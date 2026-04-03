from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class RakeshJhunjhunwalaSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


# ─── Indian Stock Data via yfinance ───────────────────────────────────────────

def _is_indian_ticker(ticker: str) -> bool:
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def _get_yfinance_line_items(ticker: str) -> list:
    """
    Build synthetic line item objects from yfinance financials/balance sheet/cashflow
    so the existing analysis functions work unchanged.
    """
    try:
        import yfinance as yf
        from types import SimpleNamespace

        stock = yf.Ticker(ticker)
        fin = stock.financials        # columns = fiscal year dates, rows = line items
        bs  = stock.balance_sheet
        cf  = stock.cashflow

        items = []
        # yfinance returns columns newest → oldest
        for col in fin.columns:
            ns = SimpleNamespace()
            ns.report_period = str(col)[:10]

            def _get(df, *keys):
                for k in keys:
                    if df is not None and k in df.index:
                        v = df.loc[k, col]
                        try:
                            f = float(v)
                            return None if (f != f) else f   # NaN → None
                        except Exception:
                            return None
                return None

            # Income statement
            ns.revenue          = _get(fin, "Total Revenue")
            ns.net_income       = _get(fin, "Net Income")
            ns.ebit             = _get(fin, "EBIT")
            ns.operating_income = _get(fin, "Operating Income")
            ns.operating_margin = (
                ns.operating_income / ns.revenue
                if ns.operating_income and ns.revenue and ns.revenue != 0
                else None
            )
            ns.earnings_per_share = _get(fin, "Basic EPS", "Diluted EPS")

            # Balance sheet
            ns.total_assets      = _get(bs, "Total Assets")
            ns.total_liabilities = _get(bs, "Total Liabilities Net Minority Interest",
                                         "Total Liabilities")
            ns.current_assets    = _get(bs, "Current Assets")
            ns.current_liabilities = _get(bs, "Current Liabilities")

            # Cash flow
            ns.free_cash_flow = _get(cf, "Free Cash Flow")
            ns.dividends_and_other_cash_distributions = _get(
                cf, "Cash Dividends Paid", "Common Stock Dividend Paid"
            )
            ns.issuance_or_purchase_of_equity_shares = _get(
                cf, "Issuance Of Capital Stock", "Repurchase Of Capital Stock"
            )

            items.append(ns)

        return items

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("yfinance line items failed for %s: %s", ticker, e)
        return []


# ─── Main Agent ────────────────────────────────────────────────────────────────

def rakesh_jhunjhunwala_agent(state: AgentState, agent_id: str = "rakesh_jhunjhunwala_agent"):
    """Analyzes stocks using Rakesh Jhunjhunwala's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    jhunjhunwala_analysis = {}

    for ticker in tickers:

        # ── Fetch data ────────────────────────────────────────────────────────
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        if _is_indian_ticker(ticker):
            financial_line_items = _get_yfinance_line_items(ticker)
        else:
            financial_line_items = search_line_items(
                ticker,
                [
                    "net_income", "earnings_per_share", "ebit", "operating_income",
                    "revenue", "operating_margin", "total_assets", "total_liabilities",
                    "current_assets", "current_liabilities", "free_cash_flow",
                    "dividends_and_other_cash_distributions",
                    "issuance_or_purchase_of_equity_shares",
                ],
                end_date,
                api_key=api_key,
            )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ── Analyses ──────────────────────────────────────────────────────────
        progress.update_status(agent_id, ticker, "Analyzing growth")
        growth_analysis = analyze_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        profitability_analysis = analyze_profitability(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balancesheet_analysis = analyze_balance_sheet(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing cash flow")
        cashflow_analysis = analyze_cash_flow(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management actions")
        management_analysis = analyze_management_actions(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value = calculate_intrinsic_value(financial_line_items, market_cap)

        # ── Score & signal ────────────────────────────────────────────────────
        total_score = (
            growth_analysis["score"]
            + profitability_analysis["score"]
            + balancesheet_analysis["score"]
            + cashflow_analysis["score"]
            + management_analysis["score"]
        )
        max_score = 24

        margin_of_safety = (
            (intrinsic_value - market_cap) / market_cap
            if intrinsic_value and market_cap
            else None
        )

        if margin_of_safety is not None and margin_of_safety >= 0.30:
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.30:
            signal = "bearish"
        else:
            quality_score = assess_quality_metrics(financial_line_items)
            if quality_score >= 0.7 and total_score >= max_score * 0.6:
                signal = "bullish"
            elif quality_score <= 0.4 or total_score <= max_score * 0.3:
                signal = "bearish"
            else:
                signal = "neutral"

        if margin_of_safety is not None:
            confidence = min(max(abs(margin_of_safety) * 150, 20), 95)
        else:
            confidence = min(max((total_score / max_score) * 100, 10), 80)

        intrinsic_value_analysis = analyze_rakesh_jhunjhunwala_style(
            financial_line_items,
            intrinsic_value=intrinsic_value,
            current_price=market_cap,
        )

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "margin_of_safety": margin_of_safety,
            "growth_analysis": growth_analysis,
            "profitability_analysis": profitability_analysis,
            "balancesheet_analysis": balancesheet_analysis,
            "cashflow_analysis": cashflow_analysis,
            "management_analysis": management_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "intrinsic_value": intrinsic_value,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Jhunjhunwala analysis")
        jhunjhunwala_output = generate_jhunjhunwala_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        jhunjhunwala_analysis[ticker] = jhunjhunwala_output.model_dump()
        progress.update_status(agent_id, ticker, "Done", analysis=jhunjhunwala_output.reasoning)

    message = HumanMessage(content=json.dumps(jhunjhunwala_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(jhunjhunwala_analysis, "Rakesh Jhunjhunwala Agent")

    state["data"]["analyst_signals"][agent_id] = jhunjhunwala_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ─── Analysis Functions ────────────────────────────────────────────────────────

def analyze_profitability(financial_line_items: list) -> dict[str, any]:
    if not financial_line_items:
        return {"score": 0, "details": "No profitability data available"}

    latest = financial_line_items[0]
    score = 0
    reasoning = []

    if (getattr(latest, 'net_income', None) and latest.net_income > 0 and
        getattr(latest, 'total_assets', None) and getattr(latest, 'total_liabilities', None) and
        latest.total_assets and latest.total_liabilities):

        shareholders_equity = latest.total_assets - latest.total_liabilities
        if shareholders_equity > 0:
            roe = (latest.net_income / shareholders_equity) * 100
            if roe > 20:
                score += 3
                reasoning.append(f"Excellent ROE: {roe:.1f}%")
            elif roe > 15:
                score += 2
                reasoning.append(f"Good ROE: {roe:.1f}%")
            elif roe > 10:
                score += 1
                reasoning.append(f"Decent ROE: {roe:.1f}%")
            else:
                reasoning.append(f"Low ROE: {roe:.1f}%")
        else:
            reasoning.append("Negative shareholders equity")
    else:
        reasoning.append("Unable to calculate ROE - missing data")

    if (getattr(latest, "operating_income", None) and latest.operating_income and
        getattr(latest, "revenue", None) and latest.revenue and latest.revenue > 0):
        operating_margin = (latest.operating_income / latest.revenue) * 100
        if operating_margin > 20:
            score += 2
            reasoning.append(f"Excellent operating margin: {operating_margin:.1f}%")
        elif operating_margin > 15:
            score += 1
            reasoning.append(f"Good operating margin: {operating_margin:.1f}%")
        elif operating_margin > 0:
            reasoning.append(f"Positive operating margin: {operating_margin:.1f}%")
        else:
            reasoning.append(f"Negative operating margin: {operating_margin:.1f}%")
    else:
        reasoning.append("Unable to calculate operating margin")

    eps_values = [getattr(item, "earnings_per_share", None) for item in financial_line_items
                  if getattr(item, "earnings_per_share", None) is not None and getattr(item, "earnings_per_share", None) > 0]

    if len(eps_values) >= 3:
        initial_eps = eps_values[-1]
        final_eps = eps_values[0]
        years = len(eps_values) - 1
        if initial_eps > 0:
            eps_cagr = ((final_eps / initial_eps) ** (1 / years) - 1) * 100
            if eps_cagr > 20:
                score += 3
                reasoning.append(f"High EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 15:
                score += 2
                reasoning.append(f"Good EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 10:
                score += 1
                reasoning.append(f"Moderate EPS CAGR: {eps_cagr:.1f}%")
            else:
                reasoning.append(f"Low EPS CAGR: {eps_cagr:.1f}%")
        else:
            reasoning.append("Cannot calculate EPS growth from negative base")
    else:
        reasoning.append("Insufficient EPS data for growth analysis")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_growth(financial_line_items: list) -> dict[str, any]:
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for growth analysis"}

    score = 0
    reasoning = []

    revenues = [getattr(item, "revenue", None) for item in financial_line_items
                if getattr(item, "revenue", None) is not None and getattr(item, "revenue", None) > 0]

    if len(revenues) >= 3:
        initial_revenue = revenues[-1]
        final_revenue = revenues[0]
        years = len(revenues) - 1
        if initial_revenue > 0:
            revenue_cagr = ((final_revenue / initial_revenue) ** (1 / years) - 1) * 100
            if revenue_cagr > 20:
                score += 3
                reasoning.append(f"Excellent revenue CAGR: {revenue_cagr:.1f}%")
            elif revenue_cagr > 15:
                score += 2
                reasoning.append(f"Good revenue CAGR: {revenue_cagr:.1f}%")
            elif revenue_cagr > 10:
                score += 1
                reasoning.append(f"Moderate revenue CAGR: {revenue_cagr:.1f}%")
            else:
                reasoning.append(f"Low revenue CAGR: {revenue_cagr:.1f}%")
        else:
            reasoning.append("Cannot calculate revenue CAGR from zero base")
    else:
        reasoning.append("Insufficient revenue data")

    net_incomes = [getattr(item, "net_income", None) for item in financial_line_items
                   if getattr(item, "net_income", None) is not None and getattr(item, "net_income", None) > 0]

    if len(net_incomes) >= 3:
        initial_income = net_incomes[-1]
        final_income = net_incomes[0]
        years = len(net_incomes) - 1
        if initial_income > 0:
            income_cagr = ((final_income / initial_income) ** (1 / years) - 1) * 100
            if income_cagr > 25:
                score += 3
                reasoning.append(f"Excellent income CAGR: {income_cagr:.1f}%")
            elif income_cagr > 20:
                score += 2
                reasoning.append(f"High income CAGR: {income_cagr:.1f}%")
            elif income_cagr > 15:
                score += 1
                reasoning.append(f"Good income CAGR: {income_cagr:.1f}%")
            else:
                reasoning.append(f"Moderate income CAGR: {income_cagr:.1f}%")
        else:
            reasoning.append("Cannot calculate income CAGR from zero base")
    else:
        reasoning.append("Insufficient net income data")

    if len(revenues) >= 3:
        declining_years = sum(1 for i in range(1, len(revenues)) if revenues[i - 1] > revenues[i])
        consistency_ratio = 1 - (declining_years / (len(revenues) - 1))
        if consistency_ratio >= 0.8:
            score += 1
            reasoning.append(f"Consistent growth ({consistency_ratio * 100:.0f}% of years)")
        else:
            reasoning.append(f"Inconsistent growth ({consistency_ratio * 100:.0f}% of years)")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_balance_sheet(financial_line_items: list) -> dict[str, any]:
    if not financial_line_items:
        return {"score": 0, "details": "No balance sheet data"}

    latest = financial_line_items[0]
    score = 0
    reasoning = []

    if (getattr(latest, "total_assets", None) and getattr(latest, "total_liabilities", None)
            and latest.total_assets and latest.total_liabilities and latest.total_assets > 0):
        debt_ratio = latest.total_liabilities / latest.total_assets
        if debt_ratio < 0.5:
            score += 2
            reasoning.append(f"Low debt ratio: {debt_ratio:.2f}")
        elif debt_ratio < 0.7:
            score += 1
            reasoning.append(f"Moderate debt ratio: {debt_ratio:.2f}")
        else:
            reasoning.append(f"High debt ratio: {debt_ratio:.2f}")
    else:
        reasoning.append("Insufficient data to calculate debt ratio")

    if (getattr(latest, "current_assets", None) and getattr(latest, "current_liabilities", None)
            and latest.current_assets and latest.current_liabilities and latest.current_liabilities > 0):
        current_ratio = latest.current_assets / latest.current_liabilities
        if current_ratio > 2.0:
            score += 2
            reasoning.append(f"Excellent liquidity: current ratio {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 1
            reasoning.append(f"Good liquidity: current ratio {current_ratio:.2f}")
        else:
            reasoning.append(f"Weak liquidity: current ratio {current_ratio:.2f}")
    else:
        reasoning.append("Insufficient data to calculate current ratio")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_cash_flow(financial_line_items: list) -> dict[str, any]:
    if not financial_line_items:
        return {"score": 0, "details": "No cash flow data"}

    latest = financial_line_items[0]
    score = 0
    reasoning = []

    if getattr(latest, "free_cash_flow", None) is not None:
        if latest.free_cash_flow > 0:
            score += 2
            reasoning.append(f"Positive free cash flow: {latest.free_cash_flow:,.0f}")
        else:
            reasoning.append(f"Negative free cash flow: {latest.free_cash_flow:,.0f}")
    else:
        reasoning.append("Free cash flow data not available")

    if getattr(latest, "dividends_and_other_cash_distributions", None) is not None:
        if latest.dividends_and_other_cash_distributions < 0:
            score += 1
            reasoning.append("Company pays dividends to shareholders")
        else:
            reasoning.append("No significant dividend payments")
    else:
        reasoning.append("No dividend payment data available")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_management_actions(financial_line_items: list) -> dict[str, any]:
    if not financial_line_items:
        return {"score": 0, "details": "No management action data"}

    latest = financial_line_items[0]
    score = 0
    reasoning = []

    issuance = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
    if issuance is not None:
        if issuance < 0:
            score += 2
            reasoning.append(f"Company buying back shares: {abs(issuance):,.0f}")
        elif issuance > 0:
            reasoning.append(f"Share issuance detected (potential dilution): {issuance:,.0f}")
        else:
            score += 1
            reasoning.append("No recent share issuance or buyback")
    else:
        reasoning.append("No data on share issuance or buybacks")

    return {"score": score, "details": "; ".join(reasoning)}


def assess_quality_metrics(financial_line_items: list) -> float:
    if not financial_line_items:
        return 0.5

    latest = financial_line_items[0]
    quality_factors = []

    if (getattr(latest, 'net_income', None) and getattr(latest, 'total_assets', None) and
            getattr(latest, 'total_liabilities', None) and latest.total_assets and latest.total_liabilities):
        shareholders_equity = latest.total_assets - latest.total_liabilities
        if shareholders_equity > 0 and latest.net_income:
            roe = latest.net_income / shareholders_equity
            if roe > 0.20:
                quality_factors.append(1.0)
            elif roe > 0.15:
                quality_factors.append(0.8)
            elif roe > 0.10:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
        else:
            quality_factors.append(0.0)
    else:
        quality_factors.append(0.5)

    if (getattr(latest, 'total_assets', None) and getattr(latest, 'total_liabilities', None) and
            latest.total_assets and latest.total_liabilities):
        debt_ratio = latest.total_liabilities / latest.total_assets
        if debt_ratio < 0.3:
            quality_factors.append(1.0)
        elif debt_ratio < 0.5:
            quality_factors.append(0.7)
        elif debt_ratio < 0.7:
            quality_factors.append(0.4)
        else:
            quality_factors.append(0.1)
    else:
        quality_factors.append(0.5)

    net_incomes = [getattr(item, "net_income", None) for item in financial_line_items[:4]
                   if getattr(item, "net_income", None) is not None and getattr(item, "net_income", None) > 0]

    if len(net_incomes) >= 3:
        declining_years = sum(1 for i in range(1, len(net_incomes)) if net_incomes[i - 1] > net_incomes[i])
        consistency = 1 - (declining_years / (len(net_incomes) - 1))
        quality_factors.append(consistency)
    else:
        quality_factors.append(0.5)

    return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5


def calculate_intrinsic_value(financial_line_items: list, market_cap: float) -> float:
    if not financial_line_items or not market_cap:
        return None

    try:
        latest = financial_line_items[0]

        if not getattr(latest, 'net_income', None) or latest.net_income <= 0:
            return None

        net_incomes = [getattr(item, "net_income", None) for item in financial_line_items[:5]
                       if getattr(item, "net_income", None) is not None and getattr(item, "net_income", None) > 0]

        if len(net_incomes) < 2:
            return latest.net_income * 12

        initial_income = net_incomes[-1]
        final_income = net_incomes[0]
        years = len(net_incomes) - 1

        historical_growth = ((final_income / initial_income) ** (1 / years) - 1) if initial_income > 0 else 0.05

        if historical_growth > 0.25:
            sustainable_growth = 0.20
        elif historical_growth > 0.15:
            sustainable_growth = historical_growth * 0.8
        elif historical_growth > 0.05:
            sustainable_growth = historical_growth * 0.9
        else:
            sustainable_growth = 0.05

        quality_score = assess_quality_metrics(financial_line_items)

        if quality_score >= 0.8:
            discount_rate = 0.12
            terminal_multiple = 18
        elif quality_score >= 0.6:
            discount_rate = 0.15
            terminal_multiple = 15
        else:
            discount_rate = 0.18
            terminal_multiple = 12

        current_earnings = latest.net_income
        dcf_value = 0

        for year in range(1, 6):
            projected_earnings = current_earnings * ((1 + sustainable_growth) ** year)
            dcf_value += projected_earnings / ((1 + discount_rate) ** year)

        year_5_earnings = current_earnings * ((1 + sustainable_growth) ** 5)
        terminal_value = (year_5_earnings * terminal_multiple) / ((1 + discount_rate) ** 5)

        return dcf_value + terminal_value

    except Exception:
        if getattr(latest, 'net_income', None) and latest.net_income > 0:
            return latest.net_income * 15
        return None


def analyze_rakesh_jhunjhunwala_style(
    financial_line_items: list,
    owner_earnings: float = None,
    intrinsic_value: float = None,
    current_price: float = None,
) -> dict[str, any]:
    profitability = analyze_profitability(financial_line_items)
    growth = analyze_growth(financial_line_items)
    balance_sheet = analyze_balance_sheet(financial_line_items)
    cash_flow = analyze_cash_flow(financial_line_items)
    management = analyze_management_actions(financial_line_items)

    total_score = (
        profitability["score"] + growth["score"] + balance_sheet["score"]
        + cash_flow["score"] + management["score"]
    )

    details = (
        f"Profitability: {profitability['details']}\n"
        f"Growth: {growth['details']}\n"
        f"Balance Sheet: {balance_sheet['details']}\n"
        f"Cash Flow: {cash_flow['details']}\n"
        f"Management Actions: {management['details']}"
    )

    if not intrinsic_value:
        intrinsic_value = calculate_intrinsic_value(financial_line_items, current_price)

    valuation_gap = (intrinsic_value - current_price) if intrinsic_value and current_price else None

    return {
        "total_score": total_score,
        "details": details,
        "owner_earnings": owner_earnings,
        "intrinsic_value": intrinsic_value,
        "current_price": current_price,
        "valuation_gap": valuation_gap,
        "breakdown": {
            "profitability": profitability,
            "growth": growth,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "management": management,
        },
    }


# ─── LLM Generation ───────────────────────────────────────────────────────────

def generate_jhunjhunwala_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> RakeshJhunjhunwalaSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Rakesh Jhunjhunwala AI agent. Decide on investment signals based on Rakesh Jhunjhunwala's principles:
            - Circle of Competence: Only invest in businesses you understand
            - Margin of Safety (> 30%): Buy at a significant discount to intrinsic value
            - Economic Moat: Look for durable competitive advantages
            - Quality Management: Seek conservative, shareholder-oriented teams
            - Financial Strength: Favor low debt, strong returns on equity
            - Long-term Horizon: Invest in businesses, not just stocks
            - Growth Focus: Look for companies with consistent earnings and revenue growth
            - Sell only if fundamentals deteriorate or valuation far exceeds intrinsic value

            When providing your reasoning, be thorough and specific by:
            1. Explaining the key factors that influenced your decision the most (both positive and negative)
            2. Highlighting how the company aligns with or violates specific Jhunjhunwala principles
            3. Providing quantitative evidence where relevant (e.g., specific margins, ROE values, debt levels)
            4. Concluding with a Jhunjhunwala-style assessment of the investment opportunity
            5. Using Rakesh Jhunjhunwala's voice and conversational style in your explanation

            For example, if bullish: "I'm particularly impressed with the consistent growth and strong balance sheet..."
            For example, if bearish: "The deteriorating margins and high debt levels concern me..."

            Follow these guidelines strictly.
            """,
        ),
        (
            "human",
            """Based on the following data, create the investment signal as Rakesh Jhunjhunwala would:

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in the following JSON format exactly:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float between 0 and 100,
              "reasoning": "string"
            }}
            """,
        ),
    ])

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_rakesh_jhunjhunwala_signal():
        return RakeshJhunjhunwalaSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=RakeshJhunjhunwalaSignal,
        state=state,
        agent_name=agent_id,
        default_factory=create_default_rakesh_jhunjhunwala_signal,
    )