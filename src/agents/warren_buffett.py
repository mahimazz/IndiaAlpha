from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ─── Indian Stock Data via yfinance ───────────────────────────────────────────

def _is_indian_ticker(ticker: str) -> bool:
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def _get_yfinance_line_items(ticker: str) -> list:
    """
    Build synthetic line item objects from yfinance data so the existing
    Buffett analysis functions work unchanged.
    """
    try:
        import yfinance as yf
        from types import SimpleNamespace

        stock = yf.Ticker(ticker)
        fin = stock.financials
        bs  = stock.balance_sheet
        cf  = stock.cashflow

        items = []
        for col in fin.columns:
            ns = SimpleNamespace()
            ns.report_period = str(col)[:10]

            def _get(df, *keys):
                for k in keys:
                    if df is not None and k in df.index:
                        try:
                            v = float(df.loc[k, col])
                            return None if v != v else v  # NaN → None
                        except Exception:
                            return None
                return None

            # Income statement
            ns.revenue                = _get(fin, "Total Revenue")
            ns.net_income             = _get(fin, "Net Income")
            ns.gross_profit           = _get(fin, "Gross Profit")
            ns.operating_income       = _get(fin, "Operating Income", "EBIT")
            ns.depreciation_and_amortization = _get(fin, "Reconciled Depreciation",
                                                     "Depreciation And Amortization")
            ns.capital_expenditure    = _get(cf, "Capital Expenditure")
            ns.free_cash_flow         = _get(cf, "Free Cash Flow")

            # Balance sheet
            ns.total_assets           = _get(bs, "Total Assets")
            ns.total_liabilities      = _get(bs, "Total Liabilities Net Minority Interest",
                                              "Total Liabilities")
            ns.shareholders_equity    = _get(bs, "Stockholders Equity",
                                             "Common Stock Equity")
            ns.current_assets         = _get(bs, "Current Assets")
            ns.current_liabilities    = _get(bs, "Current Liabilities")
            ns.outstanding_shares     = _get(bs, "Ordinary Shares Number",
                                             "Share Issued")

            # Cash flow extras
            ns.dividends_and_other_cash_distributions = _get(
                cf, "Cash Dividends Paid", "Common Stock Dividend Paid"
            )
            ns.issuance_or_purchase_of_equity_shares = _get(
                cf, "Issuance Of Capital Stock", "Repurchase Of Capital Stock"
            )

            # Derived
            ns.gross_margin = (
                ns.gross_profit / ns.revenue
                if ns.gross_profit and ns.revenue and ns.revenue != 0
                else None
            )

            items.append(ns)

        return items

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "yfinance line items failed for %s: %s", ticker, e
        )
        return []


# ─── Main Agent ────────────────────────────────────────────────────────────────

def warren_buffett_agent(state: AgentState, agent_id: str = "warren_buffett_agent"):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        if _is_indian_ticker(ticker):
            financial_line_items = _get_yfinance_line_items(ticker)
        else:
            financial_line_items = search_line_items(
                ticker,
                [
                    "capital_expenditure",
                    "depreciation_and_amortization",
                    "net_income",
                    "outstanding_shares",
                    "total_assets",
                    "total_liabilities",
                    "shareholders_equity",
                    "dividends_and_other_cash_distributions",
                    "issuance_or_purchase_of_equity_shares",
                    "gross_profit",
                    "revenue",
                    "free_cash_flow",
                ],
                end_date,
                period="ttm",
                limit=10,
                api_key=api_key,
            )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status(agent_id, ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing competitive moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status(agent_id, ticker, "Analyzing pricing power")
        pricing_power_analysis = analyze_pricing_power(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing book value growth")
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        total_score = (
            fundamental_analysis["score"] +
            consistency_analysis["score"] +
            moat_analysis["score"] +
            mgmt_analysis["score"] +
            pricing_power_analysis["score"] +
            book_value_analysis["score"]
        )

        max_possible_score = (
            10 +
            moat_analysis["max_score"] +
            mgmt_analysis["max_score"] +
            5 +
            5
        )

        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status(agent_id, ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=buffett_output.reasoning)

    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = buffett_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ─── Analysis Functions ────────────────────────────────────────────────────────

def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    earnings_values = [getattr(item, 'net_income', None) for item in financial_line_items
                       if getattr(item, 'net_income', None) is not None]

    if len(earnings_values) >= 4:
        earnings_growth = all(
            earnings_values[i] > earnings_values[i + 1]
            for i in range(len(earnings_values) - 1)
        )
        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(
                f"Total earnings growth of {growth_rate:.1%} over {len(earnings_values)} periods"
            )
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_moat(metrics: list) -> dict[str, any]:
    """Evaluate whether the company likely has a durable competitive advantage."""
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]

    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        if roe_consistency >= 0.8:
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(
                f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%})"
            )
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE: {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
    else:
        reasoning.append("Insufficient ROE history for moat analysis")

    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_avg = sum(historical_margins[:3]) / 3
        older_avg = sum(historical_margins[-3:]) / 3

        if avg_margin > 0.2 and recent_avg >= older_avg:
            moat_score += 1
            reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%})")
        elif avg_margin > 0.15:
            reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%})")
        else:
            reasoning.append(f"Low operating margins (avg: {avg_margin:.1%})")

    if len(metrics) >= 5:
        asset_turnovers = [
            m.asset_turnover for m in metrics
            if hasattr(m, 'asset_turnover') and m.asset_turnover is not None
        ]
        if len(asset_turnovers) >= 3 and any(t > 1.0 for t in asset_turnovers):
            moat_score += 1
            reasoning.append("Efficient asset utilization suggests operational moat")

    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((r - roe_avg) ** 2 for r in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0

        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((m - margin_avg) ** 2 for m in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0

        overall_stability = (roe_stability + margin_stability) / 2
        if overall_stability > 0.7:
            moat_score += 1
            reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong moat")

    moat_score = min(moat_score, max_score)
    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available",
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """Checks for share dilution or consistent buybacks, and dividend track record."""
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0
    latest = financial_line_items[0]

    issuance = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
    if issuance is not None and issuance < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
    elif issuance is not None and issuance > 0:
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    dividends = getattr(latest, "dividends_and_other_cash_distributions", None)
    if dividends is not None and dividends < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {"score": mgmt_score, "max_score": 2, "details": "; ".join(reasoning)}


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power)."""
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []

    net_income   = getattr(latest, 'net_income', None)
    depreciation = getattr(latest, 'depreciation_and_amortization', None)
    capex        = getattr(latest, 'capital_expenditure', None)

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None:   missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None:        missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    maintenance_capex = estimate_maintenance_capex(financial_line_items)

    working_capital_change = 0
    if len(financial_line_items) >= 2:
        try:
            ca_cur  = getattr(latest, 'current_assets', None)
            cl_cur  = getattr(latest, 'current_liabilities', None)
            prev    = financial_line_items[1]
            ca_prev = getattr(prev, 'current_assets', None)
            cl_prev = getattr(prev, 'current_liabilities', None)
            if all([ca_cur, cl_cur, ca_prev, cl_prev]):
                working_capital_change = (ca_cur - cl_cur) - (ca_prev - cl_prev)
                details.append(f"Working capital change: {working_capital_change:,.0f}")
        except Exception:
            pass

    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

    details.extend([
        f"Net income: {net_income:,.0f}",
        f"Depreciation: {depreciation:,.0f}",
        f"Estimated maintenance capex: {maintenance_capex:,.0f}",
        f"Owner earnings: {owner_earnings:,.0f}",
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0,
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    """Estimate maintenance capital expenditure."""
    if not financial_line_items:
        return 0

    capex_ratios = []
    for item in financial_line_items[:5]:
        capex = getattr(item, 'capital_expenditure', None)
        revenue = getattr(item, 'revenue', None)
        if capex and revenue and revenue > 0:
            capex_ratios.append(abs(capex) / revenue)

    latest = financial_line_items[0]
    latest_depreciation = getattr(latest, 'depreciation_and_amortization', None) or 0
    latest_capex = abs(getattr(latest, 'capital_expenditure', None) or 0)

    method_1 = latest_capex * 0.85
    method_2 = latest_depreciation

    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = getattr(latest, 'revenue', None) or 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0
        estimates = sorted([method_1, method_2, method_3])
        return estimates[1]
    else:
        return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """Calculate intrinsic value using enhanced DCF with owner earnings."""
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest = financial_line_items[0]
    shares_outstanding = getattr(latest, 'outstanding_shares', None)

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    details = []

    historical_earnings = [
        getattr(item, 'net_income', None) for item in financial_line_items[:5]
        if getattr(item, 'net_income', None)
    ]

    if len(historical_earnings) >= 3:
        oldest = historical_earnings[-1]
        latest_e = historical_earnings[0]
        years = len(historical_earnings) - 1
        if oldest > 0:
            historical_growth = ((latest_e / oldest) ** (1 / years)) - 1
            historical_growth = max(-0.05, min(historical_growth, 0.15))
            conservative_growth = historical_growth * 0.7
        else:
            conservative_growth = 0.03
    else:
        conservative_growth = 0.03

    stage1_growth  = min(conservative_growth, 0.08)
    stage2_growth  = min(conservative_growth * 0.5, 0.04)
    terminal_growth = 0.025
    discount_rate  = 0.10
    stage1_years   = 5
    stage2_years   = 5

    details.append(
        f"DCF: Stage1={stage1_growth:.1%} Stage2={stage2_growth:.1%} Terminal={terminal_growth:.1%}"
    )

    stage1_pv = sum(
        owner_earnings * (1 + stage1_growth) ** y / (1 + discount_rate) ** y
        for y in range(1, stage1_years + 1)
    )

    stage1_final = owner_earnings * (1 + stage1_growth) ** stage1_years
    stage2_pv = sum(
        stage1_final * (1 + stage2_growth) ** y / (1 + discount_rate) ** (stage1_years + y)
        for y in range(1, stage2_years + 1)
    )

    final_earnings = stage1_final * (1 + stage2_growth) ** stage2_years
    terminal_value = (final_earnings * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    terminal_pv    = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)

    intrinsic_value = (stage1_pv + stage2_pv + terminal_pv) * 0.85  # 15% haircut

    details.extend([
        f"Stage1 PV: {stage1_pv:,.0f}",
        f"Stage2 PV: {stage2_pv:,.0f}",
        f"Terminal PV: {terminal_pv:,.0f}",
        f"Conservative IV (15% haircut): {intrinsic_value:,.0f}",
        f"Owner earnings: {owner_earnings:,.0f}",
    ])

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "details": details,
    }


def analyze_book_value_growth(financial_line_items: list) -> dict[str, any]:
    """Analyze book value per share growth - a key Buffett metric."""
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}

    book_values = []
    for item in financial_line_items:
        equity  = getattr(item, 'shareholders_equity', None)
        shares  = getattr(item, 'outstanding_shares', None)
        if equity and shares and shares > 0:
            book_values.append(equity / shares)

    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data for growth analysis"}

    score = 0
    reasoning = []

    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)

    if growth_rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value per share growth")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("Good book value per share growth pattern")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value per share growth")
    else:
        reasoning.append("Inconsistent book value per share growth")

    cagr_score, cagr_reason = _calculate_book_value_cagr(book_values)
    score += cagr_score
    reasoning.append(cagr_reason)

    return {"score": score, "details": "; ".join(reasoning)}


def _calculate_book_value_cagr(book_values: list) -> tuple[int, str]:
    if len(book_values) < 2:
        return 0, "Insufficient data for CAGR calculation"

    oldest_bv = book_values[-1]
    latest_bv = book_values[0]
    years = len(book_values) - 1

    if oldest_bv > 0 and latest_bv > 0:
        cagr = ((latest_bv / oldest_bv) ** (1 / years)) - 1
        if cagr > 0.15:
            return 2, f"Excellent book value CAGR: {cagr:.1%}"
        elif cagr > 0.1:
            return 1, f"Good book value CAGR: {cagr:.1%}"
        else:
            return 0, f"Book value CAGR: {cagr:.1%}"
    elif oldest_bv < 0 < latest_bv:
        return 3, "Excellent: improved from negative to positive book value"
    elif oldest_bv > 0 > latest_bv:
        return 0, "Warning: declined from positive to negative book value"
    else:
        return 0, "Unable to calculate meaningful book value CAGR"


def analyze_pricing_power(financial_line_items: list, metrics: list) -> dict[str, any]:
    """Analyze pricing power - Buffett's key indicator of a business moat."""
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}

    score = 0
    reasoning = []

    gross_margins = [
        getattr(item, 'gross_margin', None) for item in financial_line_items
        if getattr(item, 'gross_margin', None) is not None
    ]

    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2
        older_avg  = sum(gross_margins[-2:]) / 2

        if recent_avg > older_avg + 0.02:
            score += 3
            reasoning.append("Expanding gross margins indicate strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving gross margins suggest good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:
            score += 1
            reasoning.append("Stable gross margins")
        else:
            reasoning.append("Declining gross margins may indicate pricing pressure")

    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%})")
        elif avg_margin > 0.3:
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%})")

    return {
        "score": score,
        "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available",
    }


# ─── LLM Generation ───────────────────────────────────────────────────────────

def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str = "warren_buffett_agent",
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with a compact prompt."""

    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),
        "consistency": analysis_data.get("consistency_analysis", {}).get("details"),
        "moat": analysis_data.get("moat_analysis", {}).get("details"),
        "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),
        "book_value": analysis_data.get("book_value_analysis", {}).get("details"),
        "management": analysis_data.get("management_analysis", {}).get("details"),
        "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),
        "market_cap": analysis_data.get("market_cap"),
        "margin_of_safety": analysis_data.get("margin_of_safety"),
    }

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Warren Buffett. Decide bullish, bearish, or neutral using only the provided facts.\n"
            "\n"
            "Checklist for decision:\n"
            "- Circle of competence\n"
            "- Competitive moat\n"
            "- Management quality\n"
            "- Financial strength\n"
            "- Valuation vs intrinsic value\n"
            "- Long-term prospects\n"
            "\n"
            "Signal rules:\n"
            "- Bullish: strong business AND margin_of_safety > 0.\n"
            "- Bearish: poor business OR clearly overvalued.\n"
            "- Neutral: good business but margin_of_safety <= 0, or mixed evidence.\n"
            "\n"
            "Confidence scale:\n"
            "- 90-100%: Exceptional business within my circle, trading at attractive price\n"
            "- 70-89%: Good business with decent moat, fair valuation\n"
            "- 50-69%: Mixed signals, would need more information or better price\n"
            "- 30-49%: Outside my expertise or concerning fundamentals\n"
            "- 10-29%: Poor business or significantly overvalued\n"
            "\n"
            "Keep reasoning under 120 characters. Do not invent data. Return JSON only."
        ),
        (
            "human",
            "Ticker: {ticker}\n"
            "Facts:\n{facts}\n\n"
            "Return exactly:\n"
            "{{\n"
            '  "signal": "bullish" | "bearish" | "neutral",\n'
            '  "confidence": int,\n'
            '  "reasoning": "short justification"\n'
            "}}"
        ),
    ])

    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
    })

    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )