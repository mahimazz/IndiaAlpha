from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap
from src.utils.indian_stocks import get_line_items_for_ticker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import math
from src.utils.api_key import get_api_key_from_state
 
 
class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
 
 
def ben_graham_agent(state: AgentState, agent_id: str = "ben_graham_agent"):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
 
    analysis_data = {}
    graham_analysis = {}
 
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)
 
        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = get_line_items_for_ticker(
            ticker,
            ["earnings_per_share", "revenue", "net_income", "book_value_per_share",
             "total_assets", "total_liabilities", "current_assets", "current_liabilities",
             "dividends_and_other_cash_distributions", "outstanding_shares"],
            end_date, period="annual", limit=10, api_key=api_key,
        )
 
        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
 
        if not metrics and not financial_line_items and not market_cap:
            graham_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "Insufficient data available for this ticker.",
            }
            progress.update_status(agent_id, ticker, "Done (no data available)")
            continue
 
        progress.update_status(agent_id, ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)
 
        progress.update_status(agent_id, ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)
 
        progress.update_status(agent_id, ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)
 
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15
 
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
 
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "earnings_analysis": earnings_analysis,
            "strength_analysis": strength_analysis,
            "valuation_analysis": valuation_analysis,
        }
 
        progress.update_status(agent_id, ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )
 
        graham_analysis[ticker] = {
            "signal": graham_output.signal,
            "confidence": graham_output.confidence,
            "reasoning": graham_output.reasoning,
        }
 
        progress.update_status(agent_id, ticker, "Done", analysis=graham_output.reasoning)
 
    message = HumanMessage(content=json.dumps(graham_analysis), name=agent_id)
 
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")
 
    state["data"]["analyst_signals"][agent_id] = graham_analysis
    progress.update_status(agent_id, None, "Done")
 
    return {"messages": [message], "data": state["data"]}
 
 
def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []
 
    if not metrics or not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}
 
    eps_vals = [
        getattr(item, "earnings_per_share", None) for item in financial_line_items
        if getattr(item, "earnings_per_share", None) is not None
    ]
 
    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}
 
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")
 
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")
 
    return {"score": score, "details": "; ".join(details)}
 
 
def analyze_financial_strength(financial_line_items: list) -> dict:
    score = 0
    details = []
 
    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}
 
    latest = financial_line_items[0]
    total_assets      = getattr(latest, "total_assets", None) or 0
    total_liabilities = getattr(latest, "total_liabilities", None) or 0
    current_assets    = getattr(latest, "current_assets", None) or 0
    current_liabilities = getattr(latest, "current_liabilities", None) or 0
 
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio.")
 
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio.")
 
    div_periods = [
        getattr(item, "dividends_and_other_cash_distributions", None)
        for item in financial_line_items
        if getattr(item, "dividends_and_other_cash_distributions", None) is not None
    ]
    if div_periods:
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years >= (len(div_periods) // 2 + 1):
            score += 1
            details.append("Company paid dividends in the majority of reported years.")
        elif div_paid_years > 0:
            details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available.")
 
    return {"score": score, "details": "; ".join(details)}
 
 
def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}
 
    latest = financial_line_items[0]
    current_assets    = getattr(latest, "current_assets", None) or 0
    total_liabilities = getattr(latest, "total_liabilities", None) or 0
    book_value_ps     = getattr(latest, "book_value_per_share", None) or 0
    eps               = getattr(latest, "earnings_per_share", None) or 0
    shares_outstanding = getattr(latest, "outstanding_shares", None) or 0
 
    details = []
    score = 0
 
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        ncav_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding
 
        details.append(f"NCAV = {net_current_asset_value:,.0f}")
        details.append(f"NCAV/share = {ncav_per_share:,.2f}, Price/share = {price_per_share:,.2f}")
 
        if net_current_asset_value > market_cap:
            score += 4
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        elif ncav_per_share >= (price_per_share * 0.67):
            score += 2
            details.append("NCAV/share >= 2/3 of price (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data.")
 
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")
 
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number.")
 
    return {"score": score, "details": "; ".join(details)}
 
 
def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BenGrahamSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (Graham Number, net-net).
            2. Emphasize financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative assumptions; focus on proven metrics.
 
            Be thorough: cite specific numbers, compare to Graham's thresholds, use his conservative voice.
            Return a rational recommendation: bullish, bearish, or neutral, with confidence (0-100) and reasoning.
            """,
        ),
        (
            "human",
            """Based on the following analysis, create a Graham-style investment signal:
 
            Analysis Data for {ticker}:
            {analysis_data}
 
            Return JSON exactly:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
        ),
    ])
 
    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
    })
 
    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0,
                               reasoning="Error in generating analysis; defaulting to neutral.")
 
    return call_llm(
        prompt=prompt,
        pydantic_model=BenGrahamSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ben_graham_signal,
    )
 