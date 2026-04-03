"""
Shared helper module for Indian stock (NSE/BSE) support via yfinance.
Import this in any agent that uses search_line_items to add Indian stock support.

Usage in any agent:
    from src.utils.indian_stocks import get_line_items_for_ticker

    # Replace search_line_items(...) with:
    financial_line_items = get_line_items_for_ticker(ticker, [...], end_date, api_key=api_key)
"""

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def is_indian_ticker(ticker: str) -> bool:
    """Check if ticker is an Indian stock (NSE or BSE)."""
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def get_yfinance_line_items(ticker: str) -> list:
    """
    Build synthetic line item objects from yfinance financials/balance sheet/cashflow.
    Returns a list of SimpleNamespace objects mimicking the LineItem model,
    so all existing agent analysis functions work unchanged.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        fin = stock.financials
        bs  = stock.balance_sheet
        cf  = stock.cashflow

        if fin is None or fin.empty:
            logger.warning("yfinance returned no financials for %s", ticker)
            return []

        items = []
        for col in fin.columns:
            ns = SimpleNamespace()
            ns.report_period = str(col)[:10]
            ns.period = "annual"
            ns.currency = "INR"
            ns.ticker = ticker

            def _get(df, *keys):
                for k in keys:
                    if df is not None and k in df.index:
                        try:
                            v = float(df.loc[k, col])
                            return None if v != v else v  # NaN → None
                        except Exception:
                            return None
                return None

            # ── Income Statement ──────────────────────────────────────────────
            ns.revenue          = _get(fin, "Total Revenue")
            ns.net_income       = _get(fin, "Net Income")
            ns.gross_profit     = _get(fin, "Gross Profit")
            ns.ebit             = _get(fin, "EBIT", "Operating Income")
            ns.operating_income = _get(fin, "Operating Income", "EBIT")
            ns.earnings_per_share = _get(fin, "Basic EPS", "Diluted EPS")
            ns.depreciation_and_amortization = _get(
                fin, "Reconciled Depreciation", "Depreciation And Amortization"
            )
            ns.research_and_development = _get(fin, "Research And Development")
            ns.operating_margin = (
                ns.operating_income / ns.revenue
                if ns.operating_income and ns.revenue and ns.revenue != 0
                else None
            )
            ns.gross_margin = (
                ns.gross_profit / ns.revenue
                if ns.gross_profit and ns.revenue and ns.revenue != 0
                else None
            )
            ns.net_margin = (
                ns.net_income / ns.revenue
                if ns.net_income and ns.revenue and ns.revenue != 0
                else None
            )

            # ── Balance Sheet ─────────────────────────────────────────────────
            ns.total_assets       = _get(bs, "Total Assets")
            ns.total_liabilities  = _get(
                bs, "Total Liabilities Net Minority Interest", "Total Liabilities"
            )
            ns.shareholders_equity = _get(bs, "Stockholders Equity", "Common Stock Equity")
            ns.current_assets     = _get(bs, "Current Assets")
            ns.current_liabilities = _get(bs, "Current Liabilities")
            ns.outstanding_shares  = _get(bs, "Ordinary Shares Number", "Share Issued")
            ns.total_debt          = _get(bs, "Total Debt")
            ns.cash_and_equivalents = _get(bs, "Cash And Cash Equivalents")
            ns.return_on_invested_capital = None  # Not directly available

            # Derived balance sheet
            ns.book_value_per_share = (
                ns.shareholders_equity / ns.outstanding_shares
                if ns.shareholders_equity and ns.outstanding_shares and ns.outstanding_shares != 0
                else None
            )
            ns.debt_to_equity = (
                ns.total_debt / ns.shareholders_equity
                if ns.total_debt and ns.shareholders_equity and ns.shareholders_equity != 0
                else None
            )

            # ── Cash Flow ─────────────────────────────────────────────────────
            ns.free_cash_flow     = _get(cf, "Free Cash Flow")
            ns.capital_expenditure = _get(cf, "Capital Expenditure")
            ns.operating_cash_flow = _get(cf, "Operating Cash Flow")
            ns.dividends_and_other_cash_distributions = _get(
                cf, "Cash Dividends Paid", "Common Stock Dividend Paid"
            )
            ns.issuance_or_purchase_of_equity_shares = _get(
                cf, "Issuance Of Capital Stock", "Repurchase Of Capital Stock"
            )
            ns.issuance_or_repayment_of_debt_securities = _get(
                cf, "Issuance Of Debt", "Repayment Of Debt"
            )

            items.append(ns)

        return items

    except Exception as e:
        logger.warning("yfinance line items failed for %s: %s", ticker, e)
        return []


def get_line_items_for_ticker(ticker, line_item_names, end_date,
                               period="annual", limit=10, api_key=None):
    """
    Drop-in replacement for search_line_items() that automatically uses
    yfinance for Indian tickers and falls back to the API for others.
    """
    from src.tools.api import search_line_items

    if is_indian_ticker(ticker):
        items = get_yfinance_line_items(ticker)
        return items[:limit]
    else:
        return search_line_items(
            ticker, line_item_names, end_date,
            period=period, limit=limit, api_key=api_key
        )