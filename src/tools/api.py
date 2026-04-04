import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


# ─── Indian Stock Helpers (yfinance) ───────────────────────────────────────────

def _is_indian_ticker(ticker: str) -> bool:
    """Check if ticker is an Indian stock (NSE or BSE)."""
    return ticker.endswith(".NS") or ticker.endswith(".BO")


def _get_yfinance_metrics(ticker: str) -> list[FinancialMetrics]:
    """Fetch financial metrics using yfinance for Indian stocks."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            logger.warning("yfinance returned empty info for %s", ticker)
            return []

        # Calculate missing fields from available data
        total_cash = info.get("totalCash")
        total_debt = info.get("totalDebt")
        market_cap = info.get("marketCap")
        gross_margins = info.get("grossMargins")
        ebitda_margins = info.get("ebitdaMargins")
        net_margin = info.get("profitMargins")
        revenue_growth = info.get("revenueGrowth")

        # Estimate ROE from ROA or net margin if missing
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")
        if roe is None and net_margin and revenue_growth is not None:
            roe = net_margin * 1.5  # rough estimate

        # Estimate current ratio from cash/debt if missing
        current_ratio = info.get("currentRatio")
        if current_ratio is None and total_cash and total_debt:
            current_ratio = total_cash / total_debt if total_debt > 0 else 1.0

        return [FinancialMetrics(
            ticker=ticker,
            report_period=datetime.datetime.now().strftime("%Y-%m-%d"),
            period="ttm",
            currency=info.get("currency", "INR"),
            market_cap=market_cap,
            price_to_earnings_ratio=info.get("trailingPE"),
            price_to_book_ratio=info.get("priceToBook"),
            price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
            return_on_equity=roe,
            return_on_assets=roa,
            net_margin=net_margin,
            operating_margin=info.get("operatingMargins") or ebitda_margins,
            revenue_growth=revenue_growth,
            earnings_growth=info.get("earningsGrowth") or revenue_growth,
            current_ratio=current_ratio,
            debt_to_equity=info.get("debtToEquity"),
            free_cash_flow_per_share=info.get("freeCashflow"),
            earnings_per_share=info.get("trailingEps"),
            revenue_per_share=info.get("revenuePerShare"),
            book_value_per_share=info.get("bookValue"),
            dividend_yield=info.get("dividendYield"),
        )]
    except Exception as e:
        logger.warning("yfinance metrics fetch failed for %s: %s", ticker, e)
        return []
    


def _get_yfinance_metrics(ticker: str) -> list[FinancialMetrics]:
    """Fetch financial metrics using yfinance for Indian stocks."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return []

        # Compute derived metrics from financials if available
        gross_margin = ev_to_ebitda = ev_to_revenue = None
        ev = info.get("enterpriseValue")
        try:
            fin = stock.financials
            gross_profit = fin.loc["Gross Profit"].iloc[0] if "Gross Profit" in fin.index else None
            total_revenue = fin.loc["Total Revenue"].iloc[0] if "Total Revenue" in fin.index else None
            ebitda = fin.loc["EBITDA"].iloc[0] if "EBITDA" in fin.index else None
            gross_margin = float(gross_profit / total_revenue) if gross_profit and total_revenue else None
            ev_to_ebitda = float(ev / ebitda) if ev and ebitda else None
            ev_to_revenue = float(ev / total_revenue) if ev and total_revenue else None
        except Exception:
            pass

        return [FinancialMetrics(
            ticker=ticker,
            report_period=datetime.datetime.now().strftime("%Y-%m-%d"),
            period="ttm",
            currency=info.get("currency", "INR"),
            market_cap=info.get("marketCap"),
            enterprise_value=ev,
            price_to_earnings_ratio=info.get("trailingPE"),
            price_to_book_ratio=info.get("priceToBook"),
            price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
            enterprise_value_to_ebitda_ratio=ev_to_ebitda,
            enterprise_value_to_revenue_ratio=ev_to_revenue,
            free_cash_flow_yield=None,
            peg_ratio=info.get("pegRatio"),
            gross_margin=gross_margin,
            operating_margin=info.get("operatingMargins"),
            net_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=info.get("debtToEquity"),
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=info.get("payoutRatio"),
            earnings_per_share=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            free_cash_flow_per_share=info.get("freeCashflow"),
        )]
    except Exception as e:
        logger.warning("yfinance metrics failed for %s: %s", ticker, e)
        return []


# ─── API Request Helper ────────────────────────────────────────────────────────

def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """Make an API request with rate limiting handling and moderate backoff."""
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        return response


# ─── Price Data ────────────────────────────────────────────────────────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # Use yfinance for Indian stocks
    if _is_indian_ticker(ticker):
        prices = _get_yfinance_prices(ticker, start_date, end_date)
        if prices:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices

    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        logger.warning("Could not fetch prices for %s (HTTP %s)", ticker, response.status_code)
        return []

    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except (ValueError, KeyError) as e:
        logger.warning("Failed to parse price data for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ─── Financial Metrics ─────────────────────────────────────────────────────────

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # Use yfinance for Indian stocks
    if _is_indian_ticker(ticker):
        metrics = _get_yfinance_metrics(ticker)
        if metrics:
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        logger.warning("Could not fetch financial metrics for %s (HTTP %s)", ticker, response.status_code)
        return []

    try:
        metrics_response = FinancialMetricsResponse(**response.json())
        financial_metrics = metrics_response.financial_metrics
    except (ValueError, KeyError) as e:
        logger.warning("Failed to parse financial metrics for %s: %s", ticker, e)
        return []

    if not financial_metrics:
        return []

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


# ─── Line Items ────────────────────────────────────────────────────────────────

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    # Not available for Indian stocks via yfinance
    if _is_indian_ticker(ticker):
        logger.info("Line items not available for Indian ticker %s via yfinance", ticker)
        return []

    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        logger.warning("Could not fetch line items for %s (HTTP %s)", ticker, response.status_code)
        return []

    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except (ValueError, KeyError) as e:
        logger.warning("Failed to parse line items for %s: %s", ticker, e)
        return []

    if not search_results:
        return []

    return search_results[:limit]


# ─── Insider Trades ────────────────────────────────────────────────────────────

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Not available for Indian stocks
    if _is_indian_ticker(ticker):
        return []

    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            logger.warning("Could not fetch insider trades for %s (HTTP %s)", ticker, response.status_code)
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse insider trades for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        if not start_date or len(insider_trades) < limit:
            break

        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


# ─── Company News ──────────────────────────────────────────────────────────────

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Not available for Indian stocks via this API
    if _is_indian_ticker(ticker):
        return []

    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            logger.warning("Could not fetch company news for %s (HTTP %s)", ticker, response.status_code)
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse company news for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        if not start_date or len(company_news) < limit:
            break

        current_end_date = min(news.date for news in company_news).split("T")[0]

        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


# ─── Market Cap ────────────────────────────────────────────────────────────────

def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    # Use yfinance for Indian stocks
    if _is_indian_ticker(ticker):
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            return info.get("marketCap")
        except Exception as e:
            logger.warning("yfinance market cap failed for %s: %s", ticker, e)
            return None

    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            logger.warning("Could not fetch company facts for %s (HTTP %s)", ticker, response.status_code)
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


# ─── Price Utilities ───────────────────────────────────────────────────────────

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)