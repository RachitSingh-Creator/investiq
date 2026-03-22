import json
import logging
import requests
from langchain.tools import tool
from tools.ticker_resolver import get_ticker
from functools import lru_cache
from utils.config import settings
from utils.safe_execution import safe_execute_sync

logger = logging.getLogger(__name__)

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}


def _empty_market_stats(company_name: str, ticker_str: str) -> dict:
    return {
        "symbol": ticker_str,
        "companyName": company_name,
        "currency": "USD",
        "currentPrice": "Data Not Available",
        "marketCap": "Data Not Available",
        "revenue": "Data Not Available",
        "revenueGrowth": "Data Not Available",
        "sector": "Data Not Available",
        "industry": "Data Not Available",
        "ebitda": "Data Not Available",
        "debtToEquity": "Data Not Available",
    }


def is_missing(value) -> bool:
    return value in (None, "", "Data Not Available", "N/A")


def merge_market_stats(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key not in merged or is_missing(merged.get(key)):
            if not is_missing(value):
                merged[key] = value
    return merged


@lru_cache(maxsize=128)
def fetch_quote_summary_data(ticker_str: str) -> dict:
    url = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker_str}"
        "?modules=price,financialData,assetProfile"
    )
    response = requests.get(url, headers=YAHOO_HEADERS, timeout=8)
    response.raise_for_status()
    payload = response.json()
    result = ((payload.get("quoteSummary") or {}).get("result") or [{}])[0]
    return result if isinstance(result, dict) else {}


@lru_cache(maxsize=128)
def fetch_chart_data(ticker_str: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_str}?range=5d&interval=1d"
    response = requests.get(url, headers=YAHOO_HEADERS, timeout=8)
    response.raise_for_status()
    payload = response.json()
    result = ((payload.get("chart") or {}).get("result") or [{}])[0]
    return result if isinstance(result, dict) else {}


def extract_chart_price(chart_payload: dict):
    meta = chart_payload.get("meta", {}) if isinstance(chart_payload, dict) else {}
    if meta.get("regularMarketPrice") is not None:
        return meta.get("regularMarketPrice")

    indicators = chart_payload.get("indicators", {}) if isinstance(chart_payload, dict) else {}
    quotes = indicators.get("quote", [])
    if quotes:
        closes = quotes[0].get("close", []) or []
        valid_closes = [close for close in closes if close is not None]
        if valid_closes:
            return valid_closes[-1]

    return "Data Not Available"


@lru_cache(maxsize=128)
def fetch_alphavantage_overview(ticker_str: str) -> dict:
    if not settings.alphavantage_api_key:
        return {}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=OVERVIEW&symbol={ticker_str}&apikey={settings.alphavantage_api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_alphavantage_income_statement(ticker_str: str) -> dict:
    if not settings.alphavantage_api_key:
        return {}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=INCOME_STATEMENT&symbol={ticker_str}&apikey={settings.alphavantage_api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_finnhub_profile(ticker_str: str) -> dict:
    if not settings.finnhub_api_key:
        return {}

    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker_str}&token={settings.finnhub_api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_finnhub_quote(ticker_str: str) -> dict:
    if not settings.finnhub_api_key:
        return {}

    url = f"https://finnhub.io/api/v1/quote?symbol={ticker_str}&token={settings.finnhub_api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _to_float(value):
    try:
        if value in (None, "", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _alphavantage_notice(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    return payload.get("Note") or payload.get("Information") or payload.get("Error Message")


def _compute_revenue_growth_from_quarters(quarterly_reports) -> float | None:
    if not isinstance(quarterly_reports, list) or len(quarterly_reports) < 5:
        return None

    current_quarter = _to_float((quarterly_reports[0] or {}).get("totalRevenue"))
    year_ago_quarter = _to_float((quarterly_reports[4] or {}).get("totalRevenue"))

    if current_quarter in (None, 0) or year_ago_quarter in (None, 0):
        return None

    return (current_quarter - year_ago_quarter) / year_ago_quarter


def build_alphavantage_income_stats(company_name: str, ticker_str: str, income_statement: dict) -> dict:
    annual_reports = income_statement.get("annualReports") if isinstance(income_statement, dict) else None
    quarterly_reports = income_statement.get("quarterlyReports") if isinstance(income_statement, dict) else None

    latest_annual_revenue = None
    if isinstance(annual_reports, list) and annual_reports:
        latest_annual_revenue = _to_float((annual_reports[0] or {}).get("totalRevenue"))

    return {
        "symbol": ticker_str,
        "companyName": company_name,
        "currency": "USD",
        "currentPrice": "Data Not Available",
        "marketCap": "Data Not Available",
        "revenue": latest_annual_revenue or "Data Not Available",
        "revenueGrowth": _compute_revenue_growth_from_quarters(quarterly_reports) or "Data Not Available",
        "sector": "Data Not Available",
        "industry": "Data Not Available",
        "ebitda": "Data Not Available",
        "debtToEquity": "Data Not Available",
    }


def build_alphavantage_stats(company_name: str, ticker_str: str, overview: dict) -> dict:
    return {
        "symbol": ticker_str,
        "companyName": overview.get("Name", company_name),
        "currency": overview.get("Currency", "USD"),
        "currentPrice": "Data Not Available",
        "marketCap": _to_float(overview.get("MarketCapitalization")) or "Data Not Available",
        "revenue": _to_float(overview.get("RevenueTTM")) or "Data Not Available",
        "revenueGrowth": _to_float(overview.get("QuarterlyRevenueGrowthYOY")) or "Data Not Available",
        "sector": overview.get("Sector", "Data Not Available"),
        "industry": overview.get("Industry", "Data Not Available"),
        "ebitda": _to_float(overview.get("EBITDA")) or "Data Not Available",
        "debtToEquity": _to_float(overview.get("DebtToEquity")) or "Data Not Available",
    }


def build_finnhub_stats(company_name: str, ticker_str: str, profile: dict, quote: dict) -> dict:
    market_cap_millions = _to_float(profile.get("marketCapitalization"))
    market_cap = market_cap_millions * 1_000_000 if market_cap_millions is not None else "Data Not Available"

    return {
        "symbol": ticker_str,
        "companyName": profile.get("name", company_name),
        "currency": profile.get("currency", "USD"),
        "currentPrice": quote.get("c") or "Data Not Available",
        "marketCap": market_cap,
        "revenue": "Data Not Available",
        "revenueGrowth": "Data Not Available",
        "sector": profile.get("finnhubIndustry", "Data Not Available"),
        "industry": "Data Not Available",
        "ebitda": "Data Not Available",
        "debtToEquity": "Data Not Available",
    }


def build_market_stats(company_name: str, ticker_str: str, summary_payload: dict, chart_payload: dict) -> dict:
    price_data = summary_payload.get("price", {}) if isinstance(summary_payload, dict) else {}
    financial_data = summary_payload.get("financialData", {}) if isinstance(summary_payload, dict) else {}
    asset_profile = summary_payload.get("assetProfile", {}) if isinstance(summary_payload, dict) else {}

    def extract_value(value, default="Data Not Available"):
        if isinstance(value, dict):
            if value.get("raw") is not None:
                return value.get("raw")
            if value.get("fmt") is not None:
                return value.get("fmt")
        if value is None:
            return default
        return value

    current_price = extract_value(price_data.get("regularMarketPrice"))
    if current_price == "Data Not Available":
        current_price = extract_chart_price(chart_payload)

    return {
        "symbol": ticker_str,
        "companyName": extract_value(price_data.get("longName"), company_name),
        "currency": extract_value(price_data.get("currency"), "USD"),
        "currentPrice": current_price,
        "marketCap": extract_value(price_data.get("marketCap")),
        "revenue": extract_value(financial_data.get("totalRevenue")),
        "revenueGrowth": extract_value(financial_data.get("revenueGrowth")),
        "sector": extract_value(asset_profile.get("sector")),
        "industry": extract_value(asset_profile.get("industry")),
        "ebitda": extract_value(financial_data.get("ebitda")),
        "debtToEquity": extract_value(financial_data.get("debtToEquity")),
    }

def fetch_market_logic(company_name: str) -> str:
    """Core market fetch logic wrapped physically by safe wrapper."""
    logger.info(f"Fetching market data for: {company_name}")
    ticker_str = get_ticker(company_name)
    
    try:
        summary_payload = {}
        chart_payload = {}

        try:
            summary_payload = fetch_quote_summary_data(ticker_str)
        except requests.RequestException as exc:
            logger.warning("Quote summary request failed for %s (%s): %s", company_name, ticker_str, exc)

        try:
            chart_payload = fetch_chart_data(ticker_str)
        except requests.RequestException as exc:
            logger.warning("Chart request failed for %s (%s): %s", company_name, ticker_str, exc)

        market_stats = build_market_stats(company_name, ticker_str, summary_payload, chart_payload)
        provider_fields: dict[str, list[str]] = {}

        yahoo_fields = [
            key for key, value in market_stats.items()
            if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
        ]
        if yahoo_fields:
            provider_fields["yahoo"] = yahoo_fields

        try:
            av_overview = fetch_alphavantage_overview(ticker_str)
            av_notice = _alphavantage_notice(av_overview)
            if av_notice:
                logger.info("Alpha Vantage overview notice for %s (%s): %s", company_name, ticker_str, av_notice)
            elif av_overview:
                av_stats = build_alphavantage_stats(company_name, ticker_str, av_overview)
                filled_fields = [
                    key for key, value in av_stats.items()
                    if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                ]
                if filled_fields:
                    provider_fields["alphavantage_overview"] = filled_fields
                market_stats = merge_market_stats(
                    market_stats,
                    av_stats,
                )
        except requests.RequestException as exc:
            logger.warning("Alpha Vantage request failed for %s (%s): %s", company_name, ticker_str, exc)

        try:
            av_income_statement = fetch_alphavantage_income_statement(ticker_str)
            av_income_notice = _alphavantage_notice(av_income_statement)
            if av_income_notice:
                logger.info("Alpha Vantage income statement notice for %s (%s): %s", company_name, ticker_str, av_income_notice)
            elif av_income_statement:
                av_income_stats = build_alphavantage_income_stats(company_name, ticker_str, av_income_statement)
                filled_fields = [
                    key for key, value in av_income_stats.items()
                    if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                ]
                if filled_fields:
                    provider_fields["alphavantage_income"] = filled_fields
                market_stats = merge_market_stats(
                    market_stats,
                    av_income_stats,
                )
        except requests.RequestException as exc:
            logger.warning("Alpha Vantage income statement request failed for %s (%s): %s", company_name, ticker_str, exc)

        try:
            finnhub_profile = fetch_finnhub_profile(ticker_str)
            finnhub_quote = fetch_finnhub_quote(ticker_str)
            if finnhub_profile or finnhub_quote:
                finnhub_stats = build_finnhub_stats(company_name, ticker_str, finnhub_profile, finnhub_quote)
                filled_fields = [
                    key for key, value in finnhub_stats.items()
                    if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                ]
                if filled_fields:
                    provider_fields["finnhub"] = filled_fields
                market_stats = merge_market_stats(
                    market_stats,
                    finnhub_stats,
                )
        except requests.RequestException as exc:
            logger.warning("Finnhub request failed for %s (%s): %s", company_name, ticker_str, exc)

        final_fields = [
            key for key, value in market_stats.items()
            if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
        ]
        logger.info(
            "Market data providers for %s (%s): %s | final fields: %s",
            company_name,
            ticker_str,
            provider_fields or {"none": []},
            final_fields,
        )

        return json.dumps(market_stats)
        
    except Exception as e:
        logger.error(f"Failed pulling market data natively for {company_name}: {e}")
        return json.dumps(_empty_market_stats(company_name, ticker_str))

@tool
def get_market_data(company_name: str) -> str:
    """
    Useful to fetch real-time market data globally for a target company securely.
    Returns current price, market cap, revenue, revenue growth, and sector correctly wrapped in timeout resilience.
    """
    return safe_execute_sync(fetch_market_logic, "market_data", company_name)
