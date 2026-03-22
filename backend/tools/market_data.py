import json
import logging
import requests
from langchain.tools import tool
from tools.ticker_resolver import get_ticker
from functools import lru_cache
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
