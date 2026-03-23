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


def missing_fields(payload: dict, fields: list[str]) -> list[str]:
    return [field for field in fields if is_missing(payload.get(field))]


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
def fetch_finnhub_profile(ticker_str: str) -> dict:
    if not settings.finnhub_api_key:
        return {}

    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker_str}&token={settings.finnhub_api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_finnhub_basic_financials(ticker_str: str) -> dict:
    if not settings.finnhub_api_key:
        return {}

    url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker_str}&metric=all&token={settings.finnhub_api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_finnhub_financials(ticker_str: str) -> dict:
    if not settings.finnhub_api_key:
        return {}

    url = f"https://finnhub.io/api/v1/stock/financials?symbol={ticker_str}&statement=ic&freq=annual&token={settings.finnhub_api_key}"
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


def _finnhub_notice(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error
    return None


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


def _extract_finnhub_metric(metric_payload: dict, keys: list[str]):
    metric = metric_payload.get("metric", {}) if isinstance(metric_payload, dict) else {}
    if not isinstance(metric, dict):
        return None

    for key in keys:
        value = _to_float(metric.get(key))
        if value is not None:
            return value

    return None


def _extract_finnhub_revenue(financials_payload: dict) -> float | None:
    data = financials_payload.get("financials") if isinstance(financials_payload, dict) else None
    if not isinstance(data, list):
        return None

    revenue_keys = [
        "revenue",
        "totalRevenue",
        "revenueFromContractWithCustomerExcludingAssessedTax",
        "salesRevenueNet",
    ]

    for entry in data:
        if not isinstance(entry, dict):
            continue
        for key in revenue_keys:
            value = _to_float(entry.get(key))
            if value is not None:
                return value

    return None


def build_finnhub_fundamental_stats(company_name: str, ticker_str: str, basic_financials: dict, financials: dict) -> dict:
    return {
        "symbol": ticker_str,
        "companyName": company_name,
        "currency": "USD",
        "currentPrice": "Data Not Available",
        "marketCap": "Data Not Available",
        "revenue": _extract_finnhub_revenue(financials) or "Data Not Available",
        "revenueGrowth": _extract_finnhub_metric(
            basic_financials,
            [
                "revenueGrowthTTMYoy",
                "revenueGrowthQuarterlyYoy",
                "revenueGrowthAnnual5Y",
            ],
        ) or "Data Not Available",
        "sector": "Data Not Available",
        "industry": "Data Not Available",
        "ebitda": _extract_finnhub_metric(
            basic_financials,
            [
                "ebitda",
                "ebitdPerShareTTM",
            ],
        ) or "Data Not Available",
        "debtToEquity": _extract_finnhub_metric(
            basic_financials,
            [
                "totalDebt/totalEquityQuarterly",
                "totalDebt/totalEquityAnnual",
            ],
        ) or "Data Not Available",
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

        finnhub_missing = missing_fields(
            market_stats,
            ["currentPrice", "marketCap", "sector", "revenue", "revenueGrowth", "ebitda", "debtToEquity"],
        )
        if finnhub_missing and settings.finnhub_api_key:
            finnhub_filled_fields: list[str] = []

            try:
                finnhub_profile = fetch_finnhub_profile(ticker_str)
                finnhub_quote = fetch_finnhub_quote(ticker_str)
                if finnhub_profile or finnhub_quote:
                    finnhub_stats = build_finnhub_stats(company_name, ticker_str, finnhub_profile, finnhub_quote)
                    finnhub_filled_fields.extend(
                        key for key, value in finnhub_stats.items()
                        if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                    )
                    market_stats = merge_market_stats(
                        market_stats,
                        finnhub_stats,
                    )
            except requests.RequestException as exc:
                logger.warning("Finnhub quote/profile request failed for %s (%s): %s", company_name, ticker_str, exc)

            finnhub_fundamentals_missing = missing_fields(
                market_stats,
                ["revenue", "revenueGrowth", "ebitda", "debtToEquity"],
            )
            if finnhub_fundamentals_missing:
                basic_financials = {}
                financials = {}

                try:
                    basic_financials = fetch_finnhub_basic_financials(ticker_str)
                except requests.RequestException as exc:
                    logger.warning("Finnhub basic financials request failed for %s (%s): %s", company_name, ticker_str, exc)

                try:
                    financials = fetch_finnhub_financials(ticker_str)
                except requests.RequestException as exc:
                    logger.warning("Finnhub financials request failed for %s (%s): %s", company_name, ticker_str, exc)

                finnhub_notice = _finnhub_notice(basic_financials) or _finnhub_notice(financials)
                if finnhub_notice:
                    logger.info("Finnhub fundamentals notice for %s (%s): %s", company_name, ticker_str, finnhub_notice)

                if basic_financials or financials:
                    finnhub_fundamental_stats = build_finnhub_fundamental_stats(
                        company_name,
                        ticker_str,
                        basic_financials,
                        financials,
                    )
                    finnhub_filled_fields.extend(
                        key for key, value in finnhub_fundamental_stats.items()
                        if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                    )
                    market_stats = merge_market_stats(
                        market_stats,
                        finnhub_fundamental_stats,
                    )

            if finnhub_filled_fields:
                provider_fields["finnhub"] = sorted(set(finnhub_filled_fields))

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
