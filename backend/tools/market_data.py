import json
import logging
import requests
import yfinance as yf
from langchain.tools import tool
from tools.ticker_resolver import get_ticker
from functools import lru_cache
from utils.config import settings
from utils.safe_execution import safe_execute_sync

logger = logging.getLogger(__name__)

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}


def infer_currency_from_symbol(ticker_str: str, fallback: str = "USD") -> str:
    symbol = (ticker_str or "").upper()
    suffix_currency_map = {
        ".NS": "INR",
        ".BO": "INR",
        ".L": "GBP",
        ".T": "JPY",
        ".DE": "EUR",
        ".PA": "EUR",
        ".AS": "EUR",
        ".MI": "EUR",
    }

    for suffix, currency in suffix_currency_map.items():
        if symbol.endswith(suffix):
            return currency

    return fallback


def is_indian_equity(ticker_str: str) -> bool:
    symbol = (ticker_str or "").upper()
    return symbol.endswith(".NS") or symbol.endswith(".BO")


def to_eodhd_symbol(ticker_str: str) -> str:
    symbol = (ticker_str or "").upper()
    if symbol.endswith(".NS"):
        return f"{symbol[:-3]}.NSE"
    if symbol.endswith(".BO"):
        return f"{symbol[:-3]}.BSE"
    return symbol


def _empty_market_stats(company_name: str, ticker_str: str) -> dict:
    return {
        "symbol": ticker_str,
        "companyName": company_name,
        "currency": infer_currency_from_symbol(ticker_str),
        "currentPrice": "Data Not Available",
        "marketCap": "Data Not Available",
        "revenue": "Data Not Available",
        "revenueGrowth": "Data Not Available",
        "sector": "Data Not Available",
        "industry": "Data Not Available",
        "ebitda": "Data Not Available",
        "debtToEquity": "Data Not Available",
        "_providerStatus": {"none": ["no_data"]},
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


@lru_cache(maxsize=128)
def fetch_quote_data(ticker_str: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker_str}"
    response = requests.get(url, headers=YAHOO_HEADERS, timeout=8)
    response.raise_for_status()
    payload = response.json()
    result = ((payload.get("quoteResponse") or {}).get("result") or [{}])[0]
    return result if isinstance(result, dict) else {}


@lru_cache(maxsize=128)
def fetch_eodhd_fundamentals(ticker_str: str) -> dict:
    if not settings.eodhd_api_key or not is_indian_equity(ticker_str):
        return {}

    eodhd_symbol = to_eodhd_symbol(ticker_str)
    url = f"https://eodhd.com/api/fundamentals/{eodhd_symbol}?api_token={settings.eodhd_api_key}&fmt=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def fetch_yfinance_payload(ticker_str: str) -> dict:
    ticker = yf.Ticker(ticker_str)
    payload: dict = {}

    try:
        info = ticker.get_info()
        if isinstance(info, dict):
            payload["info"] = info
    except Exception as exc:
        logger.warning("yfinance info request failed for %s: %s", ticker_str, exc)

    try:
        financials = ticker.financials
        if financials is not None and not financials.empty:
            payload["financials"] = financials.to_dict()
    except Exception as exc:
        logger.warning("yfinance financials request failed for %s: %s", ticker_str, exc)

    try:
        balance_sheet = ticker.balance_sheet
        if balance_sheet is not None and not balance_sheet.empty:
            payload["balance_sheet"] = balance_sheet.to_dict()
    except Exception as exc:
        logger.warning("yfinance balance sheet request failed for %s: %s", ticker_str, exc)

    return payload


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
        "currency": profile.get("currency") or infer_currency_from_symbol(ticker_str),
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


def _normalize_percentage_metric(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) > 3:
        return value / 100
    return value


def _extract_percentage(value):
    numeric = _to_float(value)
    return _normalize_percentage_metric(numeric)


def _first_numeric(values: list) -> float | None:
    for value in values:
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _extract_finnhub_share_outstanding(profile: dict, metric_payload: dict) -> float | None:
    profile_outstanding = _to_float(profile.get("shareOutstanding")) if isinstance(profile, dict) else None
    if profile_outstanding is not None:
        return profile_outstanding * 1_000_000

    metric = metric_payload.get("metric", {}) if isinstance(metric_payload, dict) else {}
    if not isinstance(metric, dict):
        return None

    metric_outstanding = _to_float(metric.get("shareOutstanding"))
    if metric_outstanding is not None:
        return metric_outstanding * 1_000_000

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


def _estimate_revenue_from_per_share(profile: dict, basic_financials: dict) -> float | None:
    shares_outstanding = _extract_finnhub_share_outstanding(profile, basic_financials)
    if shares_outstanding is None:
        return None

    sales_per_share = _extract_finnhub_metric(
        basic_financials,
        [
            "salesPerShareTTM",
            "revenuePerShareTTM",
        ],
    )
    if sales_per_share is None:
        return None

    return sales_per_share * shares_outstanding


def _extract_latest_statement_entries(financials_payload: dict, statement_key: str) -> list[dict]:
    financials = financials_payload.get("Financials") if isinstance(financials_payload, dict) else None
    if not isinstance(financials, dict):
        return []

    statement = financials.get(statement_key)
    if not isinstance(statement, dict):
        return []

    entries: list[dict] = []
    for period_key in ("yearly", "quarterly"):
        period_data = statement.get(period_key)
        if not isinstance(period_data, dict):
            continue

        sorted_keys = sorted(period_data.keys(), reverse=True)
        for key in sorted_keys:
            entry = period_data.get(key)
            if isinstance(entry, dict):
                entries.append(entry)

    return entries


def _extract_eodhd_revenue(fundamentals_payload: dict) -> float | None:
    highlights = fundamentals_payload.get("Highlights") if isinstance(fundamentals_payload, dict) else {}
    if isinstance(highlights, dict):
        for key in ("RevenueTTM", "Revenue"):
            value = _to_float(highlights.get(key))
            if value is not None:
                return value

    revenue_keys = [
        "totalRevenue",
        "Revenue",
        "revenue",
        "sales",
    ]
    for entry in _extract_latest_statement_entries(fundamentals_payload, "Income_Statement"):
        for key in revenue_keys:
            value = _to_float(entry.get(key))
            if value is not None:
                return value

    return None


def _extract_eodhd_revenue_growth(fundamentals_payload: dict) -> float | None:
    growth = fundamentals_payload.get("Growth") if isinstance(fundamentals_payload, dict) else {}
    if isinstance(growth, dict):
        for key in ("RevenueGrowthYOY", "QuarterlyRevenueGrowthYOY", "RevenueGrowthQuarterlyYoy"):
            value = _extract_percentage(growth.get(key))
            if value is not None:
                return value

    entries = _extract_latest_statement_entries(fundamentals_payload, "Income_Statement")
    revenues: list[float] = []
    for entry in entries:
        for key in ("totalRevenue", "Revenue", "revenue", "sales"):
            value = _to_float(entry.get(key))
            if value is not None:
                revenues.append(value)
                break
        if len(revenues) >= 2:
            break

    if len(revenues) >= 2 and revenues[1] not in (None, 0):
        return (revenues[0] - revenues[1]) / revenues[1]

    return None


def _extract_eodhd_debt_to_equity(fundamentals_payload: dict) -> float | None:
    balance_entries = _extract_latest_statement_entries(fundamentals_payload, "Balance_Sheet")
    for entry in balance_entries:
        debt = None
        equity = None

        for debt_key in ("totalDebt", "shortLongTermDebtTotal", "longTermDebt", "totalLiab"):
            debt = _to_float(entry.get(debt_key))
            if debt is not None:
                break

        for equity_key in ("totalStockholderEquity", "totalEquity", "commonStockEquity"):
            equity = _to_float(entry.get(equity_key))
            if equity is not None:
                break

        if debt is not None and equity not in (None, 0):
            return debt / equity

    return None


def build_eodhd_fundamental_stats(company_name: str, ticker_str: str, fundamentals_payload: dict) -> dict:
    general = fundamentals_payload.get("General") if isinstance(fundamentals_payload, dict) else {}
    highlights = fundamentals_payload.get("Highlights") if isinstance(fundamentals_payload, dict) else {}

    if not isinstance(general, dict):
        general = {}
    if not isinstance(highlights, dict):
        highlights = {}

    revenue = _extract_eodhd_revenue(fundamentals_payload)
    revenue_growth = _extract_eodhd_revenue_growth(fundamentals_payload)
    debt_to_equity = _extract_eodhd_debt_to_equity(fundamentals_payload)

    return {
        "symbol": ticker_str,
        "companyName": general.get("Name", company_name),
        "currency": general.get("CurrencyCode") or infer_currency_from_symbol(ticker_str),
        "currentPrice": "Data Not Available",
        "marketCap": _to_float(highlights.get("MarketCapitalization")) or "Data Not Available",
        "revenue": revenue or "Data Not Available",
        "revenueGrowth": revenue_growth if revenue_growth is not None else "Data Not Available",
        "sector": general.get("Sector", "Data Not Available"),
        "industry": general.get("Industry", "Data Not Available"),
        "ebitda": _to_float(highlights.get("EBITDA")) or "Data Not Available",
        "debtToEquity": debt_to_equity if debt_to_equity is not None else "Data Not Available",
    }


def _extract_yfinance_statement_values(statement_payload: dict, row_candidates: list[str]) -> list[float]:
    if not isinstance(statement_payload, dict):
        return []

    for row_name, columns in statement_payload.items():
        if str(row_name) not in row_candidates or not isinstance(columns, dict):
            continue

        values: list[float] = []
        for _, value in sorted(columns.items(), key=lambda item: str(item[0]), reverse=True):
            numeric = _to_float(value)
            if numeric is not None:
                values.append(numeric)

        if values:
            return values

    return []


def build_yfinance_fundamental_stats(company_name: str, ticker_str: str, payload: dict) -> dict:
    info = payload.get("info", {}) if isinstance(payload, dict) else {}
    financials = payload.get("financials", {}) if isinstance(payload, dict) else {}
    balance_sheet = payload.get("balance_sheet", {}) if isinstance(payload, dict) else {}

    if not isinstance(info, dict):
        info = {}

    revenue_series = _extract_yfinance_statement_values(
        financials,
        ["Total Revenue", "Operating Revenue", "Revenue"],
    )
    balance_debt_series = _extract_yfinance_statement_values(
        balance_sheet,
        ["Total Debt", "Net Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    )
    balance_equity_series = _extract_yfinance_statement_values(
        balance_sheet,
        ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    )

    revenue = _first_numeric([
        info.get("totalRevenue"),
        revenue_series[0] if revenue_series else None,
    ])

    revenue_growth = _extract_percentage(info.get("revenueGrowth"))
    if revenue_growth is None and len(revenue_series) >= 2 and revenue_series[1] not in (None, 0):
        revenue_growth = (revenue_series[0] - revenue_series[1]) / revenue_series[1]

    debt_to_equity = _first_numeric([
        _extract_percentage(info.get("debtToEquity")),
    ])
    if debt_to_equity is None and balance_debt_series and balance_equity_series and balance_equity_series[0] not in (None, 0):
        debt_to_equity = balance_debt_series[0] / balance_equity_series[0]

    market_cap = _first_numeric([
        info.get("marketCap"),
        info.get("enterpriseValue"),
    ])

    return {
        "symbol": ticker_str,
        "companyName": info.get("longName") or info.get("shortName") or company_name,
        "currency": info.get("currency") or infer_currency_from_symbol(ticker_str),
        "currentPrice": _first_numeric([info.get("currentPrice"), info.get("regularMarketPrice")]) or "Data Not Available",
        "marketCap": market_cap or "Data Not Available",
        "revenue": revenue or "Data Not Available",
        "revenueGrowth": revenue_growth if revenue_growth is not None else "Data Not Available",
        "sector": info.get("sector") or info.get("sectorDisp") or "Data Not Available",
        "industry": info.get("industry") or info.get("industryDisp") or "Data Not Available",
        "ebitda": _first_numeric([info.get("ebitda")]) or "Data Not Available",
        "debtToEquity": debt_to_equity if debt_to_equity is not None else "Data Not Available",
    }


def build_finnhub_fundamental_stats(company_name: str, ticker_str: str, profile: dict, basic_financials: dict, financials: dict) -> dict:
    revenue = _extract_finnhub_revenue(financials)
    if revenue is None:
        revenue = _estimate_revenue_from_per_share(profile, basic_financials)

    revenue_growth = _normalize_percentage_metric(
        _extract_finnhub_metric(
            basic_financials,
            [
                "revenueGrowthAnnual5Y",
                "revenueGrowthTTMYoy",
                "revenueGrowthQuarterlyYoy",
            ],
        )
    )

    return {
        "symbol": ticker_str,
        "companyName": company_name,
        "currency": infer_currency_from_symbol(ticker_str),
        "currentPrice": "Data Not Available",
        "marketCap": "Data Not Available",
        "revenue": revenue or "Data Not Available",
        "revenueGrowth": revenue_growth if revenue_growth is not None else "Data Not Available",
        "sector": "Data Not Available",
        "industry": "Data Not Available",
        "ebitda": "Data Not Available",
        "debtToEquity": _extract_finnhub_metric(
            basic_financials,
            [
                "totalDebt/totalEquityQuarterly",
                "totalDebt/totalEquityAnnual",
            ],
        ) or "Data Not Available",
    }


def build_market_stats(company_name: str, ticker_str: str, summary_payload: dict, chart_payload: dict, quote_payload: dict) -> dict:
    price_data = summary_payload.get("price", {}) if isinstance(summary_payload, dict) else {}
    financial_data = summary_payload.get("financialData", {}) if isinstance(summary_payload, dict) else {}
    asset_profile = summary_payload.get("assetProfile", {}) if isinstance(summary_payload, dict) else {}
    chart_meta = chart_payload.get("meta", {}) if isinstance(chart_payload, dict) else {}

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
    if current_price == "Data Not Available":
        current_price = extract_value(quote_payload.get("regularMarketPrice"))

    currency = extract_value(price_data.get("currency"), None)
    if is_missing(currency):
        currency = extract_value(chart_meta.get("currency"), None)
    if is_missing(currency):
        currency = extract_value(quote_payload.get("currency"), None)
    if is_missing(currency):
        currency = infer_currency_from_symbol(ticker_str)

    company_display_name = extract_value(price_data.get("longName"), None)
    if is_missing(company_display_name):
        company_display_name = extract_value(quote_payload.get("longName"), None)
    if is_missing(company_display_name):
        company_display_name = extract_value(quote_payload.get("shortName"), company_name)

    market_cap = extract_value(price_data.get("marketCap"))
    if is_missing(market_cap):
        market_cap = extract_value(chart_meta.get("marketCap"))
    if is_missing(market_cap):
        market_cap = extract_value(quote_payload.get("marketCap"))

    return {
        "symbol": ticker_str,
        "companyName": company_display_name or company_name,
        "currency": currency,
        "currentPrice": current_price,
        "marketCap": market_cap,
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
        quote_payload = {}
        finnhub_filled_fields: list[str] = []

        try:
            summary_payload = fetch_quote_summary_data(ticker_str)
        except requests.RequestException as exc:
            logger.warning("Quote summary request failed for %s (%s): %s", company_name, ticker_str, exc)

        try:
            chart_payload = fetch_chart_data(ticker_str)
        except requests.RequestException as exc:
            logger.warning("Chart request failed for %s (%s): %s", company_name, ticker_str, exc)

        try:
            quote_payload = fetch_quote_data(ticker_str)
        except requests.RequestException as exc:
            logger.warning("Quote request failed for %s (%s): %s", company_name, ticker_str, exc)

        market_stats = build_market_stats(company_name, ticker_str, summary_payload, chart_payload, quote_payload)
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
            finnhub_profile = {}
            finnhub_quote = {}

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
                        finnhub_profile,
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

        eodhd_missing = missing_fields(
            market_stats,
            ["marketCap", "revenue", "revenueGrowth", "sector", "industry", "ebitda", "debtToEquity"],
        )
        if eodhd_missing and settings.eodhd_api_key and is_indian_equity(ticker_str):
            try:
                eodhd_payload = fetch_eodhd_fundamentals(ticker_str)
                if eodhd_payload:
                    eodhd_stats = build_eodhd_fundamental_stats(company_name, ticker_str, eodhd_payload)
                    eodhd_filled_fields = sorted(
                        key for key, value in eodhd_stats.items()
                        if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                    )
                    market_stats = merge_market_stats(market_stats, eodhd_stats)
                    if eodhd_filled_fields:
                        provider_fields["eodhd"] = eodhd_filled_fields
            except requests.RequestException as exc:
                logger.warning("EODHD fundamentals request failed for %s (%s): %s", company_name, ticker_str, exc)

        yfinance_missing = missing_fields(
            market_stats,
            ["marketCap", "revenue", "revenueGrowth", "sector", "industry", "ebitda", "debtToEquity"],
        )
        if yfinance_missing and is_indian_equity(ticker_str):
            yfinance_payload = fetch_yfinance_payload(ticker_str)
            if yfinance_payload:
                yfinance_stats = build_yfinance_fundamental_stats(company_name, ticker_str, yfinance_payload)
                yfinance_filled_fields = sorted(
                    key for key, value in yfinance_stats.items()
                    if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
                )
                market_stats = merge_market_stats(market_stats, yfinance_stats)
                if yfinance_filled_fields:
                    provider_fields["yfinance"] = yfinance_filled_fields

        final_fields = [
            key for key, value in market_stats.items()
            if key not in {"symbol", "companyName", "currency"} and not is_missing(value)
        ]
        market_stats["_providerStatus"] = provider_fields or {"none": []}
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
