import logging
import re
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _tokenize(value: str) -> list[str]:
    normalized = _normalize_text(value)
    return [token for token in normalized.split() if token]


def _score_quote(company_name: str, quote: dict) -> float:
    score = 0.0
    normalized_query = _normalize_text(company_name)
    query_tokens = set(_tokenize(company_name))
    query_is_symbol_like = bool(re.fullmatch(r"[A-Z0-9.-]{2,10}", company_name.strip()))

    symbol = str(quote.get("symbol", "")).strip()
    short_name = str(quote.get("shortname") or quote.get("shortName") or "")
    long_name = str(quote.get("longname") or quote.get("longName") or "")
    exchange = str(quote.get("exchange") or quote.get("exchDisp") or "")
    quote_type = str(quote.get("quoteType") or "")

    normalized_symbol = _normalize_text(symbol)
    base_symbol = symbol.split(".")[0].lower()
    normalized_short = _normalize_text(short_name)
    normalized_long = _normalize_text(long_name)
    name_text = f"{normalized_short} {normalized_long}".strip()
    name_tokens = set(_tokenize(name_text))

    if quote_type == "EQUITY":
        score += 10

    if normalized_symbol == normalized_query:
        score += 100
    elif base_symbol == normalized_query:
        score += 80
    elif symbol.lower().startswith(normalized_query):
        score += 35

    if normalized_short == normalized_query:
        score += 90
    if normalized_long == normalized_query:
        score += 100

    if normalized_query and normalized_query in normalized_long:
        score += 50
    elif normalized_query and normalized_query in normalized_short:
        score += 45

    common_tokens = query_tokens & name_tokens
    score += len(common_tokens) * 12

    if query_tokens and query_tokens <= name_tokens:
        score += 25

    if "." not in symbol:
        score += 8
    elif not query_is_symbol_like and "." in symbol:
        score -= 12

    if query_is_symbol_like and base_symbol == company_name.strip().lower():
        score += 35

    region_markers = {
        "india": ["nse", "bse", ".ns", ".bo"],
        "japan": ["tyo", ".t"],
        "germany": ["ger", "xetra", "fra", ".de"],
        "europe": ["epa", "ams", "bru", "mil", "lis", ".de", ".as", ".pa", ".mi"],
        "uk": ["lon", ".l"],
        "us": ["nasdaq", "nyse", "nms"],
    }
    lowered_query = company_name.lower()
    for region, markers in region_markers.items():
        if region in lowered_query and any(marker in symbol.lower() or marker in exchange.lower() for marker in markers):
            score += 20

    preferred_exchange_markers = {"nasdaq", "nyse", "nse", "bse", "etra", "xetra", "tse", "tyo", "lon", "ams", "epa"}
    if any(marker in exchange.lower() for marker in preferred_exchange_markers):
        score += 5

    yahoo_score = quote.get("score")
    if isinstance(yahoo_score, (int, float)):
        score += min(float(yahoo_score), 10.0)

    return score


@lru_cache(maxsize=128)
def get_ticker(company_name: str) -> str:
    """
    Resolve a company name or ticker into the best matching equity symbol
    by scoring Yahoo Finance search results instead of relying on hardcoded maps.
    """
    logger.info("Resolving ticker for company: %s", company_name)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        data = res.json()

        quotes = data.get("quotes", [])
        if quotes:
            scored_quotes = sorted(
                (
                    (quote, _score_quote(company_name, quote))
                    for quote in quotes
                    if isinstance(quote, dict) and quote.get("symbol")
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            if scored_quotes:
                best_quote, best_score = scored_quotes[0]
                resolved_ticker = str(best_quote["symbol"])
                logger.info(
                    "Resolved %s to %s (score %.2f, exchange=%s, name=%s)",
                    company_name,
                    resolved_ticker,
                    best_score,
                    best_quote.get("exchDisp") or best_quote.get("exchange") or "unknown",
                    best_quote.get("shortname") or best_quote.get("longname") or "unknown",
                )
                return resolved_ticker

    except requests.RequestException as e:
        logger.error("Error resolving ticker for %s: %s", company_name, e)
    except Exception as e:
        logger.error("Unexpected error when resolving ticker for %s: %s", company_name, e)

    logger.warning("Using fallback string for ticker: %s", company_name)
    return company_name
