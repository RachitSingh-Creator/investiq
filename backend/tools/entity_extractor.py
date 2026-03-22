import json
import logging
import re
from typing import Any
from utils.safe_execution import async_safe_execute

logger = logging.getLogger(__name__)

KNOWN_COMPANIES = {
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "apple": "Apple",
    "aapl": "Apple",
    "amazon": "Amazon",
    "amzn": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "googl": "Alphabet",
    "meta": "Meta",
    "meta platforms": "Meta",
    "tesla": "Tesla",
    "tsla": "Tesla",
    "netflix": "Netflix",
    "nflx": "Netflix",
    "intel": "Intel",
    "intc": "Intel",
    "broadcom": "Broadcom",
    "avgo": "Broadcom",
}

LEADING_INSTRUCTION_WORDS = {
    "analyze", "compare", "summarize", "provide", "show", "tell",
    "find", "review", "explain", "evaluate", "assess", "check",
    "include", "consider", "give", "share", "list",
}

GENERIC_PHRASES = {
    "recent news", "news sentiment", "risk assessment", "revenue growth",
    "general market", "market data", "final recommendation",
    "news summary", "risk analysis", "investment recommendation",
}

GENERIC_SINGLE_WORDS = {
    "analyze", "analysis", "compare", "tell", "show", "find", "what", "how", "why",
    "should", "would", "could", "market", "revenue", "growth", "risk", "risks",
    "stock", "stocks", "company", "companies", "performance", "include", "latest",
    "recent", "news", "summary", "final", "recommendation", "document", "documents",
    "sector", "industry", "debt", "equity", "ebitda", "price", "valuation",
    "outlook", "sentiment", "with", "and", "or", "the", "for", "about", "from",
    "using", "give", "provide", "review", "evaluate", "assess", "check",
}


def normalize_company_candidate(candidate: str) -> str:
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.&'-]+$", "", candidate.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def clean_company_candidate(candidate: str) -> str | None:
    cleaned = normalize_company_candidate(candidate)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in KNOWN_COMPANIES:
        return KNOWN_COMPANIES[lowered]
    if lowered in GENERIC_PHRASES or lowered in GENERIC_SINGLE_WORDS:
        return None
    if cleaned.isdigit():
        return None

    words = cleaned.split()
    lowered_words = [word.lower() for word in words]
    if not words or len(words) > 4:
        return None
    if len(words) > 1 and all(word in KNOWN_COMPANIES for word in lowered_words):
        return None
    if lowered_words[0] in LEADING_INSTRUCTION_WORDS:
        return None
    if all(word in GENERIC_SINGLE_WORDS for word in lowered_words):
        return None
    if len(words) == 1 and lowered_words[0] in GENERIC_SINGLE_WORDS:
        return None

    return cleaned


def dedupe_companies(candidates: list[str]) -> list[str]:
    cleaned_matches: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = clean_company_candidate(candidate)
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned_matches.append(cleaned)

    return cleaned_matches


def heuristic_extract_companies(query: str) -> list[str]:
    normalized = query.lower()
    matches: list[str] = []

    for alias, company in KNOWN_COMPANIES.items():
        if alias in normalized and company not in matches:
            matches.append(company)

    capitalized_phrases = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    for phrase in capitalized_phrases:
        cleaned = normalize_company_candidate(phrase)
        if not cleaned:
            continue

        words = cleaned.split()
        if not words:
            continue

        lowered_phrase = cleaned.lower()
        if lowered_phrase in GENERIC_PHRASES:
            continue

        first_word = words[0].lower()
        if first_word in LEADING_INSTRUCTION_WORDS:
            if len(words) == 1:
                continue
            candidate = " ".join(words[1:]).strip()
            candidate = clean_company_candidate(candidate)
            if candidate and candidate not in matches:
                matches.append(candidate)
            continue

        if len(words) > 2:
            continue

        if cleaned not in matches:
            matches.append(cleaned)

    uppercase_tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
    for ticker in uppercase_tickers:
        company = KNOWN_COMPANIES.get(ticker.lower())
        if company and company not in matches:
            matches.append(company)

    return dedupe_companies(matches)

async def extract_logic(query: str, llm: Any) -> list[str]:
    """Dynamically extract company names from the user query utilizing strict prompt bounds."""
    prompt = f"""
    Extract only real public company names or stock tickers explicitly mentioned in the query.
    Do not return instruction words or finance terms such as Include, Compare, Analyze, News, Risk, Growth, Stock, Revenue, Recommendation, Market, or Summary.
    Return output strictly as a valid JSON array of company names, in mention order.
    If no company is present, return [].

    Query: "{query}"
    """
    
    response = await llm.ainvoke(prompt)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
        
    try:
        companies = json.loads(content)
        if isinstance(companies, list):
            return dedupe_companies([str(company) for company in companies])
    except Exception as e:
        logger.error(f"Failed parsing extracted entities array organically: {e}")
        pass
        
    return []

async def extract_companies(query: str, llm: Any) -> list[str]:
    """Dynamically extract company names logically mapping to safe execution wrapper."""
    result = await async_safe_execute(extract_logic, "entity_extractor", query, llm)
    
    # If the safe execution failed, it returns a JSON string, fallback gracefully to a generic array
    if isinstance(result, str):
        logger.error("Safe wrapper execution triggered fallback extracting generic list instead.")
        return heuristic_extract_companies(query)
    
    return result or heuristic_extract_companies(query)
