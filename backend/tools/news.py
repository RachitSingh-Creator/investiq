import feedparser
import json
import logging
import re
import urllib.parse
import urllib.request
from langchain.tools import tool
from utils.config import settings
from utils.llm import get_llm
from utils.safe_execution import safe_execute_sync

logger = logging.getLogger(__name__)


def extract_json_object(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def fetch_newsapi_articles(company_name: str) -> list[dict]:
    if not settings.newsapi_key:
        logger.warning("NEWSAPI_KEY is missing while NEWS_SOURCE=newsapi")
        return []

    params = {
        "q": f'"{company_name}"',
        "language": settings.news_language,
        "sortBy": settings.news_sort_by,
        "pageSize": str(settings.news_fetch_limit),
        "searchIn": "title,description",
        "apiKey": settings.newsapi_key,
    }
    url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=settings.request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.error("NewsAPI request failed for %s: %s", company_name, exc)
        return []

    if payload.get("status") != "ok":
        logger.error("NewsAPI returned non-ok status for %s: %s", company_name, payload)
        return []

    articles = []
    for article in payload.get("articles", [])[:settings.news_fetch_limit]:
        articles.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "date": article.get("publishedAt"),
                "url": article.get("url"),
            }
        )
    return articles


def fetch_google_rss_articles(company_name: str) -> list[dict]:
    query = urllib.parse.quote(company_name)
    url = f"https://news.google.com/rss/search?q={query}&hl={settings.news_language}-US&gl=US&ceid=US:{settings.news_language}"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:settings.news_fetch_limit]:
        articles.append(
            {
                "title": entry.get("title"),
                "description": entry.get("summary", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "date": entry.get("published"),
                "url": entry.get("link"),
            }
        )
    return articles


def fetch_articles(company_name: str) -> list[dict]:
    source = settings.news_source.strip().lower()

    if source == "newsapi":
        articles = fetch_newsapi_articles(company_name)
        if articles:
            return articles
        logger.warning("Falling back to Google RSS for %s after NewsAPI returned no articles", company_name)

    return fetch_google_rss_articles(company_name)

def fetch_and_analyze_news(company_name: str) -> str:
    """Core logic utilizing the configured LLM provider for sentiment analysis."""
    logger.info(f"Fetching news for: {company_name}")
    llm = get_llm()
    articles = fetch_articles(company_name)

    if not articles:
        logger.warning(f"No news found for {company_name}")
        return json.dumps({
            "articles": [],
            "sentiment": "neutral",
            "confidence": 0.5,
            "summary": "No current news found."
        })
        
    raw_text_for_llm = ""
    for i, article in enumerate(articles):
        title = article.get("title", "")
        description = article.get("description") or ""
        source = article.get("source", "Unknown")
        raw_text_for_llm += f"[{i+1}] {title} (Source: {source}) {description}\n"
        
    prompt = f"""
    Analyze the following recent news headlines for '{company_name}'.

    Headlines:
    {raw_text_for_llm}

    Return ONLY a valid JSON object matching this exact schema carefully:
    {{
        "sentiment": "positive | negative | neutral",
        "confidence": <float between 0 and 1>,
        "summary": "<concise 2-sentence summary>"
    }}

    Do not use markdown fences. If the headlines are mixed or thin, return neutral sentiment.
    """
    
    try:
        analysis = llm.invoke(prompt)
        content = analysis.content.strip()
    except Exception as exc:
        logger.warning("LLM news synthesis failed for %s, using article-only fallback: %s", company_name, exc)
        fallback_titles = [article.get("title") for article in articles[:2] if article.get("title")]
        fallback_summary = "No current news found."
        if fallback_titles:
            fallback_summary = "Recent headlines include " + "; ".join(fallback_titles) + "."
        return json.dumps(
            {
                "articles": articles,
                "sentiment": "neutral",
                "confidence": 0.5,
                "summary": fallback_summary,
            }
        )
    
    # Simple JSON extraction layer if wrapped in markdown
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
        
    parsed_sentiment = extract_json_object(content)
    if parsed_sentiment:
        payload = {
            "articles": articles,
            "sentiment": parsed_sentiment.get("sentiment", "neutral"),
            "confidence": parsed_sentiment.get("confidence", 0.5),
            "summary": parsed_sentiment.get("summary", "Recent headlines were retrieved, but the summary was incomplete."),
        }
    else:
        logger.error("Failed to parse LLM JSON for %s", company_name)
        fallback_titles = [article.get("title") for article in articles[:2] if article.get("title")]
        fallback_summary = "No current news found."
        if fallback_titles:
            fallback_summary = "Recent headlines include " + "; ".join(fallback_titles) + "."
        payload = {
            "articles": articles,
            "sentiment": "neutral",
            "confidence": 0.5,
            "summary": fallback_summary
        }
        
    return json.dumps(payload)

@tool
def get_company_news(company_name: str) -> str:
    """Useful to fetch top recent articles along with their LLM-analyzed sentiment score and summary."""
    return safe_execute_sync(fetch_and_analyze_news, "news", company_name)
