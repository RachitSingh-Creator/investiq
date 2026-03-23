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
REDDIT_HEADERS = {
    "User-Agent": "InvestIQAI/1.0 (news sentiment aggregator)"
}


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


def fetch_reddit_posts(company_name: str) -> list[dict]:
    if not settings.reddit_enabled:
        return []

    params = {
        "q": company_name,
        "sort": settings.reddit_sort,
        "limit": str(settings.reddit_post_limit),
        "restrict_sr": "false",
        "t": "month",
        "include_over_18": "false",
    }
    url = "https://www.reddit.com/search.json?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers=REDDIT_HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=settings.request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Reddit search request failed for %s: %s", company_name, exc)
        return []

    children = (((payload.get("data") or {}).get("children")) or []) if isinstance(payload, dict) else []
    posts: list[dict] = []
    company_lower = company_name.lower()
    for child in children[:settings.reddit_post_limit]:
        data = child.get("data", {}) if isinstance(child, dict) else {}
        title = data.get("title")
        selftext = data.get("selftext") or ""
        subreddit = data.get("subreddit_name_prefixed") or data.get("subreddit") or "reddit"
        permalink = data.get("permalink")

        if not title:
            continue

        combined_text = f"{title} {selftext}".lower()
        if company_lower not in combined_text:
            continue

        posts.append(
            {
                "title": title,
                "description": selftext[:280],
                "source": subreddit,
                "date": data.get("created_utc"),
                "url": f"https://www.reddit.com{permalink}" if permalink else data.get("url"),
            }
        )

    return posts


def dedupe_articles(articles: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for article in articles:
        title = (article.get("title") or "").strip().lower()
        url = (article.get("url") or "").strip().lower()
        key = f"{title}|{url}"
        if not title or key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def fetch_articles(company_name: str) -> list[dict]:
    source = settings.news_source.strip().lower()
    articles: list[dict] = []

    if source == "newsapi":
        articles = fetch_newsapi_articles(company_name)
        if not articles:
            logger.warning("Falling back to Google RSS for %s after NewsAPI returned no articles", company_name)

    if not articles:
        articles = fetch_google_rss_articles(company_name)

    combined = articles + fetch_reddit_posts(company_name)
    return dedupe_articles(combined)[: max(settings.news_fetch_limit, settings.reddit_post_limit)]


def infer_sentiment_from_articles(articles: list[dict]) -> tuple[str, float]:
    positive_terms = {
        "beat", "surge", "growth", "record", "upgrade", "strong", "bullish", "gain", "wins", "profit",
        "launch", "expands", "outperform", "buy", "optimistic",
    }
    negative_terms = {
        "miss", "drop", "decline", "lawsuit", "probe", "risk", "warning", "cut", "bearish", "fall",
        "recall", "ban", "delay", "loss", "downgrade",
    }

    score = 0
    for article in articles[:5]:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        score += sum(1 for term in positive_terms if term in text)
        score -= sum(1 for term in negative_terms if term in text)

    if score > 1:
        return "positive", min(0.8, 0.55 + score * 0.05)
    if score < -1:
        return "negative", min(0.8, 0.55 + abs(score) * 0.05)
    return "neutral", 0.5


def build_fallback_news_summary(company_name: str, articles: list[dict]) -> str:
    if not articles:
        return "No current news found."

    headlines = [article.get("title") for article in articles[:2] if article.get("title")]
    reddit_sources = [article.get("source") for article in articles if str(article.get("source", "")).lower().startswith("r/")]

    summary = "Recent headlines include " + "; ".join(headlines) + "." if headlines else "Recent coverage was retrieved."
    if reddit_sources:
        summary += " Reddit discussion was also sampled for retail sentiment context."
    return summary

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
        sentiment, confidence = infer_sentiment_from_articles(articles)
        fallback_summary = build_fallback_news_summary(company_name, articles)
        return json.dumps(
            {
                "articles": articles,
                "sentiment": sentiment,
                "confidence": confidence,
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
        sentiment, confidence = infer_sentiment_from_articles(articles)
        fallback_summary = build_fallback_news_summary(company_name, articles)
        payload = {
            "articles": articles,
            "sentiment": sentiment,
            "confidence": confidence,
            "summary": fallback_summary
        }
        
    return json.dumps(payload)

@tool
def get_company_news(company_name: str) -> str:
    """Useful to fetch top recent articles along with their LLM-analyzed sentiment score and summary."""
    return safe_execute_sync(fetch_and_analyze_news, "news", company_name)
