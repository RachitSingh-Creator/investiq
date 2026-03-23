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
COMPANY_NEWS_ALIASES = {
    "Apple": ["apple", "aapl", "iphone", "ipad", "mac"],
    "Microsoft": ["microsoft", "msft", "azure", "windows", "copilot"],
    "Alphabet": ["alphabet", "google", "goog", "googl", "youtube", "verily", "waymo"],
    "Amazon": ["amazon", "amzn", "aws", "prime"],
    "Meta": ["meta", "meta platforms", "facebook", "instagram", "whatsapp"],
    "Tesla": ["tesla", "tsla", "model 3", "model y", "semi"],
    "NVIDIA": ["nvidia", "nvda", "geforce", "cuda", "h100", "blackwell"],
}
NOISY_NEWS_TERMS = {
    "pypi", "npm", "crate", "package", "library", "sdk", "plugin", "extension",
    "release", "version", "changelog", "github", "gitlab", "docker image",
}
FINANCE_CONTEXT_TERMS = {
    "stock", "shares", "earnings", "revenue", "profit", "guidance", "analyst",
    "market", "valuation", "investor", "demand", "sales", "margin", "quarter",
    "fiscal", "forecast", "buyback", "chip", "iphone", "ai", "cloud",
}
FINANCE_SUBREDDITS = {
    "r/stocks", "r/investing", "r/wallstreetbets", "r/securityanalysis", "r/options",
    "r/valueinvesting", "r/economy", "r/technology", "r/apple", "r/nvidia",
    "r/teslainvestorsclub", "r/microsoft", "r/amazon", "r/google", "r/meta",
}
TRUSTED_NEWS_SOURCES = {
    "reuters", "bloomberg", "cnbc", "wall street journal", "wsj", "financial times",
    "marketwatch", "associated press", "ap news", "economic times", "mint",
    "business standard", "moneycontrol", "livemint", "the hindu businessline",
}
LOW_QUALITY_SOURCE_TERMS = {
    "alltoc", "reddit", "youtube", "instagram", "facebook", "blog", "medium", "substack",
}
LOW_QUALITY_HEADLINE_TERMS = {
    "#tech", "#ai", "#stocks", "viral", "hyped", "keyboard launched", "secure messaging",
    "genz", "samvadini",
}


def build_company_search_terms(company_name: str) -> list[str]:
    base_terms = COMPANY_NEWS_ALIASES.get(company_name, [])
    generic_terms = [company_name.lower()]

    cleaned = re.sub(r"[^a-zA-Z0-9\s&.-]", " ", company_name).strip().lower()
    if cleaned and cleaned not in generic_terms:
        generic_terms.append(cleaned)

    normalized_words = [word for word in re.split(r"\s+", cleaned) if len(word) > 2]
    if len(normalized_words) > 1:
        generic_terms.extend(normalized_words)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in base_terms + generic_terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return deduped


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
    query_terms = build_company_search_terms(company_name)
    query = urllib.parse.quote(f"({company_name} OR {' OR '.join(query_terms[:3])}) stock")
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


def is_relevant_financial_article(company_name: str, title: str, description: str = "", source: str = "") -> bool:
    text = f"{title} {description}".lower()
    aliases = build_company_search_terms(company_name)
    mentions_company = any(alias in text for alias in aliases)
    if not mentions_company:
        return False

    has_noisy_term = any(term in text for term in NOISY_NEWS_TERMS)
    has_finance_context = any(term in text for term in FINANCE_CONTEXT_TERMS)
    if has_noisy_term and not has_finance_context:
        return False

    source_text = str(source).lower()
    if "pypi" in source_text or "github" in source_text:
        return False
    if any(term in source_text for term in LOW_QUALITY_SOURCE_TERMS) and not any(term in source_text for term in TRUSTED_NEWS_SOURCES):
        return False
    if any(term in text for term in LOW_QUALITY_HEADLINE_TERMS):
        return False

    return True


def article_quality_score(article: dict) -> int:
    title = str(article.get("title", "")).lower()
    description = str(article.get("description", "")).lower()
    source = str(article.get("source", "")).lower()
    text = f"{title} {description}"

    score = 0
    if any(trusted in source for trusted in TRUSTED_NEWS_SOURCES):
        score += 6
    if any(term in text for term in FINANCE_CONTEXT_TERMS):
        score += 3
    if any(term in source for term in LOW_QUALITY_SOURCE_TERMS):
        score -= 4
    if any(term in text for term in LOW_QUALITY_HEADLINE_TERMS):
        score -= 5
    if "analyst" in text or "earnings" in text or "revenue" in text:
        score += 2

    return score


def infer_source_quality(articles: list[dict]) -> str:
    if any(any(trusted in str(article.get("source", "")).lower() for trusted in TRUSTED_NEWS_SOURCES) for article in articles):
        return "high"
    if articles:
        return "medium"
    return "low"


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
        if str(subreddit).lower() not in FINANCE_SUBREDDITS:
            continue
        if not is_relevant_financial_article(company_name, title, selftext, str(subreddit)):
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


def filter_relevant_articles(company_name: str, articles: list[dict]) -> list[dict]:
    filtered = [
        article
        for article in articles
        if is_relevant_financial_article(
            company_name,
            str(article.get("title", "")),
            str(article.get("description", "")),
            str(article.get("source", "")),
        )
    ]
    deduped = dedupe_articles(filtered)
    ranked = sorted(deduped, key=article_quality_score, reverse=True)
    return [article for article in ranked if article_quality_score(article) >= 0]


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
    filtered = filter_relevant_articles(company_name, combined)
    return filtered[: max(settings.news_fetch_limit, settings.reddit_post_limit)]


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
    source_quality = infer_source_quality(articles)

    summary = "Recent headlines include " + "; ".join(headlines) + "." if headlines else "Recent coverage was retrieved."
    if source_quality != "high":
        summary += " Source quality is mixed, so this summary should be treated cautiously."
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
            "summary": "No current news found.",
            "source_quality": "low",
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
                "source_quality": infer_source_quality(articles),
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
            "source_quality": infer_source_quality(articles),
        }
    else:
        logger.error("Failed to parse LLM JSON for %s", company_name)
        sentiment, confidence = infer_sentiment_from_articles(articles)
        fallback_summary = build_fallback_news_summary(company_name, articles)
        payload = {
            "articles": articles,
            "sentiment": sentiment,
            "confidence": confidence,
            "summary": fallback_summary,
            "source_quality": infer_source_quality(articles),
        }
        
    return json.dumps(payload)

@tool
def get_company_news(company_name: str) -> str:
    """Useful to fetch top recent articles along with their LLM-analyzed sentiment score and summary."""
    return safe_execute_sync(fetch_and_analyze_news, "news", company_name)
