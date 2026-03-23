import json
import asyncio
import logging
import time
import re
from typing import List, Dict, Any

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from tools.entity_extractor import extract_companies
from tools.market_data import fetch_market_logic
from tools.news import fetch_and_analyze_news
from tools.document_qa import process_documents
from utils.config import settings
from utils.llm import get_llm
from utils.safe_execution import safe_execute_sync

logger = logging.getLogger(__name__)

class AnalysisOutput(BaseModel):
    companies: List[str] = Field(description="List of companies identified.")
    market_data: Dict[str, Any] = Field(description="Market data payload extracted for companies. Provide it as a dictionary keyed by company name.")
    company_scores: Dict[str, Any] = Field(default_factory=dict, description="Per-company score and confidence summary keyed by company name.")
    news_summary: str = Field(description="A concise summary of all recent news retrieved, highlighting key events and sentiment.")
    document_insights: str = Field(description="Any insights or specific points found in the provided documents. If none found or provided, explicitly state so.")
    risk_analysis: str = Field(description="Risk analysis paragraph.")
    final_recommendation: str = Field(description="Final recommendation based on growth, risks, market positioning, and documents.")
    llm_status: str = Field(default="", description="Status message describing AI synthesis availability or fallback conditions.")
    used_fallback: bool = Field(default=False, description="Whether the response used deterministic fallback logic instead of a successful LLM synthesis.")

class AgentService:
    def __init__(self):
        self.llm = get_llm()

    def _normalize_sector_label(self, company: str, sector: Any, industry: Any) -> str:
        company_lower = str(company).lower()
        sector_text = str(sector) if isinstance(sector, str) else ""
        industry_text = str(industry) if isinstance(industry, str) else ""

        if company_lower == "meta" and sector_text.lower() == "media":
            return "Communication Services"
        if sector_text and sector_text != "Data Not Available":
            return sector_text
        if "internet content" in industry_text.lower() or "social" in industry_text.lower():
            return "Communication Services"
        return sector_text or "Data Not Available"

    def _company_has_strong_fundamentals(self, market_snapshot: Dict[str, Any]) -> bool:
        filled = 0
        for key in ("marketCap", "revenue", "revenueGrowth", "sector", "industry", "ebitda", "debtToEquity"):
            value = market_snapshot.get(key)
            if value not in (None, "", "Data Not Available", "N/A"):
                filled += 1
        return filled >= 4

    def _comparison_limitation_note(self, companies: List[str], market_data: Dict[str, Any]) -> str | None:
        if len(companies) < 2:
            return None

        weak_companies = [
            company for company in companies
            if not self._company_has_strong_fundamentals(market_data.get(company, {}) if isinstance(market_data, dict) else {})
        ]
        if not weak_companies:
            return None

        joined = ", ".join(weak_companies)
        return f"Comparison is limited because full financial data is unavailable or incomplete for {joined}."

    def _company_specific_risks(self, company: str, market_snapshot: Dict[str, Any]) -> list[str]:
        company_lower = str(company).lower()
        sector = self._normalize_sector_label(company, market_snapshot.get("sector"), market_snapshot.get("industry")).lower()
        industry = str(market_snapshot.get("industry") or "").lower()
        risks: list[str] = []

        if "meta" in company_lower or "social" in industry or "internet content" in industry:
            risks.extend([
                "advertising demand can weaken if marketers reduce spending",
                "AI infrastructure spending can pressure margins if returns take time to materialize",
                "privacy and platform regulation can affect monetization and user growth",
            ])
        elif "reliance" in company_lower or "energy" in sector or "oil" in industry or "refin" in industry:
            risks.extend([
                "energy and commodity price swings can affect refining and petrochemical profitability",
                "large capex programs can delay returns if execution or demand weakens",
                "conglomerate complexity can make capital allocation harder to evaluate",
            ])
        elif "communication services" in sector:
            risks.extend([
                "platform engagement and monetization can change quickly with competition and regulation",
            ])

        # Keep the list short and readable.
        return risks[:3]

    def _compute_company_score(self, company: str, market_snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        confidence = 35
        notes: list[str] = []
        growth_score = 20
        balance_score = 10
        sentiment_score = 10
        data_quality_score = 10

        revenue_growth = market_snapshot.get("revenueGrowth")
        debt_to_equity = market_snapshot.get("debtToEquity")
        source_quality = news_snapshot.get("source_quality", "low")
        sentiment = news_snapshot.get("sentiment", "neutral")

        filled_fields = 0
        for key in ("marketCap", "revenue", "revenueGrowth", "sector", "industry", "ebitda", "debtToEquity"):
            if market_snapshot.get(key) not in (None, "", "Data Not Available", "N/A"):
                filled_fields += 1

        confidence += min(35, filled_fields * 8)
        data_quality_score = min(20, max(0, filled_fields * 3))
        if isinstance(revenue_growth, (int, float)):
            if revenue_growth >= 0.15:
                growth_score = 38
                notes.append("strong growth")
            elif revenue_growth >= 0.05:
                growth_score = 30
                notes.append("positive growth")
            elif revenue_growth < 0:
                growth_score = 6
                notes.append("negative growth")
            else:
                growth_score = 22
                notes.append("modest growth")
        else:
            growth_score = 8
            confidence -= 12
            notes.append("growth data incomplete")

        if isinstance(debt_to_equity, (int, float)):
            if debt_to_equity <= 1:
                balance_score = 18
                notes.append("low leverage")
            elif debt_to_equity <= 3:
                balance_score = 14
                notes.append("manageable leverage")
            else:
                balance_score = 6
                notes.append("elevated leverage")
        else:
            balance_score = 8
            confidence -= 10
            notes.append("balance sheet data incomplete")

        if sentiment == "positive":
            sentiment_score = 16
            notes.append("supportive news tone")
        elif sentiment == "negative":
            sentiment_score = 5
            notes.append("negative news tone")
        else:
            sentiment_score = 10

        if source_quality == "high":
            confidence += 12
            sentiment_score = min(20, sentiment_score + 2)
            data_quality_score = min(20, data_quality_score + 4)
        elif source_quality == "medium":
            confidence += 4
            data_quality_score = min(20, data_quality_score + 1)
        else:
            confidence -= 8
            sentiment_score = max(0, sentiment_score - 2)
            data_quality_score = max(0, data_quality_score - 4)
            notes.append("mixed or weak news sources")

        if not self._company_has_strong_fundamentals(market_snapshot):
            data_quality_score = max(0, data_quality_score - 6)
            confidence -= 15
            notes.append("limited fundamental coverage")

        score = growth_score + balance_score + sentiment_score + data_quality_score
        score = max(0, min(100, int(round(score))))
        confidence = max(10, min(95, int(round(confidence))))

        if score >= 75:
            stance = "strong"
        elif score >= 60:
            stance = "constructive"
        elif score >= 45:
            stance = "watchful"
        else:
            stance = "avoid decision"

        if score >= 75 and confidence >= 70:
            decision_tag = "Growth + Quality"
        elif score >= 60:
            decision_tag = "Balanced Opportunity"
        elif confidence < 45:
            decision_tag = "Insufficient Data"
        elif isinstance(revenue_growth, (int, float)) and revenue_growth < 0:
            decision_tag = "Turnaround Risk"
        else:
            decision_tag = "Watchlist Candidate"

        return {
            "score": score,
            "confidence": confidence,
            "stance": stance,
            "notes": notes[:4],
            "decision_tag": decision_tag,
            "breakdown": {
                "growth": growth_score,
                "balance_sheet": balance_score,
                "sentiment": sentiment_score,
                "data_quality": data_quality_score,
            },
        }

    def _build_company_scores(self, companies: List[str], market_data: Dict[str, Any], news_data: Dict[str, Any]) -> Dict[str, Any]:
        scored_entries: list[tuple[str, Dict[str, Any]]] = []
        for company in companies:
            market_snapshot = market_data.get(company, {}) if isinstance(market_data, dict) else {}
            news_snapshot = news_data.get(company, {}) if isinstance(news_data, dict) else {}
            scored_entries.append((company, self._compute_company_score(company, market_snapshot, news_snapshot)))

        ranked = sorted(
            scored_entries,
            key=lambda item: (item[1].get("score", 0), item[1].get("confidence", 0)),
            reverse=True,
        )

        rank_lookup = {company: index + 1 for index, (company, _) in enumerate(ranked)}
        scorecard: Dict[str, Any] = {}
        for company, snapshot in scored_entries:
            scorecard[company] = {
                **snapshot,
                "rank": rank_lookup[company],
            }
        return scorecard

    def _build_positioning_hint(self, company: str, market_snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> str | None:
        sector = self._normalize_sector_label(company, market_snapshot.get("sector"), market_snapshot.get("industry"))
        industry = market_snapshot.get("industry")
        sentiment = news_snapshot.get("sentiment", "neutral")
        articles = news_snapshot.get("articles", []) if isinstance(news_snapshot, dict) else []

        parts: list[str] = []
        if isinstance(industry, str) and industry != "Data Not Available":
            parts.append(f"its positioning is tied closely to execution in {industry.lower()}")
        elif isinstance(sector, str) and sector != "Data Not Available":
            parts.append(f"its positioning depends on how well it executes within the {sector.lower()} sector")

        if articles:
            sources = [str(article.get("source", "")).strip() for article in articles[:2] if article.get("source")]
            if sources:
                parts.append(f"recent coverage from {', '.join(sources)} suggests investors are watching execution closely")

        if sentiment == "positive":
            parts.append("recent sentiment offers some support, but it still needs operational follow-through")
        elif sentiment == "negative":
            parts.append("negative sentiment increases pressure on management to defend the current thesis")

        if not parts:
            return f"{company}'s competitive position depends on product execution, pricing discipline, and its ability to defend demand in its end markets."

        return " ".join(parts).strip().rstrip(".") + "."

    def _build_risk_hint(self, company: str, market_snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> str | None:
        sector = self._normalize_sector_label(company, market_snapshot.get("sector"), market_snapshot.get("industry"))
        industry = market_snapshot.get("industry")
        sentiment = news_snapshot.get("sentiment", "neutral")
        articles = news_snapshot.get("articles", []) if isinstance(news_snapshot, dict) else []

        parts: list[str] = []
        if isinstance(industry, str) and industry != "Data Not Available":
            parts.append(f"{industry.lower()} demand can change quickly if customers delay spending or competitors gain share")
        elif isinstance(sector, str) and sector != "Data Not Available":
            parts.append(f"{sector.lower()} conditions can shift quickly with pricing pressure, regulation, or weaker demand")

        if sentiment == "negative":
            parts.append("recent negative news increases near-term execution and reputation risk")
        elif sentiment == "positive":
            parts.append("even supportive recent news can reverse quickly if expectations become too high")

        if articles:
            parts.append("headline-driven sentiment adds volatility, especially when the fundamental picture is incomplete")

        return (" ".join(parts).strip().rstrip(".") + ".") if parts else None
        
    def _validate_data(self, parsed: AnalysisOutput) -> AnalysisOutput:
        """Data Validation Layer preventing critical empty logic structure cleanly matching fallbacks."""
        if settings.fallback_mode == "graceful":
            if not parsed.companies:
                parsed.companies = ["Generic Market Search"]
            if not parsed.market_data:
                parsed.market_data = {"error": "Failed to explicitly gather deep financial data bounds natively."}
            if parsed.company_scores is None:
                parsed.company_scores = {}
            if not parsed.news_summary or len(parsed.news_summary) < 5:
                parsed.news_summary = "No significant news summary could be formulated properly."
            if not parsed.risk_analysis or len(parsed.risk_analysis) < 5:
                parsed.risk_analysis = "Insufficient data points gathered to establish analytical risk bounds."
            if not parsed.final_recommendation or len(parsed.final_recommendation) < 5:
                parsed.final_recommendation = "Partial evaluation. Retrying recommended based strictly on incomplete extractions."
        return parsed

    def _parse_json_payload(self, raw_payload: str, source: str, company_name: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode %s payload for %s: %s", source, company_name, exc)

        return {
            "error": f"Failed to parse {source} payload.",
            "raw": raw_payload[:500],
        }

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any] | None:
        if not raw_text:
            return None

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            logger.warning("Failed to salvage JSON object from LLM output: %s", exc)

        return None

    async def _gather_company_context(self, companies: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        market_semaphore = asyncio.Semaphore(1)

        async def gather(company_name: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
            news_task = asyncio.to_thread(safe_execute_sync, fetch_and_analyze_news, "news", company_name)

            async with market_semaphore:
                market_raw = await asyncio.to_thread(safe_execute_sync, fetch_market_logic, "market_data", company_name)

            news_raw = await news_task

            market_payload = self._parse_json_payload(market_raw, "market_data", company_name)
            news_payload = self._parse_json_payload(news_raw, "news", company_name)
            return company_name, market_payload, news_payload

        results = await asyncio.gather(*(gather(company) for company in companies))

        market_data: Dict[str, Any] = {}
        news_data: Dict[str, Any] = {}
        for company_name, market_payload, news_payload in results:
            market_data[company_name] = market_payload
            news_data[company_name] = news_payload

        return market_data, news_data

    def _summarize_documents(self, doc_store, query: str) -> str:
        if not doc_store:
            return "No documents were provided."

        try:
            docs = doc_store.similarity_search(query, k=3)
        except Exception as exc:
            logger.error("Document search failed: %s", exc)
            return "Document retrieval failed."

        if not docs:
            return "No relevant document insights were found."

        snippets = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
        if not snippets:
            return "No relevant document insights were found."

        return "\n\n".join(snippets[:3])

    def _sanitize_llm_error(self, error: Exception) -> str:
        message = str(error)
        lowered = message.lower()

        if "not found for api version" in lowered or "is not supported for generatecontent" in lowered:
            return "AI synthesis is temporarily unavailable because the configured Gemini model name is not available for this API key. Switch to a supported model such as gemini-2.5-flash or gemini-2.5-pro."
        if "resourceexhausted" in lowered or "quota exceeded" in lowered or "exceeded your current quota" in lowered:
            return "AI synthesis is temporarily unavailable because the Gemini API quota for this project is exhausted. Wait for quota reset, use a different project/key, or enable billing."
        if "reported as leaked" in lowered or "api key was reported as leaked" in lowered:
            return "AI synthesis is temporarily unavailable because the configured Gemini API key has been blocked as leaked. Replace it with a newly generated key."
        if "accessdenied" in lowered or "not authorized" in lowered or "invokemodel" in lowered:
            return "AI synthesis is temporarily unavailable because the configured AWS Bedrock credentials do not have permission to invoke the selected model."
        if "security token" in lowered or "invalidclienttokenid" in lowered:
            return "AI synthesis is temporarily unavailable because the configured AWS credentials are invalid or expired."
        if "api key" in lowered or "google" in lowered and "credential" in lowered:
            return "AI synthesis is temporarily unavailable because the configured Gemini API key is invalid or missing."
        return "AI synthesis is temporarily unavailable right now."

    def _build_fallback_recommendation(
        self,
        companies: List[str],
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        document_insights: str,
        llm_status: str,
    ) -> str:
        company = companies[0] if companies else "the requested company"
        market_snapshot = market_data.get(company, {}) if market_data else {}
        news_snapshot = news_data.get(company, {}) if news_data else {}

        price = market_snapshot.get("currentPrice", "Data Not Available")
        revenue_growth = market_snapshot.get("revenueGrowth", "Data Not Available")
        sentiment = news_snapshot.get("sentiment", "neutral")

        recommendation = (
            f"{llm_status} Review the raw data for {company} before making any decision. "
            f"Current price: {price}. Revenue growth: {revenue_growth}. News sentiment: {sentiment}."
        )
        if document_insights and document_insights != "No documents were provided.":
            recommendation += " Uploaded documents were processed and can provide additional context."

        return recommendation

    def _build_grounded_news_summary(self, companies: List[str], news_data: Dict[str, Any]) -> str:
        if not companies:
            return "No companies were identified, so no company-specific news summary could be prepared."

        lines: list[str] = []
        for company in companies:
            payload = news_data.get(company, {})
            articles = payload.get("articles", []) if isinstance(payload, dict) else []
            summary = payload.get("summary") if isinstance(payload, dict) else None
            sentiment = payload.get("sentiment", "neutral") if isinstance(payload, dict) else "neutral"
            source_quality = payload.get("source_quality", "low") if isinstance(payload, dict) else "low"

            if summary and "failed compiling sentiment" not in summary.lower():
                quality_note = " Source reliability looks limited." if source_quality != "high" else ""
                lines.append(f"- {company}: News tone looks {sentiment}. {summary}{quality_note}")
                continue

            if articles:
                titles = [article.get("title") for article in articles[:2] if article.get("title")]
                if titles:
                    joined_titles = " and ".join(titles)
                    lines.append(f"- {company}: News tone looks {sentiment}. Recent coverage mentions {joined_titles}.")
                    continue

            lines.append(f"- {company}: No meaningful recent news was found, so the news tone stays {sentiment}.")

        return "\n".join(lines)

    def _build_grounded_risk_analysis(
        self,
        companies: List[str],
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        document_insights: str,
    ) -> str:
        if not companies:
            return "Risk analysis is limited because no specific company was identified."

        paragraphs: list[str] = []
        for company in companies:
            market_snapshot = market_data.get(company, {}) if isinstance(market_data, dict) else {}
            news_snapshot = news_data.get(company, {}) if isinstance(news_data, dict) else {}

            revenue_growth = market_snapshot.get("revenueGrowth")
            debt_to_equity = market_snapshot.get("debtToEquity")
            sector = self._normalize_sector_label(company, market_snapshot.get("sector"), market_snapshot.get("industry"))
            sentiment = news_snapshot.get("sentiment", "neutral")
            has_news = bool(news_snapshot.get("articles"))
            source_quality = news_snapshot.get("source_quality", "low")
            price = market_snapshot.get("currentPrice", "Data Not Available")
            specific_risks = self._company_specific_risks(company, market_snapshot)

            missing_growth = not isinstance(revenue_growth, (int, float))
            missing_debt = not isinstance(debt_to_equity, (int, float))
            missing_sector = not isinstance(sector, str) or sector == "Data Not Available"

            if missing_growth and missing_debt and missing_sector:
                summary = f"- {company}: Detailed fundamentals were not available from the live market feed"
                if price != "Data Not Available":
                    summary += f", although the latest traded price was {price}"
                summary += ". This lowers confidence, so the biggest risk is making a decision without enough financial detail."
                if sentiment == "negative":
                    summary += " Recent news tone is negative, which adds pressure."
                elif sentiment == "neutral":
                    summary += " Recent news tone is neutral."
                if source_quality != "high":
                    summary += " News reliability is mixed, so headline signals should be treated cautiously."
                if specific_risks:
                    summary += " Key business risks include " + "; ".join(specific_risks) + "."
                extra_risk = self._build_risk_hint(company, market_snapshot, news_snapshot)
                if extra_risk:
                    summary += f" {extra_risk}"
                paragraphs.append(summary)
                continue

            company_risks: list[str] = []
            if isinstance(revenue_growth, (int, float)):
                if revenue_growth < 0:
                    company_risks.append(f"revenue is shrinking, with growth at {revenue_growth:.3f}")
                elif revenue_growth < 0.05:
                    company_risks.append(f"revenue growth is modest at {revenue_growth:.3f}")
                else:
                    company_risks.append(f"revenue growth is healthy at {revenue_growth:.3f}, but maintaining that pace may be difficult")
            else:
                company_risks.append("revenue growth data is incomplete")

            if isinstance(debt_to_equity, (int, float)):
                if debt_to_equity > 100:
                    company_risks.append(f"debt is high, with debt-to-equity at {debt_to_equity:.2f}")
                elif debt_to_equity > 30:
                    company_risks.append(f"debt-to-equity at {debt_to_equity:.2f} is worth watching")
            else:
                company_risks.append("balance sheet data is incomplete")

            if sentiment == "negative":
                company_risks.append("recent news sentiment is negative")
            elif not has_news:
                company_risks.append("there is not much recent news to confirm or challenge the investment case")
            elif source_quality != "high":
                company_risks.append("recent news coverage is available but source quality is mixed")

            if not missing_sector:
                company_risks.append(f"the {sector} sector can change quickly because of competition, execution risk, and shifts in demand")
            else:
                company_risks.append("competition, execution risk, and demand shifts can still change the outlook quickly")
            company_risks.extend(self._company_specific_risks(company, market_snapshot)[:2])
            sentence = f"- {company}: The main risks are " + ", ".join(company_risks) + "."
            extra_risk = self._build_risk_hint(company, market_snapshot, news_snapshot)
            if extra_risk:
                sentence += f" {extra_risk}"
            paragraphs.append(sentence)

        if document_insights == "No documents were provided.":
            paragraphs.append("- Note: No supporting documents were uploaded, so this view is based only on market data and recent news.")

        limitation_note = self._comparison_limitation_note(companies, market_data)
        if limitation_note:
            paragraphs.append(f"- Note: {limitation_note}")

        return "\n".join(paragraphs)

    def _build_grounded_recommendation(
        self,
        companies: List[str],
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        document_insights: str,
    ) -> str:
        if not companies:
            return "No investable recommendation can be made because no specific company was identified."

        recommendations: list[str] = []
        for company in companies:
            market_snapshot = market_data.get(company, {}) if isinstance(market_data, dict) else {}
            news_snapshot = news_data.get(company, {}) if isinstance(news_data, dict) else {}
            score_snapshot = self._compute_company_score(company, market_snapshot, news_snapshot)
            price = market_snapshot.get("currentPrice", "N/A")
            revenue_growth = market_snapshot.get("revenueGrowth")
            debt_to_equity = market_snapshot.get("debtToEquity")
            sentiment = news_snapshot.get("sentiment", "neutral")
            source_quality = news_snapshot.get("source_quality", "low")
            sector = self._normalize_sector_label(company, market_snapshot.get("sector"), market_snapshot.get("industry"))

            stance = "watchlist"
            reason_parts: list[str] = []
            missing_growth = not isinstance(revenue_growth, (int, float))
            missing_debt = not isinstance(debt_to_equity, (int, float))
            missing_sector = not isinstance(sector, str) or sector == "Data Not Available"

            if missing_growth and missing_debt and missing_sector:
                recommendations.append(
                    f"- {company}: Current view is avoid decision for now. Price is {price}. "
                    f"Score: {score_snapshot['score']}/100. Confidence: {score_snapshot['confidence']}%. "
                    f"Reason: only limited live market data was available, so this is a low-confidence view, and recent news tone is {sentiment}."
                )
                continue

            if isinstance(revenue_growth, (int, float)) and revenue_growth >= 0.08:
                stance = "constructive"
                reason_parts.append(f"revenue growth is solid at {revenue_growth:.3f}")
            elif isinstance(revenue_growth, (int, float)) and revenue_growth < 0:
                stance = "cautious"
                reason_parts.append(f"revenue growth is negative at {revenue_growth:.3f}")
            else:
                reason_parts.append("growth visibility is limited")

            if isinstance(debt_to_equity, (int, float)) and debt_to_equity > 30:
                reason_parts.append(f"debt is notable with debt-to-equity at {debt_to_equity:.2f}")

            if sentiment == "negative":
                stance = "cautious"
                reason_parts.append("recent news tone is negative")
            elif sentiment == "positive":
                reason_parts.append("recent news tone is supportive")
            else:
                reason_parts.append("recent news tone is neutral")
            if source_quality != "high":
                reason_parts.append("news-source quality is mixed")

            positioning_hint = self._build_positioning_hint(company, market_snapshot, news_snapshot)
            qualitative_note = positioning_hint if positioning_hint else ""

            reason_text = ", and ".join(reason_parts) + "."
            recommendation = (
                f"- {company}: Current view is {stance}. "
                f"Price is {price}. "
                f"Score: {score_snapshot['score']}/100. Confidence: {score_snapshot['confidence']}%. "
                f"Reason: {reason_text}"
            )
            if qualitative_note:
                recommendation += f" {qualitative_note}"
            recommendations.append(recommendation)

        if document_insights != "No documents were provided.":
            recommendations.append("- Note: Uploaded documents should be reviewed before making a decision because they may change the conclusion.")

        limitation_note = self._comparison_limitation_note(companies, market_data)
        if limitation_note:
            recommendations.append(f"- Note: {limitation_note}")

        return "\n".join(recommendations)

    def _parse_synthesis_response(
        self,
        parser: PydanticOutputParser,
        raw_content: str,
        companies: List[str],
        market_data: Dict[str, Any],
        news_data: Dict[str, Any],
        document_insights: str,
    ) -> AnalysisOutput:
        company_scores = self._build_company_scores(companies, market_data, news_data)
        try:
            parsed = parser.parse(raw_content)
            parsed.companies = companies
            parsed.market_data = market_data
            parsed.company_scores = company_scores
            if not parsed.document_insights:
                parsed.document_insights = document_insights
            return self._validate_data(parsed)
        except Exception as exc:
            logger.warning("Structured parse failed, attempting salvage parse: %s", exc)

        partial = self._extract_json_object(raw_content) or {}
        parsed = AnalysisOutput(
            companies=companies,
            market_data=market_data,
            company_scores=company_scores,
            news_summary=partial.get("news_summary") or self._build_grounded_news_summary(companies, news_data),
            document_insights=partial.get("document_insights") or document_insights,
            risk_analysis=partial.get("risk_analysis") or self._build_grounded_risk_analysis(companies, market_data, news_data, document_insights),
            final_recommendation=partial.get("final_recommendation") or self._build_grounded_recommendation(companies, market_data, news_data, document_insights),
        )
        return self._validate_data(parsed)

    async def process_query(self, query: str, documents=None) -> dict:
        start_time = time.time()
        logger.info(f"Incoming query: {query}")
        
        # 1. Entity Extraction securely wrapped in timeouts
        companies = await extract_companies(query, self.llm)
        if not companies:
            logger.warning("No specific companies detected natively formatting to default fallback context.")
            companies = ["General Market"]
            
        logger.info(f"Extracted entities: {companies}")

        # 2. Parse documents safely mapping FAISS settings
        doc_store = None
        if documents and hasattr(documents[0], "file") and documents[0].filename != "":
            try:
                task = asyncio.create_task(process_documents(documents))
                doc_store = await asyncio.wait_for(task, timeout=settings.request_timeout * 3)
                logger.info("Documents processed successfully via FAISS.")
            except asyncio.TimeoutError:
                logger.error("Document parsing FAISS indexing severely timed out skipping processing.")
        
        market_data, news_data = await self._gather_company_context(companies)
        document_insights = self._summarize_documents(doc_store, query)

        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
        synthesis_payload = {
            "query": query,
            "companies": companies,
            "market_data": market_data,
            "company_scores": self._build_company_scores(companies, market_data, news_data),
            "news_data": news_data,
            "document_insights": document_insights,
        }

        synthesis_prompt = f"""
        You are a careful financial analysis assistant.
        Use the supplied research context to generate one valid JSON object that follows the required schema exactly.

        Rules:
        - Keep "companies" exactly as provided.
        - Keep "market_data" as the provided dictionary keyed by company name.
        - Keep "company_scores" aligned with the supplied scorecard.
        - Return only JSON. Do not add markdown, commentary, or code fences.
        - Every required field must be present.
        - "news_summary" should synthesize the company news payloads into a concise readable summary with no hype.
        - "document_insights" should summarize the supplied document context, or clearly state that no documents were provided.
        - "risk_analysis" should explain downside risks, uncertainty, and data quality caveats using the supplied data only.
        - "final_recommendation" should be balanced, practical, and should not claim certainty.
        - If the news payload says no current news was found, say that plainly instead of inventing events.
        - Do not leave any field blank.

        Research context:
        {json.dumps(synthesis_payload, ensure_ascii=True, default=str)}

        {parser.get_format_instructions()}
        """

        logger.info("Executing %s synthesis...", settings.model_provider.strip().lower())
        try:
            text_output = await asyncio.wait_for(
                self.llm.ainvoke(synthesis_prompt),
                timeout=settings.request_timeout * 4,
            )
            raw_content = getattr(text_output, "content", "")
            parsed = self._parse_synthesis_response(
                parser,
                raw_content,
                companies,
                market_data,
                news_data,
                document_insights,
            )
            parsed.llm_status = ""
            parsed.used_fallback = False
            logger.info(f"Request strictly formatted successfully in {time.time() - start_time:.2f}s")
            return parsed.model_dump()
        except asyncio.TimeoutError:
            logger.error("%s synthesis timed out.", settings.model_provider.strip().lower())
            llm_status = "AI synthesis timed out."
        except Exception as e:
            logger.error("%s synthesis failed: %s", settings.model_provider.strip().lower(), e)
            llm_status = self._sanitize_llm_error(e)
        else:
            llm_status = ""
        
        if settings.response_mode == "plaintext":
            logger.info("Processing natively bypassing structured extraction directly via plaintext configs.")
            fallback = AnalysisOutput(
                companies=companies,
                market_data=market_data,
                company_scores=self._build_company_scores(companies, market_data, news_data),
                news_summary="Plaintext mode enabled. See final recommendation.",
                document_insights=document_insights,
                risk_analysis="Plaintext mode enabled. See final recommendation.",
                final_recommendation=self._build_fallback_recommendation(
                    companies, market_data, news_data, document_insights, llm_status or "Plaintext mode is enabled."
                ),
                llm_status=llm_status or "Plaintext mode is enabled.",
                used_fallback=True,
            )
            return self._validate_data(fallback).model_dump()

        fallback = AnalysisOutput(
            companies=companies,
            market_data=market_data,
            company_scores=self._build_company_scores(companies, market_data, news_data),
            news_summary=self._build_grounded_news_summary(companies, news_data),
            document_insights=document_insights,
            risk_analysis=self._build_grounded_risk_analysis(companies, market_data, news_data, document_insights),
            final_recommendation=self._build_grounded_recommendation(companies, market_data, news_data, document_insights),
            llm_status=llm_status,
            used_fallback=True,
        )
        return self._validate_data(fallback).model_dump()

    def _build_news_summary(self, news_data: Dict[str, Any]) -> str:
        if not news_data:
            return "No significant news summary could be formulated properly."

        lines = []
        for company_name, payload in news_data.items():
            summary = payload.get("summary", "No current news found.")
            sentiment = payload.get("sentiment", "neutral")
            lines.append(f"{company_name}: {summary} Sentiment: {sentiment}.")

        return " ".join(lines)
