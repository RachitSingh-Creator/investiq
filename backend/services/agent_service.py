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
    news_summary: str = Field(description="A concise summary of all recent news retrieved, highlighting key events and sentiment.")
    document_insights: str = Field(description="Any insights or specific points found in the provided documents. If none found or provided, explicitly state so.")
    risk_analysis: str = Field(description="Risk analysis paragraph.")
    final_recommendation: str = Field(description="Final recommendation based on growth, risks, market positioning, and documents.")

class AgentService:
    def __init__(self):
        self.llm = get_llm()
        
    def _validate_data(self, parsed: AnalysisOutput) -> AnalysisOutput:
        """Data Validation Layer preventing critical empty logic structure cleanly matching fallbacks."""
        if settings.fallback_mode == "graceful":
            if not parsed.companies:
                parsed.companies = ["Generic Market Search"]
            if not parsed.market_data:
                parsed.market_data = {"error": "Failed to explicitly gather deep financial data bounds natively."}
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
        async def gather(company_name: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
            market_raw, news_raw = await asyncio.gather(
                asyncio.to_thread(safe_execute_sync, fetch_market_logic, "market_data", company_name),
                asyncio.to_thread(safe_execute_sync, fetch_and_analyze_news, "news", company_name),
            )

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

            if summary and "failed compiling sentiment" not in summary.lower():
                lines.append(f"- {company}: News tone looks {sentiment}. {summary}")
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
            sector = market_snapshot.get("sector", "its sector")
            sentiment = news_snapshot.get("sentiment", "neutral")
            has_news = bool(news_snapshot.get("articles"))
            price = market_snapshot.get("currentPrice", "Data Not Available")

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

            if not missing_sector:
                company_risks.append(f"the {sector} sector can change quickly because of competition, execution risk, and shifts in demand")
            else:
                company_risks.append("competition, execution risk, and demand shifts can still change the outlook quickly")
            paragraphs.append(f"- {company}: The main risks are " + ", ".join(company_risks) + ".")

        if document_insights == "No documents were provided.":
            paragraphs.append("- Note: No supporting documents were uploaded, so this view is based only on market data and recent news.")

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

            price = market_snapshot.get("currentPrice", "N/A")
            revenue_growth = market_snapshot.get("revenueGrowth")
            debt_to_equity = market_snapshot.get("debtToEquity")
            sentiment = news_snapshot.get("sentiment", "neutral")
            sector = market_snapshot.get("sector", "Data Not Available")

            stance = "watchlist"
            reason_parts: list[str] = []
            missing_growth = not isinstance(revenue_growth, (int, float))
            missing_debt = not isinstance(debt_to_equity, (int, float))
            missing_sector = not isinstance(sector, str) or sector == "Data Not Available"

            if missing_growth and missing_debt and missing_sector:
                recommendations.append(
                    f"- {company}: Current view is watchlist. Price is {price}. "
                    f"Reason: only limited live market data was available, so this is a low-confidence view, and recent news tone is {sentiment}."
                )
                continue

            if isinstance(revenue_growth, (int, float)) and revenue_growth >= 0.08:
                stance = "cautiously positive"
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

            recommendations.append(
                f"- {company}: Current view is {stance}. "
                f"Price is {price}. "
                f"Reason: " + ", and ".join(reason_parts) + "."
            )

        if document_insights != "No documents were provided.":
            recommendations.append("- Note: Uploaded documents should be reviewed before making a decision because they may change the conclusion.")

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
        try:
            parsed = parser.parse(raw_content)
            parsed.companies = companies
            parsed.market_data = market_data
            if not parsed.document_insights:
                parsed.document_insights = document_insights
            return self._validate_data(parsed)
        except Exception as exc:
            logger.warning("Structured parse failed, attempting salvage parse: %s", exc)

        partial = self._extract_json_object(raw_content) or {}
        parsed = AnalysisOutput(
            companies=companies,
            market_data=market_data,
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
            "news_data": news_data,
            "document_insights": document_insights,
        }

        synthesis_prompt = f"""
        You are a careful financial analysis assistant.
        Use the supplied research context to generate one valid JSON object that follows the required schema exactly.

        Rules:
        - Keep "companies" exactly as provided.
        - Keep "market_data" as the provided dictionary keyed by company name.
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
                news_summary="Plaintext mode enabled. See final recommendation.",
                document_insights=document_insights,
                risk_analysis="Plaintext mode enabled. See final recommendation.",
                final_recommendation=self._build_fallback_recommendation(
                    companies, market_data, news_data, document_insights, llm_status or "Plaintext mode is enabled."
                )
            )
            return self._validate_data(fallback).model_dump()

        fallback = AnalysisOutput(
            companies=companies,
            market_data=market_data,
            news_summary=self._build_grounded_news_summary(companies, news_data),
            document_insights=document_insights,
            risk_analysis=self._build_grounded_risk_analysis(companies, market_data, news_data, document_insights),
            final_recommendation=self._build_grounded_recommendation(companies, market_data, news_data, document_insights),
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
