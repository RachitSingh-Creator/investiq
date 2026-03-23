# InvestIQ AI - Investment Research Agent

InvestIQ AI is a production-oriented investment research assistant built with FastAPI, LangChain-compatible LLM orchestration, a React frontend, and multiple live market/news/document pipelines. The system is designed to be uncertainty-aware: it prefers grounded data, uses provider fallbacks, flags incomplete coverage, and avoids inventing fundamentals when the source data is thin.

## Live Frontend

- Frontend URL: https://investiq-frontend-gpko.onrender.com/

## End-to-End Pipeline

### 1. User Intake
- The frontend sends a query plus optional uploaded files to the backend `/analyze` endpoint.
- Queries can be comparisons such as `Compare Meta and Reliance on growth, competitive positioning, and risks`.
- Documents are optional and are only used to enrich the analysis if relevant content can be extracted.

### 2. Entity Extraction
- File: [backend/tools/entity_extractor.py](d:/investiq/backend/tools/entity_extractor.py)
- Purpose: identify the companies/entities mentioned in the query.
- Output: a normalized list of company names such as `["Meta", "Reliance"]`.

### 3. Parallel Company Context Gathering
- File: [backend/services/agent_service.py](d:/investiq/backend/services/agent_service.py)
- Purpose: for each extracted company, gather:
  - market data
  - news data
  - later combine that with document insights
- Market data and news are gathered per company, then merged into the final analysis context.

## Market Data Pipeline

### Ticker Resolution
- File: [backend/tools/ticker_resolver.py](d:/investiq/backend/tools/ticker_resolver.py)
- Purpose: map a company name to the most likely tradeable ticker.
- Provider used: Yahoo Finance search endpoint.
- Logic:
  - scores candidate tickers based on symbol match, company name match, exchange, and region hints
  - prefers NSE listings for common Indian companies when the user did not explicitly request a symbol-like ticker
  - examples:
    - `Reliance` -> likely `RELIANCE.NS`
    - `TCS` -> prefers `TCS.NS` over `TCS.BO` when appropriate

### Market Data Provider Order
- File: [backend/tools/market_data.py](d:/investiq/backend/tools/market_data.py)
- Purpose: fetch live quote/fundamental data with fallbacks.

#### Base Quote Layer
- Yahoo Finance `quote` endpoint
  - used for current price, company name, currency, market cap when available
- Yahoo Finance `chart` endpoint
  - used as a fallback for current price from chart metadata or recent close
- Yahoo Finance `quoteSummary` endpoint
  - used for:
    - total revenue
    - revenue growth
    - EBITDA
    - sector
    - industry
    - debt to equity

#### India-First Fundamentals Layer
- EODHD Fundamentals API
  - used as the primary fundamentals provider for Indian stocks (`.NS`, `.BO`)
  - tries multiple exchange variants:
    - `.NSE`
    - `.BSE`
    - base symbol fallback
  - fields extracted:
    - market cap
    - revenue
    - revenue growth
    - sector
    - industry
    - EBITDA
    - debt to equity
  - parser is case-insensitive and checks multiple field-name variants to tolerate provider payload differences

#### Additional Fallback Providers
- Finnhub
  - used for profile, quote, and some metrics when configured
  - useful for:
    - current price
    - market cap
    - industry/profile signals
    - some growth and leverage metrics
- Alpha Vantage
  - used as another fallback for fundamentals, including Indian-symbol variants
  - useful for:
    - market cap
    - revenue TTM
    - quarterly revenue growth
    - sector
    - industry
    - EBITDA
- NSEPython / NSEPythonServer
  - India-specific quote and metadata fallback
  - mainly helps for:
    - current price
    - sector / industry style metadata
    - some market-cap-style metadata
  - not treated as the primary full-fundamentals source
- yfinance
  - final fallback used to extract:
    - info-based market cap / sector / industry / EBITDA
    - revenue from statements
    - revenue growth from info or statement deltas
    - debt/equity from balance-sheet context

### Market Data Safety / Validation
- Revenue growth sanity filter:
  - obviously broken values such as `-293%` are rejected rather than displayed
- Currency inference:
  - symbol suffixes such as `.NS` / `.BO` map to `INR`
- Merge logic:
  - the pipeline fills missing fields instead of overwriting already valid values
- Provider diagnostics:
  - the backend internally tracks which provider filled which fields

## News Pipeline

### News Sources
- File: [backend/tools/news.py](d:/investiq/backend/tools/news.py)
- Primary source:
  - NewsAPI if configured
- Fallback source:
  - Google News RSS
- Additional sentiment context:
  - selected finance-oriented Reddit search results if enabled

### News Filtering
- The system filters for company relevance using:
  - company aliases
  - finance-related context terms
  - source heuristics
- It rejects noisy content such as:
  - package/library release posts
  - obvious low-quality or irrelevant headlines
  - GitHub / PyPI style technical noise
- It ranks articles using:
  - trusted-source matches
  - finance-term density
  - low-quality penalties

### News Quality Handling
- The system computes `source_quality`:
  - `high`
  - `medium`
  - `low`
- This is used later in:
  - news summary wording
  - risk analysis caveats
  - recommendation confidence language
  - scorecard confidence scoring

### News Sentiment
- If LLM news synthesis works:
  - the LLM returns structured JSON with sentiment, confidence, and a short summary
- If LLM synthesis fails:
  - fallback sentiment is inferred from headline terms
  - a grounded summary is generated from the top headlines only

## Document Pipeline

### Document Processing
- File: [backend/tools/document_qa.py](d:/investiq/backend/tools/document_qa.py)
- Purpose: parse uploaded PDFs/text files and build a local searchable document store.
- Vector store:
  - FAISS
- Embeddings:
  - sentence-transformers model configured through environment variables

### Document Usage
- Relevant chunks are retrieved for the user query.
- The final analysis references document insights only if useful content is found.
- If no documents are uploaded, the system states that clearly.

## Analysis / Reasoning Pipeline

### Grounded Analysis Builder
- File: [backend/services/agent_service.py](d:/investiq/backend/services/agent_service.py)
- Purpose: transform raw market/news/document payloads into:
  - news summary
  - risk analysis
  - final recommendation
  - company scorecard

### Risk Analysis
- Uses:
  - revenue growth
  - debt/equity
  - sector / industry
  - news sentiment
  - news source quality
  - company-specific risk heuristics when enough context exists
- Example patterns:
  - ad-driven/platform businesses -> advertising, AI spend, regulatory risk
  - energy/conglomerate businesses -> commodity, capex, capital allocation risk

### Recommendation Logic
- Produces grounded stances such as:
  - `constructive`
  - `watchful`
  - `cautious`
  - `avoid decision for now`
- If fundamentals are incomplete:
  - the system explicitly marks the view as low-confidence instead of pretending completeness
- If multiple companies are compared and one side lacks data:
  - a comparison limitation note is added

## Scorecard Pipeline

### What the Scorecard Uses
- File: [backend/services/agent_service.py](d:/investiq/backend/services/agent_service.py)
- The scorecard is not hardcoded per company. It is derived from:
  - growth quality
  - balance-sheet quality
  - sentiment quality
  - data-quality coverage

### Scorecard Outputs
- For each company:
  - total score out of 100
  - confidence percentage
  - stance
  - notes
  - decision tag
  - rank among the analyzed companies
  - score breakdown:
    - Growth / 40
    - Balance Sheet / 20
    - Sentiment / 20
    - Data Quality / 20

### Decision Tags
- Generated from score + confidence + available data
- Examples:
  - `Growth + Quality`
  - `Balanced Opportunity`
  - `Watchlist Candidate`
  - `Turnaround Risk`
  - `Insufficient Data`

## LLM / Synthesis Pipeline

### LLM Usage
- File: [backend/utils/llm.py](d:/investiq/backend/utils/llm.py)
- Supported providers are configured by environment variables.
- The LLM is used for:
  - company/entity extraction
  - structured news synthesis
  - final response synthesis when available

### Structured Output
- The backend asks the LLM to produce strict JSON.
- If parsing fails:
  - the backend falls back to deterministic grounded builders
- This keeps the app resilient when:
  - quotas are exceeded
  - credentials fail
  - model output is malformed
  - timeouts occur

## Frontend Pipeline

### Frontend Stack
- File: [frontend/src/App.tsx](d:/investiq/frontend/src/App.tsx)
- Purpose: capture the query, upload files, call the API, and render:
  - analyzed entities
  - market data
  - scorecard
  - news summary
  - document insights
  - risk analysis
  - final recommendation

### Frontend Rendering Rules
- Market metrics are formatted by type:
  - prices by currency
  - market cap / revenue / EBITDA as compact numbers
  - revenue growth as percentages
- Internal diagnostic fields are hidden from the UI
- If only current price is available:
  - the UI states that only quote data is currently available

## Setup and Configurations

Create your `.env` within `backend/`:
```env
# =========================
# AWS BEDROCK CONFIG
# =========================
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=eu-north-1
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0

# =========================
# SYSTEM PERFORMANCE & MODES
# =========================
REQUEST_TIMEOUT=20
MAX_RETRIES=3
RATE_LIMIT_PER_MINUTE=5
FALLBACK_MODE=graceful
RESPONSE_MODE=structured

# =========================
# MODEL CONFIG
# =========================
MODEL_PROVIDER=bedrock
MODEL_TEMPERATURE=0.0
MODEL_MAX_TOKENS=1024

# =========================
# NEWS CONFIG
# =========================
NEWS_SOURCE=newsapi
NEWSAPI_KEY=your_newsapi_key
NEWS_FETCH_LIMIT=5
NEWS_LANGUAGE=en
NEWS_SORT_BY=publishedAt
REDDIT_ENABLED=true
REDDIT_POST_LIMIT=3
REDDIT_SORT=relevance

# =========================
# MARKET DATA FALLBACK APIS
# =========================
ALPHAVANTAGE_API_KEY=
FINNHUB_API_KEY=
EODHD_API_KEY=

# =========================
# VECTOR DATABASE CONFIG
# =========================
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Inside `frontend/`, create a standard `.env` defining the API base URL used by the frontend:
```env
VITE_API_BASE_URL=your_deployed_api_base_url
```

## Running the Application

### Deploying via Docker Compose (Recommended)
You can instantiate the complete microservice architecture efficiently natively using Docker Compose explicitly mapping environment bounds directly.

```bash
cd d:\investiq
docker-compose up --build
```
*For the deployed demo, use the live frontend URL listed above.*

### Running Locally natively
**Backend:**
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Reliability and Fallback Strategy
- All external tool calls are wrapped by [safe_execution.py](d:/investiq/backend/utils/safe_execution.py)
- Retries are controlled by:
  - `REQUEST_TIMEOUT`
  - `MAX_RETRIES`
- Failure modes handled:
  - provider timeouts
  - malformed provider payloads
  - empty news responses
  - LLM synthesis failures
  - incomplete fundamentals
- The system falls back to grounded summaries instead of empty screens or invented claims

## Security Note
**In production, AWS credentials should be managed using IAM roles instead of environment variables for enhanced security.**

## Scalability Note
In production, the system can dynamically scale structurally by logically:
- Replacing strictly memory-bound LRU caches structurally mapping natively to dedicated scalable external **Redis** containers logically.
- Migrating local FAISS components mapping directly out toward explicitly managed Vector Databases inherently scaling via API (like **Pinecone**, **ChromaDB Cloud** or **Milvus**).
- Deploying Backend Agent configurations distinctly mapping inherently across decoupled modular microservices logic securely.
- Upgrading LangChain purely using nested orchestrator blocks mapping towards dedicated native **AWS Bedrock Agents** explicitly enforcing orchestration structurally bounding context directly across environments securely.
