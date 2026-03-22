# InvestIQ AI - Investment Research Agent

A production-ready AI Investment Research Agent strictly utilizing AWS Bedrock (Claude Haiku 4.5, eu-north-1 region), FastAPI, LangChain, FAISS Vector DB, and an aesthetic Framer Motion React Front-End. Completely autonomous, prompting parallel extraction and financial reporting without brittle keyword routing.

## Architecture & Workflow

* **System Overview:** An autonomous AI orchestration engine that extracts entities from raw user queries and routes them in parallel against live semantic and financial data streams, generating robustly structured analytical JSON dynamically.
* **Agent Workflow:**
  1. Intake: Receives a dense financial query alongside optional document PDFs.
  2. Extraction: Explicitly strips and recognizes the core corporate entities.
  3. Parallel Routing: The LangChain Orchestrator splits multiple concurrent instances of async tools executing market metrics and news retrieval independently for every entity securely.
  4. Generation: Deep synthesizes the raw outputs natively formulating a strictly typed JSON layout outlining risks, news impacts, metrics, and core recommendations.
* **Tool Orchestration:** Our agent leverages `ChatBedrock` via `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` wrapped natively entirely inside local Multi-Threaded asynchronous execution limits preventing hallucination and deadlocks.
  - `Ticker Resolver`: Intelligent fuzzy searching via Yahoo API.
  - `Market Data`: Direct `yfinance` queries securely checking values aggressively.
  - `RSS News Agent`: News data is retrieved using Google News RSS feeds, configured via environment variables for flexibility and reliability. Feeds sequentially into local bedrock nodes compiling semantic classifications.
  - `FAISS RAG`: Vector Database executing local HuggingFace embeddings explicitly mapping PDF insight retrievals globally.

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
REQUEST_TIMEOUT=5
MAX_RETRIES=3
RATE_LIMIT_PER_MINUTE=5
FALLBACK_MODE=graceful
RESPONSE_MODE=structured

# =========================
# VECTOR DATABASE CONFIG
# =========================
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Inside `frontend/`, create a standard `.env` defining the native endpoints cleanly:
```env
VITE_API_BASE_URL=http://localhost:8000
```

## Running the Application

### Deploying via Docker Compose (Recommended)
You can instantiate the complete microservice architecture efficiently natively using Docker Compose explicitly mapping environment bounds directly.

```bash
cd d:\investiq
docker-compose up --build
```
*Frontend spins up internally bound to `http://localhost:3000` executing seamlessly.*

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

## Error Handling & Reliability
- **API Failures:** Wrapped universally by `/utils/safe_execution.py`. If Yahoo or Google natively misfire after configured retries natively (`MAX_RETRIES=3`), execution overrides to an internal strict JSON fallback avoiding cascading crashes natively.
- **Missing Data:** If extracted content is empty, natively intercepted via the agent orchestration layer falling back to `Generic Market Search`.
- **Timeouts:** Aggressively regulated via `REQUEST_TIMEOUT`. Nested chains terminating strictly preventing locked asynchronous states gracefully mapping into `FALLBACK_MODE`.

## Security Note
**In production, AWS credentials should be managed using IAM roles instead of environment variables for enhanced security.**

## Scalability Note
In production, the system can dynamically scale structurally by logically:
- Replacing strictly memory-bound LRU caches structurally mapping natively to dedicated scalable external **Redis** containers logically.
- Migrating local FAISS components mapping directly out toward explicitly managed Vector Databases inherently scaling via API (like **Pinecone**, **ChromaDB Cloud** or **Milvus**).
- Deploying Backend Agent configurations distinctly mapping inherently across decoupled modular microservices logic securely.
- Upgrading LangChain purely using nested orchestrator blocks mapping towards dedicated native **AWS Bedrock Agents** explicitly enforcing orchestration structurally bounding context directly across environments securely.
