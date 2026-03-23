from fastapi import FastAPI, UploadFile, File, Form, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import os
from contextlib import asynccontextmanager

from utils.config import settings
from utils.rate_limiter import check_rate_limit
from services.agent_service import AgentService
import time

logger = logging.getLogger("investiq_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Initializing InvestIQ AI Backend strictly bound on port 8000 using log level {settings.log_level}")
    yield
    logger.info("Shutting down precisely.")

app = FastAPI(title="InvestIQ AI Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    companies: List[str]
    market_data: dict
    company_scores: dict
    news_summary: str
    document_insights: str
    risk_analysis: str
    final_recommendation: str
    llm_status: str = ""
    used_fallback: bool = False

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} Method: {request.method} Status: {response.status_code} Time: {process_time:.2f}s")
    return response


@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(check_rate_limit)])
async def analyze_query(
    query: str = Form(...),
    documents: Optional[List[UploadFile]] = File(None)
):
    logger.info(f"Received /analyze efficiently mapping payload length {len(query)}")
    
    if documents and documents[0].filename == '':
        documents = None

    service = AgentService()
    try:
        result = await service.process_query(query, documents)
        return AnalyzeResponse(**result)
    except Exception as e:
        logger.error(f"Failed cleanly handling /analyze securely catching completely: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed while processing the request."
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
