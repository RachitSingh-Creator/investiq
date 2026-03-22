import io
from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
import asyncio
import logging
from utils.config import settings
from utils.safe_execution import async_safe_execute

logger = logging.getLogger(__name__)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if not embedding_model:
        # Load embedding model dynamically configured natively through env variables
        logger.info(f"Loading embedding model natively globally bound to settings: {settings.embedding_model}")
        embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return embedding_model

async def process_documents_logic(files) -> FAISS:
    """Processes uploaded files safely mapping embedded FAISS chunk limits dynamically."""
    if not files:
        return None
        
    def _read_and_split():
        all_text = ""
        for f in files:
            f.file.seek(0)
            try:
                raw_bytes = f.file.read()
                suffix = Path(f.filename or "").suffix.lower()

                if suffix == ".txt" or f.content_type == "text/plain":
                    text = raw_bytes.decode("utf-8", errors="ignore").strip()
                    if text:
                        all_text += text + "\n"
                    continue

                if suffix != ".pdf":
                    logger.warning("Skipping unsupported document type for %s", f.filename)
                    continue

                pdf_reader = PdfReader(io.BytesIO(raw_bytes))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            except Exception as e:
                logger.error(f"Failed parsing PDF extraction natively bounds natively skipped securely: {e}")
        
        # Apply chunk constraints dynamically sourced
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        return text_splitter.split_text(all_text)
        
    chunks = await asyncio.to_thread(_read_and_split)
    
    if not chunks:
        return None
        
    embeddings = get_embedding_model()
    
    def _build_faiss():
        return FAISS.from_texts(chunks, embeddings)
        
    vector_store = await asyncio.to_thread(_build_faiss)
    return vector_store

async def process_documents(files):
    """Async execution wrapper logically bound over processing."""
    try:
        # Add basic try except just in case if safe wrapper throws JSON natively
        result = await async_safe_execute(process_documents_logic, "document_qa", files)
        if isinstance(result, str):
            return None
        return result
    except Exception as e:
        logger.error(f"Execution wrapper internally caught: {e}")
        return None

def get_doc_tool(vector_store):
    def search_docs(query: str) -> str:
        """Search uploaded financial documents for insights given a query."""
        if not vector_store:
            return "No documents were uploaded to search."
        try:
            docs = vector_store.similarity_search(query, k=3)
            return "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"Failed FAISS internal similarity securely: {e}")
            import json
            return json.dumps({"error": str(e), "source": "docs"})
        
    return Tool(
        name="document_search",
        description="Search uploaded documents for context or insights dynamically.",
        func=search_docs
    )
