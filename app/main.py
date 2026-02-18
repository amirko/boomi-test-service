"""FastAPI application for RAG-as-a-Service."""
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import json

from app.config import settings
from app.models import (
    DocumentInput, SearchRequest, SearchResponse, SearchResult,
    SummaryResponse, HealthResponse, DocumentResponse, DeleteResponse
)
from app.services.embedding import get_embedding_service
from app.services.qdrant_client import get_qdrant_service
from app.services.llm_service import get_llm_service
from app.services.search import get_search_service
from app.utils.circuit_breaker import CircuitBreakerException, with_timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: Initialize services
    logger.info("Starting RAG-as-a-Service...")
    try:
        get_embedding_service()
        get_qdrant_service()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG-as-a-Service...")


app = FastAPI(
    title="RAG-as-a-Service",
    description="Retrieval-Augmented Generation service with hybrid search",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Check Qdrant connectivity
        qdrant = get_qdrant_service()
        qdrant_connected = qdrant.health_check()
        
        # Check if embedding model is loaded
        embedding_service = get_embedding_service()
        embedding_loaded = embedding_service is not None
        
        status_ok = qdrant_connected and embedding_loaded
        
        return HealthResponse(
            status="healthy" if status_ok else "degraded",
            qdrant_connected=qdrant_connected,
            embedding_model_loaded=embedding_loaded,
            details={
                "qdrant_host": settings.qdrant_host,
                "qdrant_port": settings.qdrant_port,
                "collection": settings.collection_name,
                "embedding_model": settings.embedding_model,
                "llm_provider": settings.llm_provider
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/documents", response_model=DocumentResponse)
async def ingest_document(document: DocumentInput):
    """
    Ingest a document with tenant isolation.
    
    Generates both dense and sparse vectors and stores in Qdrant.
    """
    try:
        start_time = time.time()
        
        # Generate embedding
        embedding_service = get_embedding_service()
        dense_vector = embedding_service.encode(document.content).tolist()
        
        # Store in Qdrant
        qdrant = get_qdrant_service()
        await qdrant.insert_document(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            content=document.content,
            metadata=document.metadata,
            dense_vector=dense_vector
        )
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Document ingested in {latency_ms:.2f}ms")
        
        return DocumentResponse(
            status="success",
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            message=f"Document ingested successfully in {latency_ms:.2f}ms"
        )
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Hybrid search endpoint with parallel BM25 + Vector search.
    
    Target latency: < 800ms
    """
    try:
        # Execute hybrid search
        search_service = get_search_service()
        results, latency_ms = await search_service.hybrid_search(
            tenant_id=request.tenant_id,
            query=request.query,
            top_k=request.top_k
        )
        
        # Convert to response model
        search_results = [
            SearchResult(
                document_id=r["document_id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata")
            )
            for r in results
        ]
        
        return SearchResponse(
            results=search_results,
            latency_ms=latency_ms
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/search-with-summary")
async def search_with_summary(request: SearchRequest):
    """
    Search with LLM summarization endpoint.
    
    Target latency: < 2.5s (with streaming)
    Uses circuit breaker with timeout for LLM calls.
    """
    try:
        overall_start = time.time()
        
        # Execute hybrid search
        search_service = get_search_service()
        results, search_latency_ms = await search_service.hybrid_search(
            tenant_id=request.tenant_id,
            query=request.query,
            top_k=request.top_k
        )
        
        if not results:
            return JSONResponse(
                content={
                    "results": [],
                    "summary": "No results found for your query.",
                    "latency_ms": (time.time() - overall_start) * 1000,
                    "search_latency_ms": search_latency_ms,
                    "llm_latency_ms": 0
                }
            )
        
        # Generate summary with timeout
        llm_start = time.time()
        llm_service = get_llm_service()
        
        try:
            summary = await with_timeout(
                llm_service.generate_summary(request.query, results),
                timeout=settings.llm_timeout,
                default="Summary generation timed out. Please try again."
            )
        except CircuitBreakerException:
            summary = "Summary generation timed out. Search results are still available below."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            summary = f"Summary generation failed: {str(e)}. Search results are still available below."
        
        llm_latency_ms = (time.time() - llm_start) * 1000
        total_latency_ms = (time.time() - overall_start) * 1000
        
        # Convert to response model
        search_results = [
            SearchResult(
                document_id=r["document_id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata")
            )
            for r in results
        ]
        
        return JSONResponse(
            content={
                "results": [r.model_dump() for r in search_results],
                "summary": summary,
                "latency_ms": total_latency_ms,
                "search_latency_ms": search_latency_ms,
                "llm_latency_ms": llm_latency_ms
            }
        )
    except Exception as e:
        logger.error(f"Search with summary failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search with summary failed: {str(e)}"
        )


@app.delete("/documents/{tenant_id}", response_model=DeleteResponse)
async def delete_tenant_documents(tenant_id: str):
    """Delete all documents for a tenant."""
    try:
        qdrant = get_qdrant_service()
        deleted_count = await qdrant.delete_by_tenant(tenant_id)
        
        return DeleteResponse(
            status="success",
            tenant_id=tenant_id,
            deleted_count=deleted_count,
            message=f"Deleted {deleted_count} documents for tenant {tenant_id}"
        )
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
