"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class DocumentInput(BaseModel):
    """Model for document ingestion."""
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant isolation")
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content to be indexed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class SearchRequest(BaseModel):
    """Model for search requests."""
    tenant_id: str = Field(..., description="Tenant identifier")
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")


class SearchResult(BaseModel):
    """Model for individual search result."""
    document_id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[SearchResult]
    latency_ms: float


class SummaryResponse(BaseModel):
    """Model for search with summary response."""
    results: List[SearchResult]
    summary: str
    latency_ms: float
    search_latency_ms: float
    llm_latency_ms: float


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    qdrant_connected: bool
    embedding_model_loaded: bool
    details: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Model for document ingestion response."""
    status: str
    document_id: str
    tenant_id: str
    message: str


class DeleteResponse(BaseModel):
    """Model for deletion response."""
    status: str
    tenant_id: str
    deleted_count: int
    message: str
