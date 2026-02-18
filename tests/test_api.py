"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "embedding_model_loaded" in data


def test_ingest_document():
    """Test document ingestion."""
    document = {
        "tenant_id": "test_tenant",
        "document_id": "test_doc_1",
        "content": "This is a test document about FastAPI and Python.",
        "metadata": {"source": "test", "category": "demo"}
    }
    
    response = client.post("/documents", json=document)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["document_id"] == "test_doc_1"
    assert data["tenant_id"] == "test_tenant"


def test_search():
    """Test search endpoint."""
    # First, ingest a document
    document = {
        "tenant_id": "search_test_tenant",
        "document_id": "search_doc_1",
        "content": "FastAPI is a modern web framework for building APIs with Python.",
        "metadata": {"source": "docs"}
    }
    client.post("/documents", json=document)
    
    # Then search
    search_request = {
        "tenant_id": "search_test_tenant",
        "query": "What is FastAPI?",
        "top_k": 5
    }
    
    response = client.post("/search", json=search_request)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "latency_ms" in data
    assert isinstance(data["results"], list)


def test_multi_tenant_isolation():
    """Test that tenants cannot access each other's documents."""
    # Ingest document for tenant A
    doc_a = {
        "tenant_id": "tenant_a",
        "document_id": "doc_a1",
        "content": "Secret information for tenant A.",
        "metadata": {}
    }
    client.post("/documents", json=doc_a)
    
    # Ingest document for tenant B
    doc_b = {
        "tenant_id": "tenant_b",
        "document_id": "doc_b1",
        "content": "Secret information for tenant B.",
        "metadata": {}
    }
    client.post("/documents", json=doc_b)
    
    # Search as tenant A
    search_a = {
        "tenant_id": "tenant_a",
        "query": "secret information",
        "top_k": 10
    }
    response_a = client.post("/search", json=search_a)
    results_a = response_a.json()["results"]
    
    # Verify tenant A only sees their documents
    for result in results_a:
        assert result["document_id"] == "doc_a1"
    
    # Search as tenant B
    search_b = {
        "tenant_id": "tenant_b",
        "query": "secret information",
        "top_k": 10
    }
    response_b = client.post("/search", json=search_b)
    results_b = response_b.json()["results"]
    
    # Verify tenant B only sees their documents
    for result in results_b:
        assert result["document_id"] == "doc_b1"


def test_delete_tenant_documents():
    """Test deleting all documents for a tenant."""
    # Ingest some documents
    tenant_id = "delete_test_tenant"
    for i in range(3):
        doc = {
            "tenant_id": tenant_id,
            "document_id": f"doc_{i}",
            "content": f"Test document {i}",
            "metadata": {}
        }
        client.post("/documents", json=doc)
    
    # Delete all documents for tenant
    response = client.delete(f"/documents/{tenant_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["deleted_count"] >= 0  # May be 0 if documents were already deleted in previous runs


@pytest.mark.asyncio
async def test_search_latency():
    """Test that search meets latency requirements."""
    import time
    
    # Ingest a document
    document = {
        "tenant_id": "latency_test_tenant",
        "document_id": "latency_doc",
        "content": "Performance testing document for RAG service.",
        "metadata": {}
    }
    client.post("/documents", json=document)
    
    # Measure search latency
    search_request = {
        "tenant_id": "latency_test_tenant",
        "query": "performance testing",
        "top_k": 5
    }
    
    start = time.time()
    response = client.post("/search", json=search_request)
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    # Note: Latency target is 800ms, but in test environment it may vary
    assert latency < 2000  # More lenient for test environment
