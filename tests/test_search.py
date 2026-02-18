"""Tests for search and RRF functionality."""
import pytest
from app.utils.rrf import reciprocal_rank_fusion


def test_rrf_basic():
    """Test basic RRF functionality."""
    # Create two ranked lists
    list1 = [
        {"document_id": "doc1", "content": "content1", "score": 0.9},
        {"document_id": "doc2", "content": "content2", "score": 0.8},
        {"document_id": "doc3", "content": "content3", "score": 0.7},
    ]
    
    list2 = [
        {"document_id": "doc2", "content": "content2", "score": 0.95},
        {"document_id": "doc1", "content": "content1", "score": 0.85},
        {"document_id": "doc4", "content": "content4", "score": 0.75},
    ]
    
    # Apply RRF
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    
    # doc2 should rank highest (appears first in list2, second in list1)
    assert len(merged) == 4
    assert merged[0]["document_id"] == "doc2"
    
    # All documents should have RRF scores
    for doc in merged:
        assert "score" in doc
        assert doc["score"] > 0


def test_rrf_empty_lists():
    """Test RRF with empty lists."""
    merged = reciprocal_rank_fusion([[], []], k=60)
    assert len(merged) == 0


def test_rrf_single_list():
    """Test RRF with a single list."""
    list1 = [
        {"document_id": "doc1", "content": "content1", "score": 0.9},
        {"document_id": "doc2", "content": "content2", "score": 0.8},
    ]
    
    merged = reciprocal_rank_fusion([list1], k=60)
    
    assert len(merged) == 2
    # Order should be preserved for single list
    assert merged[0]["document_id"] == "doc1"
    assert merged[1]["document_id"] == "doc2"


def test_rrf_no_overlap():
    """Test RRF with no overlapping documents."""
    list1 = [
        {"document_id": "doc1", "content": "content1", "score": 0.9},
        {"document_id": "doc2", "content": "content2", "score": 0.8},
    ]
    
    list2 = [
        {"document_id": "doc3", "content": "content3", "score": 0.95},
        {"document_id": "doc4", "content": "content4", "score": 0.85},
    ]
    
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    
    # Should have all 4 documents
    assert len(merged) == 4
    
    # All should have equal RRF scores since they appear once at same relative position
    doc_ids = [doc["document_id"] for doc in merged]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids
    assert "doc4" in doc_ids


def test_rrf_score_calculation():
    """Test that RRF scores are calculated correctly."""
    list1 = [
        {"document_id": "doc1", "content": "content1", "score": 0.9},
    ]
    
    list2 = [
        {"document_id": "doc1", "content": "content1", "score": 0.8},
    ]
    
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    
    # doc1 appears at rank 1 in both lists
    # Expected RRF score = 1/(60+1) + 1/(60+1) = 2/61
    expected_score = 2 / 61
    assert len(merged) == 1
    assert abs(merged[0]["score"] - expected_score) < 0.0001


@pytest.mark.asyncio
async def test_embedding_generation():
    """Test embedding generation."""
    from app.services.embedding import get_embedding_service
    
    embedder = get_embedding_service()
    
    # Test single encoding
    text = "This is a test sentence."
    embedding = embedder.encode(text)
    
    assert embedding is not None
    assert len(embedding) == embedder.get_dimension()
    
    # Test batch encoding
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = embedder.encode_batch(texts)
    
    assert embeddings is not None
    assert len(embeddings) == 3
    assert len(embeddings[0]) == embedder.get_dimension()


@pytest.mark.asyncio
async def test_hybrid_search():
    """Test hybrid search functionality."""
    from app.services.search import get_search_service
    from app.services.qdrant_client import get_qdrant_service
    from app.services.embedding import get_embedding_service
    
    # Setup
    search_service = get_search_service()
    qdrant = get_qdrant_service()
    embedder = get_embedding_service()
    
    tenant_id = "hybrid_test_tenant"
    
    # Ingest test documents
    test_docs = [
        {"id": "h_doc1", "content": "Python is a programming language."},
        {"id": "h_doc2", "content": "FastAPI is a web framework for Python."},
        {"id": "h_doc3", "content": "Machine learning with Python is powerful."},
    ]
    
    for doc in test_docs:
        vector = embedder.encode(doc["content"]).tolist()
        await qdrant.insert_document(
            tenant_id=tenant_id,
            document_id=doc["id"],
            content=doc["content"],
            metadata={},
            dense_vector=vector
        )
    
    # Execute hybrid search
    results, latency_ms = await search_service.hybrid_search(
        tenant_id=tenant_id,
        query="Python programming",
        top_k=3
    )
    
    # Verify results
    assert len(results) > 0
    assert latency_ms > 0
    
    # Results should contain relevant documents
    doc_ids = [r["document_id"] for r in results]
    assert any(doc_id in doc_ids for doc_id in ["h_doc1", "h_doc2", "h_doc3"])
