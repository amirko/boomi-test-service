"""Qdrant vector database client with hybrid search support."""
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue,
    SparseVector, SparseVectorParams, SparseIndexParams
)
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant client for hybrid vector search."""
    
    def __init__(self):
        """Initialize Qdrant client and create collection if needed."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with dense and sparse vector configurations."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams()
                    )
                }
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")
    
    def _create_sparse_vector(self, text: str) -> SparseVector:
        """
        Create a simple sparse vector from text (BM25-like).
        Uses word frequency as a simple sparse representation.
        """
        words = text.lower().split()
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create sparse vector using deterministic hash of words as indices
        indices = []
        values = []
        for word, freq in word_freq.items():
            # Use MD5 hash for deterministic, consistent indices across runs
            hash_obj = hashlib.md5(word.encode('utf-8'))
            idx = int(hash_obj.hexdigest()[:8], 16) % 10000  # Limit index space
            indices.append(idx)
            values.append(float(freq))
        
        return SparseVector(indices=indices, values=values)
    
    async def insert_document(
        self,
        tenant_id: str,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        dense_vector: List[float]
    ):
        """
        Insert a document with dense and sparse vectors.
        
        Args:
            tenant_id: Tenant identifier for isolation
            document_id: Unique document identifier
            content: Document content
            metadata: Additional metadata
            dense_vector: Dense embedding vector
        """
        sparse_vector = self._create_sparse_vector(content)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector
            },
            payload={
                "tenant_id": tenant_id,
                "document_id": document_id,
                "content": content,
                "metadata": metadata
            }
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        logger.info(f"Inserted document {document_id} for tenant {tenant_id}")
    
    async def search_dense(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using dense vectors.
        
        Args:
            tenant_id: Tenant identifier
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_vector),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        return [
            {
                "document_id": hit.payload.get("document_id"),
                "content": hit.payload.get("content"),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]
    
    async def search_sparse(
        self,
        tenant_id: str,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using sparse vectors (BM25-like).
        
        Args:
            tenant_id: Tenant identifier
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        sparse_vector = self._create_sparse_vector(query_text)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("sparse", sparse_vector),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        return [
            {
                "document_id": hit.payload.get("document_id"),
                "content": hit.payload.get("content"),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]
    
    async def delete_by_tenant(self, tenant_id: str) -> int:
        """
        Delete all documents for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Number of documents deleted
        """
        # Scroll through all points for the tenant and delete
        offset = None
        deleted_count = 0
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            if not results:
                break
            
            point_ids = [point.id for point in results]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            deleted_count += len(point_ids)
            
            if offset is None:
                break
        
        logger.info(f"Deleted {deleted_count} documents for tenant {tenant_id}")
        return deleted_count
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


# Global instance
_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Get or create the global Qdrant service instance."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
