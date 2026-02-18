"""Hybrid search implementation with parallel execution."""
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time

from app.services.qdrant_client import get_qdrant_service
from app.services.embedding import get_embedding_service
from app.utils.rrf import reciprocal_rank_fusion
from app.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """Hybrid search service combining dense and sparse retrieval."""
    
    def __init__(self):
        """Initialize search service with required components."""
        self.qdrant = get_qdrant_service()
        self.embedder = get_embedding_service()
    
    async def hybrid_search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Execute hybrid search with parallel dense and sparse retrieval.
        
        Args:
            tenant_id: Tenant identifier
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (merged results, latency in ms)
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Execute dense and sparse searches in parallel
        dense_task = self.qdrant.search_dense(tenant_id, query_embedding, top_k * 2)
        sparse_task = self.qdrant.search_sparse(tenant_id, query, top_k * 2)
        
        try:
            dense_results, sparse_results = await asyncio.wait_for(
                asyncio.gather(dense_task, sparse_task),
                timeout=settings.search_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout exceeded: {settings.search_timeout}s")
            # Return partial results if available
            dense_results = await dense_task
            sparse_results = []
        
        # Merge results using Reciprocal Rank Fusion
        merged_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=settings.rrf_k
        )
        
        # Return top_k results
        final_results = merged_results[:top_k]
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Hybrid search completed in {latency_ms:.2f}ms, found {len(final_results)} results")
        
        return final_results, latency_ms


# Global instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create the global search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
