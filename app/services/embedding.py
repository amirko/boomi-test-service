"""Local embedding service using sentence-transformers."""
import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Local embedding service for generating dense vectors."""
    
    def __init__(self):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = settings.embedding_dimension
        logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into a dense vector.
        
        Args:
            text: Input text to encode
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into dense vectors.
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.dimension


# Global instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
