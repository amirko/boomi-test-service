"""LLM service with streaming support for multiple providers."""
import logging
from typing import AsyncIterator, Optional, List, Dict, Any
import httpx
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service with support for multiple providers."""
    
    def __init__(self):
        """Initialize LLM client based on provider."""
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout
        
        if self.provider == "groq":
            self.client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        elif self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.llm_api_key)
        elif self.provider == "ollama":
            self.client = AsyncOpenAI(
                api_key="ollama",  # Ollama doesn't need real API key
                base_url=f"{settings.ollama_base_url}/v1"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        logger.info(f"LLM service initialized with provider: {self.provider}, model: {self.model}")
    
    def _create_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM with search context.
        
        Args:
            query: Original search query
            search_results: List of search results with content
            
        Returns:
            Formatted prompt string
        """
        # Limit to top 3-5 chunks for context pruning
        top_results = search_results[:min(5, len(search_results))]
        
        context_parts = []
        for i, result in enumerate(top_results, 1):
            content = result.get("content", "")
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following search results, provide a concise summary answering the query: "{query}"

Search Results:
{context}

Summary:"""
        
        return prompt
    
    async def generate_summary(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a summary from search results.
        
        Args:
            query: Original search query
            search_results: List of search results
            
        Returns:
            Generated summary text
        """
        if not search_results:
            return "No search results found to summarize."
        
        prompt = self._create_prompt(query, search_results)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes search results concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
                timeout=self.timeout
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def generate_summary_stream(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """
        Generate a streaming summary from search results.
        
        Args:
            query: Original search query
            search_results: List of search results
            
        Yields:
            Text chunks as they are generated
        """
        if not search_results:
            yield "No search results found to summarize."
            return
        
        prompt = self._create_prompt(query, search_results)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes search results concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
                stream=True,
                timeout=self.timeout
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
