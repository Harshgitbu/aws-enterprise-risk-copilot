"""
Async wrapper for LLM Service
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from .llm_service import UnifiedLLMService

logger = logging.getLogger(__name__)

class AsyncUnifiedLLMService:
    """Async wrapper for UnifiedLLMService"""
    
    def __init__(self):
        self.sync_service = UnifiedLLMService()
        self.executor = None
    
    async def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """
        Generate response from LLM (async wrapper)
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Creativity temperature
            
        Returns:
            Generated response
        """
        try:
            # Run sync method in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sync_service.generate_response(prompt, max_tokens, temperature)
            )
            return response
        except Exception as e:
            logger.error(f"Async LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics (async)"""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                self.sync_service.get_stats
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get LLM stats: {e}")
            return {"error": str(e)}

# Global async LLM service instance
async_llm_service = AsyncUnifiedLLMService()
