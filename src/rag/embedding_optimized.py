"""
Optimized embedding service with lazy loading and memory limits
"""
import asyncio
import logging
from typing import Optional, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class OptimizedEmbeddingService:
    """Memory-optimized embedding service with lazy loading"""
    
    _model = None
    _model_loaded = False
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 cache_ttl: int = 3600,
                 max_memory_mb: int = 150):
        self.model_name = model_name
        self.cache_ttl = cache_ttl
        self.max_memory_mb = max_memory_mb
        self.embedding_cache = {}  # Simple in-memory cache
        self.cache_timestamps = {}
        
        logger.info(f"OptimizedEmbeddingService initialized (model: {model_name})")
    
    def _load_model(self):
        """Lazy load the model only when needed"""
        if not self._model_loaded:
            try:
                logger.info(f"Lazy loading model: {self.model_name}")
                
                # Import only when needed
                from sentence_transformers import SentenceTransformer
                
                # Load with optimized settings
                self._model = SentenceTransformer(
                    self.model_name,
                    device='cpu',  # Force CPU to save memory
                    cache_folder='/tmp/model_cache'  # Use tmpfs if possible
                )
                self._model_loaded = True
                
                logger.info(f"Model loaded successfully: {self.model_name}")
                
                # Force minimal memory footprint
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with memory awareness"""
        # Check memory before loading
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 85:  # If memory > 85%
            logger.warning(f"Memory high ({mem.percent}%), cleaning cache")
            self._clear_cache()
        
        # Load model if not loaded
        if not self._model_loaded:
            self._load_model()
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=2,  # Small batch size for low memory
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def _clear_cache(self):
        """Clear embedding cache to free memory"""
        cache_size = len(self.embedding_cache)
        if cache_size > 0:
            logger.info(f"Clearing embedding cache ({cache_size} entries)")
            self.embedding_cache.clear()
            self.cache_timestamps.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
    
    def cleanup(self):
        """Cleanup model and cache"""
        if self._model_loaded:
            logger.info("Cleaning up embedding model")
            del self._model
            self._model = None
            self._model_loaded = False
            
            # Clear cache
            self._clear_cache()
            
            import gc
            gc.collect()
            logger.info("Embedding model cleanup completed")

# Global instance
embedding_service = OptimizedEmbeddingService()

async def get_embedding_service():
    """Get singleton embedding service"""
    return embedding_service
