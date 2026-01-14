"""
Memory-efficient embedding service with Redis caching
Optimized for 1GB RAM constraint - ASYNC VERSION
"""
import os
import hashlib
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio

# Import RedisCache for proper Redis connection
try:
    from backend.redis_cache import redis_cache
    REDIS_AVAILABLE = True
except ImportError as e:
    print(f"RedisCache import error: {e}")
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and caching embeddings with memory constraints"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_ttl: int = 3600):  # 1 hour TTL
        """
        Initialize embedding service
        
        Args:
            model_name: Name of the sentence transformer model
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.model_name = model_name
        self.cache_ttl = cache_ttl
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Redis cache is available globally via redis_cache instance
        self.redis_available = REDIS_AVAILABLE
        
        # Memory limits
        self.max_cache_size_mb = 50  # Max 50MB for embedding cache
        self.batch_size = 32  # Process in batches to save memory
        
    def load_model(self):
        """Lazy load the embedding model to save memory"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # Verify embedding dimension
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        
        return self.model
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use SHA256 for consistent keys
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text with caching (async)
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if error
        """
        if not text or not text.strip():
            return None
        
        cache_key = self._get_cache_key(text)
        
        # Try cache first
        if self.redis_available:
            try:
                cached = await redis_cache.get(cache_key)
                if cached:
                    if isinstance(cached, list):
                        return np.array(cached, dtype=np.float32)
                    return cached
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        
        # Generate embedding
        try:
            model = self.load_model()
            embedding = model.encode([text])[0]
            
            # Store in cache
            if self.redis_available:
                try:
                    # Convert numpy array to list for JSON serialization
                    embedding_list = embedding.tolist()
                    await redis_cache.set(cache_key, embedding_list, self.cache_ttl)
                except Exception as e:
                    logger.debug(f"Cache write error: {e}")
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts (more memory efficient, async)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            return [None] * len(texts)
        
        embeddings = [None] * len(texts)
        
        # Check cache first
        cache_keys = []
        texts_to_process = []
        process_indices = []
        
        if self.redis_available:
            for idx, text in zip(valid_indices, valid_texts):
                cache_key = self._get_cache_key(text)
                cache_keys.append(cache_key)
                
                try:
                    cached = await redis_cache.get(cache_key)
                    if cached:
                        if isinstance(cached, list):
                            embeddings[idx] = np.array(cached, dtype=np.float32)
                        else:
                            embeddings[idx] = cached
                    else:
                        texts_to_process.append(text)
                        process_indices.append(idx)
                except Exception as e:
                    logger.debug(f"Cache batch read error: {e}")
                    texts_to_process.append(text)
                    process_indices.append(idx)
        else:
            texts_to_process = valid_texts
            process_indices = valid_indices
        
        # Process remaining texts
        if texts_to_process:
            try:
                model = self.load_model()
                # Process in batches to save memory
                batch_embeddings = []
                for i in range(0, len(texts_to_process), self.batch_size):
                    batch = texts_to_process[i:i + self.batch_size]
                    batch_emb = model.encode(batch)
                    batch_embeddings.extend(batch_emb)
                
                # Store in cache and update results
                for idx, emb, text in zip(process_indices, batch_embeddings, texts_to_process):
                    embeddings[idx] = emb
                    
                    if self.redis_available:
                        try:
                            cache_key = self._get_cache_key(text)
                            await redis_cache.set(cache_key, emb.tolist(), self.cache_ttl)
                        except Exception as e:
                            logger.debug(f"Cache batch write error: {e}")
                            
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
        
        return embeddings
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics (async)"""
        stats = {
            'model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'model_loaded': self.model is not None,
            'cache_available': self.redis_available,
            'cache_ttl': self.cache_ttl,
            'max_cache_size_mb': self.max_cache_size_mb,
            'batch_size': self.batch_size
        }
        
        if self.redis_available:
            try:
                cache_stats = await redis_cache.get_stats()
                stats['redis_connected'] = cache_stats.get('connected', False)
                if cache_stats.get('connected'):
                    stats['redis_keys'] = cache_stats.get('keys', 0)
                    stats['redis_memory_mb'] = cache_stats.get('memory_used_mb', 0)
            except Exception as e:
                stats['redis_error'] = str(e)
                stats['redis_connected'] = False
        
        return stats

# Global embedding service instance
embedding_service = EmbeddingService()
