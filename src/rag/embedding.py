"""
Embedding service with Redis caching for memory efficiency
"""
from typing import List, Optional
import numpy as np
import redis
import pickle
import hashlib
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Lightweight embedding service with caching
    Uses external API (will be implemented Day 3)
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 cache_ttl: int = 3600):
        """
        Initialize embedding service
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
        
        # In-memory cache as fallback
        self.memory_cache = {}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        cache_key = self._get_cache_key(text)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to memory cache
        return self.memory_cache.get(cache_key)
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        cache_key = self._get_cache_key(text)
        
        # Cache in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    pickle.dumps(embedding)
                )
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Cache in memory
        self.memory_cache[cache_key] = embedding
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts (placeholder - will use external API Day 3)
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        # For now, return random embeddings
        # Day 3: Replace with Google Gemini/Hugging Face API
        dimension = 384  # all-MiniLM-L6-v2 dimension
        embeddings = []
        
        for text in texts:
            # Check cache first
            cached = self.get_cached_embedding(text)
            if cached is not None:
                embeddings.append(cached)
                continue
            
            # Generate random embedding (placeholder)
            # TODO: Replace with actual API call
            embedding = np.random.randn(dimension).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            # Cache it
            self.cache_embedding(text, embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts in batches for memory efficiency
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)
            
            logger.info(f"Encoded batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
        
        return np.vstack(all_embeddings)
    
    def get_stats(self) -> dict:
        """Get caching statistics"""
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                stats['redis_cache_size'] = self.redis_client.dbsize()
            except:
                stats['redis_cache_size'] = 'unknown'
        
        return stats

if __name__ == "__main__":
    # Test the embedding service
    service = EmbeddingService()
    
    test_texts = [
        "This is a test document about risk management.",
        "Enterprise risk intelligence requires careful analysis.",
        "AI can help identify potential business risks."
    ]
    
    print("Testing embedding service...")
    embeddings = service.encode(test_texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Cache stats: {service.get_stats()}")
    
    # Test caching
    print("\nTesting cache...")
    cached_embeddings = service.encode(test_texts)  # Should come from cache
    print("✅ Embedding service test complete!")
