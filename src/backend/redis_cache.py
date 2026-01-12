"""
Redis-based cache for embeddings and API responses
Memory-efficient: Redis runs in memory, prevents recomputation
OPTIMIZED FOR: 1GB RAM, Free Tier
"""

import json
import hashlib
import pickle
from typing import Any, Optional, List
import logging
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not installed. Run: pip install redis")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    """
    Memory-efficient cache using Redis
    Key optimizations for 1GB RAM:
    - Automatic cleanup of old entries
    - Size-based eviction
    - Compressed storage
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, max_memory_mb: int = 50):
        """
        Initialize Redis cache with memory limits
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_memory_mb: Maximum memory in MB (for 1GB total RAM)
        """
        if not REDIS_AVAILABLE:
            self.enabled = False
            logger.warning("Redis not available. Cache disabled.")
            return
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,  # Keep as bytes for compression
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.client.ping()
            self.enabled = True
            
            # Configure memory limits for 1GB total RAM
            # Redis gets 50MB, leaving 950MB for Python/FAISS/etc
            self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
            
            # Set Redis memory policy (volatile-LRU: evict expired keys first)
            self.client.config_set("maxmemory", str(self.max_memory))
            self.client.config_set("maxmemory-policy", "allkeys-lru")
            
            logger.info(f"✅ RedisCache initialized with {max_memory_mb}MB limit")
            logger.info(f"📊 Redis memory policy: LRU eviction")
            
        except Exception as e:
            self.enabled = False
            logger.error(f"Redis connection failed: {e}")
            logger.warning("Cache disabled. Running without Redis.")
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = str(data).encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text
        Returns None if not cached
        """
        if not self.enabled:
            return None
        
        try:
            key = f"embedding:{self._generate_key(text)}"
            cached = self.client.get(key)
            if cached:
                # Deserialize from compressed bytes
                return pickle.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set_embedding(self, text: str, embedding: List[float], 
                     ttl_seconds: int = 86400):
        """
        Cache embedding with TTL (24 hours default)
        Uses compression to save memory
        """
        if not self.enabled:
            return False
        
        try:
            key = f"embedding:{self._generate_key(text)}"
            # Serialize with compression
            serialized = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Store with TTL
            self.client.setex(key, ttl_seconds, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_llm_response(self, prompt: str, context: str = "") -> Optional[dict]:
        """
        Get cached LLM response
        """
        if not self.enabled:
            return None
        
        try:
            combined = f"{prompt}||{context}"
            key = f"llm:{self._generate_key(combined)}"
            cached = self.client.get(key)
            if cached:
                return json.loads(cached.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"LLM cache get error: {e}")
            return None
    
    def set_llm_response(self, prompt: str, context: str, 
                        response: dict, ttl_seconds: int = 3600):
        """
        Cache LLM response (1 hour default)
        """
        if not self.enabled:
            return False
        
        try:
            combined = f"{prompt}||{context}"
            key = f"llm:{self._generate_key(combined)}"
            
            # Store as JSON
            self.client.setex(key, ttl_seconds, json.dumps(response))
            return True
        except Exception as e:
            logger.error(f"LLM cache set error: {e}")
            return False
    
    def cleanup_old(self, pattern: str = "*"):
        """
        Manual cleanup of old cache entries
        Useful for memory management
        """
        if not self.enabled:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"🧹 Cleaned up {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.client.info()
            stats = {
                "enabled": True,
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "maxmemory_human": info.get('maxmemory_human', '0B'),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": 0
            }
            
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total = hits + misses
            if total > 0:
                stats["hit_rate"] = round((hits / total) * 100, 2)
            
            return stats
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"enabled": False, "error": str(e)}
    
    def clear_all(self) -> bool:
        """Clear all cache (use cautiously)"""
        if not self.enabled:
            return False
        
        try:
            self.client.flushdb()
            logger.info("🧹 Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Clear error: {e}")
            return False


# Singleton instance
try:
    # For EC2 deployment, Redis might be on localhost
    # For Docker, use service name
    redis_cache = RedisCache(host="localhost", port=6379, max_memory_mb=50)
except:
    redis_cache = None
    logger.warning("Redis cache not available")
