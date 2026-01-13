"""
Redis cache for embeddings and responses
"""
import redis.asyncio as redis
import json
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis caching wrapper"""
    
    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.client = None
    
    async def connect(self):
        """Connect to Redis"""
        if self.client is None:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                await self.client.ping()
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.client = None
                raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.client is None:
                await self.connect()
            
            value = await self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL"""
        try:
            if self.client is None:
                await self.connect()
            
            await self.client.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()

# Function that main.py expects
async def get_redis_client():
    """Get Redis client instance"""
    cache = RedisCache()
    try:
        await cache.connect()
        return cache.client
    except Exception as e:
        logger.warning(f"Failed to get Redis client: {e}")
        return None
