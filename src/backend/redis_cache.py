"""
Redis Cache Module for AWS Risk Copilot - ASYNC VERSION
Optimized for 1GB RAM constraint
"""
import os
import json
import logging
from typing import Optional, Any, Dict
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisCache:
    """Async Redis caching wrapper with sync-like interface for compatibility"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("REDIS_HOST", "redis")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = int(os.getenv("REDIS_DB", 0))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.client = None
        self.max_memory_mb = 50  # Max memory for cache in MB
        self.ttl = 3600  # Default TTL: 1 hour
        
    async def _ensure_connected(self):
        """Ensure Redis connection is established"""
        if self.client is None:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                if await self.client.ping():
                    logger.info(f"✅ Redis connection established to {self.host}:{self.port}")
                else:
                    logger.error("❌ Redis ping failed")
                    self.client = None
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                self.client = None
                raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (async)"""
        try:
            await self._ensure_connected()
            value = await self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL (async)"""
        try:
            await self._ensure_connected()
            # Convert to JSON if it's a dict/list
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            
            # Set with TTL
            actual_ttl = ttl or self.ttl
            result = await self.client.setex(key, actual_ttl, value_str)
            
            # Enforce memory limits
            await self._enforce_memory_limits()
            return bool(result)
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache (async)"""
        try:
            await self._ensure_connected()
            return bool(await self.client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
        return False
    
    async def clear(self) -> bool:
        """Clear all cache keys (use with caution, async)"""
        try:
            await self._ensure_connected()
            await self.client.flushdb()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        return False
    
    async def _enforce_memory_limits(self):
        """Enforce memory limits by removing old keys if needed (async)"""
        try:
            await self._ensure_connected()
            # Check memory usage
            info = await self.client.info('memory')
            used_memory_mb = int(info.get('used_memory', 0)) / 1024 / 1024
            
            if used_memory_mb > self.max_memory_mb:
                logger.warning(f"Redis memory high: {used_memory_mb:.1f}MB > {self.max_memory_mb}MB")
        except Exception as e:
            logger.warning(f"Memory limit enforcement error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (async)"""
        try:
            await self._ensure_connected()
            info = await self.client.info()
            return {
                "connected": True,
                "keys": await self.client.dbsize(),
                "memory_used_mb": int(info.get('used_memory', 0)) / 1024 / 1024,
                "host": self.host,
                "port": self.port
            }
        except Exception as e:
            logger.warning(f"Stats error: {e}")
        return {"connected": False, "error": "Not connected"}
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client for direct access (async)"""
        await self._ensure_connected()
        return self.client
    
    async def close(self):
        """Close Redis connection (async)"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Redis connection closed")

# Global Redis cache instance
redis_cache = RedisCache()

async def get_redis_client() -> redis.Redis:
    """Get Redis client for direct access - main.py expects this (async)"""
    return await redis_cache.get_client()
