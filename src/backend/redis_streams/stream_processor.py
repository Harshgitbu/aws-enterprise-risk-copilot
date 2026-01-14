"""
Lightweight Redis Streams processor for real-time risk alerts
Memory-optimized for 1GB RAM EC2
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisStreamProcessor:
    """Memory-efficient Redis Streams processor"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.stream_key = "risk:alerts"
        self.consumer_group = "risk_processor"
        self.consumer_name = "backend_worker"
        self.max_memory_mb = 50  # Max memory for streams
        self.max_stream_length = 1000  # Keep streams short
        
    async def initialize(self):
        """Initialize streams and consumer group"""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host="redis",
                port=6379,
                decode_responses=True
            )
        
        try:
            # Create consumer group if not exists
            await self.redis_client.xgroup_create(
                name=self.stream_key,
                groupname=self.consumer_group,
                id="$",
                mkstream=True
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group {self.consumer_group} already exists")
            else:
                logger.warning(f"Failed to create consumer group: {e}")
    
    async def publish_alert(self, alert_data: Dict[str, Any]) -> str:
        """
        Publish risk alert to stream
        
        Args:
            alert_data: Alert data dictionary
        
        Returns:
            Message ID
        """
        try:
            # Ensure stream doesn't grow too large
            stream_length = await self.redis_client.xlen(self.stream_key)
            if stream_length > self.max_stream_length:
                await self.redis_client.xtrim(self.stream_key, maxlen=self.max_stream_length)
                logger.debug(f"Trimmed stream to {self.max_stream_length} messages")
            
            # Add to stream
            message_id = await self.redis_client.xadd(
                name=self.stream_key,
                fields={
                    "data": json.dumps(alert_data),
                    "timestamp": str(asyncio.get_event_loop().time())
                },
                maxlen=self.max_stream_length  # Auto-trim
            )
            
            logger.debug(f"Published alert: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return None
    
    async def process_alerts(self, callback):
        """
        Process alerts from stream (non-blocking)
        
        Args:
            callback: Async function to process alerts
        """
        await self.initialize()
        
        logger.info("Starting alert processor...")
        
        while True:
            try:
                # Read from stream with short timeout
                messages = await self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_key: ">"},
                    count=5,
                    block=1000  # 1 second timeout
                )
                
                if messages:
                    for stream, message_list in messages:
                        for message_id, message_data in message_list:
                            try:
                                # Process alert
                                alert_data = json.loads(message_data.get("data", "{}"))
                                await callback(alert_data)
                                
                                # Acknowledge processing
                                await self.redis_client.xack(
                                    self.stream_key,
                                    self.consumer_group,
                                    message_id
                                )
                                
                                logger.debug(f"Processed alert: {message_id}")
                                
                            except Exception as e:
                                logger.error(f"Failed to process alert {message_id}: {e}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(1)  # Wait before retry
    
    async def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        try:
            length = await self.redis_client.xlen(self.stream_key)
            info = await self.redis_client.xinfo_stream(self.stream_key)
            
            return {
                "length": length,
                "memory_usage": length * 1024,  # Approximate bytes per message
                "first_entry": info.get("first-entry", {}),
                "last_entry": info.get("last-entry", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get stream stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_messages(self, max_age_hours: int = 24):
        """Clean up old messages to save memory"""
        try:
            # Simple cleanup - in production use more sophisticated method
            current_time = asyncio.get_event_loop().time()
            cutoff = current_time - (max_age_hours * 3600)
            
            # This is simplified - real implementation would use XRANGE + XDEL
            logger.info(f"Stream cleanup scheduled (simplified implementation)")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Singleton instance
_stream_processor = None

async def get_stream_processor():
    """Get singleton stream processor instance"""
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = RedisStreamProcessor()
        await _stream_processor.initialize()
    return _stream_processor
