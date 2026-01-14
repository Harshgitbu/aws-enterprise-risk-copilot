"""
Bridge between Redis Streams and WebSocket broadcasts
"""
import asyncio
import json
import logging
from typing import Dict, Any
import redis.asyncio as redis

from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

class StreamBroadcaster:
    """Broadcast Redis Stream alerts to WebSocket clients"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.client = None
        self.active = False
        self.stream_key = "risk:alerts"
        self.consumer_group = "websocket_broadcaster"
        self.consumer_name = "broadcaster_1"
        
    async def connect(self):
        """Connect to Redis"""
        if self.client is None:
            try:
                self.client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                await self.client.ping()
                logger.info(f"✅ StreamBroadcaster connected to Redis")
                
                # Create consumer group if not exists
                try:
                    await self.client.xgroup_create(
                        name=self.stream_key,
                        groupname=self.consumer_group,
                        id="$",
                        mkstream=True
                    )
                    logger.info(f"Created consumer group: {self.consumer_group}")
                except Exception as e:
                    if "BUSYGROUP" in str(e):
                        logger.info(f"Consumer group already exists: {self.consumer_group}")
                    else:
                        raise
                        
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.client = None
                raise
    
    async def broadcast_new_alerts(self):
        """Continuously read and broadcast new alerts"""
        logger.info("Starting alert broadcaster")
        
        last_id = ">"
        
        while self.active:
            try:
                # Read new messages
                messages = await self.client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_key: last_id},
                    count=10,
                    block=5000  # 5 second block
                )
                
                if messages:
                    for stream, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            # Broadcast to WebSocket clients
                            await self._broadcast_alert(message_id, message_data)
                            
                            # Acknowledge message
                            await self.client.xack(
                                self.stream_key,
                                self.consumer_group,
                                message_id
                            )
                            
                            last_id = message_id
                            
                            logger.debug(f"Broadcast alert: {message_id}")
                
                # Check memory every iteration
                await websocket_manager.check_memory_limits()
                
            except Exception as e:
                logger.error(f"Error in broadcaster: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _broadcast_alert(self, message_id: str, message_data: Dict[str, Any]):
        """Broadcast a single alert to WebSocket clients"""
        try:
            # Parse alert data
            alert_data = {}
            if "data" in message_data:
                try:
                    alert_data = json.loads(message_data["data"])
                except:
                    alert_data = {"raw_data": message_data["data"]}
            
            # Add metadata
            broadcast_message = {
                "type": "alert",
                "id": message_id,
                "timestamp": message_data.get("timestamp", ""),
                "data": alert_data,
                "broadcast_at": asyncio.get_event_loop().time()
            }
            
            # Broadcast to WebSocket clients
            await websocket_manager.broadcast_to_queue(broadcast_message, group="alerts")
            
            logger.debug(f"Queued alert for broadcast: {message_id}")
            
        except Exception as e:
            logger.error(f"Error broadcasting alert {message_id}: {e}")
    
    async def start(self):
        """Start the broadcaster"""
        if self.active:
            return
        
        await self.connect()
        self.active = True
        
        # Start broadcasting task
        asyncio.create_task(self.broadcast_new_alerts())
        
        logger.info("StreamBroadcaster started")
    
    async def stop(self):
        """Stop the broadcaster"""
        self.active = False
        
        if self.client:
            await self.client.close()
            self.client = None
        
        logger.info("StreamBroadcaster stopped")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics"""
        return {
            "active": self.active,
            "stream_key": self.stream_key,
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name,
            "redis_connected": self.client is not None
        }

# Global broadcaster instance
stream_broadcaster = StreamBroadcaster()
