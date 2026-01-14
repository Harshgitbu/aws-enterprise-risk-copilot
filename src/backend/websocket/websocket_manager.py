"""
Memory-optimized WebSocket connection manager
Limits active connections and message queues for 1GB RAM
"""
import asyncio
import json
import logging
import time
from typing import Dict, Set, Any, Optional, List
from collections import defaultdict
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections with memory limits"""
    
    def __init__(self, max_connections: int = 50, max_queue_size: int = 1000):
        self.active_connections: Set[WebSocket] = set()
        self.connection_groups: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.connection_stats: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Memory limits
        self.max_connections = max_connections
        self.max_queue_size = max_queue_size
        self.max_memory_mb = 50  # Max 50MB for WebSocket data
        self.active = False
        
        logger.info(f"WebSocketManager initialized (max: {max_connections} connections)")
    
    async def add_connection(self, websocket: WebSocket, group: str = "default"):
        """Add a WebSocket connection to manager"""
        if len(self.active_connections) >= self.max_connections:
            await self._close_oldest_connection()
        
        self.active_connections.add(websocket)
        self.connection_groups[group].add(websocket)
        
        # Initialize stats
        self.connection_stats[websocket] = {
            "connected_at": time.time(),
            "last_activity": time.time(),
            "messages_sent": 0,
            "messages_received": 0,
            "group": group,
            "client": str(websocket.client) if websocket.client else "unknown"
        }
        
        logger.info(f"New WebSocket connection added to group '{group}'")
    
    async def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from all groups
        for group_name, connections in self.connection_groups.items():
            if websocket in connections:
                connections.remove(websocket)
        
        # Remove stats
        if websocket in self.connection_stats:
            del self.connection_stats[websocket]
        
        logger.debug(f"WebSocket connection removed")
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific WebSocket"""
        try:
            message_json = json.dumps(message)
            await websocket.send_text(message_json)
            
            if websocket in self.connection_stats:
                self.connection_stats[websocket]["messages_sent"] += 1
                self.connection_stats[websocket]["last_activity"] = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            await self.remove_connection(websocket)
            return False
    
    async def broadcast_to_group(self, message: Dict[str, Any], group: str = "alerts"):
        """Broadcast a message to all connections in a group"""
        if group not in self.connection_groups:
            return 0
        
        sent_count = 0
        dead_connections = []
        
        for websocket in list(self.connection_groups[group]):
            try:
                await self.send_message(websocket, message)
                sent_count += 1
            except Exception as e:
                logger.debug(f"Failed to broadcast to connection: {e}")
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for websocket in dead_connections:
            await self.remove_connection(websocket)
        
        logger.debug(f"Broadcast to group '{group}': {sent_count} sent, {len(dead_connections)} failed")
        return sent_count
    
    async def broadcast_to_queue(self, message: Dict[str, Any], group: str = "alerts"):
        """Queue message for broadcast (non-blocking)"""
        if group not in self.message_queues:
            self.message_queues[group] = asyncio.Queue(maxsize=self.max_queue_size)
        
        try:
            await self.message_queues[group].put(message)
        except asyncio.QueueFull:
            logger.warning(f"Message queue for group '{group}' is full, dropping message")
    
    async def process_message_queues(self):
        """Process queued messages for all groups"""
        while self.active:
            for group_name, queue in self.message_queues.items():
                try:
                    # Process up to 10 messages from each queue
                    for _ in range(10):
                        if queue.empty():
                            break
                        
                        message = await queue.get()
                        await self.broadcast_to_group(message, group=group_name)
                        queue.task_done()
                        
                except Exception as e:
                    logger.error(f"Error processing queue '{group_name}': {e}")
            
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)
    
    async def check_memory_limits(self):
        """Check and enforce memory limits"""
        total_connections = len(self.active_connections)
        
        if total_connections > self.max_connections * 0.8:  # 80% threshold
            logger.warning(f"Connection count high: {total_connections}/{self.max_connections}")
        
        # Estimate memory usage (rough calculation)
        estimated_memory_mb = (
            len(self.active_connections) * 0.1 +  # Per connection overhead
            sum(q.qsize() * 0.01 for q in self.message_queues.values())  # Queue overhead
        )
        
        if estimated_memory_mb > self.max_memory_mb * 0.8:
            logger.warning(f"WebSocket memory usage high: {estimated_memory_mb:.1f}MB/{self.max_memory_mb}MB")
    
    async def _close_oldest_connection(self):
        """Close the oldest connection when at limit"""
        if not self.connection_stats:
            return
        
        # Find oldest connection
        oldest_websocket = min(
            self.connection_stats.items(),
            key=lambda x: x[1]["connected_at"]
        )[0]
        
        try:
            await oldest_websocket.close()
            logger.info(f"Closed oldest connection to make room for new connection")
        except:
            pass
        
        await self.remove_connection(oldest_websocket)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        total_queue_size = sum(q.qsize() for q in self.message_queues.values())
        
        return {
            "active_connections": len(self.active_connections),
            "connection_groups": {k: len(v) for k, v in self.connection_groups.items()},
            "queue_sizes": {k: q.qsize() for k, q in self.message_queues.items()},
            "total_queue_messages": total_queue_size,
            "connection_stats": {
                str(conn): stats for conn, stats in self.connection_stats.items()
            },
            "limits": {
                "max_connections": self.max_connections,
                "max_queue_size": self.max_queue_size,
                "max_memory_mb": self.max_memory_mb
            }
        }
    
    async def start(self):
        """Start the WebSocket manager"""
        if self.active:
            return
        
        self.active = True
        asyncio.create_task(self.process_message_queues())
        asyncio.create_task(self._monitor_connections())
        
        logger.info("WebSocketManager started")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        self.active = False
        
        # Close all connections
        for websocket in list(self.active_connections):
            try:
                await websocket.close()
            except:
                pass
        
        self.active_connections.clear()
        self.connection_groups.clear()
        self.message_queues.clear()
        self.connection_stats.clear()
        
        logger.info("WebSocketManager stopped")
    
    async def _monitor_connections(self):
        """Monitor connections for inactivity"""
        while self.active:
            try:
                current_time = time.time()
                inactive_timeout = 300  # 5 minutes
                
                dead_connections = []
                for websocket, stats in self.connection_stats.items():
                    if current_time - stats["last_activity"] > inactive_timeout:
                        dead_connections.append(websocket)
                
                for websocket in dead_connections:
                    logger.info(f"Closing inactive connection: {stats.get('client', 'unknown')}")
                    try:
                        await websocket.close()
                    except:
                        pass
                    await self.remove_connection(websocket)
                
                # Check memory limits periodically
                await self.check_memory_limits()
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
            
            await asyncio.sleep(60)  # Check every minute

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
