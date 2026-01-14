"""
Memory-efficient WebSocket manager for real-time alert broadcasting
Optimized for 1GB RAM constraint
"""
import asyncio
import json
import logging
import time
from typing import Dict, Set, Any, Optional
from collections import defaultdict
import websockets
from websockets.server import WebSocketServerProtocol
import psutil

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Memory-constrained WebSocket connection manager"""
    
    def __init__(self, max_connections: int = 50, max_memory_mb: int = 50):
        """
        Initialize WebSocket manager
        
        Args:
            max_connections: Maximum simultaneous connections
            max_memory_mb: Maximum memory allowed for WebSocket operations
        """
        self.connections: Set[WebSocketServerProtocol] = set()
        self.connection_times: Dict[WebSocketServerProtocol, float] = {}
        self.connection_stats: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        self.max_connections = max_connections
        self.max_memory_mb = max_memory_mb
        self.active = False
        self.broadcast_queue = asyncio.Queue(maxsize=1000)
        
        # Memory monitoring
        self.memory_check_interval = 30  # seconds
        self.last_memory_check = 0
        
        # Connection groups (for targeted broadcasts)
        self.connection_groups = defaultdict(set)
        
        logger.info(f"WebSocketManager initialized (max_connections={max_connections}, max_memory={max_memory_mb}MB)")
    
    async def check_memory_limits(self) -> bool:
        """Check if we're within memory limits"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Check overall memory
            if memory_mb > self.max_memory_mb:
                logger.warning(f"WebSocket memory high: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                
                # If we have too many connections, drop oldest
                if len(self.connections) > 10:
                    await self._cleanup_old_connections()
                return False
            
            return True
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return True  # Don't fail on memory check errors
    
    async def _cleanup_old_connections(self):
        """Clean up old connections to save memory"""
        if not self.connections:
            return
        
        # Sort connections by connection time
        sorted_connections = sorted(
            self.connection_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest 10% of connections
        remove_count = max(1, len(sorted_connections) // 10)
        for ws, connect_time in sorted_connections[:remove_count]:
            try:
                await self.remove_connection(ws)
                logger.info(f"Cleaned up old connection (age: {time.time() - connect_time:.0f}s)")
            except Exception as e:
                logger.error(f"Error cleaning up connection: {e}")
    
    async def add_connection(self, websocket: WebSocketServerProtocol, group: str = "default"):
        """
        Add a WebSocket connection
        
        Args:
            websocket: WebSocket connection
            group: Connection group for targeted broadcasts
        """
        # Check memory limits before adding
        if not await self.check_memory_limits():
            raise MemoryError(f"Memory limit reached ({self.max_memory_mb}MB)")
        
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            # Try to clean up old connections first
            await self._cleanup_old_connections()
            
            if len(self.connections) >= self.max_connections:
                raise ConnectionError(f"Maximum connections reached ({self.max_connections})")
        
        self.connections.add(websocket)
        self.connection_times[websocket] = time.time()
        self.connection_groups[group].add(websocket)
        
        # Initialize stats
        self.connection_stats[websocket] = {
            "connected_at": time.time(),
            "messages_sent": 0,
            "messages_received": 0,
            "last_activity": time.time(),
            "group": group
        }
        
        logger.info(f"WebSocket connection added. Total: {len(self.connections)}/{self.max_connections}")
    
    async def remove_connection(self, websocket: WebSocketServerProtocol):
        """Remove a WebSocket connection"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            
            # Remove from all groups
            for group in self.connection_groups.values():
                group.discard(websocket)
            
            # Clean up empty groups
            self.connection_groups = defaultdict(
                set,
                {k: v for k, v in self.connection_groups.items() if v}
            )
            
            # Remove stats
            if websocket in self.connection_stats:
                del self.connection_stats[websocket]
            
            if websocket in self.connection_times:
                del self.connection_times[websocket]
            
            logger.info(f"WebSocket connection removed. Total: {len(self.connections)}")
    
    async def broadcast(self, message: Dict[str, Any], group: Optional[str] = None):
        """
        Broadcast message to all connected clients or a specific group
        
        Args:
            message: Message to broadcast
            group: Optional group to target (None = broadcast to all)
        """
        if not self.connections:
            return
        
        # Convert message to JSON
        try:
            message_json = json.dumps(message)
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            return
        
        # Get connections to broadcast to
        if group:
            connections = self.connection_groups.get(group, set())
        else:
            connections = self.connections.copy()
        
        # Broadcast to connections
        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send(message_json)
                # Update stats
                if websocket in self.connection_stats:
                    self.connection_stats[websocket]["messages_sent"] += 1
                    self.connection_stats[websocket]["last_activity"] = time.time()
            except (websockets.exceptions.ConnectionClosed, ConnectionError):
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            try:
                await self.remove_connection(ws)
            except Exception as e:
                logger.debug(f"Error removing disconnected WebSocket: {e}")
        
        if connections:
            logger.debug(f"Broadcasted to {len(connections) - len(disconnected)}/{len(connections)} clients")
    
    async def broadcast_to_queue(self, message: Dict[str, Any], group: Optional[str] = None):
        """
        Add message to broadcast queue (non-blocking)
        
        Args:
            message: Message to broadcast
            group: Optional group to target
        """
        try:
            await self.broadcast_queue.put((message, group))
        except asyncio.QueueFull:
            logger.warning("Broadcast queue full, dropping message")
    
    async def process_broadcast_queue(self):
        """Process messages from broadcast queue"""
        logger.info("Starting broadcast queue processor")
        
        while self.active:
            try:
                # Get message from queue with timeout
                message, group = await asyncio.wait_for(
                    self.broadcast_queue.get(),
                    timeout=1.0
                )
                
                # Broadcast message
                await self.broadcast(message, group)
                
                # Mark task as done
                self.broadcast_queue.task_done()
                
            except asyncio.TimeoutError:
                # Check memory periodically
                current_time = time.time()
                if current_time - self.last_memory_check > self.memory_check_interval:
                    await self.check_memory_limits()
                    self.last_memory_check = current_time
                    
            except Exception as e:
                logger.error(f"Error processing broadcast queue: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                "active": self.active,
                "connections": len(self.connections),
                "max_connections": self.max_connections,
                "connection_groups": len(self.connection_groups),
                "queue_size": self.broadcast_queue.qsize(),
                "memory_usage_mb": memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "connection_stats": {
                    "total_messages_sent": sum(
                        stats["messages_sent"] 
                        for stats in self.connection_stats.values()
                    ),
                    "average_connection_age": (
                        sum(time.time() - connect_time 
                            for connect_time in self.connection_times.values()) 
                        / max(1, len(self.connection_times))
                    )
                }
            }
        except Exception as e:
            logger.error(f"Error getting WebSocket stats: {e}")
            return {"error": str(e)}
    
    async def start(self):
        """Start WebSocket manager"""
        if self.active:
            return
        
        self.active = True
        
        # Start broadcast queue processor
        asyncio.create_task(self.process_broadcast_queue())
        
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop WebSocket manager"""
        self.active = False
        
        # Close all connections
        for websocket in list(self.connections):
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
        
        self.connections.clear()
        self.connection_groups.clear()
        self.connection_stats.clear()
        self.connection_times.clear()
        
        logger.info("WebSocket manager stopped")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
