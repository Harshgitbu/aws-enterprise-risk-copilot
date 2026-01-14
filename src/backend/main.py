"""
Main FastAPI application for AWS Risk Copilot
Optimized for 1GB RAM on EC2 t3.micro
"""
import os
import sys
sys.path.append('/app/src')
import psutil
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any

# Add the src directory to Python path for imports
sys.path.append('/app/src')

# Import our modules - using relative imports
from backend.rag_integration import get_rag_pipeline

# Redis Streams Integration - Day 5
from backend.redis_streams.stream_processor import RedisStreamProcessor
from backend.websocket.websocket_manager import websocket_manager
from backend.websocket.stream_broadcaster import stream_broadcaster
import asyncio
from websockets.server import WebSocketServerProtocol
import websockets
import time
from backend.redis_cache import get_redis_client

# Initialize Redis client
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    global redis_client
    try:
        redis_client = await get_redis_client()
        print("✅ Redis client initialized successfully")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("⚠️  Running without Redis cache - performance may be reduced")
        redis_client = None
    
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()
        print("Redis client closed")

app = FastAPI(
    title="AWS Enterprise Risk Copilot",
    description="Memory-optimized RAG pipeline for risk analysis on EC2 t3.micro",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Redis Streams processor will be initialized on first use
    logger.info("✅ Application startup complete")
    # Start WebSocket manager
    await websocket_manager.start()
    logger.info("✅ WebSocket manager started")
    # Start Redis Streams broadcaster
    await stream_broadcaster.start()
    logger.info("✅ Stream broadcaster started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await stream_broadcaster.stop()
    await websocket_manager.stop()
    logger.info("Application shutdown")
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AWS Enterprise Risk Copilot",
        "status": "operational",
        "memory_constraint": "1GB RAM (EC2 t3.micro)",
        "redis_status": "connected" if redis_client else "disconnected",
        "endpoints": {
            "/health": "System health check",
            "/memory": "Memory usage statistics",
            "/status": "Detailed system status",
            "/analyze-risk": "POST - Analyze risk with RAG"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "risk-copilot",
        "version": "1.0.0",
        "redis": "connected" if redis_client else "disconnected"
    }

@app.get("/memory")
async def memory():
    """Memory usage statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "percent": process.memory_percent(),
        "total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/status")
async def status():
    """Detailed system status"""
    process = psutil.Process(os.getpid())
    
    return {
        "memory": {
            "rss_mb": process.memory_info().rss / (1024 * 1024),
            "percent": process.memory_percent()
        },
        "cpu": {
            "percent": process.cpu_percent(interval=0.1),
            "count": psutil.cpu_count()
        },
        "disk": {
            "total_gb": psutil.disk_usage('/').total / (1024**3),
            "free_gb": psutil.disk_usage('/').free / (1024**3)
        },
        "redis": "connected" if redis_client else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== DAY 5: REDIS STREAMS ALERT ENDPOINTS ====================

@app.post("/alerts/publish", tags=["alerts"])
async def publish_alert(alert: dict):
    """
    Publish a real-time alert to Redis Streams
    
    - alert: dict containing alert data (type, severity, message, source, timestamp)
    Returns: message_id if successful
    """
    try:
        processor = RedisStreamProcessor()
        await processor.initialize()
        
        # Add timestamp if not present
        if "timestamp" not in alert:
            from datetime import datetime
            alert["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        message_id = await processor.publish_alert(alert)
        
        logger.info(f"Alert published: {alert.get('type', 'unknown')} - {message_id}")
        
        return {
            "status": "success",
            "message_id": message_id,
            "alert": alert
        }
    except Exception as e:
        logger.error(f"Error publishing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        logger.info("Alert published: " + str(alert.get("type", "unknown")) + " - " + str(message_id))
async def get_alert_stats():
    """
    Get Redis Streams statistics
    
    Returns: stream statistics including length and memory usage
    """
    try:
        processor = RedisStreamProcessor()
        await processor.initialize()
        
        stats = await processor.get_stream_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting stream stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DAY 5: WEBSOCKET ENDPOINT ====================

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert notifications
    
    Clients can subscribe to receive real-time alerts as they are published
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established: {websocket.client}")
        
        # Add connection to manager
        await websocket_manager.add_connection(websocket, group="alerts")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for message (with timeout to allow health checks)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Update activity
                if websocket in websocket_manager.connection_stats:
                    websocket_manager.connection_stats[websocket]["messages_received"] += 1
                    websocket_manager.connection_stats[websocket]["last_activity"] = time.time()
                
                # Handle ping/pong or other messages
                if data == "ping":
                    await websocket.send_text("pong")
                elif data.startswith("subscribe:"):
                    # Simple subscription mechanism
                    group = data.split(":", 1)[1].strip()
                    await websocket_manager.add_connection(websocket, group=group)
                    await websocket.send_text(json.dumps({
                        "type": "subscription",
                        "group": group,
                        "status": "subscribed"
                    }))
                else:
                    # Echo back for debugging
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "message": data[:100]
                    }))
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": time.time()
                    }))
                except:
                    break  # Connection closed
                
            except websockets.exceptions.ConnectionClosed:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove connection
        try:
            await websocket_manager.remove_connection(websocket)
            logger.info(f"WebSocket connection closed: {websocket.client}")
        except:
            pass

@app.get("/ws/stats", tags=["websocket"])
async def get_websocket_stats():
    """
    Get WebSocket connection statistics
    
    Returns: WebSocket manager statistics including connections and memory usage
    """
    try:
        stats = await websocket_manager.get_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================
@app.get("/alerts/recent", tags=["alerts"])
async def get_recent_alerts(count: int = 10):
    """
    Get recent alerts from Redis Streams
    
    - count: number of alerts to retrieve (default: 10, max: 100)
    Returns: list of recent alerts
    """
    try:
        # Limit count to reasonable number
        count = min(max(1, count), 100)
        

    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/broadcaster/stats", tags=["broadcaster"])
async def get_broadcaster_stats():
    """
    Get Stream Broadcaster statistics
    
    Returns: Broadcaster statistics
    """
    try:
        stats = await stream_broadcaster.get_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting broadcaster stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        processor = RedisStreamProcessor()
        await processor.initialize()
        
        # Get recent messages
        import redis.asyncio as redis
        client = redis.Redis(host="redis", port=6379, decode_responses=True)
        
        # Get recent messages (reverse chronological)
        messages = await client.xrevrange("risk:alerts", count=count)
        
        # Format response
        alerts = []
        for msg_id, data in reversed(messages):  # Reverse to chronological order
            try:
                import json
                alert_data = json.loads(data.get("data", "{}"))
                alerts.append({
                    "id": msg_id,
                    "timestamp": data.get("timestamp", ""),
                    "data": alert_data
                })
            except:
                alerts.append({
                    "id": msg_id,
                    "timestamp": data.get("timestamp", ""),
                    "data": data
                })
        
        await client.close()
        
        return {
            "status": "success",
            "count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================================================================
@app.post("/analyze-risk")
async def analyze_risk(request: Dict[str, Any]):
    """
    Analyze risk using RAG pipeline
    
    Expected JSON:
    {
        "query": "Your risk query",
        "document_text": "Optional document text",
        "top_k": 3
    }
    """
    try:
        query = request.get("query", "")
        document_text = request.get("document_text", "")
        top_k = request.get("top_k", 3)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Get RAG pipeline
        rag_pipeline = get_rag_pipeline()
        
        # Perform analysis
        result = rag_pipeline.analyze_risk(
            query=query,
            document_text=document_text,
            top_k=top_k,
            redis_client=redis_client
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
