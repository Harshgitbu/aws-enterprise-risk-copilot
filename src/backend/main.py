"""
Main FastAPI application for AWS Risk Copilot
Optimized for 1GB RAM on EC2 t3.micro
"""
import os
import sys
sys.path.append('/app/src')
import psutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any

# Add the src directory to Python path for imports
sys.path.append('/app/src')

# Import our modules - using relative imports
from backend.rag_integration import get_rag_pipeline
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
