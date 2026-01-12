"""
FastAPI Backend for AWS Risk Copilot
OPTIMIZED FOR: 1GB RAM, Free Tier constraints
Features: RAG pipeline, memory monitoring, rate limiting
"""

import os
import sys
from typing import Optional, List
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import psutil
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from backend.rag_integration import get_rag_pipeline
from backend.redis_cache import redis_cache

# ========== Pydantic Models ==========

class RiskAnalysisRequest(BaseModel):
    """Request model for risk analysis"""
    query: str = Field(..., description="Risk analysis query", example="Analyze S3 bucket security risks")
    document_text: Optional[str] = Field(None, description="Optional document text for RAG context")
    top_k: int = Field(3, description="Number of relevant chunks to retrieve", ge=1, le=10)

class BatchProcessRequest(BaseModel):
    """Request model for batch document processing"""
    document_paths: List[str] = Field(..., description="List of document paths to process")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: str
    memory_mb: float
    memory_percent: float
    components: dict

# ========== FastAPI App ==========

app = FastAPI(
    title="AWS Risk Copilot API",
    description="Enterprise AI Risk Intelligence with Memory-Efficient RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ========== Middleware for Memory Efficiency ==========

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Memory monitoring middleware
@app.middleware("http")
async def memory_monitor_middleware(request, call_next):
    """Monitor memory usage and reject if too high"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Reject if memory exceeds 900MB (leave 100MB buffer for 1GB total)
        if memory_mb > 900:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "Memory limit exceeded",
                    "memory_mb": round(memory_mb, 2),
                    "limit_mb": 900
                }
            )
        
        response = await call_next(request)
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# ========== Dependency Injection ==========

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
        "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
        "percent": round(process.memory_percent(), 2),
        "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
    }

# ========== Routes ==========

@app.get("/")
async def root():
    """Root endpoint with service info"""
    memory = get_memory_info()
    return {
        "service": "AWS Risk Copilot API",
        "version": "1.0.0",
        "status": "operational",
        "memory_mb": memory["rss_mb"],
        "optimized_for": "1GB RAM, AWS Free Tier",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-risk",
            "batch_process": "/batch-process",
            "status": "/status",
            "memory": "/memory"
        }
    }

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    memory = get_memory_info()
    
    # Check RAG pipeline
    try:
        pipeline = get_rag_pipeline()
        components = pipeline.components_loaded
    except:
        components = {"error": "Pipeline not initialized"}
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="AWS Risk Copilot",
        memory_mb=memory["rss_mb"],
        memory_percent=memory["percent"],
        components=components
    )

@app.post("/analyze-risk")
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Analyze risk using RAG pipeline
    
    - Uses vector store for relevant context retrieval
    - Falls back to document text if provided
    - Returns structured risk analysis
    - Cached for efficiency
    """
    logger.info(f"📥 Risk analysis request: {request.query[:50]}...")
    
    try:
        pipeline = get_rag_pipeline()
        
        # Perform RAG analysis
        result = pipeline.analyze_risk_with_rag(
            query=request.query,
            document_text=request.document_text,
            top_k=request.top_k
        )
        
        # Add memory info
        memory = get_memory_info()
        if isinstance(result, dict):
            result["memory_usage_mb"] = memory["rss_mb"]
            result["available_memory_mb"] = memory["available_mb"]
        
        logger.info(f"✅ Risk analysis completed: {memory['rss_mb']}MB used")
        return result
        
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(e)}"
        )

@app.post("/batch-process")
async def batch_process(request: BatchProcessRequest):
    """Batch process documents for indexing"""
    logger.info(f"📥 Batch processing {len(request.document_paths)} documents")
    
    try:
        pipeline = get_rag_pipeline()
        
        # Check memory before processing
        memory = get_memory_info()
        if memory["rss_mb"] > 800:  # Leave 200MB buffer
            return {
                "error": "Memory too high for batch processing",
                "current_memory_mb": memory["rss_mb"],
                "suggestion": "Process fewer documents at once"
            }
        
        results = pipeline.batch_process_documents(request.document_paths)
        
        # Add memory info
        memory_after = get_memory_info()
        results["memory_before_mb"] = memory["rss_mb"]
        results["memory_after_mb"] = memory_after["rss_mb"]
        results["memory_increase_mb"] = round(memory_after["rss_mb"] - memory["rss_mb"], 2)
        
        logger.info(f"✅ Batch processed: {results['processed']} documents, {results['total_chunks']} chunks")
        return results
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    try:
        pipeline = get_rag_pipeline()
        status = pipeline.get_pipeline_status()
        
        # Add Redis cache stats
        if redis_cache and redis_cache.enabled:
            status["cache_stats"] = redis_cache.get_stats()
        
        # Add API stats
        status["api"] = {
            "start_time": app_start_time.isoformat(),
            "uptime_seconds": (datetime.now() - app_start_time).total_seconds()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"error": str(e)}

@app.get("/memory")
async def get_memory():
    """Get detailed memory information"""
    memory = get_memory_info()
    
    # Add process info
    process = psutil.Process(os.getpid())
    memory.update({
        "cpu_percent": process.cpu_percent(interval=0.1),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0,
        "constraints": {
            "max_memory_mb": 1024,
            "target_usage_mb": 800,
            "current_usage_mb": memory["rss_mb"],
            "headroom_mb": 1024 - memory["rss_mb"]
        }
    })
    
    return memory

@app.get("/cache/stats")
async def get_cache_stats():
    """Get Redis cache statistics"""
    if not redis_cache or not redis_cache.enabled:
        return {"enabled": False, "message": "Redis cache not available"}
    
    return redis_cache.get_stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear Redis cache"""
    if not redis_cache or not redis_cache.enabled:
        return {"success": False, "message": "Redis cache not available"}
    
    success = redis_cache.clear_all()
    return {"success": success, "message": "Cache cleared" if success else "Failed to clear cache"}

# ========== App Startup ==========

app_start_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting AWS Risk Copilot Backend...")
    
    # Initialize RAG pipeline (lazy loading)
    pipeline = get_rag_pipeline()
    
    # Log memory info
    memory = get_memory_info()
    logger.info(f"🧠 Memory on startup: {memory['rss_mb']}MB ({memory['percent']}%)")
    logger.info(f"📊 Available memory: {memory['available_mb']}MB")
    
    # Log component status
    logger.info(f"🔧 Components loaded: {list(pipeline.components_loaded.keys())}")
    
    logger.info("✅ Backend startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AWS Risk Copilot Backend...")
    
    # Hint to garbage collector
    import gc
    gc.collect()
    
    logger.info("✅ Backend shutdown complete")

# ========== Run the app ==========

if __name__ == "__main__":
    import uvicorn
    
    # Run with memory optimization settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for memory efficiency
        workers=1,     # Single worker for 1GB RAM
        limit_concurrency=10,  # Limit concurrent requests
        timeout_keep_alive=30  # Shorter keep-alive
    )
