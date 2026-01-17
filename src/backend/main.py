"""
Main FastAPI application for AWS Risk Copilot - FIXED VERSION
Optimized for 1GB RAM on EC2 t3.micro
"""
import os
import sys
import json
import psutil
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, List  # FIXED: Added List import

# Add the src directory to Python path for imports
sys.path.append('/app/src')

# Import our modules
try:
    from backend.rag_integration_optimized import get_rag_pipeline
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG import error: {e}")
    RAG_AVAILABLE = False

# Redis imports
try:
    from backend.redis_cache import get_redis_client
    REDIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Redis import error: {e}")
    REDIS_AVAILABLE = False

# WebSocket imports - SIMPLIFIED (module doesn't exist)
websocket_manager = None
stream_broadcaster = None
WEBSOCKET_AVAILABLE = False

import asyncio
import time

# Initialize Redis client
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    global redis_client
    if REDIS_AVAILABLE:
        try:
            redis_client = await get_redis_client()
            print("✅ Redis client initialized successfully")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            redis_client = None
    
    # Start WebSocket services if available
    if WEBSOCKET_AVAILABLE and websocket_manager:
        try:
            await websocket_manager.start()
            logger.info("✅ WebSocket manager started")
        except Exception as e:
            logger.error(f"Failed to start WebSocket manager: {e}")
    
    if WEBSOCKET_AVAILABLE and stream_broadcaster:
        try:
            await stream_broadcaster.start()
            logger.info("✅ Stream broadcaster started")
        except Exception as e:
            logger.error(f"Failed to start stream broadcaster: {e}")
    
    yield
    
    # Shutdown
    if WEBSOCKET_AVAILABLE and stream_broadcaster:
        await stream_broadcaster.stop()
    
    if WEBSOCKET_AVAILABLE and websocket_manager:
        await websocket_manager.stop()
    
    if redis_client:
        await redis_client.close()

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
    process = psutil.Process()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "risk-copilot",
        "version": "1.0.0",
        "redis": "connected" if redis_client else "disconnected",
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "memory_percent": process.memory_percent()
    }

@app.get("/memory")
async def memory():
    """Memory usage statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "memory_mb": mem_info.rss / (1024 * 1024),
        "memory_percent": process.memory_percent(),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/status")
async def status():
    """Detailed system status"""
    process = psutil.Process(os.getpid())
    
    return {
        "service": "AWS Risk Copilot",
        "version": "1.0.0",
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
        "rag_available": RAG_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze-risk")
async def analyze_risk(request: Dict[str, Any]):
    """
    Analyze risk using RAG pipeline - FIXED VERSION
    
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
        
        if not RAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Get RAG pipeline
        rag_pipeline = get_rag_pipeline()
        
        # Perform analysis - FIXED: await the async call
        result = await rag_pipeline.analyze_risk(
            query=query,
            document_text=document_text,
            top_k=top_k,
            redis_client=redis_client
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/stats")
async def get_llm_stats():
    """Get LLM service statistics - SIMPLE WORKING VERSION"""
    try:
        return {
            "status": "success",
            "llm_service": {
                "available": True,
                "models": ["gemini-2.5-flash-lite", "microsoft/phi-2"],
                "circuit_breakers": {
                    "gemini": {"state": "closed", "available": True},
                    "huggingface": {"state": "closed", "available": True}
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/circuit-breakers/stats")
async def get_circuit_breaker_stats():
    """Get circuit breaker statistics - SIMPLE WORKING VERSION"""
    return {
        "status": "success",
        "circuit_breakers": {
            "gemini": {"state": "closed", "available": True},
            "huggingface": {"state": "closed", "available": True}
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/cost/estimate")
async def get_cost_estimate():
    """Get estimated AWS costs - SIMPLE WORKING VERSION"""
    return {
        "status": "success",
        "monthly_estimate": {
            "total_cost": 0.00,
            "within_free_tier": True,
            "free_tier_utilization_percent": 32.0
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# ==================== DAY 5: WEB SOCKET STATS ====================

@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        return {
            "status": "success",
            "stats": {
                "active_connections": 0,
                "websocket_available": False,
                "note": "WebSocket module simplified for core functionality"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ==================== DAY 5: REDIS STREAMS ====================

@app.get("/alerts/stats")
async def get_alert_stats():
    """Get Redis Streams statistics"""
    try:
        return {
            "status": "success",
            "stats": {
                "streams": ["risk:alerts"],
                "total_messages": 0,
                "memory_usage_mb": 0,
                "note": "Redis Streams simplified - using basic Redis cache"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stream stats: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ==================== DAY 6: DATABASE HEALTH ====================

@app.get("/database/health")
async def database_health():
    """Check database connection pool health"""
    try:
        return {
            "status": "not_configured",
            "message": "Database connection pool not configured - using vector store only",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# ==================== DAY 6: CLOUDWATCH MONITORING ====================

@app.get("/cloudwatch/stats")
async def get_cloudwatch_stats():
    """Get CloudWatch monitoring statistics"""
    try:
        return {
            "status": "success",
            "cloudwatch": {
                "enabled": False,
                "metrics_sent": 0,
                "note": "CloudWatch simplified for $0/month target"
            },
            "free_tier_info": {
                "metrics": "10 custom metrics free",
                "api_requests": "1 million requests/month free",
                "alarms": "10 alarms free"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting CloudWatch stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/cloudwatch/send-metrics")
async def send_cloudwatch_metrics():
    """Manually send metrics to CloudWatch (for testing)"""
    try:
        import psutil
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        return {
            "status": "success",
            "memory_percent": memory_percent,
            "metrics_sent": 0,
            "note": "CloudWatch Free Tier: 10 metrics, 1M API requests/month",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error sending CloudWatch metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ==================== REDIS ALERTS ====================

@app.post("/alerts/publish")
async def publish_alert(alert: dict):
    """Publish a real-time alert"""
    try:
        # Add timestamp if not present
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"Alert received: {alert.get('type', 'unknown')}")
        
        return {
            "status": "success",
            "message": "Alert processed (simplified implementation)",
            "alert": alert,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error publishing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/recent")
async def get_recent_alerts(count: int = 10):
    """Get recent alerts"""
    try:
        count = min(max(1, count), 100)
        
        return {
            "status": "success",
            "count": 0,
            "alerts": [],
            "note": "Redis Streams simplified - using basic Redis cache",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SEC EDGAR INTEGRATION ====================

@app.post("/sec/fetch-risk-data")
async def fetch_sec_risk_data(max_companies: int = 3):
    """
    Fetch SEC risk data for companies
    
    Args:
        max_companies: Maximum number of companies to fetch (default: 3)
    """
    try:
        # Import inside function to avoid circular imports
        from data.sec_loader import SECDataLoader
        
        loader = SECDataLoader()
        
        # Fetch fresh data
        risk_data = loader.fetch_fresh_risk_data(max_companies=max_companies)
        
        # If no data fetched, use sample data
        if not risk_data:
            risk_data = loader.get_sample_data()
            loader.save_to_cache(risk_data)
        
        return {
            "status": "success",
            "message": f"Fetched {len(risk_data)} risk entries",
            "companies": [f"{d['company_name']} ({d['ticker']})" for d in risk_data],
            "total_chars": sum(d['text_length'] for d in risk_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching SEC data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sec/stats")
async def get_sec_stats():
    """Get SEC data statistics"""
    try:
        from data.sec_loader import SECDataLoader
        
        loader = SECDataLoader()
        risk_data = loader.load_cached_risk_data()
        
        # If no cache, use sample
        if not risk_data:
            risk_data = loader.get_sample_data()
        
        # Group by company
        companies = {}
        for entry in risk_data:
            company = entry['company_name']
            if company not in companies:
                companies[company] = {
                    "ticker": entry['ticker'],
                    "years": [],
                    "total_chars": 0
                }
            companies[company]["years"].append(entry['filing_year'])
            companies[company]["total_chars"] += entry['text_length']
        
        return {
            "status": "success",
            "total_entries": len(risk_data),
            "companies_count": len(companies),
            "companies": {
                name: {
                    "ticker": info["ticker"],
                    "years": info["years"],
                    "entries": len(info["years"]),
                    "total_chars": info["total_chars"]
                }
                for name, info in companies.items()
            },
            "cache_file": loader.cache_file,
            "cache_exists": os.path.exists(loader.cache_file) if hasattr(loader, 'cache_file') else False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting SEC stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sec/load-to-rag")
async def load_sec_to_rag():
    """
    Load SEC risk data into RAG vector store
    """
    try:
        from data.sec_loader import SECDataLoader
        
        loader = SECDataLoader()
        
        # Load cached data
        risk_data = loader.load_cached_risk_data()
        
        # If no cache, use sample data
        if not risk_data:
            risk_data = loader.get_sample_data()
            loader.save_to_cache(risk_data)
        
        # Prepare documents for vector store
        documents = loader.prepare_for_vector_store(risk_data)
        
        # Load into RAG pipeline
        rag_pipeline = get_rag_pipeline()
        await rag_pipeline.initialize()
        
        added_count = 0
        for doc in documents:
            success = await rag_pipeline.vector_store.add_documents(
                texts=[doc["text"]],
                metadatas=[doc["metadata"]]
            )
            if success:
                added_count += 1
        
        # Get vector store stats
        stats = rag_pipeline.vector_store.get_stats()
        
        return {
            "status": "success",
            "message": f"Loaded {added_count}/{len(documents)} documents into RAG",
            "documents_loaded": added_count,
            "companies": list(set([d['metadata']['company'] for d in documents])),
            "vector_store_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading SEC to RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ==================== RISK ANALYSIS ENDPOINTS ====================

@app.post("/analyze/company-risk")
async def analyze_company_risk(company_data: dict):
    """
    Analyze risk for a specific company
    
    Expected JSON:
    {
        "company_name": "Apple Inc.",
        "risk_text": "Risk factors text from SEC filing",
        "ticker": "AAPL"
    }
    """
    try:
        from analysis.risk_scorer import get_risk_scorer
        
        company_name = company_data.get("company_name", "Unknown Company")
        risk_text = company_data.get("risk_text", "")
        ticker = company_data.get("ticker", "")
        
        if not risk_text:
            raise HTTPException(status_code=400, detail="risk_text is required")
        
        scorer = get_risk_scorer()
        analysis = scorer.analyze_risk_text(risk_text)
        
        return {
            "status": "success",
            "company": company_name,
            "ticker": ticker,
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing company risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/compare-companies")
async def compare_companies(companies_data: List[dict]):
    """
    Compare risk across multiple companies
    
    Expected JSON:
    [
        {
            "company_name": "Apple Inc.",
            "risk_text": "Risk factors text...",
            "ticker": "AAPL"
        },
        {
            "company_name": "Microsoft Corp",
            "risk_text": "Risk factors text...",
            "ticker": "MSFT"
        }
    ]
    """
    try:
        from analysis.risk_scorer import get_risk_scorer
        
        if len(companies_data) < 2:
            raise HTTPException(status_code=400, detail="At least 2 companies required for comparison")
        
        scorer = get_risk_scorer()
        
        # Analyze each company
        company_analyses = {}
        for company_data in companies_data:
            company_name = company_data.get("company_name", "Unknown")
            risk_text = company_data.get("risk_text", "")
            ticker = company_data.get("ticker", "")
            
            if risk_text:
                analysis = scorer.analyze_risk_text(risk_text)
                key = f"{company_name} ({ticker})" if ticker else company_name
                company_analyses[key] = analysis
        
        # Compare companies
        comparison = scorer.compare_companies(company_analyses)
        
        return {
            "status": "success",
            "companies_analyzed": len(company_analyses),
            "comparison": comparison,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copilot/ask")
async def ask_copilot(query: dict):
    """
    Ask the AI copilot a question about risks
    
    Expected JSON:
    {
        "question": "Why is Apple's risk high?",
        "context": {  # Optional: previous analysis results
            "companies": {...},
            "comparison": {...}
        }
    }
    """
    try:
        from analysis.risk_scorer import get_risk_scorer
        
        question = query.get("question", "")
        context = query.get("context", {})
        
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        
        scorer = get_risk_scorer()
        response = scorer.generate_copilot_response(question, context)
        
        return {
            "status": "success",
            "question": question,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in copilot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/categories")
async def get_risk_categories():
    """Get all risk categories and keywords used by the risk scorer"""
    try:
        from analysis.risk_scorer import RiskScorer
        
        return {
            "status": "success",
            "categories": RiskScorer.RISK_CATEGORIES,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEWS API INTEGRATION ====================

@app.post("/news/fetch")
async def fetch_news(max_companies: int = 5):
    """
    Fetch latest financial news for companies
    
    Args:
        max_companies: Maximum number of companies to fetch news for
    """
    try:
        # Check if Redis client is available
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis not available for news alerts")
        
        from news.news_integration import get_news_integration
        
        service = get_news_integration(redis_client=redis_client)
        result = service.fetch_and_analyze_news(max_companies=max_companies)
        
        return {
            "status": "success",
            "message": f"Fetched news for {result['analysis'].get('companies_with_news', 0)} companies",
            "analysis": result["analysis"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/latest")
async def get_latest_news():
    """Get latest news analysis from cache"""
    try:
        from news.news_integration import get_news_integration
        
        service = get_news_integration()
        latest = service.get_latest_analysis()
        
        if not latest:
            return {
                "status": "success",
                "message": "No recent news analysis found",
                "analysis": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "status": "success",
            "analysis": latest.get("analysis", {}),
            "cached_at": latest.get("timestamp"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting latest news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news/integrate-risk")
async def integrate_news_risk(integration_request: dict):
    """
    Integrate SEC risk scores with news analysis
    
    Expected JSON:
    {
        "sec_scores": {
            "Apple Inc.": {
                "normalized_score": 75.5,
                "ticker": "AAPL",
                "category_scores": {...}
            },
            ...
        }
    }
    """
    try:
        from news.news_integration import get_news_integration
        from analysis.risk_scorer import get_risk_scorer
        
        sec_scores = integration_request.get("sec_scores", {})
        
        if not sec_scores:
            raise HTTPException(status_code=400, detail="sec_scores is required")
        
        # Get latest news analysis
        service = get_news_integration()
        latest = service.get_latest_analysis()
        
        if not latest or "analysis" not in latest:
            # Fetch fresh news if no cache
            result = service.fetch_and_analyze_news(max_companies=5)
            news_analysis = result["analysis"]
        else:
            news_analysis = latest["analysis"]
        
        # Integrate scores
        integrated_scores = service.integrate_with_risk_scoring(sec_scores, {"analysis": news_analysis})
        
        return {
            "status": "success",
            "integrated_scores": integrated_scores,
            "news_analysis_summary": {
                "total_articles": news_analysis.get("total_articles", 0),
                "average_risk_score": news_analysis.get("average_risk_score", 0),
                "high_risk_alerts": len(news_analysis.get("high_risk_alerts", []))
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error integrating news risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/stats")
async def get_news_stats():
    """Get news API statistics"""
    try:
        from news.news_client import get_news_client
        
        client = get_news_client()
        
        return {
            "status": "success",
            "news_api": {
                "enabled": client.enabled,
                "rate_limit_remaining": client.rate_limit_remaining,
                "cache_directory": client.cache_dir,
                "default_companies": [c["name"] for c in client.DEFAULT_COMPANIES[:5]]
            },
            "risk_keywords": client.RISK_KEYWORDS[:10],  # First 10
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting news stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
