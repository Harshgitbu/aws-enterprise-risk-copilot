"""
Unified LLM Service for AWS Risk Copilot
Smart routing between available LLM services
OPTIMIZED FOR: 1GB RAM, Free Tier, $0/month
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedLLMService:
    """
    Smart LLM router that works with available services
    1. Tries Gemini first (free tier: 10 RPM)
    2. Falls back to HuggingFace if available
    3. Gracefully handles missing services
    """
    
    def __init__(self):
        self.clients = {}
        self.init_clients()
        self.start_time = time.time()
        self.stats = {
            "gemini_requests": 0,
            "hf_requests": 0,
            "total_tokens": 0,
            "fallbacks_used": 0
        }
        
        logger.info(f"🔀 UnifiedLLMService initialized")
        logger.info(f"   Available clients: {list(self.clients.keys())}")
    
    def init_clients(self):
        """Initialize available clients"""
        # Try Gemini
        try:
            from llm.gemini_client import create_gemini_client
            self.clients["gemini"] = create_gemini_client()
            logger.info("✅ Gemini client loaded")
        except Exception as e:
            logger.warning(f"Gemini client not available: {e}")
        
        # Try HuggingFace
        try:
            from llm.huggingface_client import create_huggingface_client
            hf_client = create_huggingface_client()
            if hf_client and hf_client.enabled:
                self.clients["huggingface"] = hf_client
                logger.info("✅ HuggingFace client loaded")
        except Exception as e:
            logger.warning(f"HuggingFace client not available: {e}")
    
    def analyze_risk(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze risk using available services
        Memory optimized for 1GB RAM
        """
        start_time = time.time()
        
        # Truncate for memory
        if len(context) > 800:
            context = context[:800] + "..."
        
        # Try Gemini first
        if "gemini" in self.clients:
            try:
                logger.info(f"🔀 Using Gemini for: {query[:50]}...")
                result = self.clients["gemini"].analyze_risk(query, context)
                self.stats["gemini_requests"] += 1
                
                if isinstance(result, dict):
                    result["source"] = "gemini"
                    result["model"] = "gemini-2.5-flash-lite"
                    result["response_time"] = time.time() - start_time
                    
                    # Estimate tokens
                    total_tokens = (len(query) + len(context) + len(str(result))) // 4
                    self.stats["total_tokens"] += total_tokens
                    result["estimated_tokens"] = total_tokens
                
                logger.info("✅ Analysis completed with Gemini")
                return result
            except Exception as e:
                logger.warning(f"Gemini failed: {e}")
                self.stats["fallbacks_used"] += 1
        
        # Try HuggingFace fallback
        if "huggingface" in self.clients:
            try:
                logger.info(f"🔀 Falling back to HuggingFace...")
                result = self.clients["huggingface"].analyze_risk_fallback(query, context)
                self.stats["hf_requests"] += 1
                
                if "error" not in result:
                    result["source"] = "huggingface_fallback"
                    result["response_time"] = time.time() - start_time
                    
                    if "tokens_used" in result:
                        self.stats["total_tokens"] += result["tokens_used"]
                    else:
                        estimated = (len(query) + len(str(result))) // 4
                        self.stats["total_tokens"] += estimated
                        result["estimated_tokens"] = estimated
                    
                    logger.info("✅ Analysis completed with HuggingFace")
                    return result
            except Exception as e:
                logger.error(f"HuggingFace also failed: {e}")
        
        # All failed
        return {
            "error": "No LLM services available",
            "query": query[:100],
            "available_services": list(self.clients.keys()),
            "response_time": time.time() - start_time,
            "suggestions": [
                "Check API keys in .env",
                "Verify internet connection",
                "Check service status with get_service_status()"
            ]
        }
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from available service"""
        if not texts:
            return None
        
        # Limit batch size
        if len(texts) > 5:
            texts = texts[:5]
        
        # Prefer HuggingFace for embeddings
        if "huggingface" in self.clients:
            try:
                return self.clients["huggingface"].get_embeddings(texts)
            except:
                pass
        
        # No embeddings available
        logger.warning("No embedding service available")
        return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        status = {
            "uptime_hours": round((time.time() - self.start_time) / 3600, 2),
            "available_clients": list(self.clients.keys()),
            "stats": self.stats.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add client-specific info
        for name, client in self.clients.items():
            try:
                if name == "gemini":
                    status["gemini_stats"] = client.get_usage_stats()
                elif name == "huggingface":
                    status["huggingface_stats"] = client.get_usage_stats()
            except:
                pass
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check"""
        health = {"status": "operational", "services": {}}
        
        for name, client in self.clients.items():
            try:
                if name == "gemini":
                    health["services"]["gemini"] = {
                        "status": "healthy" if client.test_connection() else "unhealthy",
                        "model": "gemini-2.5-flash-lite"
                    }
                elif name == "huggingface":
                    health["services"]["huggingface"] = {
                        "status": "healthy" if client.test_connection() else "unhealthy",
                        "model": "microsoft/phi-2"
                    }
            except:
                health["services"][name] = {"status": "error"}
        
        return health


# Factory function
def create_llm_service() -> UnifiedLLMService:
    return UnifiedLLMService()
