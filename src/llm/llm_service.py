"""
Unified LLM Service for AWS Risk Copilot WITH CIRCUIT BREAKERS
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

# Import circuit breaker for monitoring
try:
    from llm.circuit_breaker_config import circuit_breaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    logger.warning("Circuit breaker configuration not available")


class UnifiedLLMService:
    """
    Smart LLM router with circuit breaker protection
    1. Tries Gemini first (free tier: 10 RPM) with circuit breaker
    2. Falls back to HuggingFace with circuit breaker
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
            "fallbacks_used": 0,
            "circuit_breaker_trips": 0,
            "last_error": None
        }
        
        logger.info(f"🔀 UnifiedLLMService initialized (Circuit Breakers: {CIRCUIT_BREAKER_AVAILABLE})")
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
        Analyze risk using available services with circuit breaker protection
        Memory optimized for 1GB RAM
        """
        start_time = time.time()
        
        # Truncate for memory
        if len(context) > 800:
            context = context[:800] + "..."
        
        # Check circuit breaker states first
        circuit_info = {}
        if CIRCUIT_BREAKER_AVAILABLE:
            circuit_info = circuit_breaker.get_circuit_stats()
            
            # Log circuit states
            for service, state in circuit_info.items():
                if state.get("circuit_state") == "open":
                    logger.warning(f"Circuit OPEN for {service}, will skip or use fallback")
        
        # Try Gemini first (if circuit not open)
        if "gemini" in self.clients:
            gemini_open = circuit_info.get("gemini", {}).get("circuit_state") == "open"
            
            if not gemini_open:
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
                    self.stats["last_error"] = str(e)
                    
                    # Check if circuit tripped
                    if "circuit" in str(e).lower() or "breaker" in str(e).lower():
                        self.stats["circuit_breaker_trips"] += 1
            else:
                logger.warning("Gemini circuit OPEN, skipping to fallback")
        
        # Try HuggingFace fallback (if circuit not open)
        if "huggingface" in self.clients:
            hf_open = circuit_info.get("huggingface", {}).get("circuit_state") == "open"
            
            if not hf_open:
                try:
                    logger.info(f"🔀 Falling back to HuggingFace...")
                    result = self.clients["huggingface"].analyze_risk_fallback(query, context)
                    self.stats["hf_requests"] += 1
                    
                    if isinstance(result, dict):
                        result["source"] = "huggingface"
                        result["response_time"] = time.time() - start_time
                    
                    logger.info("✅ Analysis completed with HuggingFace fallback")
                    return result
                except Exception as e:
                    logger.error(f"HuggingFace fallback also failed: {e}")
                    self.stats["last_error"] = str(e)
                    
                    # Check if circuit tripped
                    if "circuit" in str(e).lower() or "breaker" in str(e).lower():
                        self.stats["circuit_breaker_trips"] += 1
            else:
                logger.warning("HuggingFace circuit OPEN, no fallback available")
        
        # All services failed or circuits open
        fallback_time = time.time() - start_time
        
        return {
            "analysis": "Unable to process request. All LLM services are currently unavailable. This could be due to: 1) API rate limits exceeded, 2) Network issues, 3) Service outages. Please try again later.",
            "model": "none",
            "response_time": fallback_time,
            "status": "error",
            "source": "fallback_error",
            "circuit_states": circuit_info if CIRCUIT_BREAKER_AVAILABLE else None,
            "stats": self.stats
        }
    
    def generate_risk_analysis(self, query: str, context: str = "", relevant_docs: list = None) -> str:
        """Alias for analyze_risk for backward compatibility"""
        result = self.analyze_risk(query, context)
        return result.get("analysis", "Analysis unavailable")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics including circuit breaker states"""
        stats = {
            "uptime": time.time() - self.start_time,
            "requests": self.stats,
            "available_clients": list(self.clients.keys()),
            "client_details": {}
        }
        
        # Get client-specific stats
        for name, client in self.clients.items():
            if hasattr(client, 'get_stats'):
                stats["client_details"][name] = client.get_stats()
        
        # Get circuit breaker stats if available
        if CIRCUIT_BREAKER_AVAILABLE:
            stats["circuit_breakers"] = circuit_breaker.get_circuit_stats()
        
        return stats
    
    def reset_circuits(self):
        """Reset all circuit breakers"""
        if CIRCUIT_BREAKER_AVAILABLE:
            circuit_breaker.reset_circuit("gemini")
            circuit_breaker.reset_circuit("huggingface")
            logger.info("All circuit breakers reset")
            return True
        return False

