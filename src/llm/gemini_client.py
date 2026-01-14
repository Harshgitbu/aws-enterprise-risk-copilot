"""
Memory-optimized Gemini API client for AWS Risk Copilot
WITH CIRCUIT BREAKER PROTECTION
OPTIMIZED FOR: 1GB RAM, AWS t3.micro, Google Free Tier (10 RPM)
Model: gemini-2.5-flash-lite (Your actual free tier allocation)
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import logging

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import google.generativeai as genai
    from google.api_core import exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not installed. Run: pip install google-generativeai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import circuit breaker
try:
    from llm.circuit_wrapper import with_gemini_circuit
except ImportError:
    logger.warning("Circuit breaker not available, running without protection")
    circuit_breaker = None

# Import your existing config
try:
    from llm.config import APIConfig
    CONFIG = APIConfig()
except ImportError:
    # Fallback if config not available
    logger.warning("Could not import APIConfig, using environment variables directly")
    class CONFIG:
        GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
        GEMINI_MODEL = "gemini-2.5-flash-lite"  # Your free tier model
        GEMINI_MAX_TOKENS = 2048  # Reduced for 1GB RAM
        GEMINI_RATE_LIMIT = 8  # Your free tier: 10 RPM, using 8 for safety

@dataclass
class RateLimiter:
    """Memory-efficient rate limiter for YOUR free tier (10 RPM)"""
    def __init__(self, max_requests: int = 7, time_window: int = 60):  # Conservative: 7 not 10
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []  # Simple list of timestamps (memory efficient)
    
    def can_make_request(self) -> bool:
        """Check if we can make a request within rate limits"""
        now = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        # Check if we're under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        # Calculate wait time
        oldest = min(self.requests) if self.requests else now
        wait_time = self.time_window - (now - oldest)
        logger.warning(f"⚠️ Rate limit hit. Wait {wait_time:.1f}s")
        return False

class GeminiClient:
    """
    Gemini API client with circuit breaker protection
    Memory optimized for 1GB RAM
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or CONFIG.GEMINI_API_KEY
        self.model_name = CONFIG.GEMINI_MODEL
        self.max_tokens = CONFIG.GEMINI_MAX_TOKENS
        self.rate_limiter = RateLimiter(max_requests=CONFIG.GEMINI_RATE_LIMIT)
        self.initialized = False
        
        if not self.api_key:
            logger.warning("⚠️  No Gemini API key found")
            self.enabled = False
            return
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.initialized = True
            self.enabled = True
            logger.info(f"✅ Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            self.enabled = False
    
    @staticmethod
    def _prepare_prompt(query: str, context: str = "") -> str:
        """Prepare prompt for risk analysis (memory optimized)"""
        if len(context) > 800:
            context = context[:800] + "..."
        
        prompt = f"""You are an AWS Enterprise Risk Analysis Assistant. Analyze the following risk query.

Context from documents:
{context}

Risk Query: {query}

Provide a concise risk analysis with:
1. Risk Level (Low/Medium/High)
2. Key Risk Factors
3. AWS-specific Mitigation Recommendations
4. Estimated Impact

Keep response under 200 words for memory efficiency."""

        return prompt
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Internal method to call Gemini API
        Called by circuit breaker wrapper
        """
        if not self.initialized:
            raise Exception("Gemini client not initialized")
        
        if not self.rate_limiter.can_make_request():
            raise Exception("Rate limit exceeded")
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.1,  # Low for consistency
                }
            )
            
            if response and response.text:
                return response.text
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    def analyze_risk(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze risk with circuit breaker protection
        """
        if not self.enabled:
            raise Exception("Gemini client not enabled")
        
        start_time = time.time()
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(query, context)
            
            # Call API with circuit breaker protection
            if circuit_breaker:
                response_text = with_gemini_circuit(self._call_gemini_api)(prompt)
            else:
                response_text = self._call_gemini_api(prompt)
            
            # Parse response
            response_time = time.time() - start_time
            
            return {
                "analysis": response_text,
                "model": self.model_name,
                "response_time": response_time,
                "tokens_estimated": len(prompt) // 4 + len(response_text) // 4,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                "analysis": f"Error: {str(e)}",
                "model": self.model_name,
                "response_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "enabled": self.enabled,
            "initialized": self.initialized,
            "model": self.model_name,
            "rate_limit": CONFIG.GEMINI_RATE_LIMIT,
            "recent_requests": len(self.rate_limiter.requests)
        }

def create_gemini_client() -> Optional[GeminiClient]:
    """Factory function to create Gemini client"""
    try:
        return GeminiClient()
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None

