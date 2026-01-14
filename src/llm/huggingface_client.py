"""
Memory-optimized HuggingFace client for AWS Risk Copilot
WITH CIRCUIT BREAKER PROTECTION
Uses huggingface_hub library with smart rate limit handling
OPTIMIZED FOR: 1GB RAM, Free Tier (API: 1000/5min), $0/month
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import circuit breaker
try:
    from llm.circuit_wrapper import with_hf_circuit
except ImportError:
    logger.warning("Circuit breaker not available, running without protection")
    circuit_breaker = None

# Import your config
try:
    from llm.config import APIConfig
    CONFIG = APIConfig()
except ImportError:
    logger.warning("Could not import APIConfig")
    class CONFIG:
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        HF_LLM_MODEL = "microsoft/phi-2"
        HF_RATE_LIMIT = 30

# Try to import huggingface_hub - FIXED IMPORT
try:
    from huggingface_hub import InferenceClient, HfApi
    HF_HUB_AVAILABLE = True
    logger.debug("huggingface_hub imported successfully")
except ImportError as e:
    HF_HUB_AVAILABLE = False
    logger.warning(f"huggingface_hub import failed: {e}")

@dataclass
class HFRateLimiter:
    """Simple rate limiter for HuggingFace Free Tier"""
    def __init__(self, max_requests: int = 25, time_window: int = 300):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if we can make a request"""
        now = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        oldest = min(self.requests) if self.requests else now
        wait_time = self.time_window - (now - oldest)
        logger.warning(f"⚠️ HF Rate limit hit. Wait {wait_time:.1f}s")
        return False

class HuggingFaceClient:
    """
    HuggingFace client with circuit breaker protection
    Memory optimized for 1GB RAM
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or CONFIG.HF_TOKEN
        self.llm_model = CONFIG.HF_LLM_MODEL
        self.rate_limiter = HFRateLimiter()
        self.client = None
        self.enabled = False
        
        if not self.api_token:
            logger.warning("⚠️  No HuggingFace token found")
            return
        
        if not HF_HUB_AVAILABLE:
            logger.warning("⚠️  huggingface_hub not available")
            return
        
        try:
            self.client = InferenceClient(
                model=self.llm_model,
                token=self.api_token,
                timeout=30  # 30 second timeout
            )
            self.enabled = True
            logger.info(f"✅ HuggingFace client initialized with model: {self.llm_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace: {e}")
            self.enabled = False
    
    @staticmethod
    def _prepare_prompt(query: str, context: str = "") -> str:
        """Prepare prompt for fallback analysis (memory optimized)"""
        if len(context) > 600:  # Even shorter for HF fallback
            context = context[:600] + "..."
        
        prompt = f"""<|system|>
You are an AWS Risk Analyst assistant. Provide a brief risk analysis.

Context: {context}

Risk Query: {query}

Provide analysis with:
1. Risk Level
2. Main concerns
3. Simple recommendations

Keep response very short (under 150 words).</s>
<|user|>
Analyze the risk and be concise:</s>
<|assistant|>
"""
        
        return prompt
    
    def _call_hf_api(self, prompt: str) -> str:
        """
        Internal method to call HuggingFace API
        Called by circuit breaker wrapper
        """
        if not self.enabled or not self.client:
            raise Exception("HuggingFace client not enabled")
        
        if not self.rate_limiter.can_make_request():
            raise Exception("HuggingFace rate limit exceeded")
        
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                truncate=1024  # Limit for memory
            )
            
            if response:
                # Clean up response
                response = response.strip()
                if "```" in response:
                    # Remove markdown code blocks
                    response = response.replace("```json", "").replace("```", "").strip()
                
                return response
            else:
                raise Exception("Empty response from HuggingFace")
                
        except Exception as e:
            logger.error(f"HuggingFace API call failed: {e}")
            raise
    
    def analyze_risk_fallback(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Fallback risk analysis with circuit breaker protection
        """
        if not self.enabled:
            raise Exception("HuggingFace client not enabled")
        
        start_time = time.time()
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(query, context)
            
            # Call API with circuit breaker protection
            if circuit_breaker:
                response_text = with_hf_circuit(self._call_hf_api)(prompt)
            else:
                response_text = self._call_hf_api(prompt)
            
            # Parse response
            response_time = time.time() - start_time
            
            return {
                "analysis": response_text,
                "model": self.llm_model,
                "response_time": response_time,
                "tokens_estimated": len(prompt) // 4 + len(response_text) // 4,
                "status": "success",
                "source": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback risk analysis failed: {e}")
            return {
                "analysis": f"Error: {str(e)}",
                "model": self.llm_model,
                "response_time": time.time() - start_time,
                "status": "error",
                "error": str(e),
                "source": "fallback"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "enabled": self.enabled,
            "model": self.llm_model,
            "rate_limit": f"{self.rate_limiter.max_requests}/{self.rate_limiter.time_window}s",
            "recent_requests": len(self.rate_limiter.requests),
            "client_available": self.client is not None
        }

def create_huggingface_client() -> Optional[HuggingFaceClient]:
    """Factory function to create HuggingFace client"""
    try:
        return HuggingFaceClient()
    except Exception as e:
        logger.error(f"Failed to create HuggingFace client: {e}")
        return None

