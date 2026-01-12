"""
Memory-optimized HuggingFace client for AWS Risk Copilot
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
        """Check if request can be made within rate limits"""
        now = time.time()
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """Record a new request"""
        self.requests.append(time.time())
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded"""
        if not self.can_make_request():
            oldest_request = min(self.requests)
            wait_time = self.time_window - (time.time() - oldest_request)
            if wait_time > 0:
                logger.info(f"HF Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)
            self.requests = [req_time for req_time in self.requests 
                           if time.time() - req_time < self.time_window]
        self.record_request()


class HFResponse:
    """Container for HuggingFace response with usage stats"""
    def __init__(self, text: str, tokens_used: int = 0, response_time: float = 0.0):
        self.text = text
        self.tokens_used = tokens_used
        self.response_time = response_time
        
    def __str__(self):
        return f"HFResponse({len(self.text)} chars, {self.tokens_used} tokens, {self.response_time:.2f}s)"


class HuggingFaceClient:
    """Memory-efficient HuggingFace client using huggingface_hub"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace client with huggingface_hub
        """
        if not HF_HUB_AVAILABLE:
            logger.error("huggingface_hub library not available")
            self.enabled = False
            return
        
        self.token = token or CONFIG.HF_TOKEN or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            logger.warning("HuggingFace token not found. Features disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        try:
            # Initialize clients
            self.api = HfApi(token=self.token)
            self.inference_client = InferenceClient(
                model=CONFIG.HF_LLM_MODEL,
                token=self.token
            )
            
            # Rate limiter
            self.rate_limiter = HFRateLimiter(max_requests=25, time_window=300)
            
            # Usage tracking
            self.total_requests = 0
            self.total_tokens = 0
            self.start_time = time.time()
            
            logger.info("✅ HuggingFaceClient initialized with huggingface_hub")
            logger.info(f"📊 Using models: {CONFIG.HF_LLM_MODEL} (LLM), {CONFIG.HF_EMBEDDING_MODEL} (Embeddings)")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            self.enabled = False
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings using direct API call (InferenceClient has issues with embeddings)"""
        if not self.enabled or not texts:
            return None
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Small batch for memory
            if len(texts) > 2:
                texts = texts[:2]
            
            logger.debug(f"Getting embeddings for {len(texts)} texts")
            
            # Use direct API call since InferenceClient has issues
            import requests
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{CONFIG.HF_EMBEDDING_MODEL}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": texts},
                timeout=30
            )
            
            if response.status_code == 200:
                self.total_requests += 1
                estimated_tokens = sum(len(text) // 4 for text in texts)
                self.total_tokens += estimated_tokens
                
                embeddings = response.json()
                logger.info(f"✅ Got embeddings ({len(embeddings)} vectors)")
                return embeddings
            else:
                logger.error(f"Embedding API error: {response.status_code}")
                return None
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def generate(self, prompt: str, max_tokens: int = 128) -> Optional[HFResponse]:
        """Generate text using InferenceClient"""
        if not self.enabled:
            return None
        
        start_time = time.time()
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Truncate prompt
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            
            logger.debug(f"Generating text with {len(prompt)} chars prompt")
            
            response = self.inference_client.text_generation(
                prompt=prompt,
                max_new_tokens=min(max_tokens, 128),
                temperature=0.1
            )
            
            response_time = time.time() - start_time
            
            # Get text from response
            generated_text = str(response)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Estimate tokens
            estimated_tokens = (len(prompt) + len(generated_text)) // 4
            
            self.total_requests += 1
            self.total_tokens += estimated_tokens
            
            logger.info(f"✅ Generated {len(generated_text)} chars, {estimated_tokens} tokens")
            
            return HFResponse(
                text=generated_text,
                tokens_used=estimated_tokens,
                response_time=response_time
            )
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None
    
    def analyze_risk_fallback(self, query: str, context: str = "") -> Dict[str, Any]:
        """Fallback risk analysis"""
        risk_prompt = f"Analyze AWS risk: {query}. Context: {context[:150] if context else 'No context'}."
        
        response = self.generate(risk_prompt, max_tokens=100)
        
        if response:
            return {
                "risk_analysis": response.text,
                "source": "huggingface_fallback",
                "model": CONFIG.HF_LLM_MODEL,
                "tokens_used": response.tokens_used
            }
        else:
            return {
                "error": "HuggingFace fallback failed",
                "query": query[:100]
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        uptime = time.time() - self.start_time
        
        monthly_token_limit = 30000
        remaining_tokens = max(0, monthly_token_limit - self.total_tokens)
        
        requests_last_5min = len([r for r in self.rate_limiter.requests 
                                 if time.time() - r < 300])
        
        return {
            "enabled": self.enabled,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "remaining_tokens_month": remaining_tokens,
            "token_percentage_used": round((self.total_tokens / monthly_token_limit * 100), 2),
            "requests_last_5min": requests_last_5min,
            "rate_limit_5min": self.rate_limiter.max_requests,
            "uptime_hours": round(uptime / 3600, 2),
            "embedding_model": CONFIG.HF_EMBEDDING_MODEL,
            "llm_model": CONFIG.HF_LLM_MODEL
        }
    
    def test_connection(self) -> bool:
        """Test connection"""
        if not self.enabled:
            return False
        
        try:
            # Try to list models (limit 1 for speed)
            models = list(self.api.list_models(limit=1))
            return len(models) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def create_huggingface_client() -> HuggingFaceClient:
    """Factory function"""
    return HuggingFaceClient()