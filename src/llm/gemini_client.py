"""
Memory-optimized Gemini API client for AWS Risk Copilot
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
        """Check if request can be made within rate limits"""
        now = time.time()
        # Clean old requests (older than time_window)
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
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 0.1)  # Small buffer
            # Clean up after waiting
            self.requests = [req_time for req_time in self.requests 
                           if time.time() - req_time < self.time_window]
        self.record_request()


class GeminiResponse:
    """Container for Gemini response with usage stats"""
    def __init__(self, text: str, tokens_used: int = 0, response_time: float = 0.0):
        self.text = text
        self.tokens_used = tokens_used
        self.response_time = response_time
        
    def __str__(self):
        return f"GeminiResponse({len(self.text)} chars, {self.tokens_used} tokens, {self.response_time:.2f}s)"


class GeminiClient:
    """Memory-efficient Gemini API client with streaming and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini client with memory optimization
        
        Args:
            api_key: Google API key (defaults to config or env var)
            model: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or CONFIG.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env file")
        
        self.model_name = model or CONFIG.GEMINI_MODEL
        
        # Configure Gemini with minimal settings
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Initialize rate limiter with YOUR free tier limits (10 RPM -> 7 for safety)
        self.rate_limiter = RateLimiter(
            max_requests=CONFIG.GEMINI_RATE_LIMIT,  # Using 7 from config
            time_window=60
        )
        
        # Memory-efficient usage tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        logger.info(f"✅ GeminiClient initialized with model: {self.model_name}")
        logger.info(f"📊 Free Tier Limits: {CONFIG.GEMINI_RATE_LIMIT} RPM (Your quota: 10 RPM)")
    
    def generate(self, 
                prompt: str, 
                context: Optional[str] = None,
                stream: bool = False,  # Streaming disabled for memory efficiency
                max_tokens: int = 1024) -> GeminiResponse:
        """
        Generate response with rate limiting and error handling
        OPTIMIZED FOR: 1GB RAM, Free Tier (10 RPM)
        """
        start_time = time.time()
        
        try:
            # Apply rate limiting (YOUR free tier: 10 RPM)
            self.rate_limiter.wait_if_needed()
            
            # Prepare final prompt with memory optimization
            final_prompt = prompt
            if context:
                # Truncate context if too long for 1GB RAM
                max_context_len = 1000
                if len(context) > max_context_len:
                    context = context[:max_context_len] + "..."
                final_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            
            logger.info(f"📤 Sending request to Gemini (approx {len(final_prompt)} chars)")
            
            # Generate response with memory-efficient settings
            generation_config = {
                "temperature": 0.1,  # Low for deterministic responses
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": min(max_tokens, 1024),  # Cap at 1024 for 1GB RAM
            }
            
            response = self.model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
            
            # Extract text
            response_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    response_text += part.text
            
            response_time = time.time() - start_time
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(response_text) // 4 + len(final_prompt) // 4
            
            # Update usage stats
            self.total_requests += 1
            self.total_tokens += estimated_tokens
            
            logger.info(f"📥 Received response ({len(response_text)} chars, {estimated_tokens} tokens, {response_time:.2f}s)")
            
            return GeminiResponse(
                text=response_text,
                tokens_used=estimated_tokens,
                response_time=response_time
            )
            
        except exceptions.ResourceExhausted as e:
            logger.error(f"Rate limit exceeded: {e}")
            # Wait and retry once (longer wait for free tier)
            time.sleep(10)  # 10 seconds for free tier
            return self.generate(prompt, context, stream, max_tokens)
            
        except exceptions.GoogleAPIError as e:
            logger.error(f"Google API error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def analyze_risk(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze risk with structured JSON output
        Memory-optimized for 1GB RAM
        """
        risk_template = """You are an AWS risk intelligence analyst. Analyze the following risk query:

CONTEXT: {context}

QUERY: {query}

Provide analysis in this JSON format:
{{
    "risk_summary": "Brief summary",
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "confidence": 0.0-1.0,
    "aws_services_affected": ["service1", "service2"],
    "recommendations": ["rec1", "rec2"],
    "compliance_standards": ["standard1", "standard2"]
}}"""
        
        prompt = risk_template.format(context=context, query=query)
        response = self.generate(prompt, max_tokens=512)  # Limited for memory
        
        try:
            # Try to parse JSON from response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response.text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from response")
        
        # Fallback to text response
        return {
            "risk_analysis": response.text,
            "raw_response": response.text
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics with free tier awareness"""
        uptime = time.time() - self.start_time
        avg_tokens_per_request = self.total_tokens / max(1, self.total_requests)
        
        # Calculate remaining requests based on YOUR free tier (10 RPM)
        requests_per_day = 10 * 60 * 24  # 10 RPM * 60 minutes * 24 hours
        remaining_requests = max(0, requests_per_day - self.total_requests)
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": avg_tokens_per_request,
            "uptime_hours": uptime / 3600,
            "remaining_requests_today": remaining_requests,
            "free_tier_limit": "10 RPM (your quota)",
            "model": self.model_name
        }
    
    def test_connection(self) -> bool:
        """Test if Gemini API is accessible"""
        try:
            response = self.generate("Hello, respond with 'OK' if you can hear me.", max_tokens=10)
            return "OK" in response.text.upper()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def create_gemini_client() -> GeminiClient:
    """Factory function to create Gemini client"""
    return GeminiClient()