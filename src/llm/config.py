"""
LLM API configuration with rate limiting and quotas
OPTIMIZED FOR: AWS t3.micro 1GB RAM, Google Free Tier (10 RPM), $0/month
"""
import os
from typing import Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class APIConfig:
    """
    Configuration for external LLM APIs with free tier limits
    YOUR ACTUAL FREE TIER: gemini-2.5-flash-lite (10 RPM, 250K TPM)
    """
    
    # Google Gemini API (YOUR FREE TIER: 10 requests/minute)
    GEMINI_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"  # Your actual free tier model
    GEMINI_MAX_TOKENS: int = 2048  # Reduced for 1GB RAM efficiency
    GEMINI_RATE_LIMIT: int = 8  # Conservative: 8 instead of 10 RPM for safety
    GEMINI_REQUEST_TIMEOUT: int = 30  # seconds
    
    # Hugging Face Inference API (Free Tier: 30k tokens/month)
    HF_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_LLM_MODEL: str = "microsoft/phi-2"  # Lighter than Mistral-7B (2.7B vs 7B params)
    HF_RATE_LIMIT: int = 5  # requests per minute (free tier conservative)
    HF_REQUEST_TIMEOUT: int = 60  # seconds
    
    # Memory optimization for 1GB RAM
    MAX_CONTEXT_LENGTH: int = 1500  # chars
    MAX_RESPONSE_LENGTH: int = 800  # chars
    CACHE_MAX_ITEMS: int = 20  # Small cache for 1GB RAM
    
    # Cost tracking (Free Tier limits)
    MAX_MONTHLY_COST: float = 0.0  # $0 - stay within free tier
    TOKEN_TRACKING: bool = True
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate API configuration"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GOOGLE_API_KEY not found in environment variables")
        
        if not cls.HF_TOKEN:
            errors.append("HUGGINGFACE_TOKEN not found in environment variables")
        
        if errors:
            logger.warning(f"API configuration issues: {errors}")
            logger.warning("Some features may not work without API keys")
            return False
        
        logger.info("✅ API configuration validated successfully")
        logger.info(f"   Gemini model: {cls.GEMINI_MODEL} (Your free tier: 10 RPM)")
        logger.info(f"   HF Embedding model: {cls.HF_EMBEDDING_MODEL}")
        logger.info(f"   HF LLM model: {cls.HF_LLM_MODEL} (lightweight fallback)")
        logger.info(f"   Memory optimized for: 1GB RAM, $0/month")
        
        return True
    
    @classmethod
    def get_headers(cls, provider: str) -> dict:
        """Get headers for API requests"""
        if provider == "gemini":
            return {}
        elif provider == "huggingface":
            return {"Authorization": f"Bearer {cls.HF_TOKEN}"}
        else:
            return {}
    
    @classmethod
    def get_rate_limit(cls, provider: str) -> dict:
        """Get rate limiting configuration"""
        if provider == "gemini":
            return {
                "requests_per_minute": cls.GEMINI_RATE_LIMIT,
                "timeout": cls.GEMINI_REQUEST_TIMEOUT
            }
        elif provider == "huggingface":
            return {
                "requests_per_minute": cls.HF_RATE_LIMIT,
                "timeout": cls.HF_REQUEST_TIMEOUT
            }
        else:
            return {"requests_per_minute": 5, "timeout": 30}

# Validate on import
config_valid = APIConfig.validate_config()

if __name__ == "__main__":
    print("=== AWS RISK COPILOT - FREE TIER CONFIG ===")
    print(f"EC2: t3.micro (1GB RAM)")
    print(f"Gemini: {APIConfig.GEMINI_MODEL} (Your free tier: 10 RPM)")
    print(f"Gemini API: {'✅ Ready' if APIConfig.GEMINI_API_KEY else '❌ Missing'}")
    print(f"HF Token: {'✅ Ready' if APIConfig.HF_TOKEN else '❌ Missing'}")
    print(f"Config valid: {'✅ Yes' if config_valid else '⚠️ Partial'}")