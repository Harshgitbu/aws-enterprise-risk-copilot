"""
LLM API configuration with rate limiting and quotas
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
    """
    
    # Google Gemini API (Free Tier: 60 requests/minute)
    GEMINI_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = "gemini-1.5-flash"  # Fast, cost-effective
    GEMINI_MAX_TOKENS: int = 8192
    GEMINI_RATE_LIMIT: int = 60  # requests per minute
    GEMINI_REQUEST_TIMEOUT: int = 30  # seconds
    
    # Hugging Face Inference API (Free Tier: 30k tokens/month)
    HF_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.1"  # Efficient 7B model
    HF_RATE_LIMIT: int = 10  # requests per minute (free tier)
    HF_REQUEST_TIMEOUT: int = 60  # seconds
    
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
        
        logger.info("API configuration validated successfully")
        logger.info(f"Gemini model: {cls.GEMINI_MODEL}")
        logger.info(f"HF Embedding model: {cls.HF_EMBEDDING_MODEL}")
        logger.info(f"HF LLM model: {cls.HF_LLM_MODEL}")
        
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
            return {"requests_per_minute": 10, "timeout": 30}

# Validate on import
config_valid = APIConfig.validate_config()

if __name__ == "__main__":
    print("=== LLM API Configuration ===")
    print(f"Gemini API Key configured: {'Yes' if APIConfig.GEMINI_API_KEY else 'No'}")
    print(f"HF Token configured: {'Yes' if APIConfig.HF_TOKEN else 'No'}")
    print(f"Config valid: {config_valid}")
