"""
Circuit breaker wrapper that works with either implementation
"""
import logging

logger = logging.getLogger(__name__)

def with_gemini_circuit(func):
    """Circuit breaker wrapper for Gemini - tries multiple implementations"""
    try:
        # Try simple implementation first
        from llm.circuit_breaker_simple import with_gemini_circuit as simple_wrapper
        logger.debug("Using simple circuit breaker for Gemini")
        return simple_wrapper(func)
    except ImportError:
        try:
            # Try original implementation
            from llm.circuit_breaker_config import circuit_breaker
            logger.debug("Using config circuit breaker for Gemini")
            return circuit_breaker.with_gemini_circuit(func)
        except ImportError:
            # No circuit breaker available
            logger.debug("No circuit breaker available for Gemini")
            return func

def with_hf_circuit(func):
    """Circuit breaker wrapper for HuggingFace - tries multiple implementations"""
    try:
        # Try simple implementation first
        from llm.circuit_breaker_simple import with_hf_circuit as simple_wrapper
        logger.debug("Using simple circuit breaker for HF")
        return simple_wrapper(func)
    except ImportError:
        try:
            # Try original implementation
            from llm.circuit_breaker_config import circuit_breaker
            logger.debug("Using config circuit breaker for HF")
            return circuit_breaker.with_hf_circuit(func)
        except ImportError:
            # No circuit breaker available
            logger.debug("No circuit breaker available for HF")
            return func
