"""
Circuit breaker configuration for LLM services - FIXED VERSION
Optimized for 1GB RAM with minimal overhead
"""
import time
import logging
from circuitbreaker import circuit, CircuitBreaker

logger = logging.getLogger(__name__)

# Create circuit breaker instances with proper configuration
gemini_circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    name="gemini_api"
)

hf_circuit = CircuitBreaker(
    failure_threshold=2,
    recovery_timeout=60,
    name="huggingface_api"
)

class LLMCircuitBreaker:
    """
    Memory-efficient circuit breaker for LLM APIs
    Settings optimized for free tier APIs
    """
    
    # Circuit instances
    GEMINI_CIRCUIT = gemini_circuit
    HF_CIRCUIT = hf_circuit
    
    # Monitoring
    _circuit_states = {
        "gemini": {"state": "closed", "failures": 0, "last_failure": None},
        "huggingface": {"state": "closed", "failures": 0, "last_failure": None}
    }
    
    @classmethod
    def with_gemini_circuit(cls, func):
        """Decorator for Gemini API calls with circuit breaker"""
        @circuit(failure_threshold=3, recovery_timeout=30, name="gemini_api")
        def wrapper(*args, **kwargs):
            try:
                logger.debug("Gemini circuit: attempting call")
                result = func(*args, **kwargs)
                cls._update_state("gemini", "closed", success=True)
                return result
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                cls._update_state("gemini", "closed", success=False)
                raise
        
        return wrapper
    
    @classmethod
    def with_hf_circuit(cls, func):
        """Decorator for HuggingFace API calls with circuit breaker"""
        @circuit(failure_threshold=2, recovery_timeout=60, name="huggingface_api")
        def wrapper(*args, **kwargs):
            try:
                logger.debug("HF circuit: attempting call")
                result = func(*args, **kwargs)
                cls._update_state("huggingface", "closed", success=True)
                return result
            except Exception as e:
                logger.error(f"HF API error: {e}")
                cls._update_state("huggingface", "closed", success=False)
                raise
        
        return wrapper
    
    @classmethod
    def _update_state(cls, service: str, state: str, success: bool):
        """Update circuit state for monitoring"""
        if service not in cls._circuit_states:
            return
        
        if not success:
            cls._circuit_states[service]["failures"] += 1
            cls._circuit_states[service]["last_failure"] = time.time()
        
        cls._circuit_states[service]["state"] = state
        
        # Reset failures on success
        if success and state == "closed":
            cls._circuit_states[service]["failures"] = 0
    
    @classmethod
    def get_circuit_stats(cls):
        """Get circuit breaker statistics"""
        stats = {}
        
        for service in ["gemini", "huggingface"]:
            circuit_inst = gemini_circuit if service == "gemini" else hf_circuit
            state_data = cls._circuit_states.get(service, {})
            
            stats[service] = {
                "state": state_data.get("state", "unknown"),
                "failures": state_data.get("failures", 0),
                "last_failure": state_data.get("last_failure"),
                "circuit_state": circuit_inst.current_state,
                "fail_counter": circuit_inst.fail_counter,
                "open_until": circuit_inst.open_until if hasattr(circuit_inst, 'open_until') else None
            }
        
        return stats
    
    @classmethod
    def reset_circuit(cls, service: str):
        """Reset a circuit breaker"""
        if service == "gemini":
            gemini_circuit.close()
            gemini_circuit.fail_counter = 0
        elif service == "huggingface":
            hf_circuit.close()
            hf_circuit.fail_counter = 0
        
        if service in cls._circuit_states:
            cls._circuit_states[service] = {"state": "closed", "failures": 0, "last_failure": None}
        
        logger.info(f"Circuit reset for {service}")

# Global instance
circuit_breaker = LLMCircuitBreaker()

