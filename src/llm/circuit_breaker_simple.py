"""
Simple, working circuit breaker for LLM services
Minimal implementation for 1GB RAM
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SimpleCircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if circuit is closed and can execute"""
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                logger.info(f"Circuit {self.name} moved to half-open")
                return True
            return False
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            logger.info(f"Circuit {self.name} reset to closed after success")
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker"""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0
        logger.info(f"Circuit {self.name} manually reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
            "can_execute": self.can_execute()
        }

# Create global circuit breakers
gemini_circuit = SimpleCircuitBreaker("gemini_api", failure_threshold=3, recovery_timeout=30)
hf_circuit = SimpleCircuitBreaker("huggingface_api", failure_threshold=2, recovery_timeout=60)

class CircuitBreakerManager:
    """Manager for circuit breakers"""
    
    circuits = {
        "gemini": gemini_circuit,
        "huggingface": hf_circuit
    }
    
    @classmethod
    def get_circuit_stats(cls) -> Dict[str, Any]:
        """Get all circuit breaker statistics"""
        stats = {}
        for name, circuit in cls.circuits.items():
            stats[name] = circuit.get_stats()
        return stats
    
    @classmethod
    def reset_circuit(cls, service: str):
        """Reset a circuit breaker"""
        if service in cls.circuits:
            cls.circuits[service].reset()
            return True
        return False
    
    @classmethod
    def with_circuit(cls, service: str):
        """Decorator for circuit breaker protection"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                circuit = cls.circuits.get(service)
                if not circuit:
                    return func(*args, **kwargs)
                
                if not circuit.can_execute():
                    raise Exception(f"Circuit {service} is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    circuit.record_success()
                    return result
                except Exception as e:
                    circuit.record_failure()
                    raise
            
            return wrapper
        return decorator

# Global manager instance
circuit_manager = CircuitBreakerManager()

# Convenience functions
def with_gemini_circuit(func):
    return circuit_manager.with_circuit("gemini")(func)

def with_hf_circuit(func):
    return circuit_manager.with_circuit("huggingface")(func)
