"""
Memory optimization utilities for 1GB RAM constraint
"""
import gc
import psutil
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Optimize memory usage for constrained environments"""
    
    def __init__(self, target_mb: int = 300, warning_threshold: float = 0.7):
        self.target_mb = target_mb
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
        logger.info(f"MemoryOptimizer initialized (target: {target_mb}MB)")
    
    def get_current_usage(self) -> dict:
        """Get current memory usage"""
        mem_info = self.process.memory_info()
        total_mem = psutil.virtual_memory().total
        
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "percent": (mem_info.rss / total_mem) * 100,
            "available_mb": psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is above threshold"""
        usage = self.get_current_usage()
        return usage["rss_mb"] > self.target_mb * self.warning_threshold
    
    def optimize_memory(self) -> dict:
        """
        Perform memory optimization
        
        Returns:
            dict with optimization results
        """
        before = self.get_current_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear caches if possible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        after = self.get_current_usage()
        
        result = {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "freed_mb": before["rss_mb"] - after["rss_mb"],
            "collected_objects": collected,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Memory optimization freed {result['freed_mb']:.1f}MB")
        return result
    
    async def periodic_optimization(self, interval_seconds: int = 60):
        """Periodically optimize memory"""
        while True:
            if self.is_memory_high():
                logger.warning(f"Memory high, performing optimization")
                result = self.optimize_memory()
                
                if result["freed_mb"] < 10:  # Didn't free much
                    logger.warning(f"Optimization only freed {result['freed_mb']:.1f}MB")
            
            await asyncio.sleep(interval_seconds)

# Global optimizer instance
memory_optimizer = MemoryOptimizer(target_mb=300)

async def start_memory_optimizer():
    """Start periodic memory optimization"""
    asyncio.create_task(memory_optimizer.periodic_optimization())
    logger.info("Memory optimizer started")

