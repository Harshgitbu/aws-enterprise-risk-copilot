"""
Memory-efficient embedding service for AWS Risk Copilot
Uses sentence-transformers with lazy loading and memory monitoring
Model: all-MiniLM-L6-v2 (80MB, optimized for 1GB RAM)
"""

import numpy as np
import logging
import psutil
from typing import List, Optional
import gc

logger = logging.getLogger(__name__)

class MemoryOptimizedEmbedder:
    """
    Embedding service with memory monitoring and lazy loading
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2
        self.is_loaded = False
        
        logger.info(f"Embedder initialized (will lazy load: {model_name})")
    
    def _check_memory(self) -> bool:
        """Check if we have enough memory to load model"""
        mem = psutil.virtual_memory()
        
        if mem.percent > 85:
            logger.warning(f"Memory high ({mem.percent}%), skipping model load")
            return False
        
        # Check available memory (should be > 100MB for model)
        available_mb = mem.available / (1024 * 1024)
        if available_mb < 100:
            logger.warning(f"Available memory low ({available_mb:.1f}MB), skipping model load")
            return False
        
        return True
    
    def load_model(self) -> bool:
        """Lazy load the embedding model"""
        if self.is_loaded:
            return True
        
        if not self._check_memory():
            return False
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Set model to eval mode and disable gradients for inference
            self.model.eval()
            
            # Try to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            self.is_loaded = True
            logger.info("✅ Embedding model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import sentence-transformers: {e}")
            logger.info("Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def unload_model(self):
        """Unload model to save memory"""
        if self.model:
            del self.model
            self.model = None
        
        # Force garbage collection
        gc.collect()
        
        self.is_loaded = False
        logger.info("Embedding model unloaded")
    
    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Embed a list of texts with memory monitoring
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings or None if failed
        """
        if not self.load_model():
            logger.error("Cannot embed: model not loaded")
            return None
        
        if not texts:
            return np.array([])
        
        try:
            # Truncate long texts for memory efficiency
            truncated_texts = []
            for text in texts:
                if len(text) > 512:
                    truncated_texts.append(text[:512])
                else:
                    truncated_texts.append(text)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(truncated_texts)} texts")
            embeddings = self.model.encode(
                truncated_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Clear memory
            gc.collect()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """Embed single text"""
        result = self.embed_texts([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None
    
    def get_stats(self) -> dict:
        """Get embedder statistics"""
        mem = psutil.virtual_memory()
        
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "dimension": self.dimension,
            "memory_percent": mem.percent,
            "available_mb": mem.available / (1024 * 1024),
            "model_size_mb": 80  # Approximate size of all-MiniLM-L6-v2
        }

# Global instance for singleton pattern
_embedder_instance = None

def get_embedder() -> MemoryOptimizedEmbedder:
    """Get singleton embedder instance"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MemoryOptimizedEmbedder()
    return _embedder_instance
