"""
Optimized embedding module with lazy loading
"""

import os
from typing import List, Optional
import numpy as np

class LazySentenceTransformer:
    """Lazy loader for sentence transformers to save memory"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            # Import ONLY when needed
            from sentence_transformers import SentenceTransformer
            print(f"Loading model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Size: {self._model.get_sentence_embedding_dimension()} dimensions")
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

# Global instance with lazy loading
_embedder = None

def get_embedder():
    """Get or create the embedder instance (lazy)"""
    global _embedder
    if _embedder is None:
        _embedder = LazySentenceTransformer()
    return _embedder

def encode_texts(texts: List[str]) -> np.ndarray:
    """Encode texts with lazy loading"""
    embedder = get_embedder()
    return embedder.encode(texts)

def get_embedding_dimension() -> int:
    """Get embedding dimension"""
    embedder = get_embedder()
    return embedder.get_embedding_dimension()
