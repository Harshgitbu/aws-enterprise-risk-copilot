import faiss
import numpy as np
import pickle
import json
import os
from typing import List, Dict, Any, Optional
import psutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientFAISS:
    """
    FAISS vector store with memory optimization for 1GB RAM
    """
    
    def __init__(self, dimension: int = 384, max_vectors: int = 10000):
        """
        Initialize FAISS index with memory constraints
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            max_vectors: Maximum vectors to store (memory limit)
        """
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.current_vectors = 0
        
        # Initialize empty index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Metadata storage
        self.metadata = []
        
        logger.info(f"FAISS initialized: dim={dimension}, max_vectors={max_vectors}")
        self._log_memory()
    
    def _log_memory(self):
        """Log current memory usage"""
        mem = psutil.virtual_memory()
        logger.debug(f"Memory: {mem.used/1024/1024:.0f}MB/{mem.total/1024/1024:.0f}MB ({mem.percent}%)")
    
    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        Add vectors with memory constraints
        
        Args:
            vectors: numpy array of shape (n, dimension)
            metadatas: list of metadata dicts
        """
        if self.current_vectors + len(vectors) > self.max_vectors:
            raise MemoryError(
                f"Cannot add {len(vectors)} vectors. "
                f"Current: {self.current_vectors}, Max: {self.max_vectors}"
            )
        
        # Convert to float32 for FAISS
        vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self.current_vectors += len(vectors)
        
        logger.info(f"Added {len(vectors)} vectors. Total: {self.current_vectors}")
        self._log_memory()
        return self.current_vectors
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: query embedding
            k: number of results
            
        Returns:
            List of dicts with 'metadata' and 'distance'
        """
        if self.current_vectors == 0:
            return []
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # FAISS search
        distances, indices = self.index.search(query_vector, min(k, self.current_vectors))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # FAISS returns -1 for empty slots
                results.append({
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i]),
                    'score': 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity score
                })
        
        return results
    
    def save(self, path: str):
        """Save index and metadata"""
        # Save FAISS index
        faiss.write_index(self.index, path)
        
        # Save metadata
        meta_path = path.replace('.index', '.meta')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'current_vectors': self.current_vectors
            }, f)
        
        logger.info(f"Saved to {path} ({self.current_vectors} vectors)")
    
    def load(self, path: str):
        """Load index and metadata"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(path)
        self.current_vectors = self.index.ntotal
        
        # Load metadata
        meta_path = path.replace('.index', '.meta')
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.dimension = data['dimension']
        
        logger.info(f"Loaded from {path} ({self.current_vectors} vectors)")
        self._log_memory()
    
    def clear(self):
        """Clear index to free memory"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.current_vectors = 0
        logger.info("Index cleared")
        self._log_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'dimension': self.dimension,
            'current_vectors': self.current_vectors,
            'max_vectors': self.max_vectors,
            'memory_percent': psutil.virtual_memory().percent
        }

# Factory function
def create_vector_store(dimension: int = 384, max_vectors: int = 10000):
    """Create a new vector store instance"""
    return MemoryEfficientFAISS(dimension, max_vectors)

if __name__ == "__main__":
    # Test the implementation
    print("Testing FAISS vector store...")
    
    # Create store
    store = MemoryEfficientFAISS(dimension=384, max_vectors=1000)
    
    # Create test data
    test_vectors = np.random.randn(100, 384).astype('float32')
    test_metadata = [{"id": i, "text": f"Test document {i}"} for i in range(100)]
    
    # Add vectors
    store.add_vectors(test_vectors, test_metadata)
    
    # Test search
    query = np.random.randn(384).astype('float32')
    results = store.search(query, k=3)
    
    print(f"\nSearch results ({len(results)} found):")
    for r in results[:3]:
        print(f"  - {r['metadata']['text']} (score: {r['score']:.3f})")
    
    print("\n✅ FAISS vector store test complete!")
    print(f"Stats: {store.get_stats()}")