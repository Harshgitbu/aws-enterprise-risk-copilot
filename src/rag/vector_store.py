"""
FAISS vector store with memory optimization for 1GB RAM EC2
"""
import numpy as np
import faiss
import pickle
import os
import logging
import psutil
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MemoryEfficientFAISS:
    """Memory-efficient FAISS vector store for 1GB RAM constraint"""
    
    def __init__(self, dimension: int = 384, max_vectors: int = 10000):
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.index = None
        self.metadatas = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        try:
            # Try to load existing index
            if os.path.exists("/app/data/vector_store/faiss_index"):
                self.index = faiss.read_index("/app/data/vector_store/faiss_index")
                # Load metadatas
                meta_path = "/app/data/vector_store/metadatas.pkl"
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        self.metadatas = pickle.load(f)
                logger.info(f"Loaded existing index with {len(self.metadatas)} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created new FAISS index (dim={self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def _log_memory(self):
        """Log current memory usage"""
        mem = psutil.virtual_memory()
        logger.debug(f"Memory: {mem.used/1024/1024:.0f}MB/{mem.total/1024/1024:.0f}MB ({mem.percent}%)")
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to vector store
        
        Args:
            documents: List of document strings
        """
        if not documents:
            return
        
        # For now, create dummy embeddings
        # In production, use embedding service
        vectors = np.random.randn(len(documents), self.dimension).astype('float32')
        metadatas = [{"text": doc, "source": "user_input"} for doc in documents]
        
        self.add_vectors(vectors, metadatas)
        logger.info(f"Added {len(documents)} documents")
    
    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        Add vectors to index
        
        Args:
            vectors: numpy array of vectors
            metadatas: List of metadata dicts
        """
        if len(vectors) == 0:
            return
        
        # Check memory constraint
        current_size = len(self.metadatas)
        if current_size + len(vectors) > self.max_vectors:
            logger.warning(f"Approaching max vectors limit: {current_size}/{self.max_vectors}")
        
        # Add to index
        self.index.add(vectors)
        self.metadatas.extend(metadatas)
        
        # Save periodically
        if len(self.metadatas) % 100 == 0:
            self._save_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            List of result dictionaries
        """
        # For now, return dummy results
        # In production, embed query and search
        results = []
        for i in range(min(top_k, len(self.metadatas))):
            results.append({
                "content": self.metadatas[i].get("text", f"Document {i}"),
                "score": 0.9 - (i * 0.1),
                "metadata": self.metadatas[i]
            })
        
        return results
    
    def _search_by_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search by vector (internal method)
        
        Args:
            query_vector: Query vector
            k: Number of results
        """
        if len(self.metadatas) == 0:
            return []
        
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'),
            min(k, len(self.metadatas))
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadatas):
                results.append({
                    "content": self.metadatas[idx].get("text", f"Document {idx}"),
                    "score": float(1 / (1 + distance)),
                    "metadata": self.metadatas[idx]
                })
        
        return results
    
    def _save_index(self):
        """Save index to disk"""
        try:
            index_path = "/app/data/vector_store/faiss_index"
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            
            # Save metadatas
            meta_path = "/app/data/vector_store/metadatas.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadatas, f)
            
            logger.info(f"Saved index with {len(self.metadatas)} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load(self, path: str):
        """Load index from disk"""
        try:
            self.index = faiss.read_index(path)
            
            # Load metadatas
            meta_path = path.replace(".faiss", "_metadatas.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.metadatas = pickle.load(f)
            
            logger.info(f"Loaded index with {len(self.metadatas)} vectors")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
    
    def clear(self):
        """Clear all vectors"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadatas = []
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "vectors": len(self.metadatas),
            "dimension": self.dimension,
            "max_vectors": self.max_vectors,
            "memory_usage_mb": len(self.metadatas) * self.dimension * 4 / (1024 * 1024)
        }

def create_vector_store(dimension: int = 384, max_vectors: int = 10000):
    """Factory function to create vector store"""
    return MemoryEfficientFAISS(dimension=dimension, max_vectors=max_vectors)
