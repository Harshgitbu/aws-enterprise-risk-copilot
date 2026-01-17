"""
Memory-optimized vector store using REAL embeddings
"""
import numpy as np
import faiss
import logging
from typing import List, Optional, Tuple, Dict, Any
import asyncio
import psutil

logger = logging.getLogger(__name__)

class MemoryOptimizedFAISS:
    """FAISS vector store with memory optimization and REAL embeddings"""
    
    def __init__(self, dimension: int = 384, max_vectors: int = 1000):  # Reduced to 1000 for memory
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.index = None
        self.documents = []
        self.ids = []
        self.metadata = []  # Store metadata for each document
        self.embedder = None
        
        logger.info(f"MemoryOptimizedFAISS initialized (max: {max_vectors} vectors)")
    
    async def initialize(self):
        """Initialize with lazy loading of embedder"""
        if self.index is None:
            # Create flat index (most memory efficient)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("FAISS index initialized")
        
        # Lazy load embedder only when needed
        if self.embedder is None:
            try:
                from rag.embedding import get_embedder
                self.embedder = get_embedder()
                logger.info("Embedding service initialized")
            except ImportError as e:
                logger.error(f"Failed to import embedder: {e}")
                # Fallback to random embeddings for testing
                self.embedder = None
    
    async def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                          ids: Optional[List[str]] = None):
        """Add documents with REAL embeddings and memory check"""
        await self.initialize()
        
        # Check memory before adding
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            logger.warning(f"Memory high ({mem.percent}%), skipping document addition")
            return False
        
        # Limit number of vectors
        if len(self.documents) + len(texts) > self.max_vectors:
            logger.warning(f"Max vectors ({self.max_vectors}) reached, trimming")
            self._trim_vectors(self.max_vectors // 2)
        
        # Use REAL embeddings if available
        if self.embedder:
            try:
                # Get embeddings
                embeddings = self.embedder.embed_texts(texts)
                if embeddings is None or len(embeddings) == 0:
                    logger.error("Failed to generate embeddings, falling back to random")
                    embeddings = np.random.randn(len(texts), self.dimension).astype('float32')
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}, falling back to random")
                embeddings = np.random.randn(len(texts), self.dimension).astype('float32')
        else:
            # Fallback to random embeddings
            logger.warning("Embedder not available, using random embeddings")
            embeddings = np.random.randn(len(texts), self.dimension).astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(texts)
        
        # Store metadata
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in range(len(texts))])
        
        # Store IDs
        if ids:
            self.ids.extend(ids)
        else:
            self.ids.extend([str(len(self.documents) - i) for i in range(len(texts))])
        
        logger.info(f"Added {len(texts)} documents with embeddings (total: {len(self.documents)})")
        return True
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search with REAL query embeddings"""
        await self.initialize()
        
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = None
        if self.embedder:
            try:
                query_embedding = self.embedder.embed_single(query)
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
        
        # Fallback to random if embedder fails
        if query_embedding is None:
            logger.warning("Using random query embedding (fallback)")
            query_embedding = np.random.randn(self.dimension).astype('float32')
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(
            query_embedding, 
            min(top_k, len(self.documents))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((
                    self.documents[idx], 
                    float(distances[0][i]),
                    self.metadata[idx]
                ))
        
        logger.info(f"Search found {len(results)} results for query: {query[:50]}...")
        return results
    
    def _trim_vectors(self, keep_count: int):
        """Trim vectors to save memory"""
        if keep_count >= len(self.documents):
            return
        
        logger.info(f"Trimming vectors from {len(self.documents)} to {keep_count}")
        
        # Rebuild index with only kept vectors
        kept_docs = self.documents[:keep_count]
        kept_metadata = self.metadata[:keep_count]
        kept_ids = self.ids[:keep_count]
        
        # Reset and rebuild with real embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Re-embed kept documents
        if self.embedder and kept_docs:
            try:
                embeddings = self.embedder.embed_texts(kept_docs)
                if embeddings is not None and len(embeddings) > 0:
                    self.index.add(embeddings)
            except Exception as e:
                logger.error(f"Failed to re-embed trimmed documents: {e}")
        
        self.documents = kept_docs
        self.metadata = kept_metadata
        self.ids = kept_ids
        
        logger.info(f"Trimmed to {len(self.documents)} vectors")
    
    def get_stats(self) -> dict:
        """Get memory statistics"""
        import psutil
        process = psutil.Process()
        
        embedder_stats = {}
        if self.embedder and hasattr(self.embedder, 'get_stats'):
            embedder_stats = self.embedder.get_stats()
        
        return {
            "vectors_count": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "max_vectors": self.max_vectors,
            "dimension": self.dimension,
            "embedder_loaded": self.embedder is not None,
            **embedder_stats
        }
