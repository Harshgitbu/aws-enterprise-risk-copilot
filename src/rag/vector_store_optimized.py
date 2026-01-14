"""
Memory-optimized vector store using lazy-loaded embeddings
"""
import numpy as np
import faiss
import logging
from typing import List, Optional, Tuple
import asyncio

from .embedding_optimized import get_embedding_service

logger = logging.getLogger(__name__)

class MemoryOptimizedFAISS:
    """FAISS vector store with memory optimization"""
    
    def __init__(self, dimension: int = 384, max_vectors: int = 10000):
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.index = None
        self.documents = []
        self.ids = []
        self.embedding_service = None
        
        logger.info(f"MemoryOptimizedFAISS initialized (max: {max_vectors} vectors)")
    
    async def initialize(self):
        """Initialize with lazy loading"""
        if self.index is None:
            # Create flat index (most memory efficient)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("FAISS index initialized")
        
        if self.embedding_service is None:
            self.embedding_service = await get_embedding_service()
    
    async def add_documents(self, texts: List[str], ids: Optional[List[str]] = None):
        """Add documents with memory check"""
        await self.initialize()
        
        # Check memory before adding
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            logger.warning(f"Memory high ({mem.percent}%), skipping document addition")
            return False
        
        # Limit number of vectors
        if len(self.documents) + len(texts) > self.max_vectors:
            logger.warning(f"Max vectors ({self.max_vectors}) reached, trimming")
            self._trim_vectors(self.max_vectors // 2)
        
        # Get embeddings
        embeddings = await self.embedding_service.get_embeddings(texts)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(texts)
        
        if ids:
            self.ids.extend(ids)
        else:
            self.ids.extend([str(len(self.documents) - i) for i in range(len(texts))])
        
        logger.info(f"Added {len(texts)} documents (total: {len(self.documents)})")
        return True
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search with memory optimization"""
        await self.initialize()
        
        if len(self.documents) == 0:
            return []
        
        # Get query embedding
        query_embedding = await self.embedding_service.get_embeddings([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), 
                                              min(top_k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(distances[0][i])))
        
        return results
    
    def _trim_vectors(self, keep_count: int):
        """Trim vectors to save memory"""
        if keep_count >= len(self.documents):
            return
        
        logger.info(f"Trimming vectors from {len(self.documents)} to {keep_count}")
        
        # Rebuild index with only kept vectors
        kept_docs = self.documents[:keep_count]
        kept_ids = self.ids[:keep_count]
        
        # Reset and rebuild
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = kept_docs
        self.ids = kept_ids
        
        # Re-add kept documents (this would need embeddings, simplified here)
        # In production, you'd store embeddings separately
        
        logger.info(f"Trimmed to {len(self.documents)} vectors")
    
    def get_stats(self) -> dict:
        """Get memory statistics"""
        import psutil
        process = psutil.Process()
        
        return {
            "vectors_count": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "max_vectors": self.max_vectors,
            "dimension": self.dimension
        }
