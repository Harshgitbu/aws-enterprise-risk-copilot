"""
Memory-optimized RAG integration with lazy loading
"""
import logging
from typing import Optional, Dict, Any
import asyncio

from rag.document_processor import DocumentProcessor
from rag.vector_store_optimized import MemoryOptimizedFAISS
from llm.llm_service import UnifiedLLMService

logger = logging.getLogger(__name__)

class OptimizedRAGPipeline:
    """Memory-optimized RAG pipeline for 1GB RAM"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = MemoryOptimizedFAISS()
        self.llm_service = UnifiedLLMService()
        self.initialized = False
        
        logger.info("OptimizedRAGPipeline initialized (lazy loading enabled)")
    
    async def initialize(self):
        """Lazy initialization"""
        if not self.initialized:
            await self.vector_store.initialize()
            self.initialized = True
            logger.info("RAG pipeline fully initialized")
    
    async def analyze_risk(self, query: str, document_text: str = "", 
                          top_k: int = 3, redis_client = None) -> Dict[str, Any]:
        """Analyze risk with memory optimization"""
        await self.initialize()
        
        try:
            # Process document if provided
            if document_text:
                chunks = self.document_processor.chunk_document(document_text)
                if chunks:
                    await self.vector_store.add_documents(chunks)
            
            # Search for relevant documents
            relevant_docs = await self.vector_store.search(query, top_k=top_k)
            
            # Prepare context
            context = "\n".join([doc[0] for doc in relevant_docs])
            
            # Generate analysis using LLM
            analysis = await self.llm_service.generate_risk_analysis(
                query=query,
                context=context,
                relevant_docs=relevant_docs
            )
            
            # Get memory stats
            vector_stats = self.vector_store.get_stats()
            
            return {
                "query": query,
                "analysis": analysis,
                "relevant_documents": [
                    {"content": doc[0], "similarity_score": doc[1]} 
                    for doc in relevant_docs
                ],
                "memory_stats": {
                    "vectors_count": vector_stats["vectors_count"],
                    "memory_mb": vector_stats["memory_mb"]
                },
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {
                "query": query,
                "analysis": f"Error: {str(e)}",
                "relevant_documents": [],
                "memory_stats": {"error": "Failed to get stats"},
                "status": "error"
            }

# Global instance
_rag_pipeline = None

async def get_rag_pipeline():
    """Get singleton RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = OptimizedRAGPipeline()
    return _rag_pipeline
