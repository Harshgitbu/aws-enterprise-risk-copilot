"""
Memory-optimized RAG integration with lazy loading - FIXED SYNC VERSION
"""
import logging
from typing import Optional, Dict, Any

# Import SYNC versions
from rag.document_processor import DocumentProcessor
from rag.vector_store_optimized import MemoryOptimizedFAISS
from llm.llm_service_async import async_llm_service  # Use async service

logger = logging.getLogger(__name__)

class OptimizedRAGPipeline:
    """Memory-optimized RAG pipeline for 1GB RAM - FIXED SYNC INTERFACE"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = MemoryOptimizedFAISS()
        self.llm_service = async_llm_service
        self.initialized = False
        
        logger.info("OptimizedRAGPipeline initialized")
    
    async def initialize(self):
        """Lazy initialization"""
        if not self.initialized:
            await self.vector_store.initialize()
            self.initialized = True
            logger.info("RAG pipeline fully initialized")
    
    async def analyze_risk(self, query: str, document_text: str = "", 
                          top_k: int = 3, redis_client = None) -> Dict[str, Any]:
        """Analyze risk with memory optimization - FIXED ASYNC"""
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
            
            # Generate analysis using LLM - Use async service
            import asyncio
            analysis = await self.llm_service.generate_response(
                prompt=f"Analyze risk for: {query}\nContext: {context}",
                max_tokens=500,
                temperature=0.3
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
                "memory_stats": vector_stats,
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

def get_rag_pipeline():
    """Get singleton RAG pipeline instance - FIXED SYNC RETURN"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = OptimizedRAGPipeline()
    return _rag_pipeline
