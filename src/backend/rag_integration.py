"""
RAG (Retrieval Augmented Generation) pipeline
Integrates: Vector store + LLM + Cache
OPTIMIZED FOR: 1GB RAM, Free Tier constraints
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Memory-efficient RAG pipeline
    Components:
    1. Document processor (chunking)
    2. Vector store (FAISS - from Day 2)
    3. LLM service (from Day 3)
    4. Cache (Redis - from this file)
    5. Embedding service (from Day 2)
    """
    
    def __init__(self):
        """Initialize RAG pipeline with memory optimization"""
        self.components_loaded = {}
        self.initialize_components()
        
        logger.info(f"🔗 RAGPipeline initialized")
        logger.info(f"📊 Components: {list(self.components_loaded.keys())}")
    
    def initialize_components(self):
        """Lazy loading of components to save memory"""
        try:
            # Import only when needed
            from backend.document_processor import default_processor
            self.document_processor = default_processor
            self.components_loaded["document_processor"] = True
            logger.debug("✅ Document processor loaded")
        except ImportError as e:
            logger.warning(f"Document processor not available: {e}")
            self.document_processor = None
        
        try:
            # Import vector store from Day 2
            from rag.vector_store import create_vector_store
            self.vector_store = create_vector_store(dimension=384, max_vectors=5000)
            self.components_loaded["vector_store"] = True
            logger.debug("✅ Vector store loaded (FAISS from Day 2)")
        except ImportError as e:
            logger.warning(f"Vector store not available: {e}")
            self.vector_store = None
        
        try:
            # Import embedding service from Day 2
            from rag.embedding import EmbeddingService
            self.embedding_service = EmbeddingService()
            self.components_loaded["embedding_service"] = True
            logger.debug("✅ Embedding service loaded (with Redis caching)")
        except ImportError as e:
            logger.warning(f"Embedding service not available: {e}")
            self.embedding_service = None
        
        try:
            # Import LLM service from Day 3
            from llm.llm_service import create_llm_service
            self.llm_service = create_llm_service()
            self.components_loaded["llm_service"] = True
            logger.debug("✅ LLM service loaded")
        except ImportError as e:
            logger.warning(f"LLM service not available: {e}")
            self.llm_service = None
        
        try:
            # Import Redis cache
            from backend.redis_cache import redis_cache
            self.cache = redis_cache
            self.components_loaded["cache"] = self.cache.enabled if self.cache else False
            if self.cache and self.cache.enabled:
                logger.debug("✅ Redis cache loaded")
        except ImportError as e:
            logger.warning(f"Redis cache not available: {e}")
            self.cache = None
    
    def analyze_risk_with_rag(self, query: str, 
                            document_text: Optional[str] = None,
                            top_k: int = 3) -> Dict[str, Any]:
        """
        Main RAG pipeline: Retrieve relevant context, then analyze with LLM
        
        Args:
            query: Risk analysis query
            document_text: Optional document text to process
            top_k: Number of relevant chunks to retrieve
        
        Returns:
            Risk analysis with RAG context
        """
        start_time = datetime.now()
        
        # Check if components are available
        if not self.llm_service:
            return {
                "error": "LLM service not available",
                "query": query,
                "timestamp": start_time.isoformat()
            }
        
        # Step 1: Check cache first
        cache_key = f"rag:{hash(query + (document_text or ''))}"
        if self.cache and self.cache.enabled:
            cached = self.cache.get_llm_response(query, document_text or "")
            if cached:
                logger.info("✅ Using cached RAG response")
                cached["source"] = "cache"
                cached["response_time"] = (datetime.now() - start_time).total_seconds()
                return cached
        
        # Step 2: Process document if provided
        context_chunks = []
        if document_text and self.document_processor:
            try:
                chunks = self.document_processor.process_text(
                    document_text,
                    {"source": "user_input", "timestamp": start_time.isoformat()}
                )
                
                # Add to vector store temporarily
                if self.vector_store and self.embedding_service and len(chunks) > 0:
                    texts = [chunk.text for chunk in chunks]
                    
                    # Get embeddings using your Day 2 service
                    embeddings = self.embedding_service.encode(texts)
                    
                    if embeddings is not None and len(embeddings) > 0:
                        # Prepare metadata
                        metadatas = []
                        for chunk in chunks[:len(embeddings)]:
                            metadata = chunk.metadata.copy()
                            metadata.update({
                                "chunk_id": chunk.chunk_id,
                                "token_count": chunk.token_count,
                                "text": chunk.text[:200]  # Store snippet
                            })
                            metadatas.append(metadata)
                        
                        # Add to vector store
                        self.vector_store.add_vectors(embeddings, metadatas)
                        logger.info(f"📄 Added {len(embeddings)} document chunks to vector store")
            except Exception as e:
                logger.error(f"Document processing error: {e}")
        
        # Step 3: Retrieve relevant context from vector store
        retrieved_context = ""
        if self.vector_store and self.embedding_service and hasattr(self.vector_store, 'search'):
            try:
                # Get embedding for query
                query_embedding = self.embedding_service.encode([query])
                
                if query_embedding is not None and len(query_embedding) > 0:
                    # Search for similar vectors
                    results = self.vector_store.search(query_embedding[0], k=top_k)
                    
                    if results:
                        retrieved_context = "\n\n".join([
                            f"[Relevant Document {i+1}]: {doc['metadata'].get('text', str(doc['metadata']))}"
                            for i, doc in enumerate(results[:top_k])
                        ])
                        logger.info(f"🔍 Retrieved {len(results)} relevant chunks from vector store")
                    else:
                        logger.info("🔍 No relevant chunks found in vector store")
            except Exception as e:
                logger.error(f"Vector search error: {e}")
        
        # Step 4: Prepare final context
        final_context = ""
        if retrieved_context:
            final_context = f"Relevant documentation:\n{retrieved_context}"
        
        if document_text and not retrieved_context:
            # If no vector store results but we have document text, use first part
            final_context = f"Document context (first 500 chars):\n{document_text[:500]}..."
        
        # Step 5: Analyze with LLM
        logger.info(f"🤖 Analyzing risk with RAG context ({len(final_context)} chars)")
        
        try:
            analysis_result = self.llm_service.analyze_risk(query, final_context)
            
            # Add RAG metadata
            if isinstance(analysis_result, dict):
                analysis_result["rag_metadata"] = {
                    "has_document_context": bool(document_text),
                    "has_retrieved_context": bool(retrieved_context),
                    "retrieved_chunks_count": len(retrieved_context.split("\n\n")) if retrieved_context else 0,
                    "total_context_length": len(final_context),
                    "cache_used": False,
                    "pipeline_time": (datetime.now() - start_time).total_seconds(),
                    "vector_store_vectors": self.vector_store.current_vectors if self.vector_store else 0
                }
                
                # Cache the result
                if self.cache and self.cache.enabled:
                    self.cache.set_llm_response(
                        query, 
                        document_text or "", 
                        analysis_result,
                        ttl_seconds=3600  # Cache for 1 hour
                    )
            
            logger.info("✅ RAG analysis complete")
            return analysis_result
            
        except Exception as e:
            logger.error(f"RAG analysis error: {e}")
            return {
                "error": f"RAG analysis failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    def batch_process_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Batch process documents for indexing
        Memory-efficient: processes one at a time
        """
        if not self.document_processor or not self.vector_store or not self.embedding_service:
            return {"error": "Required components not available"}
        
        results = {
            "total_documents": len(document_paths),
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for doc_path in document_paths:
            try:
                if not os.path.exists(doc_path):
                    results["errors"].append(f"File not found: {doc_path}")
                    results["failed"] += 1
                    continue
                
                # Process in streaming mode
                chunks_added = 0
                texts_batch = []
                metadatas_batch = []
                
                for chunk in self.document_processor.process_file_streaming(doc_path):
                    texts_batch.append(chunk.text)
                    
                    metadata = chunk.metadata.copy()
                    metadata.update({
                        "chunk_id": chunk.chunk_id,
                        "token_count": chunk.token_count,
                        "text": chunk.text[:200]  # Store snippet
                    })
                    metadatas_batch.append(metadata)
                    chunks_added += 1
                    
                    # Process in batches of 5 to save memory
                    if len(texts_batch) >= 5:
                        embeddings = self.embedding_service.encode(texts_batch)
                        if embeddings is not None:
                            self.vector_store.add_vectors(embeddings, metadatas_batch)
                        
                        texts_batch = []
                        metadatas_batch = []
                
                # Process remaining batch
                if texts_batch:
                    embeddings = self.embedding_service.encode(texts_batch)
                    if embeddings is not None:
                        self.vector_store.add_vectors(embeddings, metadatas_batch)
                
                results["processed"] += 1
                results["total_chunks"] += chunks_added
                
                logger.info(f"✅ Processed {doc_path}: {chunks_added} chunks")
                
                # Manual garbage collection hint
                if results["processed"] % 3 == 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{doc_path}: {str(e)}")
                logger.error(f"Failed to process {doc_path}: {e}")
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status and health"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": self.components_loaded.copy(),
            "memory_info": self._get_memory_info()
        }
        
        # Add vector store stats if available
        if self.vector_store:
            try:
                stats = self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {}
                status["vector_store"] = {
                    "current_vectors": stats.get('current_vectors', 'unknown'),
                    "max_vectors": stats.get('max_vectors', 'unknown'),
                    "dimension": stats.get('dimension', 'unknown')
                }
            except:
                status["vector_store"] = {"status": "error"}
        
        # Add embedding service stats if available
        if self.embedding_service:
            try:
                embedding_stats = self.embedding_service.get_stats() if hasattr(self.embedding_service, 'get_stats') else {}
                status["embedding_service"] = embedding_stats
            except:
                status["embedding_service"] = {"status": "error"}
        
        # Add cache stats if available
        if self.cache and self.cache.enabled:
            try:
                status["cache"] = self.cache.get_stats()
            except:
                status["cache"] = {"status": "error"}
        
        return status
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage info"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": process.memory_percent(),
                "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
            }
        except ImportError:
            return {"error": "psutil not installed"}
        except Exception as e:
            return {"error": str(e)}


# Global RAG pipeline instance
rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Singleton factory for RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline
