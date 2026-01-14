"""
RAG Pipeline Integration for AWS Risk Copilot
Memory-optimized for 1GB RAM constraint - ASYNC VERSION
"""
import os
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
import asyncio

from rag.embedding import embedding_service
from rag.document_processor import DocumentProcessor
from rag.vector_store import MemoryEfficientFAISS
from llm.llm_service_async import async_llm_service

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Memory-efficient RAG pipeline with caching"""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.document_processor = DocumentProcessor()
        self.vector_store = MemoryEfficientFAISS()
        self.llm_service = async_llm_service
        self.cache_ttl = 3600  # 1 hour cache
        
        # Load vector store if exists
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load FAISS vector store from disk"""
        try:
            if os.path.exists("data/faiss_index.index"):
                self.vector_store.load("data/faiss_index.index")
                logger.info(f"✅ Loaded vector store with {self.vector_store.get_vector_count()} vectors")
            else:
                logger.warning("No existing vector store found. Will create new one.")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
    
    async def _get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for query"""
        try:
            from backend.redis_cache import redis_cache
            cache_key = f"rag_response:{hashlib.md5(query.encode()).hexdigest()}"
            cached = await redis_cache.get(cache_key)
            return cached
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None
    
    async def _cache_response(self, query: str, response: Dict[str, Any]):
        """Cache response for query"""
        try:
            from backend.redis_cache import redis_cache
            cache_key = f"rag_response:{hashlib.md5(query.encode()).hexdigest()}"
            await redis_cache.set(cache_key, response, self.cache_ttl)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    async def analyze_risk(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze risk using RAG pipeline (async)
        
        Args:
            query: User query about risk
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Analyzing risk query: {query[:100]}...")
        
        # Check cache first
        if use_cache:
            cached_response = await self._get_cached_response(query)
            if cached_response:
                logger.info("Cache hit for query")
                cached_response['cache_hit'] = True
                return cached_response
        
        try:
            # Generate query embedding (async)
            query_embedding = await self.embedding_service.get_embedding(query)
            if query_embedding is None:
                return {
                    "error": "Failed to generate query embedding",
                    "answer": "I apologize, but I encountered an error processing your query.",
                    "sources": [],
                    "cache_hit": False
                }
            
            # Search for similar documents
            search_results = self.vector_store.search(query_embedding, k=3)
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for i, (doc_text, similarity, metadata) in enumerate(search_results):
                context_parts.append(f"[Document {i+1}, Relevance: {similarity:.2f}]\n{doc_text}")
                if metadata and 'source' in metadata:
                    sources.append(metadata['source'])
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
            
            # Prepare prompt for LLM
            prompt = f"""You are an AI Risk Intelligence Assistant. Analyze the following query about risks, using the provided context.

Query: {query}

Context from risk documents:
{context}

Based on the context above, provide:
1. A comprehensive risk analysis
2. Potential impacts
3. Recommended mitigation strategies
4. Any limitations or assumptions in your analysis

If the context doesn't contain relevant information, acknowledge this and provide general risk management advice.

Response:"""
            
            # Get LLM response
            llm_response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            # Prepare response
            response = {
                "query": query,
                "answer": llm_response,
                "sources": sources[:5],  # Limit to top 5 sources
                "context_used": bool(context_parts),
                "documents_retrieved": len(search_results),
                "cache_hit": False
            }
            
            # Cache the response
            try:
                await self._cache_response(query, response)
            except Exception as e:
                logger.debug(f"Failed to cache response: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            return {
                "error": str(e),
                "answer": "I apologize, but I encountered an error processing your risk analysis request. Please try again.",
                "sources": [],
                "cache_hit": False
            }
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to vector store (async)
        
        Args:
            documents: List of documents with text and metadata
        """
        try:
            # Process documents
            processed_docs = []
            all_texts = []
            all_metadatas = []
            
            for doc in documents:
                chunks = self.document_processor.chunk_document(
                    doc.get('text', ''),
                    chunk_size=500,
                    chunk_overlap=50
                )
                
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({
                        'source': doc.get('source', 'unknown'),
                        'type': doc.get('type', 'document'),
                        'timestamp': doc.get('timestamp', '')
                    })
            
            # Generate embeddings in batches (async)
            embeddings = await self.embedding_service.get_embeddings_batch(all_texts)
            
            # Filter out None embeddings
            valid_data = []
            for text, embedding, metadata in zip(all_texts, embeddings, all_metadatas):
                if embedding is not None:
                    valid_data.append((text, embedding, metadata))
            
            if valid_data:
                # Add to vector store
                texts, embeds, metadatas = zip(*valid_data)
                self.vector_store.add_documents(list(texts), list(embeds), list(metadatas))
                
                # Save vector store
                self.vector_store.save("data/faiss_index.index")
                
                logger.info(f"✅ Added {len(valid_data)} document chunks to vector store")
                return {"status": "success", "added": len(valid_data)}
            else:
                return {"status": "error", "message": "No valid embeddings generated"}
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics (async)"""
        stats = {
            "vector_store": self.vector_store.get_stats(),
        }
        
        try:
            llm_stats = await self.llm_service.get_stats()
            stats["llm_service"] = llm_stats
        except Exception as e:
            stats["llm_service_error"] = str(e)
        
        try:
            embedding_stats = await self.embedding_service.get_stats()
            stats["embedding_service"] = embedding_stats
        except Exception as e:
            stats["embedding_service_error"] = str(e)
        
        return stats

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()

def get_rag_pipeline():
    """Get the global RAG pipeline instance"""
    return rag_pipeline
