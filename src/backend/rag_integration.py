"""
RAG integration module - connects vector store with LLM service
Optimized for 1GB RAM constraint
"""
import sys
import json

# Add /app/src to path for imports
sys.path.insert(0, '/app/src')

# Import with correct class names (check actual names in files)
from rag.vector_store import MemoryEfficientFAISS as VectorStore
from rag.embedding import EmbeddingService
from rag.document_processor import DocumentProcessor
from llm.llm_service import UnifiedLLMService as LLMService

class RAGPipeline:
    """Memory-optimized RAG pipeline for risk analysis"""
    
    def __init__(self):
        # Initialize with lazy loading
        self.vector_store = None
        self.embedding_service = None
        self.document_processor = None
        self.llm_service = None
        
    def initialize_components(self):
        """Lazy initialization of components to save memory"""
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
        
        if self.vector_store is None:
            self.vector_store = VectorStore(
                embedding_service=self.embedding_service,
                index_path="/app/data/vector_store/faiss_index"
            )
        
        if self.document_processor is None:
            self.document_processor = DocumentProcessor()
        
        if self.llm_service is None:
            self.llm_service = LLMService()
    
    def analyze_risk(self, query: str, document_text: str = "", top_k: int = 3, redis_client = None):
        """
        Analyze risk using RAG pipeline
        
        Args:
            query: Risk query
            document_text: Optional document text for context
            top_k: Number of results to return
            redis_client: Redis client for caching
        
        Returns:
            Dictionary with risk analysis
        """
        # Initialize components if needed
        self.initialize_components()
        
        # Check cache first
        cache_key = f"risk:{hash(query + document_text)}"
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        
        # Process document if provided
        chunks = []
        if document_text:
            chunks = self.document_processor.chunk_document(
                document_text,
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Add to vector store temporarily
            if chunks:
                self.vector_store.add_documents(chunks)
        
        # Search for relevant documents
        try:
            search_results = self.vector_store.search(query, top_k=top_k)
        except:
            # If vector store is empty, return empty results
            search_results = []
        
        # Generate context
        context = "\n".join([
            f"[Document {i+1}]: {result['content'][:200]}..."
            for i, result in enumerate(search_results)
        ]) if search_results else "No relevant documents found."
        
        # Generate risk analysis using LLM
        prompt = f"""
        Analyze the following cybersecurity/cloud risk query:
        
        Query: {query}
        
        Context from documents:
        {context}
        
        Provide a risk analysis with:
        1. Risk level (Low/Medium/High/Critical)
        2. Key findings
        3. Recommendations
        4. Immediate actions
        
        Format the response as a JSON object.
        """
        
        try:
            # Try Gemini first (free tier)
            response = self.llm_service.generate(
                prompt=prompt,
                use_gemini=True,
                max_tokens=500
            )
            
            # Parse response
            try:
                result = json.loads(response)
            except:
                # If not JSON, create structured response
                result = {
                    "risk_level": "Medium",
                    "findings": ["Analysis completed"],
                    "recommendations": ["Review the findings"],
                    "actions": ["Monitor the situation"],
                    "analysis": response[:500] if response else "No analysis generated",
                    "source": "gemini"
                }
            
            # Add search context
            result["search_results"] = [
                {
                    "content": r["content"][:100] + "...",
                    "score": r["score"]
                }
                for r in search_results[:2]
            ]
            
            # Cache the result
            if redis_client:
                try:
                    redis_client.setex(
                        cache_key,
                        300,  # 5 minutes TTL
                        json.dumps(result)
                    )
                except:
                    pass
            
            return result
            
        except Exception as e:
            # Fallback response
            return {
                "risk_level": "Unknown",
                "error": str(e),
                "source": "fallback",
                "search_results": search_results
            }

# Singleton instance
_rag_pipeline = None

def get_rag_pipeline():
    """Get singleton RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
