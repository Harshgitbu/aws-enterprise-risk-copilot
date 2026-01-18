"""
Enhanced Advanced AI Copilot with better fallback
"""
from analysis.advanced_copilot import AdvancedAICopilot
from llm.llm_service_enhanced import get_enhanced_llm_service

class EnhancedAdvancedAICopilot(AdvancedAICopilot):
    """
    Enhanced version with better fallback responses
    """
    
    def __init__(self, llm_service=None, rag_pipeline=None):
        super().__init__(llm_service=llm_service, rag_pipeline=rag_pipeline)
        self.enhanced_llm_service = get_enhanced_llm_service()
    
    async def ask(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """
        Enhanced ask method with better fallback
        """
        try:
            # Try parent class first
            response = await super().ask(query, context_type)
            
            # If parent returned fallback_error, use enhanced service
            if response.get("llm_used") == "fallback_error":
                enhanced_response = self.enhanced_llm_service.get_intelligent_response(query)
                
                # Update response with enhanced content
                response["answer"] = enhanced_response["analysis"]
                response["llm_used"] = "enhanced_fallback"
                response["source"] = enhanced_response["source"]
                response["topic"] = enhanced_response.get("topic", "general")
                response["note"] = enhanced_response.get("note", "")
                
                # Update conversation history
                self._update_history(query, response["answer"])
            
            return response
            
        except Exception as e:
            # Ultimate fallback
            import traceback
            logger.error(f"Error in enhanced copilot: {e}\n{traceback.format_exc()}")
            
            return {
                "query": query,
                "answer": f"I encountered an error: {str(e)[:100]}...\n\nFor risk analysis, I recommend:\n1. Check SEC 10-K filings for specific companies\n2. Review recent news articles\n3. Use the dashboard for data visualization\n\nError details have been logged.",
                "sources": [],
                "confidence": 0.0,
                "response_time": 0.0,
                "context_used": False,
                "llm_used": "error_fallback",
                "conversation_id": len(self.conversation_history),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

# Global instance
_enhanced_advanced_copilot = None

def get_enhanced_advanced_copilot(llm_service=None, rag_pipeline=None) -> EnhancedAdvancedAICopilot:
    """Get singleton enhanced advanced copilot instance"""
    global _enhanced_advanced_copilot
    if _enhanced_advanced_copilot is None:
        _enhanced_advanced_copilot = EnhancedAdvancedAICopilot(
            llm_service=llm_service, 
            rag_pipeline=rag_pipeline
        )
    return _enhanced_advanced_copilot
