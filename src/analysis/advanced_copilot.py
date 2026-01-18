"""
Advanced AI Copilot for AWS Risk Copilot
Uses LLM (Gemini) with RAG context for intelligent responses
Memory optimized for 1GB RAM
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class AdvancedAICopilot:
    """
    Advanced AI copilot with:
    1. LLM-powered responses (Gemini/HuggingFace)
    2. RAG context from SEC/news data
    3. Natural language understanding
    4. Multi-turn conversation memory
    """
    
    def __init__(self, llm_service=None, rag_pipeline=None):
        self.llm_service = llm_service
        self.rag_pipeline = rag_pipeline
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 messages
        
        # Load templates and configurations
        self._load_prompts()
        
        logger.info("AdvancedAICopilot initialized")
    
    def _load_prompts(self):
        """Load AI prompt templates"""
        self.prompts = {
            "risk_analysis": """
You are an expert AWS Enterprise Risk Analysis AI Copilot. 
Analyze the following risk query using the provided context.

CONTEXT FROM SEC FILINGS & NEWS:
{context}

USER QUERY: {query}

PREVIOUS CONVERSATION:
{history}

Provide a comprehensive risk analysis with:
1. **Risk Assessment** (Low/Medium/High with confidence score)
2. **Key Risk Factors** (List top 3-5 with evidence from context)
3. **Impact Analysis** (Financial, operational, reputational impact)
4. **Mitigation Recommendations** (AWS-specific solutions)
5. **Monitoring Suggestions** (What to watch for)

Format response as a professional risk analyst would present to executives.
Be specific, data-driven, and actionable.
""",
            
            "company_comparison": """
You are comparing companies based on risk profiles.

COMPANY DATA:
{company_data}

USER QUERY: {query}

Provide a comparative analysis:
1. **Overall Risk Ranking** (Highest to lowest risk)
2. **Sector Comparison** (How each company compares to sector average)
3. **Risk Factor Breakdown** (Cybersecurity, financial, regulatory, etc.)
4. **Investment Implications** (For risk-aware investors)
5. **Recommendations** (Which company is better for specific risk tolerance)

Use specific numbers and percentages from the data.
Be objective and data-driven.
""",
            
            "general_question": """
You are an AI Risk Copilot helping with enterprise risk management.

CONTEXT AVAILABLE:
- SEC 10-K risk factor data for 500+ companies
- Real-time financial news with sentiment analysis
- Market data and volatility metrics
- Historical risk trends

USER QUERY: {query}

PREVIOUS CONVERSATION:
{history}

Provide a helpful, accurate response based on available data.
If you don't have specific data, say so honestly but suggest alternatives.
Focus on risk intelligence and AWS cloud security.
"""
        }
    
    def set_llm_service(self, llm_service):
        """Set LLM service for AI responses"""
        self.llm_service = llm_service
    
    def set_rag_pipeline(self, rag_pipeline):
        """Set RAG pipeline for context retrieval"""
        self.rag_pipeline = rag_pipeline
    
    async def ask(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """
        Ask the AI copilot a question
        
        Args:
            query: User's question
            context_type: Type of context needed (risk_analysis, company_comparison, general)
            
        Returns:
            AI response with metadata
        """
        start_time = datetime.now()
        
        try:
            # Get relevant context from RAG if available
            context = await self._get_relevant_context(query)
            
            # Prepare prompt based on context type
            prompt = self._prepare_prompt(query, context, context_type)
            
            # Get AI response
            if self.llm_service:
                llm_response = self.llm_service.analyze_risk(prompt, "")
            else:
                llm_response = {"analysis": self._fallback_response(query, context)}
            
            # Process response
            response_text = llm_response.get("analysis", "")
            processed_response = self._process_ai_response(response_text)
            
            # Update conversation history
            self._update_history(query, processed_response["answer"])
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "answer": processed_response["answer"],
                "sources": processed_response["sources"],
                "confidence": processed_response["confidence"],
                "response_time": response_time,
                "context_used": bool(context),
                "llm_used": llm_response.get("source", "fallback"),
                "conversation_id": len(self.conversation_history),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in AI copilot: {e}")
            return self._error_response(query, str(e))
    
    async def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from RAG pipeline"""
        if not self.rag_pipeline:
            return ""
        
        try:
            # Search for relevant documents
            relevant_docs = await self.rag_pipeline.vector_store.search(query, top_k=3)
            
            if not relevant_docs:
                return ""
            
            # Format context from documents
            context_parts = []
            for i, (text, score, metadata) in enumerate(relevant_docs):
                source = metadata.get("company", metadata.get("source", "Unknown"))
                context_parts.append(f"[Source {i+1}: {source}, Relevance: {1-score:.2f}]\n{text[:500]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Could not get RAG context: {e}")
            return ""
    
    def _prepare_prompt(self, query: str, context: str, context_type: str) -> str:
        """Prepare AI prompt with context and history"""
        # Get conversation history
        history_text = "\n".join([
            f"User: {h['query']}\nAI: {h['answer'][:100]}..."
            for h in self.conversation_history[-3:]  # Last 3 exchanges
        ])
        
        # Get appropriate prompt template
        template = self.prompts.get(context_type, self.prompts["general_question"])
        
        # Format prompt
        prompt = template.format(
            query=query,
            context=context,
            history=history_text
        )
        
        # Truncate for token limits
        if len(prompt) > 4000:
            prompt = prompt[:4000] + "... [truncated]"
        
        return prompt
    
    def _process_ai_response(self, response: str) -> Dict[str, Any]:
        """Process AI response to extract structured information"""
        # Default values
        processed = {
            "answer": response,
            "sources": [],
            "confidence": 0.7,
            "structured_data": {}
        }
        
        # Try to extract sources from response
        source_patterns = [
            r"\[Source: ([^\]]+)\]",
            r"According to ([^,\.]+)",
            r"Based on ([^,\.]+)"
        ]
        
        for pattern in source_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                processed["sources"] = list(set(matches))
                break
        
        # Try to extract confidence/risk levels
        confidence_patterns = [
            r"(high confidence|confidence: high)",
            r"(medium confidence|confidence: medium)",
            r"(low confidence|confidence: low)"
        ]
        
        for i, pattern in enumerate(confidence_patterns):
            if re.search(pattern, response, re.IGNORECASE):
                processed["confidence"] = [0.9, 0.7, 0.4][i]
                break
        
        # Try to extract structured risk assessment
        risk_sections = {
            "risk_level": r"Risk (Level|Assessment):\s*([^\n]+)",
            "key_factors": r"Key (Risk )?Factors:\s*([^\n]+(?:\n[^\n]+){0,3})",
            "recommendations": r"(Recommendations|Mitigation):\s*([^\n]+(?:\n[^\n]+){0,5})"
        }
        
        for key, pattern in risk_sections.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                processed["structured_data"][key] = match.group(2).strip()
        
        return processed
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when LLM is unavailable"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["risk", "security", "cyber"]):
            return f"Based on available risk data: {context[:500] if context else 'No specific context found.'}\n\nFor detailed risk analysis, I recommend checking SEC 10-K filings and recent news articles about cybersecurity threats."
        
        elif "compare" in query_lower:
            return "I can help compare companies based on their risk profiles. Please specify which companies you'd like to compare, and I'll analyze their SEC risk factors, news sentiment, and market data."
        
        elif "recommend" in query_lower:
            return "For risk mitigation recommendations, consider: 1) Regular security audits, 2) Employee cybersecurity training, 3) Multi-factor authentication, 4) AWS Security Hub for monitoring, 5) Incident response planning."
        
        else:
            return f"I'm your AI Risk Copilot. You asked: '{query}'\n\nI can help with:\n- Company risk analysis\n- Comparative risk assessment\n- SEC filing insights\n- News sentiment analysis\n- Risk mitigation recommendations\n\nPlease ask a specific question about enterprise risk management."
    
    def _update_history(self, query: str, answer: str):
        """Update conversation history"""
        self.conversation_history.append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "answer": f"I encountered an error processing your question: {error}\n\nPlease try again or ask a different question about risk analysis.",
            "sources": [],
            "confidence": 0.0,
            "response_time": 0.0,
            "context_used": False,
            "llm_used": "error",
            "conversation_id": len(self.conversation_history),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            "total_exchanges": len(self.conversation_history),
            "recent_queries": [h["query"] for h in self.conversation_history[-3:]],
            "last_updated": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

# Global instance
_advanced_copilot = None

def get_advanced_copilot(llm_service=None, rag_pipeline=None) -> AdvancedAICopilot:
    """Get singleton advanced copilot instance"""
    global _advanced_copilot
    if _advanced_copilot is None:
        _advanced_copilot = AdvancedAICopilot(llm_service=llm_service, rag_pipeline=rag_pipeline)
    return _advanced_copilot
