"""
Lightweight Sentiment Analysis for 1GB RAM
Uses VADER for financial sentiment (lightweight, rule-based)
"""
import logging
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)

class LightweightSentimentAnalyzer:
    """
    Lightweight sentiment analyzer using VADER (if available) or rule-based
    """
    
    def __init__(self):
        self.vader_analyzer = None
        self._try_load_vader()
        
        # Enhanced financial dictionary
        self.financial_lexicon = {
            "positive": {
                "growth": 1.5, "profit": 1.8, "gain": 1.6, "surge": 1.7, "rally": 1.6,
                "beat": 1.4, "exceed": 1.3, "strong": 1.2, "robust": 1.3, "record": 1.5,
                "high": 1.3, "increase": 1.2, "expand": 1.2, "improve": 1.3, "recover": 1.4
            },
            "negative": {
                "loss": -1.8, "decline": -1.5, "fall": -1.4, "drop": -1.4, "plunge": -1.7,
                "miss": -1.3, "disappoint": -1.4, "weak": -1.3, "poor": -1.4, "bearish": -1.5,
                "breach": -1.7, "hack": -1.8, "lawsuit": -1.6, "investigation": -1.5,
                "fine": -1.6, "penalty": -1.7, "downgrade": -1.5, "cut": -1.4
            }
        }
        
        logger.info("LightweightSentimentAnalyzer initialized")
    
    def _try_load_vader(self):
        """Try to load VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("✅ VADER sentiment analyzer loaded")
        except ImportError:
            logger.warning("VADER not installed, using rule-based analysis")
            self.vader_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        if not text or not text.strip():
            return self._empty_sentiment()
        
        # Try VADER first if available
        if self.vader_analyzer:
            try:
                vader_result = self.vader_analyzer.polarity_scores(text)
                
                # Map VADER scores to our format
                compound = vader_result['compound']
                
                if compound >= 0.05:
                    label = "positive"
                    score = (compound + 1) / 2  # Normalize to 0-1
                elif compound <= -0.05:
                    label = "negative"
                    score = (compound + 1) / 2  # Normalize to 0-1
                else:
                    label = "neutral"
                    score = 0.5
                
                # Enhance with financial context
                enhanced = self._enhance_with_financial(text, label, score)
                
                return {
                    "label": enhanced["label"],
                    "score": enhanced["score"],
                    "confidence": abs(compound),  # Use absolute compound as confidence
                    "vader_scores": vader_result,
                    "financial_enhanced": enhanced.get("financial_data"),
                    "model_used": "vader_financial",
                    "method": "vader_enhanced"
                }
            except Exception as e:
                logger.error(f"VADER analysis failed: {e}")
        
        # Fallback to rule-based analysis
        return self._rule_based_analysis(text)
    
    def _enhance_with_financial(self, text: str, base_label: str, base_score: float) -> Dict[str, Any]:
        """Enhance sentiment with financial context"""
        text_lower = text.lower()
        
        # Calculate financial sentiment
        financial_score = 0.0
        financial_words = []
        
        for sentiment, words in self.financial_lexicon.items():
            for word, weight in words.items():
                if word in text_lower:
                    financial_score += weight
                    financial_words.append((word, weight))
        
        # Normalize financial score
        word_count = len(financial_words)
        if word_count > 0:
            financial_normalized = (financial_score / (word_count * 2) + 1) / 2  # Normalize to 0-1
        else:
            financial_normalized = 0.5
        
        # Blend scores (70% base, 30% financial)
        blended_score = (base_score * 0.7) + (financial_normalized * 0.3)
        
        # Determine final label
        if blended_score >= 0.6:
            final_label = "positive"
        elif blended_score <= 0.4:
            final_label = "negative"
        else:
            final_label = "neutral"
        
        return {
            "label": final_label,
            "score": blended_score,
            "financial_data": {
                "word_count": word_count,
                "financial_score": financial_score,
                "financial_normalized": financial_normalized,
                "words_found": financial_words
            }
        }
    
    def _rule_based_analysis(self, text: str) -> Dict[str, Any]:
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
        
        # Simple word counting
        positive_count = 0
        negative_count = 0
        
        for word in self.financial_lexicon["positive"]:
            if word in text_lower:
                positive_count += 1
        
        for word in self.financial_lexicon["negative"]:
            if word in text_lower:
                negative_count += 1
        
        total_words = positive_count + negative_count
        
        if total_words == 0:
            # Check for neutral/context words
            neutral_words = ["stable", "maintain", "steady", "unchanged", "flat", "consistent"]
            neutral_count = sum(1 for word in neutral_words if word in text_lower)
            
            if neutral_count > 0:
                return {
                    "label": "neutral",
                    "score": 0.5,
                    "confidence": 0.3,
                    "word_counts": {"positive": 0, "negative": 0, "neutral": neutral_count},
                    "model_used": "rule_based",
                    "method": "rule_based_neutral"
                }
            return self._empty_sentiment()
        
        # Calculate sentiment
        sentiment_score = positive_count / total_words if total_words > 0 else 0.5
        
        if sentiment_score >= 0.7:
            label = "positive"
        elif sentiment_score <= 0.3:
            label = "negative"
        else:
            label = "neutral"
        
        confidence = max(positive_count, negative_count) / total_words if total_words > 0 else 0.5
        
        return {
            "label": label,
            "score": sentiment_score,
            "confidence": confidence,
            "word_counts": {"positive": positive_count, "negative": negative_count, "neutral": 0},
            "model_used": "rule_based_financial",
            "method": "rule_based"
        }
    
    def _empty_sentiment(self) -> Dict[str, Any]:
        """Return empty sentiment result"""
        return {
            "label": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "word_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "model_used": "none",
            "method": "empty"
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze batch of texts"""
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_sentiment_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get sentiment summary"""
        if not analyses:
            return self._empty_summary()
        
        positive = sum(1 for a in analyses if a["label"] == "positive")
        negative = sum(1 for a in analyses if a["label"] == "negative")
        neutral = sum(1 for a in analyses if a["label"] == "neutral")
        
        total = len(analyses)
        avg_score = sum(a["score"] for a in analyses) / total
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "average_score": avg_score,
            "sentiment_distribution": {
                "positive": positive / total,
                "negative": negative / total,
                "neutral": neutral / total
            }
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Empty summary"""
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "average_score": 0.5,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
        }

# Global instance
_light_sentiment = None

def get_light_sentiment() -> LightweightSentimentAnalyzer:
    """Get singleton lightweight sentiment analyzer"""
    global _light_sentiment
    if _light_sentiment is None:
        _light_sentiment = LightweightSentimentAnalyzer()
    return _light_sentiment
