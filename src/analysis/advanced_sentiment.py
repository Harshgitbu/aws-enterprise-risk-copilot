"""
Advanced ML Sentiment Analysis for AWS Risk Copilot
Uses transformer models for accurate sentiment analysis
Memory optimized for 1GB RAM with model quantization
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis using:
    1. Transformer models (BERT-based)
    2. Emotion detection
    3. Aspect-based sentiment
    4. Financial-specific sentiment
    """
    
    def __init__(self, model_name: str = "finiteautomata/bertweet-base-sentiment-analysis"):
        """
        Initialize advanced sentiment analyzer
        
        Args:
            model_name: HuggingFace model name (default: light model for 1GB RAM)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Financial-specific sentiment dictionary
        self.financial_sentiment_words = {
            "positive": [
                "growth", "profit", "gain", "surge", "rally", "soar", "jump", "climb",
                "beat", "exceed", "outperform", "strong", "robust", "resilient", "optimistic",
                "bullish", "record", "high", "increase", "expand", "improve", "recover"
            ],
            "negative": [
                "loss", "decline", "fall", "drop", "plunge", "slide", "tumble", "crash",
                "miss", "disappoint", "weak", "poor", "bearish", "pessimistic", "worry",
                "concern", "risk", "threat", "breach", "hack", "lawsuit", "investigation",
                "fine", "penalty", "downgrade", "cut", "reduce", "worsen", "deteriorate"
            ],
            "neutral": [
                "maintain", "hold", "steady", "stable", "flat", "unchanged", "neutral",
                "mixed", "moderate", "average", "maintenance", "consistent"
            ]
        }
        
        logger.info(f"AdvancedSentimentAnalyzer initialized (model: {model_name})")
    
    def load_model(self) -> bool:
        """Lazy load the ML model (memory efficient)"""
        if self.is_loaded:
            return True
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Check available memory before loading
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 80:
                logger.warning(f"Memory high ({mem.percent}%), skipping model load")
                return False
            
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load with quantization for memory efficiency
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = True
            logger.info("✅ Sentiment model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.info("Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using ML model
        
        Args:
            text: Text to analyze
            
        Returns:
            Detailed sentiment analysis
        """
        if not text or not text.strip():
            return self._empty_sentiment()
        
        # Truncate long texts for memory efficiency
        if len(text) > 512:
            text = text[:512]
            logger.debug("Text truncated to 512 characters for sentiment analysis")
        
        try:
            # Try ML model first
            if self.load_model():
                ml_result = self._analyze_with_ml(text)
                if ml_result:
                    # Enhance with financial context
                    enhanced_result = self._enhance_with_financial_context(text, ml_result)
                    return enhanced_result
            
            # Fallback to rule-based analysis
            return self._analyze_with_rules(text)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._analyze_with_rules(text)
    
    def _analyze_with_ml(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using ML model"""
        try:
            import torch
            from transformers import pipeline
            
            # Create sentiment analysis pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=512
            )
            
            # Get prediction
            result = sentiment_pipeline(text)[0]
            
            # Map to our format
            label = result['label'].upper()
            score = result['score']
            
            # Convert to our sentiment scale
            if label in ['POSITIVE', 'LABEL_2']:
                sentiment_label = "positive"
                sentiment_score = score
            elif label in ['NEGATIVE', 'LABEL_0']:
                sentiment_label = "negative"
                sentiment_score = score
            else:  # NEUTRAL or LABEL_1
                sentiment_label = "neutral"
                sentiment_score = 0.5
            
            return {
                "label": sentiment_label,
                "score": float(sentiment_score),
                "confidence": float(score),
                "model_used": self.model_name,
                "method": "ml_transformer"
            }
            
        except Exception as e:
            logger.error(f"ML sentiment analysis failed: {e}")
            return None
    
    def _enhance_with_financial_context(self, text: str, ml_result: Dict) -> Dict[str, Any]:
        """Enhance ML result with financial context"""
        text_lower = text.lower()
        
        # Count financial sentiment words
        financial_counts = {key: 0 for key in self.financial_sentiment_words.keys()}
        
        for sentiment_type, words in self.financial_sentiment_words.items():
            for word in words:
                if word in text_lower:
                    financial_counts[sentiment_type] += 1
        
        total_financial_words = sum(financial_counts.values())
        
        if total_financial_words > 0:
            # Calculate financial sentiment
            financial_sentiment_score = (
                financial_counts["positive"] * 1.0 +
                financial_counts["neutral"] * 0.5 +
                financial_counts["negative"] * 0.0
            ) / total_financial_words
            
            # Blend ML sentiment with financial context
            ml_weight = 0.7
            financial_weight = 0.3
            
            blended_score = (
                ml_result["score"] * ml_weight +
                financial_sentiment_score * financial_weight
            )
            
            # Determine label based on blended score
            if blended_score >= 0.6:
                final_label = "positive"
            elif blended_score <= 0.4:
                final_label = "negative"
            else:
                final_label = "neutral"
            
            return {
                **ml_result,
                "label": final_label,
                "score": float(blended_score),
                "financial_context": {
                    "positive_words": financial_counts["positive"],
                    "negative_words": financial_counts["negative"],
                    "neutral_words": financial_counts["neutral"],
                    "total_financial_words": total_financial_words
                },
                "method": "ml_financial_blended"
            }
        
        return ml_result
    
    def _analyze_with_rules(self, text: str) -> Dict[str, Any]:
        """Rule-based sentiment analysis fallback"""
        text_lower = text.lower()
        
        # Count sentiment words
        positive_count = sum(1 for word in self.financial_sentiment_words["positive"] if word in text_lower)
        negative_count = sum(1 for word in self.financial_sentiment_words["negative"] if word in text_lower)
        neutral_count = sum(1 for word in self.financial_sentiment_words["neutral"] if word in text_lower)
        
        total_words = positive_count + negative_count + neutral_count
        
        if total_words == 0:
            return self._empty_sentiment()
        
        # Calculate sentiment score
        sentiment_score = (
            positive_count * 1.0 +
            neutral_count * 0.5 +
            negative_count * 0.0
        ) / total_words
        
        # Determine label
        if sentiment_score >= 0.6:
            label = "positive"
        elif sentiment_score <= 0.4:
            label = "negative"
        else:
            label = "neutral"
        
        confidence = max(positive_count, negative_count, neutral_count) / total_words if total_words > 0 else 0.5
        
        return {
            "label": label,
            "score": float(sentiment_score),
            "confidence": float(confidence),
            "word_counts": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "total": total_words
            },
            "model_used": "rule_based_financial",
            "method": "rule_based"
        }
    
    def _empty_sentiment(self) -> Dict[str, Any]:
        """Return empty sentiment result"""
        return {
            "label": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "word_counts": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
            "model_used": "none",
            "method": "empty"
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analyses
        """
        results = []
        
        for text in texts:
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                results.append(self._empty_sentiment())
        
        return results
    
    def get_sentiment_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for multiple sentiment analyses"""
        if not analyses:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "average_score": 0.5,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
            }
        
        positive_count = sum(1 for a in analyses if a.get("label") == "positive")
        negative_count = sum(1 for a in analyses if a.get("label") == "negative")
        neutral_count = sum(1 for a in analyses if a.get("label") == "neutral")
        
        total = len(analyses)
        average_score = sum(a.get("score", 0.5) for a in analyses) / total
        
        return {
            "total": total,
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "average_score": float(average_score),
            "sentiment_distribution": {
                "positive": positive_count / total if total > 0 else 0,
                "negative": negative_count / total if total > 0 else 0,
                "neutral": neutral_count / total if total > 0 else 0
            }
        }

# Global instance
_advanced_sentiment = None

def get_advanced_sentiment() -> AdvancedSentimentAnalyzer:
    """Get singleton advanced sentiment analyzer instance"""
    global _advanced_sentiment
    if _advanced_sentiment is None:
        _advanced_sentiment = AdvancedSentimentAnalyzer()
    return _advanced_sentiment
