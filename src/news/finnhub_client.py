"""
Finnhub News Client for AWS Risk Copilot
Alternative news source with financial focus
FREE TIER: 60 calls/minute, 50 calls/day
"""

import os
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time

logger = logging.getLogger(__name__)

class FinnhubClient:
    """
    Finnhub client for financial news
    https://finnhub.io (Free: 60 calls/minute, 50/day)
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("⚠️  No Finnhub API key found")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Finnhub client initialized")
    
    def get_company_news(self, symbol: str, days_back: int = 7, max_articles: int = 10) -> List[Dict]:
        """
        Get news for a specific company symbol
        
        Args:
            symbol: Stock symbol (AAPL, MSFT, etc.)
            days_back: Days to look back
            max_articles: Maximum articles to return
        """
        if not self.enabled:
            return self._get_sample_news(symbol)
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        try:
            params = {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": self.api_key
            }
            
            response = self.session.get(f"{self.BASE_URL}/company-news", params=params, timeout=30)
            
            if response.status_code == 200:
                articles = response.json()
                
                # Filter and format
                formatted_articles = []
                for article in articles[:max_articles]:
                    formatted = {
                        "source": "finnhub",
                        "symbol": symbol,
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "published_at": datetime.fromtimestamp(article.get("datetime", 0)).isoformat() if article.get("datetime") else "",
                        "sentiment": self._analyze_sentiment(article.get("headline", "") + " " + article.get("summary", "")),
                        "risk_score": self._calculate_risk_score(article)
                    }
                    formatted_articles.append(formatted)
                
                logger.info(f"Got {len(formatted_articles)} articles from Finnhub for {symbol}")
                return formatted_articles
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
        
        return self._get_sample_news(symbol)
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        text_lower = text.lower()
        positive = ["growth", "profit", "gain", "strong", "beat"]
        negative = ["loss", "decline", "fall", "weak", "miss"]
        
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {"label": "neutral", "score": 0.5}
        
        score = pos_count / total
        label = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        
        return {"label": label, "score": score}
    
    def _calculate_risk_score(self, article: Dict) -> float:
        """Calculate risk score from article"""
        text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
        
        risk_keywords = ["breach", "hack", "lawsuit", "investigation", "fine", "risk"]
        score = 50.0  # Base score
        
        for keyword in risk_keywords:
            if keyword in text:
                score += 10
        
        return min(100.0, score)
    
    def _get_sample_news(self, symbol: str) -> List[Dict]:
        """Sample news fallback"""
        return [
            {
                "source": "finnhub",
                "symbol": symbol,
                "headline": f"{symbol} shows strong quarterly results",
                "summary": f"{symbol} exceeded market expectations with record revenue.",
                "sentiment": {"label": "positive", "score": 0.8},
                "risk_score": 30.0,
                "published_at": datetime.now().isoformat()
            }
        ]

# Global instance
_finnhub_client = None

def get_finnhub_client() -> FinnhubClient:
    global _finnhub_client
    if _finnhub_client is None:
        _finnhub_client = FinnhubClient()
    return _finnhub_client
