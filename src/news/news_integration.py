"""
News Integration Service for AWS Risk Copilot
Connects news data to existing risk scoring and Redis alerts
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class NewsIntegrationService:
    """
    Integrates news data with existing risk system
    """
    
    def __init__(self, redis_client=None):
        self.news_client = None
        self.redis_client = redis_client
        self.cache_dir = "./data/news_analysis"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("NewsIntegrationService initialized")
    
    def load_news_client(self):
        """Lazy load news client"""
        if self.news_client is None:
            from news.combined_client import get_combined_client
            self.news_client = get_combined_client()
    
    def fetch_and_analyze_news(self, companies: List[Dict] = None, max_companies: int = 5) -> Dict[str, Any]:
        """
        Fetch and analyze news for companies
        
        Args:
            companies: List of company dicts
            max_companies: Maximum companies to process
            
        Returns:
            Comprehensive news analysis
        """
        self.load_news_client()
        
        logger.info(f"Fetching and analyzing news for {max_companies} companies...")
        
        # Fetch news data
        news_data = self.news_client.get_multiple_companies_news(
            companies=companies,
            max_companies=max_companies,
            articles_per_company=5
        )
        
        # Analyze news data
        analysis = self._analyze_news_data(news_data)
        
        # Send alerts for high-risk news
        self._send_news_alerts(news_data, analysis)
        
        # Save analysis to cache
        self._save_analysis_cache(analysis)
        
        return {
            "news_data": news_data,
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _analyze_news_data(self, news_data: Dict) -> Dict[str, Any]:
        """Analyze news data for risk insights"""
        analysis = {
            "total_articles": 0,
            "companies_with_news": 0,
            "average_risk_score": 0.0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "top_risk_categories": {},
            "high_risk_alerts": [],
            "company_rankings": []
        }
        
        total_risk_score = 0
        total_articles = 0
        
        for ticker, data in news_data.items():
            articles = data.get("articles", [])
            if not articles:
                continue
            
            analysis["companies_with_news"] += 1
            analysis["total_articles"] += len(articles)
            
            # Analyze sentiment distribution
            for article in articles:
                sentiment = article.get("sentiment", {}).get("label", "neutral")
                analysis["sentiment_distribution"][sentiment] += 1
                
                # Track risk categories
                for category in article.get("risk_categories", []):
                    analysis["top_risk_categories"][category] = analysis["top_risk_categories"].get(category, 0) + 1
                
                # Check for high risk alerts
                risk_score = article.get("risk_score", 0)
                if risk_score > 70:
                    analysis["high_risk_alerts"].append({
                        "company": data["company_info"]["name"],
                        "ticker": ticker,
                        "title": article.get("title", ""),
                        "risk_score": risk_score,
                        "risk_categories": article.get("risk_categories", []),
                        "published_at": article.get("published_at", "")
                    })
                
                total_risk_score += risk_score
                total_articles += 1
        
        # Calculate averages
        if total_articles > 0:
            analysis["average_risk_score"] = total_risk_score / total_articles
        
        # Sort top risk categories
        analysis["top_risk_categories"] = dict(
            sorted(analysis["top_risk_categories"].items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        # Create company rankings
        company_scores = []
        for ticker, data in news_data.items():
            if data.get("articles"):
                company_scores.append({
                    "company": data["company_info"]["name"],
                    "ticker": ticker,
                    "article_count": len(data["articles"]),
                    "average_risk_score": data.get("average_risk_score", 0),
                    "latest_article": data["articles"][0].get("published_at", "") if data["articles"] else ""
                })
        
        # Sort by risk score (highest first)
        analysis["company_rankings"] = sorted(
            company_scores, 
            key=lambda x: x["average_risk_score"], 
            reverse=True
        )
        
        # Limit high risk alerts
        analysis["high_risk_alerts"] = analysis["high_risk_alerts"][:10]
        
        logger.info(f"News analysis complete: {analysis['total_articles']} articles, "
                   f"{analysis['companies_with_news']} companies, "
                   f"avg risk: {analysis['average_risk_score']:.1f}")
        
        return analysis
    
    def _send_news_alerts(self, news_data: Dict, analysis: Dict):
        """Send news alerts to Redis (if available)"""
        if not self.redis_client:
            return
        
        try:
            # Send high risk alerts
            for alert in analysis.get("high_risk_alerts", []):
                alert_message = {
                    "type": "news_alert",
                    "severity": "high",
                    "company": alert["company"],
                    "ticker": alert["ticker"],
                    "title": alert["title"],
                    "risk_score": alert["risk_score"],
                    "categories": alert["risk_categories"],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "newsapi"
                }
                
                # Publish to Redis
                self.redis_client.publish("risk:alerts", json.dumps(alert_message))
                logger.info(f"Sent alert for {alert['company']}: {alert['title'][:50]}...")
            
            # Send summary alert
            if analysis["high_risk_alerts"]:
                summary_alert = {
                    "type": "news_summary",
                    "total_alerts": len(analysis["high_risk_alerts"]),
                    "highest_risk_company": analysis["company_rankings"][0]["company"] if analysis["company_rankings"] else "None",
                    "top_category": list(analysis["top_risk_categories"].keys())[0] if analysis["top_risk_categories"] else "None",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                
                self.redis_client.publish("risk:alerts", json.dumps(summary_alert))
                
        except Exception as e:
            logger.error(f"Error sending news alerts: {e}")
    
    def _save_analysis_cache(self, analysis: Dict):
        """Save analysis to cache file"""
        try:
            cache_file = os.path.join(self.cache_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H')}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis": analysis
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analysis cache: {e}")
    
    def get_latest_analysis(self) -> Dict[str, Any]:
        """Get latest news analysis from cache"""
        try:
            # Find most recent cache file
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith("analysis_")]
            if not cache_files:
                return {}
            
            latest_file = max(cache_files)
            cache_path = os.path.join(self.cache_dir, latest_file)
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading latest analysis: {e}")
            return {}
    
    def integrate_with_risk_scoring(self, sec_risk_scores: Dict, news_analysis: Dict) -> Dict[str, Any]:
        """
        Integrate SEC risk scores with news analysis for comprehensive risk view
        
        Args:
            sec_risk_scores: Risk scores from SEC analysis
            news_analysis: News analysis data
            
        Returns:
            Integrated risk assessment
        """
        integrated_scores = {}
        
        # For each company with SEC data
        for company_name, sec_data in sec_risk_scores.items():
            integrated_score = {
                "sec_risk_score": sec_data.get("normalized_score", 0),
                "news_risk_score": 0,
                "combined_risk_score": 0,
                "risk_factors": {
                    "sec": sec_data.get("category_scores", {}),
                    "news": {}
                },
                "sentiment": "neutral",
                "alerts": []
            }
            
            # Find matching news data (by company name or ticker)
            ticker = sec_data.get("ticker", "")
            news_key = ticker if ticker else company_name
            
            # Get news data for this company
            news_data = news_analysis.get("news_data", {}).get(news_key, {})
            if news_data and news_data.get("articles"):
                news_articles = news_data["articles"]
                
                # Calculate news risk score (average of article risk scores)
                news_scores = [a.get("risk_score", 0) for a in news_articles]
                integrated_score["news_risk_score"] = sum(news_scores) / len(news_scores) if news_scores else 0
                
                # Collect news risk factors
                for article in news_articles[:3]:  # Top 3 articles
                    for category in article.get("risk_categories", []):
                        integrated_score["risk_factors"]["news"][category] = integrated_score["risk_factors"]["news"].get(category, 0) + 1
                
                # Determine overall sentiment
                sentiments = [a.get("sentiment", {}).get("label", "neutral") for a in news_articles]
                positive_count = sentiments.count("positive")
                negative_count = sentiments.count("negative")
                
                if negative_count > positive_count:
                    integrated_score["sentiment"] = "negative"
                elif positive_count > negative_count:
                    integrated_score["sentiment"] = "positive"
                
                # Collect high risk alerts
                high_risk_articles = [a for a in news_articles if a.get("risk_score", 0) > 70]
                for article in high_risk_articles[:2]:  # Top 2 high risk
                    integrated_score["alerts"].append({
                        "title": article.get("title", ""),
                        "risk_score": article.get("risk_score", 0),
                        "categories": article.get("risk_categories", []),
                        "published_at": article.get("published_at", "")
                    })
            
            # Calculate combined risk score (weighted)
            sec_weight = 0.6  # SEC data more reliable for long-term risk
            news_weight = 0.4  # News for recent developments
            
            integrated_score["combined_risk_score"] = (
                integrated_score["sec_risk_score"] * sec_weight +
                integrated_score["news_risk_score"] * news_weight
            )
            
            # Determine final risk level
            combined_score = integrated_score["combined_risk_score"]
            if combined_score >= 70:
                integrated_score["risk_level"] = "HIGH"
            elif combined_score >= 40:
                integrated_score["risk_level"] = "MEDIUM"
            else:
                integrated_score["risk_level"] = "LOW"
            
            integrated_scores[company_name] = integrated_score
        
        return integrated_scores

# Global instance
_news_integration = None

def get_news_integration(redis_client=None) -> NewsIntegrationService:
    """Get singleton news integration instance"""
    global _news_integration
    if _news_integration is None:
        _news_integration = NewsIntegrationService(redis_client=redis_client)
    return _news_integration
