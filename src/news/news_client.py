"""
NewsAPI Client for AWS Risk Copilot
Real-time financial news with sentiment analysis
Memory optimized for 1GB RAM
FREE TIER: 100 requests/day, No credit card needed
"""

import os
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import time

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """
    NewsAPI client with sentiment analysis
    https://newsapi.org (Free: 100 requests/day)
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Target companies for risk analysis
    DEFAULT_COMPANIES = [
        {"name": "Apple", "ticker": "AAPL", "keywords": ["Apple", "AAPL", "iPhone", "Tim Cook"]},
        {"name": "Microsoft", "ticker": "MSFT", "keywords": ["Microsoft", "MSFT", "Azure", "Satya Nadella"]},
        {"name": "Amazon", "ticker": "AMZN", "keywords": ["Amazon", "AMZN", "AWS", "Jeff Bezos"]},
        {"name": "Google", "ticker": "GOOGL", "keywords": ["Google", "Alphabet", "GOOGL", "Sundar Pichai"]},
        {"name": "Meta", "ticker": "META", "keywords": ["Meta", "Facebook", "META", "Mark Zuckerberg"]},
        {"name": "Tesla", "ticker": "TSLA", "keywords": ["Tesla", "TSLA", "Elon Musk", "Electric vehicles"]},
        {"name": "NVIDIA", "ticker": "NVDA", "keywords": ["NVIDIA", "NVDA", "AI chips", "Jensen Huang"]},
    ]
    
    # Risk-related keywords for filtering
    RISK_KEYWORDS = [
        "risk", "breach", "cybersecurity", "hack", "data leak", "security",
        "lawsuit", "regulation", "compliance", "investigation", "fine", "penalty",
        "competition", "antitrust", "monopoly", "market share",
        "financial", "loss", "decline", "revenue", "profit",
        "supply chain", "disruption", "shortage", "delay",
        "layoff", "firing", "resignation", "executive departure"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI client
        
        Args:
            api_key: NewsAPI key (get free at newsapi.org)
        """
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.cache_dir = "./data/news_cache"
        self.session = requests.Session()
        self.rate_limit_remaining = 100  # Free tier limit
        self.last_request_time = 0
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.api_key:
            logger.warning("⚠️  No NewsAPI key found. Using sample data mode.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ NewsAPI client initialized (Free tier: 100 requests/day)")
    
    def _rate_limit(self):
        """Respect rate limits (1 request/second for free tier)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 1.1:  # 1 second + buffer
            sleep_time = 1.1 - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _check_rate_limit(self) -> bool:
        """Check if we have rate limit remaining"""
        return self.rate_limit_remaining > 0
    
    def get_news_for_company(self, company: Dict, days_back: int = 7, max_articles: int = 10) -> List[Dict]:
        """
        Get news articles for a specific company
        
        Args:
            company: Company dict with name, ticker, keywords
            days_back: Number of days to look back
            max_articles: Maximum articles to return
            
        Returns:
            List of news articles with sentiment analysis
        """
        if not self.enabled:
            return self._get_sample_news(company)
        
        if not self._check_rate_limit():
            logger.warning(f"Rate limit reached for {company['name']}")
            return self._get_sample_news(company)
        
        # Check cache first
        cache_key = f"{company['ticker']}_{datetime.now().strftime('%Y%m%d')}"
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.info(f"Using cached news for {company['name']}")
            return cached
        
        self._rate_limit()
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        articles = []
        
        # Search for each keyword
        for keyword in company["keywords"][:3]:  # Limit to 3 keywords for API efficiency
            try:
                params = {
                    "q": f"{keyword} risk",
                    "from": from_date.strftime("%Y-%m-%d"),
                    "to": to_date.strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": min(20, max_articles * 2),
                    "apiKey": self.api_key
                }
                
                response = self.session.get(f"{self.BASE_URL}/everything", params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update rate limit
                    if "X-RateLimit-Remaining" in response.headers:
                        self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
                    
                    if data.get("status") == "ok":
                        for article in data.get("articles", []):
                            if len(articles) >= max_articles:
                                break
                            
                            # Filter for risk-relevant articles
                            if self._is_risk_relevant(article):
                                article_with_sentiment = self._analyze_article(article, company)
                                articles.append(article_with_sentiment)
                
                time.sleep(0.5)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error fetching news for {company['name']} keyword {keyword}: {e}")
                continue
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(articles)
        
        # Limit to max_articles
        final_articles = unique_articles[:max_articles]
        
        # Cache the results
        self._save_to_cache(cache_key, final_articles)
        
        logger.info(f"Found {len(final_articles)} news articles for {company['name']}")
        return final_articles
    
    def _analyze_article(self, article: Dict, company: Dict) -> Dict:
        """
        Analyze article sentiment and extract risk info
        
        Args:
            article: Raw article from NewsAPI
            company: Company info
            
        Returns:
            Enhanced article with sentiment analysis
        """
        title = article.get("title", "")
        description = article.get("description", "")
        content = f"{title} {description}"
        
        # Simple sentiment analysis (production would use ML model)
        sentiment = self._simple_sentiment_analysis(content)
        
        # Detect risk categories
        risk_categories = self._detect_risk_categories(content)
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(content, sentiment, risk_categories)
        
        return {
            "source": "newsapi",
            "company_name": company["name"],
            "company_ticker": company["ticker"],
            "title": title,
            "description": description,
            "url": article.get("url", ""),
            "published_at": article.get("publishedAt", ""),
            "source_name": article.get("source", {}).get("name", ""),
            "sentiment": sentiment,
            "risk_categories": risk_categories,
            "risk_score": risk_score,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "extracted_at": datetime.utcnow().isoformat()
        }
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis (memory efficient)"""
        text_lower = text.lower()
        
        positive_words = ["growth", "profit", "gain", "success", "win", "positive", "strong", "beat", "rise", "up"]
        negative_words = ["loss", "decline", "fall", "drop", "negative", "weak", "miss", "down", "breach", "hack", "lawsuit", "fine"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {
                "label": "neutral",
                "score": 0.5,
                "positive_count": 0,
                "negative_count": 0
            }
        
        sentiment_score = positive_count / total
        
        if sentiment_score > 0.6:
            label = "positive"
        elif sentiment_score < 0.4:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "label": label,
            "score": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count
        }
    
    def _detect_risk_categories(self, text: str) -> List[str]:
        """Detect risk categories from text"""
        text_lower = text.lower()
        categories = []
        
        risk_category_map = {
            "cybersecurity": ["cybersecurity", "data breach", "hack", "security breach", "ransomware"],
            "financial": ["loss", "decline", "revenue drop", "profit warning", "bankruptcy"],
            "regulatory": ["lawsuit", "regulation", "compliance", "investigation", "fine", "penalty"],
            "competition": ["competition", "market share", "antitrust", "monopoly"],
            "supply_chain": ["supply chain", "shortage", "delay", "disruption"],
            "personnel": ["layoff", "firing", "resignation", "executive departure"]
        }
        
        for category, keywords in risk_category_map.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _calculate_risk_score(self, text: str, sentiment: Dict, risk_categories: List[str]) -> float:
        """Calculate risk score (0-100) for article"""
        # Base on sentiment
        sentiment_weight = 0.6
        sentiment_score = 100 * (1 - sentiment["score"])  # Inverse: negative = high risk
        
        # Risk categories weight
        categories_weight = 0.4
        categories_score = len(risk_categories) * 20  # 20 points per category
        
        # Combine scores
        total_score = (sentiment_score * sentiment_weight) + (categories_score * categories_weight)
        
        # Cap at 100
        return min(100.0, total_score)
    
    def _is_risk_relevant(self, article: Dict) -> bool:
        """Check if article is relevant to risk analysis"""
        content = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # Check if contains risk keywords
        if any(keyword in content for keyword in self.RISK_KEYWORDS):
            return True
        
        return False
    
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get("title", "").lower()
            
            # Simple duplicate detection
            is_duplicate = False
            for seen_title in seen_titles:
                if title in seen_title or seen_title in title:
                    is_duplicate = True
                    break
            
            if not is_duplicate and title:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _save_to_cache(self, key: str, data: List[Dict]):
        """Save news data to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "cached_at": datetime.utcnow().isoformat(),
                    "articles": data
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, key: str) -> Optional[List[Dict]]:
        """Load news data from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            
            if os.path.exists(cache_file):
                # Check if cache is fresh (less than 4 hours old)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 4 * 3600:  # 4 hours
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return data.get("articles", [])
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
        
        return None
    
    def _get_sample_news(self, company: Dict) -> List[Dict]:
        """Get sample news data for testing"""
        sample_articles = [
            {
                "source": "newsapi",
                "company_name": company["name"],
                "company_ticker": company["ticker"],
                "title": f"{company['name']} faces cybersecurity investigation after data breach",
                "description": f"Regulators are investigating {company['name']} following a major data breach affecting millions of users.",
                "url": f"https://example.com/news/{company['ticker'].lower()}-breach",
                "published_at": datetime.utcnow().isoformat() + "Z",
                "source_name": "Financial Times",
                "sentiment": {"label": "negative", "score": 0.2, "positive_count": 1, "negative_count": 4},
                "risk_categories": ["cybersecurity", "regulatory"],
                "risk_score": 85.0,
                "content_preview": f"{company['name']} cybersecurity breach investigation...",
                "extracted_at": datetime.utcnow().isoformat()
            },
            {
                "source": "newsapi",
                "company_name": company["name"],
                "company_ticker": company["ticker"],
                "title": f"{company['name']} reports strong quarterly earnings, beats estimates",
                "description": f"{company['name']} exceeded analyst expectations with better-than-expected revenue growth.",
                "url": f"https://example.com/news/{company['ticker'].lower()}-earnings",
                "published_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
                "source_name": "Bloomberg",
                "sentiment": {"label": "positive", "score": 0.8, "positive_count": 5, "negative_count": 1},
                "risk_categories": ["financial"],
                "risk_score": 25.0,
                "content_preview": f"{company['name']} quarterly earnings report shows growth...",
                "extracted_at": datetime.utcnow().isoformat()
            }
        ]
        
        return sample_articles
    
    def get_multiple_companies_news(self, companies: List[Dict] = None, 
                                  max_companies: int = 5, 
                                  articles_per_company: int = 5) -> Dict[str, List[Dict]]:
        """
        Get news for multiple companies
        
        Args:
            companies: List of company dicts
            max_companies: Maximum companies to fetch
            articles_per_company: Articles per company
            
        Returns:
            Dict with company_ticker -> list of articles
        """
        if companies is None:
            companies = self.DEFAULT_COMPANIES
        
        all_news = {}
        
        for i, company in enumerate(companies[:max_companies]):
            logger.info(f"Fetching news for {i+1}/{min(len(companies), max_companies)}: {company['name']}")
            
            articles = self.get_news_for_company(
                company, 
                days_back=7,
                max_articles=articles_per_company
            )
            
            all_news[company["ticker"]] = {
                "company_info": company,
                "articles": articles,
                "count": len(articles),
                "average_risk_score": self._calculate_average_risk(articles) if articles else 0
            }
            
            # Rate limiting between companies
            if i < len(companies[:max_companies]) - 1:
                time.sleep(1)
        
        return all_news
    
    def _calculate_average_risk(self, articles: List[Dict]) -> float:
        """Calculate average risk score for articles"""
        if not articles:
            return 0.0
        
        total_score = sum(article.get("risk_score", 0) for article in articles)
        return total_score / len(articles)

# Global instance
_news_client = None

def get_news_client() -> NewsAPIClient:
    """Get singleton news client instance"""
    global _news_client
    if _news_client is None:
        _news_client = NewsAPIClient()
    return _news_client
