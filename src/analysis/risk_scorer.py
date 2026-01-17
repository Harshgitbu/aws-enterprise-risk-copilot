"""
AI Risk Scoring Algorithm for AWS Risk Copilot
Analyzes SEC risk data and generates risk scores
Memory optimized for 1GB RAM
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RiskScorer:
    """
    AI-powered risk scoring system
    Combines:
    1. SEC risk factor analysis
    2. Keyword risk detection
    3. Comparative risk assessment
    """
    
    # Risk categories and keywords (weights 1-10)
    RISK_CATEGORIES = {
        "cybersecurity": {
            "keywords": ["cybersecurity", "data breach", "hack", "security incident", "unauthorized access", "ransomware", "phishing"],
            "weight": 9
        },
        "financial": {
            "keywords": ["bankruptcy", "debt", "liquidity", "cash flow", "revenue decline", "profit margin", "interest rates"],
            "weight": 8
        },
        "regulatory": {
            "keywords": ["regulation", "compliance", "lawsuit", "antitrust", "fcc", "ftc", "sec investigation", "fine"],
            "weight": 8
        },
        "competition": {
            "keywords": ["competition", "market share", "competitive pressure", "pricing pressure", "disruption"],
            "weight": 7
        },
        "supply_chain": {
            "keywords": ["supply chain", "manufacturing", "logistics", "inventory", "shortage", "delays"],
            "weight": 6
        },
        "personnel": {
            "keywords": ["key personnel", "talent retention", "employee turnover", "labor dispute", "union"],
            "weight": 5
        },
        "technology": {
            "keywords": ["technology disruption", "obsolescence", "innovation", "patent", "intellectual property"],
            "weight": 6
        }
    }
    
    def __init__(self):
        logger.info("RiskScorer initialized")
    
    def analyze_risk_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze risk text and extract risk metrics
        
        Args:
            text: Risk factor text from SEC filing
            
        Returns:
            Dictionary with risk analysis
        """
        if not text:
            return {"error": "Empty text"}
        
        text_lower = text.lower()
        results = {
            "total_risk_score": 0,
            "category_scores": {},
            "keyword_matches": {},
            "risk_sentences": [],
            "text_length": len(text),
            "word_count": len(text.split())
        }
        
        # Analyze each risk category
        for category, info in self.RISK_CATEGORIES.items():
            category_score = 0
            matches = []
            
            for keyword in info["keywords"]:
                # Count occurrences
                count = text_lower.count(keyword.lower())
                if count > 0:
                    matches.append({
                        "keyword": keyword,
                        "count": count,
                        "weight": info["weight"]
                    })
                    category_score += count * info["weight"]
            
            if matches:
                results["category_scores"][category] = {
                    "score": category_score,
                    "matches": matches,
                    "weight": info["weight"]
                }
                results["total_risk_score"] += category_score
                results["keyword_matches"][category] = len(matches)
        
        # Extract risk sentences (sentences with risk keywords)
        sentences = re.split(r'[.!?]+', text)
        risk_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for category, info in self.RISK_CATEGORIES.items():
                for keyword in info["keywords"]:
                    if keyword.lower() in sentence_lower and len(sentence.strip()) > 20:
                        risk_sentences.append({
                            "sentence": sentence.strip(),
                            "category": category,
                            "keyword": keyword
                        })
                        break
        
        results["risk_sentences"] = risk_sentences[:10]  # Limit to 10
        
        # Normalize score (0-100 scale)
        max_possible_score = sum([cat["weight"] * 10 for cat in self.RISK_CATEGORIES.values()])  # Assuming max 10 mentions
        if max_possible_score > 0:
            results["normalized_score"] = min(100, (results["total_risk_score"] / max_possible_score) * 100)
        else:
            results["normalized_score"] = 0
        
        # Risk level categorization
        normalized = results["normalized_score"]
        if normalized >= 70:
            results["risk_level"] = "HIGH"
            results["risk_color"] = "red"
        elif normalized >= 40:
            results["risk_level"] = "MEDIUM"
            results["risk_color"] = "orange"
        else:
            results["risk_level"] = "LOW"
            results["risk_color"] = "green"
        
        logger.info(f"Risk analysis complete: {results['risk_level']} ({normalized:.1f}/100)")
        return results
    
    def compare_companies(self, company_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare risk across multiple companies
        
        Args:
            company_analyses: Dict of company_name -> risk analysis
            
        Returns:
            Comparative analysis
        """
        if not company_analyses:
            return {"error": "No company analyses provided"}
        
        results = {
            "companies": {},
            "highest_risk": None,
            "lowest_risk": None,
            "average_score": 0,
            "comparison": {}
        }
        
        scores = []
        
        for company, analysis in company_analyses.items():
            score = analysis.get("normalized_score", 0)
            results["companies"][company] = {
                "score": score,
                "risk_level": analysis.get("risk_level", "UNKNOWN"),
                "top_categories": self._get_top_categories(analysis, top_n=3)
            }
            scores.append(score)
        
        if scores:
            results["average_score"] = np.mean(scores)
            
            # Find highest and lowest risk
            if company_analyses:
                companies = list(company_analyses.keys())
                scores_list = [company_analyses[c].get("normalized_score", 0) for c in companies]
                
                if scores_list:
                    max_idx = np.argmax(scores_list)
                    min_idx = np.argmin(scores_list)
                    
                    results["highest_risk"] = {
                        "company": companies[max_idx],
                        "score": scores_list[max_idx],
                        "risk_level": company_analyses[companies[max_idx]].get("risk_level", "UNKNOWN")
                    }
                    
                    results["lowest_risk"] = {
                        "company": companies[min_idx],
                        "score": scores_list[min_idx],
                        "risk_level": company_analyses[companies[min_idx]].get("risk_level", "UNKNOWN")
                    }
        
        # Generate comparison insights
        results["comparison"] = self._generate_comparison_insights(company_analyses)
        
        return results
    
    def _get_top_categories(self, analysis: Dict, top_n: int = 3) -> List[Dict]:
        """Get top risk categories from analysis"""
        category_scores = analysis.get("category_scores", {})
        if not category_scores:
            return []
        
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:top_n]
        
        return [
            {
                "category": cat,
                "score": info["score"],
                "matches": len(info["matches"])
            }
            for cat, info in sorted_categories
        ]
    
    def _generate_comparison_insights(self, company_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate AI insights from comparison"""
        insights = {
            "common_risks": [],
            "unique_risks": {},
            "recommendations": []
        }
        
        # Find common risks across companies
        all_categories = set()
        company_categories = {}
        
        for company, analysis in company_analyses.items():
            categories = set(analysis.get("category_scores", {}).keys())
            all_categories.update(categories)
            company_categories[company] = categories
        
        # Common risks (present in >50% of companies)
        common_categories = []
        for category in all_categories:
            count = sum(1 for cats in company_categories.values() if category in cats)
            if count > len(company_analyses) / 2:
                common_categories.append(category)
        
        insights["common_risks"] = common_categories
        
        # Unique risks per company
        for company, categories in company_categories.items():
            other_categories = set()
            for other_company, other_cats in company_categories.items():
                if other_company != company:
                    other_categories.update(other_cats)
            
            unique = categories - other_categories
            if unique:
                insights["unique_risks"][company] = list(unique)
        
        # Generate recommendations
        if common_categories:
            insights["recommendations"].append(
                f"Common risks across companies: {', '.join(common_categories)}. "
                "Consider industry-wide risk mitigation strategies."
            )
        
        # High risk warning
        for company, analysis in company_analyses.items():
            if analysis.get("risk_level") == "HIGH":
                insights["recommendations"].append(
                    f"{company} shows HIGH risk level. Recommend detailed review and immediate mitigation planning."
                )
        
        if not insights["recommendations"]:
            insights["recommendations"].append(
                "All companies show manageable risk levels. Continue regular monitoring."
            )
        
        return insights
    
    def generate_copilot_response(self, query: str, context: Dict) -> str:
        """
        Generate AI copilot response based on query and risk analysis context
        
        Args:
            query: User query (e.g., "Why is Apple's risk high?")
            context: Risk analysis context
            
        Returns:
            AI-generated response
        """
        query_lower = query.lower()
        
        # Simple rule-based response generation
        # In production, you'd use your LLM service here
        
        if "compare" in query_lower:
            companies = []
            for word in query_lower.split():
                if word.upper() in ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]:
                    companies.append(word.upper())
            
            if companies and "comparison" in context:
                comp = context["comparison"]
                response = f"Comparing {', '.join(companies)}:\n\n"
                
                for company in companies:
                    if company in context.get("companies", {}):
                        info = context["companies"][company]
                        response += f"{company}: Risk Score {info['score']:.1f}/100 ({info['risk_level']})\n"
                
                if comp.get("highest_risk"):
                    response += f"\nHighest risk: {comp['highest_risk']['company']} "
                    response += f"({comp['highest_risk']['score']:.1f}/100)\n"
                
                if comp.get("common_risks"):
                    response += f"\nCommon risks: {', '.join(comp['common_risks'])}"
                
                return response
        
        elif "why" in query_lower and "risk" in query_lower:
            # Extract company name from query
            company = None
            for word in query.split():
                if word in ["Apple", "Microsoft", "Amazon", "Google", "Meta"]:
                    company = word
                    break
            
            if company and company in context.get("companies", {}):
                info = context["companies"][company]
                if "top_categories" in info:
                    categories = [cat["category"] for cat in info["top_categories"][:2]]
                    return f"{company}'s risk is {info['risk_level']} primarily due to: {', '.join(categories)} risks."
        
        elif "recommendation" in query_lower or "mitigate" in query_lower:
            if "recommendations" in context.get("comparison", {}):
                recs = context["comparison"]["recommendations"]
                return "Recommendations:\n" + "\n".join([f"• {r}" for r in recs[:3]])
        
        # Default response
        return f"Based on my analysis: {context.get('average_score', 0):.1f}/100 average risk score across companies. " \
               f"Highest risk: {context.get('highest_risk', {}).get('company', 'N/A')}. " \
               f"Ask me to compare companies or explain specific risks."

# Global instance
_risk_scorer = None

def get_risk_scorer() -> RiskScorer:
    """Get singleton risk scorer instance"""
    global _risk_scorer
    if _risk_scorer is None:
        _risk_scorer = RiskScorer()
    return _risk_scorer
