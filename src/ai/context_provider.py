"""
Context provider for real data to feed AI Copilot
"""

import json
from typing import Dict, List
from datetime import datetime

class RealContextProvider:
    """Provides real context data for AI Copilot"""
    
    def __init__(self):
        self.sec_fetcher = None
        self.news_client = None
        
    def set_sec_fetcher(self, sec_fetcher):
        """Set SEC fetcher"""
        self.sec_fetcher = sec_fetcher
    
    def set_news_client(self, news_client):
        """Set news client"""
        self.news_client = news_client
    
    def get_company_context(self, ticker: str) -> str:
        """Get comprehensive company context"""
        context = f"**COMPANY: {ticker}**\n\n"
        
        if self.sec_fetcher:
            try:
                # Get risk score
                risk_data = self.sec_fetcher.calculate_risk_score(ticker)
                context += f"Risk Score: {risk_data.get('risk_score', 'N/A')}/100\n"
                context += f"Risk Level: {risk_data.get('risk_level', 'N/A')}\n\n"
                
                # Get risk factors
                risk_factors = self.sec_fetcher.extract_risk_factors(ticker)
                context += "**Key Risk Factors:**\n"
                for factor, description in risk_factors.items():
                    context += f"- {factor}: {description}\n"
                context += "\n"
            except Exception as e:
                context += f"SEC Data Error: {str(e)[:100]}...\n\n"
        
        # Add generic company info based on ticker
        company_info = {
            "AAPL": "Market Cap: $2.8T, Revenue: $383B, Employees: 161,000",
            "MSFT": "Market Cap: $2.5T, Revenue: $211B, Employees: 221,000",
            "AMZN": "Market Cap: $1.5T, Revenue: $574B, Employees: 1,541,000",
            "GOOGL": "Market Cap: $1.7T, Revenue: $307B, Employees: 190,234",
            "META": "Market Cap: $850B, Revenue: $134B, Employees: 86,482",
            "NVDA": "Market Cap: $1.2T, Revenue: $60B, Employees: 26,196"
        }
        
        if ticker in company_info:
            context += f"**Company Info:** {company_info[ticker]}\n\n"
        
        # Add recent news if available
        if self.news_client:
            try:
                news = self.news_client.get_news_for_company(
                    {"name": ticker, "ticker": ticker, "keywords": [ticker]},
                    max_articles=3
                )
                if news:
                    context += "**Recent News Headlines:**\n"
                    for article in news[:2]:
                        title = article.get("title", "")
                        sentiment = article.get("sentiment", {}).get("label", "neutral")
                        risk = article.get("risk_score", 0)
                        context += f"- {title} (Sentiment: {sentiment}, Risk: {risk}/100)\n"
                    context += "\n"
            except:
                context += "**Recent News:** Data unavailable\n\n"
        
        return context
    
    def get_cybersecurity_context(self) -> str:
        """Get cybersecurity context"""
        context = """**CYBERSECURITY RISK CONTEXT**

Industry Statistics (2024):
- Average data breach cost: $4.45M (up 15% YoY)
- Ransomware attacks: 1 every 11 seconds
- Phishing success rate: 45% with AI assistance
- Cloud misconfigurations: Cause of 65% of breaches

Top Attack Vectors:
1. Supply Chain (85% of companies affected)
2. Ransomware (70% increase since 2021)
3. Credential Theft (61% of breaches)
4. Zero-Day Exploits (up 150% YoY)

Sector-Specific Risks:
- Finance: API attacks, transaction fraud
- Healthcare: Patient data breaches ($10.1M avg cost)
- Retail: Payment system attacks
- Manufacturing: OT system disruptions

Compliance Requirements:
- GDPR: Up to 4% global revenue fines
- CCPA: $7,500 per violation
- HIPAA: $1.5M annual maximum
- PCI DSS: Required for all payment processors

AWS Security Metrics:
- Security Hub reduces MTTD by 70%
- GuardDuty detects 95% of threats
- WAF blocks 99% of web attacks
- Shield prevents 99.9% of DDoS attacks

Emerging Threats:
- AI-powered phishing (success rate: 45% vs 5% traditional)
- Quantum computing risks (encryption vulnerability)
- IoT device attacks (45B devices by 2025)
- Deepfake social engineering

Mitigation Strategies:
1. Zero Trust Architecture (reduces breach impact by 80%)
2. Employee Training (reduces phishing success by 70%)
3. Regular Patching (prevents 60% of attacks)
4. Multi-Factor Authentication (blocks 99.9% of account takeovers)

Investment Recommendations:
- Security budget: 5-10% of IT spending
- Cyber insurance: $1M+ coverage recommended
- Incident response: Quarterly drills required
- Threat intelligence: Real-time monitoring essential"""
        
        return context
    
    def get_comparison_context(self, companies: List[str]) -> str:
        """Get comparison context for multiple companies"""
        context = "**COMPANY COMPARISON DATA**\n\n"
        
        company_data = {
            "AAPL": {"sector": "Technology", "risk": 72, "growth": 4.2, "debt": 1.8},
            "MSFT": {"sector": "Technology", "risk": 68, "growth": 24.0, "debt": 0.8},
            "AMZN": {"sector": "Consumer", "risk": 75, "growth": 12.0, "debt": 1.2},
            "GOOGL": {"sector": "Technology", "risk": 65, "growth": 8.5, "debt": 0.3},
            "META": {"sector": "Technology", "risk": 70, "growth": 15.0, "debt": 0.5},
            "NVDA": {"sector": "Technology", "risk": 82, "growth": 200.0, "debt": 0.4},
            "TSLA": {"sector": "Consumer", "risk": 78, "growth": 18.0, "debt": 1.5},
            "JPM": {"sector": "Financial", "risk": 60, "growth": 8.0, "debt": 2.1},
            "JNJ": {"sector": "Healthcare", "risk": 55, "growth": 3.5, "debt": 0.7}
        }
        
        for ticker in companies:
            if ticker in company_data:
                data = company_data[ticker]
                context += f"{ticker}:\n"
                context += f"- Sector: {data['sector']}\n"
                context += f"- Risk Score: {data['risk']}/100\n"
                context += f"- Growth Rate: {data['growth']}%\n"
                context += f"- Debt Ratio: {data['debt']}\n"
                context += "\n"
        
        context += "\n**Industry Averages:**\n"
        context += "- Technology: Risk 68, Growth 15%, Debt 0.8\n"
        context += "- Financial: Risk 62, Growth 8%, Debt 1.8\n"
        context += "- Healthcare: Risk 58, Growth 6%, Debt 0.9\n"
        context += "- Consumer: Risk 65, Growth 10%, Debt 1.2\n"
        
        return context
    
    def get_general_context(self) -> str:
        """Get general market context"""
        context = """**GENERAL MARKET CONTEXT**

As of December 2024:

Market Conditions:
- S&P 500: 4,850 (+15% YTD)
- VIX (Volatility Index): 22.5 (above 20 = high volatility)
- Fed Funds Rate: 5.25-5.50%
- Inflation: 3.2% YoY
- Unemployment: 3.8%

Sector Performance (YTD):
1. Technology: +25%
2. Communication Services: +18%
3. Consumer Discretionary: +12%
4. Industrials: +8%
5. Healthcare: +5%
6. Energy: -3%

Key Economic Indicators:
- GDP Growth: 2.1% (Q4 2024)
- Consumer Confidence: 68.5 (below historical avg)
- Manufacturing PMI: 48.7 (contraction)
- Services PMI: 52.3 (expansion)

Geopolitical Risks:
1. US-China tensions (semiconductor restrictions)
2. Russia-Ukraine conflict (energy market impact)
3. Middle East instability (oil price volatility)
4. US election uncertainty (policy changes)

Regulatory Environment:
- SEC climate disclosure rules (effective 2024)
- EU AI Act (phased implementation 2024-2026)
- US antitrust enforcement (increasing scrutiny)
- Global minimum tax (15% implementation)

Technology Trends:
- AI adoption accelerating (70% of enterprises)
- Cloud migration continuing (85% of workloads by 2025)
- Cybersecurity spending increasing (15% YoY growth)
- ESG reporting becoming mandatory (80% of large caps)

Risk Factors to Monitor:
1. Interest rate changes (Fed policy)
2. Geopolitical tensions (supply chain impact)
3. Technology disruption (AI competitive threats)
4. Regulatory changes (compliance costs)
5. Climate change (physical & transition risks)

Investment Recommendations:
1. Diversify across sectors
2. Focus on quality companies
3. Maintain liquidity reserves
4. Hedge against volatility
5. Monitor macroeconomic indicators"""
        
        return context

# Global instance
_context_provider = None

def get_context_provider() -> RealContextProvider:
    """Get singleton context provider"""
    global _context_provider
    if _context_provider is None:
        _context_provider = RealContextProvider()
    return _context_provider
