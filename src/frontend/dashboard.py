"""
AWS Risk Copilot - Professional Dashboard
Fixed and working version
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
from typing import Dict, List, Any, Optional

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
API_TIMEOUT = 10

# Set page configuration
st.set_page_config(
    page_title="AWS Risk Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk badges */
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-high {
        background-color: #fee2e2;
        color: #dc2626;
    }
    
    .risk-medium {
        background-color: #fef3c7;
        color: #d97706;
    }
    
    .risk-low {
        background-color: #d1fae5;
        color: #059669;
    }
    
    /* News cards */
    .news-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class WorkingRiskDashboard:
    """Working Risk Dashboard - Fixed Version"""
    
    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or BACKEND_URL
        
    def make_request(self, endpoint: str, method: str = "GET", data: dict = None) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.backend_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=API_TIMEOUT)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=API_TIMEOUT)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Connection error: {e}")
            return None
    
    def display_header(self):
        """Display header"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🛡️ AWS Risk Copilot</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Enterprise AI Risk Intelligence | 1GB RAM • $0/month</div>', unsafe_allow_html=True)
        
        with col2:
            # Quick status
            health = self.make_request("/health")
            if health:
                mem_mb = health.get("memory_mb", 0)
                st.metric("Memory", f"{mem_mb:.0f}MB")
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    
    def display_sidebar(self):
        """Display sidebar"""
        with st.sidebar:
            st.markdown("## 🎯 Quick Actions")
            
            if st.button("📊 Run Analysis", use_container_width=True):
                st.info("Analysis running...")
            
            if st.button("📰 Fetch News", use_container_width=True):
                with st.spinner("Fetching news..."):
                    result = self.make_request("/news/fetch", method="POST", data={"max_companies": 3})
                    if result:
                        st.success(f"Fetched {result.get('analysis', {}).get('total_articles', 0)} articles")
            
            st.divider()
            
            # Quick Search
            st.markdown("### 🔍 Quick Search")
            search_query = st.text_input("Search company...", key="sidebar_search")
            
            if search_query:
                results = self.make_request(f"/search/companies?query={search_query}")
                if results and results.get("results"):
                    for company in results["results"][:3]:
                        st.write(f"**{company['name']}** ({company['ticker']})")
            
            st.divider()
            
            # System Status
            st.markdown("### ⚙️ System Status")
            health = self.make_request("/health")
            if health:
                status = health.get("status", "unknown")
                st.write(f"Status: **{status.title()}**")
    
    def display_dashboard(self):
        """Display main dashboard"""
        st.markdown("## 📊 Dashboard Overview")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Companies", "55+")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk", "8")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("AI Analyses", "1,247")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Response Time", "0.8s")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Risk Distribution")
            
            # Sample data
            sector_data = pd.DataFrame({
                'Sector': ['Technology', 'Financial', 'Healthcare', 'Consumer'],
                'Risk Score': [72, 65, 58, 45]
            })
            
            fig = px.bar(sector_data, x='Sector', y='Risk Score',
                        color='Risk Score',
                        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📰 News Sentiment")
            
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [15, 20, 7]
            })
            
            fig = px.pie(sentiment_data, values='Count', names='Sentiment',
                        color_discrete_map={'Positive': '#10b981', 'Neutral': '#6b7280', 'Negative': '#ef4444'})
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Alerts
        st.divider()
        st.markdown("### 🚨 Recent Alerts")
        
        alerts = [
            {"company": "Tesla", "issue": "Cybersecurity investigation", "risk": "High"},
            {"company": "Amazon", "issue": "FTC antitrust lawsuit", "risk": "High"},
            {"company": "Apple", "issue": "Supply chain disruption", "risk": "Medium"}
        ]
        
        for alert in alerts:
            st.markdown(f"""
            <div class="news-card">
                <strong>{alert['company']}</strong>: {alert['issue']}
                <span class="risk-badge risk-{alert['risk'].lower()}">{alert['risk']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    def display_ai_copilot(self):
        """Display AI Copilot - FIXED"""
        st.markdown("## 🤖 AI Risk Copilot")
        
        # Initialize conversation
        if 'ai_conversation' not in st.session_state:
            st.session_state.ai_conversation = [
                {"role": "ai", "message": "Hello! I'm your AI Risk Copilot. I can analyze company risks, compare businesses, and provide actionable insights. What would you like to know?"}
            ]
        
        # Display conversation
        for message in st.session_state.ai_conversation:
            if message["role"] == "ai":
                with st.chat_message("assistant"):
                    st.markdown(message["message"])
            else:
                with st.chat_message("user"):
                    st.markdown(message["message"])
        
        # Chat input - SIMPLIFIED
        st.markdown("---")
        
        # Create a text input and button instead of chat_input
        user_input = st.text_input(
            "Ask me anything about company risks...",
            key="ai_input",
            placeholder="Type your question here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", use_container_width=True, type="primary")
        
        if send_button and user_input:
            # Add user message
            st.session_state.ai_conversation.append({"role": "user", "message": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    result = self.make_request(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": user_input}
                    )
                    
                    if result and result.get("status") == "success":
                        response = result.get("response", {}).get("answer", "I couldn't generate a response.")
                    else:
                        response = "I apologize, but I'm having trouble accessing the analysis service."
                    
                except Exception as e:
                    response = f"Error: {str(e)}. Please try again."
                
                # Add AI response to conversation
                st.session_state.ai_conversation.append({"role": "ai", "message": response})
            
            st.rerun()
        
        # Example queries
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        
        examples = [
            "What are Apple's main risks?",
            "Compare Tesla and Amazon",
            "Latest news risks for Microsoft",
            "How to mitigate cybersecurity risks?"
        ]
        
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                if st.button(example, use_container_width=True, key=f"ex_{idx}"):
                    # Set the input field value
                    st.session_state.ai_input = example
                    st.rerun()
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.ai_conversation = [
                {"role": "ai", "message": "Conversation cleared. How can I help you?"}
            ]
            st.rerun()
    
    def display_companies(self):
        """Display Companies Explorer"""
        st.markdown("## 🏢 Company Explorer")
        
        # Search box
        search_query = st.text_input("Search companies by name or ticker...", key="company_search")
        
        if search_query:
            with st.spinner("Searching..."):
                results = self.make_request(f"/search/companies?query={search_query}")
                
                if results and results.get("status") == "success" and results.get("results"):
                    companies = results["results"]
                    
                    st.success(f"Found {len(companies)} companies")
                    
                    # Display in a clean table
                    df_data = []
                    for company in companies[:10]:  # Limit to 10
                        risk_score = company.get("risk_score", 50)
                        risk_level = "High" if risk_score > 70 else "Medium" if risk_score > 40 else "Low"
                        
                        df_data.append({
                            "Company": company["name"],
                            "Ticker": company["ticker"],
                            "Sector": company.get("sector", "N/A"),
                            "Risk Score": risk_score,
                            "Risk Level": risk_level,
                            "Match Score": company.get("match_score", 0)
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show details for selected company
                        if len(df_data) > 0:
                            selected_company = st.selectbox(
                                "Select a company for detailed analysis",
                                [f"{row['Company']} ({row['Ticker']})" for row in df_data]
                            )
                            
                            if selected_company and st.button("Analyze Selected Company"):
                                # Extract ticker from selection
                                ticker = selected_company.split("(")[-1].replace(")", "").strip()
                                
                                with st.spinner(f"Analyzing {ticker}..."):
                                    # Get company details
                                    details = self.make_request(f"/real/company/{ticker}")
                                    
                                    if details and details.get("status") == "success":
                                        st.markdown(f"### 📊 {details.get('name')} Analysis")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Risk Score", f"{details.get('risk_score', 0)}/100")
                                        with col2:
                                            st.metric("Risk Level", details.get("risk_level", "N/A"))
                                        with col3:
                                            st.metric("Sector", details.get("sector", "N/A"))
                                        
                                        # Risk factors
                                        risk_factors = details.get("risk_factors", {})
                                        if risk_factors:
                                            st.markdown("#### 🔍 Risk Factors")
                                            for factor, desc in risk_factors.items():
                                                with st.expander(f"{factor.replace('_', ' ').title()}"):
                                                    st.write(desc)
                else:
                    st.warning("No companies found. Try: apple, google, microsoft, amazon, tesla")
        else:
            # Show popular companies
            st.info("💡 **Try searching for:** apple, google, microsoft, amazon, tesla, nvidia, meta")
            
            # Quick buttons for popular companies
            st.markdown("#### 🚀 Popular Companies")
            popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
            
            cols = st.columns(4)
            for idx, ticker in enumerate(popular):
                with cols[idx % 4]:
                    if st.button(ticker, use_container_width=True, key=f"pop_{ticker}"):
                        # Set search query
                        st.session_state.company_search = ticker
                        st.rerun()
    
    def display_risk_analysis(self):
        """Display Risk Analysis"""
        st.markdown("## 📈 Risk Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Single Company", "Compare Companies", "Sector Analysis"])
        
        with tab1:
            st.markdown("### Single Company Analysis")
            
            company = st.selectbox(
                "Select Company",
                ["Apple (AAPL)", "Microsoft (MSFT)", "Amazon (AMZN)", "Google (GOOGL)", "Tesla (TSLA)", "Meta (META)", "NVIDIA (NVDA)"],
                key="single_company"
            )
            
            if st.button("Analyze", use_container_width=True):
                # Extract ticker
                ticker = company.split("(")[-1].replace(")", "").strip()
                
                with st.spinner("Analyzing..."):
                    # Get AI analysis
                    result = self.make_request(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": f"Analyze risks for {ticker}"}
                    )
                    
                    if result and result.get("status") == "success":
                        analysis = result.get("response", {}).get("answer", "No analysis available.")
                        st.markdown(analysis)
                    else:
                        st.error("Could not generate analysis")
        
        with tab2:
            st.markdown("### Compare Companies")
            
            selected_companies = st.multiselect(
                "Select 2-3 companies to compare",
                ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA"],
                default=["AAPL", "MSFT", "AMZN"]
            )
            
            if len(selected_companies) >= 2:
                if st.button("Compare", use_container_width=True):
                    companies_str = ", ".join(selected_companies)
                    
                    with st.spinner("Comparing..."):
                        result = self.make_request(
                            "/ai/copilot/advanced",
                            method="POST",
                            data={"query": f"Compare risk between {companies_str}"}
                        )
                        
                        if result and result.get("status") == "success":
                            comparison = result.get("response", {}).get("answer", "No comparison available.")
                            st.markdown(comparison)
                            
                            # Simple comparison chart
                            st.markdown("#### 📊 Risk Score Comparison")
                            
                            # Sample scores
                            scores = {ticker: np.random.randint(50, 85) for ticker in selected_companies}
                            
                            fig = go.Figure(data=[
                                go.Bar(x=list(scores.keys()), y=list(scores.values()),
                                      marker_color=['#ef4444' if score > 70 else '#f59e0b' if score > 40 else '#10b981' for score in scores.values()])
                            ])
                            
                            fig.update_layout(
                                title="Risk Score Comparison",
                                xaxis_title="Company",
                                yaxis_title="Risk Score (0-100)",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 2 companies")
        
        with tab3:
            st.markdown("### Sector Analysis")
            
            sector = st.selectbox(
                "Select Sector",
                ["Technology", "Financial", "Healthcare", "Consumer", "Energy"],
                key="sector_select"
            )
            
            if st.button("Analyze Sector", use_container_width=True):
                with st.spinner("Analyzing sector..."):
                    # Get sector analysis
                    result = self.make_request(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": f"Analyze risks in the {sector} sector"}
                    )
                    
                    if result and result.get("status") == "success":
                        analysis = result.get("response", {}).get("answer", "No analysis available.")
                        st.markdown(analysis)
    
    def display_news(self):
        """Display News Monitor"""
        st.markdown("## 📰 News Monitor")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📡 Fetch Latest News", use_container_width=True, type="primary"):
                with st.spinner("Fetching news..."):
                    result = self.make_request("/news/fetch", method="POST", data={"max_companies": 5})
                    
                    if result and result.get("status") == "success":
                        st.success(f"Fetched {result.get('analysis', {}).get('total_articles', 0)} articles")
                        
                        # Show stats
                        analysis = result.get("analysis", {})
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Articles", analysis.get("total_articles", 0))
                        with cols[1]:
                            st.metric("Avg Risk", f"{analysis.get('average_risk_score', 0):.1f}")
                        with cols[2]:
                            st.metric("High Risk", len(analysis.get("high_risk_alerts", [])))
                        with cols[3]:
                            st.metric("Companies", analysis.get("companies_with_news", 0))
                        
                        # Show high risk alerts
                        alerts = analysis.get("high_risk_alerts", [])
                        if alerts:
                            st.markdown("#### 🚨 High Risk Alerts")
                            for alert in alerts[:5]:
                                risk_score = alert.get("risk_score", 0)
                                risk_color = "#ef4444" if risk_score > 70 else "#f59e0b"
                                
                                st.markdown(f"""
                                <div class="news-card">
                                    <strong>{alert.get('company', 'Unknown')}</strong>
                                    <div style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0;">{alert.get('title', '')}</div>
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span style="font-size: 0.8rem; color: #9ca3af;">{alert.get('published_at', '')[:10]}</span>
                                        <span style="font-weight: 600; color: {risk_color};">Risk: {risk_score:.0f}/100</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to fetch news")
        
        with col2:
            st.markdown("#### 🔧 Settings")
            max_companies = st.slider("Max Companies", 1, 10, 5)
            days_back = st.slider("Days Back", 1, 30, 7)
        
        # News API status
        st.markdown("---")
        st.markdown("#### 🔌 API Status")
        
        news_stats = self.make_request("/news/stats")
        if news_stats:
            news_api = news_stats.get("news_api", {})
            status = "🟢 Real Data" if news_api.get("enabled") else "🟡 Sample Mode"
            st.write(f"**NewsAPI:** {status}")
            
            if news_api.get("rate_limit_remaining"):
                st.write(f"**Rate Limit:** {news_api.get('rate_limit_remaining')} requests remaining")
    
    def display_system(self):
        """Display System Health"""
        st.markdown("## ⚙️ System Health")
        
        # System Status
        health = self.make_request("/health")
        status = self.make_request("/status")
        
        if health and status:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                health_status = health.get("status", "unknown")
                st.metric("Status", health_status.title())
            
            with col2:
                mem_mb = health.get("memory_mb", 0)
                st.metric("Memory", f"{mem_mb:.0f}MB")
            
            with col3:
                cpu_percent = status.get("cpu", {}).get("percent", 0)
                st.metric("CPU", f"{cpu_percent:.1f}%")
            
            with col4:
                redis_status = health.get("redis", "unknown")
                st.metric("Redis", redis_status.title())
        
        # API Keys Status
        st.markdown("### 🔑 API Keys Status")
        
        cols = st.columns(4)
        
        with cols[0]:
            gemini_key = os.getenv("GOOGLE_API_KEY")
            status = "🟢 Configured" if gemini_key and len(gemini_key) > 10 else "🔴 Not Set"
            st.metric("Gemini AI", status)
        
        with cols[1]:
            news_key = os.getenv("NEWSAPI_KEY")
            status = "🟢 Configured" if news_key and len(news_key) > 10 else "🔴 Not Set"
            st.metric("NewsAPI", status)
        
        with cols[2]:
            finnhub_key = os.getenv("FINNHUB_API_KEY")
            status = "🟢 Configured" if finnhub_key and len(finnhub_key) > 10 else "🔴 Not Set"
            st.metric("Finnhub", status)
        
        with cols[3]:
            sec_email = os.getenv("SEC_EDGAR_EMAIL")
            status = "🟢 Configured" if sec_email else "🔴 Not Set"
            st.metric("SEC EDGAR", status)
        
        # System Information
        st.markdown("### 📊 System Information")
        
        info_cols = st.columns(2)
        
        with info_cols[0]:
            st.markdown("**Backend Information:**")
            st.write(f"- URL: `{self.backend_url}`")
            st.write(f"- Memory Limit: 1GB RAM (EC2 t3.micro)")
            st.write(f"- Cost: $0/month (AWS Free Tier)")
        
        with info_cols[1]:
            st.markdown("**Project Links:**")
            st.write("- [GitHub Repository](https://github.com/Harshgitbu/aws-enterprise-risk-copilot)")
            st.write("- [API Documentation](http://54.88.98.50:8000/docs)")
            st.write("- [Dashboard](http://54.88.98.50:8501)")
    
    def run(self):
        """Main dashboard runner"""
        # Display header
        self.display_header()
        
        # Display sidebar
        self.display_sidebar()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard",
            "🤖 AI Copilot",
            "🏢 Companies",
            "📈 Risk Analysis",
            "📰 News",
            "⚙️ System"
        ])
        
        with tab1:
            self.display_dashboard()
        
        with tab2:
            self.display_ai_copilot()
        
        with tab3:
            self.display_companies()
        
        with tab4:
            self.display_risk_analysis()
        
        with tab5:
            self.display_news()
        
        with tab6:
            self.display_system()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>"
            "AWS Risk Copilot • Built on AWS Free Tier • "
            "<a href='https://github.com/Harshgitbu/aws-enterprise-risk-copilot' target='_blank'>GitHub</a> • "
            f"Backend: {self.backend_url}"
            "</div>",
            unsafe_allow_html=True
        )

def main():
    """Main entry point"""
    # Priority order for backend URLs
    backend_urls = [
        os.getenv("BACKEND_URL"),  # Render environment variable
        "http://backend:8000",      # Docker Compose
        "http://localhost:8000",    # Local development
        "https://risk-copilot-backend.onrender.com",  # Render production
        "http://54.88.98.50:8000"   # EC2 fallback
    ]
    
    backend_url = None
    for url in backend_urls:
        if url:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    backend_url = url
                    print(f"✅ Connected to backend at {url}")
                    break
            except Exception as e:
                print(f"❌ Failed to connect to {url}: {e}")
                continue
    
    if not backend_url:
        # Default to local
        backend_url = "http://localhost:8000"
        print(f"⚠️  Using default backend: {backend_url}")
    
    # Initialize and run dashboard
    dashboard = WorkingRiskDashboard(backend_url)
    dashboard.run()

if __name__ == "__main__":
    main()
