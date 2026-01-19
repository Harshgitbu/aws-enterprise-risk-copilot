"""
Fixed Streamlit Dashboard for AWS Risk Copilot
Clean working version with all fixes
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
import re
from typing import Dict, List, Any, Optional

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
MEMORY_LIMIT_MB = 1024

# Page configuration
st.set_page_config(
    page_title="AI Risk Copilot",
    page_icon="🤖",
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
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk indicators */
    .risk-high { color: #ef4444; font-weight: 600; background: #fef2f2; padding: 4px 12px; border-radius: 20px; display: inline-block; }
    .risk-medium { color: #f59e0b; font-weight: 600; background: #fffbeb; padding: 4px 12px; border-radius: 20px; display: inline-block; }
    .risk-low { color: #10b981; font-weight: 600; background: #f0fdf4; padding: 4px 12px; border-radius: 20px; display: inline-block; }
    
    /* News alerts */
    .news-alert {
        border-left: 4px solid;
        padding: 12px 16px;
        margin: 8px 0;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .alert-high { border-left-color: #ef4444; }
    .alert-medium { border-left-color: #f59e0b; }
    .alert-low { border-left-color: #10b981; }
    
    /* Professional tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Better buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
        border: 1px solid #e5e7eb;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class EnhancedRiskCopilotDashboard:
    """Professional dashboard"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session = requests.Session()
        self.session.timeout = 10
        
    def get_backend_data(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Dict]:
        """Get data from backend with error handling"""
        try:
            if method == "GET":
                response = self.session.get(f"{self.backend_url}{endpoint}", params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(f"{self.backend_url}{endpoint}", json=data, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Backend error: {e}")
        return None
    
    def display_header(self):
        """Display professional header"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="main-header">🤖 AI Risk Copilot</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Enterprise Risk Intelligence | 1GB RAM • $0/month</div>', unsafe_allow_html=True)
        
        with col2:
            health_data = self.get_backend_data("/health")
            if health_data:
                mem_mb = health_data.get("memory_mb", 0)
                usage = (mem_mb / MEMORY_LIMIT_MB) * 100
                st.progress(usage / 100)
                st.caption(f"🚀 Memory: {mem_mb:.0f}MB")
    
    def display_sidebar(self):
        """Display sidebar"""
        with st.sidebar:
            st.markdown("## 🎯 Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Run Analysis", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            st.divider()
            
            # System Status
            st.markdown("## ⚙️ System Status")
            health = self.get_backend_data("/health")
            if health:
                status = health.get("status", "unknown")
                color = "🟢" if status == "healthy" else "🔴"
                st.markdown(f"{color} **Backend:** {status.title()}")
                
                mem_mb = health.get("memory_mb", 0)
                st.markdown(f"💾 **Memory:** {mem_mb:.0f}MB")
            
            st.divider()
            
            # Project Info
            st.markdown("## 📊 Project Info")
            st.info("""
            **AWS Free Tier**
            - EC2 t3.micro: 1GB RAM
            - Cost: $0/month
            - Real-time AI Analysis
            """)
    
    def display_dashboard_home(self):
        """Display dashboard home"""
        st.markdown("## 📊 Dashboard Overview")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Companies", "55")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk", "8")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("AI Analyses", "127")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("System", "Healthy")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Risk Distribution")
            risk_data = pd.DataFrame({
                'Sector': ['Technology', 'Financial', 'Healthcare', 'Consumer'],
                'Risk Score': [72, 65, 58, 45]
            })
            
            fig = px.bar(risk_data, x='Sector', y='Risk Score', color='Risk Score',
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
    
    def display_ai_copilot(self):
        """Display AI Copilot"""
        st.markdown("## 🤖 AI Copilot Assistant")
        
        # Initialize conversation
        if 'conversation' not in st.session_state:
            st.session_state.conversation = [
                {"role": "ai", "message": "Hello! I'm your AI Risk Copilot. How can I help you?"}
            ]
        
        # Display conversation
        for msg in st.session_state.conversation:
            if msg["role"] == "ai":
                st.markdown(f'**AI:** {msg["message"]}')
            else:
                st.markdown(f'**You:** {msg["message"]}')
        
        # Chat input
        user_input = st.text_input("Ask me anything...", key="chat_input")
        
        if st.button("Send", use_container_width=True) and user_input:
            # Add user message
            st.session_state.conversation.append({"role": "user", "message": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                result = self.get_backend_data(
                    "/ai/copilot/advanced",
                    method="POST",
                    data={"query": user_input}
                )
                
                if result:
                    response = result.get("response", {}).get("answer", "I couldn't generate a response.")
                    st.session_state.conversation.append({"role": "ai", "message": response})
            
            st.rerun()
    
    def display_company_explorer(self):
        """Display company explorer"""
        st.markdown("## 🏢 Company Explorer")
        
        # Search box
        search_query = st.text_input("Search companies...")
        
        if search_query:
            # Use search endpoint
            results = self.get_backend_data(f"/search/companies?query={search_query}")
            
            if results and results.get("status") == "success":
                companies = results.get("results", [])
                
                for company in companies:
                    with st.expander(f"{company['name']} ({company['ticker']})"):
                        st.markdown(f"**Match Score:** {company['search_score']}/100")
                        
                        if st.button(f"Analyze {company['ticker']}", key=f"btn_{company['ticker']}"):
                            st.session_state.selected_company = company['name']
                            st.rerun()
        else:
            st.info("💡 Try searching: apple, google, microsoft, tesla, amazon")
    
    def display_risk_analysis(self):
        """Display risk analysis"""
        st.markdown("## 📈 Risk Analysis")
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Single Company", "Compare Companies"],
            horizontal=True
        )
        
        if analysis_type == "Single Company":
            company = st.selectbox("Select Company", ["Apple", "Microsoft", "Amazon", "Google", "Tesla"])
            
            if st.button("Analyze", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = self.get_backend_data(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": f"Analyze risk for {company}"}
                    )
                    
                    if result:
                        response = result.get("response", {}).get("answer", "")
                        st.info(response)
        else:
            companies = st.multiselect("Select Companies", ["Apple", "Microsoft", "Amazon", "Google", "Tesla"])
            
            if len(companies) >= 2 and st.button("Compare", use_container_width=True):
                with st.spinner("Comparing..."):
                    result = self.get_backend_data(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": f"Compare {', '.join(companies)}"}
                    )
                    
                    if result:
                        response = result.get("response", {}).get("answer", "")
                        st.info(response)
    
    def display_news_monitor(self):
        """Display news monitor"""
        st.markdown("## 📰 News Monitor")
        
        if st.button("Fetch Latest News", use_container_width=True):
            with st.spinner("Fetching news..."):
                news_data = self.get_backend_data("/news/fetch?max_companies=3", method="POST")
                
                if news_data:
                    st.success(f"Fetched news data")
                    
                    # Show stats
                    analysis = news_data.get("analysis", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Articles", analysis.get("total_articles", 0))
                    with col2:
                        st.metric("Avg Risk", f"{analysis.get('average_risk_score', 0):.1f}")
                    with col3:
                        st.metric("High Risk", len(analysis.get("high_risk_alerts", [])))
        else:
            st.info("Click 'Fetch Latest News' to load real-time news data")
    
    def display_system_health(self):
        """Display system health"""
        st.markdown("## ⚙️ System Health")
        
        health_data = self.get_backend_data("/health")
        
        if health_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", health_data.get("status", "unknown").title())
            
            with col2:
                mem_mb = health_data.get("memory_mb", 0)
                st.metric("Memory", f"{mem_mb:.0f}MB")
            
            with col3:
                st.metric("Redis", health_data.get("redis", "unknown"))
        
        # News API status
        st.divider()
        st.markdown("### 🔌 API Status")
        
        news_stats = self.get_backend_data("/news/stats")
        if news_stats:
            news_api = news_stats.get("news_api", {})
            st.markdown(f"**NewsAPI:** {'✅ Enabled' if news_api.get('enabled') else '❌ Sample Mode'}")
            st.markdown(f"**Rate Limit:** {news_api.get('rate_limit_remaining', 'N/A')} remaining")
    
    def run(self):
        """Main dashboard runner"""
        # Display header and sidebar
        self.display_header()
        self.display_sidebar()
        
        # Tab navigation
        tabs = st.tabs(["📊 Dashboard", "🤖 AI Copilot", "🏢 Companies", "📈 Risk Analysis", "📰 News", "⚙️ System"])
        
        with tabs[0]:
            self.display_dashboard_home()
        
        with tabs[1]:
            self.display_ai_copilot()
        
        with tabs[2]:
            self.display_company_explorer()
        
        with tabs[3]:
            self.display_risk_analysis()
        
        with tabs[4]:
            self.display_news_monitor()
        
        with tabs[5]:
            self.display_system_health()

def main():
    """Main entry point"""
    # Try to connect to backend
    backend_urls = [
        "http://backend:8000",
        "http://localhost:8000",
        "http://18.215.157.225:8000"
    ]
    
    backend_url = None
    for url in backend_urls:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                backend_url = url
                break
        except:
            continue
    
    if backend_url:
        dashboard = EnhancedRiskCopilotDashboard(backend_url)
        dashboard.run()
    else:
        st.error("❌ Cannot connect to backend service")
        st.info("Check if backend is running and accessible.")

if __name__ == "__main__":
    main()
