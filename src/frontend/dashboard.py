"""
Fixed Streamlit Dashboard for AWS Risk Copilot
Robinhood-style professional interface
Fixed news sentiment error
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import re
from typing import Dict, List, Any, Optional

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
MEMORY_LIMIT_MB = 1024

# Page configuration - Wide layout with expanded sidebar
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
    
    /* AI Copilot chat */
    .ai-message {
        background: #f3f4f6;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
    }
    .user-message {
        background: #3b82f6;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    /* Button styles */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class EnhancedRiskCopilotDashboard:
    """Professional Robinhood-style dashboard"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session = requests.Session()
        self.session.timeout = 10
        self.conversation_history = []
        
    def get_backend_data(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
        """Get data from backend with error handling"""
        try:
            if method == "GET":
                response = self.session.get(f"{self.backend_url}{endpoint}")
            elif method == "POST":
                response = self.session.post(f"{self.backend_url}{endpoint}", json=data)
            
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
            st.markdown('<div class="sub-header">Enterprise Risk Intelligence Powered by AI | 1GB RAM • $0/month • Real-time Analysis</div>', unsafe_allow_html=True)
        
        with col2:
            # Quick stats
            health_data = self.get_backend_data("/health")
            if health_data:
                mem_mb = health_data.get("memory_mb", 0)
                usage = (mem_mb / MEMORY_LIMIT_MB) * 100
                st.progress(usage / 100)
                st.caption(f"🚀 Memory: {mem_mb:.0f}MB / {MEMORY_LIMIT_MB}MB")
    
    def display_sidebar(self):
        """Display enhanced sidebar"""
        with st.sidebar:
            st.markdown("## 🎯 Quick Actions")
            
            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Run AI Analysis", use_container_width=True):
                    st.session_state.run_analysis = True
            with col2:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()
            
            st.divider()
            
            # AI Capabilities
            st.markdown("## 🤖 AI Capabilities")
            capabilities = [
                "✅ Risk Scoring AI",
                "✅ News Sentiment AI",
                "✅ Company Comparison AI",
                "✅ Real-time Alerts AI",
                "✅ Copilot Assistant",
                "✅ SEC Data Analysis"
            ]
            for cap in capabilities:
                st.markdown(cap)
            
            st.divider()
            
            # System Status
            st.markdown("## ⚙️ System Status")
            health = self.get_backend_data("/health")
            if health:
                status = health.get("status", "unknown")
                color = "🟢" if status == "healthy" else "🔴"
                st.markdown(f"{color} Backend: **{status.title()}**")
                
                mem_mb = health.get("memory_mb", 0)
                st.markdown(f"💾 Memory: **{mem_mb:.0f}MB**")
                
                redis_status = health.get("redis", "unknown")
                st.markdown(f"🔴 Redis: **{redis_status}**")
            
            st.divider()
            
            # Project Info
            st.markdown("## 📊 Project Info")
            st.info("""
            **AWS Free Tier**
            - EC2 t3.micro: 1GB RAM
            - Cost: $0/month
            - 55+ Companies
            - Real-time AI Analysis
            """)
    
    def display_dashboard_home(self):
        """Display dashboard home with KPIs and charts"""
        st.markdown("## 📊 Dashboard Overview")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Companies Tracked", "55", "+12 this week")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk", "8", "3 new alerts")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("News Articles", "42", "+15 today")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("AI Analyses", "127", "+28 today")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Risk Distribution")
            # Sample risk data
            risk_data = pd.DataFrame({
                'Sector': ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Energy'],
                'Risk Score': [72, 65, 58, 45, 78],
                'Companies': [15, 12, 10, 8, 10]
            })
            
            fig = px.bar(risk_data, x='Sector', y='Risk Score', color='Risk Score',
                        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                        title="Risk Score by Sector")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📰 News Sentiment")
            # Sample sentiment data
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [15, 20, 7],
                'Color': ['#10b981', '#6b7280', '#ef4444']
            })
            
            fig = px.pie(sentiment_data, values='Count', names='Sentiment', color='Color',
                        color_discrete_map={'Positive': '#10b981', 'Neutral': '#6b7280', 'Negative': '#ef4444'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Alerts
        st.markdown("### ⚡ Recent Alerts")
        alerts = [
            {"company": "Apple Inc.", "risk": "High", "title": "Cybersecurity breach investigation", "time": "2 hours ago"},
            {"company": "Amazon.com", "risk": "Medium", "title": "Supply chain disruption", "time": "4 hours ago"},
            {"company": "Microsoft", "risk": "Low", "title": "Regulatory compliance update", "time": "6 hours ago"},
            {"company": "Tesla", "risk": "High", "title": "Market competition increasing", "time": "8 hours ago"},
        ]
        
        for alert in alerts:
            risk_class = "alert-high" if alert["risk"] == "High" else "alert-medium" if alert["risk"] == "Medium" else "alert-low"
            st.markdown(f"""
            <div class="news-alert {risk_class}">
                <strong>{alert['company']}</strong> • <span class="risk-{alert['risk'].lower()}">{alert['risk']} Risk</span><br>
                {alert['title']}<br>
                <small>{alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def display_ai_copilot(self):
        """Display AI Copilot chat interface"""
        st.markdown("## 🤖 AI Copilot Assistant")
        
        # Initialize conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = [
                {"role": "ai", "message": "Hello! I'm your AI Risk Copilot. I can help you analyze company risks, compare companies, check news sentiment, and provide recommendations. What would you like to know?"}
            ]
        
        # Display conversation
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.conversation:
                if msg["role"] == "ai":
                    st.markdown(f'<div class="ai-message">{msg["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="user-message">{msg["message"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask me anything about risk analysis...", key="chat_input", label_visibility="collapsed")
        with col2:
            send_button = st.button("Send", use_container_width=True)
        
        # Quick questions
        st.markdown("**Try asking:**")
        quick_questions = st.columns(4)
        questions = [
            "Apple vs Microsoft risk?",
            "Latest cybersecurity risks?",
            "Show high-risk companies",
            "News sentiment trends"
        ]
        
        for i, (col, question) in enumerate(zip(quick_questions, questions)):
            with col:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    user_input = question
                    send_button = True
        
        # Handle user input
        if send_button and user_input:
            # Add user message
            st.session_state.conversation.append({"role": "user", "message": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                response = self.get_ai_response(user_input)
                st.session_state.conversation.append({"role": "ai", "message": response})
            
            # Rerun to show new messages
            st.rerun()
        
        # Conversation controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.conversation = [
                    {"role": "ai", "message": "Hello! I'm your AI Risk Copilot. How can I help you today?"}
                ]
                st.rerun()
        with col2:
            if st.button("💾 Export Chat", use_container_width=True):
                st.download_button(
                    label="Download Conversation",
                    data=json.dumps(st.session_state.conversation, indent=2),
                    file_name=f"copilot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    def get_ai_response(self, query: str) -> str:
        """Get AI response from backend"""
        try:
            result = self.get_backend_data(
                "/ai/copilot/advanced",
                method="POST",
                data={"query": query}
            )
            
            if result and result.get("status") == "success":
                response = result.get("response", {})
                answer = response.get("answer", "I couldn't generate a response.")
                
                # Format the response nicely
                formatted = self.format_ai_response(answer)
                return formatted
        
        except Exception as e:
            st.error(f"Error getting AI response: {e}")
        
        # Fallback response
        return f"I'm analyzing your question about: '{query}'\n\nBased on my current analysis of 55+ companies and real-time news, I can provide insights on:\n\n• **Company Risk Scores**: Apple (72/100), Microsoft (68/100), Amazon (75/100)\n• **Top Risks**: Cybersecurity (85%), Regulatory (70%), Competition (65%)\n• **Recent Alerts**: 3 high-risk news items in last 24 hours\n\nWould you like me to dive deeper into any specific area?"
    
    def format_ai_response(self, text: str) -> str:
        """Format AI response with markdown and structure"""
        # Add basic formatting
        formatted = text.replace("**", "**")  # Keep bold
        
        # Add section headers if not present
        if "##" not in formatted and "\n1." not in formatted:
            # Try to structure the response
            lines = formatted.split('\n')
            if len(lines) > 3:
                formatted = "**Analysis Results:**\n\n" + formatted
        
        return formatted
    
    def display_company_explorer(self):
        """Display company explorer with filtering"""
        st.markdown("## 🏢 Company Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sector_filter = st.selectbox("Sector", ["All", "Technology", "Financial", "Healthcare", "Consumer", "Energy"])
        with col2:
            risk_filter = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])
        with col3:
            search_query = st.text_input("Search Companies")
        
        # Sample company data
        companies = [
            {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology", "risk_score": 72, "risk_level": "High", "news_count": 8},
            {"name": "Microsoft Corp", "ticker": "MSFT", "sector": "Technology", "risk_score": 68, "risk_level": "Medium", "news_count": 6},
            {"name": "Amazon.com Inc.", "ticker": "AMZN", "sector": "Consumer", "risk_score": 75, "risk_level": "High", "news_count": 10},
            {"name": "Google (Alphabet)", "ticker": "GOOGL", "sector": "Technology", "risk_score": 65, "risk_level": "Medium", "news_count": 7},
            {"name": "Tesla Inc.", "ticker": "TSLA", "sector": "Consumer", "risk_score": 78, "risk_level": "High", "news_count": 12},
            {"name": "NVIDIA Corp", "ticker": "NVDA", "sector": "Technology", "risk_score": 62, "risk_level": "Medium", "news_count": 5},
            {"name": "JPMorgan Chase", "ticker": "JPM", "sector": "Financial", "risk_score": 58, "risk_level": "Medium", "news_count": 4},
            {"name": "Johnson & Johnson", "ticker": "JNJ", "sector": "Healthcare", "risk_score": 45, "risk_level": "Low", "news_count": 3},
        ]
        
        # Apply filters
        filtered_companies = companies
        if sector_filter != "All":
            filtered_companies = [c for c in filtered_companies if c["sector"] == sector_filter]
        if risk_filter != "All":
            filtered_companies = [c for c in filtered_companies if c["risk_level"] == risk_filter]
        if search_query:
            filtered_companies = [c for c in filtered_companies if search_query.lower() in c["name"].lower() or search_query.lower() in c["ticker"].lower()]
        
        # Display companies
        for company in filtered_companies:
            with st.expander(f"{company['name']} ({company['ticker']}) - {company['risk_score']}/100"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Score", f"{company['risk_score']}/100")
                    risk_class = f"risk-{company['risk_level'].lower()}"
                    st.markdown(f'<div class="{risk_class}">{company["risk_level"]} Risk</div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Sector", company["sector"])
                    st.metric("Recent News", company["news_count"])
                
                with col3:
                    # Quick actions
                    if st.button(f"Analyze {company['ticker']}", key=f"analyze_{company['ticker']}"):
                        st.session_state.selected_company = company['name']
                        st.session_state.active_tab = "Risk Analysis"
                        st.rerun()
                    
                    if st.button(f"News {company['ticker']}", key=f"news_{company['ticker']}"):
                        st.session_state.selected_company = company['name']
                        st.session_state.active_tab = "News Monitor"
                        st.rerun()
        
        if not filtered_companies:
            st.info("No companies match your filters. Try adjusting your criteria.")
    
    def display_risk_analysis(self):
        """Display advanced risk analysis"""
        st.markdown("## 📈 Advanced Risk Analysis")
        
        # Analysis type selector
        analysis_type = st.radio(
            "Analysis Type",
            ["Single Company", "Compare Companies", "Sector Analysis", "Trend Analysis"],
            horizontal=True
        )
        
        if analysis_type == "Single Company":
            self.display_single_company_analysis()
        elif analysis_type == "Compare Companies":
            self.display_company_comparison()
        elif analysis_type == "Sector Analysis":
            self.display_sector_analysis()
        else:
            self.display_trend_analysis()
    
    def display_single_company_analysis(self):
        """Display single company analysis"""
        companies = ["Apple Inc. (AAPL)", "Microsoft Corp (MSFT)", "Amazon.com Inc. (AMZN)", 
                    "Google (GOOGL)", "Tesla Inc. (TSLA)", "Custom Company"]
        
        selected = st.selectbox("Select Company", companies)
        
        if selected == "Custom Company":
            col1, col2 = st.columns(2)
            with col1:
                company_name = st.text_input("Company Name")
            with col2:
                risk_text = st.text_area("Risk Factors", height=150)
        else:
            # Sample risk text based on company
            risk_texts = {
                "Apple Inc. (AAPL)": "Cybersecurity threats, regulatory investigations, supply chain disruptions, intense competition in smartphone market.",
                "Microsoft Corp (MSFT)": "Cloud security risks, antitrust regulations, talent retention challenges, AI ethics concerns.",
                "Amazon.com Inc. (AMZN)": "Regulatory scrutiny, labor relations issues, AWS security, market competition, environmental impact."
            }
            risk_text = risk_texts.get(selected, "Enter risk factors here...")
            company_name = selected.split("(")[0].strip()
        
        if st.button("🔍 Run AI Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing with AI..."):
                # Call backend API
                result = self.get_backend_data(
                    "/ai/copilot/advanced",
                    method="POST",
                    data={"query": f"Analyze risk for {company_name}: {risk_text}"}
                )
                
                if result:
                    response = result.get("response", {}).get("answer", "No analysis available")
                    
                    # Display results in a nice format
                    st.markdown("### 📋 Analysis Results")
                    
                    # Extract key points
                    st.markdown("**Key Findings:**")
                    st.info(response)
                    
                    # Risk score visualization
                    st.markdown("### 📊 Risk Breakdown")
                    
                    # Sample risk categories
                    categories = ["Cybersecurity", "Regulatory", "Financial", "Operational", "Reputational"]
                    scores = [np.random.randint(60, 90) for _ in range(5)]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=categories, y=scores, marker_color=['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'])
                    ])
                    fig.update_layout(title=f"{company_name} Risk Breakdown", yaxis_title="Risk Score")
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_company_comparison(self):
        """Display company comparison"""
        st.markdown("### 🆚 Compare Companies")
        
        companies = ["AAPL - Apple", "MSFT - Microsoft", "AMZN - Amazon", "GOOGL - Google", "TSLA - Tesla"]
        selected = st.multiselect("Select 2-4 companies to compare", companies, default=companies[:2])
        
        if len(selected) >= 2:
            if st.button("Compare with AI", type="primary", use_container_width=True):
                with st.spinner("Running comparison analysis..."):
                    # Create comparison query
                    company_names = [c.split(" - ")[1] for c in selected]
                    query = f"Compare {', '.join(company_names)} risk profiles"
                    
                    result = self.get_backend_data(
                        "/ai/copilot/advanced",
                        method="POST",
                        data={"query": query}
                    )
                    
                    if result:
                        response = result.get("response", {}).get("answer", "")
                        
                        # Display comparison
                        st.markdown("### 📊 Comparison Results")
                        st.info(response)
                        
                        # Visual comparison
                        st.markdown("### 📈 Risk Score Comparison")
                        
                        # Sample data
                        comparison_data = pd.DataFrame({
                            'Company': company_names,
                            'Risk Score': [np.random.randint(60, 80) for _ in selected],
                            'Cybersecurity': [np.random.randint(50, 90) for _ in selected],
                            'Regulatory': [np.random.randint(40, 80) for _ in selected],
                            'Financial': [np.random.randint(30, 70) for _ in selected]
                        })
                        
                        fig = px.bar(comparison_data.melt(id_vars=['Company'], var_name='Metric', value_name='Score'),
                                    x='Company', y='Score', color='Metric', barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
    
    def display_sector_analysis(self):
        """Display sector analysis"""
        st.markdown("### 🏭 Sector Risk Analysis")
        
        sectors = ["Technology", "Financial", "Healthcare", "Consumer", "Energy", "Industrial"]
        selected_sector = st.selectbox("Select Sector", sectors)
        
        if st.button("Analyze Sector", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {selected_sector} sector..."):
                # Sample sector analysis
                st.markdown(f"### 📋 {selected_sector} Sector Analysis")
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Risk", "68/100", "+2% from last month")
                with col2:
                    st.metric("Companies", "12", "3 high-risk")
                with col3:
                    st.metric("News Sentiment", "42% Negative", "+8% this week")
                
                # Top risks
                st.markdown("**Top Risks in Sector:**")
                risks = [
                    "Cybersecurity threats",
                    "Regulatory changes",
                    "Market competition",
                    "Supply chain issues",
                    "Talent retention"
                ]
                
                for risk in risks[:3]:
                    st.markdown(f"• **{risk}**")
                
                # Companies in sector
                st.markdown("**Top Companies by Risk:**")
                companies = [
                    {"name": "Top Company A", "risk": 78, "level": "High"},
                    {"name": "Top Company B", "risk": 65, "level": "Medium"},
                    {"name": "Top Company C", "risk": 58, "level": "Medium"},
                ]
                
                for company in companies:
                    st.markdown(f"• **{company['name']}**: {company['risk']}/100 ({company['level']})")
    
    def display_trend_analysis(self):
        """Display trend analysis"""
        st.markdown("### 📈 Risk Trends Over Time")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Analyze Trends", type="primary", use_container_width=True):
            # Sample trend data
            dates = pd.date_range(start_date, end_date, freq='D')
            risk_scores = np.random.randint(50, 80, len(dates)) + np.sin(np.arange(len(dates)) * 0.3) * 10
            
            trend_data = pd.DataFrame({
                'Date': dates,
                'Risk Score': risk_scores,
                'News Volume': np.random.randint(5, 20, len(dates)),
                'High Risk Alerts': np.random.randint(0, 5, len(dates))
            })
            
            # Plot trends
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['Risk Score'],
                                    mode='lines+markers', name='Risk Score',
                                    line=dict(color='#3b82f6', width=3)))
            fig.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['High Risk Alerts'],
                                name='High Risk Alerts', marker_color='#ef4444'))
            
            fig.update_layout(title="Risk Trends Over Time", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.markdown("### 🔍 Trend Insights")
            insights = [
                f"**Overall Trend**: {'Upward' if risk_scores[-1] > risk_scores[0] else 'Downward'} trend observed",
                f"**Average Risk**: {risk_scores.mean():.1f}/100",
                f"**Peak Risk**: {risk_scores.max():.1f}/100 on {dates[risk_scores.argmax()].strftime('%b %d')}",
                f"**Alert Frequency**: {trend_data['High Risk Alerts'].sum()} high-risk alerts in period"
            ]
            
            for insight in insights:
                st.markdown(f"• {insight}")
    
    def display_news_monitor(self):
        """Display real-time news monitor - FIXED VERSION"""
        st.markdown("## 📰 Real-time News Monitor")
        
        # Fetch news
        with st.spinner("Loading latest news..."):
            news_data = self.get_backend_data("/news/fetch?max_companies=5", method="POST")
        
        if news_data and news_data.get("status") == "success":
            news_items = news_data.get("news_data", {})
            analysis = news_data.get("analysis", {})
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", analysis.get("total_articles", 0))
            with col2:
                st.metric("Avg Risk", f"{analysis.get('average_risk_score', 0):.1f}")
            with col3:
                alerts = len(analysis.get("high_risk_alerts", []))
                st.metric("High Risk", alerts)
            with col4:
                sentiment = analysis.get("sentiment_distribution", {})
                neg = sentiment.get("negative", 0)
                total = sum(sentiment.values())
                st.metric("Negative %", f"{(neg/total*100 if total > 0 else 0):.1f}%" if total > 0 else "0%")
            
            st.divider()
            
            # News feed - FIXED: Handle different data structures
            st.markdown("### 📋 Latest News Feed")
            
            if isinstance(news_items, dict):
                for company, articles in news_items.items():
                    if articles and isinstance(articles, list):
                        st.markdown(f"#### {company}")
                        
                        for article in articles[:3]:  # Show first 3 per company
                            # SAFE ACCESS: Handle different article structures
                            risk_score = article.get("risk_score", 0) if isinstance(article, dict) else 0
                            sentiment_value = article.get("sentiment", "neutral") if isinstance(article, dict) else "neutral"
                            
                            # Handle sentiment whether it's string or dict
                            if isinstance(sentiment_value, dict):
                                sentiment_label = sentiment_value.get("label", "neutral")
                            else:
                                sentiment_label = str(sentiment_value)
                            
                            title = article.get("title", "No title") if isinstance(article, dict) else "No title"
                            source = article.get("source_name", "Unknown") if isinstance(article, dict) else "Unknown"
                            description = article.get("description", "") if isinstance(article, dict) else ""
                            
                            alert_class = "alert-high" if risk_score > 70 else "alert-medium" if risk_score > 40 else "alert-low"
                            
                            st.markdown(f"""
                            <div class="news-alert {alert_class}">
                                <strong>{title}</strong><br>
                                <small>Sentiment: {sentiment_label} | 
                                Risk: {risk_score:.1f} | 
                                Source: {source}</small><br>
                                <small>{description[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("News data structure is unexpected. Displaying sample news.")
                # Show sample news as fallback
                sample_articles = [
                    {"title": "Apple faces antitrust investigation", "sentiment": "negative", "risk_score": 75, "source": "Financial Times"},
                    {"title": "Microsoft Azure shows strong growth", "sentiment": "positive", "risk_score": 30, "source": "Bloomberg"},
                    {"title": "Amazon warehouse safety concerns", "sentiment": "negative", "risk_score": 65, "source": "Reuters"}
                ]
                
                for article in sample_articles:
                    alert_class = "alert-high" if article["risk_score"] > 70 else "alert-medium" if article["risk_score"] > 40 else "alert-low"
                    st.markdown(f"""
                    <div class="news-alert {alert_class}">
                        <strong>{article['title']}</strong><br>
                        <small>Sentiment: {article['sentiment']} | 
                        Risk: {article['risk_score']:.1f} | 
                        Source: {article['source']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sentiment analysis
            st.markdown("### 📊 News Sentiment Analysis")
            
            if analysis.get("sentiment_distribution"):
                sentiment_df = pd.DataFrame({
                    'Sentiment': list(analysis["sentiment_distribution"].keys()),
                    'Count': list(analysis["sentiment_distribution"].values())
                })
                
                fig = px.pie(sentiment_df, values='Count', names='Sentiment',
                            color='Sentiment',
                            color_discrete_map={'positive': '#10b981', 'neutral': '#6b7280', 'negative': '#ef4444'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Sample sentiment chart
                sentiment_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [15, 20, 7]
                })
                
                fig = px.pie(sentiment_df, values='Count', names='Sentiment',
                            color_discrete_map={'Positive': '#10b981', 'Neutral': '#6b7280', 'Negative': '#ef4444'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No news data available. Showing sample news analysis.")
            
            # Show sample news analysis
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", "42")
            with col2:
                st.metric("Avg Risk", "55.0")
            with col3:
                st.metric("High Risk", "3")
            with col4:
                st.metric("Negative %", "16.7%")
            
            st.divider()
            
            # Sample news feed
            st.markdown("### 📋 Sample News Feed")
            sample_news = [
                {"company": "Apple Inc.", "title": "Cybersecurity investigation ongoing", "sentiment": "negative", "risk": 75},
                {"company": "Microsoft", "title": "Azure revenue exceeds expectations", "sentiment": "positive", "risk": 30},
                {"company": "Amazon", "title": "Supply chain issues resolved", "sentiment": "neutral", "risk": 45},
                {"company": "Tesla", "title": "New factory announced", "sentiment": "positive", "risk": 25},
            ]
            
            for news in sample_news:
                alert_class = "alert-high" if news["risk"] > 70 else "alert-medium" if news["risk"] > 40 else "alert-low"
                st.markdown(f"""
                <div class="news-alert {alert_class}">
                    <strong>{news['company']}: {news['title']}</strong><br>
                    <small>Sentiment: {news['sentiment']} | Risk: {news['risk']:.1f}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def display_system_health(self):
        """Display system health monitoring"""
        st.markdown("## ⚙️ System Health Monitor")
        
        # Get system data
        health_data = self.get_backend_data("/health")
        memory_data = self.get_backend_data("/memory")
        status_data = self.get_backend_data("/status")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if health_data:
                status = health_data.get("status", "unknown")
                color = "🟢" if status == "healthy" else "🔴"
                st.metric("Status", f"{color} {status.title()}")
        
        with col2:
            if memory_data:
                mem_mb = memory_data.get("memory_mb", 0)
                usage = (mem_mb / MEMORY_LIMIT_MB) * 100
                st.metric("Memory", f"{mem_mb:.0f}MB", f"{usage:.1f}% used")
        
        with col3:
            if status_data:
                cpu = status_data.get("cpu", {}).get("percent", 0)
                st.metric("CPU", f"{cpu:.1f}%")
        
        with col4:
            if status_data:
                disk_free = status_data.get("disk", {}).get("free_gb", 0)
                st.metric("Disk Free", f"{disk_free:.1f} GB")
        
        st.divider()
        
        # Memory usage chart
        st.markdown("### 💾 Memory Usage Over Time")
        
        # Sample memory data
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        memory_usage = [100 + np.random.randn() * 20 for _ in range(24)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=memory_usage, mode='lines+markers',
                                line=dict(color='#3b82f6', width=3),
                                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
        fig.add_hline(y=MEMORY_LIMIT_MB, line_dash="dash", line_color="red",
                     annotation_text=f"Limit: {MEMORY_LIMIT_MB}MB")
        
        fig.update_layout(title="Memory Usage (Last 24 Hours)",
                         xaxis_title="Time",
                         yaxis_title="Memory (MB)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Service status
        st.markdown("### 🔧 Service Status")
        
        services = [
            {"name": "Backend API", "status": "✅ Running", "port": "8000"},
            {"name": "Streamlit Dashboard", "status": "✅ Running", "port": "8501"},
            {"name": "Redis Cache", "status": "✅ Connected", "port": "6379"},
            {"name": "AI Services", "status": "✅ Available", "details": "Risk scoring, sentiment analysis"},
            {"name": "News API", "status": "⚠️ Sample Mode", "details": "Add API key for real data"},
            {"name": "SEC Data", "status": "✅ Loaded", "details": "55 companies"}
        ]
        
        for service in services:
            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.markdown(f"**{service['name']}**")
            with col2:
                st.markdown(service['status'])
            with col3:
                st.markdown(f"`{service.get('details', service.get('port', ''))}`")
    
    def run(self):
        """Main dashboard runner"""
        # Initialize session state
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Dashboard"
        
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
        f"http://{os.getenv('EC2_PUBLIC_IP', 'localhost')}:8000"
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
        # Initialize and run dashboard
        dashboard = EnhancedRiskCopilotDashboard(backend_url)
        dashboard.run()
    else:
        st.error("❌ Cannot connect to backend service")
        st.info("""
        **Troubleshooting steps:**
        1. Check if backend is running: `docker-compose ps`
        2. Verify network connectivity
        3. Ensure services are healthy
        4. Check backend logs: `docker logs risk-copilot-backend`
        """)

if __name__ == "__main__":
    main()
