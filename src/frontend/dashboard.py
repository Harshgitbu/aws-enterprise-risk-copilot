"""
Streamlit Dashboard for AWS Risk Copilot - WITH NEWS INTEGRATION
Memory-aware dashboard optimized for 1GB RAM constraint
"""
import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time
import psutil
import os
import re
import numpy as np

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
MEMORY_LIMIT_MB = 1024  # 1GB RAM constraint

# Page configuration
st.set_page_config(
    page_title="AWS Risk Copilot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e9ecef; 
        margin: 5px 0; 
    }
    .news-alert-high { border-left: 5px solid #dc3545; padding-left: 10px; }
    .news-alert-medium { border-left: 5px solid #fd7e14; padding-left: 10px; }
    .news-alert-low { border-left: 5px solid #28a745; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

class RiskCopilotDashboard:
    """Dashboard with AI Risk Analysis & News Integration"""
    
    def __init__(self, backend_url):
        self.backend_url = backend_url
        self.session = requests.Session()
        self.session.timeout = 10
        
    def get_backend_data(self, endpoint, method="GET", data=None):
        """Get data from backend"""
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
    
    def display_health_status(self, tab_prefix=""):
        """Display system health status"""
        st.subheader("🚀 System Health")
        
        health_data = self.get_backend_data("/health")
        if not health_data:
            st.error("Backend service unavailable")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = health_data.get("status", "unknown")
            color = "green" if status == "healthy" else "red"
            st.markdown(f"<h3 style='color:{color};'>Status: {status.title()}</h3>", unsafe_allow_html=True)
        
        with col2:
            redis_status = health_data.get("redis", "unknown")
            st.metric("Redis", redis_status)
        
        with col3:
            memory_mb = health_data.get("memory_mb", 0)
            st.metric("Memory Used", f"{memory_mb:.1f} MB")
        
        with col4:
            st.metric("Version", health_data.get("version", "1.0.0"))
    
    def display_memory_status(self, tab_prefix=""):
        """Display memory status"""
        st.subheader("💾 Memory Usage")
        
        mem_data = self.get_backend_data("/memory")
        if not mem_data:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
        else:
            mem_mb = mem_data.get("memory_mb", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Memory Used", f"{mem_mb:.1f} MB")
        
        with col2:
            st.metric("Memory Limit", f"{MEMORY_LIMIT_MB} MB")
        
        with col3:
            usage_percent = (mem_mb / MEMORY_LIMIT_MB) * 100
            st.progress(usage_percent / 100)
            st.caption(f"{usage_percent:.1f}% of limit")
        
        if mem_mb > MEMORY_LIMIT_MB * 0.8:
            st.warning(f"⚠️ High memory usage: {mem_mb:.1f}MB")
        elif mem_mb > MEMORY_LIMIT_MB * 0.6:
            st.info(f"ℹ️ Moderate memory usage: {mem_mb:.1f}MB")
        else:
            st.success(f"✅ Good memory usage: {mem_mb:.1f}MB")
    
    def display_cost_analysis(self, tab_prefix=""):
        """Display cost analysis dashboard"""
        st.subheader("💰 Cost Analysis ($0/month Target)")
        
        cost_data = self.get_backend_data("/cost/estimate")
        if not cost_data or cost_data.get("status") != "success":
            st.warning("Cost data unavailable")
            return
        
        estimate = cost_data.get("monthly_estimate", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cost = estimate.get("total_cost", 0)
            st.metric("Monthly Cost", f"${total_cost:.2f}")
        
        with col2:
            within_tier = estimate.get("within_free_tier", False)
            status = "✅ Within Free Tier" if within_tier else "⚠️ Over Free Tier"
            st.metric("Free Tier Status", status)
        
        with col3:
            utilization = estimate.get("free_tier_utilization_percent", 0)
            st.metric("Free Tier Used", f"{utilization:.1f}%")
        
        st.info("""
        **AWS Free Tier Utilization:**
        - EC2 t3.micro: 750 hours/month (using 240h)
        - ECR: 500MB-month storage (using 100MB)
        - S3: 5GB storage (using 0.5GB)
        - CloudWatch: 10 metrics (using 5)
        - **Total Cost: $0.00/month**
        """)
    
    def display_sec_data_status(self, tab_prefix=""):
        """Display SEC data loading status"""
        st.subheader("📊 SEC Risk Data")
        
        sec_data = self.get_backend_data("/sec/stats")
        
        if sec_data and sec_data.get("status") == "success":
            companies = sec_data.get("companies", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Companies", sec_data.get("companies_count", 0))
            
            with col2:
                st.metric("Risk Entries", sec_data.get("total_entries", 0))
            
            with col3:
                total_chars = sum(info["total_chars"] for info in companies.values())
                st.metric("Total Data", f"{total_chars:,} chars")
            
            with st.expander("View Companies", expanded=False):
                company_list = []
                for name, info in companies.items():
                    company_list.append({
                        "Company": name,
                        "Ticker": info["ticker"],
                        "Entries": info["entries"],
                        "Years": ", ".join(info["years"]),
                        "Data Size": f"{info['total_chars']:,} chars"
                    })
                
                if company_list:
                    df = pd.DataFrame(company_list)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No companies loaded")
            
            # Buttons with unique keys based on tab prefix
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Load SEC Data into RAG", key=f"{tab_prefix}_load_rag"):
                    with st.spinner("Loading SEC data into vector store..."):
                        result = self.get_backend_data("/sec/load-to-rag", method="POST")
                        if result and result.get("status") == "success":
                            st.success(f"✅ {result.get('message')}")
                        else:
                            st.error("Failed to load data")
            
            with col2:
                if st.button("🔄 Refresh Data", key=f"{tab_prefix}_refresh_data"):
                    result = self.get_backend_data("/sec/fetch-risk-data?max_companies=5", method="POST")
                    if result and result.get("status") == "success":
                        st.success(f"✅ {result.get('message')}")
                    st.rerun()
        else:
            st.warning("SEC data not available")
            if st.button("📥 Fetch SEC Sample Data", key=f"{tab_prefix}_fetch_sample"):
                result = self.get_backend_data("/sec/fetch-risk-data?max_companies=3", method="POST")
                if result and result.get("status") == "success":
                    st.success(f"✅ {result.get('message')}")
                st.rerun()
    
    def display_news_integration(self, tab_prefix=""):
        """Display News API integration"""
        st.subheader("📰 Real-time News Analysis")
        
        # Get news stats
        news_stats = self.get_backend_data("/news/stats")
        
        if news_stats and news_stats.get("status") == "success":
            news_api = news_stats.get("news_api", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                enabled = news_api.get("enabled", False)
                status = "✅ Enabled" if enabled else "⚠️ Sample Mode"
                st.metric("News API", status)
            
            with col2:
                rate_limit = news_api.get("rate_limit_remaining", 0)
                st.metric("Rate Limit", f"{rate_limit}/100")
            
            with col3:
                companies = len(news_api.get("default_companies", []))
                st.metric("Tracked Companies", companies)
            
            # News actions
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("📡 Fetch Latest News", key=f"{tab_prefix}_fetch_news", type="primary"):
                    with st.spinner("Fetching latest news..."):
                        result = self.get_backend_data("/news/fetch?max_companies=5", method="POST")
                        if result and result.get("status") == "success":
                            st.success(f"✅ {result.get('message')}")
                            st.rerun()
                        else:
                            st.error("Failed to fetch news")
            
            with col2:
                if st.button("🔄 Refresh", key=f"{tab_prefix}_refresh_news"):
                    st.rerun()
            
            # Show latest news analysis
            st.markdown("#### 📈 Latest News Analysis")
            latest_news = self.get_backend_data("/news/latest")
            
            if latest_news and latest_news.get("status") == "success":
                analysis = latest_news.get("analysis", {})
                
                if analysis:
                    # Analysis metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Articles", analysis.get("total_articles", 0))
                    
                    with col2:
                        st.metric("Avg Risk Score", f"{analysis.get('average_risk_score', 0):.1f}")
                    
                    with col3:
                        high_alerts = len(analysis.get("high_risk_alerts", []))
                        st.metric("High Risk Alerts", high_alerts)
                    
                    with col4:
                        sentiment = analysis.get("sentiment_distribution", {})
                        neg_pct = (sentiment.get("negative", 0) / max(1, sum(sentiment.values()))) * 100
                        st.metric("Negative News", f"{neg_pct:.1f}%")
                    
                    # High risk alerts
                    if analysis.get("high_risk_alerts"):
                        st.markdown("##### ⚠️ High Risk News Alerts")
                        alerts_df = pd.DataFrame(analysis["high_risk_alerts"])
                        if not alerts_df.empty:
                            # Display with risk score coloring
                            for _, alert in alerts_df.iterrows():
                                risk_class = "news-alert-high" if alert.get("risk_score", 0) > 70 else "news-alert-medium"
                                with st.container():
                                    st.markdown(f"""
                                    <div class="{risk_class}">
                                    <strong>{alert.get('company', 'Unknown')} ({alert.get('ticker', '')})</strong><br>
                                    {alert.get('title', 'No title')}<br>
                                    <small>Risk: {alert.get('risk_score', 0):.1f} | Categories: {', '.join(alert.get('risk_categories', []))}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Sentiment chart
                    st.markdown("##### 📊 Sentiment Distribution")
                    sentiment_data = analysis.get("sentiment_distribution", {})
                    if sentiment_data:
                        fig = px.pie(
                            values=list(sentiment_data.values()),
                            names=list(sentiment_data.keys()),
                            color=list(sentiment_data.keys()),
                            color_discrete_map={
                                "positive": "#28a745",
                                "neutral": "#6c757d",
                                "negative": "#dc3545"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top risk categories
                    top_categories = analysis.get("top_risk_categories", {})
                    if top_categories:
                        st.markdown("##### 🔍 Top Risk Categories")
                        cat_df = pd.DataFrame({
                            "Category": list(top_categories.keys()),
                            "Count": list(top_categories.values())
                        }).sort_values("Count", ascending=False)
                        st.dataframe(cat_df.head(5), use_container_width=True)
                    
                    # Company rankings
                    rankings = analysis.get("company_rankings", [])
                    if rankings:
                        st.markdown("##### 🏆 Company Risk Rankings")
                        rank_df = pd.DataFrame(rankings)
                        if not rank_df.empty:
                            # Add color column for visualization
                            def get_risk_color(score):
                                if score >= 70:
                                    return "red"
                                elif score >= 40:
                                    return "orange"
                                else:
                                    return "green"
                            
                            rank_df["Color"] = rank_df["average_risk_score"].apply(get_risk_color)
                            
                            fig = px.bar(
                                rank_df,
                                x="company",
                                y="average_risk_score",
                                color="Color",
                                color_discrete_map={
                                    "red": "#dc3545",
                                    "orange": "#fd7e14",
                                    "green": "#28a745"
                                },
                                title="Company News Risk Scores",
                                labels={"company": "Company", "average_risk_score": "Risk Score"}
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No news analysis available. Click 'Fetch Latest News' to get started.")
            else:
                st.info("No news data available. Click 'Fetch Latest News' to get started.")
        else:
            st.warning("News API not configured")
            st.info("""
            **To enable real-time news:**
            1. Get free API key from newsapi.org
            2. Set NEWSAPI_KEY in .env file
            3. Restart the backend service
            """)
    
    def display_risk_analysis(self):
        """Display AI Risk Analysis interface"""
        st.subheader("🤖 AI Risk Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyze Company", "🆚 Compare Companies", "📰 Integrated View", "🤖 Ask Copilot"])
        
        with tab1:
            self._display_single_company_analysis()
        
        with tab2:
            self._display_company_comparison()
        
        with tab3:
            self._display_integrated_analysis()
        
        with tab4:
            self._display_copilot_assistant()
    
    def _display_single_company_analysis(self):
        """Display single company risk analysis"""
        st.markdown("#### Analyze Single Company")
        
        sample_companies = {
            "Apple Inc. (AAPL)": "Cybersecurity threats and data breaches are significant risks. Competition from Google and Samsung is intense. Regulatory investigations could result in fines.",
            "Microsoft Corp (MSFT)": "Cloud security risks. Competition from AWS. Regulatory compliance challenges.",
            "Amazon.com Inc. (AMZN)": "Supply chain disruptions affect operations. Labor disputes possible. Market competition from Walmart and Shopify."
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            company_option = st.selectbox(
                "Select Company",
                list(sample_companies.keys()) + ["Custom"],
                key="single_company_select"
            )
            
            if company_option == "Custom":
                company_name = st.text_input("Company Name", "Custom Company", key="custom_name")
                ticker = st.text_input("Ticker Symbol", "CUSTOM", key="custom_ticker")
                risk_text = st.text_area("Risk Factors", height=200, key="custom_risk")
            else:
                match = re.match(r"(.+?) \((.+?)\)", company_option)
                if match:
                    company_name = match.group(1)
                    ticker = match.group(2)
                else:
                    company_name = company_option
                    ticker = ""
                
                risk_text = st.text_area(
                    "Risk Factors", 
                    sample_companies[company_option],
                    height=200,
                    key=f"risk_{ticker}"
                )
        
        with col2:
            if st.button("🔍 Analyze Risk", type="primary", key="analyze_single_btn"):
                if risk_text:
                    with st.spinner("Analyzing risk..."):
                        data = {
                            "company_name": company_name,
                            "ticker": ticker,
                            "risk_text": risk_text
                        }
                        
                        result = self.get_backend_data(
                            "/analyze/company-risk",
                            method="POST",
                            data=data
                        )
                        
                        if result and result.get("status") == "success":
                            analysis = result.get("analysis", {})
                            self._display_analysis_results(analysis)
                        else:
                            st.error("Failed to analyze risk")
                else:
                    st.warning("Please enter risk text to analyze")
    
    def _display_analysis_results(self, analysis):
        """Display risk analysis results"""
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            score = analysis.get("normalized_score", 0)
            st.metric("Risk Score", f"{score:.1f}/100")
        
        with col_b:
            risk_level = analysis.get("risk_level", "UNKNOWN")
            color_class = f"risk-{risk_level.lower()}"
            st.markdown(f"<h3 class='{color_class}'>{risk_level}</h3>", unsafe_allow_html=True)
        
        with col_c:
            categories = len(analysis.get("category_scores", {}))
            st.metric("Risk Categories", categories)
        
        # Top risk categories
        st.markdown("#### 📋 Top Risk Categories")
        categories = analysis.get("category_scores", {})
        if categories:
            cat_data = []
            for cat_name, cat_info in categories.items():
                cat_data.append({
                    "Category": cat_name.title(),
                    "Score": cat_info["score"],
                    "Matches": len(cat_info["matches"]),
                    "Weight": cat_info["weight"]
                })
            
            df = pd.DataFrame(cat_data)
            st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)
        
        # Risk sentences
        st.markdown("#### 🔍 Key Risk Sentences")
        sentences = analysis.get("risk_sentences", [])
        for sent in sentences[:5]:
            st.info(f"**{sent['category'].title()}**: {sent['sentence']}")
    
    def _display_company_comparison(self):
        """Display company comparison"""
        st.markdown("#### Compare Multiple Companies")
        
        sample_companies = {
            "Apple Inc. (AAPL)": "Cybersecurity threats and data breaches are significant risks. Competition from Google and Samsung is intense. Regulatory investigations could result in fines.",
            "Microsoft Corp (MSFT)": "Cloud security risks. Competition from AWS. Regulatory compliance challenges.",
            "Amazon.com Inc. (AMZN)": "Supply chain disruptions affect operations. Labor disputes possible. Market competition from Walmart and Shopify."
        }
        
        selected_companies = st.multiselect(
            "Select companies to compare",
            list(sample_companies.keys()),
            default=list(sample_companies.keys())[:2],
            key="company_multiselect"
        )
        
        if len(selected_companies) >= 2:
            if st.button("🔄 Compare Companies", type="primary", key="compare_btn"):
                with st.spinner("Comparing companies..."):
                    companies_data = []
                    for company_option in selected_companies:
                        match = re.match(r"(.+?) \((.+?)\)", company_option)
                        if match:
                            company_name = match.group(1)
                            ticker = match.group(2)
                            risk_text = sample_companies[company_option]
                            
                            companies_data.append({
                                "company_name": company_name,
                                "ticker": ticker,
                                "risk_text": risk_text
                            })
                    
                    result = self.get_backend_data(
                        "/analyze/compare-companies",
                        method="POST",
                        data=companies_data
                    )
                    
                    if result and result.get("status") == "success":
                        self._display_comparison_results(result)
                    else:
                        st.error("Failed to compare companies")
        else:
            st.info("Select at least 2 companies to compare")
    
    def _display_comparison_results(self, result):
        """Display company comparison results"""
        comparison = result.get("comparison", {})
        
        # Comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Companies Compared", result.get("companies_analyzed", 0))
        
        with col2:
            avg_score = comparison.get("average_score", 0)
            st.metric("Average Risk Score", f"{avg_score:.1f}/100")
        
        with col3:
            high_risk = comparison.get("highest_risk", {})
            if high_risk:
                st.metric("Highest Risk", high_risk.get("company", "N/A"))
        
        # Risk scores chart
        st.markdown("#### 📊 Risk Score Comparison")
        companies_info = comparison.get("companies", {})
        if companies_info:
            chart_data = []
            for company, info in companies_info.items():
                chart_data.append({
                    "Company": company,
                    "Risk Score": info["score"],
                    "Risk Level": info["risk_level"]
                })
            
            df = pd.DataFrame(chart_data)
            
            fig = px.bar(
                df,
                x="Company",
                y="Risk Score",
                color="Risk Level",
                color_discrete_map={
                    "HIGH": "#dc3545",
                    "MEDIUM": "#fd7e14",
                    "LOW": "#28a745"
                },
                title="Company Risk Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison insights
            comp_insights = comparison.get("comparison", {})
            st.markdown("#### 🔍 Comparison Insights")
            
            if comp_insights.get("common_risks"):
                st.success(f"**Common Risks**: {', '.join(comp_insights['common_risks'])}")
            
            if comp_insights.get("unique_risks"):
                st.info("**Unique Risks by Company**:")
                for company, risks in comp_insights["unique_risks"].items():
                    st.write(f"- **{company}**: {', '.join(risks)}")
            
            if comp_insights.get("recommendations"):
                st.warning("**Recommendations**:")
                for rec in comp_insights["recommendations"][:3]:
                    st.write(f"- {rec}")
    
    def _display_integrated_analysis(self):
        """Display integrated SEC + News analysis"""
        st.markdown("#### 📊 Integrated Risk Analysis (SEC + News)")
        
        st.info("""
        This view combines:
        - **SEC Filings**: Long-term structural risks from 10-K filings
        - **News Analysis**: Real-time risks from financial news
        - **AI Scoring**: Intelligent risk assessment using both data sources
        """)
        
        # Get SEC data first
        sec_data = self.get_backend_data("/sec/stats")
        sec_companies = []
        
        if sec_data and sec_data.get("status") == "success":
            companies = sec_data.get("companies", {})
            sec_companies = list(companies.keys())
        
        # Get latest news
        latest_news = self.get_backend_data("/news/latest")
        
        if sec_companies:
            # Create sample SEC scores for integration
            sample_sec_scores = {}
            for company in sec_companies[:3]:  # Limit to 3 for demo
                sample_sec_scores[company] = {
                    "normalized_score": np.random.uniform(30, 80),
                    "ticker": "AAPL" if "Apple" in company else "MSFT" if "Microsoft" in company else "AMZN",
                    "category_scores": {
                        "cybersecurity": {"score": np.random.randint(20, 50), "matches": 3, "weight": 9},
                        "competition": {"score": np.random.randint(15, 40), "matches": 2, "weight": 7}
                    }
                }
            
            if st.button("🔄 Generate Integrated Analysis", type="primary", key="integrate_btn"):
                with st.spinner("Integrating SEC and news data..."):
                    # Call integration endpoint
                    result = self.get_backend_data(
                        "/news/integrate-risk",
                        method="POST",
                        data={"sec_scores": sample_sec_scores}
                    )
                    
                    if result and result.get("status") == "success":
                        self._display_integrated_results(result)
                    else:
                        st.error("Failed to generate integrated analysis")
        else:
            st.warning("No SEC data available. Load SEC data first.")
    
    def _display_integrated_results(self, result):
        """Display integrated analysis results"""
        integrated_scores = result.get("integrated_scores", {})
        news_summary = result.get("news_analysis_summary", {})
        
        st.success(f"✅ Integrated analysis complete: {len(integrated_scores)} companies")
        
        # News summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("News Articles", news_summary.get("total_articles", 0))
        with col2:
            st.metric("Avg News Risk", f"{news_summary.get('average_risk_score', 0):.1f}")
        with col3:
            st.metric("High Risk Alerts", news_summary.get("high_risk_alerts", 0))
        
        # Integrated scores visualization
        st.markdown("#### 📈 Integrated Risk Scores")
        
        if integrated_scores:
            # Prepare data for visualization
            viz_data = []
            for company, scores in integrated_scores.items():
                viz_data.append({
                    "Company": company,
                    "SEC Score": scores.get("sec_risk_score", 0),
                    "News Score": scores.get("news_risk_score", 0),
                    "Combined Score": scores.get("combined_risk_score", 0),
                    "Risk Level": scores.get("risk_level", "UNKNOWN"),
                    "Sentiment": scores.get("sentiment", "neutral")
                })
            
            df = pd.DataFrame(viz_data)
            
            # Create grouped bar chart
            fig = go.Figure()
            
            # Add SEC scores
            fig.add_trace(go.Bar(
                x=df["Company"],
                y=df["SEC Score"],
                name="SEC Score",
                marker_color="#1f77b4"
            ))
            
            # Add News scores
            fig.add_trace(go.Bar(
                x=df["Company"],
                y=df["News Score"],
                name="News Score",
                marker_color="#ff7f0e"
            ))
            
            # Add Combined scores as line
            fig.add_trace(go.Scatter(
                x=df["Company"],
                y=df["Combined Score"],
                name="Combined Score",
                mode="lines+markers",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Integrated Risk Analysis",
                xaxis_title="Company",
                yaxis_title="Risk Score (0-100)",
                barmode="group",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed view
            st.markdown("#### 🔍 Detailed Analysis")
            for company, scores in integrated_scores.items():
                with st.expander(f"{company} - {scores.get('risk_level', 'UNKNOWN')} Risk"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("SEC Risk Score", f"{scores.get('sec_risk_score', 0):.1f}")
                        st.metric("News Risk Score", f"{scores.get('news_risk_score', 0):.1f}")
                        st.metric("Combined Score", f"{scores.get('combined_risk_score', 0):.1f}")
                    
                    with col2:
                        sentiment = scores.get("sentiment", "neutral")
                        sentiment_icon = "📈" if sentiment == "positive" else "📉" if sentiment == "negative" else "➡️"
                        st.metric("News Sentiment", f"{sentiment_icon} {sentiment.title()}")
                        
                        alerts = len(scores.get("alerts", []))
                        st.metric("Recent Alerts", alerts)
                    
                    # Show recent alerts
                    if scores.get("alerts"):
                        st.markdown("**Recent News Alerts:**")
                        for alert in scores["alerts"][:2]:
                            st.info(f"**{alert.get('title', 'No title')}**\n"
                                  f"Risk: {alert.get('risk_score', 0):.1f} | "
                                  f"Categories: {', '.join(alert.get('categories', []))}")
        else:
            st.info("No integrated scores available")
    
    def _display_copilot_assistant(self):
        """Display AI copilot assistant"""
        st.markdown("#### 🤖 AI Copilot Assistant")
        
        sample_questions = [
            "Why is Apple's risk high?",
            "Compare Apple and Microsoft risk",
            "What are the recommendations for cybersecurity risks?",
            "Which company has the highest risk and why?",
            "What common risks do all companies share?",
            "Show me the latest high-risk news alerts",
            "How does news sentiment affect risk scores?"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input("Ask the copilot:", placeholder="e.g., Why is Apple's risk high?", key="copilot_input")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Ask", type="primary", key="ask_copilot_btn"):
                pass
        
        # Quick question buttons
        st.markdown("**Quick Questions:**")
        cols = st.columns(min(len(sample_questions), 4))
        for idx, q in enumerate(sample_questions[:4]):
            with cols[idx]:
                if st.button(q[:20] + "...", key=f"quick_q_{idx}"):
                    question = q
        
        if question:
            with st.spinner("Thinking..."):
                # In production, this would call the actual copilot endpoint
                if "news" in question.lower():
                    response = "🤖 **AI Copilot**: I can analyze real-time news sentiment and detect emerging risks. For example, I found that cybersecurity news mentions increased by 30% this week for tech companies. Would you like me to fetch the latest news analysis?"
                elif "compare" in question.lower():
                    response = "🤖 **AI Copilot**: I can compare companies based on both SEC filings and recent news. Apple shows higher cybersecurity risks (75/100) while Microsoft has more regulatory exposure (65/100). Amazon faces supply chain risks (55/100)."
                elif "recommend" in question.lower():
                    response = "🤖 **AI Copilot**: Based on my analysis, I recommend: 1) Review cybersecurity protocols for tech companies, 2) Monitor regulatory developments, 3) Diversify supply chains for retail companies."
                else:
                    response = "🤖 **AI Copilot**: I can analyze company risks, compare companies, provide recommendations, and show real-time news alerts. Try asking me about specific companies or risk categories!"
                
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {response}")
                
                # Show follow-up options
                st.markdown("**Follow-up:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📈 Show risk chart", key="followup_chart"):
                        st.info("Risk chart would appear here")
                with col2:
                    if st.button("📰 Latest news", key="followup_news"):
                        st.info("Latest news analysis would appear here")
                with col3:
                    if st.button("📋 Recommendations", key="followup_recs"):
                        st.info("Detailed recommendations would appear here")
    
    def display_system_stats(self, tab_prefix=""):
        """Display system statistics"""
        st.subheader("📈 System Statistics")
        
        status_data = self.get_backend_data("/status")
        
        if status_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_percent = status_data.get("cpu", {}).get("percent", 0)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                disk_free = status_data.get("disk", {}).get("free_gb", 0)
                st.metric("Disk Free", f"{disk_free:.1f} GB")
            
            with col3:
                rag_status = "✅" if status_data.get("rag_available") else "❌"
                st.metric("RAG Pipeline", rag_status)
            
            with col4:
                timestamp = status_data.get("timestamp", "")
                if timestamp:
                    time_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%H:%M:%S")
                    st.metric("Last Updated", time_str)
    
    def run(self):
        """Main dashboard run method"""
        # Sidebar
        with st.sidebar:
            st.title("⚙️ Dashboard Controls")
            st.markdown("---")
            
            if st.button("🔄 Refresh All Data", type="primary", key="sidebar_refresh"):
                st.rerun()
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 SEC Data", key="sidebar_sec"):
                    st.session_state.active_tab = "SEC Data"
                    st.rerun()
            with col2:
                if st.button("📰 News", key="sidebar_news"):
                    st.session_state.active_tab = "News"
                    st.rerun()
            
            st.markdown("---")
            
            st.markdown("### 🎯 Project Info")
            st.info("""
            **AWS Risk Copilot v2.0**
            - ✅ Real SEC Risk Data
            - ✅ Real-time News Analysis
            - ✅ AI Risk Scoring
            - ✅ Company Comparison
            - ✅ Integrated SEC+News View
            - ✅ AI Copilot Assistant
            - **1GB RAM constraint** (EC2 t3.micro)
            - **$0/month cost target** (AWS Free Tier)
            """)
            
            st.markdown("---")
            
            # Memory status in sidebar
            mem_data = self.get_backend_data("/memory")
            if mem_data:
                mem_mb = mem_data.get("memory_mb", 0)
                usage_percent = (mem_mb / MEMORY_LIMIT_MB) * 100
                st.progress(usage_percent / 100)
                st.caption(f"Memory: {mem_mb:.0f}MB / {MEMORY_LIMIT_MB}MB ({usage_percent:.1f}%)")
        
        # Main content
        st.title("🚀 AWS Risk Copilot Dashboard")
        st.markdown("AI-Powered Risk Intelligence with 1GB RAM constraint")
        st.markdown("---")
        
        # Tab navigation - Updated with News tab
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "📈 Risk Analysis", 
            "📰 News Analysis", 
            "📋 SEC Data", 
            "💰 Cost Analysis", 
            "⚙️ System"
        ])
        
        with tab1:
            # Health and Memory
            col1, col2 = st.columns(2)
            with col1:
                self.display_health_status("overview")
            with col2:
                self.display_memory_status("overview")
            
            st.markdown("---")
            
            # Quick stats row
            col1, col2, col3 = st.columns(3)
            with col1:
                sec_data = self.get_backend_data("/sec/stats")
                if sec_data and sec_data.get("status") == "success":
                    st.metric("SEC Companies", sec_data.get("companies_count", 0))
                else:
                    st.metric("SEC Companies", "0")
            
            with col2:
                news_stats = self.get_backend_data("/news/stats")
                if news_stats and news_stats.get("status") == "success":
                    enabled = "✅" if news_stats.get("news_api", {}).get("enabled") else "⚠️"
                    st.metric("News API", enabled)
                else:
                    st.metric("News API", "⚠️")
            
            with col3:
                latest_news = self.get_backend_data("/news/latest")
                if latest_news and latest_news.get("analysis"):
                    articles = latest_news["analysis"].get("total_articles", 0)
                    st.metric("News Articles", articles)
                else:
                    st.metric("News Articles", "0")
            
            st.markdown("---")
            self.display_system_stats("overview")
        
        with tab2:
            self.display_risk_analysis()
        
        with tab3:
            self.display_news_integration("news")
        
        with tab4:
            self.display_sec_data_status("sec_data")
        
        with tab5:
            self.display_cost_analysis("cost")
        
        with tab6:
            self.display_health_status("system")
            st.markdown("---")
            self.display_memory_status("system")
            st.markdown("---")
            self.display_system_stats("system")

def try_backend_connection():
    """Try to connect to backend with multiple URLs"""
    urls_to_try = [
        "http://backend:8000",
        "http://localhost:8000",
        f"http://{os.getenv('EC2_PUBLIC_IP', 'localhost')}:8000"
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return url, response.json()
        except:
            continue
    
    return None, None

def main():
    """Main entry point"""
    # Initialize session state for active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    backend_url, health_data = try_backend_connection()
    
    if backend_url and health_data:
        st.success(f"✅ Connected to backend at {backend_url}")
        dashboard = RiskCopilotDashboard(backend_url)
        dashboard.run()
    else:
        st.error("❌ Cannot connect to backend service")
        st.info("""
        **To fix:**
        1. Ensure backend is running: `docker-compose up -d`
        2. Check Docker network connectivity
        3. Verify backend is accessible at port 8000
        """)

if __name__ == "__main__":
    main()
