"""
Streamlit Dashboard for AWS Risk Copilot
Memory-aware dashboard optimized for 1GB RAM constraint
"""
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time
import psutil
import sys
import os

# Configuration - Try Docker service name first, fallback to localhost
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
LOCAL_BACKUP_URL = 'http://localhost:8000'

# Page configuration
st.set_page_config(
    page_title="AWS Risk Copilot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for memory awareness
st.markdown("""
<style>
    .memory-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .memory-good {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MEMORY_LIMIT_MB = 1024  # 1GB RAM constraint

class MemoryAwareDashboard:
    """Dashboard with memory awareness for 1GB RAM constraint"""
    
    def __init__(self, backend_url):
        self.backend_url = backend_url
        self.session = requests.Session()
        self.session.timeout = 5
        
    def get_backend_data(self, endpoint):
        """Get data from backend"""
        try:
            response = self.session.get(f"{self.backend_url}{endpoint}")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def display_health_status(self):
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
            ws_status = health_data.get("websocket", "unknown")
            st.metric("WebSocket", ws_status)
        
        with col4:
            st.metric("Version", health_data.get("version", "1.0.0"))
    
    def display_memory_status(self):
        """Display memory status"""
        st.subheader("💾 Memory Usage")
        
        mem_data = self.get_backend_data("/memory")
        if not mem_data:
            # Fallback to local memory check
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            mem_percent = process.memory_percent()
        else:
            mem_mb = mem_data.get("memory_mb", 0)
            mem_percent = mem_data.get("memory_percent", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Memory Used", f"{mem_mb:.1f} MB")
        
        with col2:
            st.metric("Memory Limit", f"{MEMORY_LIMIT_MB} MB")
        
        with col3:
            usage_percent = (mem_mb / MEMORY_LIMIT_MB) * 100
            st.progress(usage_percent / 100)
            st.caption(f"{usage_percent:.1f}% of limit")
        
        # Memory warning
        if mem_mb > MEMORY_LIMIT_MB * 0.8:
            st.warning(f"⚠️ High memory usage: {mem_mb:.1f}MB")
        elif mem_mb > MEMORY_LIMIT_MB * 0.6:
            st.info(f"ℹ️ Moderate memory usage: {mem_mb:.1f}MB")
        else:
            st.success(f"✅ Good memory usage: {mem_mb:.1f}MB")
    
    def display_cost_analysis(self):
        """Display cost analysis dashboard"""
        st.subheader("💰 Cost Analysis ($0/month Target)")
        
        cost_data = self.get_backend_data("/cost/estimate")
        if not cost_data or cost_data.get("status") != "success":
            st.warning("Cost data unavailable")
            return
        
        estimate = cost_data.get("monthly_estimate", {})
        
        # Overall cost
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
    
    def display_realtime_alerts(self):
        """Display real-time alerts"""
        st.subheader("🚨 Real-time Alerts")
        
        alerts_data = self.get_backend_data("/alerts/recent?count=5")
        
        if alerts_data and alerts_data.get("status") == "success":
            alerts = alerts_data.get("alerts", [])
            
            if alerts:
                for alert in alerts:
                    alert_data = alert.get("data", {})
                    with st.expander(f"{alert_data.get('type', 'Alert')} - {alert_data.get('severity', 'info').title()}"):
                        st.write(f"**Message:** {alert_data.get('message', '')}")
                        st.write(f"**Time:** {alert.get('timestamp', '')}")
            else:
                st.info("No alerts in the stream")
        else:
            st.warning("Could not load alerts")
    
    def display_circuit_breakers(self):
        """Display circuit breaker status"""
        st.subheader("⚡ Circuit Breakers")
        
        cb_data = self.get_backend_data("/circuit-breakers/stats")
        
        if cb_data and cb_data.get("status") == "success":
            circuits = cb_data.get("circuit_breakers", {})
            
            cols = st.columns(len(circuits))
            for idx, (service, data) in enumerate(circuits.items()):
                with cols[idx]:
                    state = data.get("circuit_state", "unknown")
                    color = "green" if state == "closed" else "red" if state == "open" else "orange"
                    st.markdown(f"<h4 style='color:{color};'>{service.title()}</h4>", unsafe_allow_html=True)
                    st.metric("State", state.title())
        else:
            st.warning("Circuit breaker data unavailable")
    
    def run(self):
        """Main dashboard run method"""
        # Sidebar
        with st.sidebar:
            st.title("⚙️ Dashboard Controls")
            st.markdown("---")
            
            st.markdown("### 🎯 Project Info")
            st.info("""
            **AWS Risk Copilot**
            - 1GB RAM constraint (EC2 t3.micro)
            - $0/month cost target
            - Real-time risk analysis
            - Memory-efficient RAG pipeline
            """)
        
        # Main content
        st.title("🚀 AWS Risk Copilot Dashboard")
        st.markdown("Real-time monitoring with 1GB RAM constraint")
        st.markdown("---")
        
        # Health and Memory
        col1, col2 = st.columns(2)
        with col1:
            self.display_health_status()
        with col2:
            self.display_memory_status()
        
        st.markdown("---")
        
        # Cost Analysis
        self.display_cost_analysis()
        
        st.markdown("---")
        
        # Alerts and Circuit Breakers
        col1, col2 = st.columns(2)
        with col1:
            self.display_realtime_alerts()
        with col2:
            self.display_circuit_breakers()

def try_backend_connection():
    """Try to connect to backend with multiple URLs"""
    urls_to_try = [
        "http://backend:8000",  # Docker service name
        "http://localhost:8000",  # Localhost for development
        f"http://{os.getenv('EC2_PUBLIC_IP', 'localhost')}:8000"  # EC2 public IP
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
    st.title("AWS Risk Copilot Dashboard")
    
    # Try to connect to backend
    backend_url, health_data = try_backend_connection()
    
    if backend_url and health_data:
        st.success(f"✅ Connected to backend at {backend_url}")
        
        # Initialize and run dashboard
        dashboard = MemoryAwareDashboard(backend_url)
        dashboard.run()
        
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.rerun()
    else:
        st.error("❌ Cannot connect to backend service")
        st.info("""
        **Possible solutions:**
        1. Ensure backend is running: `docker-compose up -d`
        2. Check Docker network connectivity
        3. Verify backend is accessible at port 8000
        
        **Current configuration:**
        - Backend URL attempts: backend:8000, localhost:8000
        - Docker services should be in same network
        
        **To run:**
        ```bash
        # Start all services
        docker-compose -f docker-compose.full.yml up -d
        
        # Check logs
        docker logs risk-copilot-backend
        ```
        """)

if __name__ == "__main__":
    main()
