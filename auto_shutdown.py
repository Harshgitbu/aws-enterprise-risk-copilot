#!/usr/bin/env python3
"""
Auto-shutdown script for AWS Risk Copilot
Shuts down EC2 instance during idle periods to save cost
Optimized for Free Tier (750 hours/month = ~24 hours/day)
"""
import os
import time
import logging
import requests
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoShutdownManager:
    """
    Manages auto-shutdown of EC2 instance during idle periods
    Goal: Stay within Free Tier (750 hours/month = 24h/day)
    """
    
    def __init__(self, 
                 idle_threshold_minutes: int = 30,
                 check_interval_minutes: int = 5,
                 daily_uptime_hours: int = 8):  # 8 hours/day = 240h/month (well under 750h)
        
        self.idle_threshold = idle_threshold_minutes * 60  # Convert to seconds
        self.check_interval = check_interval_minutes * 60
        self.daily_uptime = daily_uptime_hours * 3600
        self.last_activity = time.time()
        self.start_time = time.time()
        self.shutdown_enabled = True
        
        logger.info(f"AutoShutdownManager initialized")
        logger.info(f"  Idle threshold: {idle_threshold_minutes} minutes")
        logger.info(f"  Daily uptime target: {daily_uptime_hours} hours")
        logger.info(f"  Monthly target: {daily_uptime_hours * 30}h (Free Tier: 750h)")
    
    def check_service_activity(self) -> bool:
        """Check if service is active (requests being made)"""
        try:
            # Check health endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check recent activity via other endpoints
                stats_response = requests.get("http://localhost:8000/alerts/stats", timeout=5)
                
                # Update last activity time
                self.last_activity = time.time()
                
                # Log activity
                active_connections = data.get('websocket_connections', 0)
                if active_connections > 0:
                    logger.debug(f"Active WebSocket connections: {active_connections}")
                
                return True
        except requests.exceptions.RequestException:
            # Service might be starting or temporarily unavailable
            pass
        
        return False
    
    def check_uptime_limit(self) -> bool:
        """Check if we've reached daily uptime limit"""
        current_uptime = time.time() - self.start_time
        if current_uptime > self.daily_uptime:
            logger.warning(f"Daily uptime limit reached: {current_uptime/3600:.1f}h/{self.daily_uptime/3600:.1f}h")
            return True
        return False
    
    def check_idle_time(self) -> bool:
        """Check if instance has been idle beyond threshold"""
        idle_time = time.time() - self.last_activity
        if idle_time > self.idle_threshold:
            logger.warning(f"Instance idle for {idle_time/60:.1f} minutes (threshold: {self.idle_threshold/60:.1f}m)")
            return True
        return False
    
    def should_shutdown(self) -> bool:
        """Determine if instance should shutdown"""
        if not self.shutdown_enabled:
            return False
        
        # Rule 1: Check uptime limit
        if self.check_uptime_limit():
            logger.info("Shutdown reason: Daily uptime limit reached")
            return True
        
        # Rule 2: Check idle time
        if self.check_idle_time():
            # Verify service is actually idle (not just health check failing)
            if not self.check_service_activity():
                logger.info("Shutdown reason: Extended idle period")
                return True
        
        return False
    
    def schedule_shutdown(self, delay_minutes: int = 5):
        """Schedule instance shutdown"""
        if not self.should_shutdown():
            return False
        
        logger.warning(f"⚠️  Scheduling shutdown in {delay_minutes} minutes")
        
        # In production, you would:
        # 1. Send notification (SNS, email, Slack)
        # 2. Drain connections gracefully
        # 3. Stop services
        # 4. Shutdown instance
        
        # For demo, we'll just log and simulate
        shutdown_time = datetime.now() + timedelta(minutes=delay_minutes)
        logger.warning(f"Simulated shutdown scheduled for: {shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Real shutdown command (commented out for safety):
        # os.system(f"sudo shutdown -h +{delay_minutes}")
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-shutdown statistics"""
        current_time = time.time()
        idle_time = current_time - self.last_activity
        uptime = current_time - self.start_time
        
        return {
            "enabled": self.shutdown_enabled,
            "idle_time_seconds": idle_time,
            "idle_time_minutes": idle_time / 60,
            "uptime_hours": uptime / 3600,
            "daily_uptime_limit_hours": self.daily_uptime / 3600,
            "idle_threshold_minutes": self.idle_threshold / 60,
            "should_shutdown": self.should_shutdown(),
            "monthly_uptime_estimate_hours": (uptime / (24*3600)) * 30 * (self.daily_uptime/3600)
        }
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Starting auto-shutdown monitor...")
        
        while True:
            try:
                # Check service activity (updates last_activity)
                self.check_service_activity()
                
                # Check if shutdown needed
                if self.should_shutdown():
                    self.schedule_shutdown()
                
                # Log stats periodically
                stats = self.get_stats()
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"Auto-shutdown stats: {stats['idle_time_minutes']:.1f}m idle, {stats['uptime_hours']:.1f}h uptime")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Auto-shutdown monitor stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-shutdown monitor: {e}")
                time.sleep(60)

# Create systemd service file template
cat > risk-copilot-autoshutdown.service << 'EOF'
[Unit]
Description=AWS Risk Copilot Auto-Shutdown Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/aws-enterprise-risk-copilot
ExecStart=/usr/bin/python3 /home/ubuntu/aws-enterprise-risk-copilot/auto_shutdown.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

# Safety: Don't restart if shutdown was intentional
RestartPreventExitStatus=0

[Install]
WantedBy=multi-user.target
