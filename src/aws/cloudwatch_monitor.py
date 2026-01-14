"""
CloudWatch monitoring for AWS Risk Copilot
Optimized for Free Tier (10 metrics, 1 million API requests free)
"""
import os
import time
import logging
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available, CloudWatch monitoring disabled")

logger = logging.getLogger(__name__)

class CloudWatchMonitor:
    """
    Lightweight CloudWatch monitoring for 1GB RAM constraint
    Only sends critical metrics to stay within Free Tier
    """
    
    def __init__(self, 
                 namespace: str = "RiskCopilot",
                 instance_id: str = "t3.micro",
                 region: str = "us-east-1"):
        
        self.namespace = namespace
        self.instance_id = instance_id
        self.region = region
        self.enabled = False
        self.client = None
        self.metrics_sent = 0
        
        # Free tier limits: 10 custom metrics, 1M API requests/month
        self.metrics_config = {
            "MemoryUsage": {"unit": "Percent", "interval": 300},  # Every 5 minutes
            "ContainerCount": {"unit": "Count", "interval": 600},  # Every 10 minutes
            "APICalls": {"unit": "Count", "interval": 60},  # Every minute (aggregated)
            "ErrorRate": {"unit": "Percent", "interval": 300},
            "CostEstimate": {"unit": "None", "interval": 3600}  # Hourly
        }
        
        # Initialize CloudWatch client
        if BOTO3_AVAILABLE:
            try:
                self.client = boto3.client('cloudwatch', region_name=region)
                self.enabled = True
                logger.info(f"CloudWatch monitor initialized for namespace: {namespace}")
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"CloudWatch not available: {e}")
                self.enabled = False
        else:
            logger.warning("boto3 not installed, CloudWatch disabled")
    
    def _should_send_metric(self, metric_name: str) -> bool:
        """Check if we should send metric based on interval"""
        if metric_name not in self.metrics_config:
            return False
        
        # Simple interval check (in production, use last_sent timestamp)
        # For demo, we'll send every time but log frequency
        interval = self.metrics_config[metric_name]["interval"]
        
        # Simulate interval check - in real implementation, track last_sent
        return True
    
    def send_memory_metric(self, memory_percent: float):
        """Send memory usage metric to CloudWatch"""
        if not self.enabled or not self._should_send_metric("MemoryUsage"):
            return False
        
        try:
            response = self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'MemoryUsage',
                        'Dimensions': [
                            {'Name': 'InstanceType', 'Value': self.instance_id},
                            {'Name': 'Service', 'Value': 'RiskCopilot'}
                        ],
                        'Value': memory_percent,
                        'Unit': 'Percent',
                        'Timestamp': datetime.utcnow(),
                        'StorageResolution': 60  # Standard resolution for lower cost
                    }
                ]
            )
            self.metrics_sent += 1
            logger.debug(f"Memory metric sent: {memory_percent}%")
            return True
        except Exception as e:
            logger.error(f"Failed to send memory metric: {e}")
            return False
    
    def send_container_metric(self, container_count: int):
        """Send container count metric"""
        if not self.enabled or not self._should_send_metric("ContainerCount"):
            return False
        
        try:
            response = self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'ContainerCount',
                        'Dimensions': [
                            {'Name': 'InstanceType', 'Value': self.instance_id}
                        ],
                        'Value': container_count,
                        'Unit': 'Count',
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
            self.metrics_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send container metric: {e}")
            return False
    
    def send_cost_estimate(self, estimated_cost: float):
        """Send cost estimate (simulated)"""
        if not self.enabled or not self._should_send_metric("CostEstimate"):
            return False
        
        try:
            response = self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'CostEstimate',
                        'Dimensions': [
                            {'Name': 'InstanceType', 'Value': self.instance_id},
                            {'Name': 'Service', 'Value': 'RiskCopilot'}
                        ],
                        'Value': estimated_cost,
                        'Unit': 'None',
                        'Timestamp': datetime.utcnow(),
                        'StatisticValues': {
                            'SampleCount': 1.0,
                            'Sum': estimated_cost,
                            'Minimum': estimated_cost,
                            'Maximum': estimated_cost
                        }
                    }
                ]
            )
            self.metrics_sent += 1
            logger.info(f"Cost estimate sent: ${estimated_cost:.4f}")
            return True
        except Exception as e:
            logger.error(f"Failed to send cost metric: {e}")
            return False
    
    def create_cost_alarm(self, threshold: float = 0.50):
        """Create CloudWatch alarm for cost threshold ($0.50)"""
        if not self.enabled:
            return False
        
        try:
            # Note: This requires appropriate IAM permissions
            alarm_name = f"RiskCopilot-Cost-Alarm-{self.instance_id}"
            
            response = self.client.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription='Alert when estimated cost exceeds $0.50',
                ActionsEnabled=True,
                OKActions=[],
                AlarmActions=[],  # Would be SNS topic ARN in production
                InsufficientDataActions=[],
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Statistic='Maximum',
                Dimensions=[
                    {'Name': 'Currency', 'Value': 'USD'},
                    {'Name': 'ServiceName', 'Value': 'AmazonEC2'}
                ],
                Period=21600,  # 6 hours (to reduce API calls)
                EvaluationPeriods=1,
                DatapointsToAlarm=1,
                Threshold=threshold,
                ComparisonOperator='GreaterThanThreshold',
                TreatMissingData='notBreaching'
            )
            
            logger.info(f"Cost alarm created: {alarm_name} (threshold: ${threshold})")
            return True
        except Exception as e:
            logger.warning(f"Could not create cost alarm (may need IAM permissions): {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CloudWatch monitor statistics"""
        return {
            "enabled": self.enabled,
            "metrics_sent": self.metrics_sent,
            "namespace": self.namespace,
            "instance_id": self.instance_id,
            "metrics_config": self.metrics_config
        }

# Global instance
cloudwatch_monitor = CloudWatchMonitor()
