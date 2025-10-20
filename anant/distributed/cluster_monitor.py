"""
Cluster Monitor - Real-time Monitoring and Alerting

Provides comprehensive monitoring, metrics collection, and alerting for
distributed cluster health, performance, and resource utilization.
"""

import asyncio
import time
import threading
import json
import statistics
import logging
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import uuid

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate of change


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"        # System failure or severe degradation
    WARNING = "warning"          # Performance degradation or issues
    INFO = "info"               # Informational alerts
    DEBUG = "debug"             # Debug-level information


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"           # Alert is currently firing
    RESOLVED = "resolved"       # Alert condition has been resolved
    SILENCED = "silenced"       # Alert has been manually silenced
    ACKNOWLEDGED = "acknowledged"  # Alert has been acknowledged


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'metric_type': self.metric_type.value
        }


@dataclass
class Alert:
    """System alert definition."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str              # Alert condition expression
    status: AlertStatus = AlertStatus.ACTIVE
    timestamp: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'name': self.name,
            'description': self.description,
            'severity': self.severity.value,
            'condition': self.condition,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'resolved_at': self.resolved_at,
            'acknowledged_at': self.acknowledged_at,
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass 
class MonitoringConfig:
    """Configuration for cluster monitoring."""
    collection_interval: float = 10.0      # Metric collection interval
    retention_hours: int = 24              # How long to keep metrics
    max_metrics_per_node: int = 10000      # Max metrics per node
    enable_alerting: bool = True           # Enable alerting system
    alert_evaluation_interval: float = 30.0  # Alert evaluation interval
    heartbeat_timeout: float = 60.0        # Node heartbeat timeout
    slow_query_threshold: float = 5.0      # Slow query alert threshold
    high_cpu_threshold: float = 80.0       # High CPU alert threshold
    high_memory_threshold: float = 85.0    # High memory alert threshold
    disk_space_threshold: float = 90.0     # Disk space alert threshold
    enable_performance_profiling: bool = False  # Enable detailed profiling


class MetricsCollector:
    """Collects and aggregates metrics from cluster nodes."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize metrics collector."""
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.node_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        self._lock = threading.RLock()
        
    def record_metric(self, metric: MetricPoint, node_id: Optional[str] = None):
        """Record a metric point."""
        with self._lock:
            # Store in global metrics
            metric_key = f"{metric.name}:{','.join(f'{k}={v}' for k, v in sorted(metric.labels.items()))}"
            self.metrics[metric_key].append(metric)
            
            # Store in node-specific metrics if node_id provided
            if node_id:
                self.node_metrics[node_id][metric_key].append(metric)
                
    def get_metric_history(self, metric_name: str, labels: Optional[Dict[str, str]] = None,
                          node_id: Optional[str] = None, 
                          duration_seconds: int = 3600) -> List[MetricPoint]:
        """Get metric history for specified duration."""
        with self._lock:
            label_str = ','.join(f'{k}={v}' for k, v in sorted((labels or {}).items()))
            metric_key = f"{metric_name}:{label_str}"
            
            # Choose metric source
            if node_id:
                metric_deque = self.node_metrics.get(node_id, {}).get(metric_key, deque())
            else:
                metric_deque = self.metrics.get(metric_key, deque())
                
            # Filter by time
            cutoff_time = time.time() - duration_seconds
            return [m for m in metric_deque if m.timestamp >= cutoff_time]
            
    def get_current_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None,
                         node_id: Optional[str] = None) -> Optional[Union[int, float]]:
        """Get the most recent value for a metric."""
        history = self.get_metric_history(metric_name, labels, node_id, 300)  # Last 5 minutes
        if history:
            return history[-1].value
        return None
        
    def calculate_rate(self, metric_name: str, labels: Optional[Dict[str, str]] = None,
                      node_id: Optional[str] = None, window_seconds: int = 300) -> Optional[float]:
        """Calculate rate of change for a counter metric."""
        history = self.get_metric_history(metric_name, labels, node_id, window_seconds)
        if len(history) < 2:
            return None
            
        first_point = history[0]
        last_point = history[-1]
        
        time_diff = last_point.timestamp - first_point.timestamp
        value_diff = last_point.value - first_point.value
        
        if time_diff > 0:
            return value_diff / time_diff
        return None
        
    def calculate_average(self, metric_name: str, labels: Optional[Dict[str, str]] = None,
                         node_id: Optional[str] = None, window_seconds: int = 300) -> Optional[float]:
        """Calculate average value over time window."""
        history = self.get_metric_history(metric_name, labels, node_id, window_seconds)
        if not history:
            return None
        return statistics.mean(point.value for point in history)
        
    def get_collector_stats(self) -> Dict[str, Any]:
        """Get metrics collector statistics."""
        with self._lock:
            total_metrics = sum(len(deque_obj) for deque_obj in self.metrics.values())
            node_metric_counts = {
                node_id: sum(len(deque_obj) for deque_obj in metrics.values())
                for node_id, metrics in self.node_metrics.items()
            }
            
            return {
                'total_metrics': total_metrics,
                'unique_metric_names': len(self.metrics),
                'monitored_nodes': len(self.node_metrics),
                'node_metric_counts': node_metric_counts,
                'retention_hours': self.config.retention_hours
            }


class AlertManager:
    """Manages alerts and alert rules."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize alert manager."""
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
    def _setup_default_alert_rules(self):
        """Setup default system alert rules."""
        self.alert_rules = {
            'high_cpu': {
                'name': 'High CPU Usage',
                'condition': lambda metrics: self._check_threshold(metrics, 'cpu_percent', self.config.high_cpu_threshold),
                'severity': AlertSeverity.WARNING,
                'description': 'CPU usage is above threshold'
            },
            'high_memory': {
                'name': 'High Memory Usage', 
                'condition': lambda metrics: self._check_threshold(metrics, 'memory_percent', self.config.high_memory_threshold),
                'severity': AlertSeverity.WARNING,
                'description': 'Memory usage is above threshold'
            },
            'disk_space': {
                'name': 'Low Disk Space',
                'condition': lambda metrics: self._check_threshold(metrics, 'disk_percent', self.config.disk_space_threshold),
                'severity': AlertSeverity.CRITICAL,
                'description': 'Disk space is critically low'
            },
            'node_down': {
                'name': 'Node Down',
                'condition': lambda metrics: self._check_node_heartbeat(metrics),
                'severity': AlertSeverity.CRITICAL,
                'description': 'Node is not responding to heartbeat'
            },
            'slow_queries': {
                'name': 'Slow Queries',
                'condition': lambda metrics: self._check_threshold(metrics, 'query_duration', self.config.slow_query_threshold),
                'severity': AlertSeverity.WARNING,
                'description': 'Query response time is above threshold'
            }
        }
        
    def _check_threshold(self, metrics: Dict[str, Any], metric_name: str, threshold: float) -> bool:
        """Check if metric exceeds threshold."""
        value = metrics.get(metric_name)
        return value is not None and value > threshold
        
    def _check_node_heartbeat(self, metrics: Dict[str, Any]) -> bool:
        """Check if node heartbeat is stale."""
        last_heartbeat = metrics.get('last_heartbeat', 0)
        return time.time() - last_heartbeat > self.config.heartbeat_timeout
        
    def evaluate_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        new_alerts = []
        
        with self._lock:
            # Get current metrics for evaluation
            current_metrics = self._collect_alert_metrics(metrics_collector)
            
            for rule_id, rule in self.alert_rules.items():
                try:
                    # Check if condition is met
                    condition_met = rule['condition'](current_metrics)
                    
                    if condition_met:
                        # Check if alert already exists
                        existing_alert = self._find_active_alert(rule['name'])
                        
                        if not existing_alert:
                            # Create new alert
                            alert = Alert(
                                alert_id=str(uuid.uuid4()),
                                name=rule['name'],
                                description=rule['description'],
                                severity=rule['severity'],
                                condition=rule_id,
                                labels={'rule_id': rule_id}
                            )
                            
                            self.active_alerts[alert.alert_id] = alert
                            self.alert_history.append(alert)
                            new_alerts.append(alert)
                            
                            logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
                            
                    else:
                        # Check if we should resolve any active alerts
                        existing_alert = self._find_active_alert(rule['name'])
                        if existing_alert:
                            self._resolve_alert(existing_alert.alert_id)
                            
                except Exception as e:
                    logger.error(f"Alert evaluation failed for rule {rule_id}: {e}")
                    
        return new_alerts
        
    def _collect_alert_metrics(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Collect current metrics for alert evaluation."""
        metrics = {}
        
        # Collect key metrics for alert evaluation
        metric_names = ['cpu_percent', 'memory_percent', 'disk_percent', 'query_duration', 'last_heartbeat']
        
        for metric_name in metric_names:
            value = metrics_collector.get_current_value(metric_name)
            if value is not None:
                metrics[metric_name] = value
                
        return metrics
        
    def _find_active_alert(self, alert_name: str) -> Optional[Alert]:
        """Find active alert by name."""
        for alert in self.active_alerts.values():
            if alert.name == alert_name and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                logger.info(f"Alert acknowledged: {alert.name}")
                return True
        return False
        
    def _resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.name}")
                return True
        return False
        
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
            
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        with self._lock:
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
                
            return {
                'active_alerts': len(self.active_alerts),
                'total_alerts_history': len(self.alert_history),
                'alerts_by_severity': dict(severity_counts),
                'alert_rules': len(self.alert_rules)
            }


class ClusterMonitor:
    """
    Comprehensive cluster monitoring and alerting system.
    
    Features:
    - Real-time metrics collection and aggregation
    - Configurable alerting with multiple severity levels
    - Performance monitoring and profiling
    - Resource utilization tracking
    - Health checks and status monitoring
    - Historical data retention and analysis
    - Integration with external monitoring systems
    """
    
    def __init__(self, cluster_manager, message_broker, config: Optional[MonitoringConfig] = None):
        """
        Initialize cluster monitor.
        
        Args:
            cluster_manager: ClusterManager instance
            message_broker: MessageBroker for communication
            config: Monitoring configuration
        """
        self.cluster_manager = cluster_manager
        self.message_broker = message_broker
        self.config = config or MonitoringConfig()
        
        # Components
        self.metrics_collector = MetricsCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Control
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'monitoring_started': time.time(),
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'nodes_monitored': 0,
            'collection_errors': 0
        }
        
        # Event callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._metric_callbacks: List[Callable[[MetricPoint], None]] = []
        
    async def start(self) -> bool:
        """
        Start the cluster monitor.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            self._running = True
            
            # Start background tasks
            self._collection_task = asyncio.create_task(self._collection_loop())
            
            if self.config.enable_alerting:
                self._alert_task = asyncio.create_task(self._alert_loop())
                
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Cluster monitor started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cluster monitor: {e}")
            self._running = False
            return False
            
    async def stop(self):
        """Stop the cluster monitor."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._collection_task, self._alert_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        logger.info("Cluster monitor stopped")
        
    async def collect_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Collect metrics from a specific node."""
        try:
            # Get node information from cluster manager
            node = self.cluster_manager.get_node(node_id)
            if not node:
                return {}
                
            # Collect basic resource metrics
            metrics = {
                'cpu_percent': node.usage.cpu_percent,
                'memory_percent': node.usage.memory_percent,
                'memory_gb': node.usage.memory_gb,
                'disk_percent': getattr(node.usage, 'disk_percent', 0.0),
                'network_bytes_sent': getattr(node.usage, 'network_bytes_sent', 0),
                'network_bytes_recv': getattr(node.usage, 'network_bytes_recv', 0),
                'load_average': getattr(node.usage, 'load_average', 0.0),
                'uptime': time.time() - node.created_at,
                'last_heartbeat': node.last_heartbeat,
                'task_count': len(self.cluster_manager.get_node_tasks(node_id))
            }
            
            # Record metrics
            current_time = time.time()
            for metric_name, value in metrics.items():
                if value is not None:
                    metric_point = MetricPoint(
                        name=metric_name,
                        value=value,
                        timestamp=current_time,
                        labels={'node_id': node_id}
                    )
                    self.metrics_collector.record_metric(metric_point, node_id)
                    
                    # Trigger metric callbacks
                    for callback in self._metric_callbacks:
                        try:
                            callback(metric_point)
                        except Exception as e:
                            logger.warning(f"Metric callback error: {e}")
                            
            self._stats['metrics_collected'] += len(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics from node {node_id}: {e}")
            self._stats['collection_errors'] += 1
            return {}
            
    async def collect_cluster_metrics(self) -> Dict[str, Any]:
        """Collect cluster-wide aggregate metrics."""
        try:
            nodes = self.cluster_manager.get_all_nodes()
            
            if not nodes:
                return {}
                
            # Aggregate node metrics
            total_cpu = sum(node.usage.cpu_percent for node in nodes)
            total_memory = sum(node.usage.memory_percent for node in nodes)
            total_tasks = sum(len(self.cluster_manager.get_node_tasks(node.node_id)) for node in nodes)
            
            cluster_metrics = {
                'cluster_size': len(nodes),
                'avg_cpu_percent': total_cpu / len(nodes),
                'avg_memory_percent': total_memory / len(nodes),
                'total_tasks': total_tasks,
                'healthy_nodes': len([n for n in nodes if n.status.value in ['active', 'idle']]),
                'cluster_capacity': sum(node.capacity.cpu_cores for node in nodes),
                'cluster_memory_gb': sum(node.capacity.memory_gb for node in nodes)
            }
            
            # Record cluster metrics
            current_time = time.time()
            for metric_name, value in cluster_metrics.items():
                metric_point = MetricPoint(
                    name=f"cluster_{metric_name}",
                    value=value,
                    timestamp=current_time,
                    labels={'cluster': 'main'}
                )
                self.metrics_collector.record_metric(metric_point)
                
            return cluster_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect cluster metrics: {e}")
            return {}
            
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health assessment."""
        try:
            nodes = self.cluster_manager.get_all_nodes()
            current_time = time.time()
            
            # Count nodes by status
            status_counts = defaultdict(int)
            for node in nodes:
                status_counts[node.status.value] += 1
                
            # Check for recent failures
            recent_failures = 0
            for node in nodes:
                if current_time - node.last_heartbeat > self.config.heartbeat_timeout:
                    recent_failures += 1
                    
            # Calculate health score
            total_nodes = len(nodes)
            healthy_nodes = status_counts.get('active', 0) + status_counts.get('idle', 0)
            health_score = (healthy_nodes / max(1, total_nodes)) * 100
            
            # Determine overall health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "degraded"
            else:
                health_status = "critical"
                
            return {
                'health_status': health_status,
                'health_score': health_score,
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'unhealthy_nodes': total_nodes - healthy_nodes,
                'recent_failures': recent_failures,
                'status_distribution': dict(status_counts),
                'active_alerts': len(self.alert_manager.get_active_alerts()),
                'last_updated': current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to assess cluster health: {e}")
            return {
                'health_status': 'unknown',
                'health_score': 0,
                'error': str(e)
            }
            
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                # Collect metrics from all nodes
                nodes = self.cluster_manager.get_all_nodes()
                self._stats['nodes_monitored'] = len(nodes)
                
                # Collect node metrics
                for node in nodes:
                    await self.collect_node_metrics(node.node_id)
                    
                # Collect cluster-wide metrics
                await self.collect_cluster_metrics()
                
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                self._stats['collection_errors'] += 1
                await asyncio.sleep(30.0)
                
    async def _alert_loop(self):
        """Main alert evaluation loop."""
        while self._running:
            try:
                # Evaluate alert rules
                new_alerts = self.alert_manager.evaluate_alerts(self.metrics_collector)
                
                # Process new alerts
                for alert in new_alerts:
                    self._stats['alerts_triggered'] += 1
                    
                    # Trigger alert callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.warning(f"Alert callback error: {e}")
                            
                await asyncio.sleep(self.config.alert_evaluation_interval)
                
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
                await asyncio.sleep(60.0)
                
    async def _cleanup_loop(self):
        """Cleanup old metrics and alerts."""
        while self._running:
            try:
                # This would implement cleanup of old metrics
                # based on retention policy
                
                await asyncio.sleep(3600.0)  # Run cleanup every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600.0)
                
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        stats = self._stats.copy()
        
        # Add component stats
        stats.update({
            'metrics_collector': self.metrics_collector.get_collector_stats(),
            'alert_manager': self.alert_manager.get_alert_stats(),
            'uptime': time.time() - self._stats['monitoring_started'],
            'collection_rate': self._stats['metrics_collected'] / max(1, time.time() - self._stats['monitoring_started'])
        })
        
        return stats
        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert events."""
        self._alert_callbacks.append(callback)
        
    def add_metric_callback(self, callback: Callable[[MetricPoint], None]):
        """Add callback for metric collection events."""
        self._metric_callbacks.append(callback)