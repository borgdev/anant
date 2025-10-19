"""
Performance Monitor
==================

Real-time performance tracking and optimization for ANANT's Polars+Parquet operations.
Monitors system resources, query performance, and data processing efficiency.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import json
import polars as pl


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryPerformance:
    """Query performance statistics."""
    query_id: str
    query_type: str  # hypergraph, metagraph, analytics
    execution_time: float
    memory_usage: float
    rows_processed: int
    optimization_applied: bool
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    polars_thread_pool: int
    active_queries: int
    timestamp: datetime


class MetricsCollector:
    """
    Collects and aggregates performance metrics for ANANT operations.
    
    Focuses on:
    - Polars query performance
    - Parquet I/O efficiency  
    - Memory usage patterns
    - CPU utilization
    - Data processing throughput
    """
    
    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.metrics: deque = deque()
        self.query_metrics: deque = deque()
        self.system_metrics: deque = deque()
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.custom_collectors: Dict[str, Callable] = {}
        
    def start_collection(self, interval_seconds: int = 30):
        """Start metrics collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.collection_thread.start()
        print(f"📊 Metrics collection started (interval: {interval_seconds}s)")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("📊 Metrics collection stopped")
    
    def _collection_loop(self, interval_seconds: int):
        """Main collection loop."""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                self._collect_polars_metrics()
                self._cleanup_old_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Metrics collection error: {e}")
                time.sleep(interval_seconds * 2)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            import os
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Polars-specific metrics
            polars_threads = int(os.environ.get('POLARS_MAX_THREADS', '0'))
            
            metrics = SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_io=0.0,  # Would implement actual disk I/O monitoring
                network_io=0.0,  # Would implement actual network I/O monitoring
                polars_thread_pool=polars_threads,
                active_queries=0,  # Would track active Polars queries
                timestamp=datetime.now()
            )
            
            self.system_metrics.append(metrics)
            
        except ImportError:
            # Fallback metrics without psutil
            metrics = SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_io=0.0,
                network_io=0.0,
                polars_thread_pool=0,
                active_queries=0,
                timestamp=datetime.now()
            )
            self.system_metrics.append(metrics)
        except Exception as e:
            print(f"System metrics collection failed: {e}")
    
    def _collect_polars_metrics(self):
        """Collect Polars-specific performance metrics."""
        try:
            # Custom Polars metrics collection
            for name, collector in self.custom_collectors.items():
                try:
                    value = collector()
                    metric = PerformanceMetric(
                        name=name,
                        value=value,
                        unit="custom",
                        timestamp=datetime.now(),
                        tags={"source": "polars"}
                    )
                    self.metrics.append(metric)
                except Exception as e:
                    print(f"Custom collector '{name}' failed: {e}")
                    
        except Exception as e:
            print(f"Polars metrics collection failed: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(minutes=self.retention_minutes)
        
        # Clean up general metrics
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
            
        # Clean up query metrics
        while self.query_metrics and self.query_metrics[0].timestamp < cutoff_time:
            self.query_metrics.popleft()
            
        # Clean up system metrics
        while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
            self.system_metrics.popleft()
    
    def record_query_performance(self, query_performance: QueryPerformance):
        """Record performance metrics for a query."""
        self.query_metrics.append(query_performance)
    
    def add_custom_collector(self, name: str, collector: Callable[[], float]):
        """Add a custom metrics collector."""
        self.custom_collectors[name] = collector
        print(f"Added custom collector: {name}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        if not self.system_metrics:
            return {}
            
        latest_system = self.system_metrics[-1]
        
        # Calculate query performance averages
        recent_queries = [q for q in self.query_metrics if q.timestamp > datetime.now() - timedelta(minutes=5)]
        avg_query_time = sum(q.execution_time for q in recent_queries) / len(recent_queries) if recent_queries else 0
        
        return {
            "timestamp": latest_system.timestamp.isoformat(),
            "system": {
                "cpu_usage": latest_system.cpu_usage,
                "memory_usage": latest_system.memory_usage,
                "polars_threads": latest_system.polars_thread_pool,
                "active_queries": latest_system.active_queries
            },
            "queries": {
                "count_last_5min": len(recent_queries),
                "avg_execution_time": avg_query_time,
                "total_queries": len(self.query_metrics)
            },
            "custom_metrics": {
                metric.name: metric.value for metric in self.metrics
                if metric.timestamp > datetime.now() - timedelta(minutes=1)
            }
        }
    
    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Filter recent data
        recent_system = [m for m in self.system_metrics if m.timestamp > cutoff_time]
        recent_queries = [q for q in self.query_metrics if q.timestamp > cutoff_time]
        
        if not recent_system:
            return {"error": "No data available for the specified period"}
        
        # Calculate averages and statistics
        avg_cpu = sum(m.cpu_usage for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_usage for m in recent_system) / len(recent_system)
        max_cpu = max(m.cpu_usage for m in recent_system)
        max_memory = max(m.memory_usage for m in recent_system)
        
        # Query statistics
        query_count = len(recent_queries)
        avg_query_time = sum(q.execution_time for q in recent_queries) / query_count if query_count > 0 else 0
        failed_queries = len([q for q in recent_queries if q.error is not None])
        
        return {
            "period_minutes": minutes,
            "system_performance": {
                "cpu": {
                    "average": round(avg_cpu, 2),
                    "maximum": round(max_cpu, 2),
                    "current": round(recent_system[-1].cpu_usage, 2)
                },
                "memory": {
                    "average": round(avg_memory, 2),
                    "maximum": round(max_memory, 2),
                    "current": round(recent_system[-1].memory_usage, 2)
                }
            },
            "query_performance": {
                "total_queries": query_count,
                "average_execution_time": round(avg_query_time, 3),
                "failed_queries": failed_queries,
                "success_rate": round((query_count - failed_queries) / query_count * 100, 2) if query_count > 0 else 0
            },
            "recommendations": self._generate_recommendations(recent_system, recent_queries)
        }
    
    def _generate_recommendations(self, system_metrics: List[SystemMetrics], query_metrics: List[QueryPerformance]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if system_metrics:
            avg_cpu = sum(m.cpu_usage for m in system_metrics) / len(system_metrics)
            avg_memory = sum(m.memory_usage for m in system_metrics) / len(system_metrics)
            
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected. Consider increasing Polars thread pool or scaling horizontally.")
            if avg_memory > 85:
                recommendations.append("High memory usage detected. Consider optimizing queries or increasing memory allocation.")
            if avg_cpu < 30 and avg_memory < 50:
                recommendations.append("Low resource utilization. Consider consolidating workloads or reducing allocated resources.")
        
        if query_metrics:
            avg_query_time = sum(q.execution_time for q in query_metrics) / len(query_metrics)
            slow_queries = [q for q in query_metrics if q.execution_time > avg_query_time * 2]
            
            if len(slow_queries) > len(query_metrics) * 0.1:
                recommendations.append("High number of slow queries detected. Review query optimization and indexing strategies.")
            if avg_query_time > 5.0:
                recommendations.append("Average query time is high. Consider query optimization or data partitioning.")
        
        return recommendations


class PerformanceMonitor:
    """
    Main performance monitoring system for ANANT production.
    
    Provides:
    - Real-time performance tracking
    - Automated optimization recommendations
    - Performance alerting
    - Historical performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.collector = MetricsCollector(
            retention_minutes=self.config.get('retention_minutes', 60)
        )
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 85,
            'memory_warning': 75,
            'memory_critical': 90,
            'query_time_warning': 3.0,
            'query_time_critical': 10.0
        }
        self.thresholds.update(self.config.get('thresholds', {}))
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start performance monitoring."""
        self.collector.start_collection(interval_seconds)
        
        # Add default Polars collectors
        self._setup_default_collectors()
        
        print("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.collector.stop_collection()
        print("🔍 Performance monitoring stopped")
    
    def _setup_default_collectors(self):
        """Setup default performance collectors for Polars/Parquet operations."""
        
        def polars_memory_usage():
            """Estimate Polars memory usage."""
            # This would be implemented with actual Polars memory tracking
            return 0.0
        
        def parquet_io_rate():
            """Monitor Parquet I/O operations."""
            # This would track actual Parquet read/write operations
            return 0.0
        
        self.collector.add_custom_collector("polars_memory_mb", polars_memory_usage)
        self.collector.add_custom_collector("parquet_io_ops_per_sec", parquet_io_rate)
    
    def track_query(self, query_type: str, operation: Callable, *args, **kwargs) -> Any:
        """
        Track performance of a query operation.
        
        Args:
            query_type: Type of query (hypergraph, metagraph, analytics)
            operation: Function to execute and monitor
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Result of the operation
        """
        start_time = time.time()
        start_memory = self._get_current_memory_usage()
        query_id = f"{query_type}_{int(start_time)}"
        
        try:
            result = operation(*args, **kwargs)
            execution_time = time.time() - start_time
            end_memory = self._get_current_memory_usage()
            memory_usage = end_memory - start_memory
            
            # Determine rows processed (if result is a Polars DataFrame)
            rows_processed = 0
            if hasattr(result, 'height'):
                rows_processed = result.height
            elif isinstance(result, (list, tuple)):
                rows_processed = len(result)
            
            # Record performance
            performance = QueryPerformance(
                query_id=query_id,
                query_type=query_type,
                execution_time=execution_time,
                memory_usage=memory_usage,
                rows_processed=rows_processed,
                optimization_applied=False,  # Would detect actual optimizations
                timestamp=datetime.now()
            )
            
            self.collector.record_query_performance(performance)
            self._check_performance_alerts(performance)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed query
            performance = QueryPerformance(
                query_id=query_id,
                query_type=query_type,
                execution_time=execution_time,
                memory_usage=0,
                rows_processed=0,
                optimization_applied=False,
                timestamp=datetime.now(),
                error=str(e)
            )
            
            self.collector.record_query_performance(performance)
            raise
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _check_performance_alerts(self, performance: QueryPerformance):
        """Check if performance metrics trigger alerts."""
        alerts = []
        
        # Check query execution time
        if performance.execution_time > self.thresholds['query_time_critical']:
            alerts.append({
                'level': 'critical',
                'metric': 'query_execution_time',
                'value': performance.execution_time,
                'threshold': self.thresholds['query_time_critical'],
                'query_id': performance.query_id,
                'timestamp': performance.timestamp
            })
        elif performance.execution_time > self.thresholds['query_time_warning']:
            alerts.append({
                'level': 'warning',
                'metric': 'query_execution_time',
                'value': performance.execution_time,
                'threshold': self.thresholds['query_time_warning'],
                'query_id': performance.query_id,
                'timestamp': performance.timestamp
            })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def get_cluster_metrics(self, cluster_id: str) -> Dict[str, Any]:
        """Get performance metrics for a cluster."""
        current_metrics = self.collector.get_current_metrics()
        performance_summary = self.collector.get_performance_summary(60)
        
        return {
            "cluster_id": cluster_id,
            "current": current_metrics,
            "summary": performance_summary,
            "alerts": self.get_recent_alerts(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def optimize_polars_configuration(self) -> Dict[str, Any]:
        """Analyze performance and suggest Polars optimizations."""
        summary = self.collector.get_performance_summary(60)
        optimizations = []
        
        if 'system_performance' in summary:
            cpu_avg = summary['system_performance']['cpu']['average']
            memory_avg = summary['system_performance']['memory']['average']
            
            # CPU-based optimizations
            if cpu_avg < 50:
                optimizations.append({
                    'type': 'thread_pool',
                    'recommendation': 'Consider reducing Polars thread pool size to free up CPU for other processes',
                    'current_cpu': cpu_avg
                })
            elif cpu_avg > 80:
                optimizations.append({
                    'type': 'thread_pool', 
                    'recommendation': 'Consider increasing Polars thread pool size or scaling horizontally',
                    'current_cpu': cpu_avg
                })
            
            # Memory-based optimizations
            if memory_avg > 75:
                optimizations.append({
                    'type': 'streaming',
                    'recommendation': 'Enable Polars streaming mode for large datasets to reduce memory usage',
                    'current_memory': memory_avg
                })
        
        if 'query_performance' in summary:
            avg_time = summary['query_performance']['average_execution_time']
            
            if avg_time > 2.0:
                optimizations.append({
                    'type': 'query_optimization',
                    'recommendation': 'Review query patterns and consider lazy evaluation optimizations',
                    'current_avg_time': avg_time
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimizations': optimizations,
            'performance_summary': summary
        }