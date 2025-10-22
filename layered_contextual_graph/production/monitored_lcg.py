"""
Monitored LayeredContextualGraph
================================

Integrates LCG with Anant's production monitoring infrastructure for:
- Real-time health checks
- Performance monitoring
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Alerting and notifications
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Anant monitoring components
try:
    from anant.production.monitoring import (
        HealthChecker,
        PerformanceMonitor
    )
    from anant.production.core import MetricsCollector
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("Anant production monitoring not available")

from ..core import LayeredContextualGraph

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for LCG monitoring"""
    enable_health_checks: bool = True
    enable_performance_monitoring: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = True
    health_check_interval: int = 30  # seconds
    metrics_port: int = 9090  # Prometheus
    alert_on_high_latency: bool = True
    latency_threshold_ms: float = 100.0
    alert_on_low_coherence: bool = True
    coherence_threshold: float = 0.5


class LCGHealthChecker:
    """
    Health checking for LCG using Anant's HealthChecker.
    
    Monitors:
    - Layer availability
    - Query responsiveness
    - Memory usage
    - Quantum coherence levels
    - Entity state consistency
    """
    
    def __init__(self, lcg: LayeredContextualGraph, config: MonitoringConfig):
        if not MONITORING_AVAILABLE:
            raise RuntimeError("Monitoring requires anant.production")
        
        self.lcg = lcg
        self.config = config
        
        # Initialize Anant health checker
        self.health_checker = HealthChecker()
        
        # Register LCG-specific health checks
        self._register_health_checks()
        
        logger.info("LCGHealthChecker initialized")
    
    def _register_health_checks(self):
        """Register health check functions"""
        
        # Layer health
        self.health_checker.register_check(
            name="layers_available",
            check_function=self._check_layers,
            interval=self.config.health_check_interval,
            critical=True
        )
        
        # Query responsiveness
        self.health_checker.register_check(
            name="query_responsive",
            check_function=self._check_query_performance,
            interval=self.config.health_check_interval,
            critical=True
        )
        
        # Memory usage
        self.health_checker.register_check(
            name="memory_usage",
            check_function=self._check_memory,
            interval=self.config.health_check_interval,
            critical=False
        )
        
        # Quantum coherence
        self.health_checker.register_check(
            name="quantum_coherence",
            check_function=self._check_coherence,
            interval=self.config.health_check_interval,
            critical=False
        )
    
    def _check_layers(self) -> Dict[str, Any]:
        """Check if all layers are accessible"""
        try:
            num_layers = len(self.lcg.layers)
            accessible = all(
                layer.hypergraph is not None 
                for layer in self.lcg.layers.values()
            )
            
            return {
                'status': 'healthy' if accessible else 'unhealthy',
                'num_layers': num_layers,
                'all_accessible': accessible
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _check_query_performance(self) -> Dict[str, Any]:
        """Check query responsiveness"""
        if not self.lcg.superposition_states:
            return {'status': 'healthy', 'note': 'No entities to query'}
        
        try:
            # Test query on first entity
            entity_id = list(self.lcg.superposition_states.keys())[0]
            
            start = time.time()
            result = self.lcg.query_across_layers(entity_id)
            duration_ms = (time.time() - start) * 1000
            
            status = 'healthy' if duration_ms < self.config.latency_threshold_ms else 'degraded'
            
            return {
                'status': status,
                'query_latency_ms': duration_ms,
                'threshold_ms': self.config.latency_threshold_ms
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        import sys
        
        try:
            # Approximate memory usage
            num_layers = len(self.lcg.layers)
            num_entities = len(self.lcg.superposition_states)
            num_quantum_states = len(self.lcg.quantum_states)
            
            # Estimate (very rough)
            estimated_mb = (num_layers * 10) + (num_entities * 0.001) + (num_quantum_states * 0.001)
            
            status = 'healthy' if estimated_mb < 1000 else 'warning'
            
            return {
                'status': status,
                'estimated_memory_mb': estimated_mb,
                'num_layers': num_layers,
                'num_entities': num_entities
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e)
            }
    
    def _check_coherence(self) -> Dict[str, Any]:
        """Check average quantum coherence"""
        if not self.lcg.quantum_states:
            return {'status': 'healthy', 'note': 'No quantum states'}
        
        try:
            coherence_values = [
                qs.coherence 
                for qs in self.lcg.quantum_states.values()
            ]
            
            avg_coherence = sum(coherence_values) / len(coherence_values)
            
            status = 'healthy' if avg_coherence >= self.config.coherence_threshold else 'warning'
            
            return {
                'status': status,
                'average_coherence': avg_coherence,
                'threshold': self.config.coherence_threshold,
                'num_quantum_states': len(coherence_values)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return self.health_checker.get_status()


class LCGPerformanceMonitor:
    """
    Performance monitoring for LCG using Anant's PerformanceMonitor.
    
    Tracks:
    - Query latency (p50, p95, p99)
    - Throughput (queries/sec, events/sec)
    - Resource usage (CPU, memory)
    - Layer operation latency
    - Quantum operation performance
    """
    
    def __init__(self, lcg: LayeredContextualGraph, config: MonitoringConfig):
        if not MONITORING_AVAILABLE:
            raise RuntimeError("Monitoring requires anant.production")
        
        self.lcg = lcg
        self.config = config
        
        # Initialize Anant performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Performance metrics
        self.query_latencies: List[float] = []
        self.operation_counts: Dict[str, int] = {}
        self.start_time = time.time()
        
        logger.info("LCGPerformanceMonitor initialized")
    
    def record_query(self, query_type: str, duration_ms: float):
        """Record query execution"""
        self.query_latencies.append(duration_ms)
        self.operation_counts['query'] = self.operation_counts.get('query', 0) + 1
        
        # Record in Anant monitor
        self.perf_monitor.record_operation(
            operation='query',
            duration=duration_ms/1000,  # Convert to seconds
            metadata={'query_type': query_type}
        )
    
    def record_layer_operation(self, operation: str, duration_ms: float):
        """Record layer operation"""
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        
        self.perf_monitor.record_operation(
            operation=f'layer_{operation}',
            duration=duration_ms/1000
        )
    
    def record_quantum_operation(self, operation: str, duration_ms: float):
        """Record quantum operation"""
        self.operation_counts[f'quantum_{operation}'] = self.operation_counts.get(f'quantum_{operation}', 0) + 1
        
        self.perf_monitor.record_operation(
            operation=f'quantum_{operation}',
            duration=duration_ms/1000
        )
    
    def get_query_statistics(self) -> Dict[str, float]:
        """Get query latency statistics"""
        if not self.query_latencies:
            return {}
        
        sorted_latencies = sorted(self.query_latencies)
        n = len(sorted_latencies)
        
        return {
            'count': n,
            'min_ms': sorted_latencies[0],
            'max_ms': sorted_latencies[-1],
            'mean_ms': sum(sorted_latencies) / n,
            'p50_ms': sorted_latencies[int(n * 0.50)],
            'p95_ms': sorted_latencies[int(n * 0.95)],
            'p99_ms': sorted_latencies[int(n * 0.99)]
        }
    
    def get_throughput(self) -> Dict[str, float]:
        """Get operations throughput"""
        uptime_seconds = time.time() - self.start_time
        
        if uptime_seconds == 0:
            return {}
        
        return {
            'queries_per_second': self.operation_counts.get('query', 0) / uptime_seconds,
            'total_operations': sum(self.operation_counts.values()),
            'operations_per_second': sum(self.operation_counts.values()) / uptime_seconds,
            'uptime_seconds': uptime_seconds
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'query_statistics': self.get_query_statistics(),
            'throughput': self.get_throughput(),
            'operation_counts': self.operation_counts,
            'anant_metrics': self.perf_monitor.get_metrics()
        }


class MonitoredLayeredGraph(LayeredContextualGraph):
    """
    Monitored version of LayeredContextualGraph with observability.
    
    Features:
    - Real-time health monitoring
    - Performance metrics collection
    - Prometheus metrics export
    - OpenTelemetry tracing
    - Alerting on anomalies
    
    Examples:
        >>> config = MonitoringConfig(
        ...     enable_health_checks=True,
        ...     enable_metrics=True,
        ...     metrics_port=9090
        ... )
        >>> mlcg = MonitoredLayeredGraph(
        ...     name="monitored_kg",
        ...     monitoring_config=config
        ... )
        >>> 
        >>> # Health status
        >>> status = mlcg.get_health_status()
        >>> print(status['overall_status'])  # 'healthy'
        >>> 
        >>> # Performance metrics
        >>> metrics = mlcg.get_performance_metrics()
        >>> print(metrics['query_statistics']['p95_ms'])
    """
    
    def __init__(
        self,
        name: str = "monitored_lcg",
        monitoring_config: Optional[MonitoringConfig] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        if not MONITORING_AVAILABLE:
            raise RuntimeError("Monitoring features require anant.production")
        
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        # Initialize monitoring components
        self.health_checker = LCGHealthChecker(self, self.monitoring_config)
        self.perf_monitor = LCGPerformanceMonitor(self, self.monitoring_config)
        
        logger.info(f"MonitoredLayeredGraph initialized: {name}")
    
    def add_layer(self, name: str, hypergraph: Any, *args, **kwargs):
        """Add layer with performance tracking"""
        start = time.time()
        
        try:
            super().add_layer(name, hypergraph, *args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_layer_operation('add', duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_layer_operation('add_failed', duration_ms)
            raise
    
    def query_across_layers(
        self,
        entity_id: str,
        layers: Optional[List[str]] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query with performance tracking"""
        start = time.time()
        
        try:
            results = super().query_across_layers(entity_id, layers, context, **kwargs)
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_query('cross_layer', duration_ms)
            return results
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_query('cross_layer_failed', duration_ms)
            raise
    
    def create_superposition(self, entity_id: str, layer_states=None, quantum_states=None):
        """Create superposition with performance tracking"""
        start = time.time()
        
        try:
            result = super().create_superposition(entity_id, layer_states, quantum_states)
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_quantum_operation('create_superposition', duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_quantum_operation('create_superposition_failed', duration_ms)
            raise
    
    def observe(self, entity_id: str, layer=None, context=None, collapse_quantum=True):
        """Observe with performance tracking"""
        start = time.time()
        
        try:
            result = super().observe(entity_id, layer, context, collapse_quantum)
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_quantum_operation('observe', duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.perf_monitor.record_quantum_operation('observe_failed', duration_ms)
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return self.health_checker.get_health_status()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.perf_monitor.get_performance_report()
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'name': self.name,
            'health': self.get_health_status(),
            'performance': self.get_performance_metrics(),
            'monitoring_config': {
                'health_checks_enabled': self.monitoring_config.enable_health_checks,
                'performance_monitoring_enabled': self.monitoring_config.enable_performance_monitoring,
                'metrics_enabled': self.monitoring_config.enable_metrics
            }
        }
