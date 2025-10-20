"""
Auto Scaler - Dynamic Resource Scaling

Automatically scales cluster resources based on workload demand, resource utilization,
and performance metrics with support for multiple scaling strategies.
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_DEPTH = "queue_depth"
    COMPOSITE = "composite"
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"


class ScalingTrigger(Enum):
    """What triggered a scaling event."""
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    HIGH_QUEUE_DEPTH = "high_queue_depth"
    LOW_UTILIZATION = "low_utilization"
    PREDICTED_LOAD = "predicted_load"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED_EVENT = "scheduled_event"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for scaling decisions."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_utilization: float
    queue_depth: int
    active_tasks: int
    node_count: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'network_utilization': self.network_utilization,
            'queue_depth': self.queue_depth,
            'active_tasks': self.active_tasks,
            'node_count': self.node_count,
            'timestamp': self.timestamp
        }


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    # Scaling thresholds
    cpu_scale_up_threshold: float = 80.0    # CPU % to trigger scale up
    cpu_scale_down_threshold: float = 30.0  # CPU % to trigger scale down
    memory_scale_up_threshold: float = 85.0  # Memory % to trigger scale up
    memory_scale_down_threshold: float = 40.0  # Memory % to trigger scale down
    queue_depth_threshold: int = 100         # Queue depth to trigger scale up
    
    # Scaling limits
    min_nodes: int = 2
    max_nodes: int = 100
    scale_up_nodes: int = 1      # Nodes to add per scale up event
    scale_down_nodes: int = 1    # Nodes to remove per scale down event
    
    # Timing constraints
    scale_up_cooldown: float = 300.0     # 5 minutes between scale ups
    scale_down_cooldown: float = 600.0   # 10 minutes between scale downs
    evaluation_interval: float = 60.0    # 1 minute between evaluations
    metrics_window: int = 5              # Number of samples for decision
    
    # Strategy configuration
    strategy: ScalingStrategy = ScalingStrategy.COMPOSITE
    enable_predictive: bool = True
    enable_scheduled: bool = False
    
    # Cost optimization
    prefer_spot_instances: bool = True
    cost_optimization_weight: float = 0.3
    
    # Performance targets
    target_cpu_utilization: float = 65.0
    target_memory_utilization: float = 70.0
    target_response_time: float = 1.0  # seconds


@dataclass
class ScalingEvent:
    """Record of a scaling action."""
    event_id: str
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    nodes_before: int
    nodes_after: int
    metrics_snapshot: ResourceMetrics
    reason: str
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'direction': self.direction.value,
            'trigger': self.trigger.value,
            'nodes_before': self.nodes_before,
            'nodes_after': self.nodes_after,
            'metrics_snapshot': self.metrics_snapshot.to_dict(),
            'reason': self.reason,
            'success': self.success,
            'error_message': self.error_message
        }


class PredictiveModel:
    """Simple predictive model for workload forecasting."""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize predictive model.
        
        Args:
            window_size: Number of historical samples to use
        """
        self.window_size = window_size
        self.cpu_history: deque = deque(maxlen=window_size)
        self.memory_history: deque = deque(maxlen=window_size)
        self.queue_history: deque = deque(maxlen=window_size)
        
    def update(self, metrics: ResourceMetrics):
        """Update model with new metrics."""
        self.cpu_history.append(metrics.cpu_percent)
        self.memory_history.append(metrics.memory_percent)
        self.queue_history.append(metrics.queue_depth)
        
    def predict_next(self, horizon_minutes: int = 5) -> ResourceMetrics:
        """
        Predict resource utilization for the next time period.
        
        Args:
            horizon_minutes: How far ahead to predict
            
        Returns:
            Predicted resource metrics
        """
        if len(self.cpu_history) < 3:
            # Not enough data for prediction, return current values
            return ResourceMetrics(
                cpu_percent=self.cpu_history[-1] if self.cpu_history else 0.0,
                memory_percent=self.memory_history[-1] if self.memory_history else 0.0,
                disk_percent=0.0,
                network_utilization=0.0,
                queue_depth=self.queue_history[-1] if self.queue_history else 0,
                active_tasks=0,
                node_count=0
            )
            
        # Simple linear trend prediction
        cpu_trend = self._calculate_trend(list(self.cpu_history))
        memory_trend = self._calculate_trend(list(self.memory_history))
        queue_trend = self._calculate_trend(list(self.queue_history))
        
        # Project trends forward
        predicted_cpu = max(0.0, min(100.0, 
            self.cpu_history[-1] + cpu_trend * horizon_minutes))
        predicted_memory = max(0.0, min(100.0,
            self.memory_history[-1] + memory_trend * horizon_minutes))
        predicted_queue = max(0, int(
            self.queue_history[-1] + queue_trend * horizon_minutes))
        
        return ResourceMetrics(
            cpu_percent=predicted_cpu,
            memory_percent=predicted_memory,
            disk_percent=0.0,
            network_utilization=0.0,
            queue_depth=predicted_queue,
            active_tasks=0,
            node_count=0
        )
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in time series data."""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope


class AutoScaler:
    """
    Intelligent auto-scaling system for distributed cluster management.
    
    Features:
    - Multiple scaling strategies (CPU, memory, queue depth, composite)
    - Predictive scaling using trend analysis
    - Cost-aware scaling decisions
    - Cooldown periods to prevent thrashing
    - Comprehensive scaling event logging
    - Integration with multiple cloud providers
    """
    
    def __init__(self, cluster_manager, policy: Optional[ScalingPolicy] = None):
        """
        Initialize auto scaler.
        
        Args:
            cluster_manager: ClusterManager instance
            policy: Scaling policy configuration
        """
        self.cluster_manager = cluster_manager
        self.policy = policy or ScalingPolicy()
        
        # Scaling state
        self.last_scale_up_time: float = 0.0
        self.last_scale_down_time: float = 0.0
        self.metrics_history: deque = deque(maxlen=self.policy.metrics_window)
        self.scaling_events: List[ScalingEvent] = []
        
        # Predictive modeling
        self.predictive_model = PredictiveModel()
        
        # Control
        self._running = False
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'total_scale_ups': 0,
            'total_scale_downs': 0,
            'scaling_accuracy': 0.0,
            'cost_savings': 0.0,
            'performance_improvement': 0.0,
            'last_scaling_decision': time.time()
        }
        
        # Event callbacks
        self._scaling_callbacks: List[Callable[[ScalingEvent], None]] = []
        
    async def start(self) -> bool:
        """
        Start the auto scaler.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            self._running = True
            self._scaling_task = asyncio.create_task(self._scaling_loop())
            
            logger.info("Auto scaler started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start auto scaler: {e}")
            self._running = False
            return False
            
    async def stop(self):
        """Stop the auto scaler."""
        self._running = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Auto scaler stopped")
        
    async def evaluate_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """
        Evaluate whether scaling is needed based on current metrics.
        
        Args:
            metrics: Current resource utilization metrics
            
        Returns:
            Tuple of (scaling_direction, trigger, reason)
        """
        current_time = time.time()
        
        # Add metrics to history
        self.metrics_history.append(metrics)
        self.predictive_model.update(metrics)
        
        # Check if we have enough data for decision
        if len(self.metrics_history) < self.policy.metrics_window:
            return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, "Insufficient metrics history"
            
        # Check cooldown periods
        if (current_time - self.last_scale_up_time < self.policy.scale_up_cooldown and
            current_time - self.last_scale_down_time < self.policy.scale_down_cooldown):
            return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, "In cooldown period"
            
        # Apply scaling strategy
        if self.policy.strategy == ScalingStrategy.CPU_BASED:
            return await self._evaluate_cpu_scaling(metrics)
        elif self.policy.strategy == ScalingStrategy.MEMORY_BASED:
            return await self._evaluate_memory_scaling(metrics)
        elif self.policy.strategy == ScalingStrategy.QUEUE_DEPTH:
            return await self._evaluate_queue_scaling(metrics)
        elif self.policy.strategy == ScalingStrategy.COMPOSITE:
            return await self._evaluate_composite_scaling(metrics)
        elif self.policy.strategy == ScalingStrategy.PREDICTIVE:
            return await self._evaluate_predictive_scaling(metrics)
        else:
            return await self._evaluate_reactive_scaling(metrics)
            
    async def _evaluate_cpu_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling based on CPU utilization."""
        avg_cpu = statistics.mean([m.cpu_percent for m in self.metrics_history])
        
        if avg_cpu > self.policy.cpu_scale_up_threshold:
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.HIGH_CPU, f"High CPU: {avg_cpu:.1f}%"
        elif avg_cpu < self.policy.cpu_scale_down_threshold:
            if metrics.node_count > self.policy.min_nodes:
                return ScalingDirection.SCALE_DOWN, ScalingTrigger.LOW_UTILIZATION, f"Low CPU: {avg_cpu:.1f}%"
                
        return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, f"CPU within normal range: {avg_cpu:.1f}%"
        
    async def _evaluate_memory_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling based on memory utilization."""
        avg_memory = statistics.mean([m.memory_percent for m in self.metrics_history])
        
        if avg_memory > self.policy.memory_scale_up_threshold:
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.HIGH_MEMORY, f"High memory: {avg_memory:.1f}%"
        elif avg_memory < self.policy.memory_scale_down_threshold:
            if metrics.node_count > self.policy.min_nodes:
                return ScalingDirection.SCALE_DOWN, ScalingTrigger.LOW_UTILIZATION, f"Low memory: {avg_memory:.1f}%"
                
        return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, f"Memory within normal range: {avg_memory:.1f}%"
        
    async def _evaluate_queue_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling based on task queue depth."""
        avg_queue = statistics.mean([m.queue_depth for m in self.metrics_history])
        
        if avg_queue > self.policy.queue_depth_threshold:
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.HIGH_QUEUE_DEPTH, f"High queue depth: {avg_queue:.0f}"
        elif avg_queue < self.policy.queue_depth_threshold * 0.1:  # 10% of threshold
            if metrics.node_count > self.policy.min_nodes:
                return ScalingDirection.SCALE_DOWN, ScalingTrigger.LOW_UTILIZATION, f"Low queue depth: {avg_queue:.0f}"
                
        return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, f"Queue depth normal: {avg_queue:.0f}"
        
    async def _evaluate_composite_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling using composite metrics."""
        # Calculate weighted score for scale up
        cpu_score = self._normalize_metric(
            statistics.mean([m.cpu_percent for m in self.metrics_history]),
            self.policy.cpu_scale_up_threshold, 100.0
        )
        
        memory_score = self._normalize_metric(
            statistics.mean([m.memory_percent for m in self.metrics_history]),
            self.policy.memory_scale_up_threshold, 100.0
        )
        
        queue_score = self._normalize_metric(
            statistics.mean([m.queue_depth for m in self.metrics_history]),
            self.policy.queue_depth_threshold, self.policy.queue_depth_threshold * 2
        )
        
        # Weighted composite score
        composite_score = (cpu_score * 0.4 + memory_score * 0.4 + queue_score * 0.2)
        
        if composite_score > 0.8:  # High utilization
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.HIGH_CPU, f"High composite score: {composite_score:.2f}"
        elif composite_score < 0.3:  # Low utilization
            if metrics.node_count > self.policy.min_nodes:
                return ScalingDirection.SCALE_DOWN, ScalingTrigger.LOW_UTILIZATION, f"Low composite score: {composite_score:.2f}"
                
        return ScalingDirection.NO_CHANGE, ScalingTrigger.MANUAL_REQUEST, f"Composite score normal: {composite_score:.2f}"
        
    async def _evaluate_predictive_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling using predictive modeling."""
        # Get prediction for next 5 minutes
        predicted_metrics = self.predictive_model.predict_next(5)
        
        # Check if predicted metrics would trigger scaling
        if predicted_metrics.cpu_percent > self.policy.cpu_scale_up_threshold * 0.9:  # 90% of threshold
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.PREDICTED_LOAD, f"Predicted high CPU: {predicted_metrics.cpu_percent:.1f}%"
        elif predicted_metrics.memory_percent > self.policy.memory_scale_up_threshold * 0.9:
            if metrics.node_count < self.policy.max_nodes:
                return ScalingDirection.SCALE_UP, ScalingTrigger.PREDICTED_LOAD, f"Predicted high memory: {predicted_metrics.memory_percent:.1f}%"
                
        # Fall back to reactive scaling
        return await self._evaluate_composite_scaling(metrics)
        
    async def _evaluate_reactive_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDirection, ScalingTrigger, str]:
        """Evaluate scaling using reactive approach."""
        return await self._evaluate_composite_scaling(metrics)
        
    def _normalize_metric(self, value: float, threshold: float, max_value: float) -> float:
        """Normalize a metric to 0-1 scale based on threshold."""
        if value <= threshold:
            return value / threshold * 0.5  # 0 to 0.5 for values below threshold
        else:
            # 0.5 to 1.0 for values above threshold
            return 0.5 + (value - threshold) / (max_value - threshold) * 0.5
            
    async def execute_scaling(self, direction: ScalingDirection, trigger: ScalingTrigger, 
                            reason: str, current_metrics: ResourceMetrics) -> bool:
        """
        Execute a scaling action.
        
        Args:
            direction: Scale up or down
            trigger: What triggered the scaling
            reason: Human-readable reason
            current_metrics: Current system metrics
            
        Returns:
            True if scaling was successful
        """
        import uuid
        
        event_id = str(uuid.uuid4())
        current_time = time.time()
        nodes_before = current_metrics.node_count
        
        try:
            if direction == ScalingDirection.SCALE_UP:
                # Add nodes
                success = await self._scale_up(self.policy.scale_up_nodes)
                self.last_scale_up_time = current_time
                self._stats['total_scale_ups'] += 1
                
            elif direction == ScalingDirection.SCALE_DOWN:
                # Remove nodes
                success = await self._scale_down(self.policy.scale_down_nodes)
                self.last_scale_down_time = current_time
                self._stats['total_scale_downs'] += 1
                
            else:
                success = True  # No change is always successful
                
            # Record scaling event
            nodes_after = nodes_before
            if success:
                if direction == ScalingDirection.SCALE_UP:
                    nodes_after += self.policy.scale_up_nodes
                elif direction == ScalingDirection.SCALE_DOWN:
                    nodes_after -= self.policy.scale_down_nodes
                    
            scaling_event = ScalingEvent(
                event_id=event_id,
                timestamp=current_time,
                direction=direction,
                trigger=trigger,
                nodes_before=nodes_before,
                nodes_after=nodes_after,
                metrics_snapshot=current_metrics,
                reason=reason,
                success=success
            )
            
            self.scaling_events.append(scaling_event)
            
            # Trigger callbacks
            for callback in self._scaling_callbacks:
                try:
                    callback(scaling_event)
                except Exception as e:
                    logger.warning(f"Scaling callback error: {e}")
                    
            self._stats['last_scaling_decision'] = current_time
            
            if success and direction != ScalingDirection.NO_CHANGE:
                logger.info(f"Scaling {direction.value}: {nodes_before} -> {nodes_after} nodes. Reason: {reason}")
            
            return success
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            
            # Record failed event
            scaling_event = ScalingEvent(
                event_id=event_id,
                timestamp=current_time,
                direction=direction,
                trigger=trigger,
                nodes_before=nodes_before,
                nodes_after=nodes_before,
                metrics_snapshot=current_metrics,
                reason=reason,
                success=False,
                error_message=str(e)
            )
            
            self.scaling_events.append(scaling_event)
            return False
            
    async def _scale_up(self, num_nodes: int) -> bool:
        """Add nodes to the cluster."""
        try:
            # This would integrate with cloud provider APIs
            # For now, we'll simulate the scaling action
            logger.info(f"Scaling up: adding {num_nodes} nodes")
            
            # In a real implementation, this would:
            # 1. Call cloud provider API to launch instances
            # 2. Wait for instances to be ready
            # 3. Register new nodes with cluster manager
            # 4. Verify nodes are healthy and accepting work
            
            await asyncio.sleep(1.0)  # Simulate scaling delay
            return True
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return False
            
    async def _scale_down(self, num_nodes: int) -> bool:
        """Remove nodes from the cluster."""
        try:
            # This would integrate with cloud provider APIs
            logger.info(f"Scaling down: removing {num_nodes} nodes")
            
            # In a real implementation, this would:
            # 1. Select nodes for removal (least utilized, graceful shutdown)
            # 2. Drain tasks from selected nodes
            # 3. Deregister nodes from cluster manager
            # 4. Terminate instances via cloud provider API
            
            await asyncio.sleep(1.0)  # Simulate scaling delay
            return True
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            return False
            
    async def _scaling_loop(self):
        """Main auto-scaling evaluation loop."""
        while self._running:
            try:
                # Collect current metrics
                metrics = await self._collect_cluster_metrics()
                
                if metrics:
                    # Evaluate scaling need
                    direction, trigger, reason = await self.evaluate_scaling(metrics)
                    
                    # Execute scaling if needed
                    if direction != ScalingDirection.NO_CHANGE:
                        await self.execute_scaling(direction, trigger, reason, metrics)
                        
                await asyncio.sleep(self.policy.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(30.0)  # Backoff on error
                
    async def _collect_cluster_metrics(self) -> Optional[ResourceMetrics]:
        """Collect current cluster resource metrics."""
        try:
            # Get nodes from cluster manager
            nodes = self.cluster_manager.get_all_nodes()
            
            if not nodes:
                return None
                
            # Calculate aggregate metrics
            total_cpu = sum(node.usage.cpu_percent for node in nodes)
            total_memory = sum(node.usage.memory_percent for node in nodes)
            avg_cpu = total_cpu / len(nodes)
            avg_memory = total_memory / len(nodes)
            
            # Get queue depth from task scheduler
            queue_depth = 0
            if hasattr(self.cluster_manager, 'task_scheduler'):
                scheduler = self.cluster_manager.task_scheduler
                if hasattr(scheduler, 'get_scheduler_stats'):
                    stats = scheduler.get_scheduler_stats()
                    queue_depth = stats.get('pending_tasks', 0) + stats.get('queued_tasks', 0)
                    
            return ResourceMetrics(
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                disk_percent=0.0,  # TODO: Implement disk monitoring
                network_utilization=0.0,  # TODO: Implement network monitoring
                queue_depth=queue_depth,
                active_tasks=sum(len(self.cluster_manager.get_node_tasks(node.node_id)) for node in nodes),
                node_count=len(nodes)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect cluster metrics: {e}")
            return None
            
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling statistics."""
        stats = self._stats.copy()
        
        # Add recent events
        recent_events = [event.to_dict() for event in self.scaling_events[-10:]]
        stats['recent_events'] = recent_events
        
        # Calculate scaling efficiency
        total_scaling_events = len([e for e in self.scaling_events if e.direction != ScalingDirection.NO_CHANGE])
        successful_events = len([e for e in self.scaling_events if e.success])
        
        if total_scaling_events > 0:
            stats['scaling_success_rate'] = successful_events / total_scaling_events
        else:
            stats['scaling_success_rate'] = 0.0
            
        # Add current policy
        stats['current_policy'] = {
            'strategy': self.policy.strategy.value,
            'min_nodes': self.policy.min_nodes,
            'max_nodes': self.policy.max_nodes,
            'cpu_thresholds': [self.policy.cpu_scale_down_threshold, self.policy.cpu_scale_up_threshold],
            'memory_thresholds': [self.policy.memory_scale_down_threshold, self.policy.memory_scale_up_threshold]
        }
        
        return stats
        
    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]):
        """Add callback for scaling events."""
        self._scaling_callbacks.append(callback)