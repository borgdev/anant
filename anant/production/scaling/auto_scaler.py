"""
Auto Scaler
===========

Automatic scaling system for ANANT services based on performance metrics.
Handles horizontal and vertical scaling with predictive capabilities.
"""

import time
import threading
import math
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingType(Enum):
    """Type of scaling."""
    HORIZONTAL = "horizontal"  # Add/remove replicas
    VERTICAL = "vertical"      # Increase/decrease resources


@dataclass 
class ScalingPolicy:
    """Scaling policy configuration."""
    name: str
    metric: str  # cpu, memory, requests_per_second, etc.
    scale_up_threshold: float
    scale_down_threshold: float
    min_replicas: int
    max_replicas: int
    scale_up_cooldown: int  # seconds
    scale_down_cooldown: int  # seconds
    scaling_factor: float = 1.5  # Multiplier for scaling
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    event_id: str
    service_name: str
    scaling_type: ScalingType
    direction: ScalingDirection
    trigger_metric: str
    trigger_value: float
    threshold: float
    before_replicas: int
    after_replicas: int
    timestamp: datetime
    success: bool
    reason: str


@dataclass
class ServiceMetrics:
    """Current metrics for a service."""
    service_name: str
    cpu_usage: float
    memory_usage: float
    requests_per_second: float
    response_time: float
    current_replicas: int
    timestamp: datetime


class AutoScaler:
    """
    Automatic scaling system for ANANT production services.
    
    Features:
    - Horizontal scaling (replica management)
    - Vertical scaling (resource adjustment)
    - Predictive scaling based on trends
    - Custom metric-based scaling
    - Cooldown periods to prevent thrashing
    - Comprehensive scaling history
    """
    
    def __init__(self, policies: List[ScalingPolicy]):
        self.policies = {policy.name: policy for policy in policies}
        self.scaling_history: List[ScalingEvent] = []
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.is_scaling = False
        self.scaling_thread: Optional[threading.Thread] = None
        
        # Scaling state tracking
        self.last_scale_events: Dict[str, datetime] = {}
        self.scaling_counter = 0
        
        # Predictive scaling
        self.metric_history: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_window = 5  # minutes
        
        print(f"⚖️ Auto-scaler initialized with {len(policies)} policies")
    
    def start_scaling(self, interval_seconds: int = 30):
        """Start automatic scaling."""
        if self.is_scaling:
            return
        
        self.is_scaling = True
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.scaling_thread.start()
        
        print(f"⚖️ Started auto-scaling (interval: {interval_seconds}s)")
    
    def stop_scaling(self):
        """Stop automatic scaling."""
        self.is_scaling = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=30)
        
        print("⚖️ Stopped auto-scaling")
    
    def _scaling_loop(self, interval_seconds: int):
        """Main scaling loop."""
        while self.is_scaling:
            try:
                # Collect metrics for all services
                self._collect_service_metrics()
                
                # Evaluate scaling policies
                for policy_name, policy in self.policies.items():
                    if policy.enabled:
                        self._evaluate_scaling_policy(policy)
                
                # Update metric history for predictions
                self._update_metric_history()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Scaling loop error: {e}")
                time.sleep(interval_seconds * 2)
    
    def _collect_service_metrics(self):
        """Collect current metrics for all services."""
        # In a real implementation, this would collect from monitoring systems
        # For now, we'll simulate metrics for ANANT services
        
        anant_services = [
            "anant-hypergraph-api",
            "anant-metagraph-api", 
            "anant-query-processor",
            "anant-storage-manager"
        ]
        
        for service_name in anant_services:
            # Simulate metric collection
            metrics = self._simulate_service_metrics(service_name)
            self.service_metrics[service_name] = metrics
    
    def _simulate_service_metrics(self, service_name: str) -> ServiceMetrics:
        """Simulate service metrics (replace with real metric collection)."""
        import random
        
        # Get previous metrics for some continuity
        previous = self.service_metrics.get(service_name)
        base_cpu = previous.cpu_usage if previous else 30.0
        base_memory = previous.memory_usage if previous else 40.0
        base_replicas = previous.current_replicas if previous else 2
        
        # Add some variance
        cpu_usage = max(5.0, min(95.0, base_cpu + random.uniform(-10.0, 10.0)))
        memory_usage = max(10.0, min(90.0, base_memory + random.uniform(-5.0, 5.0)))
        requests_per_second = random.uniform(10.0, 100.0)
        response_time = random.uniform(0.1, 2.0)
        
        return ServiceMetrics(
            service_name=service_name,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            requests_per_second=requests_per_second,
            response_time=response_time,
            current_replicas=base_replicas,
            timestamp=datetime.now()
        )
    
    def _evaluate_scaling_policy(self, policy: ScalingPolicy):
        """Evaluate a scaling policy and trigger scaling if needed."""
        # Find services that match this policy (simplified - would use service labels/tags)
        matching_services = [
            service for service_name, service in self.service_metrics.items()
            if service_name.startswith("anant-")
        ]
        
        for service in matching_services:
            # Get metric value based on policy
            metric_value = self._get_metric_value(service, policy.metric)
            
            if metric_value is None:
                continue
            
            # Check cooldown
            if not self._is_cooldown_elapsed(service.service_name, policy):
                continue
            
            # Determine scaling direction
            direction = self._determine_scaling_direction(metric_value, policy)
            
            if direction != ScalingDirection.NONE:
                self._execute_scaling(service, policy, direction, metric_value)
    
    def _get_metric_value(self, service: ServiceMetrics, metric_name: str) -> Optional[float]:
        """Get metric value from service metrics."""
        metric_map = {
            'cpu': service.cpu_usage,
            'memory': service.memory_usage,
            'requests_per_second': service.requests_per_second,
            'response_time': service.response_time
        }
        
        return metric_map.get(metric_name)
    
    def _is_cooldown_elapsed(self, service_name: str, policy: ScalingPolicy) -> bool:
        """Check if cooldown period has elapsed."""
        last_event_key = f"{service_name}:{policy.name}"
        
        if last_event_key not in self.last_scale_events:
            return True
        
        last_event_time = self.last_scale_events[last_event_key]
        elapsed = (datetime.now() - last_event_time).total_seconds()
        
        # Use appropriate cooldown based on last scaling direction
        # For simplicity, using scale_up_cooldown as default
        cooldown = policy.scale_up_cooldown
        
        return elapsed >= cooldown
    
    def _determine_scaling_direction(self, metric_value: float, policy: ScalingPolicy) -> ScalingDirection:
        """Determine if scaling is needed and in which direction."""
        if metric_value >= policy.scale_up_threshold:
            return ScalingDirection.UP
        elif metric_value <= policy.scale_down_threshold:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE
    
    def _execute_scaling(self, service: ServiceMetrics, policy: ScalingPolicy, 
                        direction: ScalingDirection, metric_value: float):
        """Execute scaling operation."""
        event_id = f"scale-{self.scaling_counter}"
        self.scaling_counter += 1
        
        # Calculate new replica count
        current_replicas = service.current_replicas
        
        if direction == ScalingDirection.UP:
            # Scale up
            new_replicas = min(
                policy.max_replicas,
                math.ceil(current_replicas * policy.scaling_factor)
            )
            threshold = policy.scale_up_threshold
            cooldown = policy.scale_up_cooldown
        else:
            # Scale down
            new_replicas = max(
                policy.min_replicas,
                math.floor(current_replicas / policy.scaling_factor)
            )
            threshold = policy.scale_down_threshold
            cooldown = policy.scale_down_cooldown
        
        # Only scale if there's an actual change
        if new_replicas == current_replicas:
            return
        
        try:
            # Execute the scaling operation
            success = self._scale_service(service.service_name, new_replicas)
            
            # Record scaling event
            event = ScalingEvent(
                event_id=event_id,
                service_name=service.service_name,
                scaling_type=ScalingType.HORIZONTAL,
                direction=direction,
                trigger_metric=policy.metric,
                trigger_value=metric_value,
                threshold=threshold,
                before_replicas=current_replicas,
                after_replicas=new_replicas if success else current_replicas,
                timestamp=datetime.now(),
                success=success,
                reason=f"{'Scale up' if direction == ScalingDirection.UP else 'Scale down'} triggered by {policy.metric}={metric_value:.2f}"
            )
            
            self.scaling_history.append(event)
            
            # Update cooldown tracking
            cooldown_key = f"{service.service_name}:{policy.name}"
            self.last_scale_events[cooldown_key] = datetime.now()
            
            # Update service metrics
            if success:
                service.current_replicas = new_replicas
                self.service_metrics[service.service_name] = service
            
            if success:
                print(f"⚖️ Scaled {service.service_name} from {current_replicas} to {new_replicas} replicas ({direction.value})")
            else:
                print(f"❌ Failed to scale {service.service_name}")
                
        except Exception as e:
            print(f"Scaling execution error: {e}")
    
    def _scale_service(self, service_name: str, new_replicas: int) -> bool:
        """Execute the actual scaling operation."""
        try:
            # In a real implementation, this would call Kubernetes API,
            # Docker Swarm, or other orchestration system
            print(f"🔧 Scaling {service_name} to {new_replicas} replicas")
            
            # Simulate scaling delay
            time.sleep(1)
            
            # Simulate success (90% success rate)
            import random
            return random.random() > 0.1
            
        except Exception as e:
            print(f"Service scaling failed: {e}")
            return False
    
    def _update_metric_history(self):
        """Update metric history for predictive scaling."""
        current_time = datetime.now()
        
        for service_name, metrics in self.service_metrics.items():
            if service_name not in self.metric_history:
                self.metric_history[service_name] = []
            
            # Store key metrics
            metric_snapshot = {
                'timestamp': current_time,
                'cpu': metrics.cpu_usage,
                'memory': metrics.memory_usage,
                'requests_per_second': metrics.requests_per_second,
                'response_time': metrics.response_time
            }
            
            self.metric_history[service_name].append(metric_snapshot)
            
            # Keep only recent history (last hour)
            cutoff_time = current_time - timedelta(hours=1)
            self.metric_history[service_name] = [
                snapshot for snapshot in self.metric_history[service_name]
                if snapshot['timestamp'] > cutoff_time
            ]
    
    def predict_scaling_needs(self, service_name: str, minutes_ahead: int = 10) -> Dict[str, Any]:
        """Predict future scaling needs based on metric trends."""
        if service_name not in self.metric_history:
            return {"prediction": "insufficient_data"}
        
        history = self.metric_history[service_name]
        if len(history) < 5:
            return {"prediction": "insufficient_data"}
        
        # Simple linear trend analysis
        recent_history = history[-10:]  # Last 10 data points
        
        predictions = {}
        for metric in ['cpu', 'memory', 'requests_per_second']:
            values = [point[metric] for point in recent_history if isinstance(point, dict) and metric in point]
            if not values:
                continue
                
            trend = self._calculate_trend(values)
            
            # Project trend forward
            current_value = values[-1]
            predicted_value = current_value + (trend * minutes_ahead)
            
            predictions[metric] = {
                'current': current_value,
                'predicted': predicted_value,
                'trend': trend,
                'confidence': self._calculate_confidence(values)
            }
        
        # Determine if scaling will be needed
        scaling_recommendation = self._analyze_predictions(service_name, predictions)
        
        return {
            "service_name": service_name,
            "prediction_horizon_minutes": minutes_ahead,
            "predictions": predictions,
            "recommendation": scaling_recommendation,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of metric values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence level of the trend."""
        if len(values) < 3:
            return 0.0
        
        # Simple confidence based on variance
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        
        # Normalize confidence (lower variance = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - (variance / (mean_val + 1))))
        return confidence
    
    def _analyze_predictions(self, service_name: str, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze predictions and recommend scaling actions."""
        recommendations = {
            "action": "none",
            "reason": "",
            "confidence": 0.0,
            "estimated_replicas": 0
        }
        
        # Check CPU prediction
        cpu_pred = predictions.get('cpu', {})
        predicted_cpu = cpu_pred.get('predicted', 0)
        cpu_confidence = cpu_pred.get('confidence', 0)
        
        # Check against scaling thresholds
        for policy in self.policies.values():
            if policy.metric == 'cpu':
                if predicted_cpu > policy.scale_up_threshold and cpu_confidence > 0.7:
                    service_metrics = self.service_metrics.get(service_name)
                    current_replicas = service_metrics.current_replicas if service_metrics else 2
                    estimated_replicas = min(
                        policy.max_replicas,
                        math.ceil(current_replicas * policy.scaling_factor)
                    )
                    
                    recommendations.update({
                        "action": "scale_up",
                        "reason": f"Predicted CPU usage {predicted_cpu:.1f}% exceeds threshold {policy.scale_up_threshold}%",
                        "confidence": cpu_confidence,
                        "estimated_replicas": estimated_replicas
                    })
                    break
                elif predicted_cpu < policy.scale_down_threshold and cpu_confidence > 0.7:
                    service_metrics = self.service_metrics.get(service_name)
                    current_replicas = service_metrics.current_replicas if service_metrics else 2
                    estimated_replicas = max(
                        policy.min_replicas,
                        math.floor(current_replicas / policy.scaling_factor)
                    )
                    
                    recommendations.update({
                        "action": "scale_down",
                        "reason": f"Predicted CPU usage {predicted_cpu:.1f}% below threshold {policy.scale_down_threshold}%",
                        "confidence": cpu_confidence,
                        "estimated_replicas": estimated_replicas
                    })
                    break
        
        return recommendations
    
    def add_policy(self, policy: ScalingPolicy):
        """Add a new scaling policy."""
        self.policies[policy.name] = policy
        print(f"➕ Added scaling policy: {policy.name}")
    
    def remove_policy(self, policy_name: str):
        """Remove a scaling policy."""
        if policy_name in self.policies:
            del self.policies[policy_name]
            print(f"➖ Removed scaling policy: {policy_name}")
    
    def enable_policy(self, policy_name: str):
        """Enable a scaling policy."""
        if policy_name in self.policies:
            self.policies[policy_name].enabled = True
            print(f"✅ Enabled scaling policy: {policy_name}")
    
    def disable_policy(self, policy_name: str):
        """Disable a scaling policy."""
        if policy_name in self.policies:
            self.policies[policy_name].enabled = False
            print(f"❌ Disabled scaling policy: {policy_name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "is_active": self.is_scaling,
            "total_policies": len(self.policies),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled]),
            "total_scaling_events": len(self.scaling_history),
            "services_monitored": len(self.service_metrics),
            "last_scaling_event": self.scaling_history[-1].timestamp.isoformat() if self.scaling_history else None
        }
    
    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        recent_events = self.scaling_history[-limit:]
        
        return [
            {
                "event_id": event.event_id,
                "service_name": event.service_name,
                "direction": event.direction.value,
                "trigger_metric": event.trigger_metric,
                "trigger_value": event.trigger_value,
                "threshold": event.threshold,
                "before_replicas": event.before_replicas,
                "after_replicas": event.after_replicas,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success,
                "reason": event.reason
            }
            for event in recent_events
        ]
    
    def get_service_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current service metrics."""
        return {
            service_name: {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "requests_per_second": metrics.requests_per_second,
                "response_time": metrics.response_time,
                "current_replicas": metrics.current_replicas,
                "timestamp": metrics.timestamp.isoformat()
            }
            for service_name, metrics in self.service_metrics.items()
        }
    
    def manual_scale(self, service_name: str, replicas: int) -> bool:
        """Manually scale a service."""
        if service_name not in self.service_metrics:
            print(f"Service {service_name} not found")
            return False
        
        current_replicas = self.service_metrics[service_name].current_replicas
        
        try:
            success = self._scale_service(service_name, replicas)
            
            if success:
                # Record manual scaling event
                event = ScalingEvent(
                    event_id=f"manual-{self.scaling_counter}",
                    service_name=service_name,
                    scaling_type=ScalingType.HORIZONTAL,
                    direction=ScalingDirection.UP if replicas > current_replicas else ScalingDirection.DOWN,
                    trigger_metric="manual",
                    trigger_value=0.0,
                    threshold=0.0,
                    before_replicas=current_replicas,
                    after_replicas=replicas,
                    timestamp=datetime.now(),
                    success=True,
                    reason="Manual scaling operation"
                )
                
                self.scaling_history.append(event)
                self.scaling_counter += 1
                
                # Update service metrics
                self.service_metrics[service_name].current_replicas = replicas
                
                print(f"⚖️ Manually scaled {service_name} to {replicas} replicas")
            
            return success
            
        except Exception as e:
            print(f"Manual scaling failed: {e}")
            return False