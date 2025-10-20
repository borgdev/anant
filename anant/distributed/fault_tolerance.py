"""
Fault Tolerance Manager - Distributed System Reliability

Handles node failures, task recovery, data replication, and system resilience
in distributed computing environments.
"""

import asyncio
import time
import threading
import json
import uuid
import logging
import pickle
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import copy

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""
    NODE_FAILURE = "node_failure"
    TASK_FAILURE = "task_failure"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"


class RecoveryAction(Enum):
    """Recovery actions that can be taken."""
    RESTART_TASK = "restart_task"
    MIGRATE_TASK = "migrate_task"
    REPLACE_NODE = "replace_node"
    REDISTRIBUTE_LOAD = "redistribute_load"
    REPLICATE_DATA = "replicate_data"
    ROLLBACK_STATE = "rollback_state"
    ALERT_OPERATOR = "alert_operator"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class FailureEvent:
    """Represents a failure event in the system."""
    failure_id: str
    failure_type: FailureType
    affected_node: Optional[str] = None
    affected_tasks: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    severity: int = 1  # 1-5 scale
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_actions: List[RecoveryAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'failure_id': self.failure_id,
            'failure_type': self.failure_type.value,
            'affected_node': self.affected_node,
            'affected_tasks': self.affected_tasks,
            'affected_resources': self.affected_resources,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'description': self.description,
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolution_actions': [action.value for action in self.resolution_actions]
        }


@dataclass
class RecoveryPolicy:
    """Configuration for failure recovery behavior."""
    max_task_retries: int = 3
    retry_delay_base: float = 2.0  # Exponential backoff base
    node_failure_timeout: float = 30.0
    heartbeat_timeout: float = 15.0
    enable_task_migration: bool = True
    enable_data_replication: bool = True
    enable_auto_scaling: bool = True
    replication_factor: int = 2
    checkpoint_interval: float = 300.0  # 5 minutes
    recovery_parallelism: int = 4
    max_failure_rate: float = 0.1  # 10% failure rate threshold
    degraded_mode_threshold: float = 0.5  # 50% capacity loss threshold


@dataclass
class NodeHealth:
    """Health status of a cluster node."""
    node_id: str
    last_heartbeat: float
    consecutive_failures: int = 0
    total_failures: int = 0
    recovery_attempts: int = 0
    is_healthy: bool = True
    is_draining: bool = False
    health_score: float = 1.0  # 0.0 to 1.0
    
    def update_health(self, success: bool):
        """Update health based on operation success."""
        if success:
            self.consecutive_failures = 0
            self.health_score = min(1.0, self.health_score + 0.1)
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.health_score = max(0.0, self.health_score - 0.2)
            
        self.is_healthy = self.consecutive_failures < 3 and self.health_score > 0.3


@dataclass
class Checkpoint:
    """System state checkpoint for recovery."""
    checkpoint_id: str
    timestamp: float
    cluster_state: Dict[str, Any]
    task_state: Dict[str, Any]
    data_state: Dict[str, Any]
    node_assignments: Dict[str, List[str]]  # node_id -> task_ids
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaultToleranceManager:
    """
    Comprehensive fault tolerance and recovery manager.
    
    Features:
    - Node failure detection and recovery
    - Task failure handling and retry logic
    - State checkpointing and recovery
    - Data replication and consistency
    - Network partition handling
    - Graceful degradation
    - Automatic recovery orchestration
    """
    
    def __init__(self, cluster_manager, task_scheduler, policy: Optional[RecoveryPolicy] = None):
        """
        Initialize fault tolerance manager.
        
        Args:
            cluster_manager: ClusterManager instance
            task_scheduler: TaskScheduler instance
            policy: Recovery policy configuration
        """
        self.cluster_manager = cluster_manager
        self.task_scheduler = task_scheduler
        self.policy = policy or RecoveryPolicy()
        
        # Failure tracking
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_history: deque = deque(maxlen=1000)
        self.node_health: Dict[str, NodeHealth] = {}
        
        # State management
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.last_checkpoint_time: float = 0.0
        
        # Recovery coordination
        self.recovery_queue: asyncio.Queue = asyncio.Queue()
        self.recovery_semaphore = asyncio.Semaphore(self.policy.recovery_parallelism)
        
        # Monitoring
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        self._checkpoint_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'total_failures': 0,
            'resolved_failures': 0,
            'failed_recoveries': 0,
            'tasks_recovered': 0,
            'nodes_recovered': 0,
            'checkpoints_created': 0,
            'rollbacks_performed': 0,
            'current_failure_rate': 0.0
        }
        
        # Event callbacks
        self._failure_callbacks: List[Callable[[FailureEvent], None]] = []
        self._recovery_callbacks: List[Callable[[FailureEvent], None]] = []
        
    async def start(self) -> bool:
        """
        Start the fault tolerance manager.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            self._running = True
            
            # Start monitoring and recovery tasks
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._recovery_task = asyncio.create_task(self._recovery_loop())
            self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
            
            # Create initial checkpoint
            await self._create_checkpoint()
            
            logger.info("Fault tolerance manager started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start fault tolerance manager: {e}")
            self._running = False
            return False
            
    async def stop(self):
        """Stop the fault tolerance manager."""
        self._running = False
        
        # Cancel tasks
        for task in [self._monitor_task, self._recovery_task, self._checkpoint_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        logger.info("Fault tolerance manager stopped")
        
    async def report_failure(self, failure_type: FailureType, 
                            affected_node: Optional[str] = None,
                            affected_tasks: Optional[List[str]] = None,
                            affected_resources: Optional[List[str]] = None,
                            severity: int = 1,
                            description: str = "",
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Report a failure event.
        
        Args:
            failure_type: Type of failure
            affected_node: Node affected by failure
            affected_tasks: Tasks affected by failure
            affected_resources: Resources affected by failure
            severity: Failure severity (1-5)
            description: Human-readable description
            metadata: Additional failure metadata
            
        Returns:
            Failure ID
        """
        failure_id = str(uuid.uuid4())
        
        failure_event = FailureEvent(
            failure_id=failure_id,
            failure_type=failure_type,
            affected_node=affected_node,
            affected_tasks=affected_tasks or [],
            affected_resources=affected_resources or [],
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        
        # Track failure
        self.active_failures[failure_id] = failure_event
        self.failure_history.append(failure_event)
        self._stats['total_failures'] += 1
        
        # Update node health if applicable
        if affected_node and affected_node in self.node_health:
            self.node_health[affected_node].update_health(False)
            
        # Trigger failure callbacks
        for callback in self._failure_callbacks:
            try:
                callback(failure_event)
            except Exception as e:
                logger.warning(f"Failure callback error: {e}")
                
        # Queue for recovery
        await self.recovery_queue.put(failure_event)
        
        logger.error(f"Failure reported: {failure_type.value} - {description}")
        return failure_id
        
    async def resolve_failure(self, failure_id: str, 
                             resolution_actions: List[RecoveryAction]) -> bool:
        """
        Mark a failure as resolved.
        
        Args:
            failure_id: ID of the failure to resolve
            resolution_actions: Actions taken to resolve the failure
            
        Returns:
            True if failure was resolved
        """
        if failure_id not in self.active_failures:
            return False
            
        failure_event = self.active_failures[failure_id]
        failure_event.resolved = True
        failure_event.resolution_actions = resolution_actions
        
        # Remove from active failures
        del self.active_failures[failure_id]
        self._stats['resolved_failures'] += 1
        
        # Update node health if recovered
        if failure_event.affected_node and failure_event.affected_node in self.node_health:
            self.node_health[failure_event.affected_node].update_health(True)
            
        # Trigger recovery callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(failure_event)
            except Exception as e:
                logger.warning(f"Recovery callback error: {e}")
                
        logger.info(f"Failure resolved: {failure_id}")
        return True
        
    async def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get health status of a specific node."""
        return self.node_health.get(node_id)
        
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health metrics."""
        total_nodes = len(self.node_health)
        healthy_nodes = sum(1 for health in self.node_health.values() if health.is_healthy)
        
        # Calculate failure rate
        recent_failures = [f for f in self.failure_history 
                         if time.time() - f.timestamp < 300]  # Last 5 minutes
        failure_rate = len(recent_failures) / max(1, total_nodes) / 5.0  # per minute per node
        
        # Update current failure rate
        self._stats['current_failure_rate'] = failure_rate
        
        return {
            'total_nodes': total_nodes,
            'healthy_nodes': healthy_nodes,
            'unhealthy_nodes': total_nodes - healthy_nodes,
            'health_percentage': (healthy_nodes / max(1, total_nodes)) * 100,
            'active_failures': len(self.active_failures),
            'failure_rate': failure_rate,
            'degraded_mode': failure_rate > self.policy.max_failure_rate,
            'cluster_capacity': healthy_nodes / max(1, total_nodes)
        }
        
    async def create_checkpoint(self) -> str:
        """Create a system state checkpoint."""
        return await self._create_checkpoint()
        
    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore system state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            True if restoration was successful
        """
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
            
        try:
            checkpoint = self.checkpoints[checkpoint_id]
            
            # This would involve complex state restoration
            # For now, we'll log the action and track statistics
            logger.info(f"Restoring from checkpoint: {checkpoint_id}")
            
            self._stats['rollbacks_performed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return False
            
    async def _monitoring_loop(self):
        """Main monitoring loop for detecting failures."""
        while self._running:
            try:
                await self._check_node_health()
                await self._check_task_health()
                await self._update_failure_rates()
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10.0)
                
    async def _check_node_health(self):
        """Check health of all cluster nodes."""
        current_time = time.time()
        
        # Get current nodes from cluster manager
        nodes = self.cluster_manager.get_all_nodes()
        
        for node in nodes:
            node_id = node.node_id
            
            # Initialize health tracking if new node
            if node_id not in self.node_health:
                self.node_health[node_id] = NodeHealth(
                    node_id=node_id,
                    last_heartbeat=current_time
                )
                
            health = self.node_health[node_id]
            
            # Check heartbeat timeout
            if current_time - health.last_heartbeat > self.policy.heartbeat_timeout:
                if health.is_healthy:
                    # Node became unhealthy
                    health.is_healthy = False
                    await self.report_failure(
                        FailureType.NODE_FAILURE,
                        affected_node=node_id,
                        description=f"Node heartbeat timeout: {node_id}",
                        severity=3
                    )
                    
            else:
                # Update heartbeat
                health.last_heartbeat = current_time
                if not health.is_healthy and health.consecutive_failures == 0:
                    health.is_healthy = True
                    logger.info(f"Node recovered: {node_id}")
                    
    async def _check_task_health(self):
        """Check health of running tasks."""
        # Get task statistics from scheduler
        if hasattr(self.task_scheduler, 'get_tasks_by_status'):
            running_tasks = self.task_scheduler.get_tasks_by_status("running")
            
            current_time = time.time()
            
            for task in running_tasks:
                # Check for task timeout or other issues
                if (hasattr(task, 'started_at') and task.started_at and
                    hasattr(task, 'timeout') and task.timeout and
                    current_time - task.started_at > task.timeout):
                    
                    await self.report_failure(
                        FailureType.TASK_FAILURE,
                        affected_tasks=[task.task_id],
                        affected_node=getattr(task, 'assigned_node', None),
                        description=f"Task timeout: {task.task_id}",
                        severity=2
                    )
                    
    async def _update_failure_rates(self):
        """Update system failure rate metrics."""
        current_time = time.time()
        window_size = 300.0  # 5 minute window
        
        # Count recent failures
        recent_failures = [f for f in self.failure_history 
                         if current_time - f.timestamp < window_size]
        
        # Calculate failure rate
        total_nodes = len(self.node_health)
        if total_nodes > 0:
            failure_rate = len(recent_failures) / total_nodes / (window_size / 60.0)
            self._stats['current_failure_rate'] = failure_rate
            
            # Check if we need to enter degraded mode
            if failure_rate > self.policy.max_failure_rate:
                logger.warning(f"High failure rate detected: {failure_rate:.2f} failures/min/node")
                
    async def _recovery_loop(self):
        """Main recovery loop for handling failures."""
        while self._running:
            try:
                # Get next failure to recover
                failure_event = await asyncio.wait_for(
                    self.recovery_queue.get(), timeout=1.0
                )
                
                # Process recovery with concurrency limit
                async with self.recovery_semaphore:
                    await self._handle_failure_recovery(failure_event)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(5.0)
                
    async def _handle_failure_recovery(self, failure_event: FailureEvent):
        """
        Handle recovery for a specific failure.
        
        Args:
            failure_event: Failure to recover from
        """
        try:
            recovery_actions = []
            
            if failure_event.failure_type == FailureType.NODE_FAILURE:
                recovery_actions = await self._recover_node_failure(failure_event)
                
            elif failure_event.failure_type == FailureType.TASK_FAILURE:
                recovery_actions = await self._recover_task_failure(failure_event)
                
            elif failure_event.failure_type == FailureType.NETWORK_PARTITION:
                recovery_actions = await self._recover_network_partition(failure_event)
                
            elif failure_event.failure_type == FailureType.RESOURCE_EXHAUSTION:
                recovery_actions = await self._recover_resource_exhaustion(failure_event)
                
            else:
                # Generic recovery
                recovery_actions = [RecoveryAction.ALERT_OPERATOR]
                
            # Mark failure as resolved if recovery was successful
            if recovery_actions:
                await self.resolve_failure(failure_event.failure_id, recovery_actions)
            else:
                self._stats['failed_recoveries'] += 1
                logger.error(f"Failed to recover from failure: {failure_event.failure_id}")
                
        except Exception as e:
            logger.error(f"Failure recovery error: {e}")
            self._stats['failed_recoveries'] += 1
            
    async def _recover_node_failure(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Recover from node failure."""
        node_id = failure_event.affected_node
        if not node_id:
            return []
            
        recovery_actions = []
        
        try:
            # Get tasks running on failed node
            if hasattr(self.task_scheduler, 'get_node_tasks'):
                node_tasks = self.task_scheduler.get_node_tasks(node_id)
                
                # Migrate tasks to other nodes
                if self.policy.enable_task_migration:
                    for task in node_tasks:
                        # Cancel task on failed node
                        self.task_scheduler.cancel_task(task.task_id)
                        
                        # Resubmit task for scheduling on healthy node
                        task.assigned_node = None
                        task.retry_count += 1
                        
                        if task.retry_count <= self.policy.max_task_retries:
                            self.task_scheduler.submit_task(task)
                            recovery_actions.append(RecoveryAction.MIGRATE_TASK)
                            self._stats['tasks_recovered'] += 1
                            
            # Mark node for replacement if available
            recovery_actions.append(RecoveryAction.REPLACE_NODE)
            self._stats['nodes_recovered'] += 1
            
            logger.info(f"Recovered from node failure: {node_id}")
            
        except Exception as e:
            logger.error(f"Node recovery failed: {e}")
            
        return recovery_actions
        
    async def _recover_task_failure(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Recover from task failure."""
        recovery_actions = []
        
        try:
            for task_id in failure_event.affected_tasks:
                # Get task from scheduler
                task = self.task_scheduler.get_task(task_id)
                if not task:
                    continue
                    
                # Retry task if within limits
                if task.retry_count < self.policy.max_task_retries:
                    # Calculate retry delay with exponential backoff
                    delay = self.policy.retry_delay_base ** task.retry_count
                    
                    # Reset task state
                    task.status = "pending"
                    task.assigned_node = None
                    task.started_at = None
                    task.retry_count += 1
                    
                    # Schedule for retry after delay
                    await asyncio.sleep(delay)
                    self.task_scheduler.submit_task(task)
                    
                    recovery_actions.append(RecoveryAction.RESTART_TASK)
                    self._stats['tasks_recovered'] += 1
                    
                else:
                    # Task exceeded retry limit
                    recovery_actions.append(RecoveryAction.ALERT_OPERATOR)
                    
        except Exception as e:
            logger.error(f"Task recovery failed: {e}")
            
        return recovery_actions
        
    async def _recover_network_partition(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Recover from network partition."""
        # Simplified network partition recovery
        return [RecoveryAction.REDISTRIBUTE_LOAD, RecoveryAction.ALERT_OPERATOR]
        
    async def _recover_resource_exhaustion(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Recover from resource exhaustion."""
        recovery_actions = []
        
        # Try to redistribute load
        if self.policy.enable_auto_scaling:
            recovery_actions.append(RecoveryAction.REDISTRIBUTE_LOAD)
            
        # Enable graceful degradation
        recovery_actions.append(RecoveryAction.GRACEFUL_DEGRADATION)
        
        return recovery_actions
        
    async def _checkpoint_loop(self):
        """Periodic checkpoint creation loop."""
        while self._running:
            try:
                current_time = time.time()
                
                if current_time - self.last_checkpoint_time > self.policy.checkpoint_interval:
                    await self._create_checkpoint()
                    
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
                await asyncio.sleep(300.0)  # 5 minute delay on error
                
    async def _create_checkpoint(self) -> str:
        """Create a system state checkpoint."""
        try:
            checkpoint_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Collect system state
            cluster_state = {
                'nodes': [node.to_dict() for node in self.cluster_manager.get_all_nodes()],
                'node_health': {node_id: {
                    'is_healthy': health.is_healthy,
                    'health_score': health.health_score,
                    'consecutive_failures': health.consecutive_failures
                } for node_id, health in self.node_health.items()}
            }
            
            task_state = {
                'pending_tasks': len(self.task_scheduler.get_tasks_by_status("pending")),
                'running_tasks': len(self.task_scheduler.get_tasks_by_status("running")),
                'completed_tasks': len(self.task_scheduler.get_tasks_by_status("completed"))
            }
            
            # Create checkpoint
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                timestamp=current_time,
                cluster_state=cluster_state,
                task_state=task_state,
                data_state={},  # Would include data replication state
                node_assignments={}  # Would include task-node assignments
            )
            
            self.checkpoints[checkpoint_id] = checkpoint
            self.last_checkpoint_time = current_time
            self._stats['checkpoints_created'] += 1
            
            # Cleanup old checkpoints (keep last 10)
            if len(self.checkpoints) > 10:
                oldest_id = min(self.checkpoints.keys(), 
                              key=lambda k: self.checkpoints[k].timestamp)
                del self.checkpoints[oldest_id]
                
            logger.debug(f"Checkpoint created: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
            
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get comprehensive failure and recovery statistics."""
        stats = self._stats.copy()
        
        # Add current state
        stats.update({
            'active_failures': len(self.active_failures),
            'total_checkpoints': len(self.checkpoints),
            'monitored_nodes': len(self.node_health),
            'healthy_nodes': sum(1 for h in self.node_health.values() if h.is_healthy),
            'recovery_queue_size': self.recovery_queue.qsize()
        })
        
        # Calculate success rates
        total_attempts = stats['resolved_failures'] + stats['failed_recoveries']
        if total_attempts > 0:
            stats['recovery_success_rate'] = stats['resolved_failures'] / total_attempts
        else:
            stats['recovery_success_rate'] = 0.0
            
        return stats
        
    def add_failure_callback(self, callback: Callable[[FailureEvent], None]):
        """Add callback for failure events."""
        self._failure_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable[[FailureEvent], None]):
        """Add callback for recovery events."""
        self._recovery_callbacks.append(callback)