"""
Task Scheduler - Intelligent Task Distribution and Load Balancing

Handles task scheduling, load balancing, and work distribution across
cluster nodes with support for priorities, dependencies, and resource requirements.
"""

import asyncio
import time
import threading
import heapq
import uuid
from typing import Dict, List, Optional, Set, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RESOURCE_AWARE = "resource_aware"
    LOCALITY_AWARE = "locality_aware"
    PRIORITY_FIRST = "priority_first"
    FAIR_SHARE = "fair_share"


@dataclass
class ResourceRequirements:
    """Resource requirements for a task."""
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    disk_gb: float = 0.1
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    estimated_runtime: float = 60.0  # seconds
    network_intensive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'disk_gb': self.disk_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'estimated_runtime': self.estimated_runtime,
            'network_intensive': self.network_intensive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceRequirements':
        return cls(**data)


@dataclass
class Task:
    """Represents a computational task to be executed."""
    task_id: str
    function_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    dependencies: Set[str] = field(default_factory=set)  # task_ids this task depends on
    group_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'function_name': self.function_name,
            'priority': self.priority.value,
            'requirements': self.requirements.to_dict(),
            'dependencies': list(self.dependencies),
            'group_id': self.group_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'scheduled_at': self.scheduled_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'assigned_node': self.assigned_node,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'metadata': self.metadata,
            'error': self.error
        }


@dataclass
class SchedulingPolicy:
    """Configuration for task scheduling behavior."""
    strategy: SchedulingStrategy = SchedulingStrategy.RESOURCE_AWARE
    max_queue_size: int = 10000
    max_tasks_per_node: int = 100
    enable_work_stealing: bool = True
    enable_preemption: bool = False
    locality_weight: float = 0.3
    load_weight: float = 0.4
    resource_weight: float = 0.3
    retry_delay: float = 5.0
    heartbeat_interval: float = 10.0


class TaskScheduler:
    """
    Advanced task scheduler with multiple scheduling strategies.
    
    Features:
    - Priority-based task queuing
    - Resource-aware scheduling
    - Load balancing across nodes
    - Task dependency management
    - Automatic retry with backoff
    - Work stealing for load balancing
    - Real-time monitoring and metrics
    """
    
    def __init__(self, cluster_manager, policy: Optional[SchedulingPolicy] = None):
        """
        Initialize task scheduler.
        
        Args:
            cluster_manager: ClusterManager instance for node information
            policy: Scheduling policy configuration
        """
        self.cluster_manager = cluster_manager
        self.policy = policy or SchedulingPolicy()
        
        # Task storage
        self.pending_tasks: List[Task] = []  # Priority queue
        self.queued_tasks: Dict[str, Task] = {}  # task_id -> task
        self.running_tasks: Dict[str, Task] = {}  # task_id -> task
        self.completed_tasks: Dict[str, Task] = {}  # task_id -> task
        self.failed_tasks: Dict[str, Task] = {}  # task_id -> task
        
        # Node task assignments
        self.node_tasks: Dict[str, Set[str]] = defaultdict(set)  # node_id -> task_ids
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> dependent_task_ids
        
        # Synchronization
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'total_execution_time': 0.0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Event callbacks
        self._task_completed_callbacks: List[Callable[[Task], None]] = []
        self._task_failed_callbacks: List[Callable[[Task], None]] = []
        
    def start(self) -> bool:
        """
        Start the task scheduler.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduling_loop, daemon=True
            )
            self._scheduler_thread.start()
            
            logger.info("Task scheduler started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start task scheduler: {e}")
            self._running = False
            return False
            
    def stop(self):
        """Stop the task scheduler."""
        self._running = False
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
            
        logger.info("Task scheduler stopped")
        
    def submit_task(self, task: Task) -> bool:
        """
        Submit a task for execution.
        
        Args:
            task: Task to submit
            
        Returns:
            True if task submitted successfully
        """
        with self._lock:
            if len(self.pending_tasks) + len(self.queued_tasks) >= self.policy.max_queue_size:
                logger.warning(f"Task queue at capacity: {self.policy.max_queue_size}")
                return False
                
            # Validate dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks and dep_id not in self.running_tasks:
                    logger.warning(f"Task {task.task_id} has unresolved dependency: {dep_id}")
                    
            # Add to pending queue
            task.status = TaskStatus.PENDING
            heapq.heappush(self.pending_tasks, task)
            
            # Update dependency tracking
            for dep_id in task.dependencies:
                if dep_id not in self.task_dependencies:
                    self.task_dependencies[dep_id] = set()
                self.task_dependencies[dep_id].add(task.task_id)
                
            self._stats['tasks_submitted'] += 1
            logger.debug(f"Task submitted: {task.task_id}")
            return True
            
    def submit_batch(self, tasks: List[Task]) -> List[bool]:
        """
        Submit multiple tasks as a batch.
        
        Args:
            tasks: List of tasks to submit
            
        Returns:
            List of success flags for each task
        """
        results = []
        for task in tasks:
            results.append(self.submit_task(task))
        return results
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled
        """
        with self._lock:
            # Check pending tasks
            for i, task in enumerate(self.pending_tasks):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    del self.pending_tasks[i]
                    heapq.heapify(self.pending_tasks)
                    logger.info(f"Cancelled pending task: {task_id}")
                    return True
                    
            # Check queued tasks
            if task_id in self.queued_tasks:
                task = self.queued_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self.queued_tasks[task_id]
                logger.info(f"Cancelled queued task: {task_id}")
                return True
                
            # For running tasks, mark for cancellation
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                # Note: Actual cancellation depends on worker implementation
                logger.info(f"Marked running task for cancellation: {task_id}")
                return True
                
            return False
            
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self._lock:
            # Check all task collections
            for task_dict in [self.queued_tasks, self.running_tasks, 
                            self.completed_tasks, self.failed_tasks]:
                if task_id in task_dict:
                    return task_dict[task_id]
                    
            # Check pending tasks
            for task in self.pending_tasks:
                if task.task_id == task_id:
                    return task
                    
            return None
            
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        with self._lock:
            if status == TaskStatus.PENDING:
                return list(self.pending_tasks)
            elif status == TaskStatus.QUEUED:
                return list(self.queued_tasks.values())
            elif status == TaskStatus.RUNNING:
                return list(self.running_tasks.values())
            elif status == TaskStatus.COMPLETED:
                return list(self.completed_tasks.values())
            elif status == TaskStatus.FAILED:
                return list(self.failed_tasks.values())
            else:
                return []
                
    def get_node_tasks(self, node_id: str) -> List[Task]:
        """Get tasks assigned to a specific node."""
        with self._lock:
            task_ids = self.node_tasks.get(node_id, set())
            tasks = []
            
            for task_id in task_ids:
                if task_id in self.running_tasks:
                    tasks.append(self.running_tasks[task_id])
                elif task_id in self.queued_tasks:
                    tasks.append(self.queued_tasks[task_id])
                    
            return tasks
            
    def _scheduling_loop(self):
        """Main scheduling loop running in background thread."""
        while self._running:
            try:
                self._schedule_ready_tasks()
                self._check_running_tasks()
                self._process_completed_dependencies()
                time.sleep(self.policy.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}")
                time.sleep(1.0)
                
    def _schedule_ready_tasks(self):
        """Schedule tasks that are ready to run."""
        with self._lock:
            ready_tasks = []
            
            # Find tasks with satisfied dependencies
            while self.pending_tasks:
                task = heapq.heappop(self.pending_tasks)
                
                if self._are_dependencies_satisfied(task):
                    ready_tasks.append(task)
                else:
                    # Put back in queue
                    heapq.heappush(self.pending_tasks, task)
                    break
                    
            # Schedule ready tasks
            for task in ready_tasks:
                self._schedule_task(task)
                
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
        
    def _schedule_task(self, task: Task) -> bool:
        """
        Schedule a single task to an appropriate node.
        
        Args:
            task: Task to schedule
            
        Returns:
            True if task was scheduled
        """
        # Get available nodes
        available_nodes = self.cluster_manager.get_available_nodes(
            min_cpu_cores=int(task.requirements.cpu_cores),
            min_memory_gb=task.requirements.memory_gb
        )
        
        if not available_nodes:
            # No nodes available, keep task pending
            heapq.heappush(self.pending_tasks, task)
            return False
            
        # Select best node based on strategy
        selected_node = self._select_node(task, available_nodes)
        
        if not selected_node:
            heapq.heappush(self.pending_tasks, task)
            return False
            
        # Check node capacity
        node_task_count = len(self.node_tasks.get(selected_node.node_id, set()))
        if node_task_count >= self.policy.max_tasks_per_node:
            heapq.heappush(self.pending_tasks, task)
            return False
            
        # Assign task to node
        task.status = TaskStatus.QUEUED
        task.assigned_node = selected_node.node_id
        task.scheduled_at = time.time()
        
        self.queued_tasks[task.task_id] = task
        self.node_tasks[selected_node.node_id].add(task.task_id)
        
        logger.debug(f"Task {task.task_id} scheduled to node {selected_node.node_id}")
        return True
        
    def _select_node(self, task: Task, available_nodes: List[Any]) -> Optional[Any]:
        """
        Select the best node for a task based on scheduling strategy.
        
        Args:
            task: Task to schedule
            available_nodes: List of available nodes
            
        Returns:
            Selected node or None
        """
        if not available_nodes:
            return None
            
        if self.policy.strategy == SchedulingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return available_nodes[int(task.created_at) % len(available_nodes)]
            
        elif self.policy.strategy == SchedulingStrategy.LEAST_LOADED:
            # Select node with least CPU usage
            return min(available_nodes, key=lambda n: n.usage.cpu_percent)
            
        elif self.policy.strategy == SchedulingStrategy.RESOURCE_AWARE:
            # Score nodes based on resource fit
            best_node = None
            best_score = float('-inf')
            
            for node in available_nodes:
                score = self._calculate_resource_score(task, node)
                if score > best_score:
                    best_score = score
                    best_node = node
                    
            return best_node
            
        elif self.policy.strategy == SchedulingStrategy.PRIORITY_FIRST:
            # For high priority tasks, prefer nodes with more resources
            if task.priority.value <= TaskPriority.HIGH.value:
                return max(available_nodes, key=lambda n: n.capacity.cpu_cores)
            else:
                return min(available_nodes, key=lambda n: n.usage.cpu_percent)
                
        else:
            # Default to least loaded
            return min(available_nodes, key=lambda n: n.usage.cpu_percent)
            
    def _calculate_resource_score(self, task: Task, node) -> float:
        """
        Calculate how well a node fits a task's resource requirements.
        
        Args:
            task: Task to score
            node: Node to evaluate
            
        Returns:
            Score (higher is better)
        """
        # Resource availability score
        cpu_fit = 1.0 - (node.usage.cpu_percent / 100.0)
        memory_fit = 1.0 - (node.usage.memory_percent / 100.0)
        
        # Resource requirement satisfaction
        cpu_capacity = node.capacity.cpu_cores >= task.requirements.cpu_cores
        memory_capacity = node.capacity.memory_gb >= task.requirements.memory_gb
        gpu_capacity = node.capacity.gpu_count >= task.requirements.gpu_count
        
        if not (cpu_capacity and memory_capacity and gpu_capacity):
            return -1.0  # Cannot satisfy requirements
            
        # Load balancing score
        node_load = len(self.node_tasks.get(node.node_id, set()))
        load_score = 1.0 / (1.0 + node_load)
        
        # Combine scores
        resource_score = (cpu_fit + memory_fit) / 2.0
        final_score = (
            self.policy.resource_weight * resource_score +
            self.policy.load_weight * load_score
        )
        
        return final_score
        
    def _check_running_tasks(self):
        """Check status of running tasks and handle timeouts."""
        current_time = time.time()
        
        with self._lock:
            timed_out_tasks = []
            
            for task_id, task in self.running_tasks.items():
                # Check for timeout
                if (task.timeout and task.started_at and
                    current_time - task.started_at > task.timeout):
                    timed_out_tasks.append(task_id)
                    
            # Handle timed out tasks
            for task_id in timed_out_tasks:
                self._handle_task_timeout(task_id)
                
    def _handle_task_timeout(self, task_id: str):
        """Handle task timeout."""
        task = self.running_tasks.get(task_id)
        if not task:
            return
            
        logger.warning(f"Task timed out: {task_id}")
        task.error = f"Task timed out after {task.timeout} seconds"
        self._handle_task_failure(task)
        
    def _process_completed_dependencies(self):
        """Process tasks whose dependencies have completed."""
        with self._lock:
            newly_ready = []
            
            # Check if any pending tasks now have satisfied dependencies
            remaining_pending = []
            
            while self.pending_tasks:
                task = heapq.heappop(self.pending_tasks)
                
                if self._are_dependencies_satisfied(task):
                    newly_ready.append(task)
                else:
                    remaining_pending.append(task)
                    
            # Restore pending tasks
            self.pending_tasks = remaining_pending
            heapq.heapify(self.pending_tasks)
            
            # Schedule newly ready tasks
            for task in newly_ready:
                if not self._schedule_task(task):
                    # Couldn't schedule, put back in pending
                    heapq.heappush(self.pending_tasks, task)
                    
    def task_started(self, task_id: str) -> bool:
        """
        Mark a task as started.
        
        Args:
            task_id: ID of the task that started
            
        Returns:
            True if task was found and updated
        """
        with self._lock:
            if task_id in self.queued_tasks:
                task = self.queued_tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # Move to running tasks
                del self.queued_tasks[task_id]
                self.running_tasks[task_id] = task
                
                logger.debug(f"Task started: {task_id}")
                return True
                
            return False
            
    def task_completed(self, task_id: str, result: Any = None) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the completed task
            result: Task result
            
        Returns:
            True if task was found and updated
        """
        with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                
                # Update statistics
                if task.started_at:
                    execution_time = task.completed_at - task.started_at
                    self._stats['total_execution_time'] += execution_time
                    
                if task.scheduled_at:
                    queue_time = (task.started_at or task.completed_at) - task.scheduled_at
                    # Update running average of queue time
                    current_avg = self._stats['average_queue_time']
                    count = self._stats['tasks_completed']
                    self._stats['average_queue_time'] = (current_avg * count + queue_time) / (count + 1)
                    
                self._stats['tasks_completed'] += 1
                
                # Move to completed tasks
                del self.running_tasks[task_id]
                self.completed_tasks[task_id] = task
                
                # Remove from node assignment
                if task.assigned_node:
                    self.node_tasks[task.assigned_node].discard(task_id)
                    
                # Trigger completion callbacks
                for callback in self._task_completed_callbacks:
                    try:
                        callback(task)
                    except Exception as e:
                        logger.warning(f"Task completion callback failed: {e}")
                        
                logger.debug(f"Task completed: {task_id}")
                return True
                
            return False
            
    def task_failed(self, task_id: str, error: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: ID of the failed task
            error: Error message
            
        Returns:
            True if task was found and updated
        """
        with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.error = error
                self._handle_task_failure(task)
                return True
                
            return False
            
    def _handle_task_failure(self, task: Task):
        """Handle task failure with retry logic."""
        with self._lock:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
                
            # Remove from node assignment
            if task.assigned_node:
                self.node_tasks[task.assigned_node].discard(task.task_id)
                
            # Check if task should be retried
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                task.assigned_node = None
                task.started_at = None
                task.scheduled_at = None
                
                # Add delay before retry
                retry_delay = self.policy.retry_delay * (2 ** (task.retry_count - 1))
                task.created_at = time.time() + retry_delay
                
                # Put back in pending queue
                heapq.heappush(self.pending_tasks, task)
                
                self._stats['tasks_retried'] += 1
                logger.info(f"Task {task.task_id} scheduled for retry {task.retry_count}/{task.max_retries}")
                
            else:
                # Task failed permanently
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self.failed_tasks[task.task_id] = task
                
                self._stats['tasks_failed'] += 1
                
                # Trigger failure callbacks
                for callback in self._task_failed_callbacks:
                    try:
                        callback(task)
                    except Exception as e:
                        logger.warning(f"Task failure callback failed: {e}")
                        
                logger.error(f"Task failed permanently: {task.task_id} - {task.error}")
                
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Add current queue sizes
            stats.update({
                'pending_tasks': len(self.pending_tasks),
                'queued_tasks': len(self.queued_tasks),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_tasks': (len(self.pending_tasks) + len(self.queued_tasks) + 
                              len(self.running_tasks) + len(self.completed_tasks) + 
                              len(self.failed_tasks))
            })
            
            # Calculate success rate
            total_finished = stats['tasks_completed'] + stats['tasks_failed']
            if total_finished > 0:
                stats['success_rate'] = stats['tasks_completed'] / total_finished
            else:
                stats['success_rate'] = 0.0
                
            # Calculate average execution time
            if stats['tasks_completed'] > 0:
                stats['average_execution_time'] = (
                    stats['total_execution_time'] / stats['tasks_completed']
                )
            else:
                stats['average_execution_time'] = 0.0
                
            return stats
            
    def add_task_completed_callback(self, callback: Callable[[Task], None]):
        """Add callback for task completion events."""
        self._task_completed_callbacks.append(callback)
        
    def add_task_failed_callback(self, callback: Callable[[Task], None]):
        """Add callback for task failure events."""
        self._task_failed_callbacks.append(callback)