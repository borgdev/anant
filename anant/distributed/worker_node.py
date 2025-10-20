"""
Worker Node - Distributed Task Execution

Handles task execution, resource monitoring, and communication with
the cluster manager and task scheduler.
"""

import asyncio
import time
import threading
import psutil
import socket
import json
import uuid
import logging
import pickle
import traceback
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import signal
import os

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker node status."""
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    DRAINING = "draining"
    STOPPING = "stopping"
    FAILED = "failed"


class ExecutionMode(Enum):
    """Task execution modes."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    CUDA = "cuda"


@dataclass
class WorkerConfig:
    """Worker node configuration."""
    worker_id: Optional[str] = None
    host: str = "localhost"
    port: int = 0  # 0 = auto-assign
    max_concurrent_tasks: int = 4
    heartbeat_interval: float = 10.0
    task_timeout: float = 300.0
    enable_gpu: bool = True
    enable_multiprocessing: bool = True
    resource_monitoring_interval: float = 5.0
    max_memory_usage: float = 0.8  # 80% of available memory
    max_cpu_usage: float = 0.9     # 90% of available CPU
    execution_mode: ExecutionMode = ExecutionMode.THREAD
    cleanup_interval: float = 60.0
    log_level: str = "INFO"


@dataclass
class TaskExecution:
    """Represents a task execution on the worker."""
    task_id: str
    future: Future
    started_at: float
    timeout: Optional[float] = None
    execution_mode: ExecutionMode = ExecutionMode.THREAD
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0


class FunctionRegistry:
    """Registry for functions that can be executed by workers."""
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._modules: Dict[str, Any] = {}
        
    def register_function(self, name: str, func: Callable):
        """Register a function for execution."""
        self._functions[name] = func
        logger.debug(f"Registered function: {name}")
        
    def register_module(self, name: str, module: Any):
        """Register a module for function lookups."""
        self._modules[name] = module
        logger.debug(f"Registered module: {name}")
        
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a registered function."""
        # Direct function lookup
        if name in self._functions:
            return self._functions[name]
            
        # Module function lookup (e.g., "module.function")
        if '.' in name:
            module_name, func_name = name.rsplit('.', 1)
            if module_name in self._modules:
                module = self._modules[module_name]
                if hasattr(module, func_name):
                    return getattr(module, func_name)
                    
        return None
        
    def list_functions(self) -> List[str]:
        """List all registered functions."""
        functions = list(self._functions.keys())
        
        # Add module functions
        for module_name, module in self._modules.items():
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        functions.append(f"{module_name}.{attr_name}")
                        
        return functions


class WorkerNode:
    """
    Distributed worker node for task execution.
    
    Features:
    - Multi-threaded and multi-process task execution
    - Resource monitoring and limits
    - Heartbeat communication with cluster
    - Task timeout handling
    - GPU acceleration support
    - Graceful shutdown and cleanup
    """
    
    def __init__(self, config: Optional[WorkerConfig] = None):
        """
        Initialize worker node.
        
        Args:
            config: Worker configuration
        """
        self.config = config or WorkerConfig()
        self.worker_id = self.config.worker_id or str(uuid.uuid4())
        
        # Status and state
        self.status = WorkerStatus.STARTING
        self._running = False
        self._shutdown_requested = False
        
        # Task execution
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.function_registry = FunctionRegistry()
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        
        # Communication
        self._socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # Monitoring
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._resource_stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_mb': 0.0,
            'disk_usage_percent': 0.0,
            'gpu_count': 0,
            'gpu_memory_mb': 0.0
        }
        
        # Cluster communication
        self.cluster_manager = None
        self.task_scheduler = None
        
        # Event callbacks
        self._task_started_callbacks: List[Callable[[str], None]] = []
        self._task_completed_callbacks: List[Callable[[str, Any], None]] = []
        self._task_failed_callbacks: List[Callable[[str, str], None]] = []
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def start(self) -> bool:
        """
        Start the worker node.
        
        Returns:
            True if started successfully
        """
        try:
            logger.info(f"Starting worker node: {self.worker_id}")
            
            # Initialize executors
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_tasks,
                thread_name_prefix=f"worker-{self.worker_id}"
            )
            
            if self.config.enable_multiprocessing:
                self._process_executor = ProcessPoolExecutor(
                    max_workers=min(self.config.max_concurrent_tasks, mp.cpu_count())
                )
                
            # Start communication server
            self._start_server()
            
            # Start background threads
            self._running = True
            
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()
            
            self._resource_monitor_thread = threading.Thread(
                target=self._resource_monitor_loop, daemon=True
            )
            self._resource_monitor_thread.start()
            
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()
            
            self.status = WorkerStatus.IDLE
            logger.info(f"Worker node started: {self.worker_id} on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start worker node: {e}")
            self.status = WorkerStatus.FAILED
            return False
            
    def stop(self):
        """Stop the worker node gracefully."""
        logger.info(f"Stopping worker node: {self.worker_id}")
        self.status = WorkerStatus.STOPPING
        self._shutdown_requested = True
        
        # Wait for active tasks to complete or timeout
        start_time = time.time()
        timeout = 30.0  # 30 second graceful shutdown timeout
        
        while self.active_tasks and time.time() - start_time < timeout:
            logger.info(f"Waiting for {len(self.active_tasks)} tasks to complete...")
            time.sleep(1.0)
            
        # Cancel remaining tasks
        with self._lock:
            for task_id, execution in list(self.active_tasks.items()):
                if not execution.future.done():
                    execution.future.cancel()
                    logger.warning(f"Cancelled task during shutdown: {task_id}")
                    
        # Stop background threads
        self._running = False
        
        # Cleanup executors
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            
        if self._process_executor:
            self._process_executor.shutdown(wait=True)
            
        # Close server socket
        if self._socket:
            self._socket.close()
            
        logger.info(f"Worker node stopped: {self.worker_id}")
        
    def execute_task(self, task_id: str, function_name: str, 
                    args: Tuple[Any, ...], kwargs: Dict[str, Any],
                    timeout: Optional[float] = None,
                    execution_mode: Optional[ExecutionMode] = None) -> bool:
        """
        Execute a task on this worker.
        
        Args:
            task_id: Unique task identifier
            function_name: Name of function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            timeout: Task timeout in seconds
            execution_mode: Execution mode override
            
        Returns:
            True if task was accepted for execution
        """
        with self._lock:
            if task_id in self.active_tasks:
                logger.warning(f"Task already running: {task_id}")
                return False
                
            if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                logger.warning(f"Worker at capacity: {len(self.active_tasks)}")
                return False
                
            # Get function
            func = self.function_registry.get_function(function_name)
            if not func:
                logger.error(f"Unknown function: {function_name}")
                return False
                
            # Determine execution mode
            exec_mode = execution_mode or self.config.execution_mode
            
            # Submit task
            try:
                if not self._thread_executor:
                    logger.error("Thread executor not initialized")
                    return False
                    
                if exec_mode == ExecutionMode.THREAD:
                    future = self._thread_executor.submit(
                        self._execute_function, func, args, kwargs
                    )
                elif exec_mode == ExecutionMode.PROCESS and self._process_executor:
                    future = self._process_executor.submit(
                        self._execute_function, func, args, kwargs
                    )
                elif exec_mode == ExecutionMode.ASYNC:
                    # For async functions, wrap in thread executor
                    future = self._thread_executor.submit(
                        self._execute_async_function, func, args, kwargs
                    )
                else:
                    # Default to thread execution
                    future = self._thread_executor.submit(
                        self._execute_function, func, args, kwargs
                    )
                    
                # Create execution record
                execution = TaskExecution(
                    task_id=task_id,
                    future=future,
                    started_at=time.time(),
                    timeout=timeout or self.config.task_timeout,
                    execution_mode=exec_mode
                )
                
                self.active_tasks[task_id] = execution
                
                # Add completion callback
                future.add_done_callback(
                    lambda f: self._task_completed(task_id, f)
                )
                
                # Update status
                if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                    self.status = WorkerStatus.BUSY
                else:
                    self.status = WorkerStatus.ACTIVE
                    
                # Trigger start callbacks
                for callback in self._task_started_callbacks:
                    try:
                        callback(task_id)
                    except Exception as e:
                        logger.warning(f"Task start callback failed: {e}")
                        
                logger.debug(f"Task started: {task_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start task {task_id}: {e}")
                return False
                
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled
        """
        with self._lock:
            if task_id not in self.active_tasks:
                return False
                
            execution = self.active_tasks[task_id]
            if execution.future.cancel():
                del self.active_tasks[task_id]
                self._update_status()
                logger.info(f"Task cancelled: {task_id}")
                return True
                
            return False
            
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self._lock:
            if task_id not in self.active_tasks:
                return None
                
            execution = self.active_tasks[task_id]
            return {
                'task_id': task_id,
                'started_at': execution.started_at,
                'running_time': time.time() - execution.started_at,
                'execution_mode': execution.execution_mode.value,
                'done': execution.future.done(),
                'cancelled': execution.future.cancelled(),
                'memory_usage': execution.memory_usage,
                'cpu_usage': execution.cpu_usage
            }
            
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        with self._lock:
            stats = {
                'worker_id': self.worker_id,
                'status': self.status.value,
                'host': getattr(self, 'host', 'unknown'),
                'port': getattr(self, 'port', 0),
                'active_tasks': len(self.active_tasks),
                'max_concurrent_tasks': self.config.max_concurrent_tasks,
                'uptime': time.time() - getattr(self, '_start_time', time.time()),
                'resource_stats': self._resource_stats.copy(),
                'registered_functions': len(self.function_registry.list_functions())
            }
            
            # Add task details
            stats['tasks'] = []
            for task_id, execution in self.active_tasks.items():
                task_info = {
                    'task_id': task_id,
                    'running_time': time.time() - execution.started_at,
                    'execution_mode': execution.execution_mode.value
                }
                stats['tasks'].append(task_info)
                
            return stats
            
    def _execute_function(self, func: Callable, args: Tuple[Any, ...], 
                         kwargs: Dict[str, Any]) -> Any:
        """Execute a function safely with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
            
    async def _execute_async_function(self, func: Callable, args: Tuple[Any, ...], 
                                    kwargs: Dict[str, Any]) -> Any:
        """Execute an async function."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Async function execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion."""
        with self._lock:
            if task_id not in self.active_tasks:
                return
                
            execution = self.active_tasks[task_id]
            del self.active_tasks[task_id]
            
            try:
                if future.cancelled():
                    logger.info(f"Task cancelled: {task_id}")
                elif future.exception():
                    error = str(future.exception())
                    logger.error(f"Task failed: {task_id} - {error}")
                    
                    # Trigger failure callbacks
                    for callback in self._task_failed_callbacks:
                        try:
                            callback(task_id, error)
                        except Exception as e:
                            logger.warning(f"Task failure callback failed: {e}")
                            
                else:
                    result = future.result()
                    logger.debug(f"Task completed: {task_id}")
                    
                    # Trigger completion callbacks
                    for callback in self._task_completed_callbacks:
                        try:
                            callback(task_id, result)
                        except Exception as e:
                            logger.warning(f"Task completion callback failed: {e}")
                            
            except Exception as e:
                logger.error(f"Error handling task completion: {e}")
                
            finally:
                self._update_status()
                
    def _update_status(self):
        """Update worker status based on current load."""
        if self._shutdown_requested:
            self.status = WorkerStatus.DRAINING
        elif len(self.active_tasks) >= self.config.max_concurrent_tasks:
            self.status = WorkerStatus.BUSY
        elif len(self.active_tasks) > 0:
            self.status = WorkerStatus.ACTIVE
        else:
            self.status = WorkerStatus.IDLE
            
    def _start_server(self):
        """Start the worker communication server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.host, self.config.port))
        
        # Get actual port if auto-assigned
        self.host, self.port = self._socket.getsockname()
        
        self._socket.listen(5)
        
        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True
        )
        self._server_thread.start()
        
    def _server_loop(self):
        """Main server loop for handling client connections."""
        while self._running and self._socket:
            try:
                client_socket, address = self._socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()
                
            except OSError:
                if self._running:
                    logger.error("Server socket error")
                break
                
    def _handle_client(self, client_socket: socket.socket):
        """Handle client connection."""
        try:
            # Receive message
            data = client_socket.recv(4096)
            if not data:
                return
                
            message = json.loads(data.decode('utf-8'))
            response = self._process_message(message)
            
            # Send response
            response_data = json.dumps(response).encode('utf-8')
            client_socket.send(response_data)
            
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
            
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response."""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'execute_task':
                task_id = message['task_id']
                function_name = message['function_name']
                args = message.get('args', ())
                kwargs = message.get('kwargs', {})
                timeout = message.get('timeout')
                
                success = self.execute_task(task_id, function_name, args, kwargs, timeout)
                return {'success': success}
                
            elif msg_type == 'cancel_task':
                task_id = message['task_id']
                success = self.cancel_task(task_id)
                return {'success': success}
                
            elif msg_type == 'get_status':
                return {'status': self.get_worker_stats()}
                
            elif msg_type == 'get_task_status':
                task_id = message['task_id']
                status = self.get_task_status(task_id)
                return {'status': status}
                
            elif msg_type == 'register_function':
                # Note: This is simplified - in production, you'd want secure function registration
                return {'success': False, 'error': 'Dynamic function registration not supported'}
                
            else:
                return {'success': False, 'error': f'Unknown message type: {msg_type}'}
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return {'success': False, 'error': str(e)}
            
    def _heartbeat_loop(self):
        """Send periodic heartbeats to cluster manager."""
        while self._running:
            try:
                if self.cluster_manager:
                    stats = self.get_worker_stats()
                    self.cluster_manager.update_node_heartbeat(self.worker_id, stats)
                    
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)
                
    def _resource_monitor_loop(self):
        """Monitor system resources."""
        while self._running:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self._resource_stats.update({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_mb': memory.used / (1024 * 1024),
                    'disk_usage_percent': disk.percent
                })
                
                # GPU stats (if available)
                try:
                    # Try to import GPUtil for GPU monitoring
                    import GPUtil  # type: ignore
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self._resource_stats['gpu_count'] = len(gpus)
                        self._resource_stats['gpu_memory_mb'] = sum(
                            gpu.memoryUsed for gpu in gpus
                        )
                except (ImportError, Exception):
                    # GPUtil not available or GPU monitoring failed
                    pass
                    
                # Check resource limits
                if memory.percent > self.config.max_memory_usage * 100:
                    logger.warning(f"High memory usage: {memory.percent:.1f}%")
                    
                if cpu_percent > self.config.max_cpu_usage * 100:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    
                time.sleep(self.config.resource_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10.0)
                
    def _cleanup_loop(self):
        """Periodic cleanup of completed tasks and resources."""
        while self._running:
            try:
                current_time = time.time()
                
                with self._lock:
                    # Check for timed out tasks
                    timed_out_tasks = []
                    
                    for task_id, execution in self.active_tasks.items():
                        if (execution.timeout and 
                            current_time - execution.started_at > execution.timeout):
                            timed_out_tasks.append(task_id)
                            
                    # Cancel timed out tasks
                    for task_id in timed_out_tasks:
                        if self.cancel_task(task_id):
                            logger.warning(f"Task timed out and cancelled: {task_id}")
                            
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(60.0)
                
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        
    def add_task_started_callback(self, callback: Callable[[str], None]):
        """Add callback for task start events."""
        self._task_started_callbacks.append(callback)
        
    def add_task_completed_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for task completion events."""
        self._task_completed_callbacks.append(callback)
        
    def add_task_failed_callback(self, callback: Callable[[str, str], None]):
        """Add callback for task failure events."""
        self._task_failed_callbacks.append(callback)