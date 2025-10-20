"""
Distributed Backend Implementations

This module provides concrete implementations for distributed computing backends:
- Dask Backend: DataFrame and array operations
- Ray Backend: ML workloads and high-performance computing  
- Celery Backend: Task queues and background jobs
- Native Backend: Custom message-passing implementation
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# Optional imports for different backends
try:
    import dask
    import dask.distributed
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dask = None

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery = None

try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

try:
    import grpc
    import grpc.aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None


@dataclass
class BackendConfig:
    """Configuration for distributed backends."""
    backend_type: str
    connection_params: Dict[str, Any]
    worker_config: Dict[str, Any]
    performance_config: Dict[str, Any]


@dataclass
class TaskResult:
    """Result from distributed task execution."""
    task_id: str
    result: Any
    execution_time: float
    worker_id: str
    status: str = "completed"
    error: Optional[str] = None


class DistributedBackend(ABC):
    """Abstract base class for distributed computing backends."""
    
    @abstractmethod
    async def initialize(self, config: BackendConfig) -> bool:
        """Initialize the backend with configuration."""
        pass
    
    @abstractmethod
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for distributed execution."""
        pass
    
    @abstractmethod
    async def get_result(self, task_id: str) -> TaskResult:
        """Get result of a submitted task."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Shutdown the backend and cleanup resources."""
        pass
    
    @abstractmethod
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the cluster."""
        pass


class DaskBackend(DistributedBackend):
    """Dask-based distributed computing backend."""
    
    def __init__(self):
        self.client: Optional[dask.distributed.Client] = None
        self.cluster = None
        self.futures = {}
    
    async def initialize(self, config: BackendConfig) -> bool:
        """Initialize Dask cluster."""
        if not DASK_AVAILABLE:
            logger.error("Dask not available - install with: pip install 'dask[complete]'")
            return False
        
        try:
            connection_params = config.connection_params
            
            if connection_params.get('cluster_type') == 'local':
                # Local cluster
                from dask.distributed import LocalCluster
                self.cluster = LocalCluster(
                    n_workers=connection_params.get('n_workers', 4),
                    threads_per_worker=connection_params.get('threads_per_worker', 2),
                    memory_limit=connection_params.get('memory_limit', '2GB')
                )
                self.client = Client(self.cluster)
            
            elif connection_params.get('scheduler_address'):
                # Connect to existing cluster
                self.client = Client(connection_params['scheduler_address'])
            
            else:
                # Default local client
                self.client = Client()
            
            # Wait for workers to be ready
            await asyncio.sleep(2)
            
            logger.info(f"Dask backend initialized with {len(self.client.scheduler_info()['workers'])} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dask backend: {e}")
            return False
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to Dask cluster."""
        if not self.client:
            raise RuntimeError("Dask backend not initialized")
        
        # Submit task asynchronously
        future = self.client.submit(func, *args, **kwargs)
        task_id = f"dask_{future.key}"
        self.futures[task_id] = future
        
        logger.debug(f"Submitted Dask task: {task_id}")
        return task_id
    
    async def get_result(self, task_id: str) -> TaskResult:
        """Get result from Dask future."""
        if task_id not in self.futures:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.futures[task_id]
        start_time = time.time()
        
        try:
            # Get result (this will block until complete)
            result = await asyncio.get_event_loop().run_in_executor(
                None, future.result
            )
            
            execution_time = time.time() - start_time
            worker_id = getattr(future, 'worker', 'unknown')
            
            # Cleanup
            del self.futures[task_id]
            
            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id,
                status="completed"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task_id,
                result=None,
                execution_time=execution_time,
                worker_id="unknown",
                status="failed",
                error=str(e)
            )
    
    async def shutdown(self):
        """Shutdown Dask cluster."""
        if self.client:
            await asyncio.get_event_loop().run_in_executor(None, self.client.close)
        if self.cluster:
            await asyncio.get_event_loop().run_in_executor(None, self.cluster.close)
        logger.info("Dask backend shutdown")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Dask cluster information."""
        if not self.client:
            return {"status": "not_initialized"}
        
        try:
            scheduler_info = self.client.scheduler_info()
            return {
                "backend": "dask",
                "scheduler_address": scheduler_info.get('address'),
                "workers": len(scheduler_info.get('workers', {})),
                "total_cores": sum(w.get('nthreads', 0) for w in scheduler_info.get('workers', {}).values()),
                "total_memory": sum(w.get('memory_limit', 0) for w in scheduler_info.get('workers', {}).values()),
                "status": "running"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class RayBackend(DistributedBackend):
    """Ray-based distributed computing backend."""
    
    def __init__(self):
        self.initialized = False
        self.futures = {}
    
    async def initialize(self, config: BackendConfig) -> bool:
        """Initialize Ray cluster."""
        if not RAY_AVAILABLE:
            logger.error("Ray not available - install with: pip install 'ray[default]'")
            return False
        
        try:
            connection_params = config.connection_params
            
            if not ray.is_initialized():
                if connection_params.get('cluster_address'):
                    # Connect to existing cluster
                    ray.init(address=connection_params['cluster_address'])
                else:
                    # Start local cluster
                    ray.init(
                        num_cpus=connection_params.get('num_cpus'),
                        num_gpus=connection_params.get('num_gpus', 0),
                        object_store_memory=connection_params.get('object_store_memory')
                    )
            
            self.initialized = True
            logger.info(f"Ray backend initialized with {ray.cluster_resources().get('CPU', 0)} CPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray backend: {e}")
            return False
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to Ray cluster."""
        if not self.initialized:
            raise RuntimeError("Ray backend not initialized")
        
        # Convert function to Ray remote function
        remote_func = ray.remote(func)
        
        # Submit task
        future = remote_func.remote(*args, **kwargs)
        task_id = f"ray_{id(future)}"
        self.futures[task_id] = future
        
        logger.debug(f"Submitted Ray task: {task_id}")
        return task_id
    
    async def get_result(self, task_id: str) -> TaskResult:
        """Get result from Ray future."""
        if task_id not in self.futures:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.futures[task_id]
        start_time = time.time()
        
        try:
            # Get result asynchronously
            result = await asyncio.get_event_loop().run_in_executor(
                None, ray.get, future
            )
            
            execution_time = time.time() - start_time
            
            # Cleanup
            del self.futures[task_id]
            
            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                worker_id="ray_worker",
                status="completed"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task_id,
                result=None,
                execution_time=execution_time,
                worker_id="ray_worker",
                status="failed",
                error=str(e)
            )
    
    async def shutdown(self):
        """Shutdown Ray cluster."""
        if self.initialized and ray.is_initialized():
            ray.shutdown()
            self.initialized = False
        logger.info("Ray backend shutdown")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Ray cluster information."""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            resources = ray.cluster_resources()
            return {
                "backend": "ray",
                "total_cpus": resources.get('CPU', 0),
                "total_gpus": resources.get('GPU', 0),
                "total_memory": resources.get('memory', 0),
                "nodes": len(ray.nodes()),
                "status": "running"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class CeleryBackend(DistributedBackend):
    """Celery-based distributed task queue backend."""
    
    def __init__(self):
        self.app: Optional[Celery] = None
        self.results = {}
    
    async def initialize(self, config: BackendConfig) -> bool:
        """Initialize Celery app."""
        if not CELERY_AVAILABLE:
            logger.error("Celery not available - install with: pip install celery")
            return False
        
        try:
            connection_params = config.connection_params
            
            self.app = Celery(
                'anant_distributed',
                broker=connection_params.get('broker', 'redis://localhost:6379'),
                backend=connection_params.get('backend', 'redis://localhost:6379')
            )
            
            # Configure Celery
            self.app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                **config.performance_config
            )
            
            logger.info("Celery backend initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Celery backend: {e}")
            return False
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to Celery queue."""
        if not self.app:
            raise RuntimeError("Celery backend not initialized")
        
        # Create Celery task
        @self.app.task
        def celery_task():
            return func(*args, **kwargs)
        
        # Submit task
        result = celery_task.delay()
        task_id = f"celery_{result.id}"
        self.results[task_id] = result
        
        logger.debug(f"Submitted Celery task: {task_id}")
        return task_id
    
    async def get_result(self, task_id: str) -> TaskResult:
        """Get result from Celery task."""
        if task_id not in self.results:
            raise ValueError(f"Task {task_id} not found")
        
        celery_result = self.results[task_id]
        start_time = time.time()
        
        try:
            # Get result (blocks until complete)
            result = await asyncio.get_event_loop().run_in_executor(
                None, celery_result.get
            )
            
            execution_time = time.time() - start_time
            
            # Cleanup
            del self.results[task_id]
            
            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                worker_id=getattr(celery_result, 'worker', 'unknown'),
                status="completed"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task_id,
                result=None,
                execution_time=execution_time,
                worker_id="unknown",
                status="failed",
                error=str(e)
            )
    
    async def shutdown(self):
        """Shutdown Celery backend."""
        if self.app:
            # Cancel pending tasks
            for result in self.results.values():
                result.revoke()
        logger.info("Celery backend shutdown")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Celery cluster information."""
        if not self.app:
            return {"status": "not_initialized"}
        
        try:
            # Get worker stats
            stats = self.app.control.inspect().stats()
            
            return {
                "backend": "celery",
                "workers": len(stats) if stats else 0,
                "broker": str(self.app.conf.broker_url),
                "status": "running" if stats else "no_workers"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class NativeBackend(DistributedBackend):
    """Native message-passing based distributed backend."""
    
    def __init__(self):
        self.context = None
        self.socket = None
        self.workers = {}
        self.tasks = {}
    
    async def initialize(self, config: BackendConfig) -> bool:
        """Initialize native ZMQ-based backend."""
        if not ZMQ_AVAILABLE:
            logger.error("ZMQ not available - install with: pip install pyzmq")
            return False
        
        try:
            connection_params = config.connection_params
            
            # Initialize ZMQ context
            self.context = zmq.asyncio.Context()
            
            # Create coordinator socket
            self.socket = self.context.socket(zmq.ROUTER)
            bind_address = connection_params.get('bind_address', 'tcp://*:5555')
            self.socket.bind(bind_address)
            
            logger.info(f"Native backend initialized on {bind_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize native backend: {e}")
            return False
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to native distributed workers."""
        if not self.socket:
            raise RuntimeError("Native backend not initialized")
        
        task_id = f"native_{int(time.time() * 1000000)}"
        
        # Serialize task
        task_data = {
            'task_id': task_id,
            'function': func.__name__,
            'args': args,
            'kwargs': kwargs
        }
        
        # Store task
        self.tasks[task_id] = {
            'data': task_data,
            'status': 'pending',
            'submitted': time.time()
        }
        
        # Send to available worker (simplified)
        # In real implementation, would have worker discovery and load balancing
        
        logger.debug(f"Submitted native task: {task_id}")
        return task_id
    
    async def get_result(self, task_id: str) -> TaskResult:
        """Get result from native task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.tasks[task_id]
        start_time = task_info['submitted']
        
        # Simulate task execution for demo
        await asyncio.sleep(0.1)
        
        execution_time = time.time() - start_time
        
        # Cleanup
        del self.tasks[task_id]
        
        return TaskResult(
            task_id=task_id,
            result=f"Native result for {task_id}",
            execution_time=execution_time,
            worker_id="native_worker_1",
            status="completed"
        )
    
    async def shutdown(self):
        """Shutdown native backend."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logger.info("Native backend shutdown")
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get native cluster information."""
        return {
            "backend": "native",
            "workers": len(self.workers),
            "pending_tasks": len([t for t in self.tasks.values() if t['status'] == 'pending']),
            "status": "running" if self.socket else "not_initialized"
        }


class BackendFactory:
    """Factory for creating distributed backends."""
    
    @staticmethod
    def create_backend(backend_type: str) -> DistributedBackend:
        """Create backend instance based on type."""
        if backend_type == "dask":
            return DaskBackend()
        elif backend_type == "ray":
            return RayBackend()
        elif backend_type == "celery":
            return CeleryBackend()
        elif backend_type == "native":
            return NativeBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends."""
        available = []
        
        if DASK_AVAILABLE:
            available.append("dask")
        if RAY_AVAILABLE:
            available.append("ray")
        if CELERY_AVAILABLE:
            available.append("celery")
        if ZMQ_AVAILABLE:
            available.append("native")
        
        return available
    
    @staticmethod
    def get_recommended_backend(workload_type: str) -> str:
        """Get recommended backend for workload type."""
        recommendations = {
            "dataframe": "dask",
            "ml": "ray",
            "batch": "celery",
            "realtime": "native"
        }
        
        available = BackendFactory.get_available_backends()
        recommended = recommendations.get(workload_type, "dask")
        
        if recommended in available:
            return recommended
        elif available:
            return available[0]
        else:
            raise RuntimeError("No distributed backends available")


class UnifiedBackendManager:
    """Unified manager for multiple distributed backends."""
    
    def __init__(self):
        self.backends = {}
        self.active_backend = None
    
    async def initialize_backend(self, backend_type: str, config: BackendConfig) -> bool:
        """Initialize a specific backend."""
        backend = BackendFactory.create_backend(backend_type)
        success = await backend.initialize(config)
        
        if success:
            self.backends[backend_type] = backend
            if not self.active_backend:
                self.active_backend = backend_type
            logger.info(f"Backend {backend_type} initialized successfully")
        
        return success
    
    async def switch_backend(self, backend_type: str):
        """Switch to a different backend."""
        if backend_type not in self.backends:
            raise ValueError(f"Backend {backend_type} not initialized")
        
        self.active_backend = backend_type
        logger.info(f"Switched to backend: {backend_type}")
    
    async def submit_task(self, func: Callable, *args, backend: Optional[str] = None, **kwargs) -> str:
        """Submit task to specified or active backend."""
        backend_name = backend or self.active_backend
        
        if not backend_name or backend_name not in self.backends:
            raise RuntimeError(f"Backend {backend_name} not available")
        
        return await self.backends[backend_name].submit_task(func, *args, **kwargs)
    
    async def get_result(self, task_id: str, backend: Optional[str] = None) -> TaskResult:
        """Get result from specified or active backend."""
        backend_name = backend or self.active_backend
        
        if not backend_name or backend_name not in self.backends:
            raise RuntimeError(f"Backend {backend_name} not available")
        
        return await self.backends[backend_name].get_result(task_id)
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all initialized backends."""
        status = {}
        
        for backend_name, backend in self.backends.items():
            status[backend_name] = await backend.get_cluster_info()
        
        status['active_backend'] = self.active_backend
        status['available_backends'] = list(self.backends.keys())
        
        return status
    
    async def shutdown_all(self):
        """Shutdown all backends."""
        for backend_name, backend in self.backends.items():
            try:
                await backend.shutdown()
                logger.info(f"Backend {backend_name} shutdown successfully")
            except Exception as e:
                logger.error(f"Error shutting down {backend_name}: {e}")
        
        self.backends.clear()
        self.active_backend = None