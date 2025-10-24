"""
Compute Workers
==============

General-purpose distributed computing workers for FastAPI + Ray integration.
"""

import ray
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@ray.remote
class ComputeWorker:
    """General-purpose compute worker for distributed tasks."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.created_at = datetime.now()
        self.task_count = 0
        
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        return {
            "worker_id": self.worker_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "created_at": self.created_at.isoformat(),
            "task_count": self.task_count,
            "status": "ready"
        }
    
    def compute_intensive_task(self, data: List[float], operation: str = "sum") -> Dict[str, Any]:
        """Perform compute-intensive operations."""
        self.task_count += 1
        start_time = time.time()
        
        # Convert to numpy for efficient computation
        np_data = np.array(data)
        
        # Perform different operations based on request
        if operation == "sum":
            result = float(np.sum(np_data))
        elif operation == "mean":
            result = float(np.mean(np_data))
        elif operation == "std":
            result = float(np.std(np_data))
        elif operation == "fft":
            fft_result = np.fft.fft(np_data)
            result = {
                "magnitude": np.abs(fft_result).tolist(),
                "phase": np.angle(fft_result).tolist()
            }
        elif operation == "matrix_multiply":
            # Reshape data for matrix multiplication (square matrix)
            size = int(np.sqrt(len(data)))
            if size * size == len(data):
                matrix = np_data.reshape(size, size)
                result = np.dot(matrix, matrix.T).tolist()
            else:
                result = {"error": "Data size not suitable for square matrix"}
        elif operation == "eigenvalues":
            # For eigenvalue computation, create correlation matrix
            if len(data) > 1:
                matrix = np.corrcoef(np_data.reshape(-1, 1).T)
                eigenvals = np.linalg.eigvals(matrix)
                result = eigenvals.tolist()
            else:
                result = {"error": "Need more data for eigenvalue computation"}
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        processing_time = time.time() - start_time
        
        return {
            "worker_id": self.worker_id,
            "operation": operation,
            "result": result,
            "processing_time_seconds": round(processing_time, 4),
            "data_size": len(data),
            "timestamp": datetime.now().isoformat()
        }
    
    def batch_process(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of tasks."""
        self.task_count += len(batch_data)
        results = []
        
        for item in batch_data:
            data = item.get("data", [])
            operation = item.get("operation", "sum")
            task_id = item.get("task_id", f"task_{len(results)}")
            
            result = self.compute_intensive_task(data, operation)
            result["task_id"] = task_id
            results.append(result)
        
        return results
    
    def simulate_work(self, duration_seconds: float, complexity: int = 1) -> Dict[str, Any]:
        """Simulate computational work with specified duration and complexity."""
        self.task_count += 1
        start_time = time.time()
        
        # Simulate varying levels of computational complexity
        for _ in range(complexity):
            # Matrix operations to consume CPU
            size = min(100 + complexity * 10, 500)  # Limit size to prevent excessive computation
            matrix_a = np.random.random((size, size))
            matrix_b = np.random.random((size, size))
            
            # Perform matrix multiplication and other operations
            result_matrix = np.dot(matrix_a, matrix_b)
            eigenvals = np.linalg.eigvals(result_matrix[:min(50, size), :min(50, size)])  # Limit for performance
            
            # Check if we've reached the target duration
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break
            
            # Sleep briefly to allow for precise timing
            if elapsed < duration_seconds - 0.1:
                time.sleep(min(0.1, duration_seconds - elapsed))
        
        actual_duration = time.time() - start_time
        
        return {
            "worker_id": self.worker_id,
            "requested_duration": duration_seconds,
            "actual_duration": round(actual_duration, 4),
            "complexity_level": complexity,
            "matrix_size": size,
            "eigenvalue_count": len(eigenvals),
            "timestamp": datetime.now().isoformat()
        }


@ray.remote
class TaskCoordinator:
    """Coordinates tasks across multiple compute workers."""
    
    def __init__(self, num_workers: int = 3):
        self.workers = []
        self.num_workers = num_workers
        self.total_tasks_processed = 0
        
    def initialize_workers(self) -> List[str]:
        """Initialize compute workers."""
        worker_ids = []
        for i in range(self.num_workers):
            worker_id = f"compute_worker_{i}"
            worker = ComputeWorker.remote(worker_id)
            self.workers.append(worker)
            worker_ids.append(worker_id)
        
        return worker_ids
    
    def get_worker_status(self) -> List[Dict[str, Any]]:
        """Get status of all workers."""
        if not self.workers:
            return []
        
        status_refs = [worker.get_worker_info.remote() for worker in self.workers]
        worker_statuses = ray.get(status_refs)
        
        return worker_statuses
    
    def distribute_work(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute work across available workers."""
        if not self.workers:
            self.initialize_workers()
        
        if not tasks:
            return []
        
        # Distribute tasks across workers
        worker_count = len(self.workers)
        task_groups = [[] for _ in range(worker_count)]
        
        for i, task in enumerate(tasks):
            worker_index = i % worker_count
            task_groups[worker_index].append(task)
        
        # Submit batch processing tasks to workers
        batch_refs = []
        for i, task_group in enumerate(task_groups):
            if task_group:  # Only submit if there are tasks
                batch_ref = self.workers[i].batch_process.remote(task_group)
                batch_refs.append(batch_ref)
        
        # Collect results
        all_results = []
        if batch_refs:
            batch_results = ray.get(batch_refs)
            for batch_result in batch_results:
                all_results.extend(batch_result)
        
        self.total_tasks_processed += len(tasks)
        
        return all_results
    
    def parallel_compute(self, data_chunks: List[List[float]], operation: str = "sum") -> Dict[str, Any]:
        """Perform parallel computation on data chunks."""
        if not self.workers:
            self.initialize_workers()
        
        start_time = time.time()
        
        # Distribute chunks across workers
        worker_count = len(self.workers)
        chunk_refs = []
        
        for i, chunk in enumerate(data_chunks):
            worker_index = i % worker_count
            chunk_ref = self.workers[worker_index].compute_intensive_task.remote(chunk, operation)
            chunk_refs.append(chunk_ref)
        
        # Collect results
        chunk_results = ray.get(chunk_refs)
        
        # Aggregate results based on operation
        if operation in ["sum", "mean"]:
            if operation == "sum":
                total_result = sum(result["result"] for result in chunk_results if isinstance(result["result"], (int, float)))
            else:  # mean
                values = [result["result"] for result in chunk_results if isinstance(result["result"], (int, float))]
                total_result = sum(values) / len(values) if values else 0
        else:
            total_result = chunk_results  # Return all results for complex operations
        
        processing_time = time.time() - start_time
        self.total_tasks_processed += len(data_chunks)
        
        return {
            "operation": operation,
            "chunks_processed": len(data_chunks),
            "total_result": total_result,
            "chunk_results": chunk_results,
            "processing_time_seconds": round(processing_time, 4),
            "total_data_points": sum(len(chunk) for chunk in data_chunks),
            "timestamp": datetime.now().isoformat()
        }
    
    def stress_test(self, duration_seconds: float, complexity: int = 1) -> Dict[str, Any]:
        """Run stress test across all workers."""
        if not self.workers:
            self.initialize_workers()
        
        # Start stress test on all workers
        stress_refs = [
            worker.simulate_work.remote(duration_seconds, complexity)
            for worker in self.workers
        ]
        
        # Wait for all workers to complete
        stress_results = ray.get(stress_refs)
        
        # Calculate aggregate metrics
        total_duration = max(result["actual_duration"] for result in stress_results)
        avg_duration = sum(result["actual_duration"] for result in stress_results) / len(stress_results)
        
        return {
            "stress_test_summary": {
                "workers_tested": len(self.workers),
                "requested_duration": duration_seconds,
                "max_actual_duration": round(total_duration, 4),
                "avg_actual_duration": round(avg_duration, 4),
                "complexity_level": complexity
            },
            "worker_results": stress_results,
            "timestamp": datetime.now().isoformat()
        }


# Utility functions for creating and managing workers
def create_worker_pool(num_workers: int = 3) -> "TaskCoordinator":
    """Create a pool of compute workers."""
    coordinator = TaskCoordinator.remote(num_workers)
    ray.get(coordinator.initialize_workers.remote())
    return coordinator


def parallel_map_reduce(data: List[Any], map_func: str, reduce_func: str = "sum", chunk_size: int = 100) -> Dict[str, Any]:
    """Perform parallel map-reduce operations."""
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Create coordinator for this operation
    coordinator = TaskCoordinator.remote(min(len(chunks), 5))  # Limit workers
    ray.get(coordinator.initialize_workers.remote())
    
    # Map phase - apply operation to each chunk
    map_results = ray.get(coordinator.parallel_compute.remote(chunks, map_func))
    
    # Reduce phase - combine results
    if reduce_func == "sum":
        if isinstance(map_results["total_result"], (int, float)):
            final_result = map_results["total_result"]
        else:
            final_result = sum(
                result["result"] for result in map_results["chunk_results"]
                if isinstance(result["result"], (int, float))
            )
    elif reduce_func == "max":
        values = [
            result["result"] for result in map_results["chunk_results"]
            if isinstance(result["result"], (int, float))
        ]
        final_result = max(values) if values else 0
    elif reduce_func == "min":
        values = [
            result["result"] for result in map_results["chunk_results"]
            if isinstance(result["result"], (int, float))
        ]
        final_result = min(values) if values else 0
    else:
        final_result = map_results["total_result"]
    
    return {
        "map_reduce_result": final_result,
        "chunks_processed": len(chunks),
        "chunk_size": chunk_size,
        "map_function": map_func,
        "reduce_function": reduce_func,
        "detailed_results": map_results,
        "timestamp": datetime.now().isoformat()
    }


# Factory function for creating specialized workers
def create_specialized_worker(worker_type: str, config: Dict[str, Any] = None) -> Any:
    """Create specialized workers based on type."""
    config = config or {}
    
    if worker_type == "compute":
        worker_id = config.get("worker_id", f"compute_{int(time.time())}")
        return ComputeWorker.remote(worker_id)
    elif worker_type == "coordinator":
        num_workers = config.get("num_workers", 3)
        return TaskCoordinator.remote(num_workers)
    else:
        raise ValueError(f"Unknown worker type: {worker_type}")