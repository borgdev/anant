"""
GPU Memory Manager - Efficient GPU Memory Handling

Provides intelligent GPU memory management with automatic cleanup,
memory monitoring, and efficient batch processing strategies.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import time
import numpy as np

from .gpu_manager import get_gpu_manager, ComputeBackend

# Optional imports with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total: int
    allocated: int
    cached: int
    free: int
    utilization: float


class GPUMemoryManager:
    """
    Intelligent GPU memory management with automatic cleanup and monitoring.
    
    Features:
    - Automatic memory cleanup and garbage collection
    - Memory usage monitoring and alerts
    - Batch size optimization based on available memory
    - Memory-efficient operation scheduling
    - Graceful handling of out-of-memory situations
    """
    
    def __init__(self, gpu_manager=None, memory_fraction: float = 0.9):
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.memory_fraction = memory_fraction  # Max fraction of GPU memory to use
        self._device_info = self.gpu_manager.get_device_info()
        self._memory_history: List[MemoryStats] = []
        self._cleanup_threshold = 0.85  # Trigger cleanup at 85% memory usage
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        backend = self._device_info.backend
        
        if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
            return self._get_cuda_memory_stats()
        else:
            return self._get_cpu_memory_stats()
            
    def _get_cuda_memory_stats(self) -> MemoryStats:
        """Get CUDA memory statistics."""
        try:
            if hasattr(torch.cuda, 'mem_get_info'):
                free, total = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
            else:
                # Fallback for older PyTorch versions
                total = self._device_info.memory_total
                allocated = torch.cuda.memory_allocated() if hasattr(torch.cuda, 'memory_allocated') else 0
                cached = torch.cuda.memory_reserved() if hasattr(torch.cuda, 'memory_reserved') else 0
                free = total - allocated
                
            utilization = allocated / total if total > 0 else 0.0
            
            stats = MemoryStats(
                total=total,
                allocated=allocated,
                cached=cached,
                free=free,
                utilization=utilization
            )
            
            self._memory_history.append(stats)
            
            # Trigger cleanup if memory usage is high
            if utilization > self._cleanup_threshold:
                self._trigger_cleanup()
                
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory stats: {e}")
            return self._get_fallback_memory_stats()
            
    def _get_cpu_memory_stats(self) -> MemoryStats:
        """Get CPU memory statistics."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return MemoryStats(
                total=memory.total,
                allocated=memory.used,
                cached=memory.cached if hasattr(memory, 'cached') else 0,
                free=memory.available,
                utilization=memory.percent / 100.0
            )
        except ImportError:
            return self._get_fallback_memory_stats()
            
    def _get_fallback_memory_stats(self) -> MemoryStats:
        """Fallback memory statistics."""
        return MemoryStats(
            total=self._device_info.memory_total,
            allocated=0,
            cached=0,
            free=self._device_info.memory_available,
            utilization=0.0
        )
        
    def _trigger_cleanup(self):
        """Trigger memory cleanup when usage is high."""
        logger.info("Memory usage high, triggering cleanup...")
        
        backend = self._device_info.backend
        
        if backend == ComputeBackend.CUDA:
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory cleanup completed")
        
    def estimate_batch_size(self, tensor_shape: Tuple[int, ...], 
                           dtype_size: int = 4, safety_factor: float = 0.8) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            tensor_shape: Shape of a single tensor (excluding batch dimension)
            dtype_size: Size of data type in bytes (4 for float32)
            safety_factor: Safety factor to prevent OOM
            
        Returns:
            Recommended batch size
        """
        stats = self.get_memory_stats()
        available_memory = stats.free * safety_factor
        
        # Calculate memory per sample
        elements_per_sample = np.prod(tensor_shape)
        memory_per_sample = elements_per_sample * dtype_size
        
        # Account for intermediate computations (estimate 3x memory usage)
        memory_per_sample *= 3
        
        # Calculate batch size
        max_batch_size = int(available_memory / memory_per_sample)
        
        # Ensure minimum batch size of 1
        batch_size = max(1, max_batch_size)
        
        logger.debug(f"Estimated batch size: {batch_size} (available memory: {available_memory / 1024**2:.1f} MB)")
        
        return batch_size
        
    def can_fit_in_memory(self, required_memory: int) -> bool:
        """
        Check if required memory can fit in available memory.
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            True if memory is available
        """
        stats = self.get_memory_stats()
        available = stats.free * self.memory_fraction
        
        return required_memory <= available
        
    def allocate_with_retry(self, allocation_func, *args, max_retries: int = 3, **kwargs):
        """
        Allocate memory with automatic retry and cleanup on failure.
        
        Args:
            allocation_func: Function to allocate memory
            max_retries: Maximum number of retries
            *args, **kwargs: Arguments for allocation function
            
        Returns:
            Result of allocation function
        """
        for attempt in range(max_retries + 1):
            try:
                return allocation_func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries:
                    logger.warning(f"Out of memory on attempt {attempt + 1}, cleaning up...")
                    self._trigger_cleanup()
                    time.sleep(0.1)  # Brief pause after cleanup
                else:
                    raise e
                    
    def batch_process_with_memory_management(self, data: Union[np.ndarray, List], 
                                           process_func, batch_size: Optional[int] = None,
                                           **kwargs):
        """
        Process data in batches with automatic memory management.
        
        Args:
            data: Input data to process
            process_func: Function to process each batch
            batch_size: Batch size (auto-estimated if None)
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results from each batch
        """
        if isinstance(data, list):
            data_array = np.array(data)
        else:
            data_array = data
            
        n_samples = len(data_array)
        
        if batch_size is None:
            # Estimate optimal batch size
            sample_shape = data_array[0].shape if n_samples > 0 else (1,)
            batch_size = self.estimate_batch_size(sample_shape)
            
        results = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data_array[i:end_idx]
            
            # Process batch with memory retry
            try:
                result = self.allocate_with_retry(process_func, batch, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch {i}-{end_idx}: {e}")
                # Try with smaller batch size
                smaller_batch_size = max(1, batch_size // 2)
                logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                
                for j in range(i, end_idx, smaller_batch_size):
                    small_end = min(j + smaller_batch_size, end_idx)
                    small_batch = data_array[j:small_end]
                    
                    try:
                        result = self.allocate_with_retry(process_func, small_batch, **kwargs)
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"Failed to process small batch {j}-{small_end}: {e2}")
                        # Skip this batch or handle as needed
                        
        return results
        
    def monitor_memory_usage(self, operation_name: str = "operation"):
        """
        Context manager for monitoring memory usage during operations.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        return MemoryMonitor(self, operation_name)
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a comprehensive memory usage summary."""
        stats = self.get_memory_stats()
        
        summary = {
            "current_stats": {
                "total_gb": stats.total / 1024**3,
                "allocated_gb": stats.allocated / 1024**3,
                "free_gb": stats.free / 1024**3,
                "utilization_percent": stats.utilization * 100
            },
            "device_info": {
                "name": self._device_info.name,
                "backend": self._device_info.backend.value
            },
            "history_length": len(self._memory_history)
        }
        
        if self._memory_history:
            recent_stats = self._memory_history[-10:]  # Last 10 measurements
            summary["recent_peak_utilization"] = max(s.utilization for s in recent_stats) * 100
            summary["recent_avg_utilization"] = np.mean([s.utilization for s in recent_stats]) * 100
            
        return summary
        
    def clear_memory_history(self):
        """Clear memory usage history."""
        self._memory_history.clear()


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, memory_manager: GPUMemoryManager, operation_name: str):
        self.memory_manager = memory_manager
        self.operation_name = operation_name
        self.start_stats = None
        
    def __enter__(self):
        self.start_stats = self.memory_manager.get_memory_stats()
        logger.info(f"Starting {self.operation_name} - Memory usage: {self.start_stats.utilization:.1%}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_stats = self.memory_manager.get_memory_stats()
        
        memory_change = end_stats.allocated - self.start_stats.allocated
        utilization_change = end_stats.utilization - self.start_stats.utilization
        
        logger.info(f"Completed {self.operation_name}")
        logger.info(f"  Memory change: {memory_change / 1024**2:+.1f} MB")
        logger.info(f"  Utilization change: {utilization_change:+.1%}")
        logger.info(f"  Final utilization: {end_stats.utilization:.1%}")
        
        if exc_type is not None:
            logger.error(f"Operation {self.operation_name} failed with {exc_type.__name__}: {exc_val}")
            
        return False  # Don't suppress exceptions