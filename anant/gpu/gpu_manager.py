"""
GPU Manager - Central GPU Resource Management

Handles GPU detection, capability assessment, and resource management with
graceful fallbacks to CPU operations when GPU is unavailable.
"""

import logging
import platform
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Optional GPU imports with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
    torch_version = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torch_version = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    cupy_version = cp.__version__
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_version = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    PYNVML_AVAILABLE = False
    pynvml = None

import numpy as np

logger = logging.getLogger(__name__)


class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    AUTO = "auto"


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    name: str
    backend: ComputeBackend
    memory_total: int  # in bytes
    memory_available: int  # in bytes
    compute_capability: Optional[Tuple[int, int]] = None
    cuda_cores: Optional[int] = None
    is_available: bool = True
    performance_score: float = 1.0  # Relative performance metric


class GPUManager:
    """
    Central GPU resource manager with graceful fallbacks.
    
    Automatically detects available GPU resources and provides a consistent
    interface for GPU operations, falling back to CPU when necessary.
    """
    
    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._current_device: Optional[DeviceInfo] = None
        self._backend_preferences = [ComputeBackend.CUDA, ComputeBackend.CPU]
        self._initialized = False
        
        # Initialize device discovery
        self._discover_devices()
        self._select_best_device()
        
    def _discover_devices(self):
        """Discover available compute devices."""
        self._devices = []
        
        # Always add CPU as fallback
        cpu_info = DeviceInfo(
            name=f"CPU ({platform.processor()})",
            backend=ComputeBackend.CPU,
            memory_total=self._get_system_memory(),
            memory_available=self._get_available_memory(),
            performance_score=1.0
        )
        self._devices.append(cpu_info)
        
        # Discover CUDA devices
        self._discover_cuda_devices()
        
        # Log discovered devices
        logger.info(f"Discovered {len(self._devices)} compute devices")
        for device in self._devices:
            logger.info(f"  - {device.name} ({device.backend.value})")
            
    def _discover_cuda_devices(self):
        """Discover CUDA-capable devices."""
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available - skipping CUDA device discovery")
            return
            
        if not torch.cuda.is_available():
            logger.info("CUDA not available on this system")
            return
            
        try:
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA device(s)")
            
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                if hasattr(torch.cuda, 'mem_get_info'):
                    try:
                        with torch.cuda.device(i):
                            memory_available, memory_total = torch.cuda.mem_get_info()
                    except Exception:
                        memory_total = device_props.total_memory
                        memory_available = memory_total
                else:
                    memory_total = device_props.total_memory
                    memory_available = memory_total
                
                # Calculate performance score based on GPU specs
                performance_score = self._calculate_gpu_performance_score(device_props)
                
                cuda_device = DeviceInfo(
                    name=f"{device_props.name} (CUDA {i})",
                    backend=ComputeBackend.CUDA,
                    memory_total=memory_total,
                    memory_available=memory_available,
                    compute_capability=(device_props.major, device_props.minor),
                    cuda_cores=device_props.multi_processor_count * 64,  # Approximate
                    performance_score=performance_score
                )
                
                self._devices.append(cuda_device)
                logger.info(f"  CUDA Device {i}: {device_props.name}")
                logger.info(f"    Compute Capability: {device_props.major}.{device_props.minor}")
                logger.info(f"    Total Memory: {memory_total / 1024**3:.1f} GB")
                logger.info(f"    Performance Score: {performance_score:.2f}")
                
        except Exception as e:
            logger.warning(f"Error discovering CUDA devices: {e}")
            
    def _calculate_gpu_performance_score(self, device_props) -> float:
        """Calculate a performance score for GPU device."""
        base_score = 2.0  # Base score for any GPU vs CPU
        
        # Boost based on compute capability
        if hasattr(device_props, 'major') and hasattr(device_props, 'minor'):
            cc_score = device_props.major + device_props.minor * 0.1
            base_score *= (1 + cc_score * 0.5)
            
        # Boost based on memory (more memory = better for large operations)
        if hasattr(device_props, 'total_memory'):
            memory_gb = device_props.total_memory / (1024**3)
            memory_score = min(memory_gb / 8.0, 2.0)  # Cap at 2x boost for 8GB+
            base_score *= (1 + memory_score * 0.3)
            
        # Boost based on multiprocessor count
        if hasattr(device_props, 'multi_processor_count'):
            mp_score = min(device_props.multi_processor_count / 20.0, 3.0)
            base_score *= (1 + mp_score * 0.2)
            
        return base_score
        
    def _select_best_device(self):
        """Select the best available device based on performance."""
        if not self._devices:
            raise RuntimeError("No compute devices available")
            
        # Sort devices by performance score (descending)
        gpu_devices = [d for d in self._devices if d.backend != ComputeBackend.CPU]
        
        if gpu_devices:
            # Select best GPU device
            best_gpu = max(gpu_devices, key=lambda d: d.performance_score)
            self._current_device = best_gpu
            logger.info(f"Selected GPU device: {best_gpu.name}")
        else:
            # Fall back to CPU
            cpu_device = next(d for d in self._devices if d.backend == ComputeBackend.CPU)
            self._current_device = cpu_device
            logger.info("No GPU available - using CPU")
            
    def _get_system_memory(self) -> int:
        """Get total system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().total
        except ImportError:
            # Fallback estimate
            return 8 * 1024**3  # 8GB estimate
            
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # Fallback estimate
            return 4 * 1024**3  # 4GB estimate
            
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._current_device and self._current_device.backend != ComputeBackend.CPU
        
    def get_device_info(self) -> DeviceInfo:
        """Get information about the current device."""
        return self._current_device
        
    def get_all_devices(self) -> List[DeviceInfo]:
        """Get information about all available devices."""
        return self._devices.copy()
        
    def set_device(self, device_id: Optional[int] = None, backend: Optional[ComputeBackend] = None) -> bool:
        """
        Set the active compute device.
        
        Args:
            device_id: Device ID to use (None for auto-select)
            backend: Preferred backend (None for auto-select)
            
        Returns:
            True if device was successfully set
        """
        if device_id is not None:
            if 0 <= device_id < len(self._devices):
                self._current_device = self._devices[device_id]
                logger.info(f"Switched to device {device_id}: {self._current_device.name}")
                return True
            else:
                logger.warning(f"Invalid device ID: {device_id}")
                return False
                
        if backend is not None:
            # Find best device with specified backend
            compatible_devices = [d for d in self._devices if d.backend == backend]
            if compatible_devices:
                best_device = max(compatible_devices, key=lambda d: d.performance_score)
                self._current_device = best_device
                logger.info(f"Switched to {backend.value} device: {best_device.name}")
                return True
            else:
                logger.warning(f"No devices available for backend: {backend.value}")
                return False
                
        return False
        
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory information for current device."""
        if not self._current_device:
            return {"total": 0, "available": 0, "used": 0}
            
        if self._current_device.backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
            try:
                if hasattr(torch.cuda, 'mem_get_info'):
                    available, total = torch.cuda.mem_get_info()
                    used = total - available
                    return {"total": total, "available": available, "used": used}
            except Exception:
                pass
                
        # Fallback to device info
        return {
            "total": self._current_device.memory_total,
            "available": self._current_device.memory_available,
            "used": self._current_device.memory_total - self._current_device.memory_available
        }
        
    def clear_cache(self):
        """Clear GPU memory cache if applicable."""
        if self._current_device and self._current_device.backend == ComputeBackend.CUDA:
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
    def benchmark_device(self) -> Dict[str, float]:
        """Run basic benchmark on current device."""
        device_info = self.get_device_info()
        
        if device_info.backend == ComputeBackend.CPU:
            return self._benchmark_cpu()
        elif device_info.backend == ComputeBackend.CUDA:
            return self._benchmark_cuda()
        else:
            return {"error": "Unsupported device for benchmarking"}
            
    def _benchmark_cpu(self) -> Dict[str, float]:
        """Benchmark CPU operations."""
        import time
        
        # Matrix multiplication benchmark
        size = 1000
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        start = time.time()
        _ = np.dot(a, b)
        cpu_time = time.time() - start
        
        return {
            "matrix_mult_time": cpu_time,
            "operations_per_second": (size ** 3) / cpu_time,
            "device_type": "cpu"
        }
        
    def _benchmark_cuda(self) -> Dict[str, float]:
        """Benchmark CUDA operations."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available for CUDA benchmarking"}
            
        import time
        
        try:
            device = torch.cuda.current_device()
            size = 1000
            
            # Matrix multiplication benchmark
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # Warm up
            _ = torch.mm(a, b)
            torch.cuda.synchronize()
            
            start = time.time()
            _ = torch.mm(a, b)
            torch.cuda.synchronize()
            cuda_time = time.time() - start
            
            return {
                "matrix_mult_time": cuda_time,
                "operations_per_second": (size ** 3) / cuda_time,
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name()
            }
            
        except Exception as e:
            return {"error": f"CUDA benchmark failed: {e}"}


# Global GPU manager instance
_global_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get or create the global GPU manager instance."""
    global _global_gpu_manager
    
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager()
        
    return _global_gpu_manager


def reset_gpu_manager():
    """Reset the global GPU manager (useful for testing)."""
    global _global_gpu_manager
    _global_gpu_manager = None