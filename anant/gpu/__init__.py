"""
GPU Acceleration Framework for Anant

This module provides GPU acceleration capabilities with graceful fallbacks to CPU.
The framework automatically detects available GPU resources and optimizes operations
accordingly, ensuring the library works seamlessly with or without GPU hardware.

Key Features:
- Automatic GPU detection and capability assessment
- Graceful fallback to CPU operations
- Support for CUDA, OpenCL, and CPU backends
- Memory-efficient operations with automatic memory management
- Compatible with older GPUs (tested on RTX 2070+)
- Consistent API regardless of backend

Components:
- gpu_manager: Central GPU resource management
- accelerated_ops: GPU-accelerated operations
- memory_manager: Efficient GPU memory handling
- fallback_manager: CPU fallback implementations
- benchmarks: Performance comparison tools
"""

from .gpu_manager import GPUManager, get_gpu_manager
from .accelerated_ops import AcceleratedOperations
from .memory_manager import GPUMemoryManager
from .fallback_manager import FallbackManager
from .benchmarks import GPUBenchmark

# Global GPU manager instance
_gpu_manager = None

def get_gpu_context():
    """Get the global GPU context manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = get_gpu_manager()
    return _gpu_manager

def is_gpu_available():
    """Check if GPU acceleration is available."""
    return get_gpu_context().is_gpu_available()

def get_device_info():
    """Get information about available compute devices."""
    return get_gpu_context().get_device_info()

def set_device(device):
    """Set the preferred compute device."""
    return get_gpu_context().set_device(device)

__all__ = [
    'GPUManager',
    'AcceleratedOperations', 
    'GPUMemoryManager',
    'FallbackManager',
    'GPUBenchmark',
    'get_gpu_context',
    'get_gpu_manager',
    'is_gpu_available',
    'get_device_info',
    'set_device',
]