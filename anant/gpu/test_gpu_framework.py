"""
Test GPU Acceleration Framework

Simple test to verify the GPU acceleration framework works correctly
with graceful fallbacks to CPU when GPU is unavailable.
"""

import numpy as np
import time
import sys
import os

# Add the anant directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_gpu_framework():
    """Test the GPU acceleration framework."""
    print("🚀 Testing Anant GPU Acceleration Framework")
    print("=" * 60)
    
    try:
        # Import our GPU modules
        from anant.gpu import (
            get_gpu_context, 
            is_gpu_available, 
            get_device_info,
            AcceleratedOperations,
            GPUBenchmark
        )
        
        print("✅ Successfully imported GPU modules")
        
        # Get GPU context
        gpu_context = get_gpu_context()
        device_info = get_device_info()
        
        print(f"\n📱 Device Information:")
        print(f"   Name: {device_info.name}")
        print(f"   Backend: {device_info.backend.value}")
        print(f"   Memory Total: {device_info.memory_total / 1024**3:.1f} GB")
        print(f"   Performance Score: {device_info.performance_score:.2f}")
        print(f"   GPU Available: {'YES' if is_gpu_available() else 'NO'}")
        
        # Test accelerated operations
        print(f"\n🔧 Testing Accelerated Operations:")
        
        ops = AcceleratedOperations()
        
        # Test matrix multiplication
        print("   Matrix Multiplication...")
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        
        start_time = time.time()
        result = ops.matrix_multiply(a, b)
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Result shape: {result.shape}")
        
        # Test vector similarity
        print("   Vector Similarity...")
        vectors1 = np.random.randn(50, 128).astype(np.float32)
        vectors2 = np.random.randn(50, 128).astype(np.float32)
        
        start_time = time.time()
        similarity = ops.vector_similarity(vectors1, vectors2, metric='cosine')
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Similarity matrix shape: {similarity.shape}")
        
        # Test embedding lookup
        print("   Embedding Lookup...")
        embeddings = np.random.randn(1000, 64).astype(np.float32)
        indices = np.random.randint(0, 1000, 20)
        
        start_time = time.time()
        looked_up = ops.batch_embedding_lookup(embeddings, indices)
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Looked up embeddings shape: {looked_up.shape}")
        
        # Run quick benchmark
        print(f"\n⚡ Running Quick Benchmark:")
        
        benchmark = GPUBenchmark()
        speedups = benchmark.quick_benchmark()
        
        for operation, speedup in speedups.items():
            if speedup > 1.0:
                print(f"   {operation}: {speedup:.2f}x speedup 🚀")
            else:
                print(f"   {operation}: Using CPU fallback ⚠️")
        
        # Memory information
        print(f"\n💾 Memory Information:")
        memory_usage = ops.get_memory_usage()
        
        if 'error' not in memory_usage:
            if 'allocated' in memory_usage:
                print(f"   Allocated: {memory_usage['allocated'] / 1024**2:.1f} MB")
                print(f"   Cached: {memory_usage['cached'] / 1024**2:.1f} MB")
            else:
                print(f"   Backend: {memory_usage.get('backend', 'Unknown')}")
        
        print(f"\n🎉 GPU Framework Test Completed Successfully!")
        
        # Test compatibility information
        print(f"\n🔧 Compatibility Information:")
        
        try:
            import torch
            print(f"   PyTorch: {torch.__version__} ✅")
            if torch.cuda.is_available():
                print(f"   CUDA: Available ({torch.cuda.device_count()} device(s)) ✅")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"     Device {i}: {props.name}")
                    print(f"     Compute Capability: {props.major}.{props.minor}")
            else:
                print(f"   CUDA: Not available ⚠️")
        except ImportError:
            print(f"   PyTorch: Not installed ⚠️")
        
        try:
            import cupy
            print(f"   CuPy: {cupy.__version__} ✅")
        except ImportError:
            print(f"   CuPy: Not installed ⚠️")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpu_framework()
    sys.exit(0 if success else 1)