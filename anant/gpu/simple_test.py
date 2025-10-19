"""
Simple GPU Framework Test

Test the GPU acceleration framework directly without package imports.
"""

import numpy as np
import time
import sys
import os

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def test_gpu_framework_simple():
    """Test the GPU framework with simple imports."""
    print("🚀 Testing Anant GPU Acceleration Framework")
    print("=" * 60)
    
    try:
        # Import GPU modules directly
        from gpu.gpu_manager import get_gpu_manager, ComputeBackend
        from gpu.accelerated_ops import AcceleratedOperations
        from gpu.memory_manager import GPUMemoryManager
        from gpu.benchmarks import GPUBenchmark
        
        print("✅ Successfully imported GPU modules")
        
        # Get GPU manager
        gpu_manager = get_gpu_manager()
        device_info = gpu_manager.get_device_info()
        
        print(f"\n📱 Device Information:")
        print(f"   Name: {device_info.name}")
        print(f"   Backend: {device_info.backend.value}")
        print(f"   Memory Total: {device_info.memory_total / 1024**3:.1f} GB")
        print(f"   Performance Score: {device_info.performance_score:.2f}")
        print(f"   GPU Available: {'YES' if gpu_manager.is_gpu_available() else 'NO'}")
        
        # Show all available devices
        all_devices = gpu_manager.get_all_devices()
        print(f"\n🖥️  Available Devices:")
        for i, device in enumerate(all_devices):
            print(f"   [{i}] {device.name} ({device.backend.value}) - Score: {device.performance_score:.2f}")
        
        # Test accelerated operations
        print(f"\n🔧 Testing Accelerated Operations:")
        
        ops = AcceleratedOperations(gpu_manager)
        
        # Test matrix multiplication
        print("   Matrix Multiplication...")
        a = np.random.randn(200, 200).astype(np.float32)
        b = np.random.randn(200, 200).astype(np.float32)
        
        start_time = time.time()
        result = ops.matrix_multiply(a, b)
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Result shape: {result.shape}")
        print(f"     Backend used: {device_info.backend.value}")
        
        # Test vector similarity  
        print("   Vector Similarity (Cosine)...")
        vectors1 = np.random.randn(100, 64).astype(np.float32)
        vectors2 = np.random.randn(100, 64).astype(np.float32)
        
        start_time = time.time()
        similarity = ops.vector_similarity(vectors1, vectors2, metric='cosine')
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Similarity matrix shape: {similarity.shape}")
        
        # Test embedding lookup
        print("   Embedding Lookup...")
        embeddings = np.random.randn(5000, 128).astype(np.float32)
        indices = np.random.randint(0, 5000, 50)
        
        start_time = time.time()
        looked_up = ops.batch_embedding_lookup(embeddings, indices)
        end_time = time.time()
        
        print(f"     ✅ Completed in {end_time - start_time:.4f}s")
        print(f"     Looked up embeddings shape: {looked_up.shape}")
        
        # Test memory management
        print(f"\n💾 Memory Management:")
        memory_mgr = GPUMemoryManager(gpu_manager)
        memory_stats = memory_mgr.get_memory_stats()
        
        print(f"   Total Memory: {memory_stats.total / 1024**3:.1f} GB")
        print(f"   Allocated: {memory_stats.allocated / 1024**2:.1f} MB")
        print(f"   Utilization: {memory_stats.utilization:.1%}")
        
        # Estimate batch size for different operations
        batch_size_512 = memory_mgr.estimate_batch_size((512, 512))
        batch_size_1024 = memory_mgr.estimate_batch_size((1024, 1024))
        
        print(f"   Recommended batch size (512x512): {batch_size_512}")
        print(f"   Recommended batch size (1024x1024): {batch_size_1024}")
        
        # Run quick benchmark if GPU is available
        if gpu_manager.is_gpu_available():
            print(f"\n⚡ Running Quick GPU vs CPU Benchmark:")
            
            benchmark = GPUBenchmark(gpu_manager)
            speedups = benchmark.quick_benchmark()
            
            for operation, speedup in speedups.items():
                if speedup > 1.1:
                    print(f"   {operation}: {speedup:.2f}x speedup 🚀")
                elif speedup > 0.9:
                    print(f"   {operation}: ~{speedup:.2f}x (similar performance) ⚖️")
                else:
                    print(f"   {operation}: {speedup:.2f}x (CPU faster) ⚠️")
        else:
            print(f"\n⚠️  GPU not available - all operations using CPU fallback")
        
        # Test compatibility
        print(f"\n🔧 System Compatibility:")
        
        try:
            import torch
            print(f"   PyTorch: {torch.__version__} ✅")
            if torch.cuda.is_available():
                print(f"   CUDA: Available ✅")
                print(f"   CUDA Devices: {torch.cuda.device_count()}")
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    print(f"   Primary GPU: {props.name}")
                    print(f"   Compute Capability: {props.major}.{props.minor}")
                    print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
            else:
                print(f"   CUDA: Not available ⚠️")
        except ImportError:
            print(f"   PyTorch: Not installed ⚠️")
        
        try:
            import cupy
            print(f"   CuPy: {cupy.__version__} ✅")
        except ImportError:
            print(f"   CuPy: Not installed (optional) ⚠️")
            
        # Test batch processing with memory management
        print(f"\n🔄 Testing Batch Processing:")
        
        # Generate larger dataset
        large_data = [np.random.randn(100, 50).astype(np.float32) for _ in range(20)]
        
        def dummy_process(batch):
            """Dummy processing function."""
            return np.sum(batch, axis=0)
        
        start_time = time.time()
        results = memory_mgr.batch_process_with_memory_management(
            large_data, dummy_process, batch_size=5
        )
        end_time = time.time()
        
        print(f"   Processed {len(large_data)} items in {len(results)} batches")
        print(f"   Total time: {end_time - start_time:.4f}s")
        
        print(f"\n🎉 GPU Framework Test Completed Successfully!")
        print(f"\n📊 Summary:")
        print(f"   - GPU Available: {'YES' if gpu_manager.is_gpu_available() else 'NO'}")
        print(f"   - Active Backend: {device_info.backend.value}")
        print(f"   - Device: {device_info.name}")
        print(f"   - All operations completed without errors ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpu_framework_simple()
    sys.exit(0 if success else 1)