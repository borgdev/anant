"""
GPU Benchmarking - Performance Comparison Tools

Provides comprehensive benchmarking capabilities to compare GPU vs CPU
performance and help users understand the benefits of GPU acceleration.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np

from .gpu_manager import get_gpu_manager, ComputeBackend
from .accelerated_ops import AcceleratedOperations
from .memory_manager import GPUMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    operation: str
    backend: str
    execution_time: float
    throughput: float  # operations per second
    memory_usage: Optional[int] = None
    speedup: Optional[float] = None  # vs baseline
    error: Optional[str] = None


class GPUBenchmark:
    """
    Comprehensive GPU benchmarking suite.
    
    Compares performance between GPU and CPU implementations,
    providing insights into acceleration benefits and optimal configurations.
    """
    
    def __init__(self, gpu_manager=None):
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.accelerated_ops = AcceleratedOperations(self.gpu_manager)
        self.memory_manager = GPUMemoryManager(self.gpu_manager)
        
    def run_comprehensive_benchmark(self, sizes: Optional[List[int]] = None) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark across multiple operations and sizes.
        
        Args:
            sizes: List of problem sizes to test
            
        Returns:
            Dictionary of benchmark results by operation
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000]
            
        results = {
            "matrix_multiply": [],
            "vector_similarity": [],
            "embedding_lookup": [],
            "kmeans_clustering": []
        }
        
        logger.info("Starting comprehensive GPU benchmark...")
        
        for size in sizes:
            logger.info(f"Testing problem size: {size}")
            
            # Matrix multiplication benchmark
            results["matrix_multiply"].extend(
                self.benchmark_matrix_multiply(size, size)
            )
            
            # Vector similarity benchmark
            results["vector_similarity"].extend(
                self.benchmark_vector_similarity(size, 128)  # 128-dim vectors
            )
            
            # Embedding lookup benchmark
            if size <= 1000:  # Limit for memory reasons
                results["embedding_lookup"].extend(
                    self.benchmark_embedding_lookup(size * 10, size, 128)
                )
                
            # K-means clustering benchmark
            if size <= 1000:  # K-means can be expensive
                results["kmeans_clustering"].extend(
                    self.benchmark_kmeans_clustering(size, 128, min(10, size // 10))
                )
                
        logger.info("Comprehensive benchmark completed")
        return results
        
    def benchmark_matrix_multiply(self, m: int, n: int, k: Optional[int] = None) -> List[BenchmarkResult]:
        """
        Benchmark matrix multiplication: (m x k) @ (k x n).
        
        Args:
            m: Rows in first matrix
            n: Columns in second matrix
            k: Inner dimension (defaults to n)
            
        Returns:
            List of benchmark results
        """
        if k is None:
            k = n
            
        logger.info(f"Benchmarking matrix multiplication: ({m} x {k}) @ ({k} x {n})")
        
        # Generate test data
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        
        results = []
        
        # CPU benchmark (fallback)
        cpu_result = self._benchmark_operation(
            operation_name="matrix_multiply",
            operation_func=lambda: self.accelerated_ops.fallback_manager.matrix_multiply(a, b),
            backend="cpu",
            data_size=m * k + k * n,
            operation_count=m * n * k
        )
        results.append(cpu_result)
        
        # GPU benchmark (if available)
        if self.gpu_manager.is_gpu_available():
            gpu_result = self._benchmark_operation(
                operation_name="matrix_multiply",
                operation_func=lambda: self.accelerated_ops.matrix_multiply(a, b),
                backend="gpu",
                data_size=m * k + k * n,
                operation_count=m * n * k
            )
            gpu_result.speedup = cpu_result.execution_time / gpu_result.execution_time if gpu_result.execution_time > 0 else None
            results.append(gpu_result)
            
        return results
        
    def benchmark_vector_similarity(self, n_vectors: int, dim: int, 
                                   metric: str = "cosine") -> List[BenchmarkResult]:
        """
        Benchmark vector similarity computation.
        
        Args:
            n_vectors: Number of vectors
            dim: Vector dimension
            metric: Similarity metric
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking vector similarity: {n_vectors} vectors of dimension {dim}")
        
        # Generate test data
        vectors1 = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors2 = np.random.randn(n_vectors, dim).astype(np.float32)
        
        results = []
        
        # CPU benchmark
        cpu_result = self._benchmark_operation(
            operation_name="vector_similarity",
            operation_func=lambda: self.accelerated_ops.fallback_manager.vector_similarity(vectors1, vectors2, metric),
            backend="cpu",
            data_size=2 * n_vectors * dim,
            operation_count=n_vectors * n_vectors * dim
        )
        results.append(cpu_result)
        
        # GPU benchmark (if available)
        if self.gpu_manager.is_gpu_available():
            gpu_result = self._benchmark_operation(
                operation_name="vector_similarity",
                operation_func=lambda: self.accelerated_ops.vector_similarity(vectors1, vectors2, metric),
                backend="gpu",
                data_size=2 * n_vectors * dim,
                operation_count=n_vectors * n_vectors * dim
            )
            gpu_result.speedup = cpu_result.execution_time / gpu_result.execution_time if gpu_result.execution_time > 0 else None
            results.append(gpu_result)
            
        return results
        
    def benchmark_embedding_lookup(self, vocab_size: int, batch_size: int, 
                                  embedding_dim: int) -> List[BenchmarkResult]:
        """
        Benchmark embedding lookup operations.
        
        Args:
            vocab_size: Vocabulary size
            batch_size: Batch size for lookup
            embedding_dim: Embedding dimension
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking embedding lookup: vocab={vocab_size}, batch={batch_size}, dim={embedding_dim}")
        
        # Generate test data
        embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
        indices = np.random.randint(0, vocab_size, batch_size)
        
        results = []
        
        # CPU benchmark
        cpu_result = self._benchmark_operation(
            operation_name="embedding_lookup",
            operation_func=lambda: self.accelerated_ops.fallback_manager.batch_embedding_lookup(embeddings, indices),
            backend="cpu",
            data_size=vocab_size * embedding_dim + batch_size,
            operation_count=batch_size * embedding_dim
        )
        results.append(cpu_result)
        
        # GPU benchmark (if available)
        if self.gpu_manager.is_gpu_available():
            gpu_result = self._benchmark_operation(
                operation_name="embedding_lookup",
                operation_func=lambda: self.accelerated_ops.batch_embedding_lookup(embeddings, indices),
                backend="gpu",
                data_size=vocab_size * embedding_dim + batch_size,
                operation_count=batch_size * embedding_dim
            )
            gpu_result.speedup = cpu_result.execution_time / gpu_result.execution_time if gpu_result.execution_time > 0 else None
            results.append(gpu_result)
            
        return results
        
    def benchmark_kmeans_clustering(self, n_samples: int, n_features: int, 
                                   k: int, max_iters: int = 10) -> List[BenchmarkResult]:
        """
        Benchmark K-means clustering.
        
        Args:
            n_samples: Number of data points
            n_features: Number of features
            k: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Benchmarking K-means: {n_samples} samples, {n_features} features, {k} clusters")
        
        # Generate test data
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        results = []
        
        # CPU benchmark
        cpu_result = self._benchmark_operation(
            operation_name="kmeans_clustering",
            operation_func=lambda: self.accelerated_ops.fallback_manager.kmeans_clustering(data, k, max_iters),
            backend="cpu",
            data_size=n_samples * n_features,
            operation_count=n_samples * k * max_iters
        )
        results.append(cpu_result)
        
        # GPU benchmark (if available)
        if self.gpu_manager.is_gpu_available():
            gpu_result = self._benchmark_operation(
                operation_name="kmeans_clustering",
                operation_func=lambda: self.accelerated_ops.kmeans_clustering(data, k, max_iters),
                backend="gpu",
                data_size=n_samples * n_features,
                operation_count=n_samples * k * max_iters
            )
            gpu_result.speedup = cpu_result.execution_time / gpu_result.execution_time if gpu_result.execution_time > 0 else None
            results.append(gpu_result)
            
        return results
        
    def _benchmark_operation(self, operation_name: str, operation_func: Callable,
                           backend: str, data_size: int, operation_count: int,
                           warmup_runs: int = 2, benchmark_runs: int = 5) -> BenchmarkResult:
        """
        Benchmark a single operation with proper warmup and timing.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to benchmark
            backend: Backend being used (cpu/gpu)
            data_size: Size of input data
            operation_count: Number of operations performed
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Benchmark result
        """
        logger.debug(f"Benchmarking {operation_name} on {backend}")
        
        try:
            # Warmup runs
            for _ in range(warmup_runs):
                _ = operation_func()
                
            # Benchmark runs
            times = []
            memory_usage = None
            
            for _ in range(benchmark_runs):
                # Monitor memory if GPU
                if backend == "gpu" and self.gpu_manager.is_gpu_available():
                    start_memory = self.memory_manager.get_memory_stats()
                    
                start_time = time.time()
                result = operation_func()
                end_time = time.time()
                
                if backend == "gpu" and self.gpu_manager.is_gpu_available():
                    end_memory = self.memory_manager.get_memory_stats()
                    memory_usage = end_memory.allocated - start_memory.allocated
                    
                times.append(end_time - start_time)
                
            # Calculate statistics
            avg_time = np.mean(times)
            throughput = operation_count / avg_time if avg_time > 0 else 0
            
            return BenchmarkResult(
                operation=operation_name,
                backend=backend,
                execution_time=avg_time,
                throughput=throughput,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for {operation_name} on {backend}: {e}")
            return BenchmarkResult(
                operation=operation_name,
                backend=backend,
                execution_time=float('inf'),
                throughput=0.0,
                error=str(e)
            )
            
    def generate_benchmark_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results from run_comprehensive_benchmark
            
        Returns:
            Formatted report string
        """
        report = ["=" * 80]
        report.append("GPU ACCELERATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Device information
        device_info = self.gpu_manager.get_device_info()
        report.append(f"Device: {device_info.name}")
        report.append(f"Backend: {device_info.backend.value}")
        report.append(f"Performance Score: {device_info.performance_score:.2f}")
        report.append("")
        
        # Results by operation
        for operation, operation_results in results.items():
            if not operation_results:
                continue
                
            report.append(f"Operation: {operation.replace('_', ' ').title()}")
            report.append("-" * 40)
            
            # Group by backend
            cpu_results = [r for r in operation_results if r.backend == "cpu"]
            gpu_results = [r for r in operation_results if r.backend == "gpu"]
            
            for cpu_result in cpu_results:
                if cpu_result.error:
                    report.append(f"  CPU: ERROR - {cpu_result.error}")
                else:
                    report.append(f"  CPU: {cpu_result.execution_time:.4f}s ({cpu_result.throughput:.2e} ops/s)")
                    
            for gpu_result in gpu_results:
                if gpu_result.error:
                    report.append(f"  GPU: ERROR - {gpu_result.error}")
                else:
                    speedup_str = f", {gpu_result.speedup:.2f}x speedup" if gpu_result.speedup else ""
                    memory_str = f", {gpu_result.memory_usage / 1024**2:.1f} MB" if gpu_result.memory_usage else ""
                    report.append(f"  GPU: {gpu_result.execution_time:.4f}s ({gpu_result.throughput:.2e} ops/s{speedup_str}{memory_str})")
                    
            report.append("")
            
        # Summary
        all_results = [r for results_list in results.values() for r in results_list]
        gpu_results = [r for r in all_results if r.backend == "gpu" and not r.error and r.speedup]
        
        if gpu_results:
            avg_speedup = np.mean([r.speedup for r in gpu_results])
            max_speedup = max(r.speedup for r in gpu_results)
            report.append("SUMMARY")
            report.append("-" * 40)
            report.append(f"Average GPU Speedup: {avg_speedup:.2f}x")
            report.append(f"Maximum GPU Speedup: {max_speedup:.2f}x")
            report.append(f"GPU Acceleration Available: {'YES' if self.gpu_manager.is_gpu_available() else 'NO'}")
        else:
            report.append("SUMMARY")
            report.append("-" * 40)
            report.append("GPU Acceleration: Not available or not beneficial")
            
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def quick_benchmark(self) -> Dict[str, float]:
        """
        Run a quick benchmark to assess GPU vs CPU performance.
        
        Returns:
            Dictionary with speedup factors
        """
        logger.info("Running quick GPU benchmark...")
        
        # Small matrix multiplication test
        size = 512
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # CPU timing
        start = time.time()
        _ = self.accelerated_ops.fallback_manager.matrix_multiply(a, b)
        cpu_time = time.time() - start
        
        speedups = {"matrix_multiply": 1.0}  # CPU baseline
        
        # GPU timing (if available)
        if self.gpu_manager.is_gpu_available():
            try:
                # Warmup
                _ = self.accelerated_ops.matrix_multiply(a, b)
                
                start = time.time()
                _ = self.accelerated_ops.matrix_multiply(a, b)
                gpu_time = time.time() - start
                
                speedups["matrix_multiply"] = cpu_time / gpu_time if gpu_time > 0 else 1.0
                
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")
                speedups["matrix_multiply"] = 1.0
                
        logger.info(f"Quick benchmark completed - GPU speedup: {speedups['matrix_multiply']:.2f}x")
        return speedups