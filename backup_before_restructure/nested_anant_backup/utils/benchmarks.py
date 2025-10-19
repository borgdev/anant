"""
Comprehensive benchmarking framework for anant library

Performance testing and validation against HyperNetX and other alternatives
to validate the 5-10x performance improvements claimed.
"""

import polars as pl
import time
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Comprehensive performance testing framework for anant library
    
    Validates performance improvements vs HyperNetX including:
    - Construction time comparisons
    - Memory usage analysis  
    - Property operation benchmarks
    - I/O performance testing
    - Analysis algorithm comparisons
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmarking framework
        
        Parameters
        ----------
        output_dir : Path, optional
            Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.test_datasets = {}
        self._start_time = None
        
        # System info
        self.system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": self._get_platform_info()
        }
        
        logger.info(f"Benchmark framework initialized. Output: {self.output_dir}")
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information"""
        import platform
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    
    def _measure_time_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
        """
        Measure execution time and peak memory usage of a function
        
        Returns
        -------
        Tuple[Any, float, float]
            (result, execution_time_seconds, peak_memory_mb)
        """
        gc.collect()  # Clean up before measurement
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / (1024 * 1024)
        
        execution_time = end_time - start_time
        peak_memory = end_memory - start_memory
        
        return result, execution_time, peak_memory
    
    def generate_test_datasets(self, sizes: List[int] = [1000, 10000, 100000]) -> None:
        """
        Generate test datasets of various sizes
        
        Parameters
        ----------
        sizes : List[int]
            Dataset sizes to generate
        """
        import random
        
        logger.info(f"Generating test datasets for sizes: {sizes}")
        
        for size in sizes:
            # Generate random hypergraph data
            num_edges = size // 10  # Average 10 nodes per edge
            edges = {}
            
            for i in range(num_edges):
                edge_id = f"edge_{i}"
                # Random edge size between 2 and 20
                edge_size = random.randint(2, min(20, size // 100))
                nodes = [f"node_{random.randint(0, size-1)}" for _ in range(edge_size)]
                edges[edge_id] = list(set(nodes))  # Remove duplicates
            
            # Generate properties
            node_properties = {}
            for i in range(size):
                node_id = f"node_{i}"
                node_properties[node_id] = {
                    "type": random.choice(["A", "B", "C"]),
                    "value": random.uniform(0, 100),
                    "category": random.randint(1, 10)
                }
            
            edge_properties = {}
            for edge_id in edges.keys():
                edge_properties[edge_id] = {
                    "weight": random.uniform(0.1, 2.0),
                    "importance": random.uniform(0, 1),
                    "created": datetime.now().isoformat()
                }
            
            self.test_datasets[size] = {
                "edges": edges,
                "node_properties": node_properties,
                "edge_properties": edge_properties
            }
        
        logger.info(f"Generated {len(self.test_datasets)} test datasets")
    
    def benchmark_construction(self, sizes: List[int] = [1000, 10000, 100000]) -> Dict[str, Any]:
        """
        Benchmark hypergraph construction performance
        
        Parameters
        ---------- 
        sizes : List[int]
            Dataset sizes to test
            
        Returns
        -------
        Dict[str, Any]
            Construction benchmark results
        """
        logger.info("Running construction benchmarks...")
        
        if not self.test_datasets:
            self.generate_test_datasets(sizes)
        
        results = {}
        
        for size in sizes:
            if size not in self.test_datasets:
                continue
                
            dataset = self.test_datasets[size]
            size_results = {}
            
            # Test anant construction
            def create_anant_hypergraph():
                from ..classes.hypergraph import Hypergraph
                return Hypergraph(
                    setsystem=dataset["edges"],
                    node_properties=dataset["node_properties"],
                    edge_properties=dataset["edge_properties"]
                )
            
            hg_anant, time_anant, mem_anant = self._measure_time_memory(create_anant_hypergraph)
            
            size_results["anant"] = {
                "construction_time": time_anant,
                "memory_usage_mb": mem_anant,
                "num_edges": hg_anant.num_edges,
                "num_nodes": hg_anant.num_nodes,
                "num_incidences": hg_anant.num_incidences
            }
            
            # Test HyperNetX construction (if available)
            try:
                import hypernetx as hnx
                
                def create_hnx_hypergraph():
                    return hnx.Hypergraph(dataset["edges"])
                
                hg_hnx, time_hnx, mem_hnx = self._measure_time_memory(create_hnx_hypergraph)
                
                size_results["hypernetx"] = {
                    "construction_time": time_hnx,
                    "memory_usage_mb": mem_hnx,
                    "num_edges": len(hg_hnx.edges),
                    "num_nodes": len(hg_hnx.nodes),
                    "num_incidences": sum(len(hg_hnx.edges[e]) for e in hg_hnx.edges)
                }
                
                # Calculate speedup
                size_results["speedup"] = {
                    "construction_time": time_hnx / time_anant if time_anant > 0 else float('inf'),
                    "memory_reduction": (mem_hnx - mem_anant) / mem_hnx * 100 if mem_hnx > 0 else 0
                }
                
            except ImportError:
                logger.warning("HyperNetX not available for comparison")
                size_results["hypernetx"] = None
                size_results["speedup"] = None
            
            results[size] = size_results
            logger.info(f"Completed construction benchmark for size {size}")
        
        self.results["construction"] = results
        return results
    
    def benchmark_property_operations(self, sizes: List[int] = [1000, 10000, 100000]) -> Dict[str, Any]:
        """
        Benchmark property storage and retrieval operations
        
        Parameters
        ----------
        sizes : List[int]
            Dataset sizes to test
            
        Returns
        -------
        Dict[str, Any]
            Property operation benchmark results
        """
        logger.info("Running property operation benchmarks...")
        
        if not self.test_datasets:
            self.generate_test_datasets(sizes)
        
        results = {}
        
        for size in sizes:
            if size not in self.test_datasets:
                continue
                
            dataset = self.test_datasets[size]
            
            # Create hypergraphs
            from ..classes.hypergraph import Hypergraph
            hg_anant = Hypergraph(setsystem=dataset["edges"])
            
            # Test bulk property setting
            def bulk_set_anant():
                props_df = pl.DataFrame([
                    {"uid": k, **v} for k, v in dataset["node_properties"].items()
                ])
                hg_anant.add_node_properties(props_df)
            
            _, time_bulk_anant, mem_bulk_anant = self._measure_time_memory(bulk_set_anant)
            
            # Test individual property getting
            nodes_sample = list(dataset["node_properties"].keys())[:min(1000, len(dataset["node_properties"]))]
            
            def get_properties_anant():
                for node in nodes_sample:
                    hg_anant.get_node_properties(node)
            
            _, time_get_anant, _ = self._measure_time_memory(get_properties_anant)
            
            # Test property analysis
            def analyze_properties_anant():
                stats = hg_anant._node_properties.get_property_summary()
                return stats
            
            _, time_analyze_anant, _ = self._measure_time_memory(analyze_properties_anant)
            
            results[size] = {
                "anant": {
                    "bulk_set_time": time_bulk_anant,
                    "bulk_set_memory": mem_bulk_anant,
                    "get_properties_time": time_get_anant,
                    "analyze_time": time_analyze_anant,
                    "properties_per_second": len(nodes_sample) / time_get_anant if time_get_anant > 0 else float('inf')
                }
            }
            
            logger.info(f"Completed property benchmark for size {size}")
        
        self.results["property_operations"] = results
        return results
    
    def benchmark_io_operations(self, sizes: List[int] = [1000, 10000, 100000]) -> Dict[str, Any]:
        """
        Benchmark I/O performance (CSV vs Parquet)
        
        Parameters
        ----------
        sizes : List[int]
            Dataset sizes to test
            
        Returns
        -------
        Dict[str, Any]
            I/O benchmark results
        """
        logger.info("Running I/O operation benchmarks...")
        
        if not self.test_datasets:
            self.generate_test_datasets(sizes)
        
        results = {}
        temp_dir = self.output_dir / "temp_io_test"
        temp_dir.mkdir(exist_ok=True)
        
        for size in sizes:
            if size not in self.test_datasets:
                continue
                
            dataset = self.test_datasets[size]
            
            # Create hypergraph
            from ..classes.hypergraph import Hypergraph
            hg = Hypergraph(
                setsystem=dataset["edges"],
                node_properties=dataset["node_properties"],
                edge_properties=dataset["edge_properties"]
            )
            
            # Test parquet save/load
            parquet_path = temp_dir / f"hg_{size}.parquet"
            
            def save_parquet():
                from ..io.parquet_io import AnantIO
                AnantIO.save_hypergraph_parquet(hg, parquet_path)
            
            def load_parquet():
                from ..io.parquet_io import AnantIO
                return AnantIO.load_hypergraph_parquet(parquet_path)
            
            _, save_time, save_memory = self._measure_time_memory(save_parquet)
            hg_loaded, load_time, load_memory = self._measure_time_memory(load_parquet)
            
            # Get file sizes
            parquet_size = sum(f.stat().st_size for f in parquet_path.rglob("*.parquet")) / (1024 * 1024)
            
            # Test CSV comparison (basic)
            csv_path = temp_dir / f"hg_{size}.csv"
            
            def save_csv():
                df = hg.to_dataframe("incidences")
                df.write_csv(csv_path)
            
            def load_csv():
                return pl.read_csv(csv_path)
            
            _, csv_save_time, csv_save_memory = self._measure_time_memory(save_csv)
            _, csv_load_time, csv_load_memory = self._measure_time_memory(load_csv)
            
            csv_size = csv_path.stat().st_size / (1024 * 1024) if csv_path.exists() else 0
            
            results[size] = {
                "parquet": {
                    "save_time": save_time,
                    "load_time": load_time,
                    "save_memory": save_memory,
                    "load_memory": load_memory,
                    "file_size_mb": parquet_size
                },
                "csv": {
                    "save_time": csv_save_time,
                    "load_time": csv_load_time,
                    "save_memory": csv_save_memory,
                    "load_memory": csv_load_memory,
                    "file_size_mb": csv_size
                },
                "speedup": {
                    "save": csv_save_time / save_time if save_time > 0 else float('inf'),
                    "load": csv_load_time / load_time if load_time > 0 else float('inf'),
                    "size_reduction": (csv_size - parquet_size) / csv_size * 100 if csv_size > 0 else 0
                }
            }
            
            # Cleanup
            if parquet_path.exists():
                import shutil
                shutil.rmtree(parquet_path)
            if csv_path.exists():
                csv_path.unlink()
            
            logger.info(f"Completed I/O benchmark for size {size}")
        
        self.results["io_operations"] = results
        return results
    
    def benchmark_analysis_operations(self, sizes: List[int] = [1000, 10000]) -> Dict[str, Any]:
        """
        Benchmark analysis operations (degree, neighbors, etc.)
        
        Parameters
        ----------
        sizes : List[int]
            Dataset sizes to test
            
        Returns
        -------
        Dict[str, Any]
            Analysis benchmark results
        """
        logger.info("Running analysis operation benchmarks...")
        
        if not self.test_datasets:
            self.generate_test_datasets(sizes)
        
        results = {}
        
        for size in sizes:
            if size not in self.test_datasets:
                continue
                
            dataset = self.test_datasets[size]
            
            # Create hypergraph
            from ..classes.hypergraph import Hypergraph
            hg = Hypergraph(setsystem=dataset["edges"])
            
            # Sample nodes for testing
            nodes_sample = hg.nodes[:min(100, len(hg.nodes))]
            
            # Test degree computation
            def compute_degrees():
                degrees = {}
                for node in nodes_sample:
                    degrees[node] = hg.degree(node)
                return degrees
            
            _, degree_time, _ = self._measure_time_memory(compute_degrees)
            
            # Test neighbor computation
            def compute_neighbors():
                neighbors = {}
                for node in nodes_sample:
                    neighbors[node] = hg.neighbors(node)
                return neighbors
            
            _, neighbor_time, _ = self._measure_time_memory(compute_neighbors)
            
            # Test edge size computation
            edges_sample = hg.edges[:min(100, len(hg.edges))]
            
            def compute_edge_sizes():
                sizes = {}
                for edge in edges_sample:
                    sizes[edge] = hg.size_of_edge(edge)
                return sizes
            
            _, edge_size_time, _ = self._measure_time_memory(compute_edge_sizes)
            
            # Test statistics computation
            def compute_statistics():
                return hg.get_statistics()
            
            _, stats_time, _ = self._measure_time_memory(compute_statistics)
            
            results[size] = {
                "anant": {
                    "degree_computation": {
                        "time": degree_time,
                        "ops_per_second": len(nodes_sample) / degree_time if degree_time > 0 else float('inf')
                    },
                    "neighbor_computation": {
                        "time": neighbor_time,
                        "ops_per_second": len(nodes_sample) / neighbor_time if neighbor_time > 0 else float('inf')
                    },
                    "edge_size_computation": {
                        "time": edge_size_time,
                        "ops_per_second": len(edges_sample) / edge_size_time if edge_size_time > 0 else float('inf')
                    },
                    "statistics_computation": {
                        "time": stats_time
                    }
                }
            }
            
            logger.info(f"Completed analysis benchmark for size {size}")
        
        self.results["analysis_operations"] = results
        return results
    
    def run_comprehensive_benchmark(self, sizes: List[int] = [1000, 10000, 100000]) -> Dict[str, Any]:
        """
        Run all benchmark suites
        
        Parameters
        ----------
        sizes : List[int]
            Dataset sizes to test
            
        Returns
        -------
        Dict[str, Any]
            Complete benchmark results
        """
        logger.info("Starting comprehensive benchmark suite...")
        start_time = datetime.now()
        
        # Generate test data
        self.generate_test_datasets(sizes)
        
        # Run all benchmarks
        self.benchmark_construction(sizes)
        self.benchmark_property_operations(sizes)
        self.benchmark_io_operations(sizes)
        self.benchmark_analysis_operations(sizes[:2])  # Smaller sizes for analysis
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile comprehensive results
        comprehensive_results = {
            "metadata": {
                "anant_version": "0.1.0",
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "system_info": self.system_info,
                "test_sizes": sizes
            },
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        # Save results
        self.save_results(comprehensive_results)
        
        logger.info(f"Comprehensive benchmark completed in {duration:.2f} seconds")
        return comprehensive_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of benchmark results"""
        summary = {}
        
        # Construction summary
        if "construction" in self.results:
            construction_speedups = []
            memory_reductions = []
            
            for size, data in self.results["construction"].items():
                if data.get("speedup"):
                    construction_speedups.append(data["speedup"]["construction_time"])
                    memory_reductions.append(data["speedup"]["memory_reduction"])
            
            if construction_speedups:
                summary["construction"] = {
                    "avg_speedup": sum(construction_speedups) / len(construction_speedups),
                    "max_speedup": max(construction_speedups),
                    "avg_memory_reduction": sum(memory_reductions) / len(memory_reductions),
                    "max_memory_reduction": max(memory_reductions)
                }
        
        # I/O summary
        if "io_operations" in self.results:
            save_speedups = []
            load_speedups = []
            size_reductions = []
            
            for size, data in self.results["io_operations"].items():
                if data.get("speedup"):
                    save_speedups.append(data["speedup"]["save"])
                    load_speedups.append(data["speedup"]["load"])
                    size_reductions.append(data["speedup"]["size_reduction"])
            
            if save_speedups:
                summary["io_operations"] = {
                    "avg_save_speedup": sum(save_speedups) / len(save_speedups),
                    "avg_load_speedup": sum(load_speedups) / len(load_speedups),
                    "avg_size_reduction": sum(size_reductions) / len(size_reductions)
                }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json") -> None:
        """Save benchmark results to file"""
        output_file = self.output_dir / filename
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_file}")
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate human-readable benchmark report
        
        Parameters
        ----------
        results : Dict[str, Any], optional
            Benchmark results to report on
            
        Returns
        -------
        str
            Formatted report
        """
        if results is None:
            results = self.results
        
        report = ["="*80, "ANANT PERFORMANCE BENCHMARK REPORT", "="*80, ""]
        
        # System info
        if "metadata" in results:
            metadata = results["metadata"]
            report.extend([
                "System Information:",
                f"  CPU Cores: {metadata['system_info']['cpu_count']}",
                f"  Memory: {metadata['system_info']['memory_gb']:.1f} GB",
                f"  Platform: {metadata['system_info']['platform']['system']} {metadata['system_info']['platform']['release']}",
                f"  Test Date: {metadata['timestamp']}",
                ""
            ])
        
        # Summary
        if "summary" in results:
            summary = results["summary"]
            report.extend(["SUMMARY:", ""])
            
            if "construction" in summary:
                const = summary["construction"]
                report.extend([
                    f"Construction Performance:",
                    f"  Average Speedup: {const['avg_speedup']:.1f}x",
                    f"  Maximum Speedup: {const['max_speedup']:.1f}x", 
                    f"  Average Memory Reduction: {const['avg_memory_reduction']:.1f}%",
                    f"  Maximum Memory Reduction: {const['max_memory_reduction']:.1f}%",
                    ""
                ])
            
            if "io_operations" in summary:
                io = summary["io_operations"]
                report.extend([
                    f"I/O Performance:",
                    f"  Average Save Speedup: {io['avg_save_speedup']:.1f}x",
                    f"  Average Load Speedup: {io['avg_load_speedup']:.1f}x",
                    f"  Average File Size Reduction: {io['avg_size_reduction']:.1f}%",
                    ""
                ])
        
        # Detailed results
        if "results" in results:
            detailed = results["results"]
            report.extend(["DETAILED RESULTS:", ""])
            
            for benchmark_type, data in detailed.items():
                report.extend([f"{benchmark_type.upper()}:", ""])
                
                for size, size_data in data.items():
                    report.extend([f"  Dataset Size: {size:,}", ""])
                    
                    if "anant" in size_data:
                        anant_data = size_data["anant"]
                        if "construction_time" in anant_data:
                            report.append(f"    Anant Construction: {anant_data['construction_time']:.3f}s, {anant_data['memory_usage_mb']:.2f}MB")
                        
                        if "bulk_set_time" in anant_data:
                            report.append(f"    Anant Bulk Properties: {anant_data['bulk_set_time']:.3f}s")
                        
                        if "save_time" in anant_data:
                            report.append(f"    Anant Save: {anant_data['save_time']:.3f}s")
                    
                    if "hypernetx" in size_data and size_data["hypernetx"]:
                        hnx_data = size_data["hypernetx"]
                        report.append(f"    HyperNetX Construction: {hnx_data['construction_time']:.3f}s, {hnx_data['memory_usage_mb']:.2f}MB")
                    
                    if "speedup" in size_data and size_data["speedup"]:
                        speedup = size_data["speedup"]
                        if "construction_time" in speedup:
                            report.append(f"    Speedup: {speedup['construction_time']:.1f}x")
                        if "memory_reduction" in speedup:
                            report.append(f"    Memory Reduction: {speedup['memory_reduction']:.1f}%")
                    
                    report.append("")
        
        report.extend(["="*80, ""])
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / "benchmark_report.txt"
        with open(report_file, "w") as f:
            f.write(report_text)
        
        logger.info(f"Benchmark report saved to {report_file}")
        return report_text


# Convenience function
def run_quick_benchmark(sizes: List[int] = [1000, 10000]) -> str:
    """Run a quick benchmark and return the report"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark(sizes)
    return benchmark.generate_report(results)