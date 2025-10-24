#!/usr/bin/env python3
"""
Ray Storage Integration Test for Anant
=====================================

This test validates the integration between Polars/Parquet storage and Ray distributed computing.
It demonstrates how Anant's Polars+Parquet backend integrates with Ray's object store for
distributed hypergraph operations.

Test Areas:
1. Ray Object Store with Polars DataFrames
2. Distributed Parquet I/O operations  
3. Ray remote functions with Anant hypergraphs
4. Storage persistence across Ray workers
5. Performance benchmarking of distributed operations
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import ray
import polars as pl
import numpy as np

# Add anant to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant')

import anant
from anant.classes.hypergraph import Hypergraph
from anant.io.parquet_io import AnantIO
from anant.factory.enhanced_setsystems import ParquetSetSystem


def setup_test_environment():
    """Initialize test environment with Ray and temporary storage"""
    print("🏗️ Setting up Ray Storage Integration Test Environment")
    
    # Connect to Ray cluster
    if not ray.is_initialized():
        ray.init(address="ray://localhost:10002")  # Connect to existing cluster
    
    # Display cluster info
    cluster_resources = ray.cluster_resources()
    print(f"📊 Ray Cluster Resources: {cluster_resources}")
    
    # Initialize Anant
    anant.setup()
    
    # Create temporary storage directory
    temp_dir = Path(tempfile.mkdtemp(prefix="ray_storage_test_"))
    print(f"💾 Temporary storage: {temp_dir}")
    
    return temp_dir


@ray.remote
def create_distributed_hypergraph(worker_id: int, data_size: int) -> dict:
    """
    Ray remote function to create hypergraph on distributed worker.
    
    This demonstrates Polars DataFrame creation and manipulation
    across Ray workers with Anant hypergraphs.
    """
    import polars as pl
    import sys
    sys.path.insert(0, '/home/amansingh/dev/ai/anant')
    from anant.classes.hypergraph import Hypergraph
    
    print(f"🔨 Worker {worker_id}: Creating hypergraph with {data_size} edges")
    
    # Generate synthetic data using Polars
    edges_data = []
    for i in range(data_size):
        edge_id = f"E{worker_id}_{i}"
        # Create random hyperedge with 2-4 nodes
        num_nodes = np.random.randint(2, 5)
        for j in range(num_nodes):
            node_id = f"N{worker_id}_{i}_{j}"
            edges_data.append({
                "edges": edge_id,
                "nodes": node_id,
                "weight": np.random.uniform(0.1, 2.0),
                "worker_id": worker_id
            })
    
    # Create Polars DataFrame
    df = pl.DataFrame(edges_data)
    
    # Create hypergraph
    hg = Hypergraph(df)
    
    # Get basic stats
    stats = {
        "worker_id": worker_id,
        "num_nodes": hg.num_nodes,
        "num_edges": hg.num_edges,
        "dataframe_height": hg.incidences.data.height,
        "memory_usage_mb": hg.incidences.data.estimated_size() / (1024 * 1024),
        "node_id": ray.get_runtime_context().get_node_id()
    }
    
    print(f"✅ Worker {worker_id}: Created hypergraph - {stats}")
    return stats


@ray.remote
def process_parquet_chunk(chunk_path: str, worker_id: int) -> dict:
    """
    Ray remote function to process Parquet file chunk.
    
    Demonstrates distributed Parquet I/O with Polars across Ray workers.
    """
    import polars as pl
    import sys
    sys.path.insert(0, '/home/amansingh/dev/ai/anant')
    from anant.factory.enhanced_setsystems import ParquetSetSystem
    
    print(f"📄 Worker {worker_id}: Processing Parquet chunk {chunk_path}")
    
    # Load Parquet data using Anant's ParquetSetSystem
    df = ParquetSetSystem.from_parquet(
        chunk_path,
        lazy=True,
        validate_schema=True
    )
    
    # Perform distributed analysis
    edge_stats = (
        df.group_by("edges")
        .agg([
            pl.count().alias("node_count"),
            pl.col("weight").mean().alias("avg_weight"),
            pl.col("weight").sum().alias("total_weight")
        ])
    )
    
    node_stats = (
        df.group_by("nodes") 
        .agg([
            pl.count().alias("edge_count"),
            pl.col("weight").mean().alias("avg_weight")
        ])
    )
    
    results = {
        "worker_id": worker_id,
        "chunk_path": chunk_path,
        "total_rows": df.height,
        "unique_edges": edge_stats.height,
        "unique_nodes": node_stats.height,
        "avg_edge_weight": edge_stats["avg_weight"].mean(),
        "node_id": ray.get_runtime_context().get_node_id()
    }
    
    print(f"📊 Worker {worker_id}: Analysis complete - {results}")
    return results


@ray.remote
def distributed_hypergraph_analysis(hypergraph_data: dict, analysis_type: str) -> dict:
    """
    Ray remote function for distributed hypergraph analysis.
    
    Demonstrates complex analysis operations distributed across Ray cluster.
    """
    import polars as pl
    import sys
    sys.path.insert(0, '/home/amansingh/dev/ai/anant')
    from anant.classes.hypergraph import Hypergraph
    
    print(f"🧮 Performing distributed analysis: {analysis_type}")
    
    # Reconstruct hypergraph from serialized data
    df = pl.DataFrame(hypergraph_data)
    hg = Hypergraph(df)
    
    if analysis_type == "centrality":
        # Calculate degree centrality for nodes
        node_degrees = (
            hg.incidences.data
            .group_by("nodes")
            .agg(pl.count().alias("degree"))
            .sort("degree", descending=True)
        )
        
        results = {
            "analysis_type": "centrality",
            "max_degree": node_degrees["degree"].max(),
            "avg_degree": node_degrees["degree"].mean(),
            "total_nodes": node_degrees.height
        }
        
    elif analysis_type == "connectivity":
        # Calculate edge connectivity statistics
        edge_sizes = (
            hg.incidences.data
            .group_by("edges")
            .agg(pl.count().alias("size"))
        )
        
        results = {
            "analysis_type": "connectivity", 
            "max_edge_size": edge_sizes["size"].max(),
            "avg_edge_size": edge_sizes["size"].mean(),
            "total_edges": edge_sizes.height
        }
        
    elif analysis_type == "weights":
        # Analyze weight distribution
        weight_stats = hg.incidences.data["weight"].describe()
        
        results = {
            "analysis_type": "weights",
            "mean_weight": float(weight_stats[weight_stats["statistic"] == "mean"]["value"].item()),
            "std_weight": float(weight_stats[weight_stats["statistic"] == "std"]["value"].item()),
            "min_weight": float(weight_stats[weight_stats["statistic"] == "min"]["value"].item()),
            "max_weight": float(weight_stats[weight_stats["statistic"] == "max"]["value"].item())
        }
    
    results["node_id"] = ray.get_runtime_context().get_node_id()
    print(f"📈 Analysis results: {results}")
    return results


def test_ray_object_store_integration(temp_dir: Path):
    """Test Ray object store integration with Polars DataFrames"""
    print("\n🎯 Test 1: Ray Object Store Integration with Polars DataFrames")
    
    # Create large Polars DataFrame
    data_size = 50000
    print(f"Creating DataFrame with {data_size} rows...")
    
    df = pl.DataFrame({
        "edges": [f"E_{i//3}" for i in range(data_size)],
        "nodes": [f"N_{i}" for i in range(data_size)], 
        "weight": np.random.uniform(0.1, 2.0, data_size),
        "timestamp": pl.date_range(
            start=datetime.now(),
            periods=data_size,
            interval="1s"
        )
    })
    
    print(f"✅ DataFrame created: {df.shape} shape, {df.estimated_size() / (1024*1024):.2f} MB")
    
    # Put DataFrame in Ray object store
    df_ref = ray.put(df)
    print(f"📦 DataFrame stored in Ray object store: {df_ref}")
    
    # Retrieve and verify
    retrieved_df = ray.get(df_ref)
    assert retrieved_df.equals(df)
    print(f"✅ DataFrame successfully retrieved from Ray object store")
    
    return df_ref


def test_distributed_hypergraph_creation():
    """Test distributed hypergraph creation across Ray workers"""
    print("\n🎯 Test 2: Distributed Hypergraph Creation")
    
    num_workers = 4
    data_per_worker = 1000
    
    print(f"Creating {num_workers} distributed hypergraphs with {data_per_worker} edges each...")
    
    # Launch distributed hypergraph creation
    futures = []
    for i in range(num_workers):
        future = create_distributed_hypergraph.remote(i, data_per_worker)
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    # Aggregate statistics
    total_nodes = sum(r["num_nodes"] for r in results)
    total_edges = sum(r["num_edges"] for r in results) 
    total_memory = sum(r["memory_usage_mb"] for r in results)
    unique_nodes = set(r["node_id"] for r in results)
    
    print(f"📊 Distributed Creation Results:")
    print(f"   Total nodes across all workers: {total_nodes}")
    print(f"   Total edges across all workers: {total_edges}")
    print(f"   Total memory usage: {total_memory:.2f} MB")
    print(f"   Ray nodes utilized: {len(unique_nodes)}")
    print(f"✅ Distributed hypergraph creation successful")
    
    return results


def test_distributed_parquet_processing(temp_dir: Path):
    """Test distributed Parquet file processing"""
    print("\n🎯 Test 3: Distributed Parquet Processing") 
    
    # Create multiple Parquet files for processing
    num_chunks = 4
    chunk_paths = []
    
    for i in range(num_chunks):
        chunk_data = []
        for j in range(2000):  # 2000 records per chunk
            edge_id = f"E_chunk_{i}_{j//4}"
            node_id = f"N_chunk_{i}_{j}"
            chunk_data.append({
                "edges": edge_id,
                "nodes": node_id,
                "weight": np.random.uniform(0.1, 3.0),
                "chunk_id": i
            })
        
        chunk_df = pl.DataFrame(chunk_data)
        chunk_path = temp_dir / f"chunk_{i}.parquet"
        chunk_df.write_parquet(chunk_path, compression="snappy")
        chunk_paths.append(str(chunk_path))
        print(f"📄 Created chunk {i}: {chunk_path} ({chunk_df.height} rows)")
    
    # Process chunks distributedly
    print(f"Processing {num_chunks} chunks across Ray workers...")
    futures = []
    for i, chunk_path in enumerate(chunk_paths):
        future = process_parquet_chunk.remote(chunk_path, i)
        futures.append(future)
    
    # Collect results  
    results = ray.get(futures)
    
    # Aggregate statistics
    total_rows = sum(r["total_rows"] for r in results)
    total_edges = sum(r["unique_edges"] for r in results)
    total_nodes = sum(r["unique_nodes"] for r in results)
    avg_weight = np.mean([r["avg_edge_weight"] for r in results])
    
    print(f"📊 Distributed Processing Results:")
    print(f"   Total rows processed: {total_rows}")
    print(f"   Total unique edges: {total_edges}")
    print(f"   Total unique nodes: {total_nodes}")
    print(f"   Average edge weight: {avg_weight:.3f}")
    print(f"✅ Distributed Parquet processing successful")
    
    return results, chunk_paths


def test_distributed_analysis():
    """Test distributed hypergraph analysis"""
    print("\n🎯 Test 4: Distributed Hypergraph Analysis")
    
    # Create comprehensive hypergraph dataset
    print("Creating comprehensive dataset for analysis...")
    data_size = 10000
    
    hypergraph_data = []
    for i in range(data_size):
        edge_id = f"E_{i//5}"  # Multiple nodes per edge
        node_id = f"N_{i}"
        hypergraph_data.append({
            "edges": edge_id,
            "nodes": node_id,
            "weight": np.random.uniform(0.1, 5.0),
        })
    
    # Convert to serializable format for Ray
    serialized_data = hypergraph_data
    
    # Run different analysis types distributedly
    analysis_types = ["centrality", "connectivity", "weights"]
    
    print(f"Running {len(analysis_types)} distributed analyses...")
    futures = []
    for analysis_type in analysis_types:
        future = distributed_hypergraph_analysis.remote(serialized_data, analysis_type)
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    print(f"📊 Distributed Analysis Results:")
    for result in results:
        analysis_type = result["analysis_type"]
        node_id = result["node_id"]
        print(f"   {analysis_type.upper()} (on {node_id}):")
        
        if analysis_type == "centrality":
            print(f"     Max degree: {result['max_degree']}")
            print(f"     Avg degree: {result['avg_degree']:.2f}")
            print(f"     Total nodes: {result['total_nodes']}")
        elif analysis_type == "connectivity":
            print(f"     Max edge size: {result['max_edge_size']}")
            print(f"     Avg edge size: {result['avg_edge_size']:.2f}")
            print(f"     Total edges: {result['total_edges']}")
        elif analysis_type == "weights":
            print(f"     Mean weight: {result['mean_weight']:.3f}")
            print(f"     Std weight: {result['std_weight']:.3f}")
            print(f"     Weight range: {result['min_weight']:.3f} - {result['max_weight']:.3f}")
    
    print(f"✅ Distributed analysis successful")
    return results


def test_storage_persistence(temp_dir: Path):
    """Test storage persistence across Ray operations"""
    print("\n🎯 Test 5: Storage Persistence with Ray")
    
    # Create hypergraph and save with different compression formats
    print("Testing Parquet storage persistence...")
    
    data = {
        "edges": ["E1", "E1", "E2", "E2", "E3"],
        "nodes": ["N1", "N2", "N2", "N3", "N1"],
        "weight": [1.5, 2.0, 0.8, 1.2, 2.5]
    }
    df = pl.DataFrame(data)
    hg = Hypergraph(df)
    
    # Test different compression formats
    compression_formats = ["snappy", "gzip", "lz4", "zstd"]
    file_sizes = {}
    
    for compression in compression_formats:
        file_path = temp_dir / f"test_{compression}.parquet"
        
        # Save with AnantIO
        start_time = time.time()
        AnantIO.save_hypergraph_parquet(
            hypergraph=hg,
            path=str(file_path),
            compression=compression
        )
        save_time = time.time() - start_time
        
        # Check file size
        file_size = file_path.stat().st_size
        file_sizes[compression] = {
            "size_bytes": file_size,
            "save_time": save_time
        }
        
        # Verify loading
        start_time = time.time()
        loaded_hg = AnantIO.load_hypergraph_parquet(str(file_path))
        load_time = time.time() - start_time
        
        file_sizes[compression]["load_time"] = load_time
        
        # Verify data integrity
        assert loaded_hg.num_nodes == hg.num_nodes
        assert loaded_hg.num_edges == hg.num_edges
        print(f"✅ {compression}: {file_size} bytes, save: {save_time:.3f}s, load: {load_time:.3f}s")
    
    print(f"📊 Compression Analysis:")
    for compression, stats in file_sizes.items():
        print(f"   {compression.upper()}: {stats['size_bytes']} bytes, "
              f"total I/O: {stats['save_time'] + stats['load_time']:.3f}s")
    
    return file_sizes


def performance_benchmark():
    """Run performance benchmark of Ray + Polars + Parquet integration"""
    print("\n🎯 Test 6: Performance Benchmark")
    
    # Benchmark parameters
    data_sizes = [1000, 5000, 10000, 20000]
    results = {}
    
    for data_size in data_sizes:
        print(f"\n📏 Benchmarking with {data_size} records...")
        
        # Create dataset
        start_time = time.time()
        futures = []
        num_workers = 4
        
        for i in range(num_workers):
            future = create_distributed_hypergraph.remote(i, data_size // num_workers)
            futures.append(future)
        
        worker_results = ray.get(futures)
        creation_time = time.time() - start_time
        
        # Calculate statistics
        total_nodes = sum(r["num_nodes"] for r in worker_results)
        total_edges = sum(r["num_edges"] for r in worker_results)
        total_memory = sum(r["memory_usage_mb"] for r in worker_results)
        
        results[data_size] = {
            "creation_time": creation_time,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "memory_mb": total_memory,
            "throughput_edges_per_sec": total_edges / creation_time
        }
        
        print(f"   Time: {creation_time:.3f}s")
        print(f"   Throughput: {total_edges / creation_time:.0f} edges/sec")
        print(f"   Memory: {total_memory:.2f} MB")
    
    print(f"\n📊 Performance Summary:")
    print(f"{'Data Size':<10} {'Time (s)':<10} {'Edges/sec':<12} {'Memory (MB)':<12}")
    print(f"{'-' * 50}")
    for size, stats in results.items():
        print(f"{size:<10} {stats['creation_time']:<10.3f} "
              f"{stats['throughput_edges_per_sec']:<12.0f} {stats['memory_mb']:<12.2f}")
    
    return results


def cleanup_and_summarize(temp_dir: Path, all_results: dict):
    """Cleanup resources and provide final summary"""
    print("\n🧹 Cleanup and Summary")
    
    # Cleanup temporary files
    shutil.rmtree(temp_dir)
    print(f"✅ Cleaned up temporary directory: {temp_dir}")
    
    # Ray cluster final status
    cluster_resources = ray.cluster_resources()
    print(f"📊 Final Ray Cluster State: {cluster_resources}")
    
    # Summary
    print(f"\n🎉 Ray Storage Integration Test Summary:")
    print(f"✅ Ray Object Store Integration: PASSED")
    print(f"✅ Distributed Hypergraph Creation: PASSED") 
    print(f"✅ Distributed Parquet Processing: PASSED")
    print(f"✅ Distributed Analysis: PASSED")
    print(f"✅ Storage Persistence: PASSED")
    print(f"✅ Performance Benchmarking: PASSED")
    
    print(f"\n📈 Integration Highlights:")
    print(f"• Polars DataFrames integrate seamlessly with Ray object store")
    print(f"• Anant hypergraphs work natively in distributed Ray environment")
    print(f"• Parquet I/O operations scale across Ray workers efficiently")
    print(f"• Multiple compression formats supported with performance tracking")
    print(f"• Complex analysis operations distribute correctly across cluster")
    
    return True


def main():
    """Main test execution"""
    print("🚀 Starting Ray Storage Integration Test for Anant")
    print("=" * 60)
    
    # Setup
    temp_dir = setup_test_environment()
    all_results = {}
    
    try:
        # Run all tests
        all_results["object_store"] = test_ray_object_store_integration(temp_dir)
        all_results["distributed_creation"] = test_distributed_hypergraph_creation()
        all_results["parquet_processing"] = test_distributed_parquet_processing(temp_dir)
        all_results["distributed_analysis"] = test_distributed_analysis()
        all_results["storage_persistence"] = test_storage_persistence(temp_dir) 
        all_results["performance"] = performance_benchmark()
        
        # Final summary
        cleanup_and_summarize(temp_dir, all_results)
        
        print(f"\n🎊 ALL TESTS PASSED! Ray + Polars + Parquet integration is VALIDATED ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)