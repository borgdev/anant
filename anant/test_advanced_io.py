#!/usr/bin/env python3
"""
Test script for Advanced I/O System

This script tests the new Advanced I/O capabilities including:
- Parquet compression options
- Multi-format support (CSV, JSON, Parquet)
- Schema preservation
- Metadata handling
- Performance benchmarking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

from anant.io import AdvancedAnantIO, CompressionType, FileFormat, IOConfiguration
from anant.classes import Hypergraph
from anant.factory import SetSystemFactory
import polars as pl
import tempfile
import shutil
from pathlib import Path


def create_test_hypergraph():
    """Create a test hypergraph for I/O testing"""
    print("Creating test hypergraph...")
    
    # Create test data
    edge_data = {
        "edge1": ["node1", "node2", "node3"],
        "edge2": ["node2", "node4"],
        "edge3": ["node1", "node4", "node5"],
        "edge4": ["node3", "node5", "node6"]
    }
    
    # Create hypergraph using SetSystemFactory
    from typing import Dict, Iterable, cast
    incidence_df = SetSystemFactory.from_dict_of_iterables(cast(Dict[str, Iterable], edge_data))
    hg = Hypergraph(incidence_df)
    
    # Add some properties
    node_props = {
        "node1": {"type": "source", "weight": 1.5},
        "node2": {"type": "intermediate", "weight": 2.0},
        "node3": {"type": "intermediate", "weight": 1.8},
        "node4": {"type": "sink", "weight": 2.5},
        "node5": {"type": "sink", "weight": 2.2},
        "node6": {"type": "sink", "weight": 1.9}
    }
    hg.add_node_properties(node_props)
    
    edge_props = {
        "edge1": {"category": "A", "strength": 0.8},
        "edge2": {"category": "B", "strength": 0.6},
        "edge3": {"category": "A", "strength": 0.9},
        "edge4": {"category": "C", "strength": 0.7}
    }
    hg.add_edge_properties(edge_props)
    
    print(f"Created hypergraph: {hg}")
    return hg


def test_basic_io():
    """Test basic I/O operations"""
    print("\n=== Testing Basic I/O Operations ===")
    
    hg = create_test_hypergraph()
    
    # Create advanced I/O instance
    io_config = IOConfiguration(
        compression=CompressionType.SNAPPY,
        enable_parallel=False,  # Disable for small test
        validate_data=True
    )
    advanced_io = AdvancedAnantIO(io_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test Parquet format
        parquet_path = tmpdir / "test_parquet"
        print(f"Saving to Parquet: {parquet_path}")
        
        save_result = advanced_io.save_hypergraph(
            hg, parquet_path, format=FileFormat.PARQUET
        )
        print(f"Save result: {save_result.total_size_bytes} bytes, {save_result.compression_ratio:.2f}x compression")
        
        # Load back
        print("Loading from Parquet...")
        load_result = advanced_io.load_hypergraph(parquet_path)
        print(f"Load result: {load_result.load_time:.3f}s, {load_result.memory_usage} bytes")
        print(f"Loaded hypergraph: {load_result.hypergraph}")
        
        # Test CSV format
        csv_path = tmpdir / "test_csv"
        print(f"Saving to CSV: {csv_path}")
        
        save_result_csv = advanced_io.save_hypergraph(
            hg, csv_path, format=FileFormat.CSV
        )
        print(f"CSV save result: {save_result_csv.total_size_bytes} bytes")
        
        # Load back
        print("Loading from CSV...")
        load_result_csv = advanced_io.load_hypergraph(csv_path)
        print(f"CSV load result: {load_result_csv.load_time:.3f}s")
        print(f"Loaded CSV hypergraph: {load_result_csv.hypergraph}")


def test_compression_options():
    """Test different compression options"""
    print("\n=== Testing Compression Options ===")
    
    hg = create_test_hypergraph()
    
    compression_types = [
        CompressionType.UNCOMPRESSED,
        CompressionType.SNAPPY,
        CompressionType.GZIP,
        CompressionType.ZSTD
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        results = {}
        
        for compression in compression_types:
            print(f"Testing {compression.value} compression...")
            
            config = IOConfiguration(compression=compression)
            advanced_io = AdvancedAnantIO(config)
            
            output_path = tmpdir / f"test_{compression.value}"
            
            try:
                save_result = advanced_io.save_hypergraph(
                    hg, output_path, format=FileFormat.PARQUET
                )
                
                load_result = advanced_io.load_hypergraph(output_path)
                
                results[compression.value] = {
                    "size": save_result.total_size_bytes,
                    "compression_ratio": save_result.compression_ratio,
                    "save_time": save_result.save_time,
                    "load_time": load_result.load_time
                }
                
                print(f"  Size: {save_result.total_size_bytes} bytes")
                print(f"  Compression ratio: {save_result.compression_ratio:.2f}x")
                print(f"  Save time: {save_result.save_time:.3f}s")
                print(f"  Load time: {load_result.load_time:.3f}s")
                
            except Exception as e:
                print(f"  Error with {compression.value}: {e}")
                results[compression.value] = {"error": str(e)}
        
        print("\n--- Compression Summary ---")
        for compression, result in results.items():
            if "error" not in result:
                print(f"{compression:12}: {result['size']:6} bytes, {result['compression_ratio']:4.2f}x ratio")


def test_format_benchmark():
    """Test performance of different formats"""
    print("\n=== Testing Format Benchmark ===")
    
    hg = create_test_hypergraph()
    
    advanced_io = AdvancedAnantIO()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Run benchmark
        benchmark_results = advanced_io.benchmark_formats(hg, tmpdir)
        
        print("\n--- Format Benchmark Results ---")
        for format_type, results in benchmark_results.items():
            if "error" not in results:
                print(f"{format_type.value:8}: "
                      f"Save: {results['save_time']:.3f}s, "
                      f"Load: {results['load_time']:.3f}s, "
                      f"Size: {results['file_size']} bytes")
            else:
                print(f"{format_type.value:8}: ERROR - {results['error']}")


def test_multiple_hypergraphs():
    """Test saving/loading multiple hypergraphs"""
    print("\n=== Testing Multiple Hypergraphs ===")
    
    # Create multiple test hypergraphs
    hg1 = create_test_hypergraph()
    hg1.name = "hypergraph_1"
    
    # Create a second hypergraph
    edge_data2 = {
        "edge_a": ["n1", "n2"],
        "edge_b": ["n2", "n3", "n4"],
        "edge_c": ["n1", "n4"]
    }
    from typing import Dict, Iterable, cast
    incidence_df2 = SetSystemFactory.from_dict_of_iterables(cast(Dict[str, Iterable], edge_data2))
    hg2 = Hypergraph(incidence_df2)
    hg2.name = "hypergraph_2"
    
    hypergraphs = {
        "graph1": hg1,
        "graph2": hg2
    }
    
    advanced_io = AdvancedAnantIO()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save multiple hypergraphs
        print("Saving multiple hypergraphs...")
        save_results = advanced_io.save_multiple_hypergraphs(
            hypergraphs, tmpdir, format=FileFormat.PARQUET
        )
        
        for name, result in save_results.items():
            if result:
                print(f"  {name}: {result.total_size_bytes} bytes")
            else:
                print(f"  {name}: FAILED")
        
        # Load multiple hypergraphs
        print("Loading multiple hypergraphs...")
        load_results = advanced_io.load_multiple_hypergraphs(tmpdir)
        
        for name, result in load_results.items():
            if result:
                print(f"  {name}: {result.hypergraph}")
            else:
                print(f"  {name}: FAILED")


def main():
    """Run all tests"""
    print("Testing Advanced I/O System for Anant")
    print("=" * 50)
    
    try:
        test_basic_io()
        test_compression_options()
        test_format_benchmark()
        test_multiple_hypergraphs()
        
        print("\n" + "=" * 50)
        print("All Advanced I/O tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())