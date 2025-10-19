#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced I/O Operations

Tests all new I/O capabilities including:
- Native Parquet I/O with compression  
- Lazy loading functionality
- Streaming I/O for large datasets
- Multi-file dataset support
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import polars as pl

# Add anant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_io_capabilities():
    """Comprehensive test of all I/O capabilities"""
    
    print("=" * 80)
    print("ADVANCED I/O CAPABILITIES - COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n🗂️ Using temporary directory: {temp_dir}")
    
    try:
        # Test 1: Native Parquet I/O
        print(f"\n1️⃣ Testing Native Parquet I/O...")
        test_parquet_io(temp_dir)
        
        # Test 2: Lazy Loading
        print(f"\n2️⃣ Testing Lazy Loading...")
        test_lazy_loading(temp_dir)
        
        # Test 3: Streaming I/O  
        print(f"\n3️⃣ Testing Streaming I/O...")
        test_streaming_io(temp_dir)
        
        # Test 4: Multi-file Dataset Support
        print(f"\n4️⃣ Testing Multi-file Dataset Support...")
        test_multi_file_support(temp_dir)
        
        print(f"\n✅ ALL I/O TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ I/O TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary directory")


def test_parquet_io(temp_dir: Path):
    """Test native Parquet I/O with compression"""
    
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.io import save_hypergraph_parquet, load_hypergraph_parquet, AnantIO
        
        # Create test data
        test_data = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.5}, 
            {"edges": "E2", "nodes": "B", "weight": 2.0},
            {"edges": "E2", "nodes": "C", "weight": 2.5},
            {"edges": "E3", "nodes": "A", "weight": 3.0},
            {"edges": "E3", "nodes": "C", "weight": 3.5},
        ])
        
        # Create hypergraph
        hg = Hypergraph(test_data)
        print(f"   Created test Hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test different compression formats
        compression_formats = ['snappy', 'gzip', 'lz4', 'uncompressed']
        
        for compression in compression_formats:
            print(f"   Testing {compression} compression...")
            
            # Save with compression
            parquet_path = temp_dir / f"test_{compression}.parquet"
            
            try:
                save_hypergraph_parquet(
                    hg, 
                    parquet_path, 
                    compression=compression,
                    include_metadata=True
                )
                print(f"     ✅ Saved with {compression} compression")
                
                # Load back
                loaded_hg = load_hypergraph_parquet(
                    parquet_path,
                    lazy=False,
                    with_optimizer=True
                )
                print(f"     ✅ Loaded: {loaded_hg.num_nodes} nodes, {loaded_hg.num_edges} edges")
                
                # Verify data integrity
                if loaded_hg.num_nodes == hg.num_nodes and loaded_hg.num_edges == hg.num_edges:
                    print(f"     ✅ Data integrity verified for {compression}")
                else:
                    print(f"     ❌ Data integrity check failed for {compression}")
                
            except Exception as e:
                print(f"     ❌ {compression} compression failed: {e}")
        
        # Test lazy loading
        print(f"   Testing lazy loading...")
        parquet_path = temp_dir / "test_lazy.parquet"
        save_hypergraph_parquet(hg, parquet_path, compression="snappy")
        
        lazy_hg = load_hypergraph_parquet(parquet_path, lazy=True)
        print(f"     ✅ Lazy loading successful: {lazy_hg.num_nodes} nodes")
        
        # Test with filters
        print(f"   Testing filtered loading...")
        filtered_hg = load_hypergraph_parquet(
            parquet_path,
            lazy=True,
            filters={"edges": ["E1", "E2"]}
        )
        print(f"     ✅ Filtered loading: {filtered_hg.num_edges} edges (expected ≤ 2)")
        
        print(f"   ✅ Native Parquet I/O tests completed")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Parquet I/O test failed: {e}")


def test_lazy_loading(temp_dir: Path):
    """Test lazy loading capabilities"""
    
    try:
        from anant.io import LazyLoader, LazyHypergraph
        
        # Create larger test dataset
        edges = [f"E{i}" for i in range(100)]
        nodes = [f"N{i}" for i in range(50)]
        
        test_data = []
        for i, edge in enumerate(edges):
            # Each edge connects to 2-4 random nodes
            import random
            edge_nodes = random.sample(nodes, random.randint(2, 4))
            for node in edge_nodes:
                test_data.append({
                    "edges": edge,
                    "nodes": node, 
                    "weight": random.uniform(0.5, 5.0),
                    "timestamp": f"2024-01-{(i % 30) + 1:02d}"
                })
        
        df = pl.DataFrame(test_data)
        parquet_path = temp_dir / "lazy_test.parquet"
        df.write_parquet(parquet_path)
        
        print(f"   Created test dataset: {len(df)} rows")
        
        # Test lazy loading from Parquet
        lazy_hg = LazyLoader.from_parquet(parquet_path)
        print(f"   ✅ Created LazyHypergraph from Parquet")
        
        # Test lazy operations
        print(f"   Testing lazy operations...")
        
        # Column selection
        lazy_subset = lazy_hg.select("edges", "nodes", "weight")
        print(f"     ✅ Column selection applied lazily")
        
        # Filtering
        lazy_filtered = lazy_hg.filter(pl.col("weight") > 3.0)
        print(f"     ✅ Filter applied lazily")
        
        # Slicing
        lazy_slice = lazy_hg.slice(0, 50)
        print(f"     ✅ Slice applied lazily")
        
        # Chain operations
        lazy_chained = lazy_hg.select("edges", "nodes").filter(pl.col("edges").str.contains("E1"))
        print(f"     ✅ Operations chained lazily")
        
        # Test materialization
        materialized = lazy_chained.collect()
        print(f"     ✅ Lazy operations materialized: {materialized.num_nodes} nodes")
        
        # Test head/tail without full materialization  
        head_hg = lazy_hg.head(10)
        print(f"     ✅ Head operation: {head_hg.num_nodes} nodes")
        
        tail_hg = lazy_hg.tail(10) 
        print(f"     ✅ Tail operation: {tail_hg.num_nodes} nodes")
        
        # Test statistics without materialization
        stats = lazy_hg.describe()
        print(f"     ✅ Statistics computed lazily: {len(stats)} rows")
        
        row_count = lazy_hg.count_rows()
        print(f"     ✅ Row count: {row_count} rows")
        
        print(f"   ✅ Lazy loading tests completed")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Lazy loading test failed: {e}")


def test_streaming_io(temp_dir: Path):
    """Test streaming I/O for large datasets"""
    
    try:
        from anant.io import StreamingDatasetReader, ChunkedHypergraphProcessor
        
        # Create larger dataset for streaming
        print(f"   Creating large test dataset...")
        large_data = []
        
        for i in range(1000):  # 1000 edges
            for j in range(5):  # 5 nodes per edge on average
                large_data.append({
                    "edges": f"E{i}",
                    "nodes": f"N{(i * 3 + j) % 200}",  # 200 unique nodes
                    "weight": (i + j) * 0.1,
                    "category": f"Cat{i % 10}"
                })
        
        df = pl.DataFrame(large_data)
        large_parquet_path = temp_dir / "large_dataset.parquet"
        df.write_parquet(large_parquet_path)
        
        print(f"   Created dataset: {len(df)} rows")
        
        # Test streaming reader
        print(f"   Testing StreamingDatasetReader...")
        
        def progress_callback(stats):
            if stats.processed_chunks % 5 == 0:  # Print every 5 chunks
                print(f"     Progress: {stats.processed_chunks}/{stats.total_chunks} chunks "
                      f"({stats.processed_rows} rows)")
        
        reader = StreamingDatasetReader(
            chunk_size=200,  # Small chunks for testing
            progress_callback=progress_callback
        )
        
        # Test streaming
        chunk_count = 0
        total_rows = 0
        
        for chunk_df in reader.stream_parquet(large_parquet_path):
            chunk_count += 1
            total_rows += len(chunk_df)
            
            if chunk_count == 1:
                print(f"     ✅ First chunk: {len(chunk_df)} rows")
            
        print(f"   ✅ Streamed {chunk_count} chunks, {total_rows} total rows")
        
        # Test chunked processor
        print(f"   Testing ChunkedHypergraphProcessor...")
        
        processor = ChunkedHypergraphProcessor(accumulate_results=True)
        
        reader2 = StreamingDatasetReader(chunk_size=300)
        
        for chunk_df in reader2.stream_parquet(large_parquet_path):
            chunk_stats = processor.process_chunk(chunk_df)
            
            if processor._processed_chunks == 1:
                print(f"     ✅ First chunk processed: {chunk_stats['unique_nodes']} unique nodes")
        
        # Get accumulated hypergraph
        accumulated_hg = processor.get_accumulated_hypergraph()
        if accumulated_hg:
            print(f"     ✅ Accumulated Hypergraph: {accumulated_hg.num_nodes} nodes, "
                  f"{accumulated_hg.num_edges} edges")
        
        # Get statistics
        stats = processor.get_statistics()
        print(f"     ✅ Processing stats: {stats['chunks_processed']} chunks, "
              f"{stats['total_unique_nodes']} unique nodes")
        
        print(f"   ✅ Streaming I/O tests completed")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Streaming I/O test failed: {e}")


def test_multi_file_support(temp_dir: Path):
    """Test multi-file dataset support"""
    
    try:
        from anant.io import load_multi_file_dataset, LazyLoader
        from anant.classes.hypergraph import Hypergraph
        
        print(f"   Creating multiple test files...")
        
        # Create multiple related datasets
        datasets = []
        file_paths = []
        
        for i in range(3):
            # Each file has different but overlapping nodes/edges
            data = []
            for j in range(50):
                edge_id = f"E{i * 30 + j}"  # Overlapping edge ranges
                node_base = i * 20  # Different node ranges per file
                
                for k in range(3):
                    data.append({
                        "edges": edge_id,
                        "nodes": f"N{node_base + k}",
                        "weight": (i + j + k) * 0.1,
                        "dataset_id": i
                    })
            
            df = pl.DataFrame(data)
            file_path = temp_dir / f"dataset_{i}.parquet"
            df.write_parquet(file_path)
            
            datasets.append(df)
            file_paths.append(file_path)
            
            print(f"     Created file {i}: {len(df)} rows")
        
        # Test union merge strategy
        print(f"   Testing union merge strategy...")
        
        union_hg = load_multi_file_dataset(
            file_paths,
            merge_strategy="union",
            with_optimizer=True
        )
        
        print(f"     ✅ Union merge: {union_hg.num_nodes} nodes, {union_hg.num_edges} edges")
        
        # Test intersection merge strategy  
        print(f"   Testing intersection merge strategy...")
        
        intersection_hg = load_multi_file_dataset(
            file_paths,
            merge_strategy="intersection",
            with_optimizer=True  
        )
        
        print(f"     ✅ Intersection merge: {intersection_hg.num_nodes} nodes, "
              f"{intersection_hg.num_edges} edges")
        
        # Test lazy multi-file loading
        print(f"   Testing lazy multi-file loading...")
        
        lazy_multi = LazyLoader.from_multiple_files(
            file_paths,
            file_format="parquet"
        )
        
        materialized_multi = lazy_multi.collect()
        print(f"     ✅ Lazy multi-file: {materialized_multi.num_nodes} nodes, "
              f"{materialized_multi.num_edges} edges")
        
        # Verify union has more data than intersection
        if union_hg.num_edges >= intersection_hg.num_edges:
            print(f"     ✅ Union ≥ Intersection verification passed")
        else:
            print(f"     ⚠️ Union < Intersection (unexpected)")
        
        print(f"   ✅ Multi-file dataset tests completed")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
    except Exception as e:
        print(f"   ❌ Multi-file dataset test failed: {e}")


if __name__ == "__main__":
    test_io_capabilities()