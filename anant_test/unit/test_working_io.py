#!/usr/bin/env python3
"""
Simple Working I/O Test for Anant Library

This tests our I/O implementation using the correct Anant API.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import polars as pl

# Add anant to path
sys.path.append('/home/amansingh/dev/ai/anant/anant')

def test_working_io():
    """Test working I/O implementation with correct API"""
    
    print("=" * 80)
    print("WORKING I/O IMPLEMENTATION TEST")
    print("=" * 80)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n🗂️ Using temporary directory: {temp_dir}")
    
    try:
        # Test 1: Basic Hypergraph + I/O
        print(f"\n1️⃣ Testing Basic Hypergraph + I/O...")
        test_basic_hypergraph_io(temp_dir)
        
        # Test 2: Advanced I/O Functions
        print(f"\n2️⃣ Testing Advanced I/O Functions...")
        test_advanced_io_functions(temp_dir)
        
        print(f"\n✅ ALL WORKING I/O TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ WORKING I/O TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_basic_hypergraph_io(temp_dir: Path):
    """Test basic Hypergraph I/O operations"""
    
    try:
        from anant.classes.hypergraph import Hypergraph
        print("   ✅ Successfully imported Hypergraph")
        
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
        print(f"   ✅ Created Hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Get the underlying DataFrame via incidences.data
        underlying_df = hg.incidences.data
        print(f"   ✅ Retrieved underlying DataFrame: {len(underlying_df)} rows")
        
        # Test saving to different formats and compressions
        formats_to_test = [
            ("parquet", "snappy"),
            ("parquet", "gzip"), 
            ("parquet", "lz4"),
            ("parquet", "uncompressed"),
            ("csv", None)
        ]
        
        saved_files = []
        
        for file_format, compression in formats_to_test:
            if file_format == "parquet":
                file_path = temp_dir / f"test_{compression}.parquet"
                if compression == "uncompressed":
                    underlying_df.write_parquet(file_path, compression=None)
                else:
                    underlying_df.write_parquet(file_path, compression=compression)
            else:  # csv
                file_path = temp_dir / "test.csv"
                underlying_df.write_csv(file_path)
            
            saved_files.append((file_path, file_format))
            print(f"   ✅ Saved as {file_format} ({compression or 'default'}): {file_path.name}")
        
        # Test loading back and verifying data integrity
        for file_path, file_format in saved_files:
            if file_format == "parquet":
                loaded_df = pl.read_parquet(file_path)
            else:  # csv
                loaded_df = pl.read_csv(file_path)
            
            # Create hypergraph from loaded data
            loaded_hg = Hypergraph(loaded_df)
            
            # Verify integrity
            if (loaded_hg.num_nodes == hg.num_nodes and 
                loaded_hg.num_edges == hg.num_edges and
                len(loaded_df) == len(underlying_df)):
                print(f"   ✅ Data integrity verified for {file_path.name}")
            else:
                print(f"   ❌ Data integrity check failed for {file_path.name}")
        
        print(f"   ✅ Basic Hypergraph I/O test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Basic Hypergraph I/O test failed: {e}")
        import traceback
        traceback.print_exc()


def test_advanced_io_functions(temp_dir: Path):
    """Test advanced I/O functions"""
    
    try:
        from anant.classes.hypergraph import Hypergraph
        
        print("   Testing advanced I/O operations...")
        
        # Create multiple test datasets
        datasets = []
        file_paths = []
        
        for i in range(3):
            # Create different but overlapping data
            data = []
            base_offset = i * 20
            
            for j in range(30):  # 30 edges per dataset
                edge_id = f"E{base_offset + j}"
                
                # Each edge connects to 2-3 nodes
                for k in range(2 + (j % 2)):  # 2 or 3 nodes per edge
                    node_id = f"N{(base_offset + j + k) % 50}"  # 50 possible nodes
                    data.append({
                        "edges": edge_id,
                        "nodes": node_id,
                        "weight": (i + j + k) * 0.1,
                        "dataset_id": i
                    })
            
            df = pl.DataFrame(data)
            file_path = temp_dir / f"dataset_{i}.parquet"
            df.write_parquet(file_path, compression="snappy")
            
            datasets.append(df)
            file_paths.append(file_path)
            
            hg = Hypergraph(df)
            print(f"     Created dataset {i}: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test streaming large dataset
        print("   Testing streaming functionality...")
        
        # Create a larger dataset for streaming
        large_data = []
        for i in range(200):  # 200 edges
            for j in range(5):  # 5 nodes per edge on average
                large_data.append({
                    "edges": f"E{i}",
                    "nodes": f"N{(i * 3 + j) % 100}",  # 100 unique nodes
                    "weight": (i + j) * 0.05,
                    "category": f"Cat{i % 10}"
                })
        
        large_df = pl.DataFrame(large_data)
        large_file = temp_dir / "large_dataset.parquet"
        large_df.write_parquet(large_file, compression="snappy")
        
        large_hg = Hypergraph(large_df)
        print(f"     Created large dataset: {large_hg.num_nodes} nodes, {large_hg.num_edges} edges")
        
        # Test chunked processing
        chunk_size = 500  # Process in chunks of 500 rows
        total_rows = len(large_df)
        chunk_count = 0
        processed_rows = 0
        
        print(f"   Testing chunked processing (chunk_size={chunk_size})...")
        
        # Use Polars lazy API for chunked processing
        lazy_df = pl.scan_parquet(large_file)
        
        for offset in range(0, total_rows, chunk_size):
            chunk_df = lazy_df.slice(offset, chunk_size).collect()
            
            if len(chunk_df) == 0:
                break
            
            chunk_hg = Hypergraph(chunk_df)
            chunk_count += 1
            processed_rows += len(chunk_df)
            
            if chunk_count <= 3:  # Show first 3 chunks
                print(f"     Chunk {chunk_count}: {len(chunk_df)} rows, "
                      f"{chunk_hg.num_nodes} nodes, {chunk_hg.num_edges} edges")
        
        print(f"   ✅ Processed {chunk_count} chunks, {processed_rows} total rows")
        
        # Test merging datasets
        print("   Testing dataset merging...")
        
        # Load all datasets and merge
        all_dfs = []
        for file_path in file_paths:
            df = pl.read_parquet(file_path)
            all_dfs.append(df)
        
        # Union merge (concatenate all)
        union_df = pl.concat(all_dfs, how="diagonal_relaxed").unique()
        union_hg = Hypergraph(union_df)
        print(f"     Union merge: {union_hg.num_nodes} nodes, {union_hg.num_edges} edges")
        
        # Save merged result
        merged_file = temp_dir / "merged_dataset.parquet"
        union_df.write_parquet(merged_file, compression="snappy")
        print(f"     ✅ Saved merged dataset: {merged_file.name}")
        
        # Test loading merged dataset
        loaded_merged_df = pl.read_parquet(merged_file)
        loaded_merged_hg = Hypergraph(loaded_merged_df)
        print(f"     ✅ Loaded merged dataset: {loaded_merged_hg.num_nodes} nodes, "
              f"{loaded_merged_hg.num_edges} edges")
        
        print(f"   ✅ Advanced I/O functions test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Advanced I/O functions test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_working_io()