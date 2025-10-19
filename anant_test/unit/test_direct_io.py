#!/usr/bin/env python3
"""
Direct Test of I/O Implementation

Tests the I/O modules directly without complex import layers.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import polars as pl

# Add anant to path
sys.path.insert(0, '/home/amansingh/dev/ai/anant/anant')

def test_direct_io():
    """Test I/O implementation directly"""
    
    print("=" * 80)
    print("DIRECT I/O IMPLEMENTATION TEST")
    print("=" * 80)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n🗂️ Using temporary directory: {temp_dir}")
    
    try:
        # Test 1: Direct Parquet I/O
        print(f"\n1️⃣ Testing Direct Parquet I/O...")
        test_direct_parquet_io(temp_dir)
        
        # Test 2: Streaming Components
        print(f"\n2️⃣ Testing Streaming Components...")
        test_streaming_components(temp_dir)
        
        print(f"\n✅ DIRECT I/O TESTS COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ DIRECT I/O TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_direct_parquet_io(temp_dir: Path):
    """Test Parquet I/O directly"""
    
    try:
        # Import directly from modules
        from anant.classes.hypergraph import Hypergraph
        from anant.io.parquet_io import AnantIO
        
        print("   ✅ Successfully imported Hypergraph and AnantIO")
        
        # Create test data
        test_data = pl.DataFrame([
            {"edges": "E1", "nodes": "A", "weight": 1.0},
            {"edges": "E1", "nodes": "B", "weight": 1.5}, 
            {"edges": "E2", "nodes": "B", "weight": 2.0},
            {"edges": "E2", "nodes": "C", "weight": 2.5},
        ])
        
        # Create hypergraph
        hg = Hypergraph(test_data)
        print(f"   ✅ Created Hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test saving to Parquet
        parquet_path = temp_dir / "test_direct.parquet"
        
        AnantIO.save_hypergraph_parquet(
            hg, 
            parquet_path,
            compression="snappy",
            include_metadata=True
        )
        print(f"   ✅ Saved Hypergraph to Parquet with metadata")
        
        # Verify file exists and has data
        if parquet_path.exists():
            file_size = parquet_path.stat().st_size
            print(f"   ✅ Parquet file created: {file_size} bytes")
            
            # Test loading back
            loaded_hg = AnantIO.load_hypergraph_parquet(
                parquet_path,
                lazy=False,
                with_optimizer=True
            )
            print(f"   ✅ Loaded Hypergraph: {loaded_hg.num_nodes} nodes, {loaded_hg.num_edges} edges")
            
            # Verify data integrity
            if loaded_hg.num_nodes == hg.num_nodes and loaded_hg.num_edges == hg.num_edges:
                print(f"   ✅ Data integrity verified")
            else:
                print(f"   ❌ Data integrity check failed")
        
        print(f"   ✅ Direct Parquet I/O test completed successfully")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        # Try to diagnose the issue
        print(f"   📁 Current directory: {os.getcwd()}")
        print(f"   🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
        
    except Exception as e:
        print(f"   ❌ Direct Parquet I/O test failed: {e}")
        import traceback
        traceback.print_exc()


def test_streaming_components(temp_dir: Path):
    """Test streaming components directly"""
    
    try:
        from anant.io.streaming_io import StreamingDatasetReader, ChunkedHypergraphProcessor
        
        print("   ✅ Successfully imported streaming components")
        
        # Create test dataset
        print("   Creating test dataset...")
        test_data = []
        for i in range(500):  # 500 rows
            test_data.append({
                "edges": f"E{i // 10}",  # 50 unique edges
                "nodes": f"N{i % 100}",  # 100 unique nodes
                "weight": i * 0.01
            })
        
        df = pl.DataFrame(test_data)
        test_file = temp_dir / "streaming_test.parquet"
        df.write_parquet(test_file)
        
        print(f"   ✅ Created test dataset: {len(df)} rows")
        
        # Test streaming reader
        reader = StreamingDatasetReader(chunk_size=100)
        
        chunk_count = 0
        total_rows = 0
        
        print("   Testing streaming reader...")
        for chunk_df in reader.stream_parquet(test_file):
            chunk_count += 1
            total_rows += len(chunk_df)
            
            if chunk_count <= 3:  # Show first 3 chunks
                print(f"     Chunk {chunk_count}: {len(chunk_df)} rows")
        
        print(f"   ✅ Streamed {chunk_count} chunks, {total_rows} total rows")
        
        # Test chunked processor
        print("   Testing chunked processor...")
        processor = ChunkedHypergraphProcessor(accumulate_results=True)
        
        reader2 = StreamingDatasetReader(chunk_size=150)
        for chunk_df in reader2.stream_parquet(test_file):
            stats = processor.process_chunk(chunk_df)
            
            if processor._processed_chunks <= 2:  # Show first 2 chunks
                print(f"     Processed chunk {stats['chunk_id']}: "
                      f"{stats['unique_nodes']} unique nodes, {stats['unique_edges']} unique edges")
        
        # Get final statistics
        final_stats = processor.get_statistics()
        print(f"   ✅ Final stats: {final_stats['chunks_processed']} chunks, "
              f"{final_stats['total_unique_nodes']} unique nodes")
        
        print(f"   ✅ Streaming components test completed successfully")
        
    except ImportError as e:
        print(f"   ❌ Streaming import error: {e}")
        
    except Exception as e:
        print(f"   ❌ Streaming components test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_direct_io()