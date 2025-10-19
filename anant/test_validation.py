"""
Quick validation test for anant library

Simple test to verify the anant library is working correctly.
"""

import sys
import traceback


def test_anant_basic_functionality():
    """Test basic anant functionality"""
    
    print("🧪 Testing anant library basic functionality...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from anant.classes.hypergraph import Hypergraph
        from anant.classes.property_store import PropertyStore
        from anant.classes.incidence_store import IncidenceStore
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        print("   ✅ All imports successful")
        
        # Test basic hypergraph creation
        print("2. Testing hypergraph creation...")
        sample_setsystem = {
            "edge1": ["node1", "node2", "node3"],
            "edge2": ["node2", "node3", "node4"],
            "edge3": ["node1", "node4", "node5"]
        }
        
        hg = Hypergraph(setsystem=sample_setsystem)
        print(f"   ✅ Hypergraph created: {hg.num_nodes} nodes, {hg.num_edges} edges")
        
        # Test basic operations
        print("3. Testing basic operations...")
        stats = hg.get_statistics()
        assert stats["num_nodes"] > 0
        assert stats["num_edges"] > 0
        print(f"   ✅ Statistics: {stats}")
        
        # Test degree computation
        degree = hg.degree("node1")
        print(f"   ✅ Node degree computed: node1 has degree {degree}")
        
        # Test neighbors
        neighbors = hg.neighbors("node1")
        print(f"   ✅ Neighbors computed: node1 has {len(neighbors)} neighbors")
        
        # Test with properties
        print("4. Testing with properties...")
        node_props = {
            "node1": {"type": "A", "value": 10},
            "node2": {"type": "B", "value": 20},
            "node3": {"type": "A", "value": 15}
        }
        
        hg_with_props = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=node_props
        )
        
        node1_props = hg_with_props.get_node_properties("node1")
        assert node1_props["type"] == "A"
        assert node1_props["value"] == 10
        print("   ✅ Properties working correctly")
        
        # Test DataFrame conversion
        print("5. Testing DataFrame conversion...")
        df = hg.to_dataframe("incidences")
        assert len(df) > 0
        print(f"   ✅ DataFrame conversion: {len(df)} incidences")
        
        print("\n🎉 All tests passed! anant library is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Traceback:")
        traceback.print_exc()
        return False


def test_polars_backend():
    """Test that Polars backend is working"""
    print("\n🔍 Testing Polars backend...")
    
    try:
        import polars as pl
        
        # Test basic Polars functionality
        df = pl.DataFrame({
            "edge": ["edge1", "edge1", "edge2"],
            "node": ["node1", "node2", "node3"],
            "weight": [1.0, 1.0, 2.0]
        })
        
        assert len(df) == 3
        assert "edge" in df.columns
        
        # Test groupby
        grouped = df.group_by("edge").agg(pl.col("node").count())
        assert len(grouped) == 2
        
        print("   ✅ Polars backend working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Polars backend test failed: {e}")
        return False


def test_performance_basic():
    """Basic performance test"""
    print("\n⚡ Testing basic performance...")
    
    try:
        import time
        from anant.classes.hypergraph import Hypergraph
        
        # Create larger dataset
        large_setsystem = {}
        for i in range(100):
            edge_id = f"edge_{i}"
            nodes = [f"node_{j}" for j in range(i, i+5)]  # 5 nodes per edge
            large_setsystem[edge_id] = nodes
        
        start_time = time.perf_counter()
        hg = Hypergraph(setsystem=large_setsystem)
        stats = hg.get_statistics()
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        print(f"   ✅ Created hypergraph with {stats['num_nodes']} nodes, {stats['num_edges']} edges in {execution_time:.3f}s")
        
        if execution_time < 1.0:
            print("   ✅ Performance looks good (< 1s)")
        else:
            print(f"   ⚠️  Performance slower than expected ({execution_time:.3f}s)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("ANANT LIBRARY VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test basic functionality
    all_passed &= test_anant_basic_functionality()
    
    # Test Polars backend
    all_passed &= test_polars_backend()
    
    # Test basic performance
    all_passed &= test_performance_basic()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎊 ALL VALIDATION TESTS PASSED!")
        print("The anant library is ready for use.")
    else:
        print("❌ Some validation tests failed.")
        print("Please check the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)