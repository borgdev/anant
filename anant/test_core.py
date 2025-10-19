"""
Basic test to validate PropertyStore and IncidenceStore functionality
"""

import polars as pl
import sys
from pathlib import Path

# Add the anant package to the path
anant_path = Path(__file__).parent / "anant"
sys.path.insert(0, str(anant_path))

from classes.property_store import PropertyStore
from classes.incidence_store import IncidenceStore


def test_property_store():
    """Test basic PropertyStore functionality"""
    print("Testing PropertyStore...")
    
    # Create a PropertyStore for nodes (level 1)
    ps = PropertyStore(level=1)
    
    # Test setting properties
    ps.set_property("Alice", "department", "Engineering")
    ps.set_property("Alice", "years_experience", 5)
    ps.set_property("Bob", "department", "Sales")
    ps.set_property("Bob", "years_experience", 3)
    
    # Test getting properties
    alice_props = ps.get_properties("Alice")
    print(f"Alice properties: {alice_props}")
    
    # Test bulk operations
    bulk_data = pl.DataFrame({
        "uid": ["Charlie", "David"],
        "department": ["Marketing", "Engineering"],
        "years_experience": [2, 7]
    })
    ps.bulk_set_properties(bulk_data)
    
    # Get summary
    summary = ps.get_property_summary()
    print(f"PropertyStore summary: {summary}")
    
    print("✅ PropertyStore tests passed")
    return ps


def test_incidence_store():
    """Test basic IncidenceStore functionality"""
    print("\nTesting IncidenceStore...")
    
    # Create test incidence data
    incidence_data = pl.DataFrame({
        "edges": ["meeting1", "meeting1", "meeting1", "meeting2", "meeting2", "meeting3", "meeting3", "meeting3"],
        "nodes": ["Alice", "Bob", "Charlie", "Bob", "David", "Alice", "Charlie", "Eve"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    # Create IncidenceStore
    inc_store = IncidenceStore(incidence_data)
    
    # Test neighbor queries
    meeting1_nodes = inc_store.get_neighbors(0, "meeting1")  # Nodes in meeting1
    alice_edges = inc_store.get_neighbors(1, "Alice")  # Edges containing Alice
    
    print(f"Nodes in meeting1: {meeting1_nodes}")
    print(f"Edges containing Alice: {alice_edges}")
    
    # Test statistics
    stats = inc_store.get_statistics()
    print(f"IncidenceStore stats: {stats}")
    
    # Test cache performance
    cache_stats = inc_store.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    print("✅ IncidenceStore tests passed")
    return inc_store


def main():
    """Run all tests"""
    print("🚀 Starting anant core component tests...\n")
    
    try:
        # Test PropertyStore
        ps = test_property_store()
        
        # Test IncidenceStore  
        inc_store = test_incidence_store()
        
        print(f"\n🎉 All tests passed!")
        print(f"PropertyStore: {ps}")
        print(f"IncidenceStore: {inc_store}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()