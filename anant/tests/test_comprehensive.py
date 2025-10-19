"""
Comprehensive test suite for anant library

This module provides extensive unit tests, integration tests, and compatibility 
validation for all anant components.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Test fixtures and data
@pytest.fixture
def sample_setsystem():
    """Sample setsystem for testing"""
    return {
        "edge1": ["node1", "node2", "node3"],
        "edge2": ["node2", "node3", "node4"],
        "edge3": ["node1", "node4", "node5"],
        "edge4": ["node3", "node5", "node6"]
    }

@pytest.fixture
def sample_node_properties():
    """Sample node properties for testing"""
    return {
        "node1": {"type": "A", "value": 10.5, "active": True},
        "node2": {"type": "B", "value": 20.0, "active": False},
        "node3": {"type": "A", "value": 15.5, "active": True},
        "node4": {"type": "C", "value": 30.0, "active": True},
        "node5": {"type": "B", "value": 12.5, "active": False},
        "node6": {"type": "A", "value": 25.0, "active": True}
    }

@pytest.fixture
def sample_edge_properties():
    """Sample edge properties for testing"""
    return {
        "edge1": {"weight": 1.0, "importance": 0.8},
        "edge2": {"weight": 2.5, "importance": 0.6},
        "edge3": {"weight": 1.5, "importance": 0.9},
        "edge4": {"weight": 3.0, "importance": 0.7}
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestPropertyStore:
    """Test suite for PropertyStore class"""
    
    def test_basic_property_operations(self, sample_node_properties):
        """Test basic property storage and retrieval"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore(level=1)  # Node properties
        
        # Test adding properties one by one
        for uid, props in sample_node_properties.items():
            prop_data = {"uid": uid, **props}
            store.add_properties(pl.DataFrame([prop_data]))
        
        # Test retrieval
        node1_props = store.get_properties("node1")
        expected_props = sample_node_properties["node1"]
        for key, value in expected_props.items():
            assert node1_props[key] == value
        
        # Test non-existent property
        assert store.get_properties("nonexistent") == {}
        
        # Test has_properties
        assert store.has_properties("node1")
        assert not store.has_properties("nonexistent")
    
    def test_bulk_property_operations(self, sample_node_properties):
        """Test bulk property operations"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore()
        
        # Convert to DataFrame format
        props_data = [{"uid": k, **v} for k, v in sample_node_properties.items()]
        props_df = pl.DataFrame(props_data)
        
        # Bulk add
        store.add_properties(props_df)
        
        # Verify all properties added
        for uid in sample_node_properties:
            assert store.has_property(uid)
            assert store.get_property(uid) == sample_node_properties[uid]
    
    def test_property_filtering(self, sample_node_properties):
        """Test property filtering operations"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore()
        props_data = [{"uid": k, **v} for k, v in sample_node_properties.items()]
        props_df = pl.DataFrame(props_data)
        store.add_properties(props_df)
        
        # Filter by type
        type_a_nodes = store.filter_by_property("type", "A")
        expected_type_a = {"node1", "node3", "node6"}
        assert set(type_a_nodes) == expected_type_a
        
        # Filter by boolean
        active_nodes = store.filter_by_property("active", True)
        expected_active = {"node1", "node3", "node4", "node6"}
        assert set(active_nodes) == expected_active
        
        # Filter by value range
        high_value_nodes = store.filter_by_property_range("value", min_val=20.0)
        expected_high_value = {"node2", "node4", "node6"}
        assert set(high_value_nodes) == expected_high_value
    
    def test_property_analysis(self, sample_node_properties):
        """Test property analysis functionality"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore()
        props_data = [{"uid": k, **v} for k, v in sample_node_properties.items()]
        props_df = pl.DataFrame(props_data)
        store.add_properties(props_df)
        
        # Get summary
        summary = store.get_property_summary()
        
        # Check summary structure
        assert "total_entities" in summary
        assert "properties" in summary
        assert summary["total_entities"] == len(sample_node_properties)
        
        # Check property details
        value_stats = summary["properties"]["value"]
        assert "min" in value_stats
        assert "max" in value_stats
        assert "mean" in value_stats
        assert value_stats["min"] == 10.5
        assert value_stats["max"] == 30.0
    
    def test_memory_optimization(self, sample_node_properties):
        """Test memory optimization features"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore()
        props_data = [{"uid": k, **v} for k, v in sample_node_properties.items()]
        props_df = pl.DataFrame(props_data)
        store.add_properties(props_df)
        
        # Test optimization
        initial_memory = store.get_memory_usage()
        store.optimize_memory()
        optimized_memory = store.get_memory_usage()
        
        # Memory should be same or less after optimization
        assert optimized_memory <= initial_memory


class TestIncidenceStore:
    """Test suite for IncidenceStore class"""
    
    def test_basic_incidence_operations(self, sample_setsystem):
        """Test basic incidence storage and retrieval"""
        from anant.classes.incidence_store import IncidenceStore
        
        store = IncidenceStore()
        
        # Add incidences
        for edge_id, nodes in sample_setsystem.items():
            for node in nodes:
                store.add_incidence(edge_id, node)
        
        # Test edge membership
        assert "node1" in store.get_edge_members("edge1")
        assert "node2" in store.get_edge_members("edge1")
        assert "node4" not in store.get_edge_members("edge1")
        
        # Test node membership
        assert "edge1" in store.get_node_memberships("node1")
        assert "edge3" in store.get_node_memberships("node1")
        assert "edge2" not in store.get_node_memberships("node1")
    
    def test_neighbor_operations(self, sample_setsystem):
        """Test neighbor computation"""
        from anant.classes.incidence_store import IncidenceStore
        
        store = IncidenceStore()
        
        # Add incidences
        for edge_id, nodes in sample_setsystem.items():
            for node in nodes:
                store.add_incidence(edge_id, node)
        
        # Test neighbors
        node1_neighbors = store.get_neighbors("node1")
        # node1 is in edge1 (with node2, node3) and edge3 (with node4, node5)
        expected_neighbors = {"node2", "node3", "node4", "node5"}
        assert set(node1_neighbors) == expected_neighbors
        
        # Test neighbors with distance
        node1_neighbors_dist2 = store.get_neighbors("node1", max_distance=2)
        # Should include distance-2 neighbors through shared hyperedges
        assert len(node1_neighbors_dist2) >= len(expected_neighbors)
    
    def test_bulk_operations(self, sample_setsystem):
        """Test bulk incidence operations"""
        from anant.classes.incidence_store import IncidenceStore
        
        store = IncidenceStore()
        
        # Bulk add from setsystem
        store.add_setsystem(sample_setsystem)
        
        # Verify all incidences added correctly
        for edge_id, nodes in sample_setsystem.items():
            edge_members = store.get_edge_members(edge_id)
            assert set(edge_members) == set(nodes)
        
        # Test statistics
        stats = store.get_statistics()
        assert stats["num_edges"] == len(sample_setsystem)
        assert stats["num_nodes"] == len(set().union(*sample_setsystem.values()))
        assert stats["num_incidences"] == sum(len(nodes) for nodes in sample_setsystem.values())
    
    def test_caching_performance(self, sample_setsystem):
        """Test caching mechanisms"""
        from anant.classes.incidence_store import IncidenceStore
        
        store = IncidenceStore()
        store.add_setsystem(sample_setsystem)
        
        # First call - should populate cache
        neighbors1 = store.get_neighbors("node1")
        
        # Second call - should use cache (faster)
        neighbors2 = store.get_neighbors("node1")
        
        assert neighbors1 == neighbors2
        
        # Check cache statistics
        cache_stats = store.get_cache_stats()
        assert cache_stats["neighbor_hits"] > 0


class TestHypergraph:
    """Test suite for main Hypergraph class"""
    
    def test_hypergraph_construction(self, sample_setsystem, sample_node_properties, sample_edge_properties):
        """Test hypergraph construction with various inputs"""
        from anant.classes.hypergraph import Hypergraph
        
        # Test basic construction
        hg = Hypergraph(setsystem=sample_setsystem)
        assert hg.num_edges == len(sample_setsystem)
        assert hg.num_nodes == len(set().union(*sample_setsystem.values()))
        
        # Test construction with properties
        hg_with_props = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties,
            edge_properties=sample_edge_properties
        )
        assert hg_with_props.num_edges == len(sample_setsystem)
        assert len(hg_with_props.get_node_properties("node1")) > 0
        assert len(hg_with_props.get_edge_properties("edge1")) > 0
    
    def test_hypergraph_basic_operations(self, sample_setsystem):
        """Test basic hypergraph operations"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(setsystem=sample_setsystem)
        
        # Test node and edge access
        assert "node1" in hg.nodes
        assert "edge1" in hg.edges
        
        # Test degree computation
        node1_degree = hg.degree("node1")
        expected_degree = sum(1 for edge, nodes in sample_setsystem.items() if "node1" in nodes)
        assert node1_degree == expected_degree
        
        # Test edge size
        edge1_size = hg.size_of_edge("edge1")
        assert edge1_size == len(sample_setsystem["edge1"])
        
        # Test neighbors
        neighbors = hg.neighbors("node1")
        assert len(neighbors) > 0
    
    def test_hypergraph_statistics(self, sample_setsystem):
        """Test hypergraph statistics computation"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(setsystem=sample_setsystem)
        stats = hg.get_statistics()
        
        # Check required statistics
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "num_incidences" in stats
        assert "avg_edge_size" in stats
        assert "avg_node_degree" in stats
        
        # Validate statistics
        assert stats["num_edges"] == len(sample_setsystem)
        assert stats["num_nodes"] == len(set().union(*sample_setsystem.values()))
        
        expected_incidences = sum(len(nodes) for nodes in sample_setsystem.values())
        assert stats["num_incidences"] == expected_incidences
    
    def test_dataframe_conversion(self, sample_setsystem, sample_node_properties):
        """Test DataFrame conversion functionality"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties
        )
        
        # Test incidence DataFrame
        incidence_df = hg.to_dataframe("incidences")
        assert len(incidence_df) == sum(len(nodes) for nodes in sample_setsystem.values())
        assert "edge_id" in incidence_df.columns
        assert "node_id" in incidence_df.columns
        
        # Test node properties DataFrame
        node_props_df = hg.to_dataframe("node_properties")
        assert len(node_props_df) == len(sample_node_properties)
        assert "uid" in node_props_df.columns


class TestSetSystemFactory:
    """Test suite for SetSystemFactory"""
    
    def test_dict_input(self, sample_setsystem):
        """Test dictionary input processing"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        result = SetSystemFactory.from_dict(sample_setsystem)
        assert result == sample_setsystem
    
    def test_iterable_input(self):
        """Test iterable input processing"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Test list of sets
        edges = [
            {"node1", "node2", "node3"},
            {"node2", "node3", "node4"},
            {"node1", "node4", "node5"}
        ]
        
        result = SetSystemFactory.from_iterable(edges)
        assert len(result) == 3
        assert all(f"edge_{i}" in result for i in range(3))
    
    def test_dataframe_input(self, sample_setsystem):
        """Test DataFrame input processing"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Create DataFrame from setsystem
        incidence_data = []
        for edge_id, nodes in sample_setsystem.items():
            for node in nodes:
                incidence_data.append({"edge_id": edge_id, "node_id": node})
        
        df = pl.DataFrame(incidence_data)
        result = SetSystemFactory.from_dataframe(df)
        
        # Verify reconstruction
        assert len(result) == len(sample_setsystem)
        for edge_id, nodes in result.items():
            assert set(nodes) == set(sample_setsystem[edge_id])
    
    def test_numpy_input(self):
        """Test NumPy array input processing"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Binary incidence matrix
        incidence_matrix = np.array([
            [1, 1, 1, 0, 0, 0],  # edge0: nodes 0,1,2
            [0, 1, 1, 1, 0, 0],  # edge1: nodes 1,2,3
            [1, 0, 0, 1, 1, 0],  # edge2: nodes 0,3,4
        ])
        
        result = SetSystemFactory.from_numpy(incidence_matrix)
        
        assert len(result) == 3
        assert set(result["edge_0"]) == {"node_0", "node_1", "node_2"}
        assert set(result["edge_1"]) == {"node_1", "node_2", "node_3"}
        assert set(result["edge_2"]) == {"node_0", "node_3", "node_4"}
    
    def test_auto_detection(self, sample_setsystem):
        """Test automatic input format detection"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Test dict input
        result_dict = SetSystemFactory.create_setsystem(sample_setsystem)
        assert result_dict == sample_setsystem
        
        # Test list input
        edges_list = [["node1", "node2"], ["node2", "node3"]]
        result_list = SetSystemFactory.create_setsystem(edges_list)
        assert len(result_list) == 2


class TestAnantIO:
    """Test suite for I/O operations"""
    
    def test_parquet_save_load(self, sample_setsystem, sample_node_properties, sample_edge_properties, temp_dir):
        """Test parquet save and load operations"""
        from anant.classes.hypergraph import Hypergraph
        from anant.io.parquet_io import AnantIO
        
        # Create hypergraph
        hg = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties,
            edge_properties=sample_edge_properties
        )
        
        # Save to parquet
        save_path = temp_dir / "test_hypergraph"
        AnantIO.save_hypergraph_parquet(hg, save_path)
        
        # Verify files exist
        assert (save_path / "incidences.parquet").exists()
        assert (save_path / "node_properties.parquet").exists()
        assert (save_path / "edge_properties.parquet").exists()
        assert (save_path / "metadata.json").exists()
        
        # Load from parquet
        hg_loaded = AnantIO.load_hypergraph_parquet(save_path)
        
        # Verify loaded hypergraph
        assert hg_loaded.num_nodes == hg.num_nodes
        assert hg_loaded.num_edges == hg.num_edges
        assert hg_loaded.num_incidences == hg.num_incidences
        
        # Verify properties preserved
        for node in sample_node_properties:
            assert hg_loaded.get_node_properties(node) == sample_node_properties[node]
        
        for edge in sample_edge_properties:
            assert hg_loaded.get_edge_properties(edge) == sample_edge_properties[edge]
    
    def test_compression_options(self, sample_setsystem, temp_dir):
        """Test different compression options"""
        from anant.classes.hypergraph import Hypergraph
        from anant.io.parquet_io import AnantIO
        
        hg = Hypergraph(setsystem=sample_setsystem)
        
        # Test different compression algorithms
        for compression in ["snappy", "gzip", "lz4"]:
            save_path = temp_dir / f"test_{compression}"
            AnantIO.save_hypergraph_parquet(hg, save_path, compression=compression)
            
            # Verify can load
            hg_loaded = AnantIO.load_hypergraph_parquet(save_path)
            assert hg_loaded.num_nodes == hg.num_nodes
    
    def test_dataset_operations(self, sample_setsystem, temp_dir):
        """Test dataset collection operations"""
        from anant.classes.hypergraph import Hypergraph
        from anant.io.parquet_io import AnantIO
        
        # Create multiple hypergraphs
        hgs = {}
        for i in range(3):
            hg = Hypergraph(setsystem=sample_setsystem)
            hgs[f"graph_{i}"] = hg
        
        # Save as dataset
        dataset_path = temp_dir / "dataset"
        AnantIO.save_dataset(hgs, dataset_path)
        
        # Load dataset
        loaded_hgs = AnantIO.load_dataset(dataset_path)
        
        assert len(loaded_hgs) == 3
        for name in hgs:
            assert name in loaded_hgs
            assert loaded_hgs[name].num_nodes == hgs[name].num_nodes


class TestIntegration:
    """Integration tests for the complete anant library"""
    
    def test_end_to_end_workflow(self, sample_setsystem, sample_node_properties, temp_dir):
        """Test complete end-to-end workflow"""
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        
        # 1. Create hypergraph using factory
        processed_setsystem = SetSystemFactory.from_dict(sample_setsystem)
        hg = Hypergraph(
            setsystem=processed_setsystem,
            node_properties=sample_node_properties
        )
        
        # 2. Perform analysis
        stats = hg.get_statistics()
        assert stats["num_edges"] > 0
        assert stats["num_nodes"] > 0
        
        # 3. Add more properties
        additional_props = pl.DataFrame([
            {"uid": "node1", "category": "important"},
            {"uid": "node2", "category": "standard"}
        ])
        hg.add_node_properties(additional_props)
        
        # 4. Save to parquet
        save_path = temp_dir / "workflow_test"
        AnantIO.save_hypergraph_parquet(hg, save_path)
        
        # 5. Load and verify
        hg_loaded = AnantIO.load_hypergraph_parquet(save_path)
        assert hg_loaded.get_node_properties("node1")["category"] == "important"
        
        # 6. Convert to DataFrame for analysis
        df = hg_loaded.to_dataframe("incidences")
        assert len(df) > 0
    
    def test_large_scale_operations(self):
        """Test operations on larger datasets"""
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Generate larger setsystem
        large_setsystem = {}
        for i in range(1000):
            edge_id = f"edge_{i}"
            # Random edge with 3-10 nodes
            import random
            num_nodes = random.randint(3, 10)
            nodes = [f"node_{random.randint(0, 500)}" for _ in range(num_nodes)]
            large_setsystem[edge_id] = list(set(nodes))
        
        # Create hypergraph
        hg = Hypergraph(setsystem=large_setsystem)
        
        # Test performance of basic operations
        stats = hg.get_statistics()
        assert stats["num_edges"] == 1000
        
        # Test neighbor computation
        sample_nodes = list(hg.nodes)[:10]
        for node in sample_nodes:
            neighbors = hg.neighbors(node)
            # Should complete without errors
            assert isinstance(neighbors, list)
    
    def test_polars_backend_performance(self, sample_setsystem, sample_node_properties):
        """Test Polars backend performance characteristics"""
        from anant.classes.hypergraph import Hypergraph
        import time
        
        # Create hypergraph
        hg = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties
        )
        
        # Test bulk property operations (should be fast with Polars)
        start_time = time.perf_counter()
        
        # Add many properties
        bulk_props = []
        for i in range(1000):
            bulk_props.append({
                "uid": f"bulk_node_{i}",
                "value": i,
                "category": f"cat_{i % 10}"
            })
        
        bulk_df = pl.DataFrame(bulk_props)
        hg.add_node_properties(bulk_df)
        
        end_time = time.perf_counter()
        bulk_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert bulk_time < 1.0
        
        # Verify properties added
        assert len(hg.get_node_properties("bulk_node_0")) > 0
        assert hg.get_node_properties("bulk_node_999")["value"] == 999


# Test runners and utilities
def run_unit_tests():
    """Run all unit tests"""
    import pytest
    return pytest.main([__file__ + "::TestPropertyStore", "-v"])

def run_integration_tests():
    """Run integration tests"""
    import pytest
    return pytest.main([__file__ + "::TestIntegration", "-v"])

def run_all_tests():
    """Run complete test suite"""
    import pytest
    return pytest.main([__file__, "-v"])

def validate_installation():
    """Quick validation that anant is properly installed"""
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.classes.property_store import PropertyStore
        from anant.classes.incidence_store import IncidenceStore
        from anant.factory.setsystem_factory import SetSystemFactory
        from anant.io.parquet_io import AnantIO
        
        print("✅ All anant modules imported successfully")
        
        # Quick functionality test
        hg = Hypergraph(setsystem={"edge1": ["node1", "node2"]})
        stats = hg.get_statistics()
        
        print(f"✅ Basic functionality test passed")
        print(f"   Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation validation failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation when executed directly
    validate_installation()