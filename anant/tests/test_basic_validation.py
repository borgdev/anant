"""
Basic validation tests for anant library

This module provides essential tests to validate the anant library components
work correctly with the actual API.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional


# Test fixtures
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
    
    def test_property_store_creation(self):
        """Test PropertyStore creation"""
        from anant.classes.property_store import PropertyStore
        
        # Test creation with minimal parameters
        store = PropertyStore(level=1)  # Node properties
        assert store is not None
        assert len(store) == 0
    
    def test_property_operations(self, sample_node_properties):
        """Test basic property operations"""
        from anant.classes.property_store import PropertyStore
        
        store = PropertyStore(level=1)
        
        # Test bulk property setting
        props_data = [{"uid": k, **v} for k, v in sample_node_properties.items()]
        props_df = pl.DataFrame(props_data)
        store.bulk_set_properties(props_df)
        
        # Test retrieval
        node1_props = store.get_properties("node1")
        assert "type" in node1_props
        assert node1_props["type"] == "A"
        assert node1_props["value"] == 10.5
        
        # Test property summary
        summary = store.get_property_summary()
        assert "total_entities" in summary
        assert summary["total_entities"] == len(sample_node_properties)


class TestIncidenceStore:
    """Test suite for IncidenceStore class"""
    
    def test_incidence_store_creation(self, sample_setsystem):
        """Test IncidenceStore creation and basic operations"""
        from anant.classes.incidence_store import IncidenceStore
        
        # Create incidence data
        incidence_data = []
        for edge_id, nodes in sample_setsystem.items():
            for node in nodes:
                incidence_data.append({"edges": edge_id, "nodes": node})
        
        incidence_df = pl.DataFrame(incidence_data)
        store = IncidenceStore(incidence_df)
        
        assert store is not None
        
        # Test basic statistics
        stats = store.get_statistics()
        assert "num_edges" in stats
        assert "num_nodes" in stats
        assert "num_incidences" in stats


class TestHypergraph:
    """Test suite for main Hypergraph class"""
    
    def test_hypergraph_creation(self, sample_setsystem):
        """Test Hypergraph creation"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(setsystem=sample_setsystem)
        assert hg is not None
        assert hg.num_edges == len(sample_setsystem)
        
        # Verify all edges are present
        for edge_id in sample_setsystem:
            assert edge_id in hg.edges
    
    def test_hypergraph_with_properties(self, sample_setsystem, sample_node_properties, sample_edge_properties):
        """Test Hypergraph with properties"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties,
            edge_properties=sample_edge_properties
        )
        
        assert hg.num_edges == len(sample_setsystem)
        assert hg.num_nodes == len(set().union(*sample_setsystem.values()))
        
        # Test property access
        node1_props = hg.get_node_properties("node1")
        assert "type" in node1_props
        assert node1_props["type"] == "A"
        
        edge1_props = hg.get_edge_properties("edge1")
        assert "weight" in edge1_props
        assert edge1_props["weight"] == 1.0
    
    def test_hypergraph_operations(self, sample_setsystem):
        """Test basic hypergraph operations"""
        from anant.classes.hypergraph import Hypergraph
        
        hg = Hypergraph(setsystem=sample_setsystem)
        
        # Test degree computation
        node1_degree = hg.degree("node1")
        # node1 is in edge1 and edge3
        assert node1_degree == 2
        
        # Test edge size
        edge1_size = hg.size_of_edge("edge1")
        assert edge1_size == len(sample_setsystem["edge1"])
        
        # Test statistics
        stats = hg.get_statistics()
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "avg_edge_size" in stats
        assert "avg_node_degree" in stats


class TestSetSystemFactory:
    """Test suite for SetSystemFactory"""
    
    def test_dict_input(self, sample_setsystem):
        """Test dictionary input processing"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        result = SetSystemFactory.from_dict(sample_setsystem)
        assert result == sample_setsystem
    
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
    
    def test_auto_detection(self, sample_setsystem):
        """Test automatic input format detection"""
        from anant.factory.setsystem_factory import SetSystemFactory
        
        # Test dict input
        result_dict = SetSystemFactory.create_setsystem(sample_setsystem)
        assert result_dict == sample_setsystem


class TestAnantIO:
    """Test suite for I/O operations"""
    
    def test_parquet_save_load(self, sample_setsystem, sample_node_properties, temp_dir):
        """Test basic parquet save and load operations"""
        from anant.classes.hypergraph import Hypergraph
        from anant.io.parquet_io import AnantIO
        
        # Create hypergraph
        hg = Hypergraph(
            setsystem=sample_setsystem,
            node_properties=sample_node_properties
        )
        
        # Save to parquet
        save_path = temp_dir / "test_hypergraph"
        AnantIO.save_hypergraph_parquet(hg, save_path)
        
        # Verify files exist
        assert (save_path / "incidences.parquet").exists()
        assert (save_path / "metadata.json").exists()
        
        # Load from parquet
        hg_loaded = AnantIO.load_hypergraph_parquet(save_path)
        
        # Verify loaded hypergraph
        assert hg_loaded.num_nodes == hg.num_nodes
        assert hg_loaded.num_edges == hg.num_edges


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
        
        # 3. Save to parquet
        save_path = temp_dir / "workflow_test"
        AnantIO.save_hypergraph_parquet(hg, save_path)
        
        # 4. Load and verify
        hg_loaded = AnantIO.load_hypergraph_parquet(save_path)
        
        # 5. Convert to DataFrame for analysis
        df = hg_loaded.to_dataframe("incidences")
        assert len(df) > 0
        assert "edge_id" in df.columns
        assert "node_id" in df.columns
    
    def test_performance_characteristics(self, sample_setsystem):
        """Test basic performance characteristics"""
        from anant.classes.hypergraph import Hypergraph
        import time
        
        # Create hypergraph
        hg = Hypergraph(setsystem=sample_setsystem)
        
        # Test that basic operations complete quickly
        start_time = time.perf_counter()
        
        stats = hg.get_statistics()
        
        # Test neighbor computation
        sample_nodes = list(hg.nodes)[:3]
        for node in sample_nodes:
            neighbors = hg.neighbors(node)
            assert isinstance(neighbors, list)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for small dataset)
        assert execution_time < 1.0


# Utility functions
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


def run_basic_tests():
    """Run basic test suite"""
    import pytest
    return pytest.main([__file__, "-v", "-x"])


if __name__ == "__main__":
    # Run validation when executed directly
    if validate_installation():
        print("\n🧪 Running basic tests...")
        run_basic_tests()
    else:
        print("❌ Installation validation failed, skipping tests")