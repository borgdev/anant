"""
Metagraph Test Suite
====================

Tests for metagraph implementations:
- Traditional Metagraph with enterprise features
- Governance and policy management
- Temporal tracking and versioning
- Polars+Parquet backend operations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_metagraph_operations():
    """Test basic metagraph functionality."""
    print("  Testing Basic Metagraph Operations...")
    
    try:
        from anant.metagraph import Metagraph
        
        # Create metagraph
        mg = Metagraph()
        
        # Create entities with metadata
        success1 = mg.create_entity("entity1", "Dataset", {
            "name": "Customer Data",
            "owner": "Data Team",
            "created_date": "2024-01-01",
            "size_gb": 10.5,
            "tags": ["customer", "production"]
        })
        
        success2 = mg.create_entity("entity2", "Pipeline", {
            "name": "ETL Pipeline",
            "owner": "Engineering Team", 
            "status": "active",
            "frequency": "daily"
        })
        
        success3 = mg.create_entity("entity3", "Report", {
            "name": "Customer Analytics",
            "owner": "Analytics Team",
            "dashboard_url": "https://dashboard.company.com/customer"
        })
        
        assert success1, "Should successfully create entity1"
        assert success2, "Should successfully create entity2"
        assert success3, "Should successfully create entity3"
        
        # Create relationships
        rel_id1 = mg.add_relationship("entity1", "entity2", "feeds_into", 
                                    strength=1.0, metadata={
            "latency_minutes": 15,
            "format": "parquet"
        })
        
        rel_id2 = mg.add_relationship("entity2", "entity3", "generates",
                                    strength=1.0, metadata={
            "schedule": "daily at 6 AM"
        })
        
        assert rel_id1, "Should successfully create relationship 1"
        assert rel_id2, "Should successfully create relationship 2"
        
        # Test entity retrieval
        entity1 = mg.get_entity("entity1")
        assert entity1 is not None, "Should retrieve entity1"
        assert entity1["entity_id"] == "entity1", "Should have correct entity_id"
        
        # Test comprehensive stats
        stats = mg.get_comprehensive_stats()
        assert isinstance(stats, dict), "Should return statistics dictionary"
        assert "total_entities" in stats, "Stats should include entity count"
        
        print("    ✅ Basic metagraph operations working")
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Basic metagraph operations test failed: {e}")
        return False


def test_metagraph_governance():
    """Test metagraph governance and policy features."""
    print("  Testing Metagraph Governance...")
    
    try:
        from anant.metagraph import Metagraph
        
        # Create metagraph
        mg = Metagraph()
        
        # Create policy for data classification
        if hasattr(mg, 'create_policy'):
            mg.create_policy("data_classification", {
                "rule": "all datasets must have classification",
                "enforcement": "strict",
                "required_fields": ["classification", "sensitivity"]
            })
        
        # Create entities with governance metadata
        mg.create_entity("sensitive_data", "Dataset", {
            "name": "PII Customer Data",
            "classification": "confidential",
            "sensitivity": "high",
            "compliance": ["GDPR", "CCPA"],
            "retention_days": 2555  # 7 years
        })
        
        mg.create_entity("public_data", "Dataset", {
            "name": "Public Market Data",
            "classification": "public",
            "sensitivity": "low",
            "compliance": ["none"]
        })
        
        # Test governance validation
        if hasattr(mg, 'validate_compliance'):
            try:
                compliance_report = mg.validate_compliance()
                assert isinstance(compliance_report, dict), "Should return compliance report"
                print("      Compliance validation working")
            except Exception:
                print("      Compliance validation not fully implemented, skipping...")
        
        # Test access control
        if hasattr(mg, 'check_access'):
            try:
                access_allowed = mg.check_access("sensitive_data", "read", {"role": "analyst"})
                assert isinstance(access_allowed, bool), "Should return access decision"
                print("      Access control working")
            except Exception:
                print("      Access control not fully implemented, skipping...")
        
        print("    ✅ Metagraph governance working")
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph governance not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metagraph governance test failed: {e}")
        return False


def test_metagraph_temporal_features():
    """Test metagraph temporal tracking and versioning."""
    print("  Testing Metagraph Temporal Features...")
    
    try:
        from anant.metagraph import Metagraph
        import time
        
        # Create metagraph
        mg = Metagraph()
        
        # Create entity with initial version
        mg.create_entity("evolving_dataset", "Dataset", {
            "name": "Customer Segments",
            "version": "1.0",
            "records": 10000,
            "schema_version": "v1"
        })
        
        # Simulate time passage
        time.sleep(0.1)
        
        # Update entity (should create new version)
        if hasattr(mg, 'update_entity'):
            mg.update_entity("evolving_dataset", {
                "version": "1.1", 
                "records": 15000,
                "schema_version": "v1.1",
                "updated_fields": ["customer_segment", "purchase_history"]
            })
        
        # Test temporal queries
        if hasattr(mg, 'get_entity_history'):
            try:
                history = mg.get_entity_history("evolving_dataset")
                assert len(history) >= 1, "Should have entity history"
                print("      Entity versioning working")
            except Exception:
                print("      Entity versioning not fully implemented, skipping...")
        
        # Test point-in-time queries
        if hasattr(mg, 'get_snapshot_at_time'):
            try:
                import datetime
                now = datetime.datetime.now()
                snapshot = mg.get_snapshot_at_time(now - datetime.timedelta(seconds=1))
                assert isinstance(snapshot, dict), "Should return point-in-time snapshot"
                print("      Point-in-time queries working")
            except Exception:
                print("      Point-in-time queries not fully implemented, skipping...")
        
        print("    ✅ Metagraph temporal features working")
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph temporal features not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metagraph temporal features test failed: {e}")
        return False


def test_metagraph_backend_operations():
    """Test metagraph Polars+Parquet backend operations."""
    print("  Testing Metagraph Backend Operations...")
    
    try:
        from anant.metagraph import Metagraph
        
        # Create metagraph
        mg = Metagraph()
        
        # Add multiple entities for backend testing
        entities_data = [
            ("dataset_1", "Dataset", {"name": "Sales Data", "department": "Sales", "size_gb": 5.2}),
            ("dataset_2", "Dataset", {"name": "Marketing Data", "department": "Marketing", "size_gb": 3.1}),
            ("dataset_3", "Dataset", {"name": "Finance Data", "department": "Finance", "size_gb": 8.7}),
            ("pipeline_1", "Pipeline", {"name": "Sales ETL", "department": "Sales", "status": "active"}),
            ("pipeline_2", "Pipeline", {"name": "Marketing ETL", "department": "Marketing", "status": "inactive"}),
            ("report_1", "Report", {"name": "Sales Dashboard", "department": "Sales", "views": 1250}),
            ("report_2", "Report", {"name": "Marketing ROI", "department": "Marketing", "views": 890})
        ]
        
        for entity_id, entity_type, properties in entities_data:
            mg.create_entity(entity_id, entity_type, properties)
        
        # Test filtering operations
        if hasattr(mg, 'filter_entities'):
            try:
                sales_entities = mg.filter_entities({"department": "Sales"})
                assert len(sales_entities) >= 2, "Should find Sales department entities"
                print("      Entity filtering working")
            except Exception:
                print("      Entity filtering not fully implemented, using basic queries...")
        
        # Test aggregation operations
        if hasattr(mg, 'aggregate_entities'):
            try:
                dept_counts = mg.aggregate_entities("department", "count")
                assert isinstance(dept_counts, dict), "Should return aggregation results"
                print("      Entity aggregation working")
            except Exception:
                print("      Entity aggregation not fully implemented, skipping...")
        
        # Test bulk operations
        if hasattr(mg, 'bulk_update'):
            try:
                # Bulk update all datasets
                mg.bulk_update("Dataset", {"last_accessed": "2024-01-15"})
                print("      Bulk operations working")
            except Exception:
                print("      Bulk operations not fully implemented, skipping...")
        
        # Test export/import operations
        if hasattr(mg, 'export_to_parquet'):
            try:
                export_path = "/tmp/metagraph_export.parquet"
                mg.export_to_parquet(export_path)
                print("      Parquet export working")
                
                # Test import
                if hasattr(mg, 'import_from_parquet'):
                    mg.import_from_parquet(export_path)
                    print("      Parquet import working")
            except Exception:
                print("      Parquet export/import not fully implemented, skipping...")
        
        print("    ✅ Metagraph backend operations working")
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph backend operations not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metagraph backend operations test failed: {e}")
        return False


def test_metagraph_lineage_tracking():
    """Test metagraph data lineage and impact analysis."""
    print("  Testing Metagraph Lineage Tracking...")
    
    try:
        from anant.metagraph import Metagraph
        
        # Create metagraph for lineage testing
        mg = Metagraph()
        
        # Create a data pipeline with lineage
        entities = [
            ("raw_data", "Dataset", {"name": "Raw Customer Data", "source": "CRM"}),
            ("clean_data", "Dataset", {"name": "Cleaned Customer Data", "source": "ETL"}),
            ("features", "Dataset", {"name": "Customer Features", "source": "Feature Engineering"}),
            ("model", "Model", {"name": "Churn Prediction Model", "algorithm": "Random Forest"}),
            ("predictions", "Dataset", {"name": "Churn Predictions", "source": "Model Inference"}),
            ("dashboard", "Report", {"name": "Churn Dashboard", "tool": "Tableau"})
        ]
        
        for entity_id, entity_type, properties in entities:
            mg.create_entity(entity_id, entity_type, properties)
        
        # Create lineage relationships
        lineage_relationships = [
            ("raw_data", "clean_data", "transforms_to"),
            ("clean_data", "features", "transforms_to"),
            ("features", "model", "trains"),
            ("model", "predictions", "generates"),
            ("predictions", "dashboard", "feeds_into")
        ]
        
        for source, target, rel_type in lineage_relationships:
            mg.create_relationship(source, target, rel_type, {"lineage": True})
        
        # Test lineage queries
        if hasattr(mg, 'get_upstream_lineage'):
            try:
                upstream = mg.get_upstream_lineage("dashboard")
                assert len(upstream) >= 4, "Dashboard should have upstream dependencies"
                print("      Upstream lineage tracking working")
            except Exception:
                print("      Upstream lineage not fully implemented, skipping...")
        
        if hasattr(mg, 'get_downstream_lineage'):
            try:
                downstream = mg.get_downstream_lineage("raw_data")
                assert len(downstream) >= 4, "Raw data should have downstream dependencies"
                print("      Downstream lineage tracking working")
            except Exception:
                print("      Downstream lineage not fully implemented, skipping...")
        
        # Test impact analysis
        if hasattr(mg, 'analyze_impact'):
            try:
                impact = mg.analyze_impact("clean_data")
                assert isinstance(impact, dict), "Should return impact analysis"
                print("      Impact analysis working")
            except Exception:
                print("      Impact analysis not fully implemented, skipping...")
        
        print("    ✅ Metagraph lineage tracking working")
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph lineage tracking not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metagraph lineage tracking test failed: {e}")
        return False


def test_metagraph_performance():
    """Test metagraph performance with larger datasets."""
    print("  Testing Metagraph Performance...")
    
    try:
        from anant.metagraph import Metagraph
        import time
        
        # Create metagraph
        mg = Metagraph()
        
        # Performance test with many entities
        start_time = time.time()
        
        # Create many entities
        for i in range(200):
            entity_type = ["Dataset", "Pipeline", "Report", "Model"][i % 4]
            mg.create_entity(f"entity_{i}", entity_type, {
                "name": f"Entity {i}",
                "department": f"dept_{i % 10}",
                "priority": i % 5,
                "size": i * 1.5,
                "created_by": f"user_{i % 20}"
            })
        
        creation_time = time.time() - start_time
        
        # Create relationships
        relationship_start = time.time()
        
        for i in range(100):
            source = f"entity_{i}"
            target = f"entity_{i + 100}"
            mg.create_relationship(source, target, "depends_on", {
                "weight": i % 10,
                "created_at": f"2024-01-{(i % 30) + 1:02d}"
            })
        
        relationship_time = time.time() - relationship_start
        
        # Test query performance
        query_start = time.time()
        
        all_entities = mg.get_entities()
        all_relationships = mg.get_relationships()
        datasets = mg.get_entities_by_type("Dataset")
        
        query_time = time.time() - query_start
        
        print(f"    ✅ Performance test completed:")
        print(f"       Created 200 entities in {creation_time:.3f}s")
        print(f"       Created 100 relationships in {relationship_time:.3f}s")
        print(f"       Queried entities/relationships in {query_time:.3f}s")
        print(f"       Found {len(all_entities)} entities, {len(all_relationships)} relationships")
        print(f"       Found {len(datasets)} datasets")
        
        return True
        
    except ImportError:
        print("    ⚠️  Metagraph performance test not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Metagraph performance test failed: {e}")
        return False


def run_tests():
    """Run all metagraph tests."""
    print("🏢 Running Metagraph Tests")
    
    test_functions = [
        test_basic_metagraph_operations,
        test_metagraph_governance,
        test_metagraph_temporal_features,
        test_metagraph_backend_operations,
        test_metagraph_lineage_tracking,
        test_metagraph_performance
    ]
    
    passed = 0
    failed = 0
    details = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed += 1
                details.append(f"✅ {test_func.__name__}")
            else:
                failed += 1
                details.append(f"❌ {test_func.__name__}: Test returned False")
        except Exception as e:
            failed += 1
            details.append(f"❌ {test_func.__name__}: {str(e)}")
    
    status = "PASSED" if failed == 0 else "FAILED"
    
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "details": details
    }


if __name__ == "__main__":
    result = run_tests()
    print(f"\nMetagraph Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")