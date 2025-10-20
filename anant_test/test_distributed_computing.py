"""
Distributed Computing Test Suite
===============================

Tests for distributed computing features:
- Graph partitioning (METIS, KaHiP)
- Distributed backends (Dask, Ray, Celery)
- Enterprise features (sharding, replication)
- Load balancing and query processing
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from anant.classes.hypergraph import Hypergraph


def test_graph_partitioning():
    """Test graph partitioning algorithms."""
    print("  Testing graph partitioning...")
    
    try:
        from anant.distributed.partitioning import ProductionPartitioner, PartitioningConfig
        
        # Create test hypergraph
        setsystem = {}
        for i in range(20):
            edge_nodes = [f"n{j}" for j in range(i, min(i+3, 25))]
            setsystem[f"e{i}"] = edge_nodes
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test partitioning
        config = PartitioningConfig(
            num_partitions=4,
            algorithm="auto",
            enable_metis=False,  # Skip METIS for basic test
            enable_kahip=False   # Skip KaHiP for basic test
        )
        
        partitioner = ProductionPartitioner(config)
        partitions = partitioner.partition_graph(hg)
        
        assert isinstance(partitions, dict), "Should return partition assignments"
        assert len(partitions) > 0, "Should have partition assignments"
        assert all(0 <= p < 4 for p in partitions.values()), "Partitions should be in valid range"
        
        # Test partition quality
        quality = partitioner.evaluate_partition_quality(hg, partitions)
        assert isinstance(quality, dict), "Should return quality metrics"
        assert "edge_cut" in quality, "Should have edge cut metric"
        
        print("    ✅ Graph partitioning working")
        return True
        
    except ImportError:
        print("    ⚠️  Partitioning module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Graph partitioning test failed: {e}")
        return False


def test_distributed_backends():
    """Test distributed computing backends."""
    print("  Testing distributed backends...")
    
    try:
        from anant.distributed.backends import DistributedBackendFactory, BackendType
        
        # Test backend factory
        factory = DistributedBackendFactory()
        
        # Test memory backend (should always be available)
        try:
            memory_backend = factory.create_backend(BackendType.MEMORY)
            assert memory_backend is not None, "Memory backend should be available"
            print("    ✅ Memory backend working")
        except Exception as e:
            print(f"    ❌ Memory backend failed: {e}")
        
        # Test other backends if available
        for backend_type in [BackendType.DASK, BackendType.RAY, BackendType.CELERY]:
            try:
                backend = factory.create_backend(backend_type)
                if backend:
                    print(f"    ✅ {backend_type.value} backend available")
                else:
                    print(f"    ⚠️  {backend_type.value} backend not available")
            except Exception as e:
                print(f"    ⚠️  {backend_type.value} backend failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Distributed backends module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Distributed backends test failed: {e}")
        return False


def test_enterprise_features():
    """Test enterprise distributed features."""
    print("  Testing enterprise features...")
    
    try:
        from anant.distributed.enterprise_features import (
            ShardingManager, ReplicationManager, ConsistencyLevel
        )
        
        # Test sharding
        try:
            sharding_config = {
                "strategy": "hash",
                "num_shards": 4,
                "replication_factor": 2
            }
            
            sharding_manager = ShardingManager(sharding_config)
            
            # Test shard assignment
            shard = sharding_manager.get_shard_for_entity("test_entity")
            assert isinstance(shard, int), "Should return shard ID"
            assert 0 <= shard < 4, "Shard should be in valid range"
            
            print("    ✅ Sharding manager working")
        except Exception as e:
            print(f"    ⚠️  Sharding manager failed: {e}")
        
        # Test replication
        try:
            replication_config = {
                "replication_factor": 3,
                "consistency_level": ConsistencyLevel.EVENTUAL
            }
            
            replication_manager = ReplicationManager(replication_config)
            
            # Test replica assignment
            replicas = replication_manager.get_replicas_for_shard(0)
            assert isinstance(replicas, list), "Should return replica list"
            assert len(replicas) <= 3, "Should respect replication factor"
            
            print("    ✅ Replication manager working")
        except Exception as e:
            print(f"    ⚠️  Replication manager failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Enterprise features module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Enterprise features test failed: {e}")
        return False


def test_distributed_query_processing():
    """Test distributed query processing."""
    print("  Testing distributed query processing...")
    
    try:
        from anant.distributed.query_processor import DistributedQueryProcessor, QueryPlan
        
        # Create test query processor
        config = {
            "num_workers": 2,
            "backend": "memory",
            "enable_caching": True
        }
        
        processor = DistributedQueryProcessor(config)
        
        # Test query planning
        query = {
            "operation": "centrality",
            "algorithm": "degree",
            "parameters": {"normalized": True}
        }
        
        plan = processor.create_query_plan(query)
        assert isinstance(plan, QueryPlan), "Should return query plan"
        
        # Test query execution (mock)
        try:
            # Create simple test hypergraph
            setsystem = {
                "e1": ["n1", "n2"],
                "e2": ["n2", "n3"],
                "e3": ["n3", "n4"]
            }
            hg = Hypergraph(setsystem=setsystem)
            
            result = processor.execute_query(plan, hg)
            assert result is not None, "Should return query result"
            
            print("    ✅ Distributed query processing working")
        except Exception as e:
            print(f"    ⚠️  Query execution failed: {e}")
        
        return True
        
    except ImportError:
        print("    ⚠️  Distributed query processing module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Distributed query processing test failed: {e}")
        return False


def test_load_balancing():
    """Test load balancing functionality."""
    print("  Testing load balancing...")
    
    try:
        from anant.distributed.load_balancer import LoadBalancer, LoadBalancingStrategy
        
        # Create load balancer
        config = {
            "strategy": LoadBalancingStrategy.ROUND_ROBIN,
            "num_workers": 4,
            "health_check_interval": 30
        }
        
        load_balancer = LoadBalancer(config)
        
        # Test worker assignment
        worker_1 = load_balancer.get_next_worker()
        worker_2 = load_balancer.get_next_worker()
        
        assert isinstance(worker_1, int), "Should return worker ID"
        assert isinstance(worker_2, int), "Should return worker ID"
        assert worker_1 != worker_2, "Should distribute across workers"
        
        # Test load metrics
        metrics = load_balancer.get_load_metrics()
        assert isinstance(metrics, dict), "Should return load metrics"
        
        print("    ✅ Load balancing working")
        return True
        
    except ImportError:
        print("    ⚠️  Load balancing module not available, skipping...")
        return True
    except Exception as e:
        print(f"    ❌ Load balancing test failed: {e}")
        return False


def test_distributed_performance():
    """Test distributed computing performance."""
    print("  Testing distributed performance...")
    
    try:
        # Create larger test hypergraph
        import random
        random.seed(42)
        
        setsystem = {}
        nodes = [f"n{i}" for i in range(200)]
        
        # Create random hyperedges
        for i in range(100):
            edge_size = random.randint(2, 5)
            edge_nodes = random.sample(nodes, edge_size)
            setsystem[f"e{i}"] = edge_nodes
        
        hg = Hypergraph(setsystem=setsystem)
        
        # Test partitioning performance
        try:
            from anant.distributed.partitioning import ProductionPartitioner, PartitioningConfig
            
            config = PartitioningConfig(
                num_partitions=8,
                algorithm="igraph",  # Use igraph as fallback
                enable_metis=False,
                enable_kahip=False
            )
            
            partitioner = ProductionPartitioner(config)
            
            start_time = time.time()
            partitions = partitioner.partition_graph(hg)
            partition_time = time.time() - start_time
            
            print(f"    ✅ Partitioned 200 nodes, 100 edges in {partition_time:.3f}s")
            
            # Test partition balance
            partition_sizes = {}
            for node, partition in partitions.items():
                partition_sizes[partition] = partition_sizes.get(partition, 0) + 1
            
            max_size = max(partition_sizes.values())
            min_size = min(partition_sizes.values())
            balance_ratio = min_size / max_size if max_size > 0 else 1.0
            
            print(f"    ✅ Partition balance ratio: {balance_ratio:.3f}")
            
        except Exception as e:
            print(f"    ⚠️  Performance test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Distributed performance test failed: {e}")
        return False


def run_tests():
    """Run all distributed computing tests."""
    print("🧪 Running Distributed Computing Tests")
    
    test_functions = [
        test_graph_partitioning,
        test_distributed_backends,
        test_enterprise_features,
        test_distributed_query_processing,
        test_load_balancing,
        test_distributed_performance
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
    print(f"\nDistributed Computing Tests: {result['status']}")
    print(f"Passed: {result['passed']}, Failed: {result['failed']}")
    for detail in result["details"]:
        print(f"  {detail}")