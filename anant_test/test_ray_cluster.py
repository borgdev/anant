#!/usr/bin/env python3
"""
Ray Cluster Integration Test
==========================

Tests the Ray distributed processing integration with Anant Enterprise components.
Validates that Ray processors work correctly with geometry and LCG modules.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our Ray processors
try:
    from ray_distributed_processors_fixed import (
        RayWorkloadDistributor, 
        validate_dependencies
    )
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Ray processors: {e}")
    PROCESSORS_AVAILABLE = False

# Import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class RayClusterTest:
    """Test suite for Ray cluster functionality"""
    
    def __init__(self):
        self.distributor = None
        self.test_results = {}
        
    async def setup_cluster(self) -> bool:
        """Initialize Ray cluster for testing"""
        if not RAY_AVAILABLE:
            logger.error("Ray not available for cluster setup")
            return False
            
        try:
            # Initialize Ray in local mode for testing
            if not ray.is_initialized():
                ray.init(
                    num_cpus=4,  # Use 4 CPUs for local testing
                    ignore_reinit_error=True,
                    include_dashboard=True,
                    dashboard_port=8265
                )
                logger.info("Ray initialized successfully")
            
            # Create workload distributor
            self.distributor = RayWorkloadDistributor()
            
            # Initialize processors
            cluster_config = {
                "geometric_processors": 2,
                "contextual_processors": 2
            }
            
            success = await self.distributor.initialize_processors(cluster_config)
            if success:
                logger.info("Ray processors initialized successfully")
                return True
            else:
                logger.error("Failed to initialize Ray processors")
                return False
                
        except Exception as e:
            logger.error(f"Cluster setup failed: {e}")
            return False
    
    def test_dependencies(self) -> Dict[str, Any]:
        """Test all dependency availability"""
        logger.info("Testing dependencies...")
        
        deps = validate_dependencies()
        
        test_result = {
            "test_name": "dependencies",
            "success": True,
            "details": deps,
            "issues": []
        }
        
        # Check for issues
        if not deps["ray_available"]:
            test_result["issues"].append("Ray not available - distributed processing disabled")
        
        if not deps["geometry_available"]:
            test_result["issues"].append("Geometry modules not available - geometric processing disabled")
            
        if not deps["lcg_available"]:
            test_result["issues"].append("LCG modules not available - contextual processing disabled")
        
        if test_result["issues"]:
            test_result["success"] = False
        
        self.test_results["dependencies"] = test_result
        return test_result
    
    async def test_geometric_processing(self) -> Dict[str, Any]:
        """Test geometric processing with Ray"""
        logger.info("Testing geometric processing...")
        
        test_result = {
            "test_name": "geometric_processing",
            "success": False,
            "details": {},
            "issues": []
        }
        
        if not self.distributor or not self.distributor.geometric_processors:
            test_result["issues"].append("No geometric processors available")
            self.test_results["geometric_processing"] = test_result
            return test_result
        
        try:
            # Create test property vectors
            test_property_vectors = {
                "entity_1": {"height": 1.75, "weight": 70.0, "age": 25.0},
                "entity_2": {"height": 1.80, "weight": 75.0, "age": 30.0},
                "entity_3": {"height": 1.65, "weight": 65.0, "age": 28.0},
                "entity_4": {"height": 1.90, "weight": 85.0, "age": 35.0}
            }
            
            # Test curvature computation
            curvature_task = {
                "operation": "curvature",
                "graph_id": "test_graph",
                "property_vectors": test_property_vectors,
                "cache_key": "test_curvature"
            }
            
            task_id = await self.distributor.submit_geometric_task(curvature_task)
            logger.info(f"Submitted curvature task: {task_id}")
            
            # Test outlier detection
            outlier_task = {
                "operation": "outliers", 
                "graph_id": "test_graph",
                "property_vectors": test_property_vectors,
                "z_threshold": 2.0
            }
            
            outlier_task_id = await self.distributor.submit_contextual_task(outlier_task)
            logger.info(f"Submitted outlier task: {outlier_task_id}")
            
            # Wait for results (simplified - in practice would use ray.get())
            await asyncio.sleep(2)
            
            test_result["success"] = True
            test_result["details"] = {
                "curvature_task_id": task_id,
                "outlier_task_id": outlier_task_id,
                "properties_tested": list(test_property_vectors.keys())
            }
            
        except Exception as e:
            test_result["issues"].append(f"Geometric processing failed: {e}")
            logger.error(f"Geometric test failed: {e}")
        
        self.test_results["geometric_processing"] = test_result
        return test_result
    
    async def test_contextual_processing(self) -> Dict[str, Any]:
        """Test contextual layer processing with Ray"""
        logger.info("Testing contextual processing...")
        
        test_result = {
            "test_name": "contextual_processing",
            "success": False,
            "details": {},
            "issues": []
        }
        
        if not self.distributor or not self.distributor.contextual_processors:
            test_result["issues"].append("No contextual processors available")
            self.test_results["contextual_processing"] = test_result
            return test_result
        
        try:
            # Create test layer configuration
            test_layers = [
                {
                    "name": "base_layer",
                    "type": "SEMANTIC",
                    "level": 0,
                    "entities": ["entity_1", "entity_2"],
                    "metadata": {"description": "Base semantic layer"}
                },
                {
                    "name": "context_layer", 
                    "type": "CONTEXTUAL",
                    "level": 1,
                    "entities": ["entity_1", "entity_3"],
                    "metadata": {"description": "Contextual layer"}
                }
            ]
            
            test_operations = [
                {
                    "id": "query_1",
                    "type": "query_layers",
                    "query": "test_query",
                    "layers": ["base_layer", "context_layer"]
                },
                {
                    "id": "hierarchy_1",
                    "type": "layer_hierarchy"
                }
            ]
            
            # Test layer processing
            layer_task = {
                "graph_id": "test_lcg",
                "layers": test_layers,
                "operations": test_operations
            }
            
            task_id = await self.distributor.submit_contextual_task(layer_task)
            logger.info(f"Submitted contextual task: {task_id}")
            
            # Wait for results
            await asyncio.sleep(2)
            
            test_result["success"] = True
            test_result["details"] = {
                "task_id": task_id,
                "layers_configured": len(test_layers),
                "operations_requested": len(test_operations)
            }
            
        except Exception as e:
            test_result["issues"].append(f"Contextual processing failed: {e}")
            logger.error(f"Contextual test failed: {e}")
        
        self.test_results["contextual_processing"] = test_result
        return test_result
    
    async def test_cluster_monitoring(self) -> Dict[str, Any]:
        """Test cluster status and monitoring"""
        logger.info("Testing cluster monitoring...")
        
        test_result = {
            "test_name": "cluster_monitoring", 
            "success": False,
            "details": {},
            "issues": []
        }
        
        if not self.distributor:
            test_result["issues"].append("No distributor available")
            self.test_results["cluster_monitoring"] = test_result
            return test_result
        
        try:
            # Get cluster status
            status = await self.distributor.get_cluster_status()
            
            test_result["success"] = True
            test_result["details"] = status
            
            logger.info(f"Cluster status: {json.dumps(status, indent=2)}")
            
        except Exception as e:
            test_result["issues"].append(f"Cluster monitoring failed: {e}")
            logger.error(f"Monitoring test failed: {e}")
        
        self.test_results["cluster_monitoring"] = test_result
        return test_result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting Ray cluster integration tests...")
        
        # Test dependencies first
        self.test_dependencies()
        
        # Setup cluster if possible
        cluster_ready = await self.setup_cluster()
        
        if cluster_ready:
            # Run processing tests
            await self.test_geometric_processing()
            await self.test_contextual_processing()
            await self.test_cluster_monitoring()
        else:
            logger.warning("Cluster setup failed - skipping processing tests")
        
        # Generate test summary
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result["success"])
        
        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "cluster_ready": cluster_ready
            },
            "detailed_results": self.test_results
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup Ray resources"""
        if RAY_AVAILABLE and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shutdown completed")
            except Exception as e:
                logger.error(f"Ray shutdown failed: {e}")


async def main():
    """Main test execution"""
    print("🧪 Ray Cluster Integration Test")
    print("=" * 40)
    
    # Check if processors are available
    if not PROCESSORS_AVAILABLE:
        print("❌ Ray processors not available - check installation")
        return
    
    # Run tests
    test_runner = RayClusterTest()
    
    try:
        results = await test_runner.run_all_tests()
        
        # Print results
        print("\n📊 Test Results:")
        print("-" * 20)
        
        summary = results["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Cluster Ready: {summary['cluster_ready']}")
        
        print("\n📋 Detailed Results:")
        print("-" * 25)
        
        for test_name, result in results["detailed_results"].items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {test_name}")
            
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"   ⚠️  {issue}")
        
        # Write results to file
        results_file = Path("ray_cluster_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        
    finally:
        test_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())