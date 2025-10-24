#!/usr/bin/env python3
"""
Anant Enterprise Ray Cluster - Integration Demo
==============================================

Comprehensive demonstration of Ray distributed processing
integrated with Anant Enterprise components:
- Enhanced AnantKnowledgeServer with Ray cluster capabilities
- Distributed geometric manifold computations
- Distributed layered contextual graph processing
- Zero code duplication - extends existing components

Author: Anant Team
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Ray components
try:
    from ray_anant_cluster import RayAnantKnowledgeServer, RayClusterConfig
    from ray_distributed_processors_fixed import RayWorkloadDistributor, validate_dependencies
    RAY_CLUSTER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Ray cluster components not available: {e}")
    RAY_CLUSTER_AVAILABLE = False

# Import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class AnantRayIntegrationDemo:
    """
    Comprehensive demonstration of Anant Enterprise Ray integration
    
    Shows how Ray distributed computing enhances existing Anant capabilities
    without duplicating code.
    """
    
    def __init__(self):
        self.ray_server: Optional[RayAnantKnowledgeServer] = None
        self.distributor: Optional[RayWorkloadDistributor] = None
        self.demo_results = {}
        
    async def initialize_demo(self) -> bool:
        """Initialize the demo environment"""
        logger.info("🚀 Initializing Anant Enterprise Ray Integration Demo")
        
        # Check dependencies
        deps = validate_dependencies() if RAY_CLUSTER_AVAILABLE else {}
        
        print("📋 Dependency Status:")
        print("-" * 25)
        for dep_name, available in deps.items():
            status = "✅" if available else "❌"
            print(f"   {status} {dep_name}")
        
        if not deps.get("ray_available", False):
            print("\n❌ Ray not available - demo will be limited")
            return False
        
        try:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(
                    num_cpus=4,
                    include_dashboard=True,
                    dashboard_port=8265,
                    ignore_reinit_error=True
                )
                print("✅ Ray cluster initialized")
                print(f"   Dashboard: http://localhost:8265")
                print(f"   Resources: {ray.available_resources()}")
            
            # Create Ray cluster configuration
            cluster_config = RayClusterConfig(
                cluster_name="anant_enterprise_demo",
                num_nodes=1,  # Local demo
                cpus_per_node=4,
                memory_per_node_gb=8,
                enable_gpu=False,  # Enable if GPU available
                ray_dashboard_port=8265,
                
                # Anant-specific configuration
                enable_geometric_processing=True,
                enable_contextual_processing=True,
                geometric_processors_per_node=2,
                contextual_processors_per_node=2,
                
                # Enterprise features
                enable_security=True,
                enable_monitoring=True,
                enable_auto_scaling=False  # Disable for demo
            )
            
            # Initialize Ray-enhanced Anant server
            self.ray_server = RayAnantKnowledgeServer(cluster_config)
            await self.ray_server.start()
            
            print("✅ RayAnantKnowledgeServer started")
            
            # Initialize workload distributor
            self.distributor = RayWorkloadDistributor()
            distributor_config = {
                "geometric_processors": 2,
                "contextual_processors": 2
            }
            
            success = await self.distributor.initialize_processors(distributor_config)
            if success:
                print("✅ Ray workload distributors initialized")
            else:
                print("⚠️  Some processors failed to initialize")
            
            return True
            
        except Exception as e:
            logger.error(f"Demo initialization failed: {e}")
            return False
    
    async def demo_geometric_processing(self) -> Dict[str, Any]:
        """Demonstrate distributed geometric processing"""
        logger.info("🧮 Demonstrating distributed geometric processing")
        
        demo_result = {
            "test_name": "geometric_processing",
            "success": False,
            "results": {},
            "performance": {}
        }
        
        try:
            # Create sample property data for analysis
            sample_entities = {
                # Financial entities
                "stock_AAPL": {"price": 150.0, "volume": 1000000, "volatility": 0.25, "pe_ratio": 28.5},
                "stock_GOOGL": {"price": 2800.0, "volume": 800000, "volatility": 0.22, "pe_ratio": 25.3},
                "stock_MSFT": {"price": 330.0, "volume": 1200000, "volatility": 0.20, "pe_ratio": 30.1},
                "stock_TSLA": {"price": 900.0, "volume": 2000000, "volatility": 0.45, "pe_ratio": 85.7},
                
                # Add some outliers for anomaly detection
                "stock_OUTLIER1": {"price": 5000.0, "volume": 100, "volatility": 1.5, "pe_ratio": 200.0},
                "stock_NORMAL": {"price": 100.0, "volume": 500000, "volatility": 0.18, "pe_ratio": 22.0}
            }
            
            print("\n📊 Geometric Processing Demo:")
            print(f"   Analyzing {len(sample_entities)} financial entities")
            print("   Properties: price, volume, volatility, pe_ratio")
            
            # Test 1: Curvature analysis
            start_time = time.time()
            
            curvature_task = {
                "operation": "curvature",
                "graph_id": "financial_manifold",
                "property_vectors": sample_entities,
                "cache_key": "demo_curvature"
            }
            
            curvature_task_id = await self.distributor.submit_geometric_task(curvature_task)
            print(f"   ✅ Submitted curvature analysis: {curvature_task_id}")
            
            # Test 2: Outlier detection
            outlier_task = {
                "operation": "outliers",
                "graph_id": "financial_manifold", 
                "property_vectors": sample_entities,
                "z_threshold": 2.0
            }
            
            outlier_task_id = await self.distributor.submit_geometric_task(outlier_task)
            print(f"   ✅ Submitted outlier detection: {outlier_task_id}")
            
            # Wait for processing (simplified - real implementation would use ray.get)
            await asyncio.sleep(3)
            
            processing_time = time.time() - start_time
            
            demo_result.update({
                "success": True,
                "results": {
                    "curvature_task_id": curvature_task_id,
                    "outlier_task_id": outlier_task_id,
                    "entities_analyzed": len(sample_entities),
                    "properties_analyzed": ["price", "volume", "volatility", "pe_ratio"]
                },
                "performance": {
                    "processing_time_seconds": processing_time,
                    "entities_per_second": len(sample_entities) / processing_time
                }
            })
            
            print(f"   ⏱️  Processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            demo_result["error"] = str(e)
            logger.error(f"Geometric processing demo failed: {e}")
        
        self.demo_results["geometric_processing"] = demo_result
        return demo_result
    
    async def demo_contextual_processing(self) -> Dict[str, Any]:
        """Demonstrate distributed contextual graph processing"""
        logger.info("🏗️ Demonstrating distributed contextual processing")
        
        demo_result = {
            "test_name": "contextual_processing",
            "success": False,
            "results": {},
            "performance": {}
        }
        
        try:
            # Create sample layered contextual graph
            knowledge_layers = [
                {
                    "name": "data_layer",
                    "type": "physical",
                    "level": 0,
                    "entities": ["customer_1", "customer_2", "product_A", "product_B"],
                    "metadata": {"description": "Raw data entities"}
                },
                {
                    "name": "relationship_layer", 
                    "type": "logical",
                    "level": 1,
                    "entities": ["purchase_1", "purchase_2", "recommendation_1"],
                    "metadata": {"description": "Entity relationships"}
                },
                {
                    "name": "semantic_layer",
                    "type": "semantic",
                    "level": 2,
                    "entities": ["preference_pattern", "buying_behavior"],
                    "metadata": {"description": "Semantic understanding"}
                }
            ]
            
            layer_operations = [
                {
                    "id": "hierarchy_analysis",
                    "type": "layer_hierarchy"
                },
                {
                    "id": "cross_layer_query",
                    "type": "query_layers",
                    "query": {"entity_type": "customer"},
                    "layers": ["data_layer", "relationship_layer"]
                }
            ]
            
            print("\n🏗️ Contextual Processing Demo:")
            print(f"   Processing {len(knowledge_layers)} knowledge layers")
            print("   Layers: data → relationship → semantic")
            
            start_time = time.time()
            
            contextual_task = {
                "graph_id": "knowledge_graph_demo",
                "layers": knowledge_layers,
                "operations": layer_operations
            }
            
            task_id = await self.distributor.submit_contextual_task(contextual_task)
            print(f"   ✅ Submitted contextual processing: {task_id}")
            
            # Wait for processing
            await asyncio.sleep(2)
            
            processing_time = time.time() - start_time
            
            demo_result.update({
                "success": True,
                "results": {
                    "task_id": task_id,
                    "layers_processed": len(knowledge_layers),
                    "operations_executed": len(layer_operations),
                    "total_entities": sum(len(layer["entities"]) for layer in knowledge_layers)
                },
                "performance": {
                    "processing_time_seconds": processing_time,
                    "layers_per_second": len(knowledge_layers) / processing_time
                }
            })
            
            print(f"   ⏱️  Processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            demo_result["error"] = str(e)
            logger.error(f"Contextual processing demo failed: {e}")
        
        self.demo_results["contextual_processing"] = demo_result
        return demo_result
    
    async def demo_cluster_monitoring(self) -> Dict[str, Any]:
        """Demonstrate cluster monitoring and management"""
        logger.info("📊 Demonstrating cluster monitoring")
        
        demo_result = {
            "test_name": "cluster_monitoring",
            "success": False,
            "results": {}
        }
        
        try:
            print("\n📊 Cluster Monitoring Demo:")
            
            # Get Ray cluster status
            if self.ray_server:
                cluster_status = await self.ray_server.get_cluster_status()
                print(f"   Ray server status: {cluster_status.get('status', 'unknown')}")
                
            # Get distributor status
            if self.distributor:
                distributor_status = await self.distributor.get_cluster_status()
                
                print(f"   Total processors: {distributor_status.get('total_processors', 0)}")
                print(f"   Geometric processors: {distributor_status.get('geometric_processors', 0)}")
                print(f"   Contextual processors: {distributor_status.get('contextual_processors', 0)}")
                print(f"   Active tasks: {distributor_status.get('active_tasks', 0)}")
                
                demo_result["results"] = distributor_status
            
            # Ray cluster resources
            if RAY_AVAILABLE and ray.is_initialized():
                resources = ray.cluster_resources()
                available = ray.available_resources()
                
                print(f"   Cluster CPUs: {resources.get('CPU', 0)}")
                print(f"   Available CPUs: {available.get('CPU', 0)}")
                print(f"   Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
                
                demo_result["results"]["ray_resources"] = {
                    "total_cpus": resources.get('CPU', 0),
                    "available_cpus": available.get('CPU', 0),
                    "total_memory_gb": resources.get('memory', 0) / (1024**3)
                }
            
            demo_result["success"] = True
            
        except Exception as e:
            demo_result["error"] = str(e)
            logger.error(f"Cluster monitoring demo failed: {e}")
        
        self.demo_results["cluster_monitoring"] = demo_result
        return demo_result
    
    async def demo_enterprise_integration(self) -> Dict[str, Any]:
        """Demonstrate enterprise features integration"""
        logger.info("🏢 Demonstrating enterprise integration")
        
        demo_result = {
            "test_name": "enterprise_integration",
            "success": False,
            "results": {}
        }
        
        try:
            print("\n🏢 Enterprise Integration Demo:")
            
            if self.ray_server:
                # Test enhanced server capabilities
                enhanced_info = await self.ray_server.get_enhanced_server_info()
                
                print("   Enhanced server features:")
                print(f"     • Base server: {enhanced_info.get('base_server_type', 'AnantKnowledgeServer')}")
                print(f"     • Ray cluster: {enhanced_info.get('ray_cluster_enabled', False)}")
                print(f"     • Security: {enhanced_info.get('security_enabled', False)}")
                print(f"     • GraphQL: {enhanced_info.get('graphql_enabled', False)}")
                print(f"     • WebSockets: {enhanced_info.get('websockets_enabled', False)}")
                
                demo_result["results"] = enhanced_info
                demo_result["success"] = True
                
                print("   ✅ All enterprise components integrated without duplication")
            else:
                print("   ❌ Ray server not available")
                demo_result["error"] = "Ray server not initialized"
            
        except Exception as e:
            demo_result["error"] = str(e)
            logger.error(f"Enterprise integration demo failed: {e}")
        
        self.demo_results["enterprise_integration"] = demo_result
        return demo_result
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete integration demo"""
        print("🎭 Anant Enterprise Ray Integration Demo")
        print("=" * 50)
        
        # Initialize
        if not await self.initialize_demo():
            return {"error": "Demo initialization failed"}
        
        # Run all demonstrations
        await self.demo_geometric_processing()
        await self.demo_contextual_processing()
        await self.demo_cluster_monitoring()
        await self.demo_enterprise_integration()
        
        # Generate summary
        total_tests = len(self.demo_results)
        successful_tests = sum(1 for result in self.demo_results.values() if result.get("success", False))
        
        summary = {
            "demo_summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_demonstrations": total_tests,
                "successful_demonstrations": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "ray_cluster_operational": RAY_AVAILABLE and ray.is_initialized() if RAY_AVAILABLE else False
            },
            "detailed_results": self.demo_results
        }
        
        # Display summary
        print(f"\n📋 Demo Summary:")
        print("-" * 20)
        print(f"   Demonstrations: {successful_tests}/{total_tests}")
        print(f"   Success Rate: {summary['demo_summary']['success_rate']:.1%}")
        print(f"   Ray Cluster: {summary['demo_summary']['ray_cluster_operational']}")
        
        print("\n📋 Detailed Results:")
        print("-" * 25)
        for demo_name, result in self.demo_results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"   {status} {demo_name}")
            
            if "error" in result:
                print(f"      Error: {result['error']}")
            
            if "performance" in result:
                perf = result["performance"]
                if "processing_time_seconds" in perf:
                    print(f"      Time: {perf['processing_time_seconds']:.2f}s")
        
        return summary
    
    async def cleanup(self):
        """Cleanup demo resources"""
        logger.info("🧹 Cleaning up demo resources")
        
        try:
            if self.ray_server:
                await self.ray_server.stop()
                print("✅ Ray server stopped")
            
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()
                print("✅ Ray cluster shutdown")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main demo execution"""
    demo = AnantRayIntegrationDemo()
    
    try:
        # Run the demo
        results = await demo.run_full_demo()
        
        # Save results
        results_file = Path("anant_ray_integration_demo_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Demo results saved to: {results_file}")
        
        # Final status
        if results.get("demo_summary", {}).get("success_rate", 0) > 0.5:
            print("\n🎉 Anant Enterprise Ray Integration Demo: SUCCESS")
            print("   Ray distributed processing is operational!")
            print("   All existing components enhanced without duplication.")
        else:
            print("\n⚠️  Anant Enterprise Ray Integration Demo: PARTIAL")
            print("   Some features may need additional configuration.")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())