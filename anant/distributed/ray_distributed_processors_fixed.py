#!/usr/bin/env python3
"""
Ray Distributed Processors for Anant Enterprise - FIXED VERSION
==============================================================

Ray actors for distributed processing of Anant components.
Uses correct APIs from existing geometry and LCG modules.

Key Principle: EXTEND existing components, never duplicate functionality.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass

# Ray imports with proper error handling
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    logging.warning("Ray not available - distributed processing disabled")

# Import existing Anant components with proper error handling
GEOMETRY_AVAILABLE = False
LCG_AVAILABLE = False

try:
    from anant.geometry.core.property_manifold import PropertyManifold
    GEOMETRY_AVAILABLE = True
except ImportError:
    logging.warning("PropertyManifold not available")

try:
    from anant.layered_contextual_graph.core.layered_contextual_graph import LayeredContextualGraph, LayerType
    LCG_AVAILABLE = True
except ImportError:
    logging.warning("LayeredContextualGraph not available")

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Distributed processing task definition"""
    task_id: str
    task_type: str
    graph_id: str
    parameters: Dict[str, Any]
    created_at: datetime
    priority: int = 1


@dataclass
class ProcessingResult:
    """Result from distributed processing"""
    task_id: str
    success: bool
    result_data: Any
    execution_time: float
    node_id: str
    error_message: Optional[str] = None


# Only define Ray actors if Ray is available
if RAY_AVAILABLE:
    
    @ray.remote
    class RayGeometricProcessor:
        """
        Ray actor for distributed geometric computations
        
        Uses existing PropertyManifold API correctly
        """
        
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.startup_time = datetime.utcnow()
            self.processed_tasks = 0
            self.manifold_cache: Dict[str, Any] = {}
            
            logger.info(f"GeometricProcessor {node_id} initialized")
        
        def compute_property_curvature(self, task_data: Dict[str, Any]) -> ProcessingResult:
            """
            Compute property manifold curvature using existing PropertyManifold
            """
            task_id = task_data.get("task_id", f"curvature_{int(time.time())}")
            start_time = time.time()
            
            try:
                if not GEOMETRY_AVAILABLE:
                    raise RuntimeError("PropertyManifold not available")
                
                # Use existing PropertyManifold with correct API
                property_vectors = task_data["property_vectors"]
                property_weights = task_data.get("property_weights")
                
                cache_key = task_data.get("cache_key", task_id)
                
                if cache_key not in self.manifold_cache:
                    manifold = PropertyManifold(
                        property_vectors=property_vectors,
                        property_weights=property_weights
                    )
                    self.manifold_cache[cache_key] = manifold
                
                manifold = self.manifold_cache[cache_key]
                
                # Use existing methods - compute_curvature() method exists
                curvature = manifold.compute_curvature()
                
                # Use existing methods - compute_metric() method exists  
                metric = manifold.compute_metric()
                
                result_data = {
                    "scalar_curvature": curvature.scalar_curvature,
                    "sectional_curvature": curvature.sectional_curvature,
                    "gaussian_curvature": curvature.gaussian_curvature,
                    "metric_condition_number": metric.condition_number(),
                    "property_count": len(manifold.property_list),
                    "entity_count": len(manifold.entity_ids)
                }
                
                self.processed_tasks += 1
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task_id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Property curvature computation failed: {e}")
                
                return ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result_data=None,
                    execution_time=execution_time,
                    node_id=self.node_id,
                    error_message=str(e)
                )
        
        def detect_property_outliers(self, task_data: Dict[str, Any]) -> ProcessingResult:
            """
            Detect outliers using existing PropertyManifold methods
            """
            task_id = task_data.get("task_id", f"outliers_{int(time.time())}")
            start_time = time.time()
            
            try:
                if not GEOMETRY_AVAILABLE:
                    raise RuntimeError("PropertyManifold not available")
                
                # Use existing PropertyManifold
                property_vectors = task_data["property_vectors"]
                property_weights = task_data.get("property_weights")
                
                manifold = PropertyManifold(
                    property_vectors=property_vectors,
                    property_weights=property_weights
                )
                
                # Use existing method - detect_property_outliers exists
                z_threshold = task_data.get("z_threshold", 3.0)
                outliers = manifold.detect_property_outliers(z_threshold=z_threshold)
                
                result_data = {
                    "outlier_entities": outliers,
                    "outlier_count": len(outliers),
                    "threshold_used": z_threshold,
                    "total_entities": len(manifold.entity_ids)
                }
                
                self.processed_tasks += 1
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task_id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Property outlier detection failed: {e}")
                
                return ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result_data=None,
                    execution_time=execution_time,
                    node_id=self.node_id,
                    error_message=str(e)
                )
        
        def get_stats(self) -> Dict[str, Any]:
            """Get processor statistics"""
            uptime = datetime.utcnow() - self.startup_time
            
            return {
                "node_id": self.node_id,
                "processor_type": "geometric",
                "uptime_seconds": int(uptime.total_seconds()),
                "tasks_processed": self.processed_tasks,
                "manifolds_cached": len(self.manifold_cache),
                "geometry_available": GEOMETRY_AVAILABLE
            }

    @ray.remote  
    class RayContextualProcessor:
        """
        Ray actor for distributed layered contextual graph processing
        
        Uses existing LayeredContextualGraph API correctly
        """
        
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.startup_time = datetime.utcnow()
            self.processed_tasks = 0
            # Use Any type to avoid import issues
            self.lcg_cache: Dict[str, Any] = {}
            
            logger.info(f"ContextualProcessor {node_id} initialized")
        
        def process_layers(self, task_data: Dict[str, Any]) -> ProcessingResult:
            """
            Process layered contextual graph using existing LCG methods
            """
            task_id = task_data.get("task_id", f"layers_{int(time.time())}")
            start_time = time.time()
            
            try:
                if not LCG_AVAILABLE:
                    raise RuntimeError("LayeredContextualGraph not available")
                
                graph_id = task_data["graph_id"]
                
                # Create LCG instance using correct constructor
                if graph_id not in self.lcg_cache:
                    # LayeredContextualGraph constructor takes graph_id
                    lcg = LayeredContextualGraph(graph_id)
                    
                    # Add layers using correct API - add_layer(name, hypergraph, layer_type, ...)
                    for layer_config in task_data.get("layers", []):
                        # Create dummy hypergraph for now (would use real Anant hypergraph in practice)
                        dummy_hypergraph = {"entities": layer_config.get("entities", [])}
                        
                        lcg.add_layer(
                            name=layer_config["name"],
                            hypergraph=dummy_hypergraph,
                            layer_type=LayerType(layer_config["type"]),
                            level=layer_config.get("level", 0),
                            metadata=layer_config.get("metadata", {})
                        )
                    
                    self.lcg_cache[graph_id] = lcg
                
                lcg = self.lcg_cache[graph_id]
                
                # Use existing methods that actually exist
                layer_results = {}
                
                for operation in task_data.get("operations", []):
                    op_type = operation["type"]
                    
                    if op_type == "query_layers":
                        # Use existing query_across_layers method
                        query = operation["query"]
                        result = lcg.query_across_layers(
                            query=query,
                            layers=operation.get("layers", list(lcg.layers.keys()))
                        )
                        layer_results[f"query_{operation['id']}"] = result
                        
                    elif op_type == "layer_hierarchy":
                        # Use existing get_layer_hierarchy method  
                        result = lcg.get_layer_hierarchy()
                        layer_results[f"hierarchy_{operation['id']}"] = result
                
                result_data = {
                    "graph_id": graph_id,
                    "layers_count": len(lcg.layers),
                    "operations_completed": len(layer_results),
                    "results": layer_results,
                    "processing_node": self.node_id
                }
                
                self.processed_tasks += 1
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task_id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    node_id=self.node_id
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Layer processing failed: {e}")
                
                return ProcessingResult(
                    task_id=task_id,
                    success=False,
                    result_data=None,
                    execution_time=execution_time,
                    node_id=self.node_id,
                    error_message=str(e)
                )
        
        def get_stats(self) -> Dict[str, Any]:
            """Get processor statistics"""
            uptime = datetime.utcnow() - self.startup_time
            
            return {
                "node_id": self.node_id,
                "processor_type": "contextual",
                "uptime_seconds": int(uptime.total_seconds()),
                "tasks_processed": self.processed_tasks,
                "lcg_instances_cached": len(self.lcg_cache),
                "lcg_available": LCG_AVAILABLE
            }


class RayWorkloadDistributor:
    """
    Distributes workloads across Ray cluster
    
    Manages geometric and contextual processors with proper error handling
    """
    
    def __init__(self):
        self.geometric_processors: List[Any] = []
        self.contextual_processors: List[Any] = []
        self.active_tasks: Dict[str, ProcessingTask] = {}
        
    async def initialize_processors(self, cluster_config: Dict[str, Any]) -> bool:
        """Initialize Ray processors if Ray is available"""
        if not RAY_AVAILABLE:
            logger.error("Ray not available for processor initialization")
            return False
            
        try:
            # Create geometric processors if geometry is available
            if GEOMETRY_AVAILABLE:
                num_geo = cluster_config.get("geometric_processors", 2)
                for i in range(num_geo):
                    processor = RayGeometricProcessor.remote(f"geo_{i}")
                    self.geometric_processors.append(processor)
                logger.info(f"Created {num_geo} geometric processors")
            
            # Create contextual processors if LCG is available
            if LCG_AVAILABLE:
                num_ctx = cluster_config.get("contextual_processors", 2)
                for i in range(num_ctx):
                    processor = RayContextualProcessor.remote(f"ctx_{i}")
                    self.contextual_processors.append(processor)
                logger.info(f"Created {num_ctx} contextual processors")
            
            total_processors = len(self.geometric_processors) + len(self.contextual_processors)
            logger.info(f"Initialized {total_processors} total processors")
            return total_processors > 0
            
        except Exception as e:
            logger.error(f"Processor initialization failed: {e}")
            return False
    
    async def submit_geometric_task(self, task_data: Dict[str, Any]) -> str:
        """Submit geometric computation task"""
        if not self.geometric_processors:
            raise RuntimeError("No geometric processors available")
        
        task_id = f"geo_{int(time.time())}_{len(self.active_tasks)}"
        task_data["task_id"] = task_id
        
        # Simple round-robin processor selection
        processor_idx = len(self.active_tasks) % len(self.geometric_processors)
        processor = self.geometric_processors[processor_idx]
        
        # Submit task based on operation type
        operation = task_data.get("operation", "curvature")
        if operation == "curvature":
            future = processor.compute_property_curvature.remote(task_data)
        elif operation == "outliers":
            future = processor.detect_property_outliers.remote(task_data)
        else:
            raise ValueError(f"Unknown geometric operation: {operation}")
        
        # Track task
        task = ProcessingTask(
            task_id=task_id,
            task_type="geometric",
            graph_id=task_data.get("graph_id", "unknown"),
            parameters=task_data,
            created_at=datetime.utcnow()
        )
        self.active_tasks[task_id] = task
        
        logger.info(f"Submitted geometric task {task_id}")
        return task_id
    
    async def submit_contextual_task(self, task_data: Dict[str, Any]) -> str:
        """Submit contextual processing task"""
        if not self.contextual_processors:
            raise RuntimeError("No contextual processors available")
        
        task_id = f"ctx_{int(time.time())}_{len(self.active_tasks)}"
        task_data["task_id"] = task_id
        
        # Simple round-robin processor selection
        processor_idx = len(self.active_tasks) % len(self.contextual_processors)
        processor = self.contextual_processors[processor_idx]
        
        # Submit layer processing task
        future = processor.process_layers.remote(task_data)
        
        # Track task
        task = ProcessingTask(
            task_id=task_id,
            task_type="contextual", 
            graph_id=task_data.get("graph_id", "unknown"),
            parameters=task_data,
            created_at=datetime.utcnow()
        )
        self.active_tasks[task_id] = task
        
        logger.info(f"Submitted contextual task {task_id}")
        return task_id
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the entire Ray cluster"""
        if not RAY_AVAILABLE:
            return {"error": "Ray not available", "processors": 0}
        
        try:
            # Collect stats from all processors
            stats_futures = []
            
            for processor in self.geometric_processors:
                stats_futures.append(processor.get_stats.remote())
                
            for processor in self.contextual_processors:
                stats_futures.append(processor.get_stats.remote())
            
            if stats_futures:
                all_stats = ray.get(stats_futures)
            else:
                all_stats = []
            
            return {
                "geometric_processors": len(self.geometric_processors),
                "contextual_processors": len(self.contextual_processors),
                "total_processors": len(self.geometric_processors) + len(self.contextual_processors),
                "active_tasks": len(self.active_tasks),
                "processor_stats": all_stats,
                "geometry_available": GEOMETRY_AVAILABLE,
                "lcg_available": LCG_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"error": str(e)}


# Testing and validation functions
def validate_dependencies() -> Dict[str, bool]:
    """Validate all dependencies for Ray distributed processing"""
    return {
        "ray_available": RAY_AVAILABLE,
        "geometry_available": GEOMETRY_AVAILABLE, 
        "lcg_available": LCG_AVAILABLE
    }


if __name__ == "__main__":
    print("🚀 Ray Distributed Processors for Anant Enterprise - FIXED")
    print("=" * 60)
    
    deps = validate_dependencies()
    
    print("Dependencies:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}: {available}")
    
    print("\nFeatures:")
    if RAY_AVAILABLE and GEOMETRY_AVAILABLE:
        print("  ✅ Ray-distributed geometric computations")
    else:
        print("  ❌ Ray-distributed geometric computations (dependencies missing)")
        
    if RAY_AVAILABLE and LCG_AVAILABLE:
        print("  ✅ Ray-distributed contextual processing")
    else:
        print("  ❌ Ray-distributed contextual processing (dependencies missing)")
    
    if RAY_AVAILABLE:
        print("  ✅ Intelligent workload distribution")
        print("  ✅ Cluster management and monitoring")
    else:
        print("  ❌ Ray features disabled (Ray not available)")
    
    print("\nProcessors:")
    print("  🧮 RayGeometricProcessor - Property manifold computations")
    print("  🏗️ RayContextualProcessor - Layer processing")
    print("  📊 RayWorkloadDistributor - Task coordination")