"""
Distributed Graph Manager
========================

High-level interface for distributed computing across all graph types.
Provides a unified API for scaling any graph operation across clusters.

Supports:
- Automatic graph type detection
- Intelligent partitioning strategies
- Load balancing across worker nodes
- Fault tolerance and recovery
- Performance monitoring

Usage:
    from anant.distributed import DistributedGraphManager
    
    # Initialize with cluster
    dgm = DistributedGraphManager(cluster_config="cluster.yaml")
    
    # Run distributed operations on any graph type
    result = await dgm.execute("centrality", my_knowledge_graph)
    result = await dgm.execute("clustering", my_metagraph) 
    result = await dgm.execute("search", my_hierarchical_kg, query="AI research")
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
from dataclasses import dataclass

from .cluster_manager import ClusterManager, ClusterConfig
from .task_scheduler import TaskScheduler, SchedulingStrategy
from .worker_node import WorkerNode, WorkerConfig
from .message_broker import MessageBroker
from .fault_tolerance import FaultToleranceManager
from .auto_scaler import AutoScaler
from .distributed_cache import DistributedCache
from .cluster_monitor import ClusterMonitor

from .graph_operations import (
    MultiGraphDistributedOperations, 
    GraphType, 
    DistributedGraphResult,
    register_graph_operations
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedGraphConfig:
    """Configuration for distributed graph operations"""
    cluster_config_path: Optional[str] = None
    max_workers: int = 8
    auto_scaling: bool = True
    fault_tolerance: bool = True
    caching_enabled: bool = True
    monitoring_enabled: bool = True
    
    # Graph-specific settings
    auto_partition_size: int = 1000  # Nodes per partition
    partition_strategy: str = "auto"  # "auto", "manual", "load_balanced"
    
    # Performance settings
    task_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    
    # Resource settings
    memory_limit_gb: float = 8.0
    cpu_cores_per_worker: int = 2


class DistributedGraphManager:
    """
    High-level manager for distributed graph operations across all graph types
    """
    
    def __init__(self, config: Optional[DistributedGraphConfig] = None):
        self.config = config or DistributedGraphConfig()
        
        # Core distributed components
        self.cluster_manager: Optional[ClusterManager] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.message_broker: Optional[MessageBroker] = None
        self.fault_tolerance: Optional[FaultToleranceManager] = None
        self.auto_scaler: Optional[AutoScaler] = None
        self.distributed_cache: Optional[DistributedCache] = None
        self.cluster_monitor: Optional[ClusterMonitor] = None
        
        # Graph operations handler
        self.graph_operations: Optional[MultiGraphDistributedOperations] = None
        
        # State tracking
        self.is_initialized = False
        self.active_operations: Dict[str, DistributedGraphResult] = {}
        
        logger.info("Created DistributedGraphManager")
    
    async def initialize(self, cluster_nodes: Optional[List[str]] = None) -> bool:
        """
        Initialize the distributed system
        
        Args:
            cluster_nodes: List of node addresses (if None, starts local cluster)
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing distributed graph system...")
            
            # 1. Initialize cluster manager
            cluster_config = self._load_cluster_config()
            self.cluster_manager = ClusterManager(cluster_config)
            
            if cluster_nodes:
                await self.cluster_manager.initialize_cluster(cluster_nodes)
            else:
                await self.cluster_manager.start_local_cluster(self.config.max_workers)
            
            # 2. Initialize message broker
            self.message_broker = MessageBroker()
            await self.message_broker.initialize()
            
            # 3. Initialize task scheduler
            self.task_scheduler = TaskScheduler(
                cluster_manager=self.cluster_manager,
                strategy=SchedulingStrategy.RESOURCE_AWARE
            )
            
            # 4. Initialize fault tolerance
            if self.config.fault_tolerance:
                self.fault_tolerance = FaultToleranceManager(
                    cluster_manager=self.cluster_manager,
                    task_scheduler=self.task_scheduler
                )
                await self.fault_tolerance.start()
            
            # 5. Initialize auto scaler
            if self.config.auto_scaling:
                self.auto_scaler = AutoScaler(
                    cluster_manager=self.cluster_manager,
                    max_workers=self.config.max_workers
                )
                await self.auto_scaler.start()
            
            # 6. Initialize distributed cache
            if self.config.caching_enabled:
                self.distributed_cache = DistributedCache()
                await self.distributed_cache.initialize()
            
            # 7. Initialize cluster monitor
            if self.config.monitoring_enabled:
                self.cluster_monitor = ClusterMonitor(self.cluster_manager)
                await self.cluster_monitor.start()
            
            # 8. Initialize graph operations handler
            self.graph_operations = MultiGraphDistributedOperations(
                cluster_manager=self.cluster_manager,
                task_scheduler=self.task_scheduler
            )
            
            # 9. Register graph operation functions with workers
            register_graph_operations()
            
            self.is_initialized = True
            logger.info("✅ Distributed graph system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize distributed system: {e}")
            return False
    
    async def execute(self, 
                     operation: str,
                     graph: Any,
                     operation_args: Optional[Dict[str, Any]] = None,
                     num_partitions: Optional[int] = None,
                     cache_key: Optional[str] = None) -> DistributedGraphResult:
        """
        Execute a distributed operation on any graph type
        
        Args:
            operation: Operation name ("centrality", "clustering", "search", etc.)
            graph: Graph object (any supported type)
            operation_args: Operation-specific arguments
            num_partitions: Number of partitions (auto-calculated if None)
            cache_key: Optional cache key for result caching
        
        Returns:
            DistributedGraphResult with operation results
        """
        if not self.is_initialized:
            raise RuntimeError("DistributedGraphManager not initialized. Call initialize() first.")
        
        logger.info(f"🚀 Starting distributed {operation} operation")
        
        # Check cache first
        if cache_key and self.distributed_cache:
            cached_result = await self.distributed_cache.get(cache_key)
            if cached_result:
                logger.info(f"📦 Returning cached result for {operation}")
                return cached_result
        
        # Determine optimal partitions if not specified
        if num_partitions is None:
            num_partitions = self._calculate_optimal_partitions(graph, operation)
        
        # Execute distributed operation
        try:
            result = await self.graph_operations.distributed_operation(
                graph=graph,
                operation=operation,
                operation_args=operation_args or {},
                num_partitions=num_partitions
            )
            
            # Cache result if successful
            if result.success and cache_key and self.distributed_cache:
                await self.distributed_cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
            
            # Track operation
            self.active_operations[result.operation_id] = result
            
            logger.info(f"✅ Completed distributed {operation} in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Distributed operation failed: {e}")
            # Return failed result
            return DistributedGraphResult(
                operation_id=f"failed_{operation}",
                graph_type=self.graph_operations.detect_graph_type(graph),
                result_data=None,
                partition_results={},
                execution_time=0.0,
                nodes_processed=0,
                edges_processed=0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_optimal_partitions(self, graph: Any, operation: str) -> int:
        """Calculate optimal number of partitions based on graph and cluster"""
        available_workers = self.cluster_manager.get_available_workers()
        
        # Get graph size
        if hasattr(graph, 'num_nodes'):
            graph_size = graph.num_nodes
        elif hasattr(graph, 'get_statistics'):
            stats = graph.get_statistics()
            graph_size = stats.get('basic_stats', {}).get('num_nodes', 1000)
        else:
            graph_size = 1000  # Default estimate
        
        # Calculate based on target partition size
        target_partition_size = self.config.auto_partition_size
        ideal_partitions = max(1, graph_size // target_partition_size)
        
        # Limit by available workers
        optimal_partitions = min(ideal_partitions, available_workers)
        
        logger.info(f"📊 Calculated {optimal_partitions} partitions for graph with {graph_size} nodes")
        return optimal_partitions
    
    def _load_cluster_config(self) -> ClusterConfig:
        """Load cluster configuration"""
        if self.config.cluster_config_path:
            config_path = Path(self.config.cluster_config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return ClusterConfig(**config_data)
        
        # Default configuration
        return ClusterConfig(
            cluster_name="anant_graph_cluster",
            max_workers=self.config.max_workers,
            heartbeat_interval=5.0,
            failure_timeout=30.0
        )
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        status = {
            "initialized": self.is_initialized,
            "workers": self.cluster_manager.get_worker_count(),
            "available_workers": self.cluster_manager.get_available_workers(),
            "active_operations": len(self.active_operations),
        }
        
        if self.cluster_monitor:
            monitor_status = await self.cluster_monitor.get_cluster_metrics()
            status.update(monitor_status)
        
        return status
    
    async def scale_cluster(self, target_workers: int) -> bool:
        """Scale cluster to target number of workers"""
        if not self.auto_scaler:
            logger.warning("Auto-scaling not enabled")
            return False
        
        try:
            await self.auto_scaler.scale_to(target_workers)
            logger.info(f"✅ Scaled cluster to {target_workers} workers")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to scale cluster: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the distributed system"""
        logger.info("🔄 Shutting down distributed graph system...")
        
        try:
            # Stop monitoring
            if self.cluster_monitor:
                await self.cluster_monitor.stop()
            
            # Stop auto-scaling
            if self.auto_scaler:
                await self.auto_scaler.stop()
            
            # Stop fault tolerance
            if self.fault_tolerance:
                await self.fault_tolerance.stop()
            
            # Shutdown cache
            if self.distributed_cache:
                await self.distributed_cache.shutdown()
            
            # Shutdown message broker
            if self.message_broker:
                await self.message_broker.shutdown()
            
            # Shutdown cluster
            if self.cluster_manager:
                await self.cluster_manager.shutdown()
            
            self.is_initialized = False
            logger.info("✅ Distributed system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Convenience functions for common operations

async def distributed_centrality(graph: Any, **kwargs) -> DistributedGraphResult:
    """Run distributed centrality calculation"""
    dgm = DistributedGraphManager()
    await dgm.initialize()
    try:
        return await dgm.execute("centrality", graph, **kwargs)
    finally:
        await dgm.shutdown()


async def distributed_clustering(graph: Any, **kwargs) -> DistributedGraphResult:
    """Run distributed clustering analysis"""
    dgm = DistributedGraphManager()
    await dgm.initialize()
    try:
        return await dgm.execute("clustering", graph, **kwargs)
    finally:
        await dgm.shutdown()


async def distributed_search(graph: Any, query: str, **kwargs) -> DistributedGraphResult:
    """Run distributed graph search"""
    dgm = DistributedGraphManager()
    await dgm.initialize()
    try:
        return await dgm.execute("search", graph, {"query": query}, **kwargs)
    finally:
        await dgm.shutdown()


# Context manager for automatic resource management
class DistributedGraphContext:
    """Context manager for distributed graph operations"""
    
    def __init__(self, config: Optional[DistributedGraphConfig] = None):
        self.manager = DistributedGraphManager(config)
    
    async def __aenter__(self):
        await self.manager.initialize()
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.shutdown()


# Usage example with context manager
"""
async def example_usage():
    from anant import KnowledgeGraph
    from anant.distributed import DistributedGraphContext
    
    # Create a knowledge graph
    kg = KnowledgeGraph()
    # ... populate graph ...
    
    # Use distributed operations
    async with DistributedGraphContext() as dgm:
        # Run multiple distributed operations
        centrality_result = await dgm.execute("centrality", kg)
        clustering_result = await dgm.execute("clustering", kg)
        search_result = await dgm.execute("search", kg, {"query": "machine learning"})
        
        print(f"Centrality completed in {centrality_result.execution_time:.2f}s")
        print(f"Processed {centrality_result.nodes_processed} nodes")
"""