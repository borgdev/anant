"""
Distributed Computing System for Anant Knowledge Graph Library

This module provides comprehensive distributed computing capabilities for large-scale
graph operations across multiple nodes with load balancing, fault tolerance, and
auto-scaling.

Architecture Overview:
- ClusterManager: Central cluster coordination and node management
- TaskScheduler: Intelligent task distribution and load balancing
- WorkerNode: Distributed worker processes for computation
- MessageBroker: High-performance inter-node communication
- FaultTolerance: Automatic failure detection and recovery
- AutoScaler: Dynamic resource scaling based on workload
- DistributedCache: Cluster-wide caching and data sharing
- Monitor: Real-time cluster health and performance monitoring

The system supports multiple distributed computing backends:
- Dask: For dataframe and array operations
- Ray: For ML workloads and reinforcement learning
- Celery: For task queues and background jobs
- Custom: Native message-passing implementation

Key Features:
- Automatic node discovery and registration
- Dynamic load balancing with work stealing
- Fault tolerance with automatic failover
- Auto-scaling based on CPU, memory, and queue depth
- Distributed caching with consistency guarantees
- Real-time monitoring and alerting
- Support for heterogeneous node configurations
- Graceful degradation and recovery
"""

# Core distributed computing components
from .cluster_manager import ClusterManager, NodeInfo, ClusterConfig
from .task_scheduler import TaskScheduler, Task, TaskPriority, SchedulingStrategy
from .worker_node import WorkerNode, WorkerConfig, WorkerStatus
from .message_broker import MessageBroker, MessageType
from .fault_tolerance import FaultToleranceManager, FailureType
from .auto_scaler import AutoScaler, ScalingPolicy, ResourceMetrics
from .distributed_cache import DistributedCache, ConsistencyLevel
from .cluster_monitor import ClusterMonitor, MetricsCollector, AlertManager
from .strategy import DistributedBackendStrategy, BackendType, WorkloadType

# Multi-graph distributed operations
from .graph_operations import (
    MultiGraphDistributedOperations,
    GraphPartition,
    GraphType,
    DistributedGraphResult,
    GraphPartitioner,
    HypergraphPartitioner,
    KnowledgeGraphPartitioner,
    HierarchicalKnowledgeGraphPartitioner,
    MetagraphPartitioner
)

from .graph_manager import (
    DistributedGraphManager,
    DistributedGraphConfig,
    DistributedGraphContext,
    distributed_centrality,
    distributed_clustering,
    distributed_search
)

# Graph-specific operations (optional import)
try:
    from .graph_specific_ops import (
        GraphSpecificOperations,
        HypergraphOperations,
        KnowledgeGraphOperations,
        HierarchicalKnowledgeGraphOperations,
        MetagraphOperations,
        GraphOperationsFactory
    )
except ImportError:
    # Handle case where graph-specific operations are not available
    pass

# Distributed computing backends
from .backends import (
    DaskBackend, RayBackend, CeleryBackend, NativeBackend,
    BackendFactory, UnifiedBackendManager, BackendConfig, TaskResult
)

# Enterprise distributed features
from .enterprise_features import (
    EnterpriseDistributedManager,
    create_enterprise_manager,
    ShardingStrategy,
    ConsistencyModel,
    ReplicationStrategy,
    DistributedQueryProcessor,
    GraphSharder,
    HashBasedSharder,
    HierarchicalSharder,
    SemanticSharder
)

# Production-grade graph partitioning
from .partitioning import (
    ProductionPartitioner,
    PartitioningConfig,
    PartitioningAlgorithm,
    PartitioningObjective,
    PartitioningResult,
    MetisPartitioner,
    KaHiPPartitioner,
    create_production_partitioner
)

# Ray-based distributed computing
try:
    from .ray_anant_cluster import AnantRayCluster
    from .ray_distributed_processors_fixed import (
        GeometricProcessor,
        ContextualProcessor,
        MultiProcessor
    )
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

__version__ = "1.0.0"

__all__ = [
    # Core Components
    "ClusterManager",
    "TaskScheduler", 
    "WorkerNode",
    "MessageBroker",
    "FaultToleranceManager",
    "AutoScaler",
    "DistributedCache",
    "ClusterMonitor",
    
    # Strategy and Backend Selection
    "DistributedBackendStrategy",
    
    # Multi-graph distributed operations
    "MultiGraphDistributedOperations",
    "GraphPartition",
    "GraphType",
    "DistributedGraphResult",
    "GraphPartitioner",
    "HypergraphPartitioner",
    "KnowledgeGraphPartitioner", 
    "HierarchicalKnowledgeGraphPartitioner",
    "MetagraphPartitioner",
    
    # High-level graph manager
    "DistributedGraphManager",
    "DistributedGraphConfig",
    "DistributedGraphContext",
    "distributed_centrality",
    "distributed_clustering",
    "distributed_search",
    
    # Graph-specific operations (if available)
    # "GraphSpecificOperations",
    # "HypergraphOperations",
    # "KnowledgeGraphOperations", 
    # "HierarchicalKnowledgeGraphOperations",
    # "MetagraphOperations",
    # "GraphOperationsFactory",
    
    # Backend Implementations
    "DaskBackend",
    "RayBackend", 
    "CeleryBackend",
    "NativeBackend",
    "BackendFactory",
    "UnifiedBackendManager",
    "BackendConfig",
    "TaskResult",
    
    # Enterprise Features
    "EnterpriseDistributedManager",
    "create_enterprise_manager",
    "ShardingStrategy",
    "ConsistencyModel", 
    "ReplicationStrategy",
    "DistributedQueryProcessor",
    "GraphSharder",
    "HashBasedSharder",
    "HierarchicalSharder",
    "SemanticSharder",
    
    # High-level Operations (TODO: Uncomment when implemented)
    # "DistributedGraphOperations",
    # "DistributedMLPipeline",
    # "DistributedAnalytics",
    # "DistributedVisualization",
    
    # Production-grade partitioning
    "ProductionPartitioner",
    "PartitioningConfig",
    "PartitioningAlgorithm",
    "PartitioningObjective", 
    "PartitioningResult",
    "MetisPartitioner",
    "KaHiPPartitioner",
    "create_production_partitioner",
]

# Add Ray components to __all__ if available
if 'RAY_AVAILABLE' in locals() and RAY_AVAILABLE:
    __all__.extend([
        "AnantRayCluster",
        "GeometricProcessor",
        "ContextualProcessor", 
        "MultiProcessor"
    ])