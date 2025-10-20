"""
Backend Strategy and Communication Protocol Implementation

This module defines the strategy for choosing between different distributed computing
backends and communication protocols based on workload characteristics.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Types of distributed workloads."""
    GRAPH_ANALYTICS = "graph_analytics"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    DATA_PROCESSING = "data_processing"
    REAL_TIME_STREAMING = "real_time_streaming"
    BATCH_PROCESSING = "batch_processing"
    INTERACTIVE_QUERY = "interactive_query"


class CommunicationPattern(Enum):
    """Communication patterns for distributed tasks."""
    SCATTER_GATHER = "scatter_gather"
    MAP_REDUCE = "map_reduce"
    PIPELINE = "pipeline"
    BROADCAST = "broadcast"
    ALL_TO_ALL = "all_to_all"
    PEER_TO_PEER = "peer_to_peer"
    GOSSIP_PROTOCOL = "gossip"


class BackendType(Enum):
    """Available distributed computing backends."""
    DASK = "dask"
    RAY = "ray"
    CELERY = "celery"
    NATIVE = "native"
    HYBRID = "hybrid"


@dataclass
class BackendCapabilities:
    """Capabilities and characteristics of a backend."""
    backend_type: BackendType
    best_for_workloads: List[WorkloadType]
    supported_patterns: List[CommunicationPattern]
    max_nodes: int
    fault_tolerance: bool
    real_time_capable: bool
    memory_efficient: bool
    setup_complexity: str  # "low", "medium", "high"
    performance_score: float  # 0.0 to 1.0


class DistributedBackendStrategy:
    """
    Strategic decision engine for choosing distributed computing backends.
    
    This class implements the "backend selection strategy" that automatically
    chooses the best backend based on:
    - Workload characteristics
    - Cluster size and resources
    - Performance requirements
    - Fault tolerance needs
    """
    
    def __init__(self):
        """Initialize the backend strategy engine."""
        
        # Define capabilities for each backend
        self.backend_capabilities = {
            BackendType.DASK: BackendCapabilities(
                backend_type=BackendType.DASK,
                best_for_workloads=[
                    WorkloadType.DATA_PROCESSING,
                    WorkloadType.BATCH_PROCESSING,
                    WorkloadType.GRAPH_ANALYTICS
                ],
                supported_patterns=[
                    CommunicationPattern.SCATTER_GATHER,
                    CommunicationPattern.MAP_REDUCE,
                    CommunicationPattern.PIPELINE
                ],
                max_nodes=1000,
                fault_tolerance=True,
                real_time_capable=False,
                memory_efficient=True,
                setup_complexity="medium",
                performance_score=0.8
            ),
            
            BackendType.RAY: BackendCapabilities(
                backend_type=BackendType.RAY,
                best_for_workloads=[
                    WorkloadType.ML_TRAINING,
                    WorkloadType.ML_INFERENCE,
                    WorkloadType.REAL_TIME_STREAMING
                ],
                supported_patterns=[
                    CommunicationPattern.SCATTER_GATHER,
                    CommunicationPattern.BROADCAST,
                    CommunicationPattern.ALL_TO_ALL,
                    CommunicationPattern.PEER_TO_PEER
                ],
                max_nodes=10000,
                fault_tolerance=True,
                real_time_capable=True,
                memory_efficient=False,
                setup_complexity="high",
                performance_score=0.9
            ),
            
            BackendType.CELERY: BackendCapabilities(
                backend_type=BackendType.CELERY,
                best_for_workloads=[
                    WorkloadType.BATCH_PROCESSING,
                    WorkloadType.DATA_PROCESSING
                ],
                supported_patterns=[
                    CommunicationPattern.SCATTER_GATHER,
                    CommunicationPattern.PIPELINE
                ],
                max_nodes=500,
                fault_tolerance=True,
                real_time_capable=False,
                memory_efficient=True,
                setup_complexity="low",
                performance_score=0.6
            ),
            
            BackendType.NATIVE: BackendCapabilities(
                backend_type=BackendType.NATIVE,
                best_for_workloads=[
                    WorkloadType.INTERACTIVE_QUERY,
                    WorkloadType.GRAPH_ANALYTICS,
                    WorkloadType.REAL_TIME_STREAMING
                ],
                supported_patterns=[
                    CommunicationPattern.SCATTER_GATHER,
                    CommunicationPattern.MAP_REDUCE,
                    CommunicationPattern.BROADCAST,
                    CommunicationPattern.PEER_TO_PEER,
                    CommunicationPattern.GOSSIP_PROTOCOL
                ],
                max_nodes=100,
                fault_tolerance=True,
                real_time_capable=True,
                memory_efficient=True,
                setup_complexity="medium",
                performance_score=0.7
            )
        }
        
        # Communication protocol preferences
        self.communication_protocols = {
            # High-performance protocols for different scenarios
            "high_throughput": ["zmq", "grpc", "tcp"],
            "low_latency": ["zmq", "udp", "grpc"],
            "fault_tolerant": ["grpc", "tcp", "redis"],
            "simple_setup": ["redis", "tcp", "http"],
            "secure": ["grpc_tls", "https", "tcp_tls"]
        }
        
    def select_backend(self, 
                      workload_type: WorkloadType,
                      cluster_size: int,
                      performance_priority: str = "balanced",  # "speed", "memory", "balanced"
                      fault_tolerance_required: bool = True,
                      real_time_required: bool = False) -> BackendType:
        """
        Select the best backend for given requirements.
        
        Args:
            workload_type: Type of workload to run
            cluster_size: Number of nodes in cluster
            performance_priority: Performance optimization priority
            fault_tolerance_required: Whether fault tolerance is required
            real_time_required: Whether real-time processing is needed
            
        Returns:
            Recommended backend type
        """
        candidates = []
        
        for backend_type, capabilities in self.backend_capabilities.items():
            # Check basic requirements
            if workload_type not in capabilities.best_for_workloads:
                continue
                
            if cluster_size > capabilities.max_nodes:
                continue
                
            if fault_tolerance_required and not capabilities.fault_tolerance:
                continue
                
            if real_time_required and not capabilities.real_time_capable:
                continue
                
            # Calculate score based on priorities
            score = capabilities.performance_score
            
            if performance_priority == "speed":
                score *= 1.2 if capabilities.real_time_capable else 0.8
            elif performance_priority == "memory":
                score *= 1.2 if capabilities.memory_efficient else 0.8
                
            candidates.append((backend_type, score))
            
        if not candidates:
            logger.warning("No suitable backend found, falling back to NATIVE")
            return BackendType.NATIVE
            
        # Return highest scoring backend
        best_backend = max(candidates, key=lambda x: x[1])[0]
        logger.info(f"Selected backend: {best_backend.value} for workload: {workload_type.value}")
        return best_backend
        
    def select_communication_protocol(self,
                                    backend_type: BackendType,
                                    priority: str = "balanced") -> str:
        """
        Select communication protocol based on backend and priorities.
        
        Args:
            backend_type: Selected backend type
            priority: Communication priority ("throughput", "latency", "reliability")
            
        Returns:
            Recommended communication protocol
        """
        if backend_type == BackendType.RAY:
            # Ray has its own optimized protocols
            return "ray_native"
        elif backend_type == BackendType.DASK:
            # Dask typically uses TCP with custom protocols
            return "dask_native"
        elif backend_type == BackendType.CELERY:
            # Celery works well with Redis/RabbitMQ
            return "redis"
        else:
            # Native backend - choose based on priority
            if priority == "throughput":
                return "zmq"
            elif priority == "latency":
                return "zmq"
            elif priority == "reliability":
                return "grpc"
            else:
                return "zmq"  # Good balance of all factors
                
    def should_use_gossip_protocol(self,
                                  cluster_size: int,
                                  node_volatility: str = "low") -> bool:
        """
        Determine if gossip protocol should be used for cluster management.
        
        Args:
            cluster_size: Number of nodes in cluster
            node_volatility: How often nodes join/leave ("low", "medium", "high")
            
        Returns:
            True if gossip protocol is recommended
        """
        # Gossip is good for:
        # - Large clusters (>50 nodes)
        # - High node volatility
        # - Decentralized coordination
        
        if cluster_size > 50:
            return True
            
        if node_volatility in ["medium", "high"]:
            return True
            
        return False
        
    def get_hybrid_strategy(self,
                           workloads: List[WorkloadType],
                           cluster_size: int) -> Dict[WorkloadType, BackendType]:
        """
        Create a hybrid strategy using multiple backends for different workloads.
        
        Args:
            workloads: List of workload types to support
            cluster_size: Cluster size
            
        Returns:
            Mapping of workload types to recommended backends
        """
        strategy = {}
        
        for workload in workloads:
            backend = self.select_backend(
                workload_type=workload,
                cluster_size=cluster_size,
                performance_priority="balanced"
            )
            strategy[workload] = backend
            
        return strategy


# Strategic recommendations based on analysis
COMMUNICATION_STRATEGY = {
    "primary_protocols": {
        "cluster_management": "gossip",  # For node discovery and health
        "task_distribution": "grpc",     # For reliable task scheduling  
        "data_transfer": "zmq",          # For high-throughput data movement
        "monitoring": "http",            # For metrics and status
        "coordination": "redis"          # For distributed locking/coordination
    },
    
    "fallback_protocols": {
        "cluster_management": "tcp",
        "task_distribution": "tcp", 
        "data_transfer": "tcp",
        "monitoring": "tcp",
        "coordination": "zmq"
    },
    
    "security_protocols": {
        "cluster_management": "gossip_tls",
        "task_distribution": "grpc_tls",
        "data_transfer": "zmq_curve",
        "monitoring": "https",
        "coordination": "redis_tls"
    }
}

BACKEND_RECOMMENDATIONS = {
    # When to use each backend
    "use_dask_when": [
        "Large dataframe operations",
        "ETL pipelines", 
        "Batch analytics",
        "Scientific computing",
        "Cluster size 10-1000 nodes"
    ],
    
    "use_ray_when": [
        "ML model training",
        "Hyperparameter tuning",
        "Reinforcement learning",
        "Real-time inference",
        "Complex task dependencies",
        "Cluster size 10-10000 nodes"
    ],
    
    "use_celery_when": [
        "Background job processing",
        "Simple task queues",
        "Web application backends",
        "Long-running tasks",
        "Cluster size 5-500 nodes"
    ],
    
    "use_native_when": [
        "Custom graph algorithms",
        "Low-latency requirements",
        "Full control needed",
        "Small to medium clusters",
        "Prototype development",
        "Cluster size 2-100 nodes"
    ]
}