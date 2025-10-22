"""
Distributed LayeredContextualGraph
==================================

Integrates LCG with Anant's distributed computing infrastructure for:
- Horizontal scaling across multiple nodes
- Distributed consensus (Raft protocol)
- Fault tolerance and high availability
- Load balancing and auto-scaling
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Anant distributed components
try:
    from anant.distributed import (
        DistributedGraphManager,
        ClusterManager,
        FaultTolerance,
        MessageBroker,
        ConsensusProtocol
    )
    from anant.distributed.backends import (
        RedisBackend,
        EtcdBackend,
        ConsulBackend
    )
    try:
        from anant.distributed import PartitionStrategy as AnantPartitionStrategy
    except ImportError:
        AnantPartitionStrategy = None
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    AnantPartitionStrategy = None
    logging.warning("Anant distributed module not available")

from ..core import LayeredContextualGraph, Layer

logger = logging.getLogger(__name__)


class LCGPartitionStrategy(Enum):
    """Partitioning strategies for LCG"""
    BY_LAYER = "by_layer"  # Each layer on different nodes
    BY_ENTITY = "by_entity"  # Hash entities across nodes
    BY_HIERARCHY = "by_hierarchy"  # Hierarchy levels on different nodes
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class DistributedConfig:
    """Configuration for distributed LCG"""
    cluster_name: str = "lcg_cluster"
    node_id: str = "node_1"
    backend: str = "redis"  # redis, etcd, consul
    consensus_protocol: str = "raft"  # raft, paxos
    partition_strategy: LCGPartitionStrategy = LCGPartitionStrategy.BY_LAYER
    replication_factor: int = 3
    enable_auto_scaling: bool = True
    enable_fault_tolerance: bool = True
    health_check_interval: int = 30  # seconds


class LCGClusterManager:
    """
    Manages LCG cluster using Anant's ClusterManager.
    
    Handles:
    - Node registration and discovery
    - Leader election
    - Health monitoring
    - Failover and recovery
    """
    
    def __init__(self, config: DistributedConfig):
        if not DISTRIBUTED_AVAILABLE:
            raise RuntimeError("Anant distributed module required")
        
        self.config = config
        
        # Initialize Anant cluster manager
        self.cluster_manager = ClusterManager(
            cluster_name=config.cluster_name,
            node_id=config.node_id,
            backend_type=config.backend
        )
        
        # Initialize consensus
        self.consensus = ConsensusProtocol(
            protocol=config.consensus_protocol,
            cluster_manager=self.cluster_manager
        )
        
        # Track LCG instances per node
        self.lcg_registry: Dict[str, LayeredContextualGraph] = {}
        
        logger.info(f"LCGClusterManager initialized: {config.cluster_name}")
    
    def register_lcg(self, lcg_name: str, lcg: LayeredContextualGraph):
        """Register an LCG instance with the cluster"""
        self.lcg_registry[lcg_name] = lcg
        
        # Register with cluster
        self.cluster_manager.register_node(
            node_id=self.config.node_id,
            metadata={
                'lcg_name': lcg_name,
                'layers': list(lcg.layers.keys()),
                'num_entities': len(lcg.superposition_states)
            }
        )
        
        logger.info(f"Registered LCG '{lcg_name}' with cluster")
    
    def get_leader(self) -> Optional[str]:
        """Get current cluster leader"""
        return self.consensus.get_leader()
    
    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        return self.consensus.is_leader(self.config.node_id)
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get health status of all nodes"""
        return self.cluster_manager.get_cluster_health()


class DistributedLayeredGraph(LayeredContextualGraph):
    """
    Distributed version of LayeredContextualGraph.
    
    Features:
    - Layers distributed across multiple nodes
    - Automatic replication and failover
    - Consensus-based updates
    - Distributed queries with routing
    - Auto-scaling based on load
    
    Examples:
        >>> config = DistributedConfig(
        ...     cluster_name="prod_cluster",
        ...     backend="redis",
        ...     replication_factor=3
        ... )
        >>> dlcg = DistributedLayeredGraph(
        ...     name="distributed_kg",
        ...     distributed_config=config
        ... )
        >>> dlcg.add_layer("physical", hg, level=0)  # Auto-distributed
    """
    
    def __init__(
        self,
        name: str = "distributed_lcg",
        distributed_config: Optional[DistributedConfig] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        if not DISTRIBUTED_AVAILABLE:
            raise RuntimeError("Distributed features require anant.distributed")
        
        self.distributed_config = distributed_config or DistributedConfig()
        
        # Initialize cluster manager
        self.cluster_manager = LCGClusterManager(self.distributed_config)
        self.cluster_manager.register_lcg(name, self)
        
        # Initialize distributed graph manager (Anant)
        self.graph_manager = DistributedGraphManager(
            backend_type=self.distributed_config.backend,
            partition_strategy=self._get_anant_partition_strategy()
        )
        
        # Initialize fault tolerance
        if self.distributed_config.enable_fault_tolerance:
            self.fault_tolerance = FaultTolerance(
                cluster_manager=self.cluster_manager.cluster_manager,
                replication_factor=self.distributed_config.replication_factor
            )
        else:
            self.fault_tolerance = None
        
        # Initialize message broker for layer synchronization
        self.message_broker = MessageBroker(
            backend_type=self.distributed_config.backend
        )
        
        # Layer-to-node mapping
        self.layer_locations: Dict[str, Set[str]] = {}  # layer_name -> node_ids
        
        logger.info(f"DistributedLayeredGraph initialized: {name}")
    
    def _get_anant_partition_strategy(self):
        """Convert LCG partition strategy to Anant strategy"""
        if not AnantPartitionStrategy:
            return "hash"  # Fallback string
        
        if self.distributed_config.partition_strategy == LCGPartitionStrategy.BY_ENTITY:
            return AnantPartitionStrategy.HASH_BASED if hasattr(AnantPartitionStrategy, 'HASH_BASED') else "hash"
        elif self.distributed_config.partition_strategy == LCGPartitionStrategy.BY_HIERARCHY:
            return AnantPartitionStrategy.RANGE_BASED if hasattr(AnantPartitionStrategy, 'RANGE_BASED') else "range"
        else:
            return AnantPartitionStrategy.GRAPH_AWARE if hasattr(AnantPartitionStrategy, 'GRAPH_AWARE') else "graph"
    
    def add_layer(
        self,
        name: str,
        hypergraph: Any,
        *args,
        target_nodes: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add layer with automatic distribution.
        
        Args:
            name: Layer name
            hypergraph: Hypergraph for this layer
            target_nodes: Specific nodes to place layer (optional)
            *args, **kwargs: Additional arguments
        """
        # Add layer locally
        super().add_layer(name, hypergraph, *args, **kwargs)
        
        # Distribute layer across cluster
        if target_nodes is None:
            # Auto-select nodes based on strategy
            target_nodes = self._select_target_nodes(name)
        
        # Register layer location
        self.layer_locations[name] = set(target_nodes)
        
        # Replicate layer to target nodes
        for node_id in target_nodes:
            self._replicate_layer_to_node(name, node_id)
        
        # Publish layer addition event
        self.message_broker.publish(
            topic=f"lcg.{self.name}.layer.added",
            message={'layer_name': name, 'nodes': target_nodes}
        )
        
        logger.info(f"Layer '{name}' distributed to nodes: {target_nodes}")
    
    def _select_target_nodes(self, layer_name: str) -> List[str]:
        """Select target nodes for layer placement"""
        cluster_health = self.cluster_manager.get_cluster_health()
        available_nodes = [
            node_id for node_id, health in cluster_health.items()
            if health['status'] == 'healthy'
        ]
        
        if not available_nodes:
            return [self.distributed_config.node_id]
        
        # Select nodes based on partition strategy
        strategy = self.distributed_config.partition_strategy
        
        if strategy == LCGPartitionStrategy.BY_LAYER:
            # Round-robin layer placement
            layer_index = len(self.layers)
            node_index = layer_index % len(available_nodes)
            primary = available_nodes[node_index]
            
            # Add replicas
            replicas = self._select_replicas(primary, available_nodes)
            return [primary] + replicas
        
        elif strategy == LCGPartitionStrategy.BY_HIERARCHY:
            # Place by hierarchy level
            layer = self.layers[layer_name]
            level = layer.level
            node_index = level % len(available_nodes)
            primary = available_nodes[node_index]
            replicas = self._select_replicas(primary, available_nodes)
            return [primary] + replicas
        
        else:
            # Default: use replication factor
            return available_nodes[:self.distributed_config.replication_factor]
    
    def _select_replicas(
        self,
        primary: str,
        available_nodes: List[str]
    ) -> List[str]:
        """Select replica nodes"""
        replicas = []
        for node in available_nodes:
            if node != primary and len(replicas) < (self.distributed_config.replication_factor - 1):
                replicas.append(node)
        return replicas
    
    def _replicate_layer_to_node(self, layer_name: str, node_id: str):
        """Replicate layer data to target node"""
        if node_id == self.distributed_config.node_id:
            return  # Already local
        
        # Use Anant's replication
        layer = self.layers[layer_name]
        if self.fault_tolerance:
            self.fault_tolerance.replicate_data(
                data_id=f"layer_{layer_name}",
                data=layer,
                target_node=node_id
            )
    
    def query_across_layers(
        self,
        entity_id: str,
        layers: Optional[List[str]] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Distributed query across layers.
        
        Routes queries to appropriate nodes and aggregates results.
        """
        # Determine which nodes to query
        if layers:
            target_nodes = set()
            for layer_name in layers:
                target_nodes.update(self.layer_locations.get(layer_name, set()))
        else:
            # Query all nodes
            target_nodes = set()
            for nodes in self.layer_locations.values():
                target_nodes.update(nodes)
        
        if not target_nodes:
            # Fallback to local query
            return super().query_across_layers(entity_id, layers, context, **kwargs)
        
        # Execute distributed query
        results = {}
        for node_id in target_nodes:
            if node_id == self.distributed_config.node_id:
                # Local query
                local_results = super().query_across_layers(
                    entity_id, layers, context, **kwargs
                )
                results.update(local_results)
            else:
                # Remote query via message broker
                remote_results = self._remote_query(node_id, entity_id, layers, context)
                results.update(remote_results)
        
        return results
    
    def _remote_query(
        self,
        node_id: str,
        entity_id: str,
        layers: Optional[List[str]],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Execute query on remote node"""
        # Send query request
        response = self.message_broker.request(
            topic=f"lcg.{self.name}.query",
            message={
                'entity_id': entity_id,
                'layers': layers,
                'context': context
            },
            target_node=node_id,
            timeout=5.0
        )
        
        return response.get('results', {})
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get distributed cluster status"""
        return {
            'cluster_name': self.distributed_config.cluster_name,
            'node_id': self.distributed_config.node_id,
            'is_leader': self.cluster_manager.is_leader(),
            'leader': self.cluster_manager.get_leader(),
            'cluster_health': self.cluster_manager.get_cluster_health(),
            'layer_locations': {
                layer: list(nodes)
                for layer, nodes in self.layer_locations.items()
            },
            'replication_factor': self.distributed_config.replication_factor
        }


def create_distributed_cluster(
    cluster_name: str,
    num_nodes: int = 3,
    backend: str = "redis",
    **kwargs
) -> List[DistributedLayeredGraph]:
    """
    Create a distributed LCG cluster with multiple nodes.
    
    Args:
        cluster_name: Name of the cluster
        num_nodes: Number of nodes in the cluster
        backend: Backend for coordination (redis, etcd, consul)
        **kwargs: Additional config options
        
    Returns:
        List of DistributedLayeredGraph instances (one per node)
    """
    nodes = []
    
    for i in range(num_nodes):
        config = DistributedConfig(
            cluster_name=cluster_name,
            node_id=f"node_{i+1}",
            backend=backend,
            **kwargs
        )
        
        dlcg = DistributedLayeredGraph(
            name=f"{cluster_name}_lcg",
            distributed_config=config
        )
        
        nodes.append(dlcg)
    
    logger.info(f"Created {num_nodes}-node cluster: {cluster_name}")
    return nodes
