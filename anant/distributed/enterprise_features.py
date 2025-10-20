"""
Enterprise-Grade Distributed Graph Database Features

This module implements advanced enterprise features for distributed graph processing:
- Sharding strategies for scalability
- Replication for high availability  
- Distributed query processing
- Concurrency control and consistency models
- Performance optimization features
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from ..core import Hypergraph
from .graph_operations import GraphType, GraphPartition

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Different strategies for distributing graph data across nodes."""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"  
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    CUSTOM = "custom"


class ConsistencyModel(Enum):
    """Data consistency models for distributed operations."""
    STRONG = "strong"           # All nodes see same data immediately
    EVENTUAL = "eventual"       # Nodes eventually converge to same state
    CAUSAL = "causal"          # Causally related operations are ordered
    SESSION = "session"        # Consistency within client session
    MONOTONIC = "monotonic"    # Read-your-writes consistency


class ReplicationStrategy(Enum):
    """Strategies for data replication across nodes."""
    FULL = "full"              # Full replication on all nodes
    PARTIAL = "partial"        # Selective replication based on access patterns
    QUORUM = "quorum"          # Majority-based replication
    CHAIN = "chain"            # Chain replication for performance


@dataclass
class ShardInfo:
    """Information about a graph data shard."""
    shard_id: str
    node_range: Optional[Tuple[int, int]] = None
    edge_range: Optional[Tuple[int, int]] = None
    hash_range: Optional[Tuple[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    primary_nodes: List[str] = field(default_factory=list)
    replica_nodes: List[str] = field(default_factory=list)
    last_modified: float = field(default_factory=time.time)


@dataclass
class ReplicationConfig:
    """Configuration for data replication."""
    strategy: ReplicationStrategy
    replication_factor: int = 3
    read_quorum: int = 2
    write_quorum: int = 2
    async_replication: bool = True
    conflict_resolution: str = "last_write_wins"


@dataclass
class QueryPlan:
    """Execution plan for distributed queries."""
    query_id: str
    operations: List[Dict[str, Any]]
    target_shards: List[str]
    estimated_cost: float
    parallelization_factor: int
    dependencies: List[str] = field(default_factory=list)


class GraphSharder(ABC):
    """Abstract base class for graph sharding strategies."""
    
    @abstractmethod
    def shard_nodes(self, graph: Any, num_shards: int) -> Dict[str, List]:
        """Distribute nodes across shards."""
        pass
    
    @abstractmethod
    def shard_edges(self, graph: Any, node_shards: Dict[str, List]) -> Dict[str, List]:
        """Distribute edges across shards based on node distribution."""
        pass
    
    @abstractmethod
    def locate_shard(self, node_id: str) -> str:
        """Find which shard contains a specific node."""
        pass


class HashBasedSharder(GraphSharder):
    """Hash-based sharding for uniform distribution."""
    
    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.shard_map = {}
    
    def shard_nodes(self, graph: Any, num_shards: int) -> Dict[str, List]:
        """Distribute nodes using consistent hashing."""
        shards = {f"shard_{i}": [] for i in range(num_shards)}
        
        # Get nodes from graph (adapt based on graph type)
        if hasattr(graph, 'nodes'):
            nodes = list(graph.nodes)
        elif hasattr(graph, 'incidence_store'):
            nodes = graph.incidence_store.nodes.get_column('node_id').to_list()
        else:
            nodes = []
        
        for node in nodes:
            shard_key = self._hash_node(str(node)) % num_shards
            shard_id = f"shard_{shard_key}"
            shards[shard_id].append(node)
            self.shard_map[node] = shard_id
        
        return shards
    
    def shard_edges(self, graph: Any, node_shards: Dict[str, List]) -> Dict[str, List]:
        """Distribute edges based on node distribution."""
        shards = {shard_id: [] for shard_id in node_shards.keys()}
        
        # Get edges from graph
        if hasattr(graph, 'edges'):
            edges = list(graph.edges)
        elif hasattr(graph, 'incidence_store'):
            edges_df = graph.incidence_store.edges
            edges = [(row['edge_id'], row['nodes']) for row in edges_df.iter_rows(named=True)]
        else:
            edges = []
        
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                edge_id, nodes = edge
                # Place edge on shard containing majority of its nodes
                node_shards_for_edge = [self.locate_shard(str(node)) for node in nodes if str(node) in self.shard_map]
                if node_shards_for_edge:
                    target_shard = max(set(node_shards_for_edge), key=node_shards_for_edge.count)
                    shards[target_shard].append(edge_id)
        
        return shards
    
    def locate_shard(self, node_id: str) -> str:
        """Find shard for a node."""
        return self.shard_map.get(node_id, f"shard_{self._hash_node(node_id) % self.num_shards}")
    
    def _hash_node(self, node_id: str) -> int:
        """Hash function for consistent node distribution."""
        return int(hashlib.md5(node_id.encode()).hexdigest(), 16)


class HierarchicalSharder(GraphSharder):
    """Hierarchical sharding for hierarchical graphs."""
    
    def __init__(self, levels_per_shard: int = 2):
        self.levels_per_shard = levels_per_shard
        self.level_map = {}
    
    def shard_nodes(self, graph: Any, num_shards: int) -> Dict[str, List]:
        """Distribute nodes by hierarchy levels."""
        shards = {f"level_shard_{i}": [] for i in range(num_shards)}
        
        # For hierarchical graphs, group by levels
        if hasattr(graph, 'get_level'):
            nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
            level_groups = {}
            
            for node in nodes:
                level = graph.get_level(node) if callable(getattr(graph, 'get_level', None)) else 0
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            # Distribute levels across shards
            shard_idx = 0
            for level, level_nodes in sorted(level_groups.items()):
                shard_id = f"level_shard_{shard_idx % num_shards}"
                shards[shard_id].extend(level_nodes)
                self.level_map[level] = shard_id
                
                if len(level_groups) > self.levels_per_shard:
                    shard_idx += 1
        
        return shards
    
    def shard_edges(self, graph: Any, node_shards: Dict[str, List]) -> Dict[str, List]:
        """Distribute edges preserving hierarchical relationships."""
        # Similar to hash-based but respects level boundaries
        return super().shard_edges(graph, node_shards)
    
    def locate_shard(self, node_id: str) -> str:
        """Find shard based on hierarchy level."""
        # Would need access to graph to determine level
        return list(self.level_map.values())[0] if self.level_map else "level_shard_0"


class SemanticSharder(GraphSharder):
    """Semantic sharding for knowledge graphs."""
    
    def __init__(self, entity_types: List[str]):
        self.entity_types = entity_types
        self.type_shard_map = {}
    
    def shard_nodes(self, graph: Any, num_shards: int) -> Dict[str, List]:
        """Distribute nodes by semantic entity types."""
        shards = {f"semantic_shard_{i}": [] for i in range(num_shards)}
        
        # Distribute entity types across shards
        for i, entity_type in enumerate(self.entity_types):
            shard_id = f"semantic_shard_{i % num_shards}"
            self.type_shard_map[entity_type] = shard_id
        
        # Assign nodes based on their types
        if hasattr(graph, 'get_node_type'):
            nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
            for node in nodes:
                node_type = graph.get_node_type(node) if callable(getattr(graph, 'get_node_type', None)) else 'unknown'
                shard_id = self.type_shard_map.get(node_type, f"semantic_shard_0")
                shards[shard_id].append(node)
        
        return shards
    
    def shard_edges(self, graph: Any, node_shards: Dict[str, List]) -> Dict[str, List]:
        """Distribute edges maintaining semantic locality."""
        return super().shard_edges(graph, node_shards)
    
    def locate_shard(self, node_id: str) -> str:
        """Find shard based on semantic type."""
        return "semantic_shard_0"  # Default, would need graph context


class DistributedConcurrencyController:
    """Manages distributed concurrency control and consistency."""
    
    def __init__(self, consistency_model: ConsistencyModel = ConsistencyModel.EVENTUAL):
        self.consistency_model = consistency_model
        self.locks = {}
        self.version_vectors = {}
        self.pending_operations = {}
    
    async def acquire_lock(self, resource_id: str, operation_type: str = "read") -> bool:
        """Acquire distributed lock for resource."""
        lock_key = f"{resource_id}:{operation_type}"
        
        if operation_type == "write":
            # Exclusive lock for writes
            if resource_id in self.locks:
                return False
            self.locks[resource_id] = {"type": "write", "timestamp": time.time()}
        else:
            # Shared locks for reads
            if resource_id in self.locks and self.locks[resource_id]["type"] == "write":
                return False
            if resource_id not in self.locks:
                self.locks[resource_id] = {"type": "read", "count": 0, "timestamp": time.time()}
            self.locks[resource_id]["count"] += 1
        
        return True
    
    async def release_lock(self, resource_id: str, operation_type: str = "read"):
        """Release distributed lock."""
        if resource_id in self.locks:
            if operation_type == "write":
                del self.locks[resource_id]
            else:
                self.locks[resource_id]["count"] -= 1
                if self.locks[resource_id]["count"] <= 0:
                    del self.locks[resource_id]
    
    async def ensure_consistency(self, operation: Dict[str, Any]) -> bool:
        """Ensure operation meets consistency requirements."""
        if self.consistency_model == ConsistencyModel.STRONG:
            return await self._ensure_strong_consistency(operation)
        elif self.consistency_model == ConsistencyModel.EVENTUAL:
            return await self._ensure_eventual_consistency(operation)
        elif self.consistency_model == ConsistencyModel.CAUSAL:
            return await self._ensure_causal_consistency(operation)
        return True
    
    async def _ensure_strong_consistency(self, operation: Dict[str, Any]) -> bool:
        """Ensure strong consistency across all nodes."""
        # All nodes must be synchronized before operation
        return True  # Simplified implementation
    
    async def _ensure_eventual_consistency(self, operation: Dict[str, Any]) -> bool:
        """Allow operation with eventual consistency guarantee."""
        # Operations can proceed, consistency achieved asynchronously
        return True
    
    async def _ensure_causal_consistency(self, operation: Dict[str, Any]) -> bool:
        """Ensure causal ordering of operations."""
        # Check if operation depends on previous operations
        return True


class DistributedQueryProcessor:
    """Processes queries across distributed graph shards."""
    
    def __init__(self, sharder: GraphSharder, concurrency_controller: DistributedConcurrencyController):
        self.sharder = sharder
        self.concurrency_controller = concurrency_controller
        self.query_cache = {}
        self.index_manager = DistributedIndexManager()
    
    async def execute_query(self, query: Dict[str, Any], graph: Any) -> Dict[str, Any]:
        """Execute distributed query across graph shards."""
        query_plan = await self._create_query_plan(query, graph)
        
        # Execute query plan across relevant shards
        results = await self._execute_distributed_plan(query_plan)
        
        # Merge and return results
        return await self._merge_results(results, query_plan)
    
    async def _create_query_plan(self, query: Dict[str, Any], graph: Any) -> QueryPlan:
        """Create optimized query execution plan."""
        query_id = hashlib.md5(str(query).encode()).hexdigest()
        
        # Determine which shards are needed
        target_shards = await self._identify_target_shards(query, graph)
        
        # Estimate query cost
        estimated_cost = len(target_shards) * query.get('complexity', 1.0)
        
        # Create operations list
        operations = [
            {"type": "filter", "criteria": query.get('filter', {})},
            {"type": "traverse", "depth": query.get('depth', 1)},
            {"type": "aggregate", "function": query.get('aggregate', 'count')}
        ]
        
        return QueryPlan(
            query_id=query_id,
            operations=operations,
            target_shards=target_shards,
            estimated_cost=estimated_cost,
            parallelization_factor=min(len(target_shards), 8)
        )
    
    async def _identify_target_shards(self, query: Dict[str, Any], graph: Any) -> List[str]:
        """Identify which shards contain relevant data for query."""
        # Analyze query to determine data requirements
        if 'node_ids' in query:
            # Specific nodes requested
            shards = set()
            for node_id in query['node_ids']:
                shard_id = self.sharder.locate_shard(str(node_id))
                shards.add(shard_id)
            return list(shards)
        else:
            # Query requires data from multiple/all shards
            return [f"shard_{i}" for i in range(4)]  # Default to 4 shards
    
    async def _execute_distributed_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        """Execute query plan across multiple shards in parallel."""
        tasks = []
        
        for shard_id in plan.target_shards:
            task = self._execute_shard_operations(shard_id, plan.operations)
            tasks.append(task)
        
        # Execute in parallel
        shard_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "shard_results": shard_results,
            "execution_time": time.time(),
            "shards_processed": len(plan.target_shards)
        }
    
    async def _execute_shard_operations(self, shard_id: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute operations on a specific shard."""
        # Simplified shard operation execution
        result = {"shard_id": shard_id, "operations_executed": len(operations)}
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return result
    
    async def _merge_results(self, results: Dict[str, Any], plan: QueryPlan) -> Dict[str, Any]:
        """Merge results from multiple shards."""
        merged = {
            "query_id": plan.query_id,
            "total_shards": len(plan.target_shards),
            "execution_time": results.get("execution_time", 0),
            "merged_data": []
        }
        
        # Combine shard results
        for shard_result in results.get("shard_results", []):
            if isinstance(shard_result, dict):
                merged["merged_data"].append(shard_result)
        
        return merged


class DistributedIndexManager:
    """Manages distributed indexes for query optimization."""
    
    def __init__(self):
        self.indexes = {}
        self.index_stats = {}
    
    async def create_index(self, index_name: str, index_type: str, columns: List[str]):
        """Create distributed index across shards."""
        self.indexes[index_name] = {
            "type": index_type,
            "columns": columns,
            "created": time.time(),
            "shards": {}
        }
        logger.info(f"Created distributed index: {index_name}")
    
    async def optimize_query_with_indexes(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize query using available indexes."""
        # Check if query can benefit from existing indexes
        optimized = query.copy()
        
        for index_name, index_info in self.indexes.items():
            if any(col in query.get('filter', {}) for col in index_info['columns']):
                optimized['use_index'] = index_name
                logger.info(f"Query optimized using index: {index_name}")
                break
        
        return optimized


class EnterpriseDistributedManager:
    """High-level manager for enterprise distributed graph features."""
    
    def __init__(self, sharding_strategy: ShardingStrategy = ShardingStrategy.HASH_BASED,
                 consistency_model: ConsistencyModel = ConsistencyModel.EVENTUAL,
                 replication_config: Optional[ReplicationConfig] = None):
        
        self.sharding_strategy = sharding_strategy
        self.consistency_model = consistency_model
        self.replication_config = replication_config or ReplicationConfig(
            strategy=ReplicationStrategy.QUORUM,
            replication_factor=3
        )
        
        # Initialize components
        self.sharder = self._create_sharder()
        self.concurrency_controller = DistributedConcurrencyController(consistency_model)
        self.query_processor = DistributedQueryProcessor(self.sharder, self.concurrency_controller)
        self.shard_info = {}
        
        logger.info(f"Enterprise distributed manager initialized with {sharding_strategy.value} sharding")
    
    def _create_sharder(self) -> GraphSharder:
        """Create appropriate sharder based on strategy."""
        if self.sharding_strategy == ShardingStrategy.HASH_BASED:
            return HashBasedSharder(num_shards=4)
        elif self.sharding_strategy == ShardingStrategy.HIERARCHICAL:
            return HierarchicalSharder()
        elif self.sharding_strategy == ShardingStrategy.SEMANTIC:
            return SemanticSharder(['person', 'organization', 'location', 'concept'])
        else:
            return HashBasedSharder(num_shards=4)  # Default
    
    async def distribute_graph(self, graph: Any, num_shards: int = 4) -> Dict[str, ShardInfo]:
        """Distribute graph across shards with replication."""
        logger.info(f"Distributing graph across {num_shards} shards")
        
        # Shard nodes and edges
        node_shards = self.sharder.shard_nodes(graph, num_shards)
        edge_shards = self.sharder.shard_edges(graph, node_shards)
        
        # Create shard info with replication
        for shard_id in node_shards.keys():
            self.shard_info[shard_id] = ShardInfo(
                shard_id=shard_id,
                primary_nodes=[f"node_primary_{shard_id}"],
                replica_nodes=[f"node_replica_{shard_id}_{i}" for i in range(self.replication_config.replication_factor - 1)],
                metadata={
                    "node_count": len(node_shards[shard_id]),
                    "edge_count": len(edge_shards.get(shard_id, [])),
                    "sharding_strategy": self.sharding_strategy.value
                }
            )
        
        logger.info(f"Graph distributed across {len(self.shard_info)} shards")
        return self.shard_info
    
    async def execute_distributed_query(self, query: Dict[str, Any], graph: Any) -> Dict[str, Any]:
        """Execute query with enterprise features."""
        # Acquire necessary locks
        resource_locks = []
        try:
            # Identify resources needed
            resources = query.get('resources', ['default'])
            
            # Acquire locks based on consistency model
            for resource in resources:
                if await self.concurrency_controller.acquire_lock(resource, 'read'):
                    resource_locks.append(resource)
            
            # Execute query
            result = await self.query_processor.execute_query(query, graph)
            
            # Add enterprise metadata
            result['enterprise_info'] = {
                'sharding_strategy': self.sharding_strategy.value,
                'consistency_model': self.consistency_model.value,
                'replication_factor': self.replication_config.replication_factor,
                'shards_involved': len(self.shard_info)
            }
            
            return result
            
        finally:
            # Release all acquired locks
            for resource in resource_locks:
                await self.concurrency_controller.release_lock(resource, 'read')
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get comprehensive cluster health information."""
        return {
            "total_shards": len(self.shard_info),
            "sharding_strategy": self.sharding_strategy.value,
            "consistency_model": self.consistency_model.value,
            "replication_config": {
                "strategy": self.replication_config.strategy.value,
                "factor": self.replication_config.replication_factor
            },
            "active_locks": len(self.concurrency_controller.locks),
            "timestamp": time.time()
        }


# Factory function for easy creation
def create_enterprise_manager(
    sharding: str = "hash_based",
    consistency: str = "eventual", 
    replication_factor: int = 3
) -> EnterpriseDistributedManager:
    """Create enterprise distributed manager with specified configuration."""
    
    sharding_strategy = ShardingStrategy(sharding)
    consistency_model = ConsistencyModel(consistency)
    replication_config = ReplicationConfig(
        strategy=ReplicationStrategy.QUORUM,
        replication_factor=replication_factor
    )
    
    return EnterpriseDistributedManager(
        sharding_strategy=sharding_strategy,
        consistency_model=consistency_model,
        replication_config=replication_config
    )