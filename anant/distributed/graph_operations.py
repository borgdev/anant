"""
Multi-Graph Distributed Operations
==================================

Unified distributed computing framework supporting all four graph types:
- Hypergraph: Traditional hypergraph operations with METIS partitioning
- Metagraph: Enterprise hierarchical knowledge management
- KnowledgeGraph: Semantic hypergraph with reasoning
- HierarchicalKnowledgeGraph: Multi-level knowledge organization

This module provides production-grade graph-aware partitioning using METIS, KaHiP,
and other industry-standard algorithms, with automatic algorithm selection and
performance optimization for each graph type's unique characteristics.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Union, Type, Protocol, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from pathlib import Path

from ..classes.hypergraph import Hypergraph
from ..classes.incidence_store import IncidenceStore
from ..kg.core import KnowledgeGraph, SemanticHypergraph
from ..kg.hierarchical import HierarchicalKnowledgeGraph
from ..metagraph.core.metagraph import Metagraph

from .cluster_manager import ClusterManager
from .task_scheduler import TaskScheduler, Task
from .worker_node import WorkerNode
from .message_broker import MessageBroker
from .partitioning import (
    ProductionPartitioner, PartitioningConfig, PartitioningAlgorithm,
    PartitioningObjective, PartitioningResult, create_production_partitioner
)

# Import graph-specific operations
try:
    from .graph_specific_ops import GraphOperationsFactory
except ImportError:
    # Handle circular import if needed
    GraphOperationsFactory = None

logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Enumeration of supported graph types"""
    HYPERGRAPH = "hypergraph"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HIERARCHICAL_KNOWLEDGE_GRAPH = "hierarchical_knowledge_graph"
    METAGRAPH = "metagraph"


@dataclass
class GraphPartition:
    """Represents a partition of a graph for distributed processing"""
    partition_id: str
    graph_type: GraphType
    node_ids: List[str]
    edge_ids: List[str]
    data: Union[pl.DataFrame, Dict[str, Any]]
    metadata: Dict[str, Any]
    size_estimate: int
    dependencies: Optional[List[str]] = None  # Other partitions this depends on


@dataclass
class DistributedGraphResult:
    """Result from distributed graph operation"""
    operation_id: str
    graph_type: GraphType
    result_data: Any
    partition_results: Dict[str, Any]
    execution_time: float
    nodes_processed: int
    edges_processed: int
    success: bool
    error_message: Optional[str] = None


class GraphPartitioner(ABC):
    """Abstract base class for graph partitioning strategies"""
    
    @abstractmethod
    def partition(self, graph: Any, num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Partition graph into specified number of partitions"""
        pass
    
    @abstractmethod
    def estimate_partition_cost(self, partition: GraphPartition) -> float:
        """Estimate computational cost of processing this partition"""
        pass


class HypergraphPartitioner(GraphPartitioner):
    """Production-grade partitioner for traditional hypergraphs using METIS/KaHiP."""
    
    def __init__(self):
        self.production_partitioner = create_production_partitioner(
            algorithm="auto",  # Auto-select best available algorithm
            objective="min_edge_cut"
        )
    
    def partition(self, graph: Hypergraph, num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Partition hypergraph using production-grade algorithms (METIS, KaHiP, etc.)."""
        logger.info(f"Partitioning Hypergraph with {graph.num_nodes} nodes and {graph.num_edges} edges using production algorithms")
        
        # Configure partitioning
        config = PartitioningConfig(
            algorithm=PartitioningAlgorithm.AUTO,
            objective=PartitioningObjective.MIN_EDGE_CUT,
            num_partitions=num_partitions,
            imbalance_tolerance=kwargs.get('imbalance_tolerance', 0.03),
            seed=kwargs.get('seed', None)
        )
        
        # Perform production-grade partitioning
        partitioning_result = self.production_partitioner.partition(graph, config)
        
        if not partitioning_result.success:
            logger.warning(f"Production partitioning failed: {partitioning_result.error_message}")
            return self._fallback_partition(graph, num_partitions)
        
        # Convert PartitioningResult to GraphPartitions
        partitions = []
        partition_assignment = partitioning_result.partition_assignment
        
        # Get graph data
        incidence_data = graph.incidences.data
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        
        for partition_id in range(num_partitions):
            # Find nodes in this partition
            partition_nodes = [nodes[i] for i, part in enumerate(partition_assignment) if part == partition_id]
            
            if not partition_nodes:
                continue
            
            # Find edges that connect to these nodes
            partition_edge_mask = incidence_data.filter(
                pl.col("node_id").is_in(partition_nodes)
            ).select("edge_id").unique()
            
            partition_edges = partition_edge_mask.to_series().to_list()
            
            # Extract partition data
            partition_data = incidence_data.filter(
                pl.col("edge_id").is_in(partition_edges)
            )
            
            partition = GraphPartition(
                partition_id=f"hypergraph_metis_partition_{partition_id}",
                graph_type=GraphType.HYPERGRAPH,
                node_ids=partition_nodes,
                edge_ids=partition_edges,
                data=partition_data,
                metadata={
                    "partition_idx": partition_id,
                    "algorithm": partitioning_result.algorithm_used.value,
                    "edge_cut": partitioning_result.edge_cut,
                    "quality_metrics": partitioning_result.quality_metrics,
                    "execution_time": partitioning_result.execution_time
                },
                size_estimate=len(partition_nodes) + len(partition_edges)
            )
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} hypergraph partitions using {partitioning_result.algorithm_used.value}")
        logger.info(f"Partitioning quality - Edge cut: {partitioning_result.edge_cut}, "
                   f"Load balance: {partitioning_result.quality_metrics.get('load_balance', 0):.3f}")
        
        return partitions
    
    def _fallback_partition(self, graph: Hypergraph, num_partitions: int) -> List[GraphPartition]:
        """Fallback to simple partitioning if production algorithms fail."""
        logger.info("Using fallback round-robin partitioning")
        
        # Get graph data
        incidence_data = graph.incidences.data
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        
        partitions = []
        nodes_per_partition = len(nodes) // num_partitions + 1
        
        for i in range(num_partitions):
            start_idx = i * nodes_per_partition
            end_idx = min((i + 1) * nodes_per_partition, len(nodes))
            
            if start_idx >= len(nodes):
                break
                
            partition_nodes = nodes[start_idx:end_idx]
            
            # Find edges that connect to these nodes
            partition_edge_mask = incidence_data.filter(
                pl.col("node_id").is_in(partition_nodes)
            ).select("edge_id").unique()
            
            partition_edges = partition_edge_mask.to_series().to_list()
            
            # Extract partition data
            partition_data = incidence_data.filter(
                pl.col("edge_id").is_in(partition_edges)
            )
            
            partition = GraphPartition(
                partition_id=f"hypergraph_fallback_partition_{i}",
                graph_type=GraphType.HYPERGRAPH,
                node_ids=partition_nodes,
                edge_ids=partition_edges,
                data=partition_data,
                metadata={"partition_idx": i, "algorithm": "round_robin_fallback"},
                size_estimate=len(partition_nodes) + len(partition_edges)
            )
            partitions.append(partition)
        
        return partitions
    
    def estimate_partition_cost(self, partition: GraphPartition) -> float:
        """Estimate cost based on nodes and edges count with algorithm quality."""
        base_cost = len(partition.node_ids) * 1.0 + len(partition.edge_ids) * 2.0
        
        # Adjust cost based on partitioning quality
        quality_metrics = partition.metadata.get('quality_metrics', {})
        quality_factor = quality_metrics.get('efficiency', 1.0)
        
        return base_cost / max(quality_factor, 0.1)  # Better quality = lower effective cost


class KnowledgeGraphPartitioner(GraphPartitioner):
    """Production-grade partitioner for semantic knowledge graphs."""
    
    def __init__(self):
        self.production_partitioner = create_production_partitioner(
            algorithm="auto",
            objective="min_edge_cut"  # Could be "min_communication" for semantic locality
        )
    
    def partition(self, graph: KnowledgeGraph, num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Partition knowledge graph with semantic awareness and production algorithms."""
        logger.info(f"Partitioning KnowledgeGraph with semantic awareness using production algorithms")
        
        # Check if graph has entity types for semantic partitioning
        try:
            entity_types = getattr(graph, 'entity_types', [])
            if callable(entity_types):
                entity_types = entity_types()
            if not isinstance(entity_types, list):
                entity_types = []
        except:
            entity_types = []
        
        # Decide partitioning strategy
        if entity_types and len(entity_types) >= num_partitions:
            # Use semantic partitioning by entity types
            return self._semantic_partition_by_entity_types(graph, entity_types, num_partitions, **kwargs)
        else:
            # Use production-grade general partitioning with semantic objective
            return self._production_partition_with_semantic_objective(graph, num_partitions, **kwargs)
    
    def _production_partition_with_semantic_objective(self, graph: KnowledgeGraph, 
                                                    num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Use production algorithms with semantic-aware objectives."""
        
        # Configure for semantic graphs - prefer communication minimization
        config = PartitioningConfig(
            algorithm=PartitioningAlgorithm.AUTO,
            objective=PartitioningObjective.MIN_COMMUNICATION,  # Better for semantic locality
            num_partitions=num_partitions,
            imbalance_tolerance=kwargs.get('imbalance_tolerance', 0.05),  # Slightly more tolerance for semantic balance
            seed=kwargs.get('seed', None)
        )
        
        # Perform partitioning
        partitioning_result = self.production_partitioner.partition(graph, config)
        
        if not partitioning_result.success:
            logger.warning(f"Production partitioning failed: {partitioning_result.error_message}")
            # Fall back to hypergraph partitioning
            hypergraph_partitioner = HypergraphPartitioner()
            partitions = hypergraph_partitioner.partition(graph, num_partitions)
            # Update partition type and metadata
            for partition in partitions:
                partition.graph_type = GraphType.KNOWLEDGE_GRAPH
                partition.partition_id = partition.partition_id.replace("hypergraph", "knowledge_graph_fallback")
            return partitions
        
        # Convert to GraphPartitions with semantic metadata
        partitions = []
        partition_assignment = partitioning_result.partition_assignment
        
        # Get graph data (assume similar structure to hypergraph)
        if hasattr(graph, 'incidences') and hasattr(graph.incidences, 'data'):
            incidence_data = graph.incidences.data
            nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
            edges = list(graph.edges) if hasattr(graph, 'edges') else []
        else:
            logger.warning("Knowledge graph structure not recognized, creating empty partitions")
            return []
        
        for partition_id in range(num_partitions):
            # Find nodes in this partition
            partition_nodes = [nodes[i] for i, part in enumerate(partition_assignment) if part == partition_id]
            
            if not partition_nodes:
                continue
            
            # Find edges for this partition
            partition_edge_mask = incidence_data.filter(
                pl.col("node_id").is_in(partition_nodes)
            ).select("edge_id").unique()
            
            partition_edges = partition_edge_mask.to_series().to_list()
            
            # Extract partition data
            partition_data = incidence_data.filter(
                pl.col("edge_id").is_in(partition_edges)
            )
            
            # Add semantic metadata
            semantic_metadata = {
                "partition_idx": partition_id,
                "algorithm": partitioning_result.algorithm_used.value,
                "partitioning_objective": "semantic_locality",
                "communication_volume": partitioning_result.communication_volume,
                "quality_metrics": partitioning_result.quality_metrics,
                "execution_time": partitioning_result.execution_time
            }
            
            partition = GraphPartition(
                partition_id=f"knowledge_graph_semantic_partition_{partition_id}",
                graph_type=GraphType.KNOWLEDGE_GRAPH,
                node_ids=partition_nodes,
                edge_ids=partition_edges,
                data=partition_data,
                metadata=semantic_metadata,
                size_estimate=len(partition_nodes) + len(partition_edges)
            )
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} semantic knowledge graph partitions using {partitioning_result.algorithm_used.value}")
        logger.info(f"Semantic partitioning quality - Communication volume: {partitioning_result.communication_volume}, "
                   f"Load balance: {partitioning_result.quality_metrics.get('load_balance', 0):.3f}")
        
        return partitions
    
    def _semantic_partition_by_entity_types(self, graph: KnowledgeGraph, entity_types: List[str], 
                                          num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Partition by grouping similar entity types with production quality."""
        logger.info(f"Using entity-type based semantic partitioning for {len(entity_types)} types")
        
        partitions = []
        types_per_partition = len(entity_types) // num_partitions + 1
        
        for i in range(num_partitions):
            start_idx = i * types_per_partition
            end_idx = min((i + 1) * types_per_partition, len(entity_types))
            
            if start_idx >= len(entity_types):
                break
            
            partition_types = entity_types[start_idx:end_idx]
            
            # Find nodes of these entity types
            partition_nodes = []
            if hasattr(graph, 'get_nodes_by_type'):
                for entity_type in partition_types:
                    try:
                        nodes_of_type = graph.get_nodes_by_type(entity_type)
                        if isinstance(nodes_of_type, list):
                            partition_nodes.extend(nodes_of_type)
                    except:
                        continue
            
            if not partition_nodes:
                # Fallback to simple partitioning
                all_nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
                nodes_per_partition = len(all_nodes) // num_partitions + 1
                start_node_idx = i * nodes_per_partition
                end_node_idx = min((i + 1) * nodes_per_partition, len(all_nodes))
                partition_nodes = all_nodes[start_node_idx:end_node_idx]
            
            if hasattr(graph, 'incidences') and hasattr(graph.incidences, 'data'):
                incidence_data = graph.incidences.data
                
                # Find edges for these nodes
                if partition_nodes:
                    partition_edge_mask = incidence_data.filter(
                        pl.col("node_id").is_in(partition_nodes)
                    ).select("edge_id").unique()
                    
                    partition_edges = partition_edge_mask.to_series().to_list()
                    
                    # Extract partition data
                    partition_data = incidence_data.filter(
                        pl.col("edge_id").is_in(partition_edges)
                    )
                else:
                    partition_edges = []
                    partition_data = incidence_data.filter(pl.lit(False))  # Empty dataframe
            else:
                partition_edges = []
                partition_data = pl.DataFrame()  # Empty dataframe
            
            semantic_metadata = {
                "partition_idx": i,
                "algorithm": "entity_type_semantic",
                "entity_types": partition_types,
                "semantic_locality": "high",
                "partitioning_strategy": "entity_type_grouping"
            }
            
            partition = GraphPartition(
                partition_id=f"knowledge_graph_entity_partition_{i}",
                graph_type=GraphType.KNOWLEDGE_GRAPH,
                node_ids=partition_nodes,
                edge_ids=partition_edges,
                data=partition_data,
                metadata=semantic_metadata,
                size_estimate=len(partition_nodes) + len(partition_edges)
            )
            partitions.append(partition)
        
        
        logger.info(f"Created {len(partitions)} entity-type based semantic partitions")
        return partitions

    @performance_monitor("hierarchical_partitioning")
    def estimate_partition_cost(self, partition: GraphPartition) -> float:
        """Estimate cost considering semantic complexity"""
        base_cost = len(partition.node_ids) * 1.5 + len(partition.edge_ids) * 3.0
        # Add semantic processing overhead
        return base_cost * 1.2


class HierarchicalKnowledgeGraphPartitioner(GraphPartitioner):
    """Partitioner for hierarchical knowledge graphs"""
    
    def partition(self, graph: HierarchicalKnowledgeGraph, num_partitions: int, 
                 **kwargs) -> List[GraphPartition]:
        """Partition hierarchical knowledge graph by levels"""
        logger.info(f"Partitioning HierarchicalKnowledgeGraph with {len(graph.levels)} levels")
        
        partitions = []
        
        # Strategy 1: Partition by hierarchy levels
        if len(graph.levels) >= num_partitions:
            partitions = self._partition_by_levels(graph, num_partitions)
        else:
            # Strategy 2: Partition individual levels if few levels
            partitions = self._partition_within_levels(graph, num_partitions)
        
        return partitions
    
    def _partition_by_levels(self, graph: HierarchicalKnowledgeGraph, 
                           num_partitions: int) -> List[GraphPartition]:
        """Partition by assigning levels to partitions"""
        partitions = []
        levels = list(graph.levels.keys())
        levels_per_partition = len(levels) // num_partitions + 1
        
        for i in range(num_partitions):
            start_idx = i * levels_per_partition
            end_idx = min((i + 1) * levels_per_partition, len(levels))
            
            if start_idx >= len(levels):
                break
                
            partition_levels = levels[start_idx:end_idx]
            
            # Collect all entities and relationships from these levels
            partition_nodes = []
            partition_edges = []
            
            for level_id in partition_levels:
                level_kg = graph.get_level_subgraph(level_id)
                if level_kg:
                    partition_nodes.extend(level_kg.nodes)
                    partition_edges.extend(level_kg.edges)
            
            # Create combined data
            combined_data = None
            for level_id in partition_levels:
                level_kg = graph.get_level_subgraph(level_id)
                if level_kg:
                    level_data = level_kg.incidences.data
                    if combined_data is None:
                        combined_data = level_data
                    else:
                        combined_data = pl.concat([combined_data, level_data])
            
            if combined_data is None:
                combined_data = pl.DataFrame({
                    'edge_id': [], 'node_id': [], 'weight': []
                }).with_columns([
                    pl.col('edge_id').cast(pl.Utf8),
                    pl.col('node_id').cast(pl.Utf8),
                    pl.col('weight').cast(pl.Float64)
                ])
            
            partition = GraphPartition(
                partition_id=f"hierarchical_kg_partition_{i}",
                graph_type=GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH,
                node_ids=partition_nodes,
                edge_ids=partition_edges,
                data=combined_data,
                metadata={
                    "partition_idx": i,
                    "strategy": "level_based",
                    "levels": partition_levels
                },
                size_estimate=len(partition_nodes) + len(partition_edges)
            )
            partitions.append(partition)
        
        return partitions
    
    def _partition_within_levels(self, graph: HierarchicalKnowledgeGraph, 
                               num_partitions: int) -> List[GraphPartition]:
        """Partition by splitting larger levels internally"""
        partitions = []
        partitions_per_level = num_partitions // len(graph.levels) + 1
        partition_idx = 0
        
        for level_id, level_info in graph.levels.items():
            level_kg = graph.get_level_subgraph(level_id)
            if not level_kg:
                continue
                
            # Use knowledge graph partitioner for this level
            kg_partitioner = KnowledgeGraphPartitioner()
            level_partitions = kg_partitioner.partition(level_kg, partitions_per_level)
            
            # Update partition metadata
            for partition in level_partitions:
                partition.partition_id = f"hierarchical_kg_partition_{partition_idx}"
                partition.graph_type = GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH
                partition.metadata["level_id"] = level_id
                partition.metadata["level_name"] = level_info.get("name", level_id)
                partitions.append(partition)
                partition_idx += 1
        
        return partitions[:num_partitions]  # Limit to requested number
    
    def estimate_partition_cost(self, partition: GraphPartition) -> float:
        """Estimate cost considering hierarchical complexity"""
        base_cost = len(partition.node_ids) * 2.0 + len(partition.edge_ids) * 4.0
        # Add hierarchical navigation overhead
        return base_cost * 1.5


class MetagraphPartitioner(GraphPartitioner):
    """Partitioner for enterprise metagraphs"""
    
    def partition(self, graph: Metagraph, num_partitions: int, **kwargs) -> List[GraphPartition]:
        """Partition metagraph considering enterprise structure"""
        logger.info(f"Partitioning Metagraph with enterprise considerations")
        
        # Strategy: Partition by organizational hierarchy levels
        partitions = []
        
        # Simple fallback partitioning for metagraph
        try:
            # Try to get available levels
            available_levels = getattr(graph, 'get_available_levels', lambda: [])()
            if not available_levels:
                available_levels = []
        except:
            available_levels = []
        
        if available_levels and len(available_levels) > 0:
            partitions = self._partition_by_enterprise_levels(graph, available_levels, num_partitions)
        else:
            # Fall back to entity-based partitioning
            partitions = self._partition_by_entities(graph, num_partitions)
        
        return partitions
    
    def _partition_by_enterprise_levels(self, graph: Metagraph, levels: List[str], 
                                      num_partitions: int) -> List[GraphPartition]:
        """Partition by enterprise hierarchy levels"""
        partitions = []
        levels_per_partition = len(levels) // num_partitions + 1
        
        for i in range(num_partitions):
            start_idx = i * levels_per_partition
            end_idx = min((i + 1) * levels_per_partition, len(levels))
            
            if start_idx >= len(levels):
                break
                
            partition_levels = levels[start_idx:end_idx]
            
            # Collect entities from these levels
            partition_entities = []
            partition_relationships = []
            
            for level in partition_levels:
                level_data = graph.get_level(level)
                if level_data is not None:
                    level_entities = level_data.select("entity_id").to_series().to_list()
                    partition_entities.extend(level_entities)
            
            # Get relationships involving these entities
            # This would require accessing the underlying relationship data
            
            partition = GraphPartition(
                partition_id=f"metagraph_partition_{i}",
                graph_type=GraphType.METAGRAPH,
                node_ids=partition_entities,
                edge_ids=partition_relationships,
                data={"levels": partition_levels, "entities": partition_entities},
                metadata={
                    "partition_idx": i,
                    "strategy": "enterprise_level_based",
                    "levels": partition_levels
                },
                size_estimate=len(partition_entities)
            )
            partitions.append(partition)
        
        return partitions
    
    def _partition_by_entities(self, graph: Metagraph, num_partitions: int) -> List[GraphPartition]:
        """Simple entity-based partitioning"""
        # This would need access to all entities in the metagraph
        # For now, create placeholder partitions
        partitions = []
        
        for i in range(num_partitions):
            partition = GraphPartition(
                partition_id=f"metagraph_partition_{i}",
                graph_type=GraphType.METAGRAPH,
                node_ids=[],
                edge_ids=[],
                data={},
                metadata={"partition_idx": i, "strategy": "entity_based"},
                size_estimate=0
            )
            partitions.append(partition)
        
        return partitions
    
    def estimate_partition_cost(self, partition: GraphPartition) -> float:
        """Estimate cost considering enterprise complexity"""
        base_cost = len(partition.node_ids) * 3.0
        # Add enterprise governance overhead
        return base_cost * 2.0


class MultiGraphDistributedOperations:
    """
    Main class for distributed operations across all graph types
    """
    
    def __init__(self, cluster_manager: ClusterManager, task_scheduler: TaskScheduler):
        self.cluster_manager = cluster_manager
        self.task_scheduler = task_scheduler
        
        # Initialize partitioners for each graph type
        self.partitioners = {
            GraphType.HYPERGRAPH: HypergraphPartitioner(),
            GraphType.KNOWLEDGE_GRAPH: KnowledgeGraphPartitioner(),
            GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH: HierarchicalKnowledgeGraphPartitioner(),
            GraphType.METAGRAPH: MetagraphPartitioner()
        }
        
        logger.info("Initialized MultiGraphDistributedOperations")
    
    def detect_graph_type(self, graph: Any) -> GraphType:
        """Automatically detect the type of graph"""
        if isinstance(graph, HierarchicalKnowledgeGraph):
            return GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH
        elif isinstance(graph, KnowledgeGraph):
            return GraphType.KNOWLEDGE_GRAPH
        elif isinstance(graph, Metagraph):
            return GraphType.METAGRAPH
        elif isinstance(graph, Hypergraph):
            return GraphType.HYPERGRAPH
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    async def distributed_operation(self, 
                                  graph: Any,
                                  operation: str,
                                  operation_args: Dict[str, Any] = None,
                                  num_partitions: Optional[int] = None) -> DistributedGraphResult:
        """
        Execute a distributed operation on any supported graph type
        
        Args:
            graph: The graph object (any supported type)
            operation: Operation name (e.g., 'centrality', 'clustering', 'search')
            operation_args: Arguments for the operation
            num_partitions: Number of partitions (auto-detected if None)
        
        Returns:
            DistributedGraphResult with aggregated results
        """
        start_time = asyncio.get_event_loop().time()
        operation_id = f"dist_op_{int(start_time)}"
        
        try:
            # Detect graph type
            graph_type = self.detect_graph_type(graph)
            logger.info(f"Detected graph type: {graph_type.value}")
            
            # Determine number of partitions
            if num_partitions is None:
                num_partitions = min(4, self.cluster_manager.get_available_workers())
            
            # Partition the graph
            partitioner = self.partitioners[graph_type]
            partitions = partitioner.partition(graph, num_partitions)
            
            logger.info(f"Created {len(partitions)} partitions for {operation}")
            
            # Create and submit tasks
            tasks = []
            for partition in partitions:
                task = Task(
                    task_id=f"{operation_id}_partition_{partition.partition_id}",
                    function_name=f"execute_{operation}",
                    args={
                        "partition": partition,
                        "operation_args": operation_args or {},
                        "graph_type": graph_type.value
                    },
                    metadata={
                        "operation": operation,
                        "graph_type": graph_type.value,
                        "partition_id": partition.partition_id
                    }
                )
                tasks.append(task)
            
            # Submit tasks to scheduler
            task_results = await self._execute_distributed_tasks(tasks)
            
            # Aggregate results
            aggregated_result = self._aggregate_results(
                operation_id, graph_type, operation, task_results, start_time
            )
            
            logger.info(f"Completed distributed {operation} on {graph_type.value}")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error in distributed operation: {e}")
            return DistributedGraphResult(
                operation_id=operation_id,
                graph_type=graph_type if 'graph_type' in locals() else GraphType.HYPERGRAPH,
                result_data=None,
                partition_results={},
                execution_time=asyncio.get_event_loop().time() - start_time,
                nodes_processed=0,
                edges_processed=0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_distributed_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks across the cluster"""
        task_results = {}
        
        # Submit all tasks
        submitted_tasks = []
        for task in tasks:
            submitted_task = await self.task_scheduler.submit_task(task)
            submitted_tasks.append((task.task_id, submitted_task))
        
        # Wait for all results
        for task_id, submitted_task in submitted_tasks:
            try:
                result = await submitted_task
                task_results[task_id] = result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                task_results[task_id] = {"error": str(e), "success": False}
        
        return task_results
    
    def _aggregate_results(self, operation_id: str, graph_type: GraphType, operation: str,
                         task_results: Dict[str, Any], start_time: float) -> DistributedGraphResult:
        """Aggregate results from distributed tasks"""
        
        # Basic aggregation (can be enhanced for specific operations)
        successful_results = {k: v for k, v in task_results.items() 
                            if isinstance(v, dict) and v.get("success", True)}
        
        nodes_processed = sum(r.get("nodes_processed", 0) for r in successful_results.values())
        edges_processed = sum(r.get("edges_processed", 0) for r in successful_results.values())
        
        # Operation-specific result aggregation
        aggregated_data = self._aggregate_operation_specific_results(operation, successful_results)
        
        return DistributedGraphResult(
            operation_id=operation_id,
            graph_type=graph_type,
            result_data=aggregated_data,
            partition_results=task_results,
            execution_time=asyncio.get_event_loop().time() - start_time,
            nodes_processed=nodes_processed,
            edges_processed=edges_processed,
            success=len(successful_results) > 0,
            error_message=None if len(successful_results) > 0 else "All tasks failed"
        )
    
    def _aggregate_operation_specific_results(self, operation: str, 
                                            results: Dict[str, Any]) -> Any:
        """Aggregate results based on the specific operation type"""
        
        if operation == "centrality":
            # Merge centrality scores
            merged_scores = {}
            for result in results.values():
                if "centrality_scores" in result:
                    merged_scores.update(result["centrality_scores"])
            return {"centrality_scores": merged_scores}
        
        elif operation == "clustering":
            # Merge cluster assignments
            merged_clusters = {}
            for result in results.values():
                if "clusters" in result:
                    merged_clusters.update(result["clusters"])
            return {"clusters": merged_clusters}
        
        elif operation == "search":
            # Merge search results
            merged_results = []
            for result in results.values():
                if "search_results" in result:
                    merged_results.extend(result["search_results"])
            return {"search_results": merged_results}
        
        else:
            # Generic aggregation
            return {"partition_results": list(results.values())}


# Graph-specific operation implementations that can be called by workers
class GraphOperationExecutor:
    """Executor for graph operations on partitions"""
    
    @staticmethod
    def execute_centrality(partition: GraphPartition, operation_args: Dict[str, Any], 
                         graph_type: str) -> Dict[str, Any]:
        """Execute centrality calculation on a partition"""
        try:
            if graph_type == GraphType.HYPERGRAPH.value:
                return GraphOperationExecutor._hypergraph_centrality(partition, operation_args)
            elif graph_type == GraphType.KNOWLEDGE_GRAPH.value:
                return GraphOperationExecutor._knowledge_graph_centrality(partition, operation_args)
            elif graph_type == GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH.value:
                return GraphOperationExecutor._hierarchical_kg_centrality(partition, operation_args)
            elif graph_type == GraphType.METAGRAPH.value:
                return GraphOperationExecutor._metagraph_centrality(partition, operation_args)
            else:
                raise ValueError(f"Unsupported graph type for centrality: {graph_type}")
        except Exception as e:
            return {"error": str(e), "success": False}
    
    @staticmethod
    def _hypergraph_centrality(partition: GraphPartition, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate centrality for hypergraph partition"""
        # Simplified centrality calculation
        centrality_scores = {}
        for node_id in partition.node_ids:
            # Simple degree-based centrality
            node_edges = partition.data.filter(pl.col("node_id") == node_id)
            centrality_scores[node_id] = len(node_edges)
        
        return {
            "centrality_scores": centrality_scores,
            "nodes_processed": len(partition.node_ids),
            "edges_processed": len(partition.edge_ids),
            "success": True
        }
    
    @staticmethod
    def _knowledge_graph_centrality(partition: GraphPartition, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic centrality for knowledge graph partition"""
        # Enhanced centrality considering semantic relationships
        centrality_scores = {}
        
        # Weight relationships by semantic importance
        for node_id in partition.node_ids:
            node_edges = partition.data.filter(pl.col("node_id") == node_id)
            semantic_weight = len(node_edges) * 1.2  # Semantic bonus
            centrality_scores[node_id] = semantic_weight
        
        return {
            "centrality_scores": centrality_scores,
            "nodes_processed": len(partition.node_ids),
            "edges_processed": len(partition.edge_ids),
            "success": True
        }
    
    @staticmethod
    def _hierarchical_kg_centrality(partition: GraphPartition, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hierarchical centrality for hierarchical knowledge graph partition"""
        centrality_scores = {}
        
        # Consider hierarchical position in centrality
        for node_id in partition.node_ids:
            node_edges = partition.data.filter(pl.col("node_id") == node_id)
            base_centrality = len(node_edges)
            
            # Bonus for nodes that bridge levels
            level_bonus = 1.5 if "cross_level" in partition.metadata.get("strategy", "") else 1.0
            centrality_scores[node_id] = base_centrality * level_bonus
        
        return {
            "centrality_scores": centrality_scores,
            "nodes_processed": len(partition.node_ids),
            "edges_processed": len(partition.edge_ids),
            "success": True
        }
    
    @staticmethod
    def _metagraph_centrality(partition: GraphPartition, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enterprise centrality for metagraph partition"""
        centrality_scores = {}
        
        # Enterprise-aware centrality calculation
        for node_id in partition.node_ids:
            # Base centrality with enterprise importance weighting
            enterprise_weight = 2.0  # Higher weight for enterprise entities
            centrality_scores[node_id] = enterprise_weight
        
        return {
            "centrality_scores": centrality_scores,
            "nodes_processed": len(partition.node_ids),
            "edges_processed": 0,  # Metagraph edges handled differently
            "success": True
        }


# Register operations with workers
def register_graph_operations():
    """Register graph operations with worker nodes"""
    from .worker_node import WorkerNode
    
    # Register executors
    WorkerNode.register_function("execute_centrality", GraphOperationExecutor.execute_centrality)
    # Additional operations can be registered here
    
    logger.info("Registered graph operations with worker nodes")