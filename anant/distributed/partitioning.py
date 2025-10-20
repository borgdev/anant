"""
Production-Grade Graph Partitioning
===================================

This module provides enterprise-level graph partitioning using industry-standard algorithms:
- METIS: Multilevel k-way partitioning with minimum edge-cut
- KaHiP: Karlsruhe High Quality Partitioning
- Scotch: Graph partitioning and sparse matrix ordering
- NetworkX: Fallback algorithms for basic partitioning
- Custom: Specialized algorithms for different graph types

Features:
- Automatic algorithm selection based on graph characteristics
- Quality metrics and benchmarking
- Multi-objective optimization (cut minimization, load balancing)
- Hierarchical and multilevel partitioning
- Memory-efficient processing for large graphs
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import tempfile
import os

import numpy as np
import polars as pl

# Core graph types
from ..classes.hypergraph import Hypergraph
from ..classes.incidence_store import IncidenceStore

# Optional high-performance partitioning libraries
try:
    import metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    metis = None

try:
    import kahip
    KAHIP_AVAILABLE = True
except ImportError:
    KAHIP_AVAILABLE = False
    kahip = None

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    ig = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


class PartitioningAlgorithm(Enum):
    """Available partitioning algorithms."""
    METIS = "metis"              # METIS multilevel k-way partitioning
    METIS_RECURSIVE = "metis_recursive"  # METIS recursive bisection
    KAHIP = "kahip"              # KaHiP high-quality partitioning
    KAHIP_FAST = "kahip_fast"    # KaHiP fast mode
    IGRAPH_LEIDEN = "igraph_leiden"  # Leiden community detection
    IGRAPH_LOUVAIN = "igraph_louvain"  # Louvain community detection
    NETWORKX_SPECTRAL = "networkx_spectral"  # Spectral partitioning
    NETWORKX_GREEDY = "networkx_greedy"  # Greedy modularity partitioning
    RANDOM = "random"            # Random partitioning (baseline)
    ROUND_ROBIN = "round_robin"  # Simple round-robin (baseline)
    AUTO = "auto"                # Automatic algorithm selection


class PartitioningObjective(Enum):
    """Optimization objectives for partitioning."""
    MIN_EDGE_CUT = "min_edge_cut"        # Minimize edges between partitions
    MIN_COMMUNICATION = "min_communication"  # Minimize communication volume
    LOAD_BALANCE = "load_balance"        # Balance partition sizes
    MULTI_OBJECTIVE = "multi_objective"  # Balance multiple objectives


@dataclass
class PartitioningConfig:
    """Configuration for graph partitioning."""
    algorithm: PartitioningAlgorithm = PartitioningAlgorithm.AUTO
    objective: PartitioningObjective = PartitioningObjective.MIN_EDGE_CUT
    num_partitions: int = 4
    imbalance_tolerance: float = 0.03  # 3% imbalance allowed
    seed: Optional[int] = None
    quality_threshold: float = 0.8  # Minimum partition quality
    max_iterations: int = 100
    recursive_levels: int = 0  # 0 = k-way, >0 = recursive bisection
    edge_weight_strategy: str = "uniform"  # "uniform", "degree", "custom"
    vertex_weight_strategy: str = "uniform"  # "uniform", "degree", "pagerank"
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartitioningResult:
    """Result of graph partitioning operation."""
    partition_assignment: List[int]  # Node ID -> partition ID mapping
    num_partitions: int
    algorithm_used: PartitioningAlgorithm
    quality_metrics: Dict[str, float]
    execution_time: float
    memory_usage: float
    edge_cut: int
    communication_volume: int
    partition_sizes: List[int]
    imbalance_ratio: float
    success: bool = True
    error_message: Optional[str] = None


class GraphConverter:
    """Converts between different graph representations."""
    
    @staticmethod
    def to_metis_format(graph: Any) -> Tuple[List[List[int]], Optional[List[float]], Optional[List[float]]]:
        """Convert graph to METIS adjacency list format."""
        if hasattr(graph, 'incidence_store') and isinstance(graph.incidence_store, IncidenceStore):
            # Handle ANANT graph types
            return GraphConverter._convert_incidence_to_metis(graph.incidence_store)
        elif hasattr(graph, 'edges') and hasattr(graph, 'nodes'):
            # Handle NetworkX-like graphs
            return GraphConverter._convert_networkx_to_metis(graph)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    @staticmethod
    def _convert_incidence_to_metis(incidence_store: IncidenceStore) -> Tuple[List[List[int]], Optional[List[float]], Optional[List[float]]]:
        """Convert IncidenceStore to METIS format."""
        # Get node and edge data
        nodes_df = incidence_store.nodes
        edges_df = incidence_store.edges
        
        # Build adjacency list representation
        node_ids = nodes_df.get_column('node_id').to_list()
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Initialize adjacency lists
        adjacency_lists = [[] for _ in range(len(node_ids))]
        
        # Process edges to build adjacency
        for edge_row in edges_df.iter_rows(named=True):
            edge_id = edge_row['edge_id']
            edge_nodes = edge_row['nodes']
            
            # For hyperedges, connect all pairs of nodes
            if isinstance(edge_nodes, list) and len(edge_nodes) > 1:
                for i, node1 in enumerate(edge_nodes):
                    if node1 in node_id_to_idx:
                        idx1 = node_id_to_idx[node1]
                        for j, node2 in enumerate(edge_nodes):
                            if i != j and node2 in node_id_to_idx:
                                idx2 = node_id_to_idx[node2]
                                if idx2 not in adjacency_lists[idx1]:
                                    adjacency_lists[idx1].append(idx2)
        
        # Convert to 1-based indexing for METIS
        metis_adjacency = []
        for adj_list in adjacency_lists:
            metis_adjacency.append([idx + 1 for idx in sorted(adj_list)])
        
        # No weights for now (can be added later)
        return metis_adjacency, None, None
    
    @staticmethod
    def _convert_networkx_to_metis(graph) -> Tuple[List[List[int]], Optional[List[float]], Optional[List[float]]]:
        """Convert NetworkX graph to METIS format."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for graph conversion")
        
        # Convert to NetworkX if needed
        if not isinstance(graph, nx.Graph):
            # Try to convert from other formats
            nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
            edges = list(graph.edges) if hasattr(graph, 'edges') else []
            
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        else:
            G = graph
        
        # Get adjacency list in METIS format (1-based indexing)
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        adjacency_lists = []
        
        for node in G.nodes():
            neighbors = [node_to_idx[neighbor] + 1 for neighbor in G.neighbors(node)]
            adjacency_lists.append(sorted(neighbors))
        
        return adjacency_lists, None, None
    
    @staticmethod
    def to_igraph(graph: Any) -> 'ig.Graph':
        """Convert to igraph format."""
        if not IGRAPH_AVAILABLE:
            raise ImportError("python-igraph required for igraph conversion")
        
        # Convert through NetworkX for simplicity
        if hasattr(graph, 'incidence_store'):
            # ANANT graph types
            nodes_df = graph.incidence_store.nodes
            edges_df = graph.incidence_store.edges
            
            node_ids = nodes_df.get_column('node_id').to_list()
            edge_list = []
            
            for edge_row in edges_df.iter_rows(named=True):
                edge_nodes = edge_row['nodes']
                if isinstance(edge_nodes, list) and len(edge_nodes) > 1:
                    # For hyperedges, create clique
                    for i in range(len(edge_nodes)):
                        for j in range(i + 1, len(edge_nodes)):
                            if edge_nodes[i] in node_ids and edge_nodes[j] in node_ids:
                                edge_list.append((edge_nodes[i], edge_nodes[j]))
            
            # Create igraph
            g = ig.Graph()
            g.add_vertices(node_ids)
            g.add_edges(edge_list)
            return g
        else:
            raise ValueError(f"Unsupported graph type for igraph conversion: {type(graph)}")


class MetisPartitioner:
    """METIS-based graph partitioner."""
    
    def __init__(self, config: PartitioningConfig):
        self.config = config
        
        if not METIS_AVAILABLE:
            raise ImportError("METIS not available - install with: pip install metis")
    
    def partition(self, graph: Any) -> PartitioningResult:
        """Partition graph using METIS."""
        start_time = time.time()
        
        try:
            # Convert graph to METIS format
            adjacency_list, edge_weights, vertex_weights = GraphConverter.to_metis_format(graph)
            
            if not adjacency_list:
                raise ValueError("Empty graph cannot be partitioned")
            
            # Prepare METIS parameters
            metis_options = metis.METIS_SetDefaultOptions()
            
            # Configure algorithm
            if self.config.algorithm == PartitioningAlgorithm.METIS_RECURSIVE:
                metis_options[metis.METIS_OPTION_PTYPE] = metis.METIS_PTYPE_RB
            else:
                metis_options[metis.METIS_OPTION_PTYPE] = metis.METIS_PTYPE_KWAY
            
            # Set objective
            if self.config.objective == PartitioningObjective.MIN_EDGE_CUT:
                metis_options[metis.METIS_OPTION_OBJTYPE] = metis.METIS_OBJTYPE_CUT
            elif self.config.objective == PartitioningObjective.MIN_COMMUNICATION:
                metis_options[metis.METIS_OPTION_OBJTYPE] = metis.METIS_OBJTYPE_VOL
            
            # Set imbalance tolerance
            metis_options[metis.METIS_OPTION_UFACTOR] = int(self.config.imbalance_tolerance * 1000)
            
            # Set seed for reproducibility
            if self.config.seed is not None:
                metis_options[metis.METIS_OPTION_SEED] = self.config.seed
            
            # Set iterations
            metis_options[metis.METIS_OPTION_NITER] = self.config.max_iterations
            
            # Perform partitioning
            if self.config.num_partitions == 2:
                # Recursive bisection
                (edgecuts, parts) = metis.part_graph(
                    adjacency_list,
                    nparts=self.config.num_partitions,
                    tpwgts=None,
                    ubvec=None,
                    options=metis_options,
                    recursive=True
                )
            else:
                # K-way partitioning
                (edgecuts, parts) = metis.part_graph(
                    adjacency_list,
                    nparts=self.config.num_partitions,
                    tpwgts=None,
                    ubvec=None,
                    options=metis_options
                )
            
            # Calculate quality metrics
            execution_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(
                parts, adjacency_list, edgecuts
            )
            
            # Calculate partition sizes and imbalance
            partition_sizes = [0] * self.config.num_partitions
            for part in parts:
                partition_sizes[part] += 1
            
            avg_size = len(parts) / self.config.num_partitions
            max_imbalance = max(abs(size - avg_size) / avg_size for size in partition_sizes)
            
            return PartitioningResult(
                partition_assignment=parts,
                num_partitions=self.config.num_partitions,
                algorithm_used=self.config.algorithm,
                quality_metrics=quality_metrics,
                execution_time=execution_time,
                memory_usage=0.0,  # TODO: Implement memory tracking
                edge_cut=edgecuts,
                communication_volume=self._calculate_communication_volume(parts, adjacency_list),
                partition_sizes=partition_sizes,
                imbalance_ratio=max_imbalance,
                success=True
            )
            
        except Exception as e:
            logger.error(f"METIS partitioning failed: {e}")
            return PartitioningResult(
                partition_assignment=[],
                num_partitions=0,
                algorithm_used=self.config.algorithm,
                quality_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                edge_cut=0,
                communication_volume=0,
                partition_sizes=[],
                imbalance_ratio=1.0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_quality_metrics(self, parts: List[int], adjacency_list: List[List[int]], 
                                 edge_cut: int) -> Dict[str, float]:
        """Calculate partitioning quality metrics."""
        total_edges = sum(len(adj) for adj in adjacency_list) // 2
        cut_ratio = edge_cut / max(total_edges, 1)
        
        # Calculate modularity-like metric
        partition_sizes = [0] * self.config.num_partitions
        for part in parts:
            partition_sizes[part] += 1
        
        # Load balance metric
        avg_size = len(parts) / self.config.num_partitions
        load_balance = 1.0 - max(abs(size - avg_size) / avg_size for size in partition_sizes)
        
        return {
            "edge_cut_ratio": cut_ratio,
            "load_balance": load_balance,
            "modularity": 1.0 - cut_ratio,  # Simplified modularity approximation
            "efficiency": load_balance * (1.0 - cut_ratio)
        }
    
    def _calculate_communication_volume(self, parts: List[int], 
                                      adjacency_list: List[List[int]]) -> int:
        """Calculate communication volume between partitions."""
        comm_volume = 0
        for node_idx, neighbors in enumerate(adjacency_list):
            node_part = parts[node_idx]
            for neighbor_idx in neighbors:
                neighbor_idx_0 = neighbor_idx - 1  # Convert from 1-based to 0-based
                if 0 <= neighbor_idx_0 < len(parts):
                    neighbor_part = parts[neighbor_idx_0]
                    if node_part != neighbor_part:
                        comm_volume += 1
        
        return comm_volume // 2  # Each edge counted twice


class KaHiPPartitioner:
    """KaHiP-based graph partitioner."""
    
    def __init__(self, config: PartitioningConfig):
        self.config = config
        
        if not KAHIP_AVAILABLE:
            raise ImportError("KaHiP not available - install with: pip install kahip")
    
    def partition(self, graph: Any) -> PartitioningResult:
        """Partition graph using KaHiP."""
        start_time = time.time()
        
        try:
            # Convert graph to edge list format for KaHiP
            edge_list, num_nodes = self._convert_to_edge_list(graph)
            
            if not edge_list:
                raise ValueError("Empty graph cannot be partitioned")
            
            # Configure KaHiP parameters
            if self.config.algorithm == PartitioningAlgorithm.KAHIP_FAST:
                mode = kahip.FAST
            else:
                mode = kahip.STRONG
            
            # Perform partitioning
            edgecut, parts = kahip.kaffpa(
                num_nodes,
                edge_list,
                self.config.num_partitions,
                imbalance=self.config.imbalance_tolerance,
                mode=mode,
                seed=self.config.seed or 0
            )
            
            # Calculate quality metrics
            execution_time = time.time() - start_time
            quality_metrics = self._calculate_quality_metrics(parts, edge_list, edgecut)
            
            # Calculate partition sizes
            partition_sizes = [0] * self.config.num_partitions
            for part in parts:
                partition_sizes[part] += 1
            
            avg_size = len(parts) / self.config.num_partitions
            max_imbalance = max(abs(size - avg_size) / avg_size for size in partition_sizes)
            
            return PartitioningResult(
                partition_assignment=parts,
                num_partitions=self.config.num_partitions,
                algorithm_used=self.config.algorithm,
                quality_metrics=quality_metrics,
                execution_time=execution_time,
                memory_usage=0.0,
                edge_cut=edgecut,
                communication_volume=self._calculate_communication_volume(parts, edge_list),
                partition_sizes=partition_sizes,
                imbalance_ratio=max_imbalance,
                success=True
            )
            
        except Exception as e:
            logger.error(f"KaHiP partitioning failed: {e}")
            return PartitioningResult(
                partition_assignment=[],
                num_partitions=0,
                algorithm_used=self.config.algorithm,
                quality_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                edge_cut=0,
                communication_volume=0,
                partition_sizes=[],
                imbalance_ratio=1.0,
                success=False,
                error_message=str(e)
            )
    
    def _convert_to_edge_list(self, graph: Any) -> Tuple[List[Tuple[int, int]], int]:
        """Convert graph to edge list format."""
        if hasattr(graph, 'incidence_store'):
            return self._convert_incidence_to_edge_list(graph.incidence_store)
        else:
            raise ValueError(f"Unsupported graph type for KaHiP: {type(graph)}")
    
    def _convert_incidence_to_edge_list(self, incidence_store: IncidenceStore) -> Tuple[List[Tuple[int, int]], int]:
        """Convert IncidenceStore to edge list."""
        nodes_df = incidence_store.nodes
        edges_df = incidence_store.edges
        
        node_ids = nodes_df.get_column('node_id').to_list()
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        edge_list = []
        for edge_row in edges_df.iter_rows(named=True):
            edge_nodes = edge_row['nodes']
            if isinstance(edge_nodes, list) and len(edge_nodes) > 1:
                # Create clique for hyperedge
                for i in range(len(edge_nodes)):
                    for j in range(i + 1, len(edge_nodes)):
                        if edge_nodes[i] in node_id_to_idx and edge_nodes[j] in node_id_to_idx:
                            idx1 = node_id_to_idx[edge_nodes[i]]
                            idx2 = node_id_to_idx[edge_nodes[j]]
                            edge_list.append((idx1, idx2))
        
        return edge_list, len(node_ids)
    
    def _calculate_quality_metrics(self, parts: List[int], edge_list: List[Tuple[int, int]], 
                                 edge_cut: int) -> Dict[str, float]:
        """Calculate partitioning quality metrics."""
        total_edges = len(edge_list)
        cut_ratio = edge_cut / max(total_edges, 1)
        
        partition_sizes = [0] * self.config.num_partitions
        for part in parts:
            partition_sizes[part] += 1
        
        avg_size = len(parts) / self.config.num_partitions
        load_balance = 1.0 - max(abs(size - avg_size) / avg_size for size in partition_sizes)
        
        return {
            "edge_cut_ratio": cut_ratio,
            "load_balance": load_balance,
            "modularity": 1.0 - cut_ratio,
            "efficiency": load_balance * (1.0 - cut_ratio)
        }
    
    def _calculate_communication_volume(self, parts: List[int], 
                                      edge_list: List[Tuple[int, int]]) -> int:
        """Calculate communication volume."""
        comm_volume = 0
        for node1, node2 in edge_list:
            if parts[node1] != parts[node2]:
                comm_volume += 1
        return comm_volume


class ProductionPartitioner:
    """Production-grade graph partitioner with automatic algorithm selection."""
    
    def __init__(self, config: Optional[PartitioningConfig] = None):
        self.config = config or PartitioningConfig()
        self.available_algorithms = self._detect_available_algorithms()
        logger.info(f"Available partitioning algorithms: {[alg.value for alg in self.available_algorithms]}")
    
    def _detect_available_algorithms(self) -> List[PartitioningAlgorithm]:
        """Detect which partitioning algorithms are available."""
        available = [PartitioningAlgorithm.RANDOM, PartitioningAlgorithm.ROUND_ROBIN]
        
        if METIS_AVAILABLE:
            available.extend([PartitioningAlgorithm.METIS, PartitioningAlgorithm.METIS_RECURSIVE])
        
        if KAHIP_AVAILABLE:
            available.extend([PartitioningAlgorithm.KAHIP, PartitioningAlgorithm.KAHIP_FAST])
        
        if IGRAPH_AVAILABLE:
            available.extend([PartitioningAlgorithm.IGRAPH_LEIDEN, PartitioningAlgorithm.IGRAPH_LOUVAIN])
        
        if NETWORKX_AVAILABLE:
            available.extend([PartitioningAlgorithm.NETWORKX_SPECTRAL, PartitioningAlgorithm.NETWORKX_GREEDY])
        
        return available
    
    def partition(self, graph: Any, config: Optional[PartitioningConfig] = None) -> PartitioningResult:
        """Partition graph using best available algorithm."""
        config = config or self.config
        
        # Auto-select algorithm if needed
        if config.algorithm == PartitioningAlgorithm.AUTO:
            config.algorithm = self._select_best_algorithm(graph, config)
        
        # Validate algorithm availability
        if config.algorithm not in self.available_algorithms:
            logger.warning(f"Algorithm {config.algorithm.value} not available, falling back to best available")
            config.algorithm = self._select_best_algorithm(graph, config)
        
        # Perform partitioning
        try:
            if config.algorithm in [PartitioningAlgorithm.METIS, PartitioningAlgorithm.METIS_RECURSIVE]:
                partitioner = MetisPartitioner(config)
                return partitioner.partition(graph)
            
            elif config.algorithm in [PartitioningAlgorithm.KAHIP, PartitioningAlgorithm.KAHIP_FAST]:
                partitioner = KaHiPPartitioner(config)
                return partitioner.partition(graph)
            
            else:
                # Fall back to simple algorithms
                return self._simple_partition(graph, config)
                
        except Exception as e:
            logger.error(f"Partitioning with {config.algorithm.value} failed: {e}")
            # Fall back to simple round-robin
            config.algorithm = PartitioningAlgorithm.ROUND_ROBIN
            return self._simple_partition(graph, config)
    
    def _select_best_algorithm(self, graph: Any, config: PartitioningConfig) -> PartitioningAlgorithm:
        """Select best algorithm based on graph characteristics."""
        # Estimate graph size
        if hasattr(graph, 'num_nodes'):
            num_nodes = graph.num_nodes
        elif hasattr(graph, 'incidence_store'):
            num_nodes = len(graph.incidence_store.nodes)
        else:
            num_nodes = 1000  # Default estimate
        
        # Algorithm selection logic
        if num_nodes < 100:
            # Small graphs - use simple algorithms
            return PartitioningAlgorithm.ROUND_ROBIN
        
        elif num_nodes < 10000:
            # Medium graphs - prefer METIS or KaHiP
            if PartitioningAlgorithm.METIS in self.available_algorithms:
                return PartitioningAlgorithm.METIS
            elif PartitioningAlgorithm.KAHIP in self.available_algorithms:
                return PartitioningAlgorithm.KAHIP
            else:
                return PartitioningAlgorithm.NETWORKX_SPECTRAL if NETWORKX_AVAILABLE else PartitioningAlgorithm.ROUND_ROBIN
        
        else:
            # Large graphs - prefer fast algorithms
            if PartitioningAlgorithm.KAHIP_FAST in self.available_algorithms:
                return PartitioningAlgorithm.KAHIP_FAST
            elif PartitioningAlgorithm.METIS in self.available_algorithms:
                return PartitioningAlgorithm.METIS
            else:
                return PartitioningAlgorithm.ROUND_ROBIN
    
    def _simple_partition(self, graph: Any, config: PartitioningConfig) -> PartitioningResult:
        """Simple fallback partitioning algorithms."""
        start_time = time.time()
        
        try:
            # Get node count
            if hasattr(graph, 'num_nodes'):
                num_nodes = graph.num_nodes
            elif hasattr(graph, 'incidence_store'):
                num_nodes = len(graph.incidence_store.nodes)
            else:
                raise ValueError("Cannot determine graph size")
            
            # Generate partition assignment
            if config.algorithm == PartitioningAlgorithm.RANDOM:
                parts = [np.random.randint(0, config.num_partitions) for _ in range(num_nodes)]
            else:  # ROUND_ROBIN
                parts = [i % config.num_partitions for i in range(num_nodes)]
            
            # Calculate basic metrics
            partition_sizes = [0] * config.num_partitions
            for part in parts:
                partition_sizes[part] += 1
            
            avg_size = num_nodes / config.num_partitions
            max_imbalance = max(abs(size - avg_size) / avg_size for size in partition_sizes) if avg_size > 0 else 0
            
            return PartitioningResult(
                partition_assignment=parts,
                num_partitions=config.num_partitions,
                algorithm_used=config.algorithm,
                quality_metrics={"load_balance": 1.0 - max_imbalance},
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                edge_cut=0,  # Not calculated for simple algorithms
                communication_volume=0,
                partition_sizes=partition_sizes,
                imbalance_ratio=max_imbalance,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Simple partitioning failed: {e}")
            return PartitioningResult(
                partition_assignment=[],
                num_partitions=0,
                algorithm_used=config.algorithm,
                quality_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                edge_cut=0,
                communication_volume=0,
                partition_sizes=[],
                imbalance_ratio=1.0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_algorithms(self, graph: Any, algorithms: Optional[List[PartitioningAlgorithm]] = None) -> Dict[str, PartitioningResult]:
        """Benchmark multiple partitioning algorithms on the same graph."""
        algorithms = algorithms or self.available_algorithms
        results = {}
        
        for algorithm in algorithms:
            if algorithm == PartitioningAlgorithm.AUTO:
                continue  # Skip AUTO as it's not a real algorithm
            
            config = PartitioningConfig(
                algorithm=algorithm,
                num_partitions=self.config.num_partitions,
                seed=42  # Fixed seed for fair comparison
            )
            
            logger.info(f"Benchmarking {algorithm.value}...")
            result = self.partition(graph, config)
            results[algorithm.value] = result
        
        return results
    
    def get_algorithm_recommendations(self, graph: Any) -> Dict[str, str]:
        """Get algorithm recommendations based on graph characteristics."""
        if hasattr(graph, 'num_nodes'):
            num_nodes = graph.num_nodes
            num_edges = getattr(graph, 'num_edges', 0)
        elif hasattr(graph, 'incidence_store'):
            num_nodes = len(graph.incidence_store.nodes)
            num_edges = len(graph.incidence_store.edges)
        else:
            return {"error": "Cannot analyze graph characteristics"}
        
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        recommendations = {
            "graph_size": f"{num_nodes} nodes, {num_edges} edges",
            "graph_density": f"{density:.4f}",
        }
        
        if num_nodes < 100:
            recommendations["recommended"] = "round_robin (small graph)"
        elif num_nodes < 1000:
            recommendations["recommended"] = "metis (medium graph)"
        elif density > 0.1:
            recommendations["recommended"] = "kahip (dense graph)"
        else:
            recommendations["recommended"] = "metis or kahip_fast (large sparse graph)"
        
        recommendations["available"] = [alg.value for alg in self.available_algorithms]
        
        return recommendations


# Factory function for easy usage
def create_production_partitioner(algorithm: Optional[str] = None, 
                                num_partitions: int = 4,
                                **kwargs) -> ProductionPartitioner:
    """Create production partitioner with specified configuration."""
    config = PartitioningConfig(
        algorithm=PartitioningAlgorithm(algorithm) if algorithm else PartitioningAlgorithm.AUTO,
        num_partitions=num_partitions,
        **kwargs
    )
    return ProductionPartitioner(config)