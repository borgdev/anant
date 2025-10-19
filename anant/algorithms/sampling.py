"""
Intelligent Sampling for Large-Scale Hypergraph Analysis
======================================================

Adaptive sampling strategies that preserve graph properties while
reducing computational complexity for large hypergraphs.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
import logging
from collections import defaultdict
from ..utils.decorators import performance_monitor

if TYPE_CHECKING:
    from ..classes.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


class SmartSampler:
    """
    Intelligent sampler that preserves hypergraph structure
    
    This class implements various sampling strategies designed to maintain
    critical graph properties while reducing computational load for analytics.
    """
    
    def __init__(self, 
                 hypergraph,
                 strategy: str = 'adaptive',
                 preserve_structure: bool = True,
                 random_seed: int = 42):
        """
        Initialize the smart sampler
        
        Args:
            hypergraph: Anant Hypergraph instance
            strategy: Sampling strategy ('adaptive', 'degree_based', 'random', 'stratified')
            preserve_structure: Whether to preserve critical graph properties
            random_seed: Seed for reproducible sampling
        """
        self.hypergraph = hypergraph
        self.strategy = strategy
        self.preserve_structure = preserve_structure
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Cache graph statistics for intelligent sampling decisions
        self._graph_stats = self._compute_graph_statistics()
        
    def _compute_graph_statistics(self) -> Dict[str, Any]:
        """Compute basic graph statistics for sampling decisions"""
        
        stats = self.hypergraph.incidences.get_statistics()
        
        # Add degree distribution analysis
        degree_data = {}
        for node in self.hypergraph.nodes:
            degree = self.hypergraph.incidences.get_node_degree(node)
            degree_data[node] = degree
        
        degrees = list(degree_data.values())
        stats.update({
            'degree_distribution': {
                'mean': np.mean(degrees) if degrees else 0,
                'std': np.std(degrees) if degrees else 0,
                'min': np.min(degrees) if degrees else 0,
                'max': np.max(degrees) if degrees else 0,
                'percentiles': {
                    '25': np.percentile(degrees, 25) if degrees else 0,
                    '50': np.percentile(degrees, 50) if degrees else 0,
                    '75': np.percentile(degrees, 75) if degrees else 0,
                    '90': np.percentile(degrees, 90) if degrees else 0
                }
            },
            'node_degrees': degree_data
        })
        
        return stats
    
    @performance_monitor
    def determine_optimal_sample_size(self, 
                                    target_algorithm: str = 'general',
                                    max_nodes: Optional[int] = None,
                                    min_nodes: int = 100) -> int:
        """
        Determine optimal sample size based on graph properties and target algorithm
        
        Args:
            target_algorithm: Algorithm that will use the sample ('clustering', 'centrality', 'general')
            max_nodes: Maximum number of nodes to include
            min_nodes: Minimum number of nodes to include
            
        Returns:
            Optimal sample size
        """
        
        total_nodes = self._graph_stats['num_nodes']
        
        if total_nodes <= min_nodes:
            return total_nodes
        
        # Algorithm-specific scaling factors
        scaling_factors = {
            'clustering': 0.3,  # Clustering algorithms are computationally expensive
            'centrality': 0.5,  # Centrality can handle moderate sizes
            'general': 0.4,     # General purpose
            'structural': 0.6   # Structural analysis is typically faster
        }
        
        base_factor = scaling_factors.get(target_algorithm, 0.4)
        
        # Adjust based on graph density
        avg_edge_size = self._graph_stats.get('avg_edge_size', 2)
        if avg_edge_size > 5:  # Dense hypergraph
            base_factor *= 0.7
        elif avg_edge_size < 2:  # Sparse hypergraph  
            base_factor *= 1.2
        
        # Calculate optimal size
        optimal_size = int(total_nodes * base_factor)
        
        # Apply constraints
        if max_nodes:
            optimal_size = min(optimal_size, max_nodes)
        optimal_size = max(optimal_size, min_nodes)
        
        logger.info(f"Determined optimal sample size: {optimal_size} from {total_nodes} total nodes")
        return optimal_size
    
    @performance_monitor
    def adaptive_sample(self, 
                       sample_size: Optional[int] = None,
                       algorithm: str = 'general') -> 'Hypergraph':
        """
        Create adaptive sample that preserves graph properties
        
        Args:
            sample_size: Target sample size (auto-determined if None)
            algorithm: Target algorithm for optimization
            
        Returns:
            Sampled hypergraph
        """
        
        if sample_size is None:
            sample_size = self.determine_optimal_sample_size(algorithm)
        
        # Ensure sample_size is valid
        total_nodes = len(self.hypergraph.nodes)
        if sample_size is None:
            sample_size = max(100, total_nodes // 4)  # fallback
        sample_size = max(1, min(sample_size, total_nodes))
        
        if sample_size >= total_nodes:
            logger.info("Sample size >= total nodes, returning original hypergraph")
            return self.hypergraph
        
        if self.strategy == 'degree_based':
            return self._degree_based_sample(sample_size)
        elif self.strategy == 'stratified':
            return self._stratified_sample(sample_size)
        elif self.strategy == 'random':
            return self._random_sample(sample_size)
        else:  # adaptive (default)
            return self._adaptive_sample(sample_size)
    
    def _adaptive_sample(self, sample_size: int) -> 'Hypergraph':
        """Adaptive sampling that combines multiple strategies"""
        
        # Analyze degree distribution to choose best strategy
        degree_stats = self._graph_stats['degree_distribution']
        degree_std = degree_stats['std']
        degree_mean = degree_stats['mean']
        
        # If degree distribution is highly skewed, use degree-based sampling
        if degree_std > degree_mean * 0.8:
            logger.info("Using degree-based sampling for skewed distribution")
            return self._degree_based_sample(sample_size)
        else:
            logger.info("Using stratified sampling for balanced distribution")
            return self._stratified_sample(sample_size)
    
    def _degree_based_sample(self, sample_size: int) -> 'Hypergraph':
        """Sample nodes based on degree distribution"""
        
        node_degrees = self._graph_stats['node_degrees']
        
        # Separate high-degree and low-degree nodes
        degree_threshold = self._graph_stats['degree_distribution']['percentiles']['75']
        
        high_degree_nodes = [node for node, degree in node_degrees.items() 
                           if degree >= degree_threshold]
        low_degree_nodes = [node for node, degree in node_degrees.items() 
                          if degree < degree_threshold]
        
        # Sample proportionally but ensure high-degree nodes are well represented
        high_degree_ratio = min(0.4, len(high_degree_nodes) / len(node_degrees))
        high_degree_sample_size = int(sample_size * high_degree_ratio)
        low_degree_sample_size = sample_size - high_degree_sample_size
        
        # Sample from each group
        sampled_nodes = set()
        
        if high_degree_nodes:
            sampled_high = np.random.choice(
                high_degree_nodes, 
                size=min(high_degree_sample_size, len(high_degree_nodes)),
                replace=False
            )
            sampled_nodes.update(sampled_high)
        
        remaining_needed = sample_size - len(sampled_nodes)
        if low_degree_nodes and remaining_needed > 0:
            sampled_low = np.random.choice(
                low_degree_nodes,
                size=min(remaining_needed, len(low_degree_nodes)),
                replace=False
            )
            sampled_nodes.update(sampled_low)
        
        return self._create_subgraph(sampled_nodes)
    
    def _stratified_sample(self, sample_size: int) -> 'Hypergraph':
        """Stratified sampling preserving edge size distribution"""
        
        # Group edges by size
        edge_sizes = {}
        for edge in self.hypergraph.edges:
            size = self.hypergraph.incidences.get_edge_size(edge)
            edge_sizes[edge] = size
        
        # Create size strata
        size_groups = defaultdict(list)
        for edge, size in edge_sizes.items():
            size_groups[size].append(edge)
        
        # Sample edges proportionally from each stratum
        total_edges = len(self.hypergraph.edges)
        sampled_edges = set()
        
        for size, edges in size_groups.items():
            stratum_proportion = len(edges) / total_edges
            stratum_sample_size = max(1, int(sample_size * stratum_proportion * 0.3))
            
            if len(edges) <= stratum_sample_size:
                sampled_edges.update(edges)
            else:
                sampled_stratum = np.random.choice(
                    edges, size=stratum_sample_size, replace=False
                )
                sampled_edges.update(sampled_stratum)
        
        # Get nodes from sampled edges
        sampled_nodes = set()
        for edge in sampled_edges:
            edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
            sampled_nodes.update(edge_nodes)
        
        # If we need more nodes, add random high-degree nodes
        if len(sampled_nodes) < sample_size:
            remaining_nodes = set(self.hypergraph.nodes) - sampled_nodes
            node_degrees = self._graph_stats['node_degrees']
            
            # Sort remaining nodes by degree
            sorted_remaining = sorted(
                remaining_nodes, 
                key=lambda x: node_degrees.get(x, 0), 
                reverse=True
            )
            
            additional_needed = sample_size - len(sampled_nodes)
            sampled_nodes.update(sorted_remaining[:additional_needed])
        
        # If we have too many nodes, keep the highest degree ones
        elif len(sampled_nodes) > sample_size:
            node_degrees = self._graph_stats['node_degrees']
            sorted_nodes = sorted(
                sampled_nodes,
                key=lambda x: node_degrees.get(x, 0),
                reverse=True
            )
            sampled_nodes = set(sorted_nodes[:sample_size])
        
        return self._create_subgraph(sampled_nodes)
    
    def _random_sample(self, sample_size: int) -> 'Hypergraph':
        """Simple random sampling of nodes"""
        
        all_nodes = list(self.hypergraph.nodes)
        sampled_nodes = np.random.choice(
            all_nodes, size=min(sample_size, len(all_nodes)), replace=False
        )
        
        return self._create_subgraph(set(sampled_nodes))
    
    def _create_subgraph(self, sampled_nodes: set) -> 'Hypergraph':
        """Create subgraph from sampled nodes"""
        
        from ..classes.hypergraph import Hypergraph
        
        # Get all edges that contain at least one sampled node
        relevant_edges = set()
        for node in sampled_nodes:
            node_edges = self.hypergraph.incidences.get_node_edges(node)
            relevant_edges.update(node_edges)
        
        # Build edge dictionary for subgraph
        edge_dict = {}
        for edge in relevant_edges:
            edge_nodes = self.hypergraph.incidences.get_edge_nodes(edge)
            # Keep only sampled nodes in each edge
            filtered_nodes = [node for node in edge_nodes if node in sampled_nodes]
            if len(filtered_nodes) >= 1:  # Keep edges with at least 1 sampled node
                edge_dict[edge] = filtered_nodes
        
        # Create new hypergraph
        subgraph = Hypergraph.from_dict(edge_dict)
        
        # Copy relevant properties if they exist
        if hasattr(self.hypergraph, '_edge_properties') and self.hypergraph._edge_properties:
            for edge in edge_dict:
                if edge in self.hypergraph._edge_properties.data:
                    edge_props = self.hypergraph._edge_properties.data[edge]
                    subgraph.add_edge_properties(edge, **edge_props)
        
        if hasattr(self.hypergraph, '_node_properties') and self.hypergraph._node_properties:
            for node in sampled_nodes:
                if node in self.hypergraph._node_properties.data:
                    node_props = self.hypergraph._node_properties.data[node]
                    subgraph.add_node_properties(node, **node_props)
        
        logger.info(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
        return subgraph


@performance_monitor
def auto_scale_algorithm(hypergraph,
                        algorithm_func,
                        algorithm_name: str = 'unknown',
                        max_nodes: int = 5000,
                        **algorithm_kwargs):
    """
    Automatically scale algorithm execution based on graph size
    
    Args:
        hypergraph: Input hypergraph
        algorithm_func: Algorithm function to execute
        algorithm_name: Name of algorithm for logging
        max_nodes: Maximum nodes before sampling is applied
        **algorithm_kwargs: Arguments to pass to algorithm
        
    Returns:
        Algorithm result (potentially on sampled graph)
    """
    
    total_nodes = len(hypergraph.nodes)
    
    if total_nodes <= max_nodes:
        logger.info(f"Running {algorithm_name} on full graph ({total_nodes} nodes)")
        return algorithm_func(hypergraph, **algorithm_kwargs)
    
    # Use smart sampling
    logger.info(f"Graph too large ({total_nodes} nodes), applying smart sampling")
    sampler = SmartSampler(hypergraph, strategy='adaptive')
    
    # Determine optimal sample size for the algorithm
    sample_size = sampler.determine_optimal_sample_size(
        target_algorithm=algorithm_name.lower(),
        max_nodes=max_nodes
    )
    
    sampled_graph = sampler.adaptive_sample(sample_size, algorithm_name.lower())
    
    logger.info(f"Running {algorithm_name} on sampled graph ({len(sampled_graph.nodes)} nodes)")
    result = algorithm_func(sampled_graph, **algorithm_kwargs)
    
    # For clustering results, don't extend to full graph (too expensive)
    # Just return results for sampled nodes with a warning
    if isinstance(result, dict) and sampled_graph.nodes:
        sample_node = next(iter(sampled_graph.nodes))
        if sample_node in result:
            logger.info(f"Clustering result contains {len(result)} sampled nodes out of {len(hypergraph.nodes)} total nodes")
            # For performance, we don't extend clustering results to full graph
    
    return result


def _extend_node_results(full_graph, sampled_graph, sample_results: Dict) -> Dict:
    """
    Extend node-based results from sample to full graph
    
    Uses neighbor-based interpolation for missing nodes.
    """
    
    extended_results = sample_results.copy()
    sampled_nodes = set(sampled_graph.nodes)
    missing_nodes = set(full_graph.nodes) - sampled_nodes
    
    if not missing_nodes:
        return extended_results
    
    logger.info(f"Extending results to {len(missing_nodes)} missing nodes")
    
    # For each missing node, estimate value based on neighbors
    for missing_node in missing_nodes:
        # Find neighbors of missing node in full graph
        neighbor_edges = full_graph.incidences.get_node_edges(missing_node)
        neighbor_values = []
        
        for edge in neighbor_edges:
            edge_nodes = full_graph.incidences.get_edge_nodes(edge)
            for node in edge_nodes:
                if node != missing_node and node in extended_results:
                    neighbor_values.append(extended_results[node])
        
        if neighbor_values:
            # Use mean of neighbor values
            extended_results[missing_node] = np.mean(neighbor_values)
        else:
            # Use global mean as fallback
            if sample_results:
                extended_results[missing_node] = np.mean(list(sample_results.values()))
            else:
                extended_results[missing_node] = 0.0
    
    return extended_results


def get_sampling_recommendations(hypergraph) -> Dict[str, Any]:
    """
    Get sampling recommendations for a hypergraph
    
    Args:
        hypergraph: Input hypergraph
        
    Returns:
        Dictionary with sampling recommendations
    """
    
    sampler = SmartSampler(hypergraph)
    stats = sampler._graph_stats
    
    recommendations = {
        'total_nodes': stats['num_nodes'],
        'total_edges': stats['num_edges'],
        'recommended_sampling': stats['num_nodes'] > 1000,
        'optimal_sample_sizes': {
            'clustering': sampler.determine_optimal_sample_size('clustering'),
            'centrality': sampler.determine_optimal_sample_size('centrality'),
            'structural': sampler.determine_optimal_sample_size('structural'),
            'general': sampler.determine_optimal_sample_size('general')
        },
        'recommended_strategy': 'degree_based' if stats['degree_distribution']['std'] > stats['degree_distribution']['mean'] * 0.8 else 'stratified',
        'graph_characteristics': {
            'avg_degree': stats['degree_distribution']['mean'],
            'degree_variance': stats['degree_distribution']['std'],
            'avg_edge_size': stats['avg_edge_size'],
            'density_category': 'dense' if stats['avg_edge_size'] > 5 else 'sparse' if stats['avg_edge_size'] < 2 else 'moderate'
        }
    }
    
    return recommendations