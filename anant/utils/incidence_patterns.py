"""
Incidence Pattern Analysis Module for Anant

This module provides comprehensive analysis of incidence patterns in hypergraphs,
including motif detection, structural pattern recognition, topological analysis,
and pattern-based clustering and classification.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
from enum import Enum
import polars as pl
import numpy as np
import math
from dataclasses import dataclass
from collections import defaultdict, Counter
import itertools

from .property_types import PropertyType, PropertyTypeManager


class PatternType(Enum):
    """Types of incidence patterns"""
    STAR = "star"                    # One node connected to many edges
    CHAIN = "chain"                  # Linear sequence of connections
    CYCLE = "cycle"                  # Circular connection pattern
    CLIQUE = "clique"               # All nodes connected to same edges
    BIPARTITE = "bipartite"         # Two distinct node sets
    TREE = "tree"                   # Hierarchical pattern
    MESH = "mesh"                   # Grid-like pattern
    HUB = "hub"                     # High-degree central node
    BRIDGE = "bridge"               # Edge connecting components
    COMMUNITY = "community"         # Dense substructure


class MotifSize(Enum):
    """Size categories for motifs"""
    SMALL = "small"      # 2-3 nodes
    MEDIUM = "medium"    # 4-6 nodes  
    LARGE = "large"      # 7+ nodes


@dataclass
class IncidenceMotif:
    """Detected incidence motif"""
    motif_id: str
    pattern_type: PatternType
    size: MotifSize
    nodes: List[Any]
    edges: List[Any]
    frequency: int
    significance_score: float
    structural_properties: Dict[str, Any]


@dataclass
class PatternStatistics:
    """Statistics for incidence patterns"""
    pattern_type: PatternType
    total_count: int
    size_distribution: Dict[int, int]
    avg_size: float
    frequency_distribution: List[int]
    centrality_scores: Dict[str, float]


@dataclass
class TopologicalFeatures:
    """Topological features of incidence structure"""
    connectivity: float
    clustering_coefficient: float
    path_length_distribution: Dict[int, int]
    degree_distribution: Dict[int, int]
    betweenness_centrality: Dict[Any, float]
    closeness_centrality: Dict[Any, float]
    eigenvector_centrality: Dict[Any, float]


class IncidencePatternAnalyzer:
    """
    Comprehensive analyzer for incidence patterns in hypergraphs
    
    Features:
    - Motif detection and characterization
    - Structural pattern recognition
    - Topological analysis
    - Pattern-based clustering
    - Anomalous pattern detection
    - Pattern evolution tracking
    """
    
    def __init__(self, max_motif_size: int = 6, significance_threshold: float = 0.05):
        self.max_motif_size = max_motif_size
        self.significance_threshold = significance_threshold
        self.type_manager = PropertyTypeManager()
        self.pattern_cache = {}
        self.motif_cache = {}
        
    def detect_incidence_motifs(
        self,
        incidence_df: pl.DataFrame,
        node_col: str = "node",
        edge_col: str = "edge",
        min_frequency: int = 2
    ) -> List[IncidenceMotif]:
        """
        Detect recurring motifs in incidence structure
        
        Args:
            incidence_df: DataFrame with node-edge incidence relationships
            node_col: Column name for nodes
            edge_col: Column name for edges
            min_frequency: Minimum frequency for motif detection
            
        Returns:
            List of detected motifs
        """
        if node_col not in incidence_df.columns or edge_col not in incidence_df.columns:
            return []
            
        motifs = []
        
        # Build adjacency structure
        edge_to_nodes = defaultdict(set)
        node_to_edges = defaultdict(set)
        
        for row in incidence_df.iter_rows():
            node = row[incidence_df.columns.index(node_col)]
            edge = row[incidence_df.columns.index(edge_col)]
            edge_to_nodes[edge].add(node)
            node_to_edges[node].add(edge)
        
        # Detect different motif types
        star_motifs = self._detect_star_motifs(node_to_edges, edge_to_nodes, min_frequency)
        chain_motifs = self._detect_chain_motifs(node_to_edges, edge_to_nodes, min_frequency)
        clique_motifs = self._detect_clique_motifs(node_to_edges, edge_to_nodes, min_frequency)
        hub_motifs = self._detect_hub_motifs(node_to_edges, min_frequency)
        
        motifs.extend(star_motifs)
        motifs.extend(chain_motifs)
        motifs.extend(clique_motifs)
        motifs.extend(hub_motifs)
        
        # Sort by significance score
        motifs.sort(key=lambda x: x.significance_score, reverse=True)
        
        return motifs
    
    def _detect_star_motifs(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]],
        min_frequency: int
    ) -> List[IncidenceMotif]:
        """Detect star patterns (one edge connecting multiple nodes)"""
        motifs = []
        star_patterns = defaultdict(int)
        
        # Group by edge size (number of nodes)
        for edge, nodes in edge_to_nodes.items():
            if len(nodes) >= 3:  # Star requires at least 3 nodes
                size = len(nodes)
                star_patterns[size] += 1
        
        # Create motifs for frequent star sizes
        motif_id = 0
        for size, frequency in star_patterns.items():
            if frequency >= min_frequency:
                # Find example edges of this size
                example_edges = [
                    edge for edge, nodes in edge_to_nodes.items() 
                    if len(nodes) == size
                ]
                if not example_edges:
                    continue
                    
                example_edges = example_edges[:min(5, len(example_edges))]  # Limit examples
                
                example_nodes = []
                for edge in example_edges:
                    example_nodes.extend(list(edge_to_nodes[edge]))
                
                motif_size = MotifSize.SMALL if size <= 3 else (
                    MotifSize.MEDIUM if size <= 6 else MotifSize.LARGE
                )
                
                significance = self._calculate_motif_significance(
                    frequency, len(edge_to_nodes), PatternType.STAR
                )
                
                motif = IncidenceMotif(
                    motif_id=f"star_{size}_{motif_id}",
                    pattern_type=PatternType.STAR,
                    size=motif_size,
                    nodes=list(set(example_nodes)),
                    edges=example_edges,
                    frequency=frequency,
                    significance_score=significance,
                    structural_properties={
                        "star_size": size,
                        "edge_count": len(example_edges),
                        "node_count": len(set(example_nodes))
                    }
                )
                motifs.append(motif)
                motif_id += 1
        
        return motifs
    
    def _detect_chain_motifs(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]],
        min_frequency: int
    ) -> List[IncidenceMotif]:
        """Detect chain patterns (linear sequences)"""
        motifs = []
        
        # Find nodes that connect exactly 2 edges (potential chain elements)
        chain_nodes = [
            node for node, edges in node_to_edges.items() 
            if len(edges) == 2
        ]
        
        if len(chain_nodes) < min_frequency:
            return motifs
        
        # Group chain nodes by their connecting edge pairs
        chain_patterns = defaultdict(list)
        for node in chain_nodes:
            edges = tuple(sorted(node_to_edges[node]))
            chain_patterns[edges].append(node)
        
        motif_id = 0
        for edge_pair, nodes in chain_patterns.items():
            if len(nodes) >= min_frequency:
                # Calculate chain length
                all_nodes_in_edges = set()
                for edge in edge_pair:
                    all_nodes_in_edges.update(edge_to_nodes[edge])
                
                chain_length = len(all_nodes_in_edges)
                motif_size = MotifSize.SMALL if chain_length <= 3 else (
                    MotifSize.MEDIUM if chain_length <= 6 else MotifSize.LARGE
                )
                
                significance = self._calculate_motif_significance(
                    len(nodes), len(chain_nodes), PatternType.CHAIN
                )
                
                motif = IncidenceMotif(
                    motif_id=f"chain_{motif_id}",
                    pattern_type=PatternType.CHAIN,
                    size=motif_size,
                    nodes=list(all_nodes_in_edges),
                    edges=list(edge_pair),
                    frequency=len(nodes),
                    significance_score=significance,
                    structural_properties={
                        "chain_length": chain_length,
                        "bridge_nodes": len(nodes),
                        "edge_pair": edge_pair
                    }
                )
                motifs.append(motif)
                motif_id += 1
        
        return motifs
    
    def _detect_clique_motifs(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]],
        min_frequency: int
    ) -> List[IncidenceMotif]:
        """Detect clique patterns (nodes sharing multiple edges)"""
        motifs = []
        
        # Find sets of nodes that share multiple edges
        node_pairs_to_edges = defaultdict(set)
        
        for edge, nodes in edge_to_nodes.items():
            if len(nodes) >= 2:
                for node1, node2 in itertools.combinations(nodes, 2):
                    pair = tuple(sorted([node1, node2]))
                    node_pairs_to_edges[pair].add(edge)
        
        # Group by number of shared edges
        clique_patterns = defaultdict(list)
        for pair, shared_edges in node_pairs_to_edges.items():
            if len(shared_edges) >= 2:  # At least 2 shared edges for clique
                clique_patterns[len(shared_edges)].append((pair, shared_edges))
        
        motif_id = 0
        for shared_count, cliques in clique_patterns.items():
            if len(cliques) >= min_frequency:
                # Take a sample of cliques
                sample_cliques = cliques[:min(5, len(cliques))]
                
                all_nodes = set()
                all_edges = set()
                for pair, edges in sample_cliques:
                    all_nodes.update(pair)
                    all_edges.update(edges)
                
                motif_size = MotifSize.SMALL if len(all_nodes) <= 3 else (
                    MotifSize.MEDIUM if len(all_nodes) <= 6 else MotifSize.LARGE
                )
                
                significance = self._calculate_motif_significance(
                    len(cliques), len(node_pairs_to_edges), PatternType.CLIQUE
                )
                
                motif = IncidenceMotif(
                    motif_id=f"clique_{shared_count}_{motif_id}",
                    pattern_type=PatternType.CLIQUE,
                    size=motif_size,
                    nodes=list(all_nodes),
                    edges=list(all_edges),
                    frequency=len(cliques),
                    significance_score=significance,
                    structural_properties={
                        "shared_edges": shared_count,
                        "clique_count": len(cliques),
                        "density": len(all_edges) / (len(all_nodes) * (len(all_nodes) - 1) / 2) if len(all_nodes) > 1 else 0
                    }
                )
                motifs.append(motif)
                motif_id += 1
        
        return motifs
    
    def _detect_hub_motifs(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        min_frequency: int
    ) -> List[IncidenceMotif]:
        """Detect hub patterns (high-degree nodes)"""
        motifs = []
        
        # Calculate degree distribution
        degrees = {node: len(edges) for node, edges in node_to_edges.items()}
        degree_counts = Counter(degrees.values())
        
        # Find degree thresholds for hubs
        sorted_degrees = sorted(degrees.values(), reverse=True)
        if len(sorted_degrees) < min_frequency:
            return motifs
        
        # Consider top 10% as potential hubs
        hub_threshold = sorted_degrees[min(len(sorted_degrees) // 10, len(sorted_degrees) - 1)]
        
        hub_nodes = [node for node, degree in degrees.items() if degree >= hub_threshold]
        
        if len(hub_nodes) >= min_frequency:
            # Group hubs by similar degree
            hub_groups = defaultdict(list)
            for node in hub_nodes:
                degree_range = (degrees[node] // 5) * 5  # Group by 5s
                hub_groups[degree_range].append(node)
            
            motif_id = 0
            for degree_range, nodes in hub_groups.items():
                if len(nodes) >= min_frequency:
                    # Collect edges for these hubs
                    all_edges = set()
                    for node in nodes[:5]:  # Limit to first 5 for efficiency
                        all_edges.update(node_to_edges[node])
                    
                    avg_degree = sum(degrees[node] for node in nodes) / len(nodes)
                    motif_size = MotifSize.SMALL if avg_degree <= 3 else (
                        MotifSize.MEDIUM if avg_degree <= 6 else MotifSize.LARGE
                    )
                    
                    significance = self._calculate_motif_significance(
                        len(nodes), len(node_to_edges), PatternType.HUB
                    )
                    
                    motif = IncidenceMotif(
                        motif_id=f"hub_{degree_range}_{motif_id}",
                        pattern_type=PatternType.HUB,
                        size=motif_size,
                        nodes=nodes[:10],  # Limit node list
                        edges=list(all_edges)[:20],  # Limit edge list
                        frequency=len(nodes),
                        significance_score=significance,
                        structural_properties={
                            "avg_degree": avg_degree,
                            "degree_range": degree_range,
                            "hub_count": len(nodes),
                            "total_edges": len(all_edges)
                        }
                    )
                    motifs.append(motif)
                    motif_id += 1
        
        return motifs
    
    def _calculate_motif_significance(
        self,
        observed_frequency: int,
        total_possible: int,
        pattern_type: PatternType
    ) -> float:
        """Calculate statistical significance of motif frequency"""
        if total_possible == 0:
            return 0.0
            
        # Simple frequency-based significance
        frequency_ratio = observed_frequency / total_possible
        
        # Weight by pattern complexity
        complexity_weight = {
            PatternType.STAR: 1.0,
            PatternType.CHAIN: 1.2,
            PatternType.CLIQUE: 1.5,
            PatternType.HUB: 1.3
        }.get(pattern_type, 1.0)
        
        significance = frequency_ratio * complexity_weight * math.log(observed_frequency + 1)
        return min(significance, 1.0)  # Cap at 1.0
    
    def analyze_pattern_statistics(
        self,
        motifs: List[IncidenceMotif]
    ) -> Dict[PatternType, PatternStatistics]:
        """
        Analyze statistics of detected patterns
        
        Args:
            motifs: List of detected motifs
            
        Returns:
            Dictionary mapping pattern types to their statistics
        """
        pattern_stats = {}
        
        # Group motifs by type
        motifs_by_type = defaultdict(list)
        for motif in motifs:
            motifs_by_type[motif.pattern_type].append(motif)
        
        for pattern_type, type_motifs in motifs_by_type.items():
            # Size distribution
            sizes = []
            for motif in type_motifs:
                if 'star_size' in motif.structural_properties:
                    sizes.append(motif.structural_properties['star_size'])
                elif 'chain_length' in motif.structural_properties:
                    sizes.append(motif.structural_properties['chain_length'])
                else:
                    sizes.append(len(motif.nodes))
            
            size_dist = Counter(sizes)
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            
            # Frequency distribution
            frequencies = [motif.frequency for motif in type_motifs]
            
            # Centrality scores (placeholder - would compute from actual graph)
            centrality_scores = {
                "betweenness": 0.5,
                "closeness": 0.5,
                "eigenvector": 0.5
            }
            
            stats = PatternStatistics(
                pattern_type=pattern_type,
                total_count=len(type_motifs),
                size_distribution=dict(size_dist),
                avg_size=avg_size,
                frequency_distribution=frequencies,
                centrality_scores=centrality_scores
            )
            
            pattern_stats[pattern_type] = stats
        
        return pattern_stats
    
    def compute_topological_features(
        self,
        incidence_df: pl.DataFrame,
        node_col: str = "node",
        edge_col: str = "edge"
    ) -> TopologicalFeatures:
        """
        Compute topological features of incidence structure
        
        Args:
            incidence_df: DataFrame with node-edge incidence relationships
            node_col: Column name for nodes
            edge_col: Column name for edges
            
        Returns:
            TopologicalFeatures object
        """
        if node_col not in incidence_df.columns or edge_col not in incidence_df.columns:
            return TopologicalFeatures(
                connectivity=0.0,
                clustering_coefficient=0.0,
                path_length_distribution={},
                degree_distribution={},
                betweenness_centrality={},
                closeness_centrality={},
                eigenvector_centrality={}
            )
        
        # Build adjacency structures
        edge_to_nodes = defaultdict(set)
        node_to_edges = defaultdict(set)
        
        for row in incidence_df.iter_rows():
            node = row[incidence_df.columns.index(node_col)]
            edge = row[incidence_df.columns.index(edge_col)]
            edge_to_nodes[edge].add(node)
            node_to_edges[node].add(edge)
        
        # Compute basic features
        connectivity = self._compute_connectivity(node_to_edges, edge_to_nodes)
        clustering_coeff = self._compute_clustering_coefficient(node_to_edges, edge_to_nodes)
        
        # Degree distribution
        degrees = {node: len(edges) for node, edges in node_to_edges.items()}
        degree_dist = Counter(degrees.values())
        
        # Path length distribution (simplified)
        path_lengths = self._compute_path_lengths(node_to_edges, edge_to_nodes)
        
        # Centrality measures (simplified implementations)
        betweenness = self._compute_betweenness_centrality(node_to_edges, edge_to_nodes)
        closeness = self._compute_closeness_centrality(node_to_edges, edge_to_nodes)
        eigenvector = self._compute_eigenvector_centrality(node_to_edges, edge_to_nodes)
        
        return TopologicalFeatures(
            connectivity=connectivity,
            clustering_coefficient=clustering_coeff,
            path_length_distribution=dict(path_lengths),
            degree_distribution=dict(degree_dist),
            betweenness_centrality=betweenness,
            closeness_centrality=closeness,
            eigenvector_centrality=eigenvector
        )
    
    def _compute_connectivity(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> float:
        """Compute overall connectivity measure"""
        if not node_to_edges or not edge_to_nodes:
            return 0.0
        
        total_nodes = len(node_to_edges)
        total_edges = len(edge_to_nodes)
        
        if total_nodes <= 1:
            return 0.0
        
        # Maximum possible connections
        max_connections = total_nodes * (total_nodes - 1) / 2
        
        # Actual connections (approximate)
        actual_connections = sum(len(edges) for edges in node_to_edges.values()) / 2
        
        return min(actual_connections / max_connections, 1.0)
    
    def _compute_clustering_coefficient(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> float:
        """Compute clustering coefficient"""
        if len(node_to_edges) < 3:
            return 0.0
        
        clustering_coeffs = []
        
        for node, edges in node_to_edges.items():
            if len(edges) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Find neighbors (nodes sharing edges)
            neighbors = set()
            for edge in edges:
                neighbors.update(edge_to_nodes[edge])
            neighbors.discard(node)
            
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count connections between neighbors
            neighbor_connections = 0
            for n1, n2 in itertools.combinations(neighbors, 2):
                shared_edges = node_to_edges[n1] & node_to_edges[n2]
                if shared_edges:
                    neighbor_connections += 1
            
            # Clustering coefficient for this node
            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            if possible_connections > 0:
                clustering_coeffs.append(neighbor_connections / possible_connections)
            else:
                clustering_coeffs.append(0.0)
        
        return sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _compute_path_lengths(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> Counter:
        """Compute distribution of shortest path lengths"""
        # Simplified implementation using BFS
        path_lengths = []
        nodes = list(node_to_edges.keys())
        
        # Sample a subset for efficiency
        sample_size = min(10, len(nodes))
        sample_nodes = nodes[:sample_size]
        
        for start_node in sample_nodes:
            visited = {start_node}
            current_level = {start_node}
            distance = 0
            
            while current_level and distance < 6:  # Limit search depth
                next_level = set()
                for node in current_level:
                    # Find neighbors through shared edges
                    for edge in node_to_edges[node]:
                        for neighbor in edge_to_nodes[edge]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_level.add(neighbor)
                                path_lengths.append(distance + 1)
                
                current_level = next_level
                distance += 1
        
        return Counter(path_lengths)
    
    def _compute_betweenness_centrality(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> Dict[Any, float]:
        """Compute simplified betweenness centrality"""
        centrality = {}
        
        for node in node_to_edges:
            # Simplified: based on degree and position
            degree = len(node_to_edges[node])
            
            # Count how many edge pairs this node bridges
            bridge_count = 0
            for e1, e2 in itertools.combinations(node_to_edges[node], 2):
                # Check if these edges have few other shared nodes
                shared_nodes = edge_to_nodes[e1] & edge_to_nodes[e2]
                if len(shared_nodes) <= 2:  # This node is important bridge
                    bridge_count += 1
            
            centrality[node] = float(bridge_count * degree) / (len(node_to_edges) + 1)
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1
        if max_centrality > 0:
            centrality = {node: val / max_centrality for node, val in centrality.items()}
        
        return centrality
    
    def _compute_closeness_centrality(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> Dict[Any, float]:
        """Compute simplified closeness centrality"""
        centrality = {}
        
        for node in node_to_edges:
            # Simplified: based on inverse average distance to others
            total_distance = 0
            reachable_count = 0
            
            # BFS to compute distances
            visited = {node}
            current_level = {node}
            distance = 0
            
            while current_level and distance < 4:  # Limit depth
                next_level = set()
                distance += 1
                
                for n in current_level:
                    for edge in node_to_edges[n]:
                        for neighbor in edge_to_nodes[edge]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_level.add(neighbor)
                                total_distance += distance
                                reachable_count += 1
                
                current_level = next_level
            
            if reachable_count > 0:
                centrality[node] = float(reachable_count) / total_distance
            else:
                centrality[node] = 0.0
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1
        if max_centrality > 0:
            centrality = {node: val / max_centrality for node, val in centrality.items()}
        
        return centrality
    
    def _compute_eigenvector_centrality(
        self,
        node_to_edges: Dict[Any, Set[Any]],
        edge_to_nodes: Dict[Any, Set[Any]]
    ) -> Dict[Any, float]:
        """Compute simplified eigenvector centrality"""
        centrality = {}
        
        # Simplified: iterative approach
        nodes = list(node_to_edges.keys())
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return centrality
        
        # Initialize with equal values
        scores = {node: 1.0 for node in nodes}
        
        # Power iteration (simplified)
        for _ in range(10):  # 10 iterations
            new_scores = {node: 0.0 for node in nodes}
            
            for node in nodes:
                # Score based on neighbors' scores
                for edge in node_to_edges[node]:
                    for neighbor in edge_to_nodes[edge]:
                        if neighbor != node:
                            new_scores[node] += scores[neighbor]
            
            # Normalize
            total_score = sum(new_scores.values())
            if total_score > 0:
                scores = {node: score / total_score for node, score in new_scores.items()}
            else:
                break
        
        return scores
    
    def detect_anomalous_patterns(
        self,
        motifs: List[IncidenceMotif],
        significance_threshold: Optional[float] = None
    ) -> List[IncidenceMotif]:
        """
        Detect anomalous patterns based on statistical significance
        
        Args:
            motifs: List of detected motifs
            significance_threshold: Threshold for anomaly detection
            
        Returns:
            List of anomalous motifs
        """
        if significance_threshold is None:
            significance_threshold = self.significance_threshold
        
        # Calculate significance distribution
        significances = [motif.significance_score for motif in motifs]
        if not significances:
            return []
        
        mean_sig = np.mean(significances)
        std_sig = np.std(significances)
        
        anomalous_motifs = []
        
        for motif in motifs:
            # Z-score based anomaly detection
            z_score = (motif.significance_score - mean_sig) / (std_sig + 1e-10)
            
            # Consider both very high and very low significance as anomalous
            if abs(z_score) > 2.0:  # 2 standard deviations
                anomalous_motifs.append(motif)
        
        return anomalous_motifs
    
    def generate_pattern_report(
        self,
        incidence_df: pl.DataFrame,
        node_col: str = "node",
        edge_col: str = "edge"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pattern analysis report
        
        Args:
            incidence_df: DataFrame with node-edge incidence relationships
            node_col: Column name for nodes
            edge_col: Column name for edges
            
        Returns:
            Comprehensive pattern analysis report
        """
        # Detect motifs
        motifs = self.detect_incidence_motifs(incidence_df, node_col, edge_col)
        
        # Analyze pattern statistics
        pattern_stats = self.analyze_pattern_statistics(motifs)
        
        # Compute topological features
        topo_features = self.compute_topological_features(incidence_df, node_col, edge_col)
        
        # Detect anomalies
        anomalous_patterns = self.detect_anomalous_patterns(motifs)
        
        # Build report
        report = {
            "dataset_summary": {
                "total_nodes": len(incidence_df[node_col].unique()),
                "total_edges": len(incidence_df[edge_col].unique()),
                "total_incidences": len(incidence_df)
            },
            "motif_analysis": {
                "total_motifs": len(motifs),
                "motifs_by_type": {
                    pattern_type.value: len([m for m in motifs if m.pattern_type == pattern_type])
                    for pattern_type in PatternType
                },
                "top_motifs": sorted(motifs, key=lambda x: x.significance_score, reverse=True)[:10]
            },
            "pattern_statistics": {
                pattern_type.value: {
                    "total_count": stats.total_count,
                    "avg_size": stats.avg_size,
                    "size_distribution": stats.size_distribution
                }
                for pattern_type, stats in pattern_stats.items()
            },
            "topological_features": {
                "connectivity": topo_features.connectivity,
                "clustering_coefficient": topo_features.clustering_coefficient,
                "degree_distribution": dict(list(topo_features.degree_distribution.items())[:10]),  # Top 10
                "avg_path_length": sum(
                    length * count for length, count in topo_features.path_length_distribution.items()
                ) / sum(topo_features.path_length_distribution.values()) if topo_features.path_length_distribution else 0
            },
            "anomalous_patterns": {
                "count": len(anomalous_patterns),
                "patterns": [
                    {
                        "motif_id": motif.motif_id,
                        "pattern_type": motif.pattern_type.value,
                        "significance_score": motif.significance_score,
                        "frequency": motif.frequency
                    }
                    for motif in anomalous_patterns[:10]  # Top 10 anomalies
                ]
            }
        }
        
        return report