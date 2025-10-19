"""
Hypergraph Pattern Detection
===========================

Advanced pattern detection and motif analysis for hypergraphs,
including structural similarity and anomaly detection.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import logging
from collections import defaultdict, Counter
from itertools import combinations
from ..utils.decorators import performance_monitor
from ..utils.extras import safe_import

logger = logging.getLogger(__name__)

# Optional dependencies
scipy = safe_import('scipy')
networkx = safe_import('networkx')


@performance_monitor
def find_hypergraph_motifs(hypergraph,
                          motif_size: int = 3,
                          max_motifs: int = 100,
                          min_frequency: int = 2) -> Dict[str, Any]:
    """
    Find recurring motifs (substructures) in the hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        motif_size: Size of motifs to search for (number of nodes)
        max_motifs: Maximum number of motifs to return
        min_frequency: Minimum frequency for a pattern to be considered a motif
        
    Returns:
        Dictionary with discovered motifs and their frequencies
    """
    try:
        data = hypergraph.incidences.data
        
        if len(hypergraph.nodes) < motif_size:
            logger.warning(f"Graph too small for motif size {motif_size}")
            return {'motifs': [], 'statistics': {}}
        
        # Extract edge patterns
        edge_patterns = defaultdict(list)
        
        for edge_id in hypergraph.edges:
            edge_nodes = (
                data
                .filter(pl.col(hypergraph.incidences.edge_column) == edge_id)
                [hypergraph.incidences.node_column]
                .to_list()
            )
            
            if len(edge_nodes) >= motif_size:
                # Generate all combinations of specified size
                for node_combo in combinations(sorted(edge_nodes), motif_size):
                    pattern_key = tuple(node_combo)
                    edge_patterns[pattern_key].append(edge_id)
        
        # Find frequently occurring patterns
        frequent_patterns = {
            pattern: edges for pattern, edges in edge_patterns.items()
            if len(edges) >= min_frequency
        }
        
        # Analyze structural properties of motifs
        motifs = []
        for pattern, edges in frequent_patterns.items():
            motif_analysis = _analyze_motif_structure(hypergraph, pattern, edges)
            motifs.append({
                'pattern': pattern,
                'frequency': len(edges),
                'edges': edges,
                'analysis': motif_analysis
            })
        
        # Sort by frequency and limit results
        motifs.sort(key=lambda x: x['frequency'], reverse=True)
        motifs = motifs[:max_motifs]
        
        statistics = {
            'total_patterns_found': len(edge_patterns),
            'frequent_patterns': len(frequent_patterns),
            'motif_size': motif_size,
            'min_frequency': min_frequency,
            'avg_frequency': np.mean([m['frequency'] for m in motifs]) if motifs else 0
        }
        
        logger.info(f"Found {len(motifs)} motifs of size {motif_size}")
        return {'motifs': motifs, 'statistics': statistics}
        
    except Exception as e:
        logger.error(f"Error finding hypergraph motifs: {e}")
        return {'motifs': [], 'statistics': {}}


@performance_monitor
def detect_recurring_patterns(hypergraph,
                             pattern_types: List[str] = ['edge_size', 'node_degree', 'weight_distribution'],
                             min_support: float = 0.1) -> Dict[str, Any]:
    """
    Detect various types of recurring patterns in the hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        pattern_types: Types of patterns to detect
        min_support: Minimum support threshold for pattern significance
        
    Returns:
        Dictionary with detected patterns by type
    """
    try:
        patterns = {}
        
        if 'edge_size' in pattern_types:
            patterns['edge_size'] = _detect_edge_size_patterns(hypergraph, min_support)
        
        if 'node_degree' in pattern_types:
            patterns['node_degree'] = _detect_node_degree_patterns(hypergraph, min_support)
        
        if 'weight_distribution' in pattern_types:
            patterns['weight_distribution'] = _detect_weight_patterns(hypergraph, min_support)
        
        if 'connectivity' in pattern_types:
            patterns['connectivity'] = _detect_connectivity_patterns(hypergraph, min_support)
        
        # Cross-pattern analysis
        patterns['cross_patterns'] = _analyze_cross_patterns(patterns)
        
        logger.info(f"Detected patterns across {len(pattern_types)} categories")
        return patterns
        
    except Exception as e:
        logger.error(f"Error detecting recurring patterns: {e}")
        return {}


@performance_monitor
def structural_similarity_analysis(hypergraph,
                                 similarity_metric: str = 'jaccard',
                                 threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyze structural similarity between nodes and edges.
    
    Args:
        hypergraph: Anant Hypergraph instance
        similarity_metric: Similarity metric ('jaccard', 'cosine', 'hamming')
        threshold: Threshold for considering nodes/edges similar
        
    Returns:
        Dictionary with similarity analysis results
    """
    try:
        # Node similarity analysis
        node_similarity = _calculate_node_similarity(hypergraph, similarity_metric, threshold)
        
        # Edge similarity analysis
        edge_similarity = _calculate_edge_similarity(hypergraph, similarity_metric, threshold)
        
        # Find similarity clusters
        similarity_clusters = _find_similarity_clusters(node_similarity, edge_similarity, threshold)
        
        analysis = {
            'node_similarity': node_similarity,
            'edge_similarity': edge_similarity,
            'similarity_clusters': similarity_clusters,
            'similarity_metric': similarity_metric,
            'threshold': threshold,
            'statistics': {
                'avg_node_similarity': _calculate_avg_similarity(node_similarity),
                'avg_edge_similarity': _calculate_avg_similarity(edge_similarity),
                'num_node_clusters': len(similarity_clusters.get('node_clusters', [])),
                'num_edge_clusters': len(similarity_clusters.get('edge_clusters', []))
            }
        }
        
        logger.info(f"Completed structural similarity analysis with {similarity_metric} metric")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in structural similarity analysis: {e}")
        return {}


@performance_monitor
def anomaly_detection(hypergraph,
                     anomaly_types: List[str] = ['outlier_nodes', 'outlier_edges', 'unusual_patterns'],
                     sensitivity: float = 2.0) -> Dict[str, Any]:
    """
    Detect anomalies and outliers in the hypergraph structure.
    
    Args:
        hypergraph: Anant Hypergraph instance
        anomaly_types: Types of anomalies to detect
        sensitivity: Sensitivity for anomaly detection (higher = more sensitive)
        
    Returns:
        Dictionary with detected anomalies
    """
    try:
        anomalies = {}
        
        if 'outlier_nodes' in anomaly_types:
            anomalies['outlier_nodes'] = _detect_outlier_nodes(hypergraph, sensitivity)
        
        if 'outlier_edges' in anomaly_types:
            anomalies['outlier_edges'] = _detect_outlier_edges(hypergraph, sensitivity)
        
        if 'unusual_patterns' in anomaly_types:
            anomalies['unusual_patterns'] = _detect_unusual_patterns(hypergraph, sensitivity)
        
        if 'weight_anomalies' in anomaly_types:
            anomalies['weight_anomalies'] = _detect_weight_anomalies(hypergraph, sensitivity)
        
        # Summary statistics
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values() if isinstance(anomaly_list, list))
        
        anomalies['summary'] = {
            'total_anomalies': total_anomalies,
            'anomaly_types_detected': len(anomalies),
            'sensitivity_used': sensitivity,
            'anomaly_rate': total_anomalies / (len(hypergraph.nodes) + len(hypergraph.edges))
        }
        
        logger.info(f"Detected {total_anomalies} anomalies across {len(anomaly_types)} categories")
        return anomalies
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {}


@performance_monitor
def pattern_frequency_analysis(hypergraph,
                              pattern_sizes: List[int] = [2, 3, 4],
                              frequency_threshold: int = 1) -> Dict[str, Any]:
    """
    Analyze frequency of different structural patterns.
    
    Args:
        hypergraph: Anant Hypergraph instance
        pattern_sizes: List of pattern sizes to analyze
        frequency_threshold: Minimum frequency for pattern inclusion
        
    Returns:
        Dictionary with pattern frequency analysis
    """
    try:
        frequency_analysis = {}
        
        for size in pattern_sizes:
            if size <= len(hypergraph.nodes):
                size_patterns = find_hypergraph_motifs(
                    hypergraph, size, max_motifs=1000, min_frequency=frequency_threshold
                )
                frequency_analysis[f'size_{size}'] = size_patterns
        
        # Analyze frequency distributions
        all_frequencies = []
        for size_data in frequency_analysis.values():
            frequencies = [motif['frequency'] for motif in size_data.get('motifs', [])]
            all_frequencies.extend(frequencies)
        
        if all_frequencies:
            frequency_stats = {
                'mean_frequency': float(np.mean(all_frequencies)),
                'std_frequency': float(np.std(all_frequencies)),
                'max_frequency': int(np.max(all_frequencies)),
                'min_frequency': int(np.min(all_frequencies)),
                'total_patterns': len(all_frequencies)
            }
        else:
            frequency_stats = {'mean_frequency': 0, 'total_patterns': 0}
        
        # Find power law characteristics
        power_law_analysis = _analyze_frequency_distribution(all_frequencies)
        
        result = {
            'pattern_frequencies': frequency_analysis,
            'frequency_statistics': frequency_stats,
            'power_law_analysis': power_law_analysis,
            'pattern_sizes_analyzed': pattern_sizes
        }
        
        logger.info(f"Analyzed pattern frequencies for sizes {pattern_sizes}")
        return result
        
    except Exception as e:
        logger.error(f"Error in pattern frequency analysis: {e}")
        return {}


def _analyze_motif_structure(hypergraph, pattern: Tuple, edges: List[str]) -> Dict[str, Any]:
    """Analyze structural properties of a motif."""
    try:
        data = hypergraph.incidences.data
        
        # Calculate motif statistics
        motif_edges_data = data.filter(
            pl.col(hypergraph.incidences.edge_column).is_in(edges)
        )
        
        # Node participation in motif
        node_participation = (
            motif_edges_data
            .group_by(hypergraph.incidences.node_column)
            .agg([pl.count().alias('edge_count')])
        )
        
        analysis = {
            'node_count': len(pattern),
            'edge_count': len(edges),
            'avg_node_participation': float(node_participation['edge_count'].mean()) if len(node_participation) > 0 else 0,
            'density': len(edges) / len(pattern) if len(pattern) > 0 else 0,
            'pattern_nodes': list(pattern)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing motif structure: {e}")
        return {}


def _detect_edge_size_patterns(hypergraph, min_support: float) -> Dict[str, Any]:
    """Detect patterns in edge sizes."""
    try:
        data = hypergraph.incidences.data
        
        edge_sizes = (
            data
            .group_by(hypergraph.incidences.edge_column)
            .agg([pl.count().alias('edge_size')])
        )
        
        size_counts = edge_sizes['edge_size'].value_counts().sort('edge_size')
        total_edges = len(hypergraph.edges)
        
        # Find frequent size patterns
        frequent_sizes = []
        for row in size_counts.iter_rows(named=True):
            size = row['edge_size']
            count = row['count']
            support = count / total_edges
            
            if support >= min_support:
                frequent_sizes.append({
                    'size': size,
                    'count': count,
                    'support': support
                })
        
        return {
            'frequent_sizes': frequent_sizes,
            'size_distribution': size_counts.to_dict(as_series=False),
            'total_edges': total_edges
        }
        
    except Exception as e:
        logger.error(f"Error detecting edge size patterns: {e}")
        return {}


def _detect_node_degree_patterns(hypergraph, min_support: float) -> Dict[str, Any]:
    """Detect patterns in node degrees."""
    try:
        data = hypergraph.incidences.data
        
        node_degrees = (
            data
            .group_by(hypergraph.incidences.node_column)
            .agg([pl.count().alias('degree')])
        )
        
        degree_counts = node_degrees['degree'].value_counts().sort('degree')
        total_nodes = len(hypergraph.nodes)
        
        # Find frequent degree patterns
        frequent_degrees = []
        for row in degree_counts.iter_rows(named=True):
            degree = row['degree']
            count = row['count']
            support = count / total_nodes
            
            if support >= min_support:
                frequent_degrees.append({
                    'degree': degree,
                    'count': count,
                    'support': support
                })
        
        return {
            'frequent_degrees': frequent_degrees,
            'degree_distribution': degree_counts.to_dict(as_series=False),
            'total_nodes': total_nodes
        }
        
    except Exception as e:
        logger.error(f"Error detecting node degree patterns: {e}")
        return {}


def _detect_weight_patterns(hypergraph, min_support: float) -> Dict[str, Any]:
    """Detect patterns in edge weights."""
    try:
        data = hypergraph.incidences.data
        
        # Find numeric columns that could be weights
        numeric_columns = [
            col for col in data.columns 
            if col not in [hypergraph.incidences.node_column, hypergraph.incidences.edge_column]
            and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if not numeric_columns:
            return {'message': 'No numeric columns found for weight analysis'}
        
        weight_patterns = {}
        
        for col in numeric_columns:
            # Bin weights and find frequent patterns
            weight_data = data[col].drop_nulls()
            if len(weight_data) == 0:
                continue
            
            # Create bins
            q1, q3 = weight_data.quantile(0.25), weight_data.quantile(0.75)
            bins = [weight_data.min(), q1, q3, weight_data.max()]
            
            # Simple binning logic
            bin_labels = []
            for val in weight_data.to_list():
                if val <= q1:
                    bin_labels.append('low')
                elif val <= q3:
                    bin_labels.append('medium')
                else:
                    bin_labels.append('high')
            
            bin_counts = Counter(bin_labels)
            total_values = len(bin_labels)
            
            frequent_bins = {
                bin_name: {'count': count, 'support': count / total_values}
                for bin_name, count in bin_counts.items()
                if count / total_values >= min_support
            }
            
            weight_patterns[col] = {
                'frequent_bins': frequent_bins,
                'total_values': total_values,
                'bins_used': bins
            }
        
        return weight_patterns
        
    except Exception as e:
        logger.error(f"Error detecting weight patterns: {e}")
        return {}


def _detect_connectivity_patterns(hypergraph, min_support: float) -> Dict[str, Any]:
    """Detect connectivity patterns."""
    try:
        data = hypergraph.incidences.data
        
        # Analyze node co-occurrence patterns
        edge_node_sets = {}
        for edge_id in hypergraph.edges:
            edge_nodes = (
                data
                .filter(pl.col(hypergraph.incidences.edge_column) == edge_id)
                [hypergraph.incidences.node_column]
                .to_list()
            )
            edge_node_sets[edge_id] = set(edge_nodes)
        
        # Find frequently co-occurring node pairs
        pair_counts = Counter()
        for nodes in edge_node_sets.values():
            if len(nodes) >= 2:
                for pair in combinations(sorted(nodes), 2):
                    pair_counts[pair] += 1
        
        total_pairs = len(pair_counts)
        frequent_pairs = [
            {
                'pair': pair,
                'count': count,
                'support': count / total_pairs
            }
            for pair, count in pair_counts.items()
            if count / total_pairs >= min_support
        ]
        
        return {
            'frequent_pairs': frequent_pairs[:100],  # Limit results
            'total_unique_pairs': total_pairs,
            'avg_co_occurrence': np.mean(list(pair_counts.values())) if pair_counts else 0
        }
        
    except Exception as e:
        logger.error(f"Error detecting connectivity patterns: {e}")
        return {}


def _analyze_cross_patterns(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze relationships between different pattern types."""
    try:
        cross_analysis = {
            'pattern_correlations': {},
            'combined_insights': []
        }
        
        # This would require more sophisticated analysis
        # For now, provide basic cross-pattern statistics
        pattern_counts = {
            pattern_type: len(pattern_data.get('frequent_patterns', pattern_data.get('frequent_sizes', [])))
            for pattern_type, pattern_data in patterns.items()
            if isinstance(pattern_data, dict)
        }
        
        cross_analysis['pattern_type_counts'] = pattern_counts
        cross_analysis['total_patterns'] = sum(pattern_counts.values())
        
        return cross_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing cross patterns: {e}")
        return {}


def _calculate_node_similarity(hypergraph, metric: str, threshold: float) -> Dict[str, Any]:
    """Calculate similarity between nodes."""
    try:
        data = hypergraph.incidences.data
        
        # Create node-edge incidence matrix
        nodes = list(hypergraph.nodes)
        edges = list(hypergraph.edges)
        
        node_edge_matrix = {}
        for node in nodes:
            node_edges = set(
                data
                .filter(pl.col(hypergraph.incidences.node_column) == node)
                [hypergraph.incidences.edge_column]
                .to_list()
            )
            node_edge_matrix[node] = node_edges
        
        # Calculate pairwise similarities
        similarities = {}
        for i, node1 in enumerate(nodes):
            similarities[node1] = {}
            for j, node2 in enumerate(nodes[i:], i):
                if node1 == node2:
                    similarity = 1.0
                else:
                    edges1, edges2 = node_edge_matrix[node1], node_edge_matrix[node2]
                    
                    if metric == 'jaccard':
                        if len(edges1 | edges2) == 0:
                            similarity = 0.0
                        else:
                            similarity = len(edges1 & edges2) / len(edges1 | edges2)
                    elif metric == 'cosine':
                        if len(edges1) == 0 or len(edges2) == 0:
                            similarity = 0.0
                        else:
                            similarity = len(edges1 & edges2) / np.sqrt(len(edges1) * len(edges2))
                    else:
                        # Default to Jaccard
                        similarity = len(edges1 & edges2) / len(edges1 | edges2) if len(edges1 | edges2) > 0 else 0.0
                
                similarities[node1][node2] = similarity
                if node1 != node2:
                    if node2 not in similarities:
                        similarities[node2] = {}
                    similarities[node2][node1] = similarity
        
        return similarities
        
    except Exception as e:
        logger.error(f"Error calculating node similarity: {e}")
        return {}


def _calculate_edge_similarity(hypergraph, metric: str, threshold: float) -> Dict[str, Any]:
    """Calculate similarity between edges."""
    try:
        data = hypergraph.incidences.data
        
        # Create edge-node matrix
        edges = list(hypergraph.edges)
        edge_node_matrix = {}
        
        for edge in edges:
            edge_nodes = set(
                data
                .filter(pl.col(hypergraph.incidences.edge_column) == edge)
                [hypergraph.incidences.node_column]
                .to_list()
            )
            edge_node_matrix[edge] = edge_nodes
        
        # Calculate pairwise similarities
        similarities = {}
        for i, edge1 in enumerate(edges):
            similarities[edge1] = {}
            for j, edge2 in enumerate(edges[i:], i):
                if edge1 == edge2:
                    similarity = 1.0
                else:
                    nodes1, nodes2 = edge_node_matrix[edge1], edge_node_matrix[edge2]
                    
                    if metric == 'jaccard':
                        if len(nodes1 | nodes2) == 0:
                            similarity = 0.0
                        else:
                            similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2)
                    elif metric == 'cosine':
                        if len(nodes1) == 0 or len(nodes2) == 0:
                            similarity = 0.0
                        else:
                            similarity = len(nodes1 & nodes2) / np.sqrt(len(nodes1) * len(nodes2))
                    else:
                        similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2) if len(nodes1 | nodes2) > 0 else 0.0
                
                similarities[edge1][edge2] = similarity
                if edge1 != edge2:
                    if edge2 not in similarities:
                        similarities[edge2] = {}
                    similarities[edge2][edge1] = similarity
        
        return similarities
        
    except Exception as e:
        logger.error(f"Error calculating edge similarity: {e}")
        return {}


def _find_similarity_clusters(node_similarity: Dict, edge_similarity: Dict, threshold: float) -> Dict[str, Any]:
    """Find clusters of similar nodes and edges."""
    try:
        # Simple clustering based on similarity threshold
        node_clusters = []
        edge_clusters = []
        
        # Node clustering
        used_nodes = set()
        for node1, similarities in node_similarity.items():
            if node1 in used_nodes:
                continue
            
            cluster = {node1}
            used_nodes.add(node1)
            
            for node2, similarity in similarities.items():
                if node2 not in used_nodes and similarity >= threshold:
                    cluster.add(node2)
                    used_nodes.add(node2)
            
            if len(cluster) > 1:
                node_clusters.append(list(cluster))
        
        # Edge clustering
        used_edges = set()
        for edge1, similarities in edge_similarity.items():
            if edge1 in used_edges:
                continue
            
            cluster = {edge1}
            used_edges.add(edge1)
            
            for edge2, similarity in similarities.items():
                if edge2 not in used_edges and similarity >= threshold:
                    cluster.add(edge2)
                    used_edges.add(edge2)
            
            if len(cluster) > 1:
                edge_clusters.append(list(cluster))
        
        return {
            'node_clusters': node_clusters,
            'edge_clusters': edge_clusters
        }
        
    except Exception as e:
        logger.error(f"Error finding similarity clusters: {e}")
        return {'node_clusters': [], 'edge_clusters': []}


def _calculate_avg_similarity(similarities: Dict[str, Dict[str, float]]) -> float:
    """Calculate average similarity across all pairs."""
    try:
        all_similarities = []
        for node1, sim_dict in similarities.items():
            for node2, similarity in sim_dict.items():
                if node1 != node2:  # Exclude self-similarity
                    all_similarities.append(similarity)
        
        return float(np.mean(all_similarities)) if all_similarities else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating average similarity: {e}")
        return 0.0


def _detect_outlier_nodes(hypergraph, sensitivity: float) -> List[Dict[str, Any]]:
    """Detect outlier nodes based on degree and other properties."""
    try:
        data = hypergraph.incidences.data
        
        # Calculate node degrees
        node_degrees = (
            data
            .group_by(hypergraph.incidences.node_column)
            .agg([pl.count().alias('degree')])
        )
        
        degrees = node_degrees['degree'].to_numpy()
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        
        # Find outliers using z-score
        outliers = []
        for row in node_degrees.iter_rows(named=True):
            node = row[hypergraph.incidences.node_column]
            degree = row['degree']
            z_score = abs(degree - mean_degree) / std_degree if std_degree > 0 else 0
            
            if z_score > sensitivity:
                outliers.append({
                    'node': node,
                    'degree': degree,
                    'z_score': z_score,
                    'type': 'high_degree' if degree > mean_degree else 'low_degree'
                })
        
        return outliers
        
    except Exception as e:
        logger.error(f"Error detecting outlier nodes: {e}")
        return []


def _detect_outlier_edges(hypergraph, sensitivity: float) -> List[Dict[str, Any]]:
    """Detect outlier edges based on size and other properties."""
    try:
        data = hypergraph.incidences.data
        
        # Calculate edge sizes
        edge_sizes = (
            data
            .group_by(hypergraph.incidences.edge_column)
            .agg([pl.count().alias('size')])
        )
        
        sizes = edge_sizes['size'].to_numpy()
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Find outliers using z-score
        outliers = []
        for row in edge_sizes.iter_rows(named=True):
            edge = row[hypergraph.incidences.edge_column]
            size = row['size']
            z_score = abs(size - mean_size) / std_size if std_size > 0 else 0
            
            if z_score > sensitivity:
                outliers.append({
                    'edge': edge,
                    'size': size,
                    'z_score': z_score,
                    'type': 'large_edge' if size > mean_size else 'small_edge'
                })
        
        return outliers
        
    except Exception as e:
        logger.error(f"Error detecting outlier edges: {e}")
        return []


def _detect_unusual_patterns(hypergraph, sensitivity: float) -> List[Dict[str, Any]]:
    """Detect unusual structural patterns."""
    try:
        # Find nodes or edges with unusual connectivity patterns
        unusual_patterns = []
        
        # Look for isolated components
        data = hypergraph.incidences.data
        
        # Find nodes with very unique edge participation patterns
        node_edge_participation = defaultdict(set)
        edge_node_participation = defaultdict(set)
        
        for row in data.iter_rows(named=True):
            node = row[hypergraph.incidences.node_column]
            edge = row[hypergraph.incidences.edge_column]
            node_edge_participation[node].add(edge)
            edge_node_participation[edge].add(node)
        
        # Find nodes that participate in edges with very different sizes
        for node, edges in node_edge_participation.items():
            edge_sizes = [len(edge_node_participation[edge]) for edge in edges]
            if len(edge_sizes) > 1:
                size_variance = np.var(edge_sizes)
                if size_variance > sensitivity * np.mean(edge_sizes):
                    unusual_patterns.append({
                        'type': 'diverse_edge_participation',
                        'node': node,
                        'edge_sizes': edge_sizes,
                        'variance': size_variance
                    })
        
        return unusual_patterns
        
    except Exception as e:
        logger.error(f"Error detecting unusual patterns: {e}")
        return []


def _detect_weight_anomalies(hypergraph, sensitivity: float) -> List[Dict[str, Any]]:
    """Detect anomalies in edge weights."""
    try:
        data = hypergraph.incidences.data
        
        # Find numeric columns
        numeric_columns = [
            col for col in data.columns 
            if col not in [hypergraph.incidences.node_column, hypergraph.incidences.edge_column]
            and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if not numeric_columns:
            return []
        
        anomalies = []
        
        for col in numeric_columns:
            values = data[col].drop_nulls().to_numpy()
            if len(values) < 2:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                continue
            
            # Find outliers
            for i, val in enumerate(values):
                z_score = abs(val - mean_val) / std_val
                if z_score > sensitivity:
                    anomalies.append({
                        'column': col,
                        'value': val,
                        'z_score': z_score,
                        'type': 'weight_outlier'
                    })
        
        return anomalies[:100]  # Limit results
        
    except Exception as e:
        logger.error(f"Error detecting weight anomalies: {e}")
        return []


def _analyze_frequency_distribution(frequencies: List[int]) -> Dict[str, Any]:
    """Analyze frequency distribution for power law characteristics."""
    try:
        if not frequencies:
            return {'power_law_fit': False, 'analysis': 'No data'}
        
        frequencies = np.array(frequencies)
        frequencies = frequencies[frequencies > 0]  # Remove zeros
        
        if len(frequencies) < 5:
            return {'power_law_fit': False, 'analysis': 'Insufficient data'}
        
        # Simple power law analysis
        log_freq = np.log(frequencies)
        log_rank = np.log(np.arange(1, len(frequencies) + 1))
        
        # Linear regression in log-log space
        slope, intercept = np.polyfit(log_rank, log_freq, 1)
        
        # R-squared
        y_pred = slope * log_rank + intercept
        ss_res = np.sum((log_freq - y_pred) ** 2)
        ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        power_law_fit = r_squared > 0.8 and slope < -0.5  # Reasonable power law criteria
        
        return {
            'power_law_fit': power_law_fit,
            'slope': slope,
            'r_squared': r_squared,
            'analysis': 'Power law detected' if power_law_fit else 'No clear power law'
        }
        
    except Exception as e:
        logger.error(f"Error analyzing frequency distribution: {e}")
        return {'power_law_fit': False, 'analysis': 'Analysis failed'}