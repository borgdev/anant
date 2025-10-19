"""
Temporal Analysis for Hypergraphs

This module implements temporal analysis algorithms for dynamic hypergraphs,
including evolution metrics, temporal clustering, and time-series analytics.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from ..classes.hypergraph import Hypergraph
from .centrality import degree_centrality, s_centrality, eigenvector_centrality
from .clustering import spectral_clustering, community_detection


@dataclass
class TemporalSnapshot:
    """Represents a hypergraph at a specific time point"""
    timestamp: Union[int, float, str]
    hypergraph: Hypergraph
    metadata: Optional[Dict[str, Any]] = None


class TemporalHypergraph:
    """
    Temporal hypergraph for dynamic analysis.
    
    Manages a sequence of hypergraph snapshots over time and provides
    methods for temporal analysis.
    """
    
    def __init__(self, snapshots: Optional[List[TemporalSnapshot]] = None):
        """
        Initialize temporal hypergraph.
        
        Args:
            snapshots: List of temporal snapshots (optional)
        """
        self.snapshots = snapshots or []
        self._sorted = False
    
    def add_snapshot(self, snapshot: TemporalSnapshot):
        """Add a temporal snapshot"""
        self.snapshots.append(snapshot)
        self._sorted = False
    
    def sort_snapshots(self):
        """Sort snapshots by timestamp"""
        if not self._sorted:
            self.snapshots.sort(key=lambda s: s.timestamp)
            self._sorted = True
    
    @property
    def timestamps(self) -> List[Union[int, float, str]]:
        """Get all timestamps"""
        return [s.timestamp for s in self.snapshots]
    
    @property
    def num_snapshots(self) -> int:
        """Number of temporal snapshots"""
        return len(self.snapshots)
    
    def get_snapshot(self, timestamp: Union[int, float, str]) -> Optional[TemporalSnapshot]:
        """Get snapshot for specific timestamp"""
        for snapshot in self.snapshots:
            if snapshot.timestamp == timestamp:
                return snapshot
        return None
    
    def get_snapshot_at_index(self, index: int) -> TemporalSnapshot:
        """Get snapshot at specific index"""
        self.sort_snapshots()
        return self.snapshots[index]


def temporal_degree_evolution(
    temporal_hg: TemporalHypergraph,
    node_id: Optional[str] = None
) -> Union[Dict[str, pl.DataFrame], pl.DataFrame]:
    """
    Analyze degree evolution over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        node_id: Specific node to analyze (if None, analyze all nodes)
        
    Returns:
        DataFrame with degree evolution data
    """
    temporal_hg.sort_snapshots()
    
    evolution_data = []
    
    for snapshot in temporal_hg.snapshots:
        hg = snapshot.hypergraph
        timestamp = snapshot.timestamp
        
        # Calculate node degree centralities
        degree_cents = degree_centrality(hg, normalized=True)['nodes']
        
        for node, degree in degree_cents.items():
            if node_id is None or node == node_id:
                evolution_data.append({
                    'timestamp': timestamp,
                    'node': node,
                    'degree': degree,
                    'num_edges': hg.num_edges,
                    'num_nodes': hg.num_nodes
                })
    
    evolution_df = pl.DataFrame(evolution_data)
    
    if node_id is not None:
        return evolution_df.filter(pl.col('node') == node_id)
    else:
        # Return dictionary with DataFrames for each node
        result = {}
        for node in evolution_df['node'].unique():
            result[node] = evolution_df.filter(pl.col('node') == node)
        return result


def temporal_centrality_evolution(
    temporal_hg: TemporalHypergraph,
    centrality_measures: List[str] = ['degree', 's_centrality', 'eigenvector'],
    s_parameter: int = 1
) -> pl.DataFrame:
    """
    Analyze evolution of multiple centrality measures over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        centrality_measures: List of centrality measures to compute
        s_parameter: Parameter for s-centrality
        
    Returns:
        DataFrame with centrality evolution data
    """
    temporal_hg.sort_snapshots()
    
    evolution_data = []
    
    for snapshot in temporal_hg.snapshots:
        hg = snapshot.hypergraph
        timestamp = snapshot.timestamp
        
        # Compute centrality measures
        centralities = {}
        
        if 'degree' in centrality_measures:
            centralities['degree'] = degree_centrality(hg, normalized=True)['nodes']
        
        if 's_centrality' in centrality_measures:
            centralities['s_centrality'] = s_centrality(hg, s=s_parameter, normalized=True)
        
        if 'eigenvector' in centrality_measures:
            centralities['eigenvector'] = eigenvector_centrality(hg, normalized=True)
        
        # Collect data for all nodes
        all_nodes = set()
        for measure_dict in centralities.values():
            all_nodes.update(measure_dict.keys())
        
        for node in all_nodes:
            row_data = {
                'timestamp': timestamp,
                'node': node,
                'num_edges': hg.num_edges,
                'num_nodes': hg.num_nodes
            }
            
            # Add centrality scores
            for measure_name, measure_dict in centralities.items():
                row_data[measure_name] = measure_dict.get(node, 0.0)
            
            evolution_data.append(row_data)
    
    return pl.DataFrame(evolution_data)


def temporal_clustering_evolution(
    temporal_hg: TemporalHypergraph,
    n_clusters: int = 3,
    method: str = "spectral"
) -> pl.DataFrame:
    """
    Analyze clustering evolution over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        n_clusters: Number of clusters to find
        method: Clustering method ("spectral" or "community")
        
    Returns:
        DataFrame with clustering evolution data
    """
    temporal_hg.sort_snapshots()
    
    evolution_data = []
    
    for snapshot in temporal_hg.snapshots:
        hg = snapshot.hypergraph
        timestamp = snapshot.timestamp
        
        try:
            if method == "spectral":
                clusters = spectral_clustering(hg, n_clusters=n_clusters, method="node")
            elif method == "community":
                clusters = community_detection(hg, method="modularity")
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Record cluster assignments
            for node, cluster_id in clusters.items():
                evolution_data.append({
                    'timestamp': timestamp,
                    'node': node,
                    'cluster': cluster_id,
                    'num_edges': hg.num_edges,
                    'num_nodes': hg.num_nodes,
                    'method': method
                })
                
        except Exception as e:
            # Skip snapshots that can't be clustered
            print(f"Warning: Could not cluster snapshot at {timestamp}: {e}")
            continue
    
    return pl.DataFrame(evolution_data)


def stability_analysis(
    temporal_hg: TemporalHypergraph,
    window_size: int = 3,
    metric: str = "jaccard"
) -> pl.DataFrame:
    """
    Analyze structural stability over time using sliding windows.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        window_size: Size of sliding window for stability analysis
        metric: Stability metric ("jaccard", "node_overlap", "edge_overlap")
        
    Returns:
        DataFrame with stability scores over time
    """
    temporal_hg.sort_snapshots()
    
    if len(temporal_hg.snapshots) < window_size:
        raise ValueError(f"Need at least {window_size} snapshots for stability analysis")
    
    stability_data = []
    
    for i in range(len(temporal_hg.snapshots) - window_size + 1):
        window_snapshots = temporal_hg.snapshots[i:i + window_size]
        
        # Calculate stability within window
        total_stability = 0.0
        comparisons = 0
        
        for j in range(len(window_snapshots)):
            for k in range(j + 1, len(window_snapshots)):
                hg1 = window_snapshots[j].hypergraph
                hg2 = window_snapshots[k].hypergraph
                
                if metric == "jaccard":
                    stability = jaccard_similarity(hg1, hg2)
                elif metric == "node_overlap":
                    stability = node_overlap(hg1, hg2)
                elif metric == "edge_overlap":
                    stability = edge_overlap(hg1, hg2)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                total_stability += stability
                comparisons += 1
        
        avg_stability = total_stability / comparisons if comparisons > 0 else 0.0
        
        # Use middle timestamp of window
        middle_idx = i + window_size // 2
        middle_timestamp = temporal_hg.snapshots[middle_idx].timestamp
        
        stability_data.append({
            'timestamp': middle_timestamp,
            'window_start': window_snapshots[0].timestamp,
            'window_end': window_snapshots[-1].timestamp,
            'stability_score': avg_stability,
            'window_size': window_size,
            'metric': metric
        })
    
    return pl.DataFrame(stability_data)


def jaccard_similarity(hg1: Hypergraph, hg2: Hypergraph) -> float:
    """Calculate Jaccard similarity between two hypergraphs"""
    # Compare edge sets
    edges1 = set(hg1.edges)
    edges2 = set(hg2.edges)
    
    if len(edges1) == 0 and len(edges2) == 0:
        return 1.0
    
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    
    return intersection / union if union > 0 else 0.0


def node_overlap(hg1: Hypergraph, hg2: Hypergraph) -> float:
    """Calculate node overlap between two hypergraphs"""
    nodes1 = set(hg1.nodes)
    nodes2 = set(hg2.nodes)
    
    if len(nodes1) == 0 and len(nodes2) == 0:
        return 1.0
    
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    
    return intersection / union if union > 0 else 0.0


def edge_overlap(hg1: Hypergraph, hg2: Hypergraph) -> float:
    """Calculate edge content overlap between two hypergraphs"""
    # Compare actual edge contents, not just edge IDs
    total_similarity = 0.0
    total_edges = 0
    
    all_edges = set(hg1.edges).union(set(hg2.edges))
    
    for edge in all_edges:
        nodes1 = set()
        nodes2 = set()
        
        if edge in hg1.edges:
            # Get nodes for this edge in hg1
            edge_nodes1 = (
                hg1._incidence_store.data
                .filter(pl.col("edges") == edge)
                .select("nodes")
                .to_series()
                .to_list()
            )
            nodes1 = set(edge_nodes1)
        
        if edge in hg2.edges:
            # Get nodes for this edge in hg2
            edge_nodes2 = (
                hg2._incidence_store.data
                .filter(pl.col("edges") == edge)
                .select("nodes")
                .to_series()
                .to_list()
            )
            nodes2 = set(edge_nodes2)
        
        # Calculate Jaccard similarity for this edge
        if len(nodes1) == 0 and len(nodes2) == 0:
            edge_similarity = 1.0
        else:
            intersection = len(nodes1.intersection(nodes2))
            union = len(nodes1.union(nodes2))
            edge_similarity = intersection / union if union > 0 else 0.0
        
        total_similarity += edge_similarity
        total_edges += 1
    
    return total_similarity / total_edges if total_edges > 0 else 0.0


def temporal_motif_analysis(
    temporal_hg: TemporalHypergraph,
    motif_sizes: List[int] = [2, 3, 4]
) -> pl.DataFrame:
    """
    Analyze evolution of hypergraph motifs over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        motif_sizes: List of motif sizes to analyze
        
    Returns:
        DataFrame with motif counts over time
    """
    temporal_hg.sort_snapshots()
    
    motif_data = []
    
    for snapshot in temporal_hg.snapshots:
        hg = snapshot.hypergraph
        timestamp = snapshot.timestamp
        
        for motif_size in motif_sizes:
            # Count edges of specific size (motifs)
            edge_size_counts = (
                hg._incidence_store.data
                .group_by("edges")
                .agg(pl.count("nodes").alias("edge_size"))
                .filter(pl.col("edge_size") == motif_size)
                .shape[0]
            )
            
            motif_data.append({
                'timestamp': timestamp,
                'motif_size': motif_size,
                'count': edge_size_counts,
                'total_edges': hg.num_edges,
                'proportion': edge_size_counts / hg.num_edges if hg.num_edges > 0 else 0.0
            })
    
    return pl.DataFrame(motif_data)


def growth_analysis(
    temporal_hg: TemporalHypergraph,
    smoothing_window: int = 1
) -> pl.DataFrame:
    """
    Analyze growth patterns of hypergraph over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        smoothing_window: Window size for smoothing growth rates
        
    Returns:
        DataFrame with growth analysis data
    """
    temporal_hg.sort_snapshots()
    
    growth_data = []
    
    for i, snapshot in enumerate(temporal_hg.snapshots):
        hg = snapshot.hypergraph
        timestamp = snapshot.timestamp
        
        # Basic size metrics
        num_nodes = hg.num_nodes
        num_edges = hg.num_edges
        num_incidences = hg.num_incidences
        
        # Growth rates (if not first snapshot)
        node_growth_rate = 0.0
        edge_growth_rate = 0.0
        incidence_growth_rate = 0.0
        
        if i > 0:
            prev_hg = temporal_hg.snapshots[i - 1].hypergraph
            
            node_growth_rate = (num_nodes - prev_hg.num_nodes) / prev_hg.num_nodes if prev_hg.num_nodes > 0 else 0.0
            edge_growth_rate = (num_edges - prev_hg.num_edges) / prev_hg.num_edges if prev_hg.num_edges > 0 else 0.0
            incidence_growth_rate = (num_incidences - prev_hg.num_incidences) / prev_hg.num_incidences if prev_hg.num_incidences > 0 else 0.0
        
        # Density metrics
        node_density = num_incidences / (num_nodes * num_edges) if (num_nodes * num_edges) > 0 else 0.0
        avg_edge_size = num_incidences / num_edges if num_edges > 0 else 0.0
        avg_node_degree = num_incidences / num_nodes if num_nodes > 0 else 0.0
        
        growth_data.append({
            'timestamp': timestamp,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_incidences': num_incidences,
            'node_growth_rate': node_growth_rate,
            'edge_growth_rate': edge_growth_rate,
            'incidence_growth_rate': incidence_growth_rate,
            'node_density': node_density,
            'avg_edge_size': avg_edge_size,
            'avg_node_degree': avg_node_degree
        })
    
    growth_df = pl.DataFrame(growth_data)
    
    # Apply smoothing if requested
    if smoothing_window > 1:
        growth_cols = ['node_growth_rate', 'edge_growth_rate', 'incidence_growth_rate']
        
        for col in growth_cols:
            smoothed_col = f"{col}_smoothed"
            growth_df = growth_df.with_columns([
                pl.col(col).rolling_mean(window_size=smoothing_window).alias(smoothed_col)
            ])
    
    return growth_df


def persistence_analysis(
    temporal_hg: TemporalHypergraph,
    entity_type: str = "nodes"
) -> pl.DataFrame:
    """
    Analyze persistence of nodes or edges over time.
    
    Args:
        temporal_hg: Temporal hypergraph instance
        entity_type: "nodes" or "edges" to analyze
        
    Returns:
        DataFrame with persistence analysis data
    """
    temporal_hg.sort_snapshots()
    
    if entity_type not in ["nodes", "edges"]:
        raise ValueError("entity_type must be 'nodes' or 'edges'")
    
    # Track appearance and disappearance of entities
    entity_timeline = {}
    
    for i, snapshot in enumerate(temporal_hg.snapshots):
        timestamp = snapshot.timestamp
        
        if entity_type == "nodes":
            entities = set(snapshot.hypergraph.nodes)
        else:
            entities = set(snapshot.hypergraph.edges)
        
        for entity in entities:
            if entity not in entity_timeline:
                entity_timeline[entity] = {
                    'first_appearance': timestamp,
                    'last_appearance': timestamp,
                    'appearances': [timestamp],
                    'first_index': i,
                    'last_index': i
                }
            else:
                entity_timeline[entity]['last_appearance'] = timestamp
                entity_timeline[entity]['appearances'].append(timestamp)
                entity_timeline[entity]['last_index'] = i
    
    # Calculate persistence metrics
    persistence_data = []
    
    for entity, timeline in entity_timeline.items():
        appearances = timeline['appearances']
        total_snapshots = len(temporal_hg.snapshots)
        
        # Persistence metrics
        lifespan = timeline['last_index'] - timeline['first_index'] + 1
        persistence_ratio = len(appearances) / lifespan
        global_persistence = len(appearances) / total_snapshots
        
        # Intermittency (gaps in appearance)
        expected_appearances = lifespan
        actual_appearances = len(appearances)
        intermittency = 1.0 - (actual_appearances / expected_appearances)
        
        persistence_data.append({
            'entity': entity,
            'entity_type': entity_type,
            'first_appearance': timeline['first_appearance'],
            'last_appearance': timeline['last_appearance'],
            'lifespan_snapshots': lifespan,
            'total_appearances': actual_appearances,
            'persistence_ratio': persistence_ratio,
            'global_persistence': global_persistence,
            'intermittency': intermittency
        })
    
    return pl.DataFrame(persistence_data)