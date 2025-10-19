"""
Property Correlation Analysis
============================

Advanced correlation analysis for hypergraph node and edge properties,
including cross-property analysis and correlation visualization.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from collections import defaultdict
from ..utils.decorators import performance_monitor
from ..utils.extras import safe_import

logger = logging.getLogger(__name__)

# Optional dependencies
scipy = safe_import('scipy')
sklearn = safe_import('sklearn')


@performance_monitor
def property_correlation_analysis(hypergraph,
                                 property_columns: Optional[List[str]] = None,
                                 correlation_method: str = 'pearson',
                                 min_correlation: float = 0.1) -> Dict[str, Any]:
    """
    Comprehensive property correlation analysis for hypergraph.
    
    Args:
        hypergraph: Anant Hypergraph instance
        property_columns: List of property columns to analyze
        correlation_method: Correlation method ('pearson', 'spearman', 'kendall')
        min_correlation: Minimum correlation threshold for reporting
        
    Returns:
        Dictionary with correlation analysis results
    """
    try:
        data = hypergraph.incidences.data
        
        # Auto-detect numeric columns if not specified
        if property_columns is None:
            property_columns = [
                col for col in data.columns 
                if col not in [hypergraph.incidences.node_column, hypergraph.incidences.edge_column]
                and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
        
        if len(property_columns) < 2:
            logger.warning("Need at least 2 numeric properties for correlation analysis")
            return {'correlations': {}, 'significant_pairs': [], 'summary': {}}
        
        # Node-level correlation analysis
        node_correlations = node_property_correlations(
            hypergraph, property_columns, correlation_method, min_correlation
        )
        
        # Edge-level correlation analysis
        edge_correlations = edge_property_correlations(
            hypergraph, property_columns, correlation_method, min_correlation
        )
        
        # Cross-property analysis
        cross_analysis = cross_property_analysis(
            hypergraph, property_columns, correlation_method
        )
        
        # Compile results
        results = {
            'correlations': {
                'node_level': node_correlations,
                'edge_level': edge_correlations,
                'cross_property': cross_analysis
            },
            'significant_pairs': _extract_significant_pairs(
                [node_correlations, edge_correlations], min_correlation
            ),
            'summary': {
                'total_properties': len(property_columns),
                'node_significant_pairs': len(node_correlations.get('significant_pairs', [])),
                'edge_significant_pairs': len(edge_correlations.get('significant_pairs', [])),
                'correlation_method': correlation_method,
                'min_correlation_threshold': min_correlation
            }
        }
        
        logger.info(f"Property correlation analysis completed for {len(property_columns)} properties")
        return results
        
    except Exception as e:
        logger.error(f"Error in property correlation analysis: {e}")
        return {'correlations': {}, 'significant_pairs': [], 'summary': {}}


@performance_monitor
def node_property_correlations(hypergraph,
                              property_columns: List[str],
                              correlation_method: str = 'pearson',
                              min_correlation: float = 0.1) -> Dict[str, Any]:
    """
    Analyze correlations between node-level properties.
    
    Args:
        hypergraph: Anant Hypergraph instance
        property_columns: List of property columns to analyze
        correlation_method: Correlation method
        min_correlation: Minimum correlation threshold
        
    Returns:
        Dictionary with node-level correlation results
    """
    try:
        data = hypergraph.incidences.data
        
        # Aggregate properties at node level
        node_aggregations = []
        for col in property_columns:
            if col in data.columns:
                node_aggregations.extend([
                    pl.col(col).mean().alias(f'{col}_mean'),
                    pl.col(col).std().alias(f'{col}_std'),
                    pl.col(col).sum().alias(f'{col}_sum'),
                    pl.col(col).count().alias(f'{col}_count')
                ])
        
        node_data = (
            data
            .group_by(hypergraph.incidences.node_column)
            .agg(node_aggregations)
        )
        
        # Calculate correlations
        correlation_matrix = {}
        significant_pairs = []
        
        aggregated_columns = [col for col in node_data.columns 
                            if col != hypergraph.incidences.node_column]
        
        for i, col1 in enumerate(aggregated_columns):
            correlation_matrix[col1] = {}
            for j, col2 in enumerate(aggregated_columns):
                if i <= j:
                    correlation = _calculate_correlation(
                        node_data, col1, col2, correlation_method
                    )
                    correlation_matrix[col1][col2] = correlation
                    
                    if abs(correlation) >= min_correlation and col1 != col2:
                        significant_pairs.append({
                            'property1': col1,
                            'property2': col2,
                            'correlation': correlation,
                            'abs_correlation': abs(correlation)
                        })
        
        # Sort significant pairs by correlation strength
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix,
            'significant_pairs': significant_pairs,
            'node_count': len(node_data),
            'property_count': len(aggregated_columns)
        }
        
    except Exception as e:
        logger.error(f"Error in node property correlations: {e}")
        return {'correlation_matrix': {}, 'significant_pairs': [], 'node_count': 0}


@performance_monitor
def edge_property_correlations(hypergraph,
                              property_columns: List[str],
                              correlation_method: str = 'pearson',
                              min_correlation: float = 0.1) -> Dict[str, Any]:
    """
    Analyze correlations between edge-level properties.
    
    Args:
        hypergraph: Anant Hypergraph instance
        property_columns: List of property columns to analyze
        correlation_method: Correlation method
        min_correlation: Minimum correlation threshold
        
    Returns:
        Dictionary with edge-level correlation results
    """
    try:
        data = hypergraph.incidences.data
        
        # Aggregate properties at edge level
        edge_aggregations = []
        for col in property_columns:
            if col in data.columns:
                edge_aggregations.extend([
                    pl.col(col).mean().alias(f'{col}_mean'),
                    pl.col(col).std().alias(f'{col}_std'),
                    pl.col(col).sum().alias(f'{col}_sum'),
                    pl.col(col).max().alias(f'{col}_max'),
                    pl.col(col).min().alias(f'{col}_min')
                ])
        
        # Add edge size
        edge_aggregations.append(pl.count().alias('edge_size'))
        
        edge_data = (
            data
            .group_by(hypergraph.incidences.edge_column)
            .agg(edge_aggregations)
        )
        
        # Calculate correlations
        correlation_matrix = {}
        significant_pairs = []
        
        aggregated_columns = [col for col in edge_data.columns 
                            if col != hypergraph.incidences.edge_column]
        
        for i, col1 in enumerate(aggregated_columns):
            correlation_matrix[col1] = {}
            for j, col2 in enumerate(aggregated_columns):
                if i <= j:
                    correlation = _calculate_correlation(
                        edge_data, col1, col2, correlation_method
                    )
                    correlation_matrix[col1][col2] = correlation
                    
                    if abs(correlation) >= min_correlation and col1 != col2:
                        significant_pairs.append({
                            'property1': col1,
                            'property2': col2,
                            'correlation': correlation,
                            'abs_correlation': abs(correlation)
                        })
        
        # Sort significant pairs by correlation strength
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix,
            'significant_pairs': significant_pairs,
            'edge_count': len(edge_data),
            'property_count': len(aggregated_columns)
        }
        
    except Exception as e:
        logger.error(f"Error in edge property correlations: {e}")
        return {'correlation_matrix': {}, 'significant_pairs': [], 'edge_count': 0}


@performance_monitor
def cross_property_analysis(hypergraph,
                           property_columns: List[str],
                           correlation_method: str = 'pearson') -> Dict[str, Any]:
    """
    Analyze cross-level correlations between node and edge properties.
    
    Args:
        hypergraph: Anant Hypergraph instance
        property_columns: List of property columns to analyze
        correlation_method: Correlation method
        
    Returns:
        Dictionary with cross-property analysis results
    """
    try:
        data = hypergraph.incidences.data
        
        # Get node-level and edge-level aggregated data
        node_data = node_property_correlations(hypergraph, property_columns)
        edge_data = edge_property_correlations(hypergraph, property_columns)
        
        # Analyze relationship between node properties and edge properties
        # This requires joining node and edge data through the incidence structure
        
        cross_correlations = {}
        
        # For each node property vs edge property combination
        node_props = [col for col in node_data.get('correlation_matrix', {}).keys()]
        edge_props = [col for col in edge_data.get('correlation_matrix', {}).keys()]
        
        # Calculate node-edge property relationships
        # This is a complex analysis that requires careful handling of the hypergraph structure
        
        # Simplified version: analyze property distributions
        property_distributions = _analyze_property_distributions(hypergraph, property_columns)
        
        return {
            'node_edge_relationships': cross_correlations,
            'property_distributions': property_distributions,
            'analysis_method': correlation_method
        }
        
    except Exception as e:
        logger.error(f"Error in cross-property analysis: {e}")
        return {'node_edge_relationships': {}, 'property_distributions': {}}


@performance_monitor
def correlation_visualization_data(hypergraph,
                                  property_columns: Optional[List[str]] = None,
                                  correlation_method: str = 'pearson') -> Dict[str, Any]:
    """
    Prepare data for correlation visualization.
    
    Args:
        hypergraph: Anant Hypergraph instance
        property_columns: List of property columns to analyze
        correlation_method: Correlation method
        
    Returns:
        Dictionary with visualization-ready data
    """
    try:
        # Get comprehensive correlation analysis
        analysis = property_correlation_analysis(
            hypergraph, property_columns, correlation_method
        )
        
        # Prepare data for different visualization types
        viz_data = {
            'correlation_heatmap': _prepare_heatmap_data(analysis),
            'network_graph': _prepare_network_data(analysis),
            'scatter_plots': _prepare_scatter_data(hypergraph, property_columns),
            'distribution_plots': _prepare_distribution_data(hypergraph, property_columns)
        }
        
        return viz_data
        
    except Exception as e:
        logger.error(f"Error preparing correlation visualization data: {e}")
        return {}


def _calculate_correlation(data: pl.DataFrame,
                          col1: str,
                          col2: str,
                          method: str = 'pearson') -> float:
    """Calculate correlation between two columns."""
    try:
        # Extract values
        values1 = data[col1].to_numpy()
        values2 = data[col2].to_numpy()
        
        # Remove NaN values
        mask = ~(np.isnan(values1) | np.isnan(values2))
        if not mask.any():
            return 0.0
        
        values1 = values1[mask]
        values2 = values2[mask]
        
        if len(values1) < 2:
            return 0.0
        
        # Calculate correlation
        if method == 'pearson':
            correlation = np.corrcoef(values1, values2)[0, 1]
        elif method == 'spearman' and scipy:
            correlation, _ = scipy.stats.spearmanr(values1, values2)
        elif method == 'kendall' and scipy:
            correlation, _ = scipy.stats.kendalltau(values1, values2)
        else:
            # Fallback to Pearson
            correlation = np.corrcoef(values1, values2)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating correlation between {col1} and {col2}: {e}")
        return 0.0


def _extract_significant_pairs(correlation_results: List[Dict], min_correlation: float) -> List[Dict]:
    """Extract significant correlation pairs from multiple result sets."""
    all_pairs = []
    
    for result in correlation_results:
        significant_pairs = result.get('significant_pairs', [])
        for pair in significant_pairs:
            if abs(pair['correlation']) >= min_correlation:
                all_pairs.append(pair)
    
    # Remove duplicates and sort by correlation strength
    unique_pairs = []
    seen_pairs = set()
    
    for pair in all_pairs:
        key = tuple(sorted([pair['property1'], pair['property2']]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(pair)
    
    unique_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    return unique_pairs


def _analyze_property_distributions(hypergraph, property_columns: List[str]) -> Dict[str, Any]:
    """Analyze property value distributions."""
    try:
        data = hypergraph.incidences.data
        distributions = {}
        
        for col in property_columns:
            if col in data.columns:
                col_data = data[col]
                distributions[col] = {
                    'mean': float(col_data.mean()) if col_data.mean() is not None else 0.0,
                    'std': float(col_data.std()) if col_data.std() is not None else 0.0,
                    'min': float(col_data.min()) if col_data.min() is not None else 0.0,
                    'max': float(col_data.max()) if col_data.max() is not None else 0.0,
                    'median': float(col_data.median()) if col_data.median() is not None else 0.0,
                    'null_count': int(col_data.null_count())
                }
        
        return distributions
        
    except Exception as e:
        logger.error(f"Error analyzing property distributions: {e}")
        return {}


def _prepare_heatmap_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for correlation heatmap visualization."""
    try:
        correlations = analysis.get('correlations', {})
        node_corr = correlations.get('node_level', {}).get('correlation_matrix', {})
        edge_corr = correlations.get('edge_level', {}).get('correlation_matrix', {})
        
        return {
            'node_correlations': node_corr,
            'edge_correlations': edge_corr
        }
        
    except Exception as e:
        logger.error(f"Error preparing heatmap data: {e}")
        return {}


def _prepare_network_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for correlation network visualization."""
    try:
        significant_pairs = analysis.get('significant_pairs', [])
        
        nodes = set()
        edges = []
        
        for pair in significant_pairs:
            prop1, prop2 = pair['property1'], pair['property2']
            nodes.add(prop1)
            nodes.add(prop2)
            edges.append({
                'source': prop1,
                'target': prop2,
                'weight': abs(pair['correlation']),
                'correlation': pair['correlation']
            })
        
        return {
            'nodes': [{'id': node} for node in nodes],
            'edges': edges
        }
        
    except Exception as e:
        logger.error(f"Error preparing network data: {e}")
        return {'nodes': [], 'edges': []}


def _prepare_scatter_data(hypergraph, property_columns: Optional[List[str]]) -> Dict[str, Any]:
    """Prepare data for scatter plot visualizations."""
    try:
        if not property_columns:
            return {}
        
        data = hypergraph.incidences.data
        scatter_data = {}
        
        # Prepare scatter plots for top correlated pairs
        for i, col1 in enumerate(property_columns):
            for j, col2 in enumerate(property_columns[i+1:], i+1):
                if col1 in data.columns and col2 in data.columns:
                    scatter_data[f'{col1}_vs_{col2}'] = {
                        'x': data[col1].to_list(),
                        'y': data[col2].to_list(),
                        'x_label': col1,
                        'y_label': col2
                    }
        
        return scatter_data
        
    except Exception as e:
        logger.error(f"Error preparing scatter data: {e}")
        return {}


def _prepare_distribution_data(hypergraph, property_columns: Optional[List[str]]) -> Dict[str, Any]:
    """Prepare data for distribution visualizations."""
    try:
        if not property_columns:
            return {}
        
        data = hypergraph.incidences.data
        distribution_data = {}
        
        for col in property_columns:
            if col in data.columns:
                values = data[col].drop_nulls().to_list()
                distribution_data[col] = {
                    'values': values,
                    'histogram_bins': min(50, max(10, len(values) // 20))
                }
        
        return distribution_data
        
    except Exception as e:
        logger.error(f"Error preparing distribution data: {e}")
        return {}