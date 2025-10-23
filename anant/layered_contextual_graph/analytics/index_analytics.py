"""
Index-Based Analytics for LCG
==============================

Uses PropertyStore indices for fast property-based queries across layers.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..core import LayeredContextualGraph

logger = logging.getLogger(__name__)


class IndexAnalytics:
    """
    Index-based analytics leveraging Anant's PropertyStore indices.
    
    PropertyStore maintains indices for fast property lookups.
    This class uses those indices for cross-layer analysis.
    
    Examples:
        >>> ia = IndexAnalytics(lcg)
        >>> 
        >>> # Build cross-layer property index
        >>> index = ia.build_cross_layer_index("category")
        >>> 
        >>> # Fast lookup by property value
        >>> entities = ia.find_by_property_value("category", "science")
        >>> 
        >>> # Property co-occurrence analysis
        >>> cooccur = ia.analyze_property_cooccurrence()
    """
    
    def __init__(self, lcg: LayeredContextualGraph):
        self.lcg = lcg
        self.property_indices: Dict[str, Dict[Any, Set[Tuple[str, str]]]] = {}  # prop_name -> {value: {(layer, entity_id)}}
        
        logger.info("IndexAnalytics initialized")
    
    def build_cross_layer_index(
        self,
        property_name: str,
        layers: Optional[List[str]] = None
    ) -> Dict[Any, Set[Tuple[str, str]]]:
        """
        Build cross-layer index for a property.
        
        Creates a reverse index: property_value -> {(layer_name, entity_id)}
        
        Args:
            property_name: Property to index
            layers: Specific layers (or all if None)
            
        Returns:
            Index mapping values to (layer, entity) tuples
        """
        index = defaultdict(set)
        
        search_layers = layers or list(self.lcg.layers.keys())
        
        for layer_name in search_layers:
            if layer_name not in self.lcg.layers:
                continue
            
            layer = self.lcg.layers[layer_name]
            
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Index nodes
            for node_id in prop_store.get_nodes_with_properties():
                node_props = prop_store.get_node_properties(node_id)
                if property_name in node_props:
                    value = node_props[property_name]
                    index[value].add((layer_name, node_id))
            
            # Index edges
            for edge_id in prop_store.get_edges_with_properties():
                edge_props = prop_store.get_edge_properties(edge_id)
                if property_name in edge_props:
                    value = edge_props[property_name]
                    index[value].add((layer_name, edge_id))
        
        # Cache the index
        self.property_indices[property_name] = dict(index)
        
        logger.info(f"Built index for property '{property_name}': {len(index)} unique values")
        return dict(index)
    
    def find_by_property_value(
        self,
        property_name: str,
        value: Any,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        Fast lookup: find all entities with specific property value.
        
        Args:
            property_name: Property to search
            value: Value to find
            use_cache: Use cached index if available
            
        Returns:
            Dict of layer_name -> list of entity IDs
        """
        # Build index if not cached
        if property_name not in self.property_indices or not use_cache:
            self.build_cross_layer_index(property_name)
        
        index = self.property_indices.get(property_name, {})
        entities = index.get(value, set())
        
        # Group by layer
        results = defaultdict(list)
        for layer_name, entity_id in entities:
            results[layer_name].append(entity_id)
        
        return dict(results)
    
    def analyze_property_cooccurrence(
        self,
        layers: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], int]:
        """
        Analyze which properties co-occur on the same entities.
        
        Useful for understanding property relationships.
        
        Args:
            layers: Specific layers to analyze
            
        Returns:
            Dict of (prop1, prop2) -> co-occurrence count
        """
        cooccurrence = defaultdict(int)
        
        search_layers = layers or list(self.lcg.layers.keys())
        
        for layer_name in search_layers:
            if layer_name not in self.lcg.layers:
                continue
            
            layer = self.lcg.layers[layer_name]
            
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Analyze nodes
            for node_id in prop_store.get_nodes_with_properties():
                props = list(prop_store.get_node_properties(node_id).keys())
                
                # Count co-occurrences
                for i, prop1 in enumerate(props):
                    for prop2 in props[i+1:]:
                        pair = tuple(sorted([prop1, prop2]))
                        cooccurrence[pair] += 1
        
        return dict(cooccurrence)
    
    def get_property_coverage(
        self,
        property_name: str
    ) -> Dict[str, Any]:
        """
        Analyze coverage of a property across layers.
        
        Args:
            property_name: Property to analyze
            
        Returns:
            Coverage statistics per layer
        """
        coverage = {}
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Count entities with this property
            nodes_with_prop = 0
            total_nodes = len(prop_store.get_nodes_with_properties())
            
            for node_id in prop_store.get_nodes_with_properties():
                if property_name in prop_store.get_node_properties(node_id):
                    nodes_with_prop += 1
            
            coverage[layer_name] = {
                'has_property': nodes_with_prop,
                'total_entities': total_nodes,
                'coverage_pct': (nodes_with_prop / total_nodes * 100) if total_nodes > 0 else 0,
                'level': layer.level
            }
        
        return coverage
    
    def find_property_gaps(self) -> Dict[str, List[str]]:
        """
        Find which properties are missing in which layers.
        
        Returns:
            Dict of property_name -> list of layers where it's missing
        """
        # Collect all properties across all layers
        all_properties = set()
        layer_properties = {}
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            stats = prop_store.get_statistics()
            
            props = set(stats.get('node_property_names', []))
            props.update(stats.get('edge_property_names', []))
            
            all_properties.update(props)
            layer_properties[layer_name] = props
        
        # Find gaps
        gaps = {}
        for prop in all_properties:
            missing_layers = [
                layer_name for layer_name, props in layer_properties.items()
                if prop not in props
            ]
            if missing_layers:
                gaps[prop] = missing_layers
        
        return gaps


def build_property_indices(
    lcg: LayeredContextualGraph,
    properties: Optional[List[str]] = None
) -> Dict[str, Dict[Any, Set[Tuple[str, str]]]]:
    """
    Convenience function to build indices for multiple properties.
    
    Args:
        lcg: LayeredContextualGraph
        properties: List of properties to index (or all if None)
        
    Returns:
        Dict of property_name -> index
    """
    ia = IndexAnalytics(lcg)
    
    if properties is None:
        # Get all unique properties across layers
        all_props = set()
        for layer in lcg.layers.values():
            if hasattr(layer.hypergraph, 'properties'):
                stats = layer.hypergraph.properties.get_statistics()
                all_props.update(stats.get('node_property_names', []))
                all_props.update(stats.get('edge_property_names', []))
        properties = list(all_props)
    
    indices = {}
    for prop in properties:
        indices[prop] = ia.build_cross_layer_index(prop)
    
    logger.info(f"Built indices for {len(properties)} properties")
    return indices
