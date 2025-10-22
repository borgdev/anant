"""
Property-Level Analytics for LCG
================================

Analyzes properties across layers to:
- Derive contexts automatically from property patterns
- Track property evolution through hierarchy
- Detect property-based relationships
- Build property indices for fast querying
- Cluster entities by property similarity
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..core import LayeredContextualGraph, Layer, Context, ContextType

logger = logging.getLogger(__name__)


@dataclass
class PropertyBasedContext:
    """Context derived from property patterns"""
    name: str
    context_type: ContextType
    property_patterns: Dict[str, Any]  # property_name -> expected_value/pattern
    applicable_layers: Set[str]
    confidence: float = 1.0  # How confident we are in this context
    derived_from: List[str] = field(default_factory=list)  # Which properties contributed
    
    def matches(self, properties: Dict[str, Any]) -> bool:
        """Check if properties match this context"""
        matches = 0
        total = len(self.property_patterns)
        
        for prop_name, expected in self.property_patterns.items():
            if prop_name in properties:
                if isinstance(expected, (list, set)):
                    if properties[prop_name] in expected:
                        matches += 1
                elif callable(expected):
                    if expected(properties[prop_name]):
                        matches += 1
                elif properties[prop_name] == expected:
                    matches += 1
        
        return (matches / total) >= self.confidence if total > 0 else False


class PropertyAnalytics:
    """
    Property-level analytics for LayeredContextualGraph.
    
    Analyzes properties across layers using Anant's PropertyStore to:
    - Derive contexts from property patterns
    - Track property evolution through layers
    - Build property-based indices
    - Detect property anomalies
    - Cluster entities by properties
    
    Examples:
        >>> pa = PropertyAnalytics(lcg)
        >>> 
        >>> # Derive contexts from properties
        >>> contexts = pa.derive_contexts()
        >>> print(f"Found {len(contexts)} property-based contexts")
        >>> 
        >>> # Analyze property distribution
        >>> dist = pa.analyze_property_distribution("layer_name")
        >>> 
        >>> # Track property evolution
        >>> evolution = pa.track_property_evolution("entity_1")
    """
    
    def __init__(self, lcg: LayeredContextualGraph):
        self.lcg = lcg
        self.property_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("PropertyAnalytics initialized")
    
    def derive_contexts(
        self,
        min_confidence: float = 0.7,
        min_support: int = 2  # Min entities that must share pattern
    ) -> List[PropertyBasedContext]:
        """
        Derive contexts automatically from property patterns across layers.
        
        Analyzes property combinations and creates contexts for common patterns.
        
        Args:
            min_confidence: Minimum confidence for context (0-1)
            min_support: Minimum number of entities sharing pattern
            
        Returns:
            List of derived PropertyBasedContext objects
        """
        derived_contexts = []
        
        # Analyze each layer
        for layer_name, layer in self.lcg.layers.items():
            # Get all properties from this layer's hypergraph
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Analyze node properties
            node_contexts = self._derive_from_node_properties(
                layer_name, prop_store, min_confidence, min_support
            )
            derived_contexts.extend(node_contexts)
            
            # Analyze edge properties
            edge_contexts = self._derive_from_edge_properties(
                layer_name, prop_store, min_confidence, min_support
            )
            derived_contexts.extend(edge_contexts)
        
        logger.info(f"Derived {len(derived_contexts)} contexts from properties")
        return derived_contexts
    
    def _derive_from_node_properties(
        self,
        layer_name: str,
        prop_store: Any,
        min_confidence: float,
        min_support: int
    ) -> List[PropertyBasedContext]:
        """Derive contexts from node property patterns"""
        contexts = []
        
        # Get all nodes with properties
        nodes_with_props = prop_store.get_nodes_with_properties()
        
        if len(nodes_with_props) < min_support:
            return contexts
        
        # Find common property patterns
        property_patterns = defaultdict(lambda: defaultdict(int))
        
        for node_id in nodes_with_props:
            node_props = prop_store.get_node_properties(node_id)
            
            for prop_name, prop_value in node_props.items():
                # Count occurrences of each property=value pair
                property_patterns[prop_name][prop_value] += 1
        
        # Create contexts for common patterns
        for prop_name, value_counts in property_patterns.items():
            for value, count in value_counts.items():
                if count >= min_support:
                    confidence = count / len(nodes_with_props)
                    
                    if confidence >= min_confidence:
                        context = PropertyBasedContext(
                            name=f"{layer_name}_{prop_name}_{value}",
                            context_type=ContextType.DOMAIN,  # Could be smarter
                            property_patterns={prop_name: value},
                            applicable_layers={layer_name},
                            confidence=confidence,
                            derived_from=[f"node.{prop_name}"]
                        )
                        contexts.append(context)
        
        return contexts
    
    def _derive_from_edge_properties(
        self,
        layer_name: str,
        prop_store: Any,
        min_confidence: float,
        min_support: int
    ) -> List[PropertyBasedContext]:
        """Derive contexts from edge property patterns"""
        contexts = []
        
        # Similar to node properties but for edges
        edges_with_props = prop_store.get_edges_with_properties()
        
        if len(edges_with_props) < min_support:
            return contexts
        
        property_patterns = defaultdict(lambda: defaultdict(int))
        
        for edge_id in edges_with_props:
            edge_props = prop_store.get_edge_properties(edge_id)
            
            for prop_name, prop_value in edge_props.items():
                property_patterns[prop_name][prop_value] += 1
        
        # Create contexts for common patterns
        for prop_name, value_counts in property_patterns.items():
            for value, count in value_counts.items():
                if count >= min_support:
                    confidence = count / len(edges_with_props)
                    
                    if confidence >= min_confidence:
                        context = PropertyBasedContext(
                            name=f"{layer_name}_edge_{prop_name}_{value}",
                            context_type=ContextType.DOMAIN,
                            property_patterns={f"edge.{prop_name}": value},
                            applicable_layers={layer_name},
                            confidence=confidence,
                            derived_from=[f"edge.{prop_name}"]
                        )
                        contexts.append(context)
        
        return contexts
    
    def analyze_property_distribution(
        self,
        layer_name: str,
        property_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze distribution of properties in a layer.
        
        Args:
            layer_name: Layer to analyze
            property_name: Specific property (or all if None)
            
        Returns:
            Distribution statistics
        """
        if layer_name not in self.lcg.layers:
            return {}
        
        layer = self.lcg.layers[layer_name]
        
        if not hasattr(layer.hypergraph, 'properties'):
            return {'error': 'No properties available'}
        
        prop_store = layer.hypergraph.properties
        
        # Analyze node properties
        node_dist = self._analyze_node_property_distribution(
            prop_store, property_name
        )
        
        # Analyze edge properties
        edge_dist = self._analyze_edge_property_distribution(
            prop_store, property_name
        )
        
        return {
            'layer': layer_name,
            'node_properties': node_dist,
            'edge_properties': edge_dist,
            'total_nodes_with_props': len(prop_store.get_nodes_with_properties()),
            'total_edges_with_props': len(prop_store.get_edges_with_properties())
        }
    
    def _analyze_node_property_distribution(
        self,
        prop_store: Any,
        property_name: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze node property distribution"""
        distribution = {}
        
        nodes = prop_store.get_nodes_with_properties()
        
        if property_name:
            # Analyze specific property
            values = []
            for node_id in nodes:
                props = prop_store.get_node_properties(node_id)
                if property_name in props:
                    values.append(props[property_name])
            
            if values:
                distribution[property_name] = self._compute_distribution_stats(values)
        else:
            # Analyze all properties
            all_props = defaultdict(list)
            for node_id in nodes:
                props = prop_store.get_node_properties(node_id)
                for pname, pvalue in props.items():
                    all_props[pname].append(pvalue)
            
            for pname, values in all_props.items():
                distribution[pname] = self._compute_distribution_stats(values)
        
        return distribution
    
    def _analyze_edge_property_distribution(
        self,
        prop_store: Any,
        property_name: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze edge property distribution"""
        distribution = {}
        
        edges = prop_store.get_edges_with_properties()
        
        if property_name:
            values = []
            for edge_id in edges:
                props = prop_store.get_edge_properties(edge_id)
                if property_name in props:
                    values.append(props[property_name])
            
            if values:
                distribution[property_name] = self._compute_distribution_stats(values)
        else:
            all_props = defaultdict(list)
            for edge_id in edges:
                props = prop_store.get_edge_properties(edge_id)
                for pname, pvalue in props.items():
                    all_props[pname].append(pvalue)
            
            for pname, values in all_props.items():
                distribution[pname] = self._compute_distribution_stats(values)
        
        return distribution
    
    def _compute_distribution_stats(self, values: List[Any]) -> Dict[str, Any]:
        """Compute statistics for a distribution of values"""
        if not values:
            return {}
        
        stats = {
            'count': len(values),
            'unique_values': len(set(values))
        }
        
        # For numeric values
        if all(isinstance(v, (int, float)) for v in values) and NUMPY_AVAILABLE:
            arr = np.array(values)
            stats.update({
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            })
        
        # Value counts
        value_counts = defaultdict(int)
        for v in values:
            value_counts[str(v)] += 1
        
        stats['value_counts'] = dict(sorted(
            value_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10
        
        return stats
    
    def track_property_evolution(
        self,
        entity_id: str,
        property_name: str
    ) -> Dict[str, Any]:
        """
        Track how a property evolves across layers for an entity.
        
        Useful for understanding how properties change through the hierarchy.
        
        Args:
            entity_id: Entity to track
            property_name: Property to track
            
        Returns:
            Evolution data across layers
        """
        if entity_id not in self.lcg.superposition_states:
            return {'error': f'Entity {entity_id} not found'}
        
        superpos = self.lcg.superposition_states[entity_id]
        evolution = {}
        
        # Check each layer
        for layer_name, layer_state in superpos.layer_states.items():
            layer = self.lcg.layers.get(layer_name)
            
            if not layer or not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Try to find property value in this layer
            # Note: entity_id might map to different node/edge IDs in each layer
            value = None
            
            # Check as node
            try:
                node_props = prop_store.get_node_properties(layer_state)
                value = node_props.get(property_name)
            except:
                pass
            
            if value is not None:
                evolution[layer_name] = {
                    'value': value,
                    'level': layer.level,
                    'type': 'node'
                }
        
        # Sort by level
        sorted_evolution = dict(sorted(
            evolution.items(),
            key=lambda x: x[1]['level']
        ))
        
        return {
            'entity_id': entity_id,
            'property_name': property_name,
            'evolution': sorted_evolution,
            'num_layers': len(sorted_evolution)
        }
    
    def find_entities_by_properties(
        self,
        property_filters: Dict[str, Any],
        layers: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Find entities matching property filters across layers.
        
        Args:
            property_filters: Dict of property_name -> desired_value
            layers: Specific layers to search (or all if None)
            
        Returns:
            Dict of layer_name -> list of matching entity IDs
        """
        results = {}
        
        search_layers = layers or list(self.lcg.layers.keys())
        
        for layer_name in search_layers:
            if layer_name not in self.lcg.layers:
                continue
            
            layer = self.lcg.layers[layer_name]
            
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Search nodes
            matching_nodes = []
            for node_id in prop_store.get_nodes_with_properties():
                node_props = prop_store.get_node_properties(node_id)
                
                # Check if all filters match
                if all(node_props.get(k) == v for k, v in property_filters.items()):
                    matching_nodes.append(node_id)
            
            if matching_nodes:
                results[layer_name] = matching_nodes
        
        return results
    
    def get_property_summary(self) -> Dict[str, Any]:
        """Get summary of all properties across all layers"""
        summary = {
            'total_layers': len(self.lcg.layers),
            'layers': {}
        }
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            stats = prop_store.get_statistics()
            
            summary['layers'][layer_name] = {
                'level': layer.level,
                'nodes_with_properties': stats.get('nodes_with_properties', 0),
                'edges_with_properties': stats.get('edges_with_properties', 0),
                'total_node_properties': stats.get('total_node_properties', 0),
                'total_edge_properties': stats.get('total_edge_properties', 0),
                'unique_node_properties': len(stats.get('node_property_names', [])),
                'unique_edge_properties': len(stats.get('edge_property_names', []))
            }
        
        return summary


def derive_contexts_from_properties(
    lcg: LayeredContextualGraph,
    min_confidence: float = 0.7,
    min_support: int = 2,
    auto_apply: bool = False
) -> List[PropertyBasedContext]:
    """
    Convenience function to derive contexts from properties.
    
    Args:
        lcg: LayeredContextualGraph to analyze
        min_confidence: Minimum confidence for contexts
        min_support: Minimum entities sharing pattern
        auto_apply: Automatically add contexts to LCG
        
    Returns:
        List of derived contexts
    """
    pa = PropertyAnalytics(lcg)
    contexts = pa.derive_contexts(min_confidence, min_support)
    
    if auto_apply:
        for prop_context in contexts:
            # Convert PropertyBasedContext to regular Context
            context = Context(
                name=prop_context.name,
                context_type=prop_context.context_type,
                applicable_layers=prop_context.applicable_layers,
                parameters={'property_patterns': prop_context.property_patterns},
                priority=int(prop_context.confidence * 10)
            )
            lcg.contexts[context.name] = context
        
        logger.info(f"Auto-applied {len(contexts)} property-based contexts to LCG")
    
    return contexts
