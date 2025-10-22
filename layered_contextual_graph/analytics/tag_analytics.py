"""
Tag-Based Analytics for LCG
============================

Analyzes tags in PropertyStore for:
- Clustering entities by tags
- Tag-based context derivation
- Tag evolution across layers
- Tag-based recommendations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, Counter
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..core import LayeredContextualGraph, Context, ContextType

logger = logging.getLogger(__name__)


class TagAnalytics:
    """
    Tag-based analytics for LayeredContextualGraph.
    
    Analyzes tags (if present in properties) across layers to:
    - Cluster entities by shared tags
    - Derive contexts from tag patterns
    - Track tag evolution through hierarchy
    - Recommend related entities by tags
    
    Examples:
        >>> ta = TagAnalytics(lcg)
        >>> 
        >>> # Cluster by tags
        >>> clusters = ta.cluster_by_tags("category_tags")
        >>> 
        >>> # Find similar entities by tags
        >>> similar = ta.find_similar_by_tags("entity_1")
        >>> 
        >>> # Derive tag-based contexts
        >>> contexts = ta.derive_tag_contexts()
    """
    
    def __init__(self, lcg: LayeredContextualGraph, tag_property: str = "tags"):
        self.lcg = lcg
        self.tag_property = tag_property  # Default property name for tags
        
        logger.info(f"TagAnalytics initialized (tag_property='{tag_property}')")
    
    def cluster_by_tags(
        self,
        tag_property: Optional[str] = None,
        min_cluster_size: int = 2
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Cluster entities by shared tags.
        
        Args:
            tag_property: Property containing tags (or use default)
            min_cluster_size: Minimum entities per cluster
            
        Returns:
            Dict of tag -> list of (layer_name, entity_id) tuples
        """
        tag_prop = tag_property or self.tag_property
        clusters = defaultdict(list)
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            
            # Get entities with tags
            for node_id in prop_store.get_nodes_with_properties():
                props = prop_store.get_node_properties(node_id)
                
                if tag_prop in props:
                    tags = props[tag_prop]
                    
                    # Handle different tag formats
                    if isinstance(tags, str):
                        tags = [tags]
                    elif isinstance(tags, (list, set, tuple)):
                        tags = list(tags)
                    else:
                        continue
                    
                    for tag in tags:
                        clusters[tag].append((layer_name, node_id))
        
        # Filter by cluster size
        filtered_clusters = {
            tag: entities
            for tag, entities in clusters.items()
            if len(entities) >= min_cluster_size
        }
        
        logger.info(f"Found {len(filtered_clusters)} tag clusters")
        return filtered_clusters
    
    def find_similar_by_tags(
        self,
        entity_id: str,
        layer_name: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Find similar entities based on tag overlap.
        
        Uses Jaccard similarity on tag sets.
        
        Args:
            entity_id: Entity to find similar ones for
            layer_name: Layer containing the entity
            top_k: Number of similar entities to return
            
        Returns:
            List of (layer_name, entity_id, similarity_score) tuples
        """
        if layer_name not in self.lcg.layers:
            return []
        
        layer = self.lcg.layers[layer_name]
        
        if not hasattr(layer.hypergraph, 'properties'):
            return []
        
        prop_store = layer.hypergraph.properties
        
        # Get tags for target entity
        target_props = prop_store.get_node_properties(entity_id)
        if self.tag_property not in target_props:
            return []
        
        target_tags = set(self._normalize_tags(target_props[self.tag_property]))
        
        if not target_tags:
            return []
        
        # Compare with all other entities
        similarities = []
        
        for other_layer_name, other_layer in self.lcg.layers.items():
            if not hasattr(other_layer.hypergraph, 'properties'):
                continue
            
            other_prop_store = other_layer.hypergraph.properties
            
            for other_id in other_prop_store.get_nodes_with_properties():
                if other_id == entity_id and other_layer_name == layer_name:
                    continue
                
                other_props = other_prop_store.get_node_properties(other_id)
                if self.tag_property not in other_props:
                    continue
                
                other_tags = set(self._normalize_tags(other_props[self.tag_property]))
                
                if other_tags:
                    # Jaccard similarity
                    intersection = len(target_tags & other_tags)
                    union = len(target_tags | other_tags)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0:
                        similarities.append((other_layer_name, other_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def _normalize_tags(self, tags: Any) -> List[str]:
        """Normalize tags to list of strings"""
        if isinstance(tags, str):
            return [tags]
        elif isinstance(tags, (list, set, tuple)):
            return [str(t) for t in tags]
        else:
            return []
    
    def derive_tag_contexts(
        self,
        min_support: int = 3
    ) -> List[Context]:
        """
        Derive contexts from common tag patterns.
        
        Args:
            min_support: Minimum entities sharing tag
            
        Returns:
            List of tag-based contexts
        """
        # Cluster by tags
        clusters = self.cluster_by_tags(min_cluster_size=min_support)
        
        contexts = []
        
        for tag, entities in clusters.items():
            # Get applicable layers
            applicable_layers = set(layer_name for layer_name, _ in entities)
            
            # Create context
            context = Context(
                name=f"tag_{tag}",
                context_type=ContextType.DOMAIN,
                applicable_layers=applicable_layers,
                parameters={
                    'tag': tag,
                    'tag_property': self.tag_property,
                    'support': len(entities)
                },
                priority=min(10, len(entities))  # Higher priority for more common tags
            )
            
            contexts.append(context)
        
        logger.info(f"Derived {len(contexts)} tag-based contexts")
        return contexts
    
    def analyze_tag_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of tags across layers.
        
        Returns:
            Tag distribution statistics
        """
        tag_stats = {
            'total_tags': 0,
            'unique_tags': set(),
            'tags_per_layer': {},
            'most_common_tags': []
        }
        
        all_tags = Counter()
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            layer_tags = Counter()
            
            for node_id in prop_store.get_nodes_with_properties():
                props = prop_store.get_node_properties(node_id)
                
                if self.tag_property in props:
                    tags = self._normalize_tags(props[self.tag_property])
                    layer_tags.update(tags)
                    all_tags.update(tags)
                    tag_stats['unique_tags'].update(tags)
            
            tag_stats['tags_per_layer'][layer_name] = {
                'total': sum(layer_tags.values()),
                'unique': len(layer_tags),
                'most_common': layer_tags.most_common(5)
            }
        
        tag_stats['total_tags'] = sum(all_tags.values())
        tag_stats['unique_tags'] = len(tag_stats['unique_tags'])
        tag_stats['most_common_tags'] = all_tags.most_common(10)
        
        return tag_stats
    
    def track_tag_evolution(
        self,
        tag: str
    ) -> Dict[str, Any]:
        """
        Track how a specific tag appears across layers.
        
        Args:
            tag: Tag to track
            
        Returns:
            Evolution data
        """
        evolution = {}
        
        for layer_name, layer in self.lcg.layers.items():
            if not hasattr(layer.hypergraph, 'properties'):
                continue
            
            prop_store = layer.hypergraph.properties
            entities_with_tag = []
            
            for node_id in prop_store.get_nodes_with_properties():
                props = prop_store.get_node_properties(node_id)
                
                if self.tag_property in props:
                    tags = self._normalize_tags(props[self.tag_property])
                    if tag in tags:
                        entities_with_tag.append(node_id)
            
            if entities_with_tag:
                evolution[layer_name] = {
                    'count': len(entities_with_tag),
                    'level': layer.level,
                    'entities': entities_with_tag[:10]  # Sample
                }
        
        return {
            'tag': tag,
            'evolution': dict(sorted(evolution.items(), key=lambda x: x[1]['level'])),
            'total_occurrences': sum(e['count'] for e in evolution.values())
        }


def cluster_by_tags(
    lcg: LayeredContextualGraph,
    tag_property: str = "tags",
    min_cluster_size: int = 2,
    auto_apply_contexts: bool = False
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Convenience function for tag-based clustering.
    
    Args:
        lcg: LayeredContextualGraph
        tag_property: Property containing tags
        min_cluster_size: Minimum entities per cluster
        auto_apply_contexts: Automatically create contexts for clusters
        
    Returns:
        Tag clusters
    """
    ta = TagAnalytics(lcg, tag_property)
    clusters = ta.cluster_by_tags(min_cluster_size=min_cluster_size)
    
    if auto_apply_contexts:
        contexts = ta.derive_tag_contexts(min_support=min_cluster_size)
        for context in contexts:
            lcg.contexts[context.name] = context
        logger.info(f"Auto-applied {len(contexts)} tag-based contexts")
    
    return clusters
