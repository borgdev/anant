"""
Graph Fusion Module
=================

Combines multiple graph types into unified structures using various fusion strategies.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of graph fusion operation"""
    fused_graph: Any
    fusion_strategy: str
    source_graphs: List[str]
    fusion_metadata: Dict[str, Any]
    node_mapping: Dict[str, str] = field(default_factory=dict)
    edge_mapping: Dict[str, str] = field(default_factory=dict)
    conflicts_resolved: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EntityAlignment:
    """Alignment between entities across graphs"""
    entity_id: str
    source_entities: Dict[str, str]  # graph_id -> original_entity_id
    confidence: float
    alignment_method: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EntityMatcher:
    """Matches entities across different graph types"""
    
    def __init__(self):
        """Initialize entity matcher"""
        self.matching_cache = {}
    
    def find_entity_alignments(self, graphs: List[Any], **kwargs) -> List[EntityAlignment]:
        """Find entity alignments across multiple graphs"""
        
        matching_threshold = kwargs.get('matching_threshold', 0.8)
        use_semantic_matching = kwargs.get('use_semantic_matching', True)
        use_structural_matching = kwargs.get('use_structural_matching', True)
        
        alignments = []
        
        # Extract entities from each graph
        graph_entities = {}
        for i, graph in enumerate(graphs):
            graph_id = f"graph_{i}"
            entities = self._extract_entities_from_graph(graph)
            graph_entities[graph_id] = entities
        
        # Find potential alignments
        graph_ids = list(graph_entities.keys())
        
        for i, graph_id1 in enumerate(graph_ids):
            for graph_id2 in graph_ids[i+1:]:
                
                entities1 = graph_entities[graph_id1]
                entities2 = graph_entities[graph_id2]
                
                # Compare entities between graphs
                for entity1 in entities1:
                    for entity2 in entities2:
                        
                        # Calculate matching confidence
                        confidence = 0.0
                        matching_methods = []
                        
                        # String similarity matching
                        string_sim = self._string_similarity(entity1['id'], entity2['id'])
                        if string_sim > 0.7:
                            confidence += string_sim * 0.4
                            matching_methods.append('string_similarity')
                        
                        # Property-based matching
                        if use_semantic_matching:
                            prop_sim = self._property_similarity(entity1.get('properties', {}), 
                                                               entity2.get('properties', {}))
                            if prop_sim > 0.6:
                                confidence += prop_sim * 0.6
                                matching_methods.append('property_similarity')
                        
                        # Structural matching (neighborhood similarity)
                        if use_structural_matching:
                            struct_sim = self._structural_similarity(entity1, entity2, graphs[i], graphs[i+1])
                            if struct_sim > 0.5:
                                confidence += struct_sim * 0.3
                                matching_methods.append('structural_similarity')
                        
                        # Create alignment if confidence is high enough
                        if confidence >= matching_threshold:
                            alignment_id = str(uuid.uuid4())
                            
                            alignment = EntityAlignment(
                                entity_id=alignment_id,
                                source_entities={
                                    graph_id1: entity1['id'],
                                    graph_id2: entity2['id']
                                },
                                confidence=confidence,
                                alignment_method='+'.join(matching_methods),
                                properties={
                                    'string_similarity': string_sim,
                                    'property_similarity': prop_sim if use_semantic_matching else 0.0,
                                    'structural_similarity': struct_sim if use_structural_matching else 0.0
                                }
                            )
                            
                            alignments.append(alignment)
        
        # Remove duplicate alignments and conflicts
        alignments = self._resolve_alignment_conflicts(alignments)
        
        logger.info(f"Found {len(alignments)} entity alignments across {len(graphs)} graphs")
        return alignments
    
    def _extract_entities_from_graph(self, graph: Any) -> List[Dict[str, Any]]:
        """Extract entities and their properties from a graph"""
        
        entities = []
        
        try:
            # Get nodes (entities) from graph
            if hasattr(graph, 'nodes'):
                for node in graph.nodes:
                    entity = {
                        'id': node,
                        'type': 'node',
                        'properties': {}
                    }
                    
                    # Get node properties if available
                    if hasattr(graph, 'get_node_data'):
                        node_data = graph.get_node_data(node)
                        if node_data:
                            entity['properties'] = node_data
                    elif hasattr(graph, 'properties') and hasattr(graph.properties, 'get_node_data'):
                        node_data = graph.properties.get_node_data(node)
                        if node_data:
                            entity['properties'] = node_data
                    
                    entities.append(entity)
            
            # For metagraphs, also include meta-nodes
            if hasattr(graph, 'meta_nodes'):
                for meta_node in graph.meta_nodes:
                    entity = {
                        'id': meta_node,
                        'type': 'meta_node',
                        'properties': {}
                    }
                    
                    if hasattr(graph, 'get_meta_node_data'):
                        meta_data = graph.get_meta_node_data(meta_node)
                        if meta_data:
                            entity['properties'] = meta_data
                    
                    entities.append(entity)
        
        except Exception as e:
            logger.warning(f"Error extracting entities from graph: {e}")
        
        return entities
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity between two strings"""
        
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity on character bigrams
        def get_bigrams(s):
            s = s.lower()
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(str(str1))
        bigrams2 = get_bigrams(str(str2))
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _property_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Calculate similarity between entity properties"""
        
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0
        
        # Compare common properties
        common_keys = set(props1.keys()) & set(props2.keys())
        
        if not common_keys:
            return 0.0
        
        total_similarity = 0.0
        
        for key in common_keys:
            val1, val2 = props1[key], props2[key]
            
            if val1 == val2:
                similarity = 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                similarity = self._string_similarity(val1, val2)
            else:
                similarity = 0.5  # Some similarity for having the same property
            
            total_similarity += similarity
        
        return total_similarity / len(common_keys)
    
    def _structural_similarity(self, entity1: Dict, entity2: Dict, graph1: Any, graph2: Any) -> float:
        """Calculate structural similarity based on graph neighborhood"""
        
        try:
            # Get neighbors of each entity
            neighbors1 = self._get_entity_neighbors(entity1['id'], graph1)
            neighbors2 = self._get_entity_neighbors(entity2['id'], graph2)
            
            if not neighbors1 and not neighbors2:
                return 1.0
            if not neighbors1 or not neighbors2:
                return 0.0
            
            # Simple neighborhood overlap measure
            # In practice, this would be more sophisticated
            return 0.5  # Placeholder - would implement neighborhood comparison
        
        except Exception:
            return 0.0
    
    def _get_entity_neighbors(self, entity_id: str, graph: Any) -> List[str]:
        """Get neighbors of an entity in a graph"""
        
        neighbors = []
        
        try:
            if hasattr(graph, 'get_node_edges'):
                edges = graph.get_node_edges(entity_id)
                for edge in edges:
                    edge_nodes = graph.get_edge_nodes(edge) if hasattr(graph, 'get_edge_nodes') else []
                    neighbors.extend([node for node in edge_nodes if node != entity_id])
        
        except Exception:
            pass
        
        return neighbors
    
    def _resolve_alignment_conflicts(self, alignments: List[EntityAlignment]) -> List[EntityAlignment]:
        """Resolve conflicts in entity alignments"""
        
        # Remove duplicate alignments (same entities aligned multiple times)
        seen_pairs = set()
        filtered_alignments = []
        
        for alignment in sorted(alignments, key=lambda a: a.confidence, reverse=True):
            # Create a signature for this alignment
            entities = tuple(sorted(alignment.source_entities.items()))
            
            if entities not in seen_pairs:
                seen_pairs.add(entities)
                filtered_alignments.append(alignment)
        
        return filtered_alignments


class GraphFusion:
    """Combines multiple graphs into unified structures"""
    
    def __init__(self):
        """Initialize graph fusion engine"""
        self.entity_matcher = EntityMatcher()
        self.fusion_stats = {}
        
        logger.info("GraphFusion initialized")
    
    def fuse_graphs(self, graphs: List[Any], fusion_strategy: str = "unified", **kwargs) -> FusionResult:
        """
        Fuse multiple graphs into a unified structure
        
        Parameters
        ----------
        graphs : List[Any]
            List of graphs to fuse
        fusion_strategy : str
            Fusion strategy: "unified", "layered", "hierarchical", "meta"
        **kwargs : dict
            Fusion options
            
        Returns
        -------
        FusionResult
            Result of the fusion operation
        """
        
        if not graphs:
            raise ValueError("No graphs provided for fusion")
        
        logger.info(f"Fusing {len(graphs)} graphs using '{fusion_strategy}' strategy")
        
        # Find entity alignments
        alignments = self.entity_matcher.find_entity_alignments(graphs, **kwargs)
        
        # Execute fusion based on strategy
        if fusion_strategy == "unified":
            return self._fuse_unified(graphs, alignments, **kwargs)
        elif fusion_strategy == "layered":
            return self._fuse_layered(graphs, alignments, **kwargs)
        elif fusion_strategy == "hierarchical":
            return self._fuse_hierarchical(graphs, alignments, **kwargs)
        elif fusion_strategy == "meta":
            return self._fuse_meta(graphs, alignments, **kwargs)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def _fuse_unified(self, graphs: List[Any], alignments: List[EntityAlignment], **kwargs) -> FusionResult:
        """Fuse graphs into a single unified knowledge graph"""
        
        from ..kg.core import KnowledgeGraph
        
        # Create unified knowledge graph
        fused_kg = KnowledgeGraph(name="unified_fused_graph")
        
        node_mapping = {}
        edge_mapping = {}
        conflicts_resolved = []
        
        # Create entity mapping from alignments
        entity_id_map = {}  # original_entity -> unified_entity_id
        
        for alignment in alignments:
            unified_id = alignment.entity_id
            for graph_id, original_id in alignment.source_entities.items():
                entity_id_map[f"{graph_id}:{original_id}"] = unified_id
        
        # Add nodes from all graphs
        for i, graph in enumerate(graphs):
            graph_id = f"graph_{i}"
            
            try:
                if hasattr(graph, 'nodes'):
                    for node in graph.nodes:
                        # Determine unified node ID
                        key = f"{graph_id}:{node}"
                        if key in entity_id_map:
                            unified_id = entity_id_map[key]
                        else:
                            unified_id = f"unified_{node}_{i}"
                        
                        node_mapping[key] = unified_id
                        
                        # Get node properties
                        node_data = {}
                        if hasattr(graph, 'get_node_data'):
                            props = graph.get_node_data(node)
                            if props:
                                node_data.update(props)
                        
                        # Add provenance information
                        node_data.update({
                            'source_graph': graph_id,
                            'source_graph_type': graph.__class__.__name__,
                            'original_id': node
                        })
                        
                        # Add to unified graph (merge if already exists)
                        if fused_kg.has_node(unified_id):
                            # Merge properties
                            existing_data = fused_kg.get_node_data(unified_id) or {}
                            merged_data = {**existing_data, **node_data}
                            fused_kg.set_node_data(unified_id, merged_data)
                            
                            conflicts_resolved.append({
                                'type': 'node_merge',
                                'unified_id': unified_id,
                                'sources': [existing_data.get('source_graph'), graph_id]
                            })
                        else:
                            fused_kg.add_node(unified_id, data=node_data)
            
            except Exception as e:
                logger.warning(f"Error processing nodes from graph {i}: {e}")
        
        # Add edges from all graphs
        for i, graph in enumerate(graphs):
            graph_id = f"graph_{i}"
            
            try:
                if hasattr(graph, 'edges'):
                    for edge in graph.edges:
                        # Map edge nodes to unified IDs
                        if hasattr(graph, 'get_edge_nodes'):
                            edge_nodes = graph.get_edge_nodes(edge)
                        else:
                            edge_nodes = list(edge) if hasattr(edge, '__iter__') else [edge]
                        
                        if len(edge_nodes) >= 2:
                            source_key = f"{graph_id}:{edge_nodes[0]}"
                            target_key = f"{graph_id}:{edge_nodes[1]}"
                            
                            unified_source = node_mapping.get(source_key, f"unified_{edge_nodes[0]}_{i}")
                            unified_target = node_mapping.get(target_key, f"unified_{edge_nodes[1]}_{i}")
                            
                            # Get edge properties
                            edge_data = {}
                            if hasattr(graph, 'get_edge_data'):
                                props = graph.get_edge_data(edge)
                                if props:
                                    edge_data.update(props)
                            
                            # Add provenance
                            edge_data.update({
                                'source_graph': graph_id,
                                'original_edge': str(edge)
                            })
                            
                            # Add relationship to unified graph
                            relation_type = edge_data.get('type', 'related_to')
                            fused_kg.add_relationship(unified_source, unified_target, relation_type, **edge_data)
            
            except Exception as e:
                logger.warning(f"Error processing edges from graph {i}: {e}")
        
        return FusionResult(
            fused_graph=fused_kg,
            fusion_strategy="unified",
            source_graphs=[f"graph_{i}" for i in range(len(graphs))],
            fusion_metadata={
                'alignments_used': len(alignments),
                'total_nodes': fused_kg.num_nodes(),
                'total_edges': fused_kg.num_edges(),
                'conflicts_resolved': len(conflicts_resolved)
            },
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
            conflicts_resolved=conflicts_resolved
        )
    
    def _fuse_layered(self, graphs: List[Any], alignments: List[EntityAlignment], **kwargs) -> FusionResult:
        """Fuse graphs into a layered hierarchical structure"""
        
        from ..kg.hierarchical import HierarchicalKnowledgeGraph
        
        # Create hierarchical knowledge graph with each source graph as a level
        fused_hkg = HierarchicalKnowledgeGraph(name="layered_fused_graph")
        
        node_mapping = {}
        edge_mapping = {}
        
        # Create a level for each graph
        for i, graph in enumerate(graphs):
            level_name = f"level_{i}_{graph.__class__.__name__}"
            
            # Add level to hierarchical graph
            fused_hkg.add_level(level_name, description=f"Level from graph {i}")
            
            # Add nodes to this level
            if hasattr(graph, 'nodes'):
                for node in graph.nodes:
                    node_data = {}
                    if hasattr(graph, 'get_node_data'):
                        props = graph.get_node_data(node)
                        if props:
                            node_data.update(props)
                    
                    node_data['source_graph'] = i
                    fused_hkg.add_node_to_level(level_name, node, data=node_data)
                    node_mapping[f"graph_{i}:{node}"] = node
        
        # Add cross-level connections based on alignments
        for alignment in alignments:
            # Create cross-level connections for aligned entities
            source_entities = list(alignment.source_entities.items())
            
            for i, (graph_id1, entity1) in enumerate(source_entities):
                for graph_id2, entity2 in source_entities[i+1:]:
                    
                    level1 = f"level_{graph_id1.split('_')[1]}_{graphs[int(graph_id1.split('_')[1])].__class__.__name__}"
                    level2 = f"level_{graph_id2.split('_')[1]}_{graphs[int(graph_id2.split('_')[1])].__class__.__name__}"
                    
                    fused_hkg.add_cross_level_edge(
                        level1, entity1,
                        level2, entity2,
                        edge_type="aligned_entity",
                        confidence=alignment.confidence
                    )
        
        return FusionResult(
            fused_graph=fused_hkg,
            fusion_strategy="layered",
            source_graphs=[f"graph_{i}" for i in range(len(graphs))],
            fusion_metadata={
                'levels_created': len(graphs),
                'alignments_used': len(alignments),
                'cross_level_edges': len(alignments)
            },
            node_mapping=node_mapping,
            edge_mapping=edge_mapping
        )
    
    def _fuse_hierarchical(self, graphs: List[Any], alignments: List[EntityAlignment], **kwargs) -> FusionResult:
        """Fuse graphs into a hierarchical structure based on graph types"""
        
        # Group graphs by type and create hierarchy
        # More complex implementation would analyze graph structure
        return self._fuse_layered(graphs, alignments, **kwargs)
    
    def _fuse_meta(self, graphs: List[Any], alignments: List[EntityAlignment], **kwargs) -> FusionResult:
        """Fuse graphs into a metagraph structure"""
        
        from ..metagraph.core.metagraph import Metagraph
        
        # Create metagraph
        fused_mg = Metagraph(name="meta_fused_graph")
        
        node_mapping = {}
        
        # Add all nodes and create meta-nodes for graph types
        for i, graph in enumerate(graphs):
            graph_type = graph.__class__.__name__
            meta_node_id = f"META_{graph_type}_{i}"
            
            # Create meta-node representing this graph
            fused_mg.add_meta_node(meta_node_id, {
                'represents': graph_type,
                'source_graph_index': i,
                'node_count': getattr(graph, 'num_nodes', lambda: 0)()
            })
            
            # Add regular nodes
            if hasattr(graph, 'nodes'):
                for node in graph.nodes:
                    node_data = {}
                    if hasattr(graph, 'get_node_data'):
                        props = graph.get_node_data(node)
                        if props:
                            node_data.update(props)
                    
                    unified_id = f"{graph_type}_{i}_{node}"
                    fused_mg.add_node(unified_id, **node_data)
                    
                    # Connect to meta-node
                    fused_mg.add_meta_relationship(meta_node_id, unified_id, relation_type="contains")
                    
                    node_mapping[f"graph_{i}:{node}"] = unified_id
        
        # Add meta-relationships based on alignments
        for alignment in alignments:
            # Create meta-relationships between aligned entities
            entities = list(alignment.source_entities.values())
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    fused_mg.add_meta_relationship(
                        entity1, entity2,
                        relation_type="aligned_entity",
                        confidence=alignment.confidence
                    )
        
        return FusionResult(
            fused_graph=fused_mg,
            fusion_strategy="meta",
            source_graphs=[f"graph_{i}" for i in range(len(graphs))],
            fusion_metadata={
                'meta_nodes_created': len(graphs),
                'alignments_used': len(alignments)
            },
            node_mapping=node_mapping
        )