"""
Graph Type Converters
==================

Specialized converters for transforming between different graph types.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import polars as pl
from collections import defaultdict

from ..classes.hypergraph import Hypergraph
from ..kg.core import KnowledgeGraph
from ..metagraph.core.metagraph import Metagraph

logger = logging.getLogger(__name__)


class HypergraphToKGConverter:
    """Convert Hypergraph to KnowledgeGraph format"""
    
    def __init__(self):
        """Initialize converter"""
        self.conversion_stats = {}
        logger.info("HypergraphToKGConverter initialized")
    
    def convert(self, hypergraph: Hypergraph, **kwargs) -> KnowledgeGraph:
        """
        Convert Hypergraph to KnowledgeGraph
        
        Parameters
        ----------
        hypergraph : Hypergraph
            Source hypergraph to convert
        **kwargs : dict
            Conversion options:
            - edge_strategy: "pairwise", "star", "clique" (default: "pairwise")
            - preserve_properties: bool (default: True)
            - semantic_inference: bool (default: True)
            
        Returns
        -------
        KnowledgeGraph
            Converted knowledge graph
        """
        edge_strategy = kwargs.get('edge_strategy', 'pairwise')
        preserve_properties = kwargs.get('preserve_properties', True)
        semantic_inference = kwargs.get('semantic_inference', True)
        
        # Create new knowledge graph
        kg = KnowledgeGraph(
            name=f"KG_from_{hypergraph.name}",
            enable_reasoning=semantic_inference
        )
        
        # Convert nodes
        nodes_converted = 0
        for node_id in hypergraph.nodes:
            # Add node to KG
            node_data = {}
            
            if preserve_properties and hasattr(hypergraph, 'properties'):
                node_props = hypergraph.properties.get_node_data(node_id)
                if node_props:
                    node_data.update(node_props)
            
            # Infer semantic type if possible
            if semantic_inference:
                node_type = self._infer_node_type(node_id, node_data, hypergraph)
                if node_type:
                    node_data['semantic_type'] = node_type
            
            kg.add_node(node_id, data=node_data)
            nodes_converted += 1
        
        # Convert hyperedges to graph edges
        edges_converted = 0
        
        for edge_id in hypergraph.edges:
            edge_nodes = hypergraph.get_edge_nodes(edge_id)
            
            if len(edge_nodes) < 2:
                continue  # Skip edges with insufficient nodes
            
            # Get edge properties
            edge_data = {}
            if preserve_properties and hasattr(hypergraph, 'properties'):
                edge_props = hypergraph.properties.get_edge_data(edge_id)
                if edge_props:
                    edge_data.update(edge_props)
            
            # Convert hyperedge based on strategy
            if edge_strategy == 'pairwise':
                # Create pairwise connections between all nodes in hyperedge
                for i, node1 in enumerate(edge_nodes):
                    for node2 in edge_nodes[i+1:]:
                        relation_type = edge_data.get('type', 'connected_via')
                        
                        # Add semantic context
                        relationship_data = {
                            'original_hyperedge': edge_id,
                            'hyperedge_size': len(edge_nodes),
                            **edge_data
                        }
                        
                        kg.add_relationship(node1, node2, relation_type, **relationship_data)
                        edges_converted += 1
            
            elif edge_strategy == 'star':
                # Create star pattern with first node as center
                center_node = edge_nodes[0]
                for other_node in edge_nodes[1:]:
                    relation_type = edge_data.get('type', 'connected_in')
                    
                    relationship_data = {
                        'original_hyperedge': edge_id,
                        'pattern': 'star',
                        'center_node': center_node,
                        **edge_data
                    }
                    
                    kg.add_relationship(center_node, other_node, relation_type, **relationship_data)
                    edges_converted += 1
            
            elif edge_strategy == 'clique':
                # Create clique (all-to-all connections)
                for node1 in edge_nodes:
                    for node2 in edge_nodes:
                        if node1 != node2:
                            relation_type = edge_data.get('type', 'clique_member')
                            
                            relationship_data = {
                                'original_hyperedge': edge_id,
                                'pattern': 'clique',
                                **edge_data
                            }
                            
                            kg.add_relationship(node1, node2, relation_type, **relationship_data)
                            edges_converted += 1
        
        # Store conversion statistics
        self.conversion_stats = {
            'source_type': 'Hypergraph',
            'target_type': 'KnowledgeGraph',
            'nodes_converted': nodes_converted,
            'edges_converted': edges_converted,
            'edge_strategy': edge_strategy,
            'preserve_properties': preserve_properties,
            'semantic_inference': semantic_inference
        }
        
        logger.info(f"Converted Hypergraph to KnowledgeGraph: {nodes_converted} nodes, {edges_converted} edges")
        
        return kg
    
    def _infer_node_type(self, node_id: str, node_data: dict, hypergraph: Hypergraph) -> Optional[str]:
        """Infer semantic type of a node based on context"""
        
        # Check explicit type in data
        if 'type' in node_data:
            return node_data['type']
        
        # Infer from node ID patterns
        if isinstance(node_id, str):
            node_lower = node_id.lower()
            
            # Common entity type patterns
            if any(pattern in node_lower for pattern in ['person', 'user', 'author', 'people']):
                return 'Person'
            elif any(pattern in node_lower for pattern in ['org', 'company', 'corp', 'institution']):
                return 'Organization'
            elif any(pattern in node_lower for pattern in ['place', 'location', 'city', 'country']):
                return 'Place'
            elif any(pattern in node_lower for pattern in ['event', 'meeting', 'conference']):
                return 'Event'
            elif any(pattern in node_lower for pattern in ['concept', 'idea', 'topic']):
                return 'Concept'
        
        # Infer from connectivity patterns
        try:
            node_edges = hypergraph.get_node_edges(node_id)
            node_degree = len(node_edges)
            
            if node_degree > 10:
                return 'Hub'  # Highly connected nodes
            elif node_degree == 1:
                return 'Leaf'  # Terminal nodes
            else:
                return 'Entity'  # General entity
        except:
            return 'Entity'


class KGToMetagraphMigrator:
    """Migrate KnowledgeGraph to Metagraph format"""
    
    def __init__(self):
        """Initialize migrator"""
        self.migration_stats = {}
        logger.info("KGToMetagraphMigrator initialized")
    
    def migrate(self, kg: KnowledgeGraph, **kwargs) -> Metagraph:
        """
        Migrate KnowledgeGraph to Metagraph
        
        Parameters
        ----------
        kg : KnowledgeGraph
            Source knowledge graph to migrate
        **kwargs : dict
            Migration options:
            - create_meta_nodes: bool (default: True)
            - infer_meta_relationships: bool (default: True)
            - preserve_semantics: bool (default: True)
            
        Returns
        -------
        Metagraph
            Migrated metagraph
        """
        create_meta_nodes = kwargs.get('create_meta_nodes', True)
        infer_meta_relationships = kwargs.get('infer_meta_relationships', True)
        preserve_semantics = kwargs.get('preserve_semantics', True)
        
        # Create new metagraph
        metagraph = Metagraph(name=f"MG_from_{kg.name}")
        
        # Migrate nodes
        nodes_migrated = 0
        entity_types = defaultdict(list)
        
        for node_id in kg.nodes:
            # Get node data
            node_data = kg.get_node_data(node_id) or {}
            
            # Determine entity type for meta-structure
            entity_type = node_data.get('semantic_type', 'Entity')
            entity_types[entity_type].append(node_id)
            
            # Add to metagraph with type information
            if preserve_semantics:
                meta_node_data = {
                    'original_kg_node': node_id,
                    'entity_type': entity_type,
                    **node_data
                }
            else:
                meta_node_data = {'entity_type': entity_type}
            
            metagraph.add_node(node_id, **meta_node_data)
            nodes_migrated += 1
        
        # Migrate relationships
        relationships_migrated = 0
        
        for edge in kg.edges:
            edge_data = kg.get_edge_data(edge) or {}
            edge_nodes = list(edge) if hasattr(edge, '__iter__') else [edge]
            
            if len(edge_nodes) >= 2:
                source, target = edge_nodes[0], edge_nodes[1]
                relation_type = edge_data.get('type', 'related_to')
                
                # Add relationship with metadata
                if preserve_semantics:
                    meta_edge_data = {
                        'original_kg_edge': str(edge),
                        'relation_type': relation_type,
                        **edge_data
                    }
                else:
                    meta_edge_data = {'relation_type': relation_type}
                
                metagraph.add_relationship(source, target, **meta_edge_data)
                relationships_migrated += 1
        
        # Create meta-nodes (nodes that represent collections/types)
        if create_meta_nodes:
            meta_nodes_created = 0
            
            for entity_type, entities in entity_types.items():
                if len(entities) > 1:  # Only create meta-nodes for types with multiple instances
                    meta_node_id = f"META_{entity_type}"
                    
                    metagraph.add_meta_node(meta_node_id, {
                        'type': 'meta_node',
                        'represents': entity_type,
                        'instance_count': len(entities),
                        'instances': entities[:10]  # Store sample instances
                    })
                    
                    # Connect meta-node to instances
                    for entity in entities:
                        metagraph.add_meta_relationship(
                            meta_node_id, entity, 
                            relation_type='instance_of'
                        )
                    
                    meta_nodes_created += 1
        
        # Infer meta-relationships (relationships between types/patterns)
        if infer_meta_relationships:
            meta_relationships_created = self._infer_meta_relationships(metagraph, entity_types)
        else:
            meta_relationships_created = 0
        
        # Store migration statistics
        self.migration_stats = {
            'source_type': 'KnowledgeGraph',
            'target_type': 'Metagraph',
            'nodes_migrated': nodes_migrated,
            'relationships_migrated': relationships_migrated,
            'meta_nodes_created': meta_nodes_created if create_meta_nodes else 0,
            'meta_relationships_created': meta_relationships_created,
            'entity_types_found': len(entity_types),
            'create_meta_nodes': create_meta_nodes,
            'preserve_semantics': preserve_semantics
        }
        
        logger.info(f"Migrated KnowledgeGraph to Metagraph: {nodes_migrated} nodes, {relationships_migrated} relationships")
        
        return metagraph
    
    def _infer_meta_relationships(self, metagraph: Metagraph, entity_types: Dict[str, List[str]]) -> int:
        """Infer relationships between entity types based on instance relationships"""
        
        meta_relationships = defaultdict(int)
        
        # Analyze relationships between instances of different types
        for relationship in metagraph.get_relationships():
            source, target = relationship['source'], relationship['target']
            
            # Find types of source and target
            source_type = None
            target_type = None
            
            for entity_type, entities in entity_types.items():
                if source in entities:
                    source_type = entity_type
                if target in entities:
                    target_type = entity_type
            
            if source_type and target_type and source_type != target_type:
                meta_rel_key = (source_type, target_type)
                meta_relationships[meta_rel_key] += 1
        
        # Create meta-relationships for frequent patterns
        meta_rels_created = 0
        threshold = 2  # Minimum occurrences to create meta-relationship
        
        for (source_type, target_type), count in meta_relationships.items():
            if count >= threshold:
                meta_source = f"META_{source_type}"
                meta_target = f"META_{target_type}"
                
                if metagraph.has_meta_node(meta_source) and metagraph.has_meta_node(meta_target):
                    metagraph.add_meta_relationship(
                        meta_source, meta_target,
                        relation_type='type_relationship',
                        strength=count,
                        pattern_frequency=count
                    )
                    meta_rels_created += 1
        
        return meta_rels_created


class GraphConverter:
    """Generic graph converter that routes to appropriate specialized converters"""
    
    def __init__(self, source_type: str, target_type: str):
        """Initialize converter for specific type pair"""
        self.source_type = source_type
        self.target_type = target_type
        
        # Load appropriate converter
        if source_type == 'hypergraph' and target_type == 'knowledge_graph':
            self.converter = HypergraphToKGConverter()
        elif source_type == 'knowledge_graph' and target_type == 'metagraph':
            self.converter = KGToMetagraphMigrator()
        else:
            # Generic conversion using intermediate steps
            self.converter = self._create_generic_converter()
        
        logger.info(f"GraphConverter initialized: {source_type} -> {target_type}")
    
    def convert(self, graph: Any, **kwargs) -> Any:
        """Convert graph using appropriate converter"""
        if hasattr(self.converter, 'convert'):
            return self.converter.convert(graph, **kwargs)
        elif hasattr(self.converter, 'migrate'):
            return self.converter.migrate(graph, **kwargs)
        else:
            return self.converter(graph, **kwargs)
    
    def _create_generic_converter(self):
        """Create generic converter for unsupported type pairs"""
        def generic_convert(graph, **kwargs):
            # For now, return the original graph
            # In a full implementation, this would handle multi-step conversions
            logger.warning(f"Generic conversion {self.source_type} -> {self.target_type} not fully implemented")
            return graph
        
        return generic_convert