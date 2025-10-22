"""
I/O Operations for Hierarchical Knowledge Graph
==============================================

This module handles all input/output operations for hierarchical knowledge graphs,
including format conversion, serialization, import/export capabilities, and data interchange.

Key Features:
- JSON serialization and deserialization
- NetworkX graph conversion (directed and undirected)
- GEXF format export for Gephi visualization
- GraphML format support for graph analysis tools
- CSV/TSV export for tabular data analysis
- Pickle serialization for fast loading/saving
- XML export with hierarchical structure preservation
"""

from typing import Dict, List, Any, Optional, Union, TextIO, BinaryIO
from pathlib import Path
import json
import csv
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
import logging
import tempfile
import gzip

logger = logging.getLogger(__name__)


class IOOperations:
    """
    Handles input/output operations for hierarchical knowledge graphs.
    
    This class provides comprehensive I/O capabilities for saving, loading,
    and converting hierarchical knowledge graphs to various formats suitable
    for analysis, visualization, and data interchange.
    
    Features:
    - Multiple format support (JSON, NetworkX, GEXF, GraphML, CSV, XML)
    - Compressed and uncompressed serialization
    - Metadata preservation during format conversion
    - Batch import/export operations
    - Format validation and error handling
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize I/O operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
    
    # =====================================================================
    # JSON SERIALIZATION
    # =====================================================================
    
    def to_json(self, 
                file_path: Optional[Union[str, Path]] = None,
                include_metadata: bool = True,
                compress: bool = False) -> Union[str, None]:
        """
        Serialize hierarchical knowledge graph to JSON format.
        
        Args:
            file_path: Path to save JSON file (None returns JSON string)
            include_metadata: Include metadata and timestamps
            compress: Use gzip compression
            
        Returns:
            JSON string if file_path is None, otherwise None
        """
        # Build comprehensive JSON representation
        json_data = {
            'format': 'hierarchical_knowledge_graph',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'name': self.hkg.name,
            'settings': {
                'enable_semantic_reasoning': self.hkg.enable_semantic_reasoning,
                'enable_temporal_tracking': self.hkg.enable_temporal_tracking
            }
        }
        
        # Export hierarchy structure
        json_data['hierarchy'] = {
            'levels': self._export_levels(include_metadata),
            'level_order': self.hkg.level_order,
            'entity_levels': self.hkg.entity_levels
        }
        
        # Export entities and relationships
        json_data['entities'] = self._export_entities(include_metadata)
        json_data['relationships'] = self._export_relationships(include_metadata)
        json_data['cross_level_relationships'] = self._export_cross_level_relationships(include_metadata)
        
        # Add statistics if metadata included
        if include_metadata:
            json_data['metadata'] = {
                'total_entities': self.hkg.num_nodes(),
                'total_relationships': self.hkg.num_edges(),
                'total_levels': len(self.hkg.levels),
                'cross_level_relationships_count': len(self.hkg.cross_level_relationships),
                'export_timestamp': datetime.now().isoformat()
            }
        
        # Convert to JSON string
        json_string = json.dumps(json_data, indent=2, default=self._json_serializer)
        
        # Save to file if path provided
        if file_path:
            file_path = Path(file_path)
            
            if compress:
                with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                    f.write(json_string)
                logger.info(f"Saved compressed JSON to {file_path}.gz")
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_string)
                logger.info(f"Saved JSON to {file_path}")
            
            return None
        
        return json_string
    
    def from_json(self, 
                  source: Union[str, Path, Dict[str, Any]],
                  validate_format: bool = True) -> bool:
        """
        Load hierarchical knowledge graph from JSON format.
        
        Args:
            source: JSON file path, JSON string, or dictionary
            validate_format: Validate JSON format before loading
            
        Returns:
            Success status
        """
        try:
            # Parse JSON data
            if isinstance(source, dict):
                json_data = source
            elif isinstance(source, (str, Path)):
                source_path = Path(source)
                if source_path.exists():
                    # Load from file
                    if source_path.suffix == '.gz':
                        with gzip.open(source_path, 'rt', encoding='utf-8') as f:
                            json_data = json.load(f)
                    else:
                        with open(source_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                else:
                    # Parse as JSON string
                    json_data = json.loads(source)
            else:
                raise ValueError(f"Invalid source type: {type(source)}")
            
            # Validate format
            if validate_format and not self._validate_json_format(json_data):
                return False
            
            # Clear existing data
            self.hkg.clear()
            
            # Load settings
            settings = json_data.get('settings', {})
            self.hkg.enable_semantic_reasoning = settings.get('enable_semantic_reasoning', True)
            self.hkg.enable_temporal_tracking = settings.get('enable_temporal_tracking', False)
            self.hkg.name = json_data.get('name', 'Loaded_HKG')
            
            # Load hierarchy structure
            hierarchy_data = json_data.get('hierarchy', {})
            self._import_levels(hierarchy_data.get('levels', {}))
            self.hkg.level_order = hierarchy_data.get('level_order', {})
            self.hkg.entity_levels = hierarchy_data.get('entity_levels', {})
            
            # Load entities and relationships
            self._import_entities(json_data.get('entities', []))
            self._import_relationships(json_data.get('relationships', []))
            self._import_cross_level_relationships(json_data.get('cross_level_relationships', []))
            
            logger.info(f"Successfully loaded hierarchical knowledge graph from JSON")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from JSON: {e}")
            return False
    
    def _export_levels(self, include_metadata: bool) -> Dict[str, Any]:
        """Export level information to dictionary."""
        levels_data = {}
        
        for level_id, level_info in self.hkg.levels.items():
            level_data = {
                'id': level_id,
                'name': level_info.get('name', level_id),
                'description': level_info.get('description', ''),
                'level_order': self.hkg.level_order.get(level_id, 0),
                'entity_count': len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
            }
            
            if include_metadata:
                level_data['metadata'] = level_info.get('metadata', {})
                level_data['created_at'] = level_info.get('created_at', '')
            
            levels_data[level_id] = level_data
        
        return levels_data
    
    def _export_entities(self, include_metadata: bool) -> List[Dict[str, Any]]:
        """Export entity information to list."""
        entities_data = []
        
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if entity_data:
                exported_entity = {
                    'id': entity_id,
                    'type': entity_data.get('type', 'entity'),
                    'properties': entity_data.get('properties', {}),
                    'level': self.hkg.entity_levels.get(entity_id)
                }
                
                if include_metadata and 'metadata' in entity_data:
                    exported_entity['metadata'] = entity_data['metadata']
                
                entities_data.append(exported_entity)
        
        return entities_data
    
    def _export_relationships(self, include_metadata: bool) -> List[Dict[str, Any]]:
        """Export relationship information to list."""
        relationships_data = []
        
        all_relationships = self.hkg.knowledge_graph.get_all_relationships()
        for relationship in all_relationships:
            exported_rel = {
                'id': relationship.get('id', ''),
                'source_entity': relationship.get('source_entity', ''),
                'target_entity': relationship.get('target_entity', ''),
                'relationship_type': relationship.get('relationship_type', ''),
                'properties': relationship.get('properties', {})
            }
            
            if include_metadata and 'metadata' in relationship:
                exported_rel['metadata'] = relationship['metadata']
            
            relationships_data.append(exported_rel)
        
        return relationships_data
    
    def _export_cross_level_relationships(self, include_metadata: bool) -> List[Dict[str, Any]]:
        """Export cross-level relationship information."""
        cross_level_data = []
        
        for relationship in self.hkg.cross_level_relationships:
            exported_rel = relationship.copy()
            
            if not include_metadata:
                # Remove metadata fields
                exported_rel.pop('created_at', None)
                exported_rel.pop('metadata', None)
            
            cross_level_data.append(exported_rel)
        
        return cross_level_data
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for non-standard types."""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return obj.__dict__
        else:
            return str(obj)
    
    def _validate_json_format(self, json_data: Dict[str, Any]) -> bool:
        """Validate JSON format for hierarchical knowledge graph."""
        required_fields = ['format', 'hierarchy', 'entities']
        
        for field in required_fields:
            if field not in json_data:
                logger.error(f"Missing required field in JSON: {field}")
                return False
        
        if json_data.get('format') != 'hierarchical_knowledge_graph':
            logger.error(f"Invalid format: {json_data.get('format')}")
            return False
        
        return True
    
    # =====================================================================
    # NETWORKX CONVERSION
    # =====================================================================
    
    def to_networkx(self, 
                   include_cross_level: bool = True,
                   directed: bool = False) -> 'nx.Graph':
        """
        Convert hierarchical knowledge graph to NetworkX graph.
        
        Args:
            include_cross_level: Include cross-level relationships
            directed: Create directed graph
            
        Returns:
            NetworkX Graph or DiGraph object
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for graph conversion. Install with: pip install networkx")
        
        # Create appropriate graph type
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes with attributes
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            node_attrs = {
                'level': self.hkg.entity_levels.get(entity_id, 'unassigned'),
                'level_order': self.hkg.level_order.get(self.hkg.entity_levels.get(entity_id, ''), 0)
            }
            
            if entity_data:
                node_attrs.update({
                    'entity_type': entity_data.get('type', 'entity'),
                    **entity_data.get('properties', {})
                })
            
            G.add_node(entity_id, **node_attrs)
        
        # Add edges from regular relationships
        all_relationships = self.hkg.knowledge_graph.get_all_relationships()
        for relationship in all_relationships:
            source = relationship.get('source_entity')
            target = relationship.get('target_entity')
            
            if source and target:
                edge_attrs = {
                    'relationship_type': relationship.get('relationship_type', ''),
                    'is_cross_level': False,
                    **relationship.get('properties', {})
                }
                
                G.add_edge(source, target, **edge_attrs)
        
        # Add cross-level relationships if requested
        if include_cross_level:
            for relationship in self.hkg.cross_level_relationships:
                source = relationship.get('source_entity')
                target = relationship.get('target_entity')
                
                if source and target:
                    edge_attrs = {
                        'relationship_type': relationship.get('relationship_type', ''),
                        'is_cross_level': True,
                        'source_level': relationship.get('source_level', ''),
                        'target_level': relationship.get('target_level', ''),
                        'level_distance': relationship.get('level_distance', 0),
                        **relationship.get('properties', {})
                    }
                    
                    G.add_edge(source, target, **edge_attrs)
        
        # Add graph attributes
        G.graph.update({
            'name': self.hkg.name,
            'type': 'hierarchical_knowledge_graph',
            'levels': len(self.hkg.levels),
            'enable_semantic_reasoning': self.hkg.enable_semantic_reasoning,
            'export_timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Converted to NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def from_networkx(self, 
                     nx_graph: 'nx.Graph',
                     level_attribute: str = 'level',
                     entity_type_attribute: str = 'entity_type') -> bool:
        """
        Load hierarchical knowledge graph from NetworkX graph.
        
        Args:
            nx_graph: NetworkX graph object
            level_attribute: Node attribute containing level information
            entity_type_attribute: Node attribute containing entity type
            
        Returns:
            Success status
        """
        try:
            # Clear existing data
            self.hkg.clear()
            
            # Extract graph metadata
            graph_attrs = nx_graph.graph
            self.hkg.name = graph_attrs.get('name', 'NetworkX_Import')
            
            # Create levels from node attributes
            levels_found = set()
            for node_id, node_attrs in nx_graph.nodes(data=True):
                level = node_attrs.get(level_attribute, 'default_level')
                levels_found.add(level)
            
            # Create levels
            for i, level in enumerate(sorted(levels_found)):
                level_order = i
                if level_attribute == 'level_order' and level in nx_graph.nodes():
                    level_order = nx_graph.nodes[level].get('level_order', i)
                
                self.hkg.create_level(level, level.replace('_', ' ').title(), level_order=level_order)
            
            # Add entities
            for node_id, node_attrs in nx_graph.nodes(data=True):
                entity_type = node_attrs.get(entity_type_attribute, 'entity')
                level = node_attrs.get(level_attribute, 'default_level')
                
                # Extract properties (exclude special attributes)
                properties = {k: v for k, v in node_attrs.items() 
                            if k not in [level_attribute, entity_type_attribute, 'level_order']}
                properties['type'] = entity_type
                
                self.hkg.add_entity_to_level(node_id, entity_type, properties, level)
            
            # Add relationships
            for source, target, edge_attrs in nx_graph.edges(data=True):
                relationship_type = edge_attrs.get('relationship_type', 'related_to')
                is_cross_level = edge_attrs.get('is_cross_level', False)
                
                # Extract edge properties
                properties = {k: v for k, v in edge_attrs.items() 
                            if k not in ['relationship_type', 'is_cross_level', 'source_level', 'target_level', 'level_distance']}
                
                if is_cross_level:
                    self.hkg.add_cross_level_relationship(source, target, relationship_type, properties)
                else:
                    self.hkg.add_relationship(source, target, relationship_type, properties)
            
            logger.info(f"Successfully imported from NetworkX graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import from NetworkX: {e}")
            return False
    
    # =====================================================================
    # GEXF FORMAT (Gephi)
    # =====================================================================
    
    def to_gexf(self, file_path: Union[str, Path], include_cross_level: bool = True) -> bool:
        """
        Export hierarchical knowledge graph to GEXF format for Gephi visualization.
        
        Args:
            file_path: Path to save GEXF file
            include_cross_level: Include cross-level relationships
            
        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            
            # Create GEXF XML structure
            root = ET.Element("gexf")
            root.set("xmlns", "http://www.gexf.net/1.2draft")
            root.set("version", "1.2")
            
            # Meta information
            meta = ET.SubElement(root, "meta")
            meta.set("lastmodifieddate", datetime.now().isoformat())
            ET.SubElement(meta, "creator").text = "HierarchicalKnowledgeGraph"
            ET.SubElement(meta, "description").text = f"Hierarchical Knowledge Graph: {self.hkg.name}"
            
            # Graph element
            graph_elem = ET.SubElement(root, "graph")
            graph_elem.set("mode", "static")
            graph_elem.set("defaultedgetype", "undirected")
            
            # Node attributes definition
            node_attributes = ET.SubElement(graph_elem, "attributes")
            node_attributes.set("class", "node")
            
            # Define standard attributes
            attr_level = ET.SubElement(node_attributes, "attribute")
            attr_level.set("id", "0")
            attr_level.set("title", "level")
            attr_level.set("type", "string")
            
            attr_type = ET.SubElement(node_attributes, "attribute")
            attr_type.set("id", "1")
            attr_type.set("title", "entity_type")
            attr_type.set("type", "string")
            
            attr_order = ET.SubElement(node_attributes, "attribute")
            attr_order.set("id", "2")
            attr_order.set("title", "level_order")
            attr_order.set("type", "integer")
            
            # Nodes
            nodes_elem = ET.SubElement(graph_elem, "nodes")
            for entity_id in self.hkg.nodes():
                node_elem = ET.SubElement(nodes_elem, "node")
                node_elem.set("id", str(entity_id))
                node_elem.set("label", str(entity_id))
                
                # Node attributes
                attvalues = ET.SubElement(node_elem, "attvalues")
                
                level = self.hkg.entity_levels.get(entity_id, 'unassigned')
                level_attvalue = ET.SubElement(attvalues, "attvalue")
                level_attvalue.set("for", "0")
                level_attvalue.set("value", level)
                
                entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
                entity_type = entity_data.get('type', 'entity') if entity_data else 'entity'
                type_attvalue = ET.SubElement(attvalues, "attvalue")
                type_attvalue.set("for", "1")
                type_attvalue.set("value", entity_type)
                
                level_order = self.hkg.level_order.get(level, 0)
                order_attvalue = ET.SubElement(attvalues, "attvalue")
                order_attvalue.set("for", "2")
                order_attvalue.set("value", str(level_order))
            
            # Edges
            edges_elem = ET.SubElement(graph_elem, "edges")
            edge_id = 0
            
            # Regular relationships
            all_relationships = self.hkg.knowledge_graph.get_all_relationships()
            for relationship in all_relationships:
                source = relationship.get('source_entity')
                target = relationship.get('target_entity')
                
                if source and target:
                    edge_elem = ET.SubElement(edges_elem, "edge")
                    edge_elem.set("id", str(edge_id))
                    edge_elem.set("source", str(source))
                    edge_elem.set("target", str(target))
                    edge_elem.set("label", relationship.get('relationship_type', ''))
                    edge_id += 1
            
            # Cross-level relationships
            if include_cross_level:
                for relationship in self.hkg.cross_level_relationships:
                    source = relationship.get('source_entity')
                    target = relationship.get('target_entity')
                    
                    if source and target:
                        edge_elem = ET.SubElement(edges_elem, "edge")
                        edge_elem.set("id", str(edge_id))
                        edge_elem.set("source", str(source))
                        edge_elem.set("target", str(target))
                        edge_elem.set("label", f"CROSS: {relationship.get('relationship_type', '')}")
                        edge_id += 1
            
            # Write to file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)  # Pretty print
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            
            logger.info(f"Exported GEXF format to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export GEXF: {e}")
            return False
    
    # =====================================================================
    # CSV EXPORT
    # =====================================================================
    
    def to_csv(self, 
              base_path: Union[str, Path],
              export_entities: bool = True,
              export_relationships: bool = True,
              export_levels: bool = True) -> Dict[str, Path]:
        """
        Export hierarchical knowledge graph to CSV files.
        
        Args:
            base_path: Base directory for CSV files
            export_entities: Export entities to CSV
            export_relationships: Export relationships to CSV
            export_levels: Export level information to CSV
            
        Returns:
            Dictionary mapping content type to file paths
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        try:
            # Export entities
            if export_entities:
                entities_file = base_path / f"{self.hkg.name}_entities.csv"
                with open(entities_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow(['entity_id', 'entity_type', 'level', 'level_order', 'properties_json'])
                    
                    # Data
                    for entity_id in self.hkg.nodes():
                        entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
                        entity_type = entity_data.get('type', 'entity') if entity_data else 'entity'
                        level = self.hkg.entity_levels.get(entity_id, 'unassigned')
                        level_order = self.hkg.level_order.get(level, 0)
                        properties = entity_data.get('properties', {}) if entity_data else {}
                        
                        writer.writerow([
                            entity_id,
                            entity_type,
                            level,
                            level_order,
                            json.dumps(properties)
                        ])
                
                exported_files['entities'] = entities_file
            
            # Export relationships
            if export_relationships:
                relationships_file = base_path / f"{self.hkg.name}_relationships.csv"
                with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'source_entity', 'target_entity', 'relationship_type', 
                        'is_cross_level', 'source_level', 'target_level', 'properties_json'
                    ])
                    
                    # Regular relationships
                    all_relationships = self.hkg.knowledge_graph.get_all_relationships()
                    for relationship in all_relationships:
                        source = relationship.get('source_entity', '')
                        target = relationship.get('target_entity', '')
                        rel_type = relationship.get('relationship_type', '')
                        properties = relationship.get('properties', {})
                        
                        source_level = self.hkg.entity_levels.get(source, 'unassigned')
                        target_level = self.hkg.entity_levels.get(target, 'unassigned')
                        
                        writer.writerow([
                            source, target, rel_type, False,
                            source_level, target_level, json.dumps(properties)
                        ])
                    
                    # Cross-level relationships
                    for relationship in self.hkg.cross_level_relationships:
                        writer.writerow([
                            relationship.get('source_entity', ''),
                            relationship.get('target_entity', ''),
                            relationship.get('relationship_type', ''),
                            True,
                            relationship.get('source_level', ''),
                            relationship.get('target_level', ''),
                            json.dumps(relationship.get('properties', {}))
                        ])
                
                exported_files['relationships'] = relationships_file
            
            # Export levels
            if export_levels:
                levels_file = base_path / f"{self.hkg.name}_levels.csv"
                with open(levels_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow(['level_id', 'level_name', 'level_order', 'entity_count', 'description'])
                    
                    # Data
                    for level_id, level_info in self.hkg.levels.items():
                        writer.writerow([
                            level_id,
                            level_info.get('name', level_id),
                            self.hkg.level_order.get(level_id, 0),
                            len(self.hkg.hierarchy_ops.get_entities_at_level(level_id)),
                            level_info.get('description', '')
                        ])
                
                exported_files['levels'] = levels_file
            
            logger.info(f"Exported CSV files to {base_path}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return {}
    
    # =====================================================================
    # PICKLE SERIALIZATION (Fast Binary Format)
    # =====================================================================
    
    def to_pickle(self, file_path: Union[str, Path], compress: bool = True) -> bool:
        """
        Serialize hierarchical knowledge graph to pickle format.
        
        Args:
            file_path: Path to save pickle file
            compress: Use gzip compression
            
        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            
            # Prepare data for pickling
            pickle_data = {
                'name': self.hkg.name,
                'settings': {
                    'enable_semantic_reasoning': self.hkg.enable_semantic_reasoning,
                    'enable_temporal_tracking': self.hkg.enable_temporal_tracking
                },
                'levels': self.hkg.levels,
                'level_graphs': self.hkg.level_graphs,
                'level_order': self.hkg.level_order,
                'entity_levels': self.hkg.entity_levels,
                'cross_level_relationships': self.hkg.cross_level_relationships,
                'knowledge_graph': self.hkg.knowledge_graph,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            if compress:
                with gzip.open(f"{file_path}.gz", 'wb') as f:
                    pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved compressed pickle to {file_path}.gz")
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved pickle to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pickle: {e}")
            return False
    
    def from_pickle(self, file_path: Union[str, Path]) -> bool:
        """
        Load hierarchical knowledge graph from pickle format.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            
            # Determine if compressed
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    pickle_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    pickle_data = pickle.load(f)
            
            # Clear existing data
            self.hkg.clear()
            
            # Restore data
            self.hkg.name = pickle_data.get('name', 'Loaded_HKG')
            settings = pickle_data.get('settings', {})
            self.hkg.enable_semantic_reasoning = settings.get('enable_semantic_reasoning', True)
            self.hkg.enable_temporal_tracking = settings.get('enable_temporal_tracking', False)
            
            self.hkg.levels = pickle_data.get('levels', {})
            self.hkg.level_graphs = pickle_data.get('level_graphs', {})
            self.hkg.level_order = pickle_data.get('level_order', {})
            self.hkg.entity_levels = pickle_data.get('entity_levels', {})
            self.hkg.cross_level_relationships = pickle_data.get('cross_level_relationships', [])
            self.hkg.knowledge_graph = pickle_data.get('knowledge_graph')
            
            logger.info(f"Successfully loaded from pickle: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pickle: {e}")
            return False
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def _import_levels(self, levels_data: Dict[str, Any]):
        """Import level information from dictionary."""
        for level_id, level_info in levels_data.items():
            level_name = level_info.get('name', level_id)
            level_description = level_info.get('description', '')
            level_order = level_info.get('level_order', 0)
            
            self.hkg.create_level(level_id, level_name, level_description, level_order)
    
    def _import_entities(self, entities_data: List[Dict[str, Any]]):
        """Import entity information from list."""
        for entity_info in entities_data:
            entity_id = entity_info.get('id', '')
            entity_type = entity_info.get('type', 'entity')
            properties = entity_info.get('properties', {})
            level_id = entity_info.get('level')
            
            if entity_id:
                if level_id:
                    self.hkg.add_entity_to_level(entity_id, entity_type, properties, level_id)
                else:
                    self.hkg.add_entity(entity_id, properties)
    
    def _import_relationships(self, relationships_data: List[Dict[str, Any]]):
        """Import relationship information from list."""
        for rel_info in relationships_data:
            source = rel_info.get('source_entity', '')
            target = rel_info.get('target_entity', '')
            rel_type = rel_info.get('relationship_type', 'related_to')
            properties = rel_info.get('properties', {})
            
            if source and target:
                self.hkg.add_relationship(source, target, rel_type, properties)
    
    def _import_cross_level_relationships(self, cross_level_data: List[Dict[str, Any]]):
        """Import cross-level relationship information."""
        for rel_info in cross_level_data:
            source = rel_info.get('source_entity', '')
            target = rel_info.get('target_entity', '')
            rel_type = rel_info.get('relationship_type', 'cross_level_related')
            properties = rel_info.get('properties', {})
            
            if source and target:
                self.hkg.add_cross_level_relationship(source, target, rel_type, properties, validate_hierarchy=False)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported I/O formats."""
        return [
            'json',
            'json.gz',  # Compressed JSON
            'pickle',
            'pickle.gz',  # Compressed pickle
            'csv',  # CSV export (multiple files)
            'gexf',  # Gephi format
            'networkx'  # NetworkX graph object
        ]
    
    def get_format_info(self) -> Dict[str, Dict[str, str]]:
        """Get detailed information about supported formats."""
        return {
            'json': {
                'description': 'Human-readable JSON format with full metadata',
                'use_case': 'Data interchange, debugging, configuration',
                'file_size': 'Large',
                'load_speed': 'Medium'
            },
            'pickle': {
                'description': 'Python binary format for fast loading',
                'use_case': 'Production use, caching, performance',
                'file_size': 'Small',
                'load_speed': 'Fast'
            },
            'csv': {
                'description': 'Tabular format for analysis tools',
                'use_case': 'Data analysis, Excel, reporting',
                'file_size': 'Medium',
                'load_speed': 'N/A (export only)'
            },
            'gexf': {
                'description': 'Gephi XML format for visualization',
                'use_case': 'Network visualization, Gephi import',
                'file_size': 'Large',
                'load_speed': 'N/A (export only)'
            },
            'networkx': {
                'description': 'NetworkX graph object for analysis',
                'use_case': 'Graph algorithms, network analysis',
                'file_size': 'N/A (in-memory)',
                'load_speed': 'Fast'
            }
        }