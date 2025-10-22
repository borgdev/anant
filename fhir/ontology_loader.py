"""
FHIR Ontology Loader for Hierarchical Knowledge Graph
====================================================

This module loads FHIR schema files (turtle/RDF format) into ANANT's
hierarchical knowledge graph structure, preserving the semantic relationships
and class hierarchies defined in the FHIR ontologies.

Features:
- Load multiple turtle files (rim.ttl, fhir.ttl, w5.ttl)
- Extract class hierarchies and relationships
- Create hierarchical levels based on FHIR structure
- Preserve semantic relationships and properties
- Support for FHIR-specific ontology patterns
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import zipfile
import tempfile

# ANANT imports
from anant.kg import HierarchicalKnowledgeGraph

# Optional RDF dependencies
try:
    import rdflib
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL
    HAS_RDFLIB = True
    
    # FHIR-specific namespaces
    FHIR_NS = Namespace("http://hl7.org/fhir/")
    RIM_NS = Namespace("http://hl7.org/orim/class/")
    DT_NS = Namespace("http://hl7.org/orim/datatype/")
    VS_NS = Namespace("http://hl7.org/orim/valueset/")
    CS_NS = Namespace("http://hl7.org/orim/codesystem/")
    
except ImportError:
    HAS_RDFLIB = False
    print("Warning: rdflib not available. Install with: pip install rdflib")
    
    # Create dummy values for when rdflib is not available
    Graph = None
    Namespace = None
    URIRef = None
    Literal = None
    FHIR_NS = None
    RIM_NS = None
    DT_NS = None
    VS_NS = None
    CS_NS = None
    RDF = None
    RDFS = None
    OWL = None

logger = logging.getLogger(__name__)


class FHIROntologyLoader:
    """
    Loads FHIR ontology files into hierarchical knowledge graph.
    
    This class handles the parsing of FHIR turtle files and creates
    a structured hierarchy that reflects the FHIR specification.
    """
    
    def __init__(self, schema_dir: str = "schema"):
        """
        Initialize the FHIR ontology loader.
        
        Args:
            schema_dir: Directory containing FHIR schema files
        """
        self.schema_dir = Path(schema_dir)
        self.graphs = {}  # filename -> rdflib.Graph
        self.hkg = None
        self.class_hierarchy = {}
        self.loaded_files = []
        
        if not HAS_RDFLIB:
            raise ImportError("rdflib is required for ontology loading. Install with: pip install rdflib")
            
        logger.info(f"Initialized FHIR ontology loader for {schema_dir}")
    
    def load_ontology_files(self) -> Dict[str, Any]:
        """
        Load all FHIR ontology files from the schema directory.
        
        Returns:
            Dictionary with loading statistics and metadata
        """
        results = {
            'loaded_files': [],
            'total_triples': 0,
            'classes_found': 0,
            'properties_found': 0,
            'errors': []
        }
        
        # Find and load turtle files
        turtle_files = [
            'rim.ttl',
            'fhir.ttl', 
            'w5.ttl'
        ]
        
        for file_name in turtle_files:
            file_path = self.schema_dir / file_name
            
            if file_path.exists():
                try:
                    graph = self._load_turtle_file(file_path)
                    self.graphs[file_name] = graph
                    
                    results['loaded_files'].append(str(file_path))
                    results['total_triples'] += len(graph)
                    
                    logger.info(f"Loaded {file_name}: {len(graph)} triples")
                    
                except Exception as e:
                    error_msg = f"Failed to load {file_name}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            else:
                logger.warning(f"Schema file not found: {file_path}")
        
        # Check for compressed files
        zip_file = self.schema_dir / 'fhir.rdf.ttl.zip'
        if zip_file.exists():
            try:
                graph = self._load_compressed_turtle(zip_file)
                self.graphs['fhir_compressed.ttl'] = graph
                
                results['loaded_files'].append(str(zip_file))
                results['total_triples'] += len(graph)
                
                logger.info(f"Loaded compressed FHIR: {len(graph)} triples")
                
            except Exception as e:
                error_msg = f"Failed to load compressed file: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Analyze loaded ontologies
        if self.graphs:
            classes, properties = self._analyze_ontologies()
            results['classes_found'] = len(classes)
            results['properties_found'] = len(properties)
        
        self.loaded_files = results['loaded_files']
        return results
    
    def _load_turtle_file(self, file_path: Path) -> Graph:
        """Load a single turtle file into an RDF graph."""
        graph = Graph()
        graph.parse(file_path, format='turtle')
        return graph
    
    def _load_compressed_turtle(self, zip_path: Path) -> Graph:
        """Load turtle content from a compressed file."""
        graph = Graph()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find turtle files in the archive
            turtle_files = [f for f in zip_file.namelist() if f.endswith('.ttl')]
            
            for turtle_file in turtle_files:
                with zip_file.open(turtle_file) as f:
                    content = f.read().decode('utf-8')
                    graph.parse(data=content, format='turtle')
        
        return graph
    
    def _analyze_ontologies(self) -> Tuple[Set[str], Set[str]]:
        """
        Analyze loaded ontologies to extract classes and properties.
        
        Returns:
            Tuple of (classes, properties) found in the ontologies
        """
        classes = set()
        properties = set()
        
        for graph in self.graphs.values():
            # Find classes (rdfs:Class or owl:Class)
            for subj in graph.subjects(RDF.type, RDFS.Class):
                classes.add(str(subj))
            
            for subj in graph.subjects(RDF.type, OWL.Class):
                classes.add(str(subj))
            
            # Find properties (rdf:Property, owl:ObjectProperty, owl:DatatypeProperty)
            for subj in graph.subjects(RDF.type, RDF.Property):
                properties.add(str(subj))
            
            for subj in graph.subjects(RDF.type, OWL.ObjectProperty):
                properties.add(str(subj))
            
            for subj in graph.subjects(RDF.type, OWL.DatatypeProperty):
                properties.add(str(subj))
        
        return classes, properties
    
    def create_hierarchical_knowledge_graph(self) -> HierarchicalKnowledgeGraph:
        """
        Create a hierarchical knowledge graph from loaded FHIR ontologies.
        
        Returns:
            Populated HierarchicalKnowledgeGraph instance
        """
        if not self.graphs:
            raise ValueError("No ontology files loaded. Call load_ontology_files() first.")
        
        # Initialize hierarchical knowledge graph
        self.hkg = HierarchicalKnowledgeGraph("FHIR_Ontology")
        
        # Create hierarchical levels based on FHIR structure
        self._create_fhir_levels()
        
        # Extract and add classes
        self._add_classes_to_graph()
        
        # Extract and add properties
        self._add_properties_to_graph()
        
        # Extract and add relationships
        self._add_relationships_to_graph()
        
        logger.info(f"Created hierarchical knowledge graph with {self.hkg.num_nodes} nodes and {self.hkg.num_edges} edges")
        
        return self.hkg
    
    def _create_fhir_levels(self):
        """Create hierarchical levels based on FHIR ontology structure."""
        levels = [
            {
                'id': 'meta',
                'name': 'Meta Level',
                'description': 'Meta-classes and abstract concepts',
                'order': 0
            },
            {
                'id': 'datatypes',
                'name': 'Data Types',
                'description': 'FHIR primitive and complex data types',
                'order': 1
            },
            {
                'id': 'resources',
                'name': 'Resources',
                'description': 'FHIR resource types',
                'order': 2
            },
            {
                'id': 'elements',
                'name': 'Elements',
                'description': 'FHIR resource elements and components',
                'order': 3
            },
            {
                'id': 'valuesets',
                'name': 'Value Sets',
                'description': 'FHIR value sets and code systems',
                'order': 4
            }
        ]
        
        for level in levels:
            self.hkg.create_level(
                level['id'],
                level['name'],
                level['description'],
                level['order']
            )
    
    def _add_classes_to_graph(self):
        """Extract and add classes from ontologies to the knowledge graph."""
        for graph_name, graph in self.graphs.items():
            # Find all classes
            classes = set()
            
            for subj in graph.subjects(RDF.type, RDFS.Class):
                classes.add(subj)
            
            for subj in graph.subjects(RDF.type, OWL.Class):
                classes.add(subj)
            
            # Add classes to appropriate levels
            for class_uri in classes:
                self._add_class_to_appropriate_level(class_uri, graph, graph_name)
    
    def _add_class_to_appropriate_level(self, class_uri: URIRef, graph: Graph, source_file: str):
        """Add a class to the appropriate hierarchical level."""
        class_str = str(class_uri)
        
        # Determine appropriate level based on namespace and type
        level_id = self._determine_class_level(class_uri, graph)
        
        # Extract class properties
        properties = self._extract_class_properties(class_uri, graph, source_file)
        
        # Add to knowledge graph
        self.hkg.add_entity_to_level(
            class_str,
            level_id,
            'Class',
            properties
        )
    
    def _determine_class_level(self, class_uri: URIRef, graph: Graph) -> str:
        """Determine which hierarchical level a class should be placed in."""
        class_str = str(class_uri)
        
        # Check namespace to determine level
        if class_str.startswith(str(DT_NS)):
            return 'datatypes'
        elif class_str.startswith(str(RIM_NS)):
            return 'meta'
        elif class_str.startswith(str(FHIR_NS)):
            # Check if it's a resource type
            if any(graph.triples((class_uri, RDFS.subClassOf, None))):
                # Has superclass, likely a specific resource
                return 'resources'
            else:
                # No clear superclass, might be meta-level
                return 'meta'
        elif class_str.startswith(str(VS_NS)) or class_str.startswith(str(CS_NS)):
            return 'valuesets'
        else:
            # Default to elements level
            return 'elements'
    
    def _extract_class_properties(self, class_uri: URIRef, graph: Graph, source_file: str) -> Dict[str, Any]:
        """Extract properties for a class from the RDF graph."""
        properties = {
            'uri': str(class_uri),
            'source_file': source_file,
            'type': 'Class',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Extract label
        for label in graph.objects(class_uri, RDFS.label):
            properties['label'] = str(label)
            break
        
        # Extract comment/description
        for comment in graph.objects(class_uri, RDFS.comment):
            properties['description'] = str(comment)
            break
        
        # Extract additional properties
        for pred, obj in graph.predicate_objects(class_uri):
            pred_str = str(pred)
            
            if pred_str.endswith('title'):
                properties['title'] = str(obj)
            elif pred_str.endswith('terms'):
                properties['terms'] = str(obj)
        
        # Extract superclasses
        superclasses = []
        for superclass in graph.objects(class_uri, RDFS.subClassOf):
            superclasses.append(str(superclass))
        
        if superclasses:
            properties['superclasses'] = superclasses
        
        return properties
    
    def _add_properties_to_graph(self):
        """Extract and add properties from ontologies to the knowledge graph."""
        for graph_name, graph in self.graphs.values():
            # Find all properties
            properties = set()
            
            for subj in graph.subjects(RDF.type, RDF.Property):
                properties.add(subj)
            
            for subj in graph.subjects(RDF.type, OWL.ObjectProperty):
                properties.add(subj)
            
            for subj in graph.subjects(RDF.type, OWL.DatatypeProperty):
                properties.add(subj)
            
            # Add properties to knowledge graph
            for prop_uri in properties:
                self._add_property_to_graph(prop_uri, graph, graph_name)
    
    def _add_property_to_graph(self, prop_uri: URIRef, graph: Graph, source_file: str):
        """Add a property to the knowledge graph."""
        prop_str = str(prop_uri)
        
        # Extract property metadata
        properties = {
            'uri': prop_str,
            'source_file': source_file,
            'type': 'Property',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Extract label and description
        for label in graph.objects(prop_uri, RDFS.label):
            properties['label'] = str(label)
            break
        
        for comment in graph.objects(prop_uri, RDFS.comment):
            properties['description'] = str(comment)
            break
        
        # Extract domain and range
        for domain in graph.objects(prop_uri, RDFS.domain):
            properties['domain'] = str(domain)
            break
        
        for range_obj in graph.objects(prop_uri, RDFS.range):
            properties['range'] = str(range_obj)
            break
        
        # Add to elements level (properties are elements)
        self.hkg.add_entity_to_level(
            prop_str,
            'elements',
            'Property',
            properties
        )
    
    def _add_relationships_to_graph(self):
        """Extract and add relationships from ontologies."""
        relationship_id = 0
        
        for graph_name, graph in self.graphs.items():
            # Add subclass relationships
            for subj, obj in graph.subject_objects(RDFS.subClassOf):
                relationship_id += 1
                
                self.hkg.add_cross_level_relationship(
                    f"subclass_{relationship_id}",
                    str(subj),
                    str(obj),
                    'subClassOf',
                    {
                        'source_file': graph_name,
                        'relationship_type': 'subClassOf',
                        'semantic_weight': 0.9
                    }
                )
            
            # Add domain/range relationships for properties
            for prop in graph.subjects(RDF.type, RDF.Property):
                for domain in graph.objects(prop, RDFS.domain):
                    relationship_id += 1
                    
                    self.hkg.add_cross_level_relationship(
                        f"domain_{relationship_id}",
                        str(prop),
                        str(domain),
                        'domain',
                        {
                            'source_file': graph_name,
                            'relationship_type': 'domain',
                            'semantic_weight': 0.8
                        }
                    )
                
                for range_obj in graph.objects(prop, RDFS.range):
                    relationship_id += 1
                    
                    self.hkg.add_cross_level_relationship(
                        f"range_{relationship_id}",
                        str(prop),
                        str(range_obj),
                        'range',
                        {
                            'source_file': graph_name,
                            'relationship_type': 'range',
                            'semantic_weight': 0.8
                        }
                    )
    
    def get_ontology_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded ontologies.
        
        Returns:
            Dictionary with ontology statistics
        """
        if not self.hkg:
            return {'error': 'No hierarchical knowledge graph created yet'}
        
        stats = self.hkg.get_hierarchy_statistics()
        
        # Add ontology-specific statistics
        stats.update({
            'loaded_files': self.loaded_files,
            'source_graphs': len(self.graphs),
            'total_triples': sum(len(g) for g in self.graphs.values()),
            'levels': {
                level_id: {
                    'entities': len(self.hkg.get_entities_at_level(level_id)),
                    'metadata': self.hkg.get_level_metadata(level_id)
                }
                for level_id in ['meta', 'datatypes', 'resources', 'elements', 'valuesets']
            }
        })
        
        return stats


def load_fhir_ontologies(schema_dir: str = "schema") -> Tuple[HierarchicalKnowledgeGraph, Dict[str, Any]]:
    """
    Convenience function to load FHIR ontologies into a hierarchical knowledge graph.
    
    Args:
        schema_dir: Directory containing FHIR schema files
        
    Returns:
        Tuple of (HierarchicalKnowledgeGraph, statistics)
    """
    loader = FHIROntologyLoader(schema_dir)
    
    # Load ontology files
    load_results = loader.load_ontology_files()
    
    if load_results['errors']:
        logger.warning(f"Encountered {len(load_results['errors'])} errors during loading")
    
    # Create hierarchical knowledge graph
    hkg = loader.create_hierarchical_knowledge_graph()
    
    # Get statistics
    stats = loader.get_ontology_statistics()
    stats['load_results'] = load_results
    
    return hkg, stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load FHIR ontologies
    hkg, stats = load_fhir_ontologies("schema")
    
    print("\n=== FHIR Ontology Loading Results ===")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Levels created: {stats['total_levels']}")
    print(f"Source files: {len(stats['loaded_files'])}")
    
    print("\nLevel breakdown:")
    for level_id, level_info in stats['levels'].items():
        print(f"  {level_id}: {level_info['entities']} entities")
    
    if stats['load_results']['errors']:
        print(f"\nErrors encountered: {len(stats['load_results']['errors'])}")
        for error in stats['load_results']['errors']:
            print(f"  - {error}")