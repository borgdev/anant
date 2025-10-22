"""
Advanced Ontology Processing System
==================================

Automatic hierarchy construction from various ontology formats with
intelligent pattern recognition and Schema.org compatibility.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import polars as pl
import time
from urllib.parse import urlparse

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies
rdflib = safe_import('rdflib')
owlready2 = safe_import('owlready2')

logger = logging.getLogger(__name__)


class OntologyFormat(Enum):
    """Supported ontology file formats"""
    RDF_XML = "rdf/xml"
    TURTLE = "turtle"
    N3 = "n3" 
    NT = "nt"
    JSON_LD = "json-ld"
    OWL_XML = "owl/xml"
    AUTO = "auto"


@dataclass
class OntologyEntity:
    """Represents an ontology entity (class, property, or instance)"""
    uri: str
    name: str
    entity_type: str  # 'class', 'property', 'instance'
    namespace: str
    labels: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Tuple[str, str]] = field(default_factory=list)  # (predicate, target)
    confidence: float = 1.0


@dataclass
class OntologyHierarchy:
    """Represents a complete ontology hierarchy"""
    classes: Dict[str, OntologyEntity] = field(default_factory=dict)
    properties: Dict[str, OntologyEntity] = field(default_factory=dict)
    instances: Dict[str, OntologyEntity] = field(default_factory=dict)
    hierarchy_graph: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    reverse_hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # child -> parents
    namespaces: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OntologyProcessor:
    """
    Advanced ontology processing with automatic hierarchy construction
    
    Features:
    - Multi-format support (RDF, OWL, JSON-LD, TTL, Schema.org)
    - Intelligent pattern recognition for class/property identification
    - Automatic hierarchy extraction and validation
    - Polars-optimized processing for large ontologies
    - Confidence scoring and validation
    """
    
    def __init__(self, kg_or_config=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ontology processor
        
        Args:
            kg_or_config: Either a KnowledgeGraph object or configuration dictionary
            config: Configuration dictionary (if first arg is KG)
        """
        # Handle different initialization patterns
        if hasattr(kg_or_config, 'nodes') and hasattr(kg_or_config, 'edges'):
            # First argument is a knowledge graph
            self.kg = kg_or_config
            self.config = config or {}
        elif isinstance(kg_or_config, dict):
            # First argument is config
            self.kg = None
            self.config = kg_or_config
        else:
            # Default initialization
            self.kg = None
            self.config = config or {}
        
        # Processing configuration
        self.settings = {
            'max_hierarchy_depth': self.config.get('max_hierarchy_depth', 10),
            'min_confidence_threshold': self.config.get('min_confidence_threshold', 0.7),
            'enable_polars_optimization': self.config.get('enable_polars_optimization', True),
            'batch_size': self.config.get('batch_size', 1000),
            'enable_inference': self.config.get('enable_inference', True),
            'strict_validation': self.config.get('strict_validation', False)
        }
        
        # Pattern recognition for different ontology formats
        self.patterns = {
            'class_indicators': [
                'owl:Class', 'rdfs:Class', 'Class', 'Type', 'Concept', 'Category',
                'Thing', 'Entity', 'Object', 'schema:Thing'
            ],
            'property_indicators': [
                'owl:ObjectProperty', 'owl:DatatypeProperty', 'rdf:Property',
                'Property', 'Relation', 'Attribute', 'Field'
            ],
            'hierarchy_relations': [
                'rdfs:subClassOf', 'subClassOf', 'rdfs:subPropertyOf', 'subPropertyOf',
                'owl:equivalentClass', 'owl:sameAs', 'schema:subOrganization'
            ],
            'instance_relations': [
                'rdf:type', 'a', 'instanceof', 'memberOf', 'schema:memberOf'
            ],
            'common_prefixes': {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'owl': 'http://www.w3.org/2002/07/owl#',
                'xsd': 'http://www.w3.org/2001/XMLSchema#',
                'foaf': 'http://xmlns.com/foaf/0.1/',
                'dc': 'http://purl.org/dc/elements/1.1/',
                'dcterms': 'http://purl.org/dc/terms/',
                'skos': 'http://www.w3.org/2004/02/skos/core#',
                'schema': 'http://schema.org/',
                'void': 'http://rdfs.org/ns/void#'
            }
        }
        
        # Processing caches
        self._entity_cache = {}
        self._hierarchy_cache = {}
        self._pattern_cache = {}
        
        logger.info("Ontology Processor initialized")
    
    @performance_monitor("ontology_processing")
    def process_ontology_file(self, 
                            file_path: str,
                            format: Optional[str] = None,
                            namespace_filter: Optional[Set[str]] = None) -> OntologyHierarchy:
        """
        Process an ontology file and extract hierarchy
        
        Args:
            file_path: Path to ontology file
            format: File format (auto-detected if None)
            namespace_filter: Only process entities from these namespaces
            
        Returns:
            Complete ontology hierarchy
        """
        
        logger.info(f"Processing ontology file: {file_path}")
        
        with PerformanceProfiler("ontology_file_processing") as profiler:
            
            # Detect format
            if format is None:
                format = self._detect_file_format(file_path)
            
            profiler.checkpoint("format_detected")
            
            # Load raw data based on format
            if format in ['rdf', 'xml', 'owl']:
                raw_data = self._load_rdf_data(file_path, format)
            elif format == 'json-ld':
                raw_data = self._load_jsonld_data(file_path)
            elif format == 'ttl':
                raw_data = self._load_turtle_data(file_path)
            elif format == 'json':
                raw_data = self._load_json_data(file_path)
            else:
                raise ValueError(f"Unsupported ontology format: {format}")
            
            profiler.checkpoint("data_loaded")
            
            # Extract entities and relationships
            hierarchy = self._extract_ontology_hierarchy(raw_data, namespace_filter)
            
            profiler.checkpoint("hierarchy_extracted")
            
            # Build hierarchy relationships
            self._build_hierarchy_graph(hierarchy)
            
            profiler.checkpoint("hierarchy_built")
            
            # Validate and optimize
            if self.settings['strict_validation']:
                self._validate_hierarchy(hierarchy)
            
            profiler.checkpoint("validation_complete")
        
        report = profiler.get_report()
        logger.info(f"Ontology processing completed in {report['total_execution_time']:.2f}s")
        logger.info(f"Extracted: {len(hierarchy.classes)} classes, {len(hierarchy.properties)} properties, {len(hierarchy.instances)} instances")
        
        return hierarchy
    
    def _detect_file_format(self, file_path: str) -> str:
        """Auto-detect ontology file format"""
        
        path = Path(file_path)
        extension = path.suffix.lower()
        
        format_map = {
            '.rdf': 'rdf',
            '.owl': 'owl',
            '.ttl': 'ttl',
            '.n3': 'n3',
            '.jsonld': 'json-ld',
            '.json': 'json',
            '.xml': 'xml'
        }
        
        detected_format = format_map.get(extension, 'rdf')
        
        # Content-based detection for ambiguous extensions
        if extension in ['.xml', '.json']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1KB
                
                if 'owl:Ontology' in content or 'rdf:RDF' in content:
                    detected_format = 'owl'
                elif '@context' in content and 'schema.org' in content:
                    detected_format = 'json-ld'
                elif 'schema.org' in content or '"@type"' in content:
                    detected_format = 'json-ld'
            except Exception as e:
                logger.warning(f"Content detection failed: {e}")
        
        logger.info(f"Detected format: {detected_format}")
        return detected_format
    
    def _load_rdf_data(self, file_path: str, format: str) -> Dict[str, Any]:
        """Load RDF/OWL data using rdflib"""
        
        if not rdflib:
            raise ImportError("rdflib is required for RDF/OWL processing")
        
        graph = rdflib.Graph()
        
        try:
            # Format mapping for rdflib
            rdflib_format = {
                'rdf': 'xml',
                'owl': 'xml',
                'xml': 'xml',
                'ttl': 'turtle',
                'n3': 'n3'
            }.get(format, 'xml')
            
            graph.parse(file_path, format=rdflib_format)
            
            logger.info(f"Loaded RDF graph with {len(graph)} triples")
            
            # Extract structured data
            return self._extract_rdf_structure(graph)
            
        except Exception as e:
            logger.error(f"Failed to load RDF data: {str(e)}")
            raise
    
    def _extract_rdf_structure(self, graph) -> Dict[str, Any]:
        """Extract structured data from RDF graph"""
        
        structure = {
            'triples': [],
            'namespaces': {},
            'classes': set(),
            'properties': set(),
            'instances': set()
        }
        
        # Extract namespaces
        for prefix, namespace in graph.namespaces():
            structure['namespaces'][str(prefix)] = str(namespace)
        
        # Extract all triples
        for subj, pred, obj in graph:
            triple = {
                'subject': str(subj),
                'predicate': str(pred),
                'object': str(obj),
                'object_type': 'uri' if isinstance(obj, rdflib.URIRef) else 'literal'
            }
            structure['triples'].append(triple)
            
            # Classify entities
            pred_str = str(pred)
            if 'type' in pred_str.lower() or pred_str.endswith('#type'):
                obj_str = str(obj)
                if any(indicator in obj_str for indicator in self.patterns['class_indicators']):
                    structure['classes'].add(str(subj))
                elif any(indicator in obj_str for indicator in self.patterns['property_indicators']):
                    structure['properties'].add(str(subj))
                else:
                    structure['instances'].add(str(subj))
        
        return structure
    
    def _load_jsonld_data(self, file_path: str) -> Dict[str, Any]:
        """Load JSON-LD data"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            if isinstance(data, list):
                entities = data
            elif isinstance(data, dict):
                if '@graph' in data:
                    entities = data['@graph']
                else:
                    entities = [data]
            else:
                entities = []
            
            return {
                'entities': entities,
                'context': data.get('@context', {}) if isinstance(data, dict) else {},
                'format': 'json-ld'
            }
            
        except Exception as e:
            logger.error(f"Failed to load JSON-LD data: {str(e)}")
            raise
    
    def _load_turtle_data(self, file_path: str) -> Dict[str, Any]:
        """Load Turtle (TTL) data"""
        return self._load_rdf_data(file_path, 'ttl')
    
    def _load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load plain JSON data (Schema.org format)"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'data': data,
                'format': 'json'
            }
            
        except Exception as e:
            logger.error(f"Failed to load JSON data: {str(e)}")
            raise
    
    def _extract_ontology_hierarchy(self, 
                                   raw_data: Dict[str, Any],
                                   namespace_filter: Optional[Set[str]] = None) -> OntologyHierarchy:
        """Extract ontology hierarchy from raw data"""
        
        hierarchy = OntologyHierarchy()
        
        if 'triples' in raw_data:
            # RDF-based data
            self._process_rdf_triples(raw_data, hierarchy, namespace_filter)
        elif 'entities' in raw_data:
            # JSON-LD data
            self._process_jsonld_entities(raw_data, hierarchy, namespace_filter)
        elif 'data' in raw_data:
            # Plain JSON data
            self._process_json_data(raw_data, hierarchy, namespace_filter)
        
        # Extract namespaces
        hierarchy.namespaces = raw_data.get('namespaces', {})
        
        return hierarchy
    
    def _process_rdf_triples(self, 
                           raw_data: Dict[str, Any], 
                           hierarchy: OntologyHierarchy,
                           namespace_filter: Optional[Set[str]]) -> None:
        """Process RDF triples to build hierarchy"""
        
        triples = raw_data['triples']
        
        if self.settings['enable_polars_optimization'] and len(triples) > 1000:
            # Use Polars for large datasets
            self._process_triples_with_polars(triples, hierarchy, namespace_filter)
        else:
            # Process triples directly
            for triple in triples:
                self._process_single_triple(triple, hierarchy, namespace_filter)
    
    def _process_triples_with_polars(self, 
                                   triples: List[Dict[str, Any]], 
                                   hierarchy: OntologyHierarchy,
                                   namespace_filter: Optional[Set[str]]) -> None:
        """Process triples using Polars for performance"""
        
        logger.info("Using Polars optimization for large ontology processing")
        
        # Create DataFrame
        df = pl.DataFrame(triples)
        
        # Filter by namespace if specified
        if namespace_filter:
            namespace_patterns = '|'.join(namespace_filter)
            df = df.filter(
                pl.col('subject').str.contains(namespace_patterns) |
                pl.col('object').str.contains(namespace_patterns)
            )
        
        # Extract classes
        class_triples = df.filter(
            pl.col('predicate').str.contains('type|subClassOf|equivalentClass')
        )
        
        for row in class_triples.iter_rows(named=True):
            if self._is_class_triple(row):
                entity = self._create_entity_from_triple(row, 'class')
                hierarchy.classes[entity.uri] = entity
        
        # Extract properties
        prop_triples = df.filter(
            pl.col('predicate').str.contains('Property|domain|range')
        )
        
        for row in prop_triples.iter_rows(named=True):
            if self._is_property_triple(row):
                entity = self._create_entity_from_triple(row, 'property')
                hierarchy.properties[entity.uri] = entity
        
        # Extract hierarchy relationships
        hierarchy_triples = df.filter(
            pl.col('predicate').str.contains('subClassOf|subPropertyOf|equivalentClass')
        )
        
        for row in hierarchy_triples.iter_rows(named=True):
            self._add_hierarchy_relationship(row, hierarchy)
    
    def _process_single_triple(self, 
                             triple: Dict[str, Any], 
                             hierarchy: OntologyHierarchy,
                             namespace_filter: Optional[Set[str]]) -> None:
        """Process a single RDF triple"""
        
        subject = triple['subject']
        predicate = triple['predicate']
        obj = triple['object']
        
        # Apply namespace filter
        if namespace_filter:
            subject_ns = self._extract_namespace_from_uri(subject)
            object_ns = self._extract_namespace_from_uri(obj)
            
            if subject_ns not in namespace_filter and object_ns not in namespace_filter:
                return
        
        # Identify entity types based on predicate
        if self._is_class_definition(predicate, obj):
            entity = self._create_ontology_entity(subject, 'class', triple)
            hierarchy.classes[subject] = entity
        
        elif self._is_property_definition(predicate, obj):
            entity = self._create_ontology_entity(subject, 'property', triple)
            hierarchy.properties[subject] = entity
        
        elif self._is_hierarchy_relation(predicate):
            self._add_hierarchy_relationship(triple, hierarchy)
        
        elif self._is_instance_relation(predicate):
            entity = self._create_ontology_entity(subject, 'instance', triple)
            hierarchy.instances[subject] = entity
    
    def _process_jsonld_entities(self, 
                               raw_data: Dict[str, Any], 
                               hierarchy: OntologyHierarchy,
                               namespace_filter: Optional[Set[str]]) -> None:
        """Process JSON-LD entities"""
        
        entities = raw_data['entities']
        context = raw_data.get('context', {})
        
        for entity_data in entities:
            if not isinstance(entity_data, dict):
                continue
            
            entity_id = entity_data.get('@id') or entity_data.get('id', '')
            entity_type = entity_data.get('@type') or entity_data.get('type', '')
            
            # Apply namespace filter
            if namespace_filter:
                entity_ns = self._extract_namespace_from_uri(entity_id)
                if entity_ns not in namespace_filter:
                    continue
            
            # Create ontology entity
            if self._is_class_type(entity_type):
                entity = self._create_entity_from_jsonld(entity_data, 'class', context)
                hierarchy.classes[entity.uri] = entity
            
            elif self._is_property_type(entity_type):
                entity = self._create_entity_from_jsonld(entity_data, 'property', context)
                hierarchy.properties[entity.uri] = entity
            
            else:
                entity = self._create_entity_from_jsonld(entity_data, 'instance', context)
                hierarchy.instances[entity.uri] = entity
            
            # Extract relationships
            self._extract_jsonld_relationships(entity_data, hierarchy, context)
    
    def _process_json_data(self, 
                          raw_data: Dict[str, Any], 
                          hierarchy: OntologyHierarchy,
                          namespace_filter: Optional[Set[str]]) -> None:
        """Process plain JSON data (Schema.org style)"""
        
        data = raw_data['data']
        
        if isinstance(data, dict):
            # Single entity or structured schema
            if 'types' in data or 'classes' in data:
                # Structured schema format
                self._process_structured_schema(data, hierarchy, namespace_filter)
            else:
                # Single entity
                self._process_json_entity(data, hierarchy, namespace_filter)
        
        elif isinstance(data, list):
            # List of entities
            for entity_data in data:
                self._process_json_entity(entity_data, hierarchy, namespace_filter)
    
    def _process_structured_schema(self, 
                                  data: Dict[str, Any], 
                                  hierarchy: OntologyHierarchy,
                                  namespace_filter: Optional[Set[str]]) -> None:
        """Process structured schema format"""
        
        # Process types/classes
        types_data = data.get('types', data.get('classes', {}))
        for type_name, type_info in types_data.items():
            entity = OntologyEntity(
                uri=type_info.get('id', f"schema:{type_name}"),
                name=type_name,
                entity_type='class',
                namespace='schema',
                labels=[type_info.get('label', type_name)],
                comments=[type_info.get('comment', '')],
                properties=type_info
            )
            hierarchy.classes[entity.uri] = entity
        
        # Process properties
        props_data = data.get('properties', {})
        for prop_name, prop_info in props_data.items():
            entity = OntologyEntity(
                uri=prop_info.get('id', f"schema:{prop_name}"),
                name=prop_name,
                entity_type='property',
                namespace='schema',
                labels=[prop_info.get('label', prop_name)],
                comments=[prop_info.get('comment', '')],
                properties=prop_info
            )
            hierarchy.properties[entity.uri] = entity
    
    def _process_json_entity(self, 
                           entity_data: Dict[str, Any], 
                           hierarchy: OntologyHierarchy,
                           namespace_filter: Optional[Set[str]]) -> None:
        """Process single JSON entity"""
        
        entity_type = entity_data.get('type', entity_data.get('@type', 'unknown'))
        entity_id = entity_data.get('id', entity_data.get('@id', ''))
        
        # Apply namespace filter
        if namespace_filter:
            entity_ns = self._extract_namespace_from_uri(entity_id)
            if entity_ns not in namespace_filter:
                return
        
        # Determine entity category
        if self._is_class_type(entity_type):
            category = 'class'
            target_dict = hierarchy.classes
        elif self._is_property_type(entity_type):
            category = 'property'
            target_dict = hierarchy.properties
        else:
            category = 'instance'
            target_dict = hierarchy.instances
        
        entity = OntologyEntity(
            uri=entity_id,
            name=entity_data.get('name', entity_data.get('label', '')),
            entity_type=category,
            namespace=self._extract_namespace_from_uri(entity_id),
            labels=self._extract_labels(entity_data),
            comments=self._extract_comments(entity_data),
            properties=entity_data
        )
        
        target_dict[entity.uri] = entity
    
    def _build_hierarchy_graph(self, hierarchy: OntologyHierarchy) -> None:
        """Build hierarchy graph from relationships"""
        
        # Initialize graphs
        hierarchy.hierarchy_graph = defaultdict(list)
        hierarchy.reverse_hierarchy = defaultdict(list)
        
        # Process explicit hierarchy relationships from triples/entities
        for entity in list(hierarchy.classes.values()) + list(hierarchy.properties.values()):
            for predicate, target in entity.relationships:
                if self._is_hierarchy_predicate(predicate):
                    # target is parent of entity
                    hierarchy.hierarchy_graph[target].append(entity.uri)
                    hierarchy.reverse_hierarchy[entity.uri].append(target)
        
        # Infer additional relationships using pattern matching
        if self.settings['enable_inference']:
            self._infer_hierarchy_relationships(hierarchy)
    
    def _infer_hierarchy_relationships(self, hierarchy: OntologyHierarchy) -> None:
        """Infer additional hierarchy relationships"""
        
        logger.info("Inferring additional hierarchy relationships")
        
        # Name-based inference for classes
        class_names = list(hierarchy.classes.keys())
        
        for i, class1_uri in enumerate(class_names):
            class1 = hierarchy.classes[class1_uri]
            
            for j, class2_uri in enumerate(class_names[i+1:], i+1):
                class2 = hierarchy.classes[class2_uri]
                
                # Check if one is a specialization of another based on naming
                if self._is_specialization_by_name(class1.name, class2.name):
                    # class1 is more specific than class2
                    hierarchy.hierarchy_graph[class2_uri].append(class1_uri)
                    hierarchy.reverse_hierarchy[class1_uri].append(class2_uri)
                
                elif self._is_specialization_by_name(class2.name, class1.name):
                    # class2 is more specific than class1
                    hierarchy.hierarchy_graph[class1_uri].append(class2_uri)
                    hierarchy.reverse_hierarchy[class2_uri].append(class1_uri)
    
    def _is_specialization_by_name(self, specific_name: str, general_name: str) -> bool:
        """Check if one name represents a specialization of another"""
        
        specific_lower = specific_name.lower()
        general_lower = general_name.lower()
        
        # Specific name contains general name
        if general_lower in specific_lower and specific_lower != general_lower:
            return True
        
        # Common patterns: SpecificType -> Type, SubCategory -> Category
        if (specific_lower.startswith(general_lower) or 
            specific_lower.endswith(general_lower)):
            return True
        
        # Word-based analysis
        specific_words = set(re.findall(r'\b\w+\b', specific_lower))
        general_words = set(re.findall(r'\b\w+\b', general_lower))
        
        # General words are subset of specific words
        if general_words and general_words.issubset(specific_words) and len(specific_words) > len(general_words):
            return True
        
        return False
    
    def _validate_hierarchy(self, hierarchy: OntologyHierarchy) -> None:
        """Validate hierarchy consistency"""
        
        logger.info("Validating hierarchy consistency")
        
        # Check for cycles
        cycles = self._detect_cycles(hierarchy.hierarchy_graph)
        if cycles:
            logger.warning(f"Detected {len(cycles)} cycles in hierarchy")
            for cycle in cycles[:5]:  # Report first 5 cycles
                logger.warning(f"Cycle detected: {' -> '.join(cycle)}")
        
        # Check depth limits
        max_depth = self._calculate_max_depth(hierarchy.hierarchy_graph)
        if max_depth > self.settings['max_hierarchy_depth']:
            logger.warning(f"Hierarchy depth ({max_depth}) exceeds limit ({self.settings['max_hierarchy_depth']})")
        
        # Update metadata
        hierarchy.metadata.update({
            'validation_timestamp': time.time(),
            'cycles_detected': len(cycles),
            'max_depth': max_depth,
            'validation_passed': len(cycles) == 0 and max_depth <= self.settings['max_hierarchy_depth']
        })
    
    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect cycles in hierarchy graph"""
        
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for child in graph.get(node, []):
                dfs(child, path + [node])
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _calculate_max_depth(self, graph: Dict[str, List[str]]) -> int:
        """Calculate maximum depth of hierarchy"""
        
        def get_depth(node: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if node in visited:
                return 0
            
            visited.add(node)
            
            children = graph.get(node, [])
            if not children:
                return 1
            
            max_child_depth = 0
            for child in children:
                child_depth = get_depth(child, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return 1 + max_child_depth
        
        # Find root nodes (nodes with no parents)
        all_children = set()
        for children in graph.values():
            all_children.update(children)
        
        roots = [node for node in graph.keys() if node not in all_children]
        
        if not roots:
            # No clear roots, check all nodes
            roots = list(graph.keys())
        
        max_depth = 0
        for root in roots:
            depth = get_depth(root)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    # Helper methods for entity classification
    
    def _is_class_triple(self, triple: Dict[str, Any]) -> bool:
        """Check if triple defines a class"""
        predicate = triple['predicate'].lower()
        obj = triple['object'].lower()
        
        return (any(indicator in predicate for indicator in ['type', 'subclassof']) and
                any(indicator in obj for indicator in ['class', 'type', 'concept']))
    
    def _is_property_triple(self, triple: Dict[str, Any]) -> bool:
        """Check if triple defines a property"""
        predicate = triple['predicate'].lower()
        obj = triple['object'].lower()
        
        return (any(indicator in predicate for indicator in ['type', 'domain', 'range']) and
                any(indicator in obj for indicator in ['property', 'relation']))
    
    def _is_class_definition(self, predicate: str, obj: str) -> bool:
        """Check if this is a class definition"""
        return any(indicator in obj for indicator in self.patterns['class_indicators'])
    
    def _is_property_definition(self, predicate: str, obj: str) -> bool:
        """Check if this is a property definition"""
        return any(indicator in obj for indicator in self.patterns['property_indicators'])
    
    def _is_hierarchy_relation(self, predicate: str) -> bool:
        """Check if predicate represents hierarchy relation"""
        return any(indicator in predicate for indicator in self.patterns['hierarchy_relations'])
    
    def _is_instance_relation(self, predicate: str) -> bool:
        """Check if predicate represents instance relation"""
        return any(indicator in predicate for indicator in self.patterns['instance_relations'])
    
    def _is_class_type(self, entity_type: str) -> bool:
        """Check if entity type represents a class"""
        return any(indicator in entity_type for indicator in self.patterns['class_indicators'])
    
    def _is_property_type(self, entity_type: str) -> bool:
        """Check if entity type represents a property"""
        return any(indicator in entity_type for indicator in self.patterns['property_indicators'])
    
    def _is_hierarchy_predicate(self, predicate: str) -> bool:
        """Check if predicate represents hierarchy"""
        return any(indicator in predicate.lower() for indicator in ['subclass', 'subproperty', 'parent', 'child'])
    
    # Entity creation helpers
    
    def _create_ontology_entity(self, uri: str, entity_type: str, triple: Dict[str, Any]) -> OntologyEntity:
        """Create ontology entity from triple"""
        
        return OntologyEntity(
            uri=uri,
            name=self._extract_name_from_uri(uri),
            entity_type=entity_type,
            namespace=self._extract_namespace_from_uri(uri),
            properties={'source_triple': triple}
        )
    
    def _create_entity_from_triple(self, triple: Dict[str, Any], entity_type: str) -> OntologyEntity:
        """Create entity from triple data"""
        
        subject = triple['subject']
        
        return OntologyEntity(
            uri=subject,
            name=self._extract_name_from_uri(subject),
            entity_type=entity_type,
            namespace=self._extract_namespace_from_uri(subject),
            properties=triple
        )
    
    def _create_entity_from_jsonld(self, data: Dict[str, Any], entity_type: str, context: Dict[str, Any]) -> OntologyEntity:
        """Create entity from JSON-LD data"""
        
        entity_id = data.get('@id', data.get('id', ''))
        
        return OntologyEntity(
            uri=entity_id,
            name=data.get('name', data.get('label', self._extract_name_from_uri(entity_id))),
            entity_type=entity_type,
            namespace=self._extract_namespace_from_uri(entity_id),
            labels=self._extract_labels(data),
            comments=self._extract_comments(data),
            properties=data
        )
    
    def _add_hierarchy_relationship(self, triple: Dict[str, Any], hierarchy: OntologyHierarchy) -> None:
        """Add hierarchy relationship to an entity"""
        
        subject = triple['subject']
        predicate = triple['predicate']
        obj = triple['object']
        
        # Find the entity and add relationship
        entity = None
        if subject in hierarchy.classes:
            entity = hierarchy.classes[subject]
        elif subject in hierarchy.properties:
            entity = hierarchy.properties[subject]
        elif subject in hierarchy.instances:
            entity = hierarchy.instances[subject]
        
        if entity:
            entity.relationships.append((predicate, obj))
    
    def _extract_jsonld_relationships(self, data: Dict[str, Any], hierarchy: OntologyHierarchy, context: Dict[str, Any]) -> None:
        """Extract relationships from JSON-LD data"""
        
        entity_id = data.get('@id', data.get('id', ''))
        
        # Look for hierarchy relationships
        for key, value in data.items():
            if key.startswith('@'):
                continue
            
            if self._is_hierarchy_predicate(key):
                # Add relationship to appropriate entity
                if entity_id in hierarchy.classes:
                    hierarchy.classes[entity_id].relationships.append((key, str(value)))
                elif entity_id in hierarchy.properties:
                    hierarchy.properties[entity_id].relationships.append((key, str(value)))
                elif entity_id in hierarchy.instances:
                    hierarchy.instances[entity_id].relationships.append((key, str(value)))
    
    # Utility methods
    
    def _extract_namespace_from_uri(self, uri: str) -> str:
        """Extract namespace from URI"""
        
        if not uri:
            return 'unknown'
        
        # Check for known prefixes
        for prefix, namespace_uri in self.patterns['common_prefixes'].items():
            if namespace_uri in uri:
                return prefix
        
        # Parse URI structure
        try:
            parsed = urlparse(uri)
            if parsed.netloc:
                return parsed.netloc.replace('www.', '')
        except Exception:
            pass
        
        # Fallback patterns
        if '#' in uri:
            return uri.split('#')[0].split('/')[-1] or 'unknown'
        elif '/' in uri:
            return uri.split('/')[-2] if uri.endswith('/') else uri.split('/')[-1].split('.')[0]
        elif ':' in uri and not uri.startswith('http'):
            return uri.split(':')[0]
        
        return 'local'
    
    def _extract_name_from_uri(self, uri: str) -> str:
        """Extract readable name from URI"""
        
        if not uri:
            return 'unnamed'
        
        # Extract from fragment or path
        if '#' in uri:
            name = uri.split('#')[-1]
        elif '/' in uri:
            name = uri.split('/')[-1]
        else:
            name = uri
        
        # Clean up the name
        name = name.replace('_', ' ').replace('-', ' ')
        
        return name or 'unnamed'
    
    def _extract_labels(self, data: Dict[str, Any]) -> List[str]:
        """Extract labels from entity data"""
        
        labels = []
        
        # Common label fields
        label_fields = ['label', 'name', 'title', 'rdfs:label', '@label']
        
        for field in label_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    labels.extend(str(v) for v in value)
                else:
                    labels.append(str(value))
        
        return labels
    
    def _extract_comments(self, data: Dict[str, Any]) -> List[str]:
        """Extract comments from entity data"""
        
        comments = []
        
        # Common comment fields
        comment_fields = ['comment', 'description', 'note', 'rdfs:comment', '@comment']
        
        for field in comment_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    comments.extend(str(v) for v in value)
                else:
                    comments.append(str(value))
        
        return comments
    
    def export_to_knowledge_graph(self, hierarchy: OntologyHierarchy) -> Dict[str, Any]:
        """
        Export ontology hierarchy to knowledge graph format
        
        Args:
            hierarchy: Processed ontology hierarchy
            
        Returns:
            Knowledge graph compatible structure
        """
        
        logger.info("Exporting ontology to knowledge graph format")
        
        kg_data = {
            'entities': {},
            'relationships': [],
            'hierarchy_levels': {},
            'metadata': hierarchy.metadata
        }
        
        # Export entities
        all_entities = {}
        all_entities.update(hierarchy.classes)
        all_entities.update(hierarchy.properties)
        all_entities.update(hierarchy.instances)
        
        for uri, entity in all_entities.items():
            kg_data['entities'][uri] = {
                'id': uri,
                'name': entity.name,
                'type': entity.entity_type,
                'namespace': entity.namespace,
                'labels': entity.labels,
                'comments': entity.comments,
                'properties': entity.properties
            }
        
        # Export relationships
        relationship_id = 0
        for parent, children in hierarchy.hierarchy_graph.items():
            for child in children:
                kg_data['relationships'].append({
                    'id': f"rel_{relationship_id}",
                    'source': parent,
                    'target': child,
                    'type': 'subClassOf',
                    'confidence': 1.0
                })
                relationship_id += 1
        
        # Create hierarchy levels based on depth
        levels = self._create_hierarchy_levels(hierarchy.hierarchy_graph)
        kg_data['hierarchy_levels'] = levels
        
        logger.info(f"Exported {len(all_entities)} entities and {len(kg_data['relationships'])} relationships")
        
        return kg_data
    
    def _create_hierarchy_levels(self, hierarchy_graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create hierarchy levels based on graph structure"""
        
        levels = defaultdict(list)
        
        # Calculate depth for each entity
        def get_depth(node: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if node in visited:
                return 0
            
            visited.add(node)
            
            # Find all parents
            parents = []
            for parent, children in hierarchy_graph.items():
                if node in children:
                    parents.append(parent)
            
            if not parents:
                return 0  # Root level
            
            max_parent_depth = 0
            for parent in parents:
                parent_depth = get_depth(parent, visited.copy())
                max_parent_depth = max(max_parent_depth, parent_depth)
            
            return max_parent_depth + 1
        
        # Assign entities to levels
        all_entities = set(hierarchy_graph.keys())
        for children in hierarchy_graph.values():
            all_entities.update(children)
        
        for entity in all_entities:
            depth = get_depth(entity)
            levels[f"level_{depth}"].append(entity)
        
        return dict(levels)
    
    # ============================================================================
    # Public API Methods for Test Compatibility
    # ============================================================================
    
    def build_class_hierarchy(self) -> Dict[str, Any]:
        """
        Build class hierarchy from loaded ontology data
        
        Returns:
            Dict containing class hierarchy information
        """
        try:
            if not hasattr(self, 'ontology_data') or not self.ontology_data:
                logger.warning("No ontology data loaded, creating basic hierarchy from KG")
                return self._build_hierarchy_from_kg()
            
            hierarchy = OntologyHierarchy()
            
            # Extract classes from ontology data
            for entity_uri, entity_data in self.ontology_data.items():
                if entity_data.get('entity_type') == 'class':
                    entity = OntologyEntity(
                        uri=entity_uri,
                        name=entity_data.get('name', entity_uri.split(':')[-1]),
                        entity_type='class',
                        namespace=entity_data.get('namespace', ''),
                        labels=entity_data.get('labels', []),
                        comments=entity_data.get('comments', [])
                    )
                    hierarchy.classes[entity_uri] = entity
            
            # Build hierarchy relationships
            hierarchy_graph = {}
            for entity_uri, entity_data in self.ontology_data.items():
                if 'subclass_of' in entity_data:
                    parent = entity_data['subclass_of']
                    if parent not in hierarchy_graph:
                        hierarchy_graph[parent] = []
                    hierarchy_graph[parent].append(entity_uri)
            
            hierarchy.hierarchy_graph = hierarchy_graph
            
            return {
                'classes': hierarchy.classes,
                'subclasses': {child: parent for parent, children in hierarchy_graph.items() for child in children},
                'hierarchy_graph': hierarchy_graph,
                'total_classes': len(hierarchy.classes)
            }
            
        except Exception as e:
            logger.error(f"Class hierarchy building failed: {e}")
            return self._build_hierarchy_from_kg()
    
    def _build_hierarchy_from_kg(self) -> Dict[str, Any]:
        """Build basic hierarchy from knowledge graph structure"""
        
        classes = {}
        subclasses = {}
        
        # Check if KG is available
        if not self.kg:
            return {
                'classes': classes,
                'subclasses': subclasses,
                'hierarchy_graph': {},
                'total_classes': 0
            }
        
        # Extract entity types from KG as classes
        for node_id in self.kg.nodes:
            node_type = self.kg.get_node_type(node_id)
            if node_type and node_type not in classes:
                classes[node_type] = {
                    'uri': f'kg:{node_type}',
                    'name': node_type,
                    'entity_type': 'class',
                    'instances': []
                }
        
        # Look for hierarchical relationships in edge types
        for edge in self.kg.edges:
            edge_type = self.kg.get_edge_type(edge)
            if edge_type and 'subclass' in edge_type.lower():
                # This would be a subclass relationship
                edge_nodes = list(edge)
                if len(edge_nodes) >= 2:
                    subclasses[edge_nodes[0]] = edge_nodes[1]
        
        return {
            'classes': classes,
            'subclasses': subclasses,
            'hierarchy_graph': {},
            'total_classes': len(classes)
        }
    
    def get_schema_org_compatibility(self) -> Dict[str, Any]:
        """
        Check compatibility with Schema.org vocabulary
        
        Returns:
            Dict containing compatibility information
        """
        try:
            schema_org_terms = {
                'Thing', 'Person', 'Organization', 'Place', 'Event', 'CreativeWork',
                'Product', 'Offer', 'Review', 'Rating', 'ContactPoint', 'PostalAddress',
                'name', 'description', 'url', 'image', 'sameAs', 'identifier'
            }
            
            found_terms = set()
            total_entities = 0
            
            # Check ontology data for Schema.org terms
            if hasattr(self, 'ontology_data') and self.ontology_data:
                for entity_uri, entity_data in self.ontology_data.items():
                    total_entities += 1
                    entity_name = entity_data.get('name', entity_uri.split(':')[-1])
                    
                    if entity_name in schema_org_terms:
                        found_terms.add(entity_name)
                    
                    # Check for schema.org namespace
                    if 'schema.org' in entity_uri or entity_uri.startswith('schema:'):
                        found_terms.add(entity_name)
            
            # Check KG for Schema.org-like patterns (if available)
            if self.kg:
                for node_id in self.kg.nodes:
                    total_entities += 1
                    node_type = self.kg.get_node_type(node_id)
                    if node_type and node_type in schema_org_terms:
                        found_terms.add(node_type)
                        
                    node_data = self.kg.properties.get_node_data(node_id) or {}
                    for prop_name in node_data.keys():
                        if prop_name in schema_org_terms:
                            found_terms.add(prop_name)
            
            compatibility_score = len(found_terms) / len(schema_org_terms) if schema_org_terms else 0
            
            return {
                'is_compatible': compatibility_score > 0.1,  # At least 10% overlap
                'compatibility_score': compatibility_score,
                'found_schema_org_terms': list(found_terms),
                'total_schema_org_terms': len(schema_org_terms),
                'total_entities_checked': total_entities,
                'missing_core_terms': list(schema_org_terms - found_terms)
            }
            
        except Exception as e:
            logger.error(f"Schema.org compatibility check failed: {e}")
            return {
                'is_compatible': False,
                'compatibility_score': 0.0,
                'error': str(e)
            }


# Test Schema.org processing capabilities
def test_schemaorg_processing():
    """Test function to demonstrate Schema.org ontology processing"""
    
    logger.info("Testing Schema.org ontology processing")
    
    # Create sample Schema.org data
    schemaorg_data = {
        "@context": "http://schema.org/",
        "@graph": [
            {
                "@id": "schema:Thing",
                "@type": "rdfs:Class",
                "rdfs:label": "Thing",
                "rdfs:comment": "The most generic type of item."
            },
            {
                "@id": "schema:Person",
                "@type": "rdfs:Class",
                "rdfs:label": "Person",
                "rdfs:comment": "A person (alive, dead, undead, or fictional).",
                "rdfs:subClassOf": "schema:Thing"
            },
            {
                "@id": "schema:Organization",
                "@type": "rdfs:Class",
                "rdfs:label": "Organization",
                "rdfs:comment": "An organization such as a school, NGO, corporation, club, etc.",
                "rdfs:subClassOf": "schema:Thing"
            },
            {
                "@id": "schema:name",
                "@type": "rdf:Property",
                "rdfs:label": "name",
                "rdfs:comment": "The name of the item.",
                "schema:domainIncludes": "schema:Thing"
            }
        ]
    }
    
    # Save test data
    test_file = "/tmp/test_schemaorg.jsonld"
    with open(test_file, 'w') as f:
        json.dump(schemaorg_data, f, indent=2)
    
    # Process ontology
    processor = OntologyProcessor()
    hierarchy = processor.process_ontology_file(test_file, format='json-ld')
    
    # Export results
    kg_data = processor.export_to_knowledge_graph(hierarchy)
    
    logger.info(f"Processing complete: {len(hierarchy.classes)} classes, {len(hierarchy.properties)} properties")
    
    return hierarchy, kg_data


if __name__ == "__main__":
    # Run test
    test_hierarchy, test_kg_data = test_schemaorg_processing()
    print("Schema.org ontology processing test completed successfully!")