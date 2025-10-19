"""
Ontology Analysis Module
=======================

Advanced ontology processing, schema extraction, and semantic analysis
for knowledge graphs, with special support for FIBO-style ontologies.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import json

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies for advanced ontology processing
rdflib = safe_import('rdflib')
owlready2 = safe_import('owlready2')

logger = logging.getLogger(__name__)


@dataclass 
class OntologyClass:
    """Represents an ontology class with metadata"""
    uri: str
    name: str
    namespace: str
    parent_classes: List[str]
    subclasses: List[str]
    properties: List[str]
    instances_count: int = 0
    description: Optional[str] = None


@dataclass
class OntologyProperty:
    """Represents an ontology property"""
    uri: str
    name: str
    namespace: str
    domain: List[str]
    range: List[str]
    property_type: str  # 'object', 'data', 'annotation'
    usage_count: int = 0
    description: Optional[str] = None


@dataclass
class OntologyStats:
    """Ontology statistics and metrics"""
    total_classes: int
    total_properties: int
    total_instances: int
    max_depth: int
    avg_children_per_class: float
    namespace_distribution: Dict[str, int]
    complexity_metrics: Dict[str, Any]


class OntologyAnalyzer:
    """
    Comprehensive ontology analysis and schema extraction
    
    Features:
    - Class hierarchy analysis
    - Property usage patterns
    - Schema complexity metrics
    - FIBO ontology support
    - Semantic validation
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize ontology analyzer
        
        Args:
            knowledge_graph: KnowledgeGraph instance to analyze
        """
        self.kg = knowledge_graph
        
        # Analysis caches
        self._class_hierarchy = None
        self._property_analysis = None
        self._ontology_stats = None
        
        # Generic ontology patterns (domain-agnostic)
        self.ontology_patterns = {
            'class_indicators': ['ontology/', 'Class', 'Type', 'Kind', 'Concept', 'Category'],
            'property_indicators': ['has', 'is', 'relates', 'connects', 'contains', 'includes', 'part'],
            'hierarchy_indicators': ['subClassOf', 'subclass', 'extends', 'inherits', 'isa', 'typeof'],
            'common_namespaces': {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'owl': 'http://www.w3.org/2002/07/owl#',
                'skos': 'http://www.w3.org/2004/02/skos/core#',
                'foaf': 'http://xmlns.com/foaf/0.1/',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }
        }
        
        logger.info("Ontology Analyzer initialized")
    
    @performance_monitor("ontology_class_analysis")
    def analyze_class_hierarchy(self) -> Dict[str, OntologyClass]:
        """
        Analyze the class hierarchy in the knowledge graph
        
        Returns:
            Dictionary mapping class URIs to OntologyClass objects
        """
        
        if self._class_hierarchy is not None:
            return self._class_hierarchy
        
        logger.info("Analyzing class hierarchy...")
        
        with PerformanceProfiler("class_hierarchy_analysis") as profiler:
            
            profiler.checkpoint("start_analysis")
            
            classes = {}
            
            # Identify ontology classes
            for entity_type, entities in self.kg._entity_types.items():
                if self._is_ontology_class(entity_type):
                    
                    # Extract class information
                    class_uri = self._get_representative_entity(entities)
                    namespace = self._extract_namespace(class_uri)
                    
                    ontology_class = OntologyClass(
                        uri=class_uri,
                        name=entity_type,
                        namespace=namespace,
                        parent_classes=[],
                        subclasses=[],
                        properties=[],
                        instances_count=len(entities)
                    )
                    
                    classes[class_uri] = ontology_class
            
            profiler.checkpoint("classes_identified")
            
            # Build hierarchy relationships
            self._build_class_relationships(classes)
            
            profiler.checkpoint("hierarchy_built")
            
            # Analyze properties for each class
            self._analyze_class_properties(classes)
            
            profiler.checkpoint("properties_analyzed")
        
        self._class_hierarchy = classes
        
        report = profiler.get_report()
        logger.info(f"Class hierarchy analysis completed in {report['total_execution_time']:.2f}s")
        logger.info(f"Found {len(classes)} ontology classes")
        
        return classes
    
    def _is_ontology_class(self, entity_type: str) -> bool:
        """Determine if an entity type represents an ontology class"""
        
        # Check for generic ontology class patterns
        for pattern in self.ontology_patterns['class_indicators']:
            if pattern.lower() in entity_type.lower():
                return True
        
        # Check URI structure (common in semantic web ontologies)
        if 'ontology/' in entity_type or '#' in entity_type or '/owl#' in entity_type:
            return True
        
        # Check for camelCase or PascalCase naming (common in ontology classes)
        if entity_type and entity_type[0].isupper() and any(c.isupper() for c in entity_type[1:]):
            return True
        
        # Simple heuristic: entities that are single words and capitalized are likely classes
        if entity_type and entity_type[0].isupper() and entity_type.isalpha():
            return True
        
        # Check for common class patterns like "Person", "Company", "Organization"
        common_class_terms = ['person', 'company', 'organization', 'agent', 'employee', 'manager']
        if any(term in entity_type.lower() for term in common_class_terms):
            return True
        
        return False
    
    def _get_representative_entity(self, entities: Set[str]) -> str:
        """Get a representative entity URI from a set"""
        # Prefer the shortest URI or the one that looks most like a class definition
        entities_list = list(entities)
        
        # Sort by length and URI structure preferences
        entities_list.sort(key=lambda x: (
            len(x),
            0 if 'ontology/' in x else 1,
            0 if x.endswith('Class') or x.endswith('Type') else 1
        ))
        
        return entities_list[0]
    
    def _extract_namespace(self, uri: str) -> str:
        """Extract namespace from URI"""
        
        # Check for known standard namespaces
        for prefix, namespace_uri in self.ontology_patterns['common_namespaces'].items():
            if namespace_uri in uri:
                return prefix
        
        # Generic namespace extraction from URI structure
        if 'ontology/' in uri:
            parts = uri.split('ontology/')
            if len(parts) > 1:
                namespace_part = parts[0].split('/')[-1] or parts[0].split('/')[-2]
                return namespace_part
        
        # Extract from hash fragment
        if '#' in uri:
            return uri.split('#')[0].split('/')[-1] or 'unknown'
        
        # Use domain as namespace
        if uri.startswith('http'):
            from urllib.parse import urlparse
            parsed = urlparse(uri)
            return parsed.netloc.replace('www.', '')
        
        # Handle local or prefixed names
        if ':' in uri and not uri.startswith('http'):
            return uri.split(':')[0]
        
        return 'local'
    
    def _build_class_relationships(self, classes: Dict[str, OntologyClass]):
        """Build parent-child relationships between classes"""
        
        # For each class, find potential parent/child relationships
        for class_uri, ontology_class in classes.items():
            
            # Look for hierarchical relationships in the knowledge graph
            # This is domain-specific and would need to be adapted based on 
            # how hierarchies are represented in the specific ontology
            
            parent_candidates = self._find_parent_classes(class_uri, classes)
            child_candidates = self._find_child_classes(class_uri, classes)
            
            ontology_class.parent_classes = parent_candidates
            ontology_class.subclasses = child_candidates
    
    def _find_parent_classes(self, class_uri: str, all_classes: Dict[str, OntologyClass]) -> List[str]:
        """Find parent classes for a given class"""
        parents = []
        
        # Strategy 1: Look for rdfs:subClassOf or similar relationships
        incident_edges = self.kg.incidences.get_node_edges(class_uri)
        
        for edge in incident_edges:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge)
            
            # Look for subclass relationships
            for node in edge_nodes:
                if node != class_uri and node in all_classes:
                    # Check if this represents a subclass relationship
                    if self._represents_subclass_relationship(edge, edge_nodes, class_uri, node):
                        parents.append(node)
        
        # Strategy 2: Namespace-based hierarchy
        class_namespace = self._extract_namespace(class_uri)
        class_name = all_classes[class_uri].name
        
        for other_uri, other_class in all_classes.items():
            if (other_uri != class_uri and 
                other_class.namespace == class_namespace and
                self._is_parent_by_naming(class_name, other_class.name)):
                parents.append(other_uri)
        
        return parents
    
    def _find_child_classes(self, class_uri: str, all_classes: Dict[str, OntologyClass]) -> List[str]:
        """Find child classes for a given class"""
        children = []
        
        # Look for classes that have this class as parent
        for other_uri, other_class in all_classes.items():
            if class_uri in self._find_parent_classes(other_uri, all_classes):
                children.append(other_uri)
        
        return children
    
    def _represents_subclass_relationship(self, edge: str, edge_nodes: List[str], 
                                        child_uri: str, parent_uri: str) -> bool:
        """Check if an edge represents a subclass relationship"""
        
        # Look for subclass indicators in edge or intermediate nodes
        subclass_indicators = ['subClassOf', 'subclass', 'extends', 'inherits', 'isa']
        
        for node in edge_nodes:
            for indicator in subclass_indicators:
                if indicator.lower() in node.lower():
                    return True
        
        return False
    
    def _is_parent_by_naming(self, child_name: str, parent_name: str) -> bool:
        """Determine parent-child relationship based on naming conventions"""
        
        # Common patterns: SpecificThing -> Thing, FinancialInstrument -> Instrument
        if child_name != parent_name and parent_name in child_name:
            return True
        
        # Length-based heuristic: shorter names are often more general
        if len(parent_name) < len(child_name) and parent_name.lower() in child_name.lower():
            return True
        
        return False
    
    def _analyze_class_properties(self, classes: Dict[str, OntologyClass]):
        """Analyze properties associated with each class"""
        
        for class_uri, ontology_class in classes.items():
            properties = []
            
            # Find all entities of this class type
            class_entities = self.kg.get_entities_by_type(ontology_class.name)
            
            # Analyze properties used by these entities
            property_usage = Counter()
            
            for entity in list(class_entities)[:100]:  # Sample for performance
                incident_edges = self.kg.incidences.get_node_edges(entity)
                
                for edge in incident_edges:
                    edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                    
                    # Extract properties from edge
                    edge_properties = self._extract_properties_from_edge(edge, edge_nodes, entity)
                    for prop in edge_properties:
                        property_usage[prop] += 1
            
            # Keep most common properties
            ontology_class.properties = [prop for prop, count in property_usage.most_common(10)]
    
    def _extract_properties_from_edge(self, edge: str, edge_nodes: List[str], entity: str) -> List[str]:
        """Extract property names from an edge"""
        properties = []
        
        for node in edge_nodes:
            if node != entity:
                # Check if this node represents a property
                for indicator in self.ontology_patterns['property_indicators']:
                    if indicator in node.lower():
                        prop_name = self._normalize_property_name(node)
                        properties.append(prop_name)
                        break
        
        return properties
    
    def _normalize_property_name(self, property_node: str) -> str:
        """Normalize property name"""
        # Extract meaningful part from URI
        if '/' in property_node or '#' in property_node:
            separator = '#' if '#' in property_node else '/'
            property_node = property_node.split(separator)[-1]
        
        return property_node
    
    @performance_monitor("ontology_property_analysis")
    def analyze_properties(self) -> Dict[str, OntologyProperty]:
        """
        Analyze all properties in the ontology
        
        Returns:
            Dictionary mapping property URIs to OntologyProperty objects
        """
        
        if self._property_analysis is not None:
            return self._property_analysis
        
        logger.info("Analyzing ontology properties...")
        
        properties = {}
        
        # Analyze relationship types as properties
        for rel_type, relationships in self.kg._relationship_types.items():
            
            if self._is_ontology_property(rel_type):
                
                # Analyze domain and range
                domain_classes = set()
                range_classes = set()
                
                for edge in list(relationships)[:50]:  # Sample for performance
                    edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                    
                    if len(edge_nodes) >= 2:
                        # First node as domain, others as range
                        domain_entity = edge_nodes[0]
                        range_entities = edge_nodes[1:]
                        
                        domain_type = self.kg.get_entity_type(domain_entity)
                        if domain_type:
                            domain_classes.add(domain_type)
                        
                        for range_entity in range_entities:
                            range_type = self.kg.get_entity_type(range_entity)
                            if range_type:
                                range_classes.add(range_type)
                
                # Create property object
                namespace = self._extract_namespace(rel_type)
                
                ontology_property = OntologyProperty(
                    uri=rel_type,
                    name=rel_type,
                    namespace=namespace,
                    domain=list(domain_classes),
                    range=list(range_classes),
                    property_type=self._determine_property_type(rel_type),
                    usage_count=len(relationships)
                )
                
                properties[rel_type] = ontology_property
        
        self._property_analysis = properties
        
        logger.info(f"Found {len(properties)} ontology properties")
        
        return properties
    
    def _is_ontology_property(self, rel_type: str) -> bool:
        """Determine if a relationship type represents an ontology property"""
        
        # Check for property indicators
        for indicator in self.ontology_patterns['property_indicators']:
            if indicator in rel_type.lower():
                return True
        
        # Check for URI structure
        if 'ontology/' in rel_type or '#' in rel_type:
            return True
        
        return len(rel_type.split('_')) > 1  # Compound names often properties
    
    def _determine_property_type(self, property_name: str) -> str:
        """Determine the type of property (object, data, annotation)"""
        
        # Simple heuristics - could be enhanced with actual ontology parsing
        data_indicators = ['value', 'amount', 'date', 'number', 'count', 'id']
        annotation_indicators = ['label', 'comment', 'description', 'note']
        
        property_lower = property_name.lower()
        
        for indicator in data_indicators:
            if indicator in property_lower:
                return 'data'
        
        for indicator in annotation_indicators:
            if indicator in property_lower:
                return 'annotation'
        
        return 'object'  # Default to object property
    
    @performance_monitor("ontology_statistics")
    def calculate_ontology_statistics(self) -> OntologyStats:
        """
        Calculate comprehensive ontology statistics
        
        Returns:
            Detailed ontology metrics
        """
        
        if self._ontology_stats is not None:
            return self._ontology_stats
        
        logger.info("Calculating ontology statistics...")
        
        classes = self.analyze_class_hierarchy()
        properties = self.analyze_properties()
        
        # Basic counts
        total_classes = len(classes)
        total_properties = len(properties)
        total_instances = sum(cls.instances_count for cls in classes.values())
        
        # Hierarchy depth analysis
        max_depth = self._calculate_max_hierarchy_depth(classes)
        
        # Average children per class
        total_children = sum(len(cls.subclasses) for cls in classes.values())
        avg_children = total_children / total_classes if total_classes > 0 else 0
        
        # Namespace distribution
        namespace_dist = Counter()
        for cls in classes.values():
            namespace_dist[cls.namespace] += 1
        for prop in properties.values():
            namespace_dist[prop.namespace] += 1
        
        # Complexity metrics
        complexity_metrics = self._calculate_complexity_metrics(classes, properties)
        
        stats = OntologyStats(
            total_classes=total_classes,
            total_properties=total_properties,
            total_instances=total_instances,
            max_depth=max_depth,
            avg_children_per_class=avg_children,
            namespace_distribution=dict(namespace_dist),
            complexity_metrics=complexity_metrics
        )
        
        self._ontology_stats = stats
        
        logger.info(f"Ontology statistics: {total_classes} classes, {total_properties} properties, depth {max_depth}")
        
        return stats
    
    def _calculate_max_hierarchy_depth(self, classes: Dict[str, OntologyClass]) -> int:
        """Calculate maximum depth of class hierarchy"""
        
        def get_depth(class_uri: str, visited: Optional[Set[str]] = None) -> int:
            if visited is None:
                visited = set()
            
            if class_uri in visited:  # Avoid cycles
                return 0
            
            visited.add(class_uri)
            
            if class_uri not in classes:
                return 0
            
            ontology_class = classes[class_uri]
            
            if not ontology_class.subclasses:
                return 1
            
            max_child_depth = 0
            for child_uri in ontology_class.subclasses:
                child_depth = get_depth(child_uri, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return 1 + max_child_depth
        
        # Find root classes (classes with no parents)
        root_classes = [uri for uri, cls in classes.items() if not cls.parent_classes]
        
        if not root_classes:
            # No clear roots, check all classes
            root_classes = list(classes.keys())
        
        max_depth = 0
        for root_uri in root_classes:
            depth = get_depth(root_uri)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_complexity_metrics(self, classes: Dict[str, OntologyClass], 
                                    properties: Dict[str, OntologyProperty]) -> Dict[str, Any]:
        """Calculate ontology complexity metrics"""
        
        # Property distribution across classes
        properties_per_class = [len(cls.properties) for cls in classes.values()]
        avg_properties_per_class = sum(properties_per_class) / len(properties_per_class) if properties_per_class else 0
        
        # Domain/range complexity
        avg_domain_size = sum(len(prop.domain) for prop in properties.values()) / len(properties) if properties else 0
        avg_range_size = sum(len(prop.range) for prop in properties.values()) / len(properties) if properties else 0
        
        # Property usage distribution
        property_usage = [prop.usage_count for prop in properties.values()]
        total_usage = sum(property_usage)
        avg_property_usage = total_usage / len(property_usage) if property_usage else 0
        
        return {
            'avg_properties_per_class': avg_properties_per_class,
            'avg_domain_size': avg_domain_size,
            'avg_range_size': avg_range_size,
            'avg_property_usage': avg_property_usage,
            'property_type_distribution': self._get_property_type_distribution(properties)
        }
    
    def _get_property_type_distribution(self, properties: Dict[str, OntologyProperty]) -> Dict[str, int]:
        """Get distribution of property types"""
        type_count = Counter()
        for prop in properties.values():
            type_count[prop.property_type] += 1
        return dict(type_count)
    
    def get_domain_analysis(self) -> Dict[str, Any]:
        """
        Generic domain analysis for any ontology
        
        Returns:
            Domain-specific insights and namespace-based metrics
        """
        
        domain_analysis = {
            'namespace_modules': {},
            'cross_namespace_relationships': [],
            'coverage_metrics': {},
            'semantic_patterns': {}
        }
        
        classes = self.analyze_class_hierarchy()
        properties = self.analyze_properties()
        
        # Analyze by namespace/domain modules
        namespace_groups = defaultdict(list)
        for cls in classes.values():
            namespace_groups[cls.namespace].append(cls)
        
        for namespace, namespace_classes in namespace_groups.items():
            domain_analysis['namespace_modules'][namespace] = {
                'class_count': len(namespace_classes),
                'classes': [cls.name for cls in namespace_classes],
                'instance_count': sum(cls.instances_count for cls in namespace_classes),
                'properties': self._get_namespace_properties(namespace, properties)
            }
        
        # Cross-namespace relationship analysis
        domain_analysis['cross_namespace_relationships'] = self._analyze_cross_namespace_relationships(classes, properties)
        
        # Coverage metrics
        domain_analysis['coverage_metrics'] = self._calculate_domain_coverage(classes, properties)
        
        # Semantic patterns
        domain_analysis['semantic_patterns'] = self._identify_semantic_patterns(classes, properties)
        
        return domain_analysis
    
    def _get_namespace_properties(self, namespace: str, properties: Dict[str, OntologyProperty]) -> List[str]:
        """Get properties belonging to a specific namespace"""
        return [prop.name for prop in properties.values() if prop.namespace == namespace]
    
    def _analyze_cross_namespace_relationships(self, classes: Dict[str, OntologyClass], 
                                             properties: Dict[str, OntologyProperty]) -> List[Dict[str, Any]]:
        """Analyze relationships that cross namespace boundaries"""
        cross_relationships = []
        
        for prop in properties.values():
            # Check if property connects different namespaces
            domain_namespaces = set()
            range_namespaces = set()
            
            for domain_class in prop.domain:
                if domain_class in classes:
                    domain_namespaces.add(classes[domain_class].namespace)
            
            for range_class in prop.range:
                if range_class in classes:
                    range_namespaces.add(classes[range_class].namespace)
            
            # If property connects different namespaces
            if len(domain_namespaces) > 1 or len(range_namespaces) > 1 or \
               (domain_namespaces and range_namespaces and domain_namespaces != range_namespaces):
                cross_relationships.append({
                    'property': prop.name,
                    'domain_namespaces': list(domain_namespaces),
                    'range_namespaces': list(range_namespaces),
                    'usage_count': prop.usage_count
                })
        
        return cross_relationships
    
    def _calculate_domain_coverage(self, classes: Dict[str, OntologyClass], 
                                  properties: Dict[str, OntologyProperty]) -> Dict[str, Any]:
        """Calculate coverage metrics for the domain"""
        total_entities = sum(cls.instances_count for cls in classes.values())
        
        # Classes with instances vs total classes
        classes_with_instances = sum(1 for cls in classes.values() if cls.instances_count > 0)
        
        # Properties with usage vs total properties  
        properties_with_usage = sum(1 for prop in properties.values() if prop.usage_count > 0)
        
        return {
            'class_instantiation_rate': classes_with_instances / len(classes) if classes else 0,
            'property_usage_rate': properties_with_usage / len(properties) if properties else 0,
            'total_entities': total_entities,
            'avg_instances_per_class': total_entities / len(classes) if classes else 0,
            'classes_without_instances': len(classes) - classes_with_instances,
            'properties_without_usage': len(properties) - properties_with_usage
        }
    
    def _identify_semantic_patterns(self, classes: Dict[str, OntologyClass], 
                                   properties: Dict[str, OntologyProperty]) -> Dict[str, Any]:
        """Identify common semantic patterns in the ontology"""
        patterns = {
            'naming_conventions': {},
            'hierarchy_patterns': {},
            'property_patterns': {}
        }
        
        # Analyze naming conventions
        class_names = [cls.name for cls in classes.values()]
        
        # Check for common prefixes/suffixes
        prefixes = Counter()
        suffixes = Counter()
        
        for name in class_names:
            if len(name) > 3:
                prefixes[name[:3]] += 1
                suffixes[name[-3:]] += 1
        
        patterns['naming_conventions'] = {
            'common_prefixes': dict(prefixes.most_common(5)),
            'common_suffixes': dict(suffixes.most_common(5)),
            'camel_case_usage': sum(1 for name in class_names if any(c.isupper() for c in name[1:])),
            'underscore_usage': sum(1 for name in class_names if '_' in name)
        }
        
        # Hierarchy patterns
        hierarchy_depths = []
        for cls in classes.values():
            depth = len(cls.parent_classes)
            hierarchy_depths.append(depth)
        
        patterns['hierarchy_patterns'] = {
            'max_inheritance_depth': max(hierarchy_depths) if hierarchy_depths else 0,
            'avg_inheritance_depth': sum(hierarchy_depths) / len(hierarchy_depths) if hierarchy_depths else 0,
            'classes_with_multiple_parents': sum(1 for cls in classes.values() if len(cls.parent_classes) > 1)
        }
        
        # Property patterns
        property_types = Counter(prop.property_type for prop in properties.values())
        patterns['property_patterns'] = {
            'property_type_distribution': dict(property_types),
            'avg_domain_size': sum(len(prop.domain) for prop in properties.values()) / len(properties) if properties else 0,
            'avg_range_size': sum(len(prop.range) for prop in properties.values()) / len(properties) if properties else 0
        }
        
        return patterns
    
    def export_ontology_schema(self, format: str = 'json') -> str:
        """
        Export ontology schema in specified format
        
        Args:
            format: Export format ('json', 'ttl', 'owl')
            
        Returns:
            Serialized ontology schema
        """
        
        classes = self.analyze_class_hierarchy()
        properties = self.analyze_properties()
        stats = self.calculate_ontology_statistics()
        
        schema = {
            'metadata': {
                'total_classes': stats.total_classes,
                'total_properties': stats.total_properties,
                'namespaces': self.kg.namespaces,
                'generated_by': 'ANANT OntologyAnalyzer'
            },
            'classes': {uri: {
                'name': cls.name,
                'namespace': cls.namespace,
                'parent_classes': cls.parent_classes,
                'subclasses': cls.subclasses,
                'properties': cls.properties,
                'instances_count': cls.instances_count
            } for uri, cls in classes.items()},
            'properties': {uri: {
                'name': prop.name,
                'namespace': prop.namespace,
                'domain': prop.domain,
                'range': prop.range,
                'property_type': prop.property_type,
                'usage_count': prop.usage_count
            } for uri, prop in properties.items()},
            'statistics': {
                'max_depth': stats.max_depth,
                'avg_children_per_class': stats.avg_children_per_class,
                'namespace_distribution': stats.namespace_distribution,
                'complexity_metrics': stats.complexity_metrics
            }
        }
        
        if format == 'json':
            return json.dumps(schema, indent=2)
        elif format == 'ttl':
            return self._export_as_turtle(schema)
        elif format == 'owl':
            return self._export_as_owl(schema)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_as_turtle(self, schema: Dict) -> str:
        """Export schema as Turtle/TTL format"""
        # Basic TTL export - would need full implementation
        ttl_lines = [
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix : <http://example.org/ontology#> .",
            ""
        ]
        
        # Add class definitions
        for cls_data in schema['classes'].values():
            ttl_lines.append(f":{cls_data['name']} a owl:Class .")
            if cls_data['parent_classes']:
                for parent in cls_data['parent_classes']:
                    ttl_lines.append(f":{cls_data['name']} rdfs:subClassOf :{parent} .")
            ttl_lines.append("")
        
        return "\n".join(ttl_lines)
    
    def _export_as_owl(self, schema: Dict) -> str:
        """Export schema as OWL format"""
        # Basic OWL export - would need full implementation
        owl_xml = '<?xml version="1.0"?>\n'
        owl_xml += '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
        owl_xml += '         xmlns:owl="http://www.w3.org/2002/07/owl#"\n'
        owl_xml += '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">\n'
        owl_xml += '  <owl:Ontology/>\n'
        
        # Add classes (simplified)
        for cls_data in schema['classes'].values():
            owl_xml += f'  <owl:Class rdf:about="#{cls_data["name"]}"/>\n'
        
        owl_xml += '</rdf:RDF>'
        
        return owl_xml


class SchemaExtractor:
    """
    Extract schema information from various ontology formats
    """
    
    def __init__(self):
        """Initialize schema extractor"""
        self.supported_formats = ['rdf', 'owl', 'ttl', 'n3', 'json-ld']
    
    def extract_from_file(self, file_path: str, format: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract schema from ontology file
        
        Args:
            file_path: Path to ontology file
            format: Ontology format (auto-detected if None)
            
        Returns:
            Extracted schema information
        """
        
        if not rdflib:
            raise ImportError("rdflib is required for ontology file processing")
        
        # Auto-detect format from file extension
        if format is None:
            format = self._detect_format(file_path)
        
        # Load ontology
        graph = rdflib.Graph()
        
        try:
            graph.parse(file_path, format=format)
            logger.info(f"Loaded ontology with {len(graph)} triples")
            
            return self._extract_schema_from_graph(graph)
            
        except Exception as e:
            logger.error(f"Failed to parse ontology file: {str(e)}")
            return {}
    
    def _detect_format(self, file_path: str) -> str:
        """Detect ontology format from file extension"""
        extension = file_path.lower().split('.')[-1]
        
        format_mapping = {
            'rdf': 'xml',
            'owl': 'xml',
            'ttl': 'turtle',
            'n3': 'n3',
            'jsonld': 'json-ld'
        }
        
        return format_mapping.get(extension, 'xml')
    
    def _extract_schema_from_graph(self, graph) -> Dict[str, Any]:
        """Extract schema information from RDF graph"""
        
        schema = {
            'classes': {},
            'properties': {},
            'namespaces': {},
            'metadata': {}
        }
        
        # Extract namespaces
        for prefix, namespace in graph.namespaces():
            schema['namespaces'][str(prefix)] = str(namespace)
        
        # Extract classes (simplified)
        if rdflib:
            try:
                OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
                RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
                
                for cls in graph.subjects(rdflib.RDF.type, OWL.Class):
                    class_uri = str(cls)
                    
                    # Get class label
                    labels = list(graph.objects(cls, RDFS.label))
                    class_name = str(labels[0]) if labels else class_uri.split('#')[-1]
                    
                    schema['classes'][class_uri] = {
                        'name': class_name,
                        'uri': class_uri
                    }
                
                # Extract properties (simplified)
                for prop in graph.subjects(rdflib.RDF.type, OWL.ObjectProperty):
                    prop_uri = str(prop)
                    
                    labels = list(graph.objects(prop, RDFS.label))
                    prop_name = str(labels[0]) if labels else prop_uri.split('#')[-1]
                    
                    schema['properties'][prop_uri] = {
                        'name': prop_name,
                        'uri': prop_uri,
                        'type': 'object'
                    }
            except Exception as e:
                logger.warning(f"RDF processing error: {str(e)}")
                # Fallback to basic parsing
        
        return schema