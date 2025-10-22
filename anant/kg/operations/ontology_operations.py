"""
Ontology Operations for Knowledge Graph

Handles ontology-related operations including:
- Class hierarchies and inheritance
- Property definitions and constraints
- Schema validation and consistency checking
- Ontology alignment and merging
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from collections import defaultdict, deque

from ...exceptions import KnowledgeGraphError
from ...utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class OntologyOperations:
    """
    Ontology operations for knowledge graph
    
    Provides capabilities for managing ontological structures,
    class hierarchies, and schema validation.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize OntologyOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Ontology namespace prefixes
        self.namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
            'skos': 'http://www.w3.org/2004/02/skos/core#'
        }
    
    @performance_monitor("kg_define_class")
    def define_class(self, class_uri: str, parent_classes: Optional[List[str]] = None,
                    properties: Optional[Dict[str, Any]] = None, 
                    constraints: Optional[Dict[str, Any]] = None) -> bool:
        """
        Define an ontology class
        
        Parameters
        ----------
        class_uri : str
            URI of the class to define
        parent_classes : Optional[List[str]]
            List of parent class URIs
        properties : Optional[Dict[str, Any]]
            Class properties and their definitions
        constraints : Optional[Dict[str, Any]]
            Class constraints and restrictions
            
        Returns
        -------
        bool
            True if class was successfully defined
        """
        try:
            # Add class node if not exists
            if class_uri not in self.kg.nodes:
                self.kg.add_node(class_uri, {
                    'type': 'owl:Class',
                    'label': self._extract_local_name(class_uri)
                })
            
            # Add subclass relationships
            if parent_classes:
                for parent_uri in parent_classes:
                    # Ensure parent class exists
                    if parent_uri not in self.kg.nodes:
                        self.kg.add_node(parent_uri, {
                            'type': 'owl:Class',
                            'label': self._extract_local_name(parent_uri)
                        })
                    
                    # Add subclass relationship
                    self.kg.add_edge((class_uri, parent_uri), {
                        'type': 'rdfs:subClassOf',
                        'source': 'ontology_definition'
                    })
            
            # Add class properties
            if properties:
                for prop_name, prop_def in properties.items():
                    prop_uri = f"{class_uri}#{prop_name}"
                    
                    # Add property definition
                    self.kg.add_node(prop_uri, {
                        'type': 'owl:ObjectProperty',
                        'domain': class_uri,
                        'range': prop_def.get('range'),
                        'label': prop_name,
                        'description': prop_def.get('description', '')
                    })
                    
                    # Link property to class
                    self.kg.add_edge((class_uri, prop_uri), {
                        'type': 'rdfs:domain',
                        'source': 'ontology_definition'
                    })
            
            # Add constraints
            if constraints:
                constraint_uri = f"{class_uri}_constraints"
                self.kg.add_node(constraint_uri, {
                    'type': 'owl:Restriction',
                    'on_class': class_uri,
                    'constraints': constraints
                })
                
                self.kg.add_edge((class_uri, constraint_uri), {
                    'type': 'owl:equivalentClass',
                    'source': 'ontology_definition'
                })
            
            self.logger.info(f"Defined class: {class_uri}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error defining class {class_uri}: {e}")
            raise KnowledgeGraphError(f"Failed to define class: {e}")
    
    @performance_monitor("kg_get_class_hierarchy")
    def get_class_hierarchy(self, root_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Get class hierarchy structure
        
        Parameters
        ----------
        root_class : Optional[str]
            Root class to start hierarchy from (if None, finds all roots)
            
        Returns
        -------
        Dict[str, Any]
            Hierarchical structure of classes
        """
        try:
            # Find all classes
            classes = set()
            subclass_relations = defaultdict(list)
            
            # Check nodes using the proper interface
            for node in self.kg.nodes:
                # Get node type from semantic metadata
                node_type = self.kg.get_node_type(node)
                if node_type == 'owl:Class':
                    classes.add(node)
                # Also check properties for type information
                try:
                    if hasattr(self.kg, 'properties') and self.kg.properties:
                        node_props = self.kg.properties.get_node_properties(node)
                        if node_props and node_props.get('type') == 'owl:Class':
                            classes.add(node)
                except:
                    pass
            
            # Find subclass relationships using edge types
            for edge in self.kg.edges:
                edge_type = self.kg.get_edge_type(edge)
                if edge_type == 'rdfs:subClassOf':
                    if len(edge) >= 2:  # Ensure edge has at least 2 nodes
                        child_class, parent_class = edge[0], edge[1]
                        subclass_relations[parent_class].append(child_class)
            
            def build_hierarchy(class_uri: str, visited: Optional[set] = None) -> Dict[str, Any]:
                if visited is None:
                    visited = set()
                
                if class_uri in visited:
                    return {'uri': class_uri, 'children': [], 'circular': True}
                
                visited.add(class_uri)
                
                children = []
                for child_uri in subclass_relations.get(class_uri, []):
                    child_hierarchy = build_hierarchy(child_uri, visited.copy())
                    children.append(child_hierarchy)
                
                # Get node data safely
                node_label = class_uri  # Default to URI
                try:
                    if hasattr(self.kg, 'properties') and self.kg.properties:
                        node_props = self.kg.properties.get_node_properties(class_uri)
                        if node_props:
                            node_label = node_props.get('label', class_uri)
                except:
                    pass  # Use default if property access fails
                
                return {
                    'uri': class_uri,
                    'label': node_label,
                    'children': children
                }
            
            # Build hierarchy from root(s)
            if root_class:
                if root_class in classes:
                    return build_hierarchy(root_class)
                else:
                    raise KnowledgeGraphError(f"Root class not found: {root_class}")
            else:
                # Find root classes (classes with no parents)
                child_classes = set()
                for parent, children in subclass_relations.items():
                    child_classes.update(children)
                
                root_classes = classes - child_classes
                
                hierarchy = {
                    'roots': []
                }
                
                for root_uri in root_classes:
                    root_hierarchy = build_hierarchy(root_uri)
                    hierarchy['roots'].append(root_hierarchy)
                
                return hierarchy
            
        except Exception as e:
            self.logger.error(f"Error getting class hierarchy: {e}")
            raise KnowledgeGraphError(f"Failed to get class hierarchy: {e}")
    
    @performance_monitor("kg_define_property")
    def define_property(self, property_uri: str, property_type: str = 'owl:ObjectProperty',
                       domain: Optional[str] = None, range_val: Optional[str] = None,
                       characteristics: Optional[List[str]] = None) -> bool:
        """
        Define an ontology property
        
        Parameters
        ----------
        property_uri : str
            URI of the property to define
        property_type : str, default 'owl:ObjectProperty'
            Type of property (ObjectProperty, DataProperty, etc.)
        domain : Optional[str]
            Domain class for the property
        range_val : Optional[str]
            Range class/datatype for the property
        characteristics : Optional[List[str]]
            Property characteristics (functional, transitive, etc.)
            
        Returns
        -------
        bool
            True if property was successfully defined
        """
        try:
            # Add property node
            property_data = {
                'type': property_type,
                'label': self._extract_local_name(property_uri)
            }
            
            if domain:
                property_data['domain'] = domain
            if range_val:
                property_data['range'] = range_val
            if characteristics:
                # Store characteristics as a comma-separated string for compatibility
                property_data['characteristics'] = ','.join(characteristics) if isinstance(characteristics, list) else str(characteristics)
            
            self.kg.add_node(property_uri, property_data)
            
            # Add domain relationship
            if domain:
                if domain not in self.kg.nodes:
                    self.kg.add_node(domain, {
                        'type': 'owl:Class',
                        'label': self._extract_local_name(domain)
                    })
                
                self.kg.add_edge((property_uri, domain), {
                    'type': 'rdfs:domain',
                    'source': 'ontology_definition'
                })
            
            # Add range relationship
            if range_val:
                if range_val not in self.kg.nodes and not range_val.startswith('xsd:'):
                    self.kg.add_node(range_val, {
                        'type': 'owl:Class',
                        'label': self._extract_local_name(range_val)
                    })
                
                self.kg.add_edge((property_uri, range_val), {
                    'type': 'rdfs:range',
                    'source': 'ontology_definition'
                })
            
            self.logger.info(f"Defined property: {property_uri}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error defining property {property_uri}: {e}")
            raise KnowledgeGraphError(f"Failed to define property: {e}")
    
    def validate_instance(self, instance_uri: str, class_uri: str) -> Dict[str, Any]:
        """
        Validate an instance against class constraints
        
        Parameters
        ----------
        instance_uri : str
            URI of the instance to validate
        class_uri : str
            URI of the class to validate against
            
        Returns
        -------
        Dict[str, Any]
            Validation results including errors and warnings
        """
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'checked_constraints': []
            }
            
            # Check if instance exists
            if instance_uri not in self.kg.nodes:
                validation_result['errors'].append(f"Instance not found: {instance_uri}")
                validation_result['valid'] = False
                return validation_result
            
            # Check if class exists
            if class_uri not in self.kg.nodes:
                validation_result['errors'].append(f"Class not found: {class_uri}")
                validation_result['valid'] = False
                return validation_result
            
            # Get class constraints
            constraints = self._get_class_constraints(class_uri)
            
            # Validate against constraints
            for constraint_type, constraint_value in constraints.items():
                validation_result['checked_constraints'].append(constraint_type)
                
                if constraint_type == 'required_properties':
                    self._validate_required_properties(
                        instance_uri, constraint_value, validation_result
                    )
                elif constraint_type == 'property_ranges':
                    self._validate_property_ranges(
                        instance_uri, constraint_value, validation_result
                    )
                elif constraint_type == 'cardinality':
                    self._validate_cardinality(
                        instance_uri, constraint_value, validation_result
                    )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating instance {instance_uri}: {e}")
            raise KnowledgeGraphError(f"Failed to validate instance: {e}")
    
    def _get_class_constraints(self, class_uri: str) -> Dict[str, Any]:
        """Get all constraints for a class"""
        constraints = {}
        
        # Find constraint nodes
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if (edge_data.get('type') == 'owl:equivalentClass' and 
                edge[0] == class_uri):
                constraint_node = edge[1]
                constraint_data = self.kg.nodes.get(constraint_node, {})
                if constraint_data.get('type') == 'owl:Restriction':
                    constraints.update(constraint_data.get('constraints', {}))
        
        return constraints
    
    def _validate_required_properties(self, instance_uri: str, required_props: List[str],
                                    validation_result: Dict[str, Any]):
        """Validate required properties"""
        instance_properties = set()
        
        # Get all properties of the instance
        for edge in self.kg.edges:
            if edge[0] == instance_uri:
                edge_data = self.kg.edges[edge]
                prop_type = edge_data.get('type')
                if prop_type and not prop_type.startswith('rdf:') and not prop_type.startswith('rdfs:'):
                    instance_properties.add(prop_type)
        
        # Check for missing required properties
        for required_prop in required_props:
            if required_prop not in instance_properties:
                validation_result['errors'].append(
                    f"Missing required property: {required_prop}"
                )
                validation_result['valid'] = False
    
    def _validate_property_ranges(self, instance_uri: str, property_ranges: Dict[str, str],
                                validation_result: Dict[str, Any]):
        """Validate property value ranges"""
        for edge in self.kg.edges:
            if edge[0] == instance_uri:
                edge_data = self.kg.edges[edge]
                prop_type = edge_data.get('type')
                
                if prop_type in property_ranges:
                    expected_range = property_ranges[prop_type]
                    value_node = edge[1]
                    value_data = self.kg.nodes.get(value_node, {})
                    
                    # Check if value matches expected range
                    if not self._check_range_compatibility(value_data, expected_range):
                        validation_result['errors'].append(
                            f"Property {prop_type} value does not match range {expected_range}"
                        )
                        validation_result['valid'] = False
    
    def _validate_cardinality(self, instance_uri: str, cardinality_constraints: Dict[str, Dict],
                            validation_result: Dict[str, Any]):
        """Validate property cardinality constraints"""
        property_counts = defaultdict(int)
        
        # Count property occurrences
        for edge in self.kg.edges:
            if edge[0] == instance_uri:
                edge_data = self.kg.edges[edge]
                prop_type = edge_data.get('type')
                if prop_type:
                    property_counts[prop_type] += 1
        
        # Check cardinality constraints
        for prop_type, constraints in cardinality_constraints.items():
            count = property_counts.get(prop_type, 0)
            
            if 'min' in constraints and count < constraints['min']:
                validation_result['errors'].append(
                    f"Property {prop_type} has {count} values, minimum {constraints['min']} required"
                )
                validation_result['valid'] = False
            
            if 'max' in constraints and count > constraints['max']:
                validation_result['errors'].append(
                    f"Property {prop_type} has {count} values, maximum {constraints['max']} allowed"
                )
                validation_result['valid'] = False
    
    def _check_range_compatibility(self, value_data: Dict[str, Any], expected_range: str) -> bool:
        """Check if a value is compatible with expected range"""
        value_type = value_data.get('type')
        
        # Direct type match
        if value_type == expected_range:
            return True
        
        # Check if value type is subclass of expected range
        if value_type and self._is_subclass(value_type, expected_range):
            return True
        
        # XSD datatype compatibility
        if expected_range.startswith('xsd:'):
            return self._check_xsd_compatibility(value_data, expected_range)
        
        return False
    
    def _is_subclass(self, child_class: str, parent_class: str) -> bool:
        """Check if child_class is a subclass of parent_class"""
        if not child_class or not parent_class:
            return False
        
        # BFS to find subclass relationship
        visited = set()
        queue = deque([child_class])
        
        while queue:
            current_class = queue.popleft()
            if current_class in visited:
                continue
            
            visited.add(current_class)
            
            if current_class == parent_class:
                return True
            
            # Find parent classes
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if (edge_data.get('type') == 'rdfs:subClassOf' and 
                    edge[0] == current_class):
                    queue.append(edge[1])
        
        return False
    
    def _check_xsd_compatibility(self, value_data: Dict[str, Any], xsd_type: str) -> bool:
        """Check XSD datatype compatibility"""
        value = value_data.get('value')
        
        if xsd_type == 'xsd:string':
            return isinstance(value, str)
        elif xsd_type == 'xsd:integer':
            return isinstance(value, int)
        elif xsd_type == 'xsd:decimal' or xsd_type == 'xsd:double':
            return isinstance(value, (int, float))
        elif xsd_type == 'xsd:boolean':
            return isinstance(value, bool)
        elif xsd_type == 'xsd:date' or xsd_type == 'xsd:dateTime':
            # Basic date/datetime check (could be enhanced)
            return isinstance(value, str) and len(value) >= 10
        
        return False
    
    def get_property_hierarchy(self, root_property: Optional[str] = None) -> Dict[str, Any]:
        """
        Get property hierarchy structure
        
        Parameters
        ----------
        root_property : Optional[str]
            Root property to start hierarchy from
            
        Returns
        -------
        Dict[str, Any]
            Hierarchical structure of properties
        """
        try:
            # Similar to class hierarchy but for properties
            properties = set()
            subproperty_relations = defaultdict(list)
            
            # Find all properties
            for node in self.kg.nodes:
                # Get node type from semantic metadata
                node_type = self.kg.get_node_type(node)
                if node_type and 'Property' in node_type:
                    properties.add(node)
                # Also check properties for type information
                try:
                    if hasattr(self.kg, 'properties') and self.kg.properties:
                        node_props = self.kg.properties.get_node_properties(node)
                        if node_props:
                            prop_type = node_props.get('type', '')
                            if 'Property' in prop_type:
                                properties.add(node)
                except:
                    pass
            
            # Find subproperty relationships
            for edge in self.kg.edges:
                edge_type = self.kg.get_edge_type(edge)
                if edge_type == 'rdfs:subPropertyOf':
                    if len(edge) >= 2:
                        child_prop, parent_prop = edge[0], edge[1]
                        subproperty_relations[parent_prop].append(child_prop)
            
            def build_property_hierarchy(prop_uri: str, visited: Optional[set] = None) -> Dict[str, Any]:
                if visited is None:
                    visited = set()
                
                if prop_uri in visited:
                    return {'uri': prop_uri, 'children': [], 'circular': True}
                
                visited.add(prop_uri)
                
                children = []
                for child_uri in subproperty_relations.get(prop_uri, []):
                    child_hierarchy = build_property_hierarchy(child_uri, visited.copy())
                    children.append(child_hierarchy)
                
                # Get property data safely
                prop_label = prop_uri
                prop_type = 'unknown'
                prop_domain = None
                prop_range = None
                
                try:
                    if hasattr(self.kg, 'properties') and self.kg.properties:
                        prop_props = self.kg.properties.get_node_properties(prop_uri)
                        if prop_props:
                            prop_label = prop_props.get('label', prop_uri)
                            prop_type = prop_props.get('type', 'unknown')
                            prop_domain = prop_props.get('domain')
                            prop_range = prop_props.get('range')
                except:
                    pass  # Use defaults if property access fails
                
                return {
                    'uri': prop_uri,
                    'label': prop_label,
                    'type': prop_type,
                    'domain': prop_domain,
                    'range': prop_range,
                    'children': children
                }
            
            if root_property:
                if root_property in properties:
                    return build_property_hierarchy(root_property)
                else:
                    raise KnowledgeGraphError(f"Root property not found: {root_property}")
            else:
                # Find root properties
                child_properties = set()
                for parent, children in subproperty_relations.items():
                    child_properties.update(children)
                
                root_properties = properties - child_properties
                
                hierarchy = {'roots': []}
                for root_uri in root_properties:
                    root_hierarchy = build_property_hierarchy(root_uri)
                    hierarchy['roots'].append(root_hierarchy)
                
                return hierarchy
            
        except Exception as e:
            self.logger.error(f"Error getting property hierarchy: {e}")
            raise KnowledgeGraphError(f"Failed to get property hierarchy: {e}")
    
    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI"""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri
    
    def add_namespace(self, prefix: str, namespace_uri: str):
        """
        Add a namespace prefix
        
        Parameters
        ----------
        prefix : str
            Namespace prefix
        namespace_uri : str
            Full namespace URI
        """
        self.namespaces[prefix] = namespace_uri
        self.logger.info(f"Added namespace {prefix}: {namespace_uri}")
    
    def expand_uri(self, prefixed_uri: str) -> str:
        """
        Expand a prefixed URI to full URI
        
        Parameters
        ----------
        prefixed_uri : str
            Prefixed URI (e.g., 'owl:Class')
            
        Returns
        -------
        str
            Full URI
        """
        if ':' in prefixed_uri:
            prefix, local_name = prefixed_uri.split(':', 1)
            if prefix in self.namespaces:
                return f"{self.namespaces[prefix]}{local_name}"
        
        return prefixed_uri
    
    def compress_uri(self, full_uri: str) -> str:
        """
        Compress a full URI to prefixed form if possible
        
        Parameters
        ----------
        full_uri : str
            Full URI
            
        Returns
        -------
        str
            Prefixed URI if possible, otherwise original URI
        """
        for prefix, namespace_uri in self.namespaces.items():
            if full_uri.startswith(namespace_uri):
                local_name = full_uri[len(namespace_uri):]
                return f"{prefix}:{local_name}"
        
        return full_uri