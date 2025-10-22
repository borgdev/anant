"""
Reasoning Operations for Knowledge Graph

Handles logical reasoning and inference operations including:
- Rule-based inference
- Transitive closure computation
- Consistency checking
- Entailment and subsumption reasoning
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
import logging
from collections import defaultdict, deque
import itertools

from ...exceptions import KnowledgeGraphError
from ...utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class ReasoningOperations:
    """
    Reasoning operations for knowledge graph
    
    Provides logical reasoning capabilities including inference,
    consistency checking, and rule application.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize ReasoningOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Built-in inference rules
        self.built_in_rules = {
            'transitivity': self._rule_transitivity,
            'symmetry': self._rule_symmetry,
            'inverse': self._rule_inverse,
            'subclass': self._rule_subclass_inheritance,
            'subproperty': self._rule_subproperty_inheritance,
            'domain_range': self._rule_domain_range_inference,
            'functional': self._rule_functional_property,
            'inverse_functional': self._rule_inverse_functional_property
        }
        
        # Custom rules storage
        self.custom_rules = {}
        
        # Reasoning cache
        self.inference_cache = {}
    
    @performance_monitor("kg_apply_inference_rules")
    def apply_inference_rules(self, rules: Optional[List[str]] = None, 
                            max_iterations: int = 10) -> Dict[str, Any]:
        """
        Apply inference rules to derive new knowledge
        
        Parameters
        ----------
        rules : Optional[List[str]]
            Specific rules to apply (if None, applies all available rules)
        max_iterations : int, default 10
            Maximum number of inference iterations
            
        Returns
        -------
        Dict[str, Any]
            Results of inference including new triples and statistics
        """
        try:
            if rules is None:
                rules = list(self.built_in_rules.keys()) + list(self.custom_rules.keys())
            
            initial_edge_count = len(self.kg.edges)
            inference_stats = {
                'iterations': 0,
                'new_triples': [],
                'rules_applied': defaultdict(int),
                'total_new_triples': 0
            }
            
            for iteration in range(max_iterations):
                iteration_new_triples = []
                
                # Apply each rule
                for rule_name in rules:
                    if rule_name in self.built_in_rules:
                        rule_func = self.built_in_rules[rule_name]
                    elif rule_name in self.custom_rules:
                        rule_func = self.custom_rules[rule_name]
                    else:
                        self.logger.warning(f"Unknown rule: {rule_name}")
                        continue
                    
                    # Apply rule and collect new triples
                    new_triples = rule_func()
                    if new_triples:
                        iteration_new_triples.extend(new_triples)
                        inference_stats['rules_applied'][rule_name] += len(new_triples)
                
                # Add new triples to knowledge graph
                for subject, predicate, obj, metadata in iteration_new_triples:
                    edge_key = (subject, obj)
                    if edge_key not in self.kg.edges:
                        edge_data = {'type': predicate, 'inferred': True}
                        edge_data.update(metadata)
                        self.kg.add_edge(edge_key, edge_data)
                
                inference_stats['new_triples'].extend(iteration_new_triples)
                inference_stats['iterations'] = iteration + 1
                
                # Stop if no new triples were inferred
                if not iteration_new_triples:
                    break
                
                self.logger.info(f"Iteration {iteration + 1}: {len(iteration_new_triples)} new triples")
            
            inference_stats['total_new_triples'] = len(self.kg.edges) - initial_edge_count
            
            # Clear cache after inference
            self.inference_cache.clear()
            
            return inference_stats
            
        except Exception as e:
            self.logger.error(f"Error applying inference rules: {e}")
            raise KnowledgeGraphError(f"Failed to apply inference rules: {e}")
    
    def _rule_transitivity(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply transitivity rule for transitive properties"""
        new_triples = []
        
        # Find transitive properties
        transitive_properties = set()
        for node in self.kg.nodes:
            node_data = self.kg.nodes[node]
            characteristics = node_data.get('characteristics', [])
            if 'transitive' in characteristics:
                transitive_properties.add(node)
        
        # Apply transitivity: if P(a,b) and P(b,c) then P(a,c)
        for prop in transitive_properties:
            # Find all edges with this property
            prop_edges = []
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == prop:
                    prop_edges.append(edge)
            
            # Find transitive connections
            for edge1 in prop_edges:
                for edge2 in prop_edges:
                    if edge1[1] == edge2[0]:  # b connects edge1 and edge2
                        new_subject = edge1[0]
                        new_object = edge2[1]
                        
                        # Check if this triple already exists
                        if (new_subject, new_object) not in self.kg.edges:
                            new_triples.append((
                                new_subject, prop, new_object,
                                {'rule': 'transitivity', 'source_edges': [edge1, edge2]}
                            ))
        
        return new_triples
    
    def _rule_symmetry(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply symmetry rule for symmetric properties"""
        new_triples = []
        
        # Find symmetric properties
        symmetric_properties = set()
        for node in self.kg.nodes:
            node_data = self.kg.nodes[node]
            characteristics = node_data.get('characteristics', [])
            if 'symmetric' in characteristics:
                symmetric_properties.add(node)
        
        # Apply symmetry: if P(a,b) then P(b,a)
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            prop = edge_data.get('type')
            
            if prop in symmetric_properties:
                subject, obj = edge
                reverse_edge = (obj, subject)
                
                # Check if reverse edge already exists
                if reverse_edge not in self.kg.edges:
                    new_triples.append((
                        obj, prop, subject,
                        {'rule': 'symmetry', 'source_edge': edge}
                    ))
        
        return new_triples
    
    def _rule_inverse(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply inverse property rules"""
        new_triples = []
        
        # Find inverse property relationships
        inverse_pairs = {}
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') == 'owl:inverseOf':
                prop1, prop2 = edge
                inverse_pairs[prop1] = prop2
                inverse_pairs[prop2] = prop1
        
        # Apply inverse rule: if P1(a,b) and P1 inverseOf P2, then P2(b,a)
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            prop = edge_data.get('type')
            
            if prop in inverse_pairs:
                inverse_prop = inverse_pairs[prop]
                subject, obj = edge
                inverse_edge = (obj, subject)
                
                # Check if inverse edge exists
                if inverse_edge not in self.kg.edges:
                    new_triples.append((
                        obj, inverse_prop, subject,
                        {'rule': 'inverse', 'source_edge': edge}
                    ))
        
        return new_triples
    
    def _rule_subclass_inheritance(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply subclass inheritance: if X instanceof A and A subClassOf B, then X instanceof B"""
        new_triples = []
        
        # Find subclass relationships
        subclass_map = {}
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') == 'rdfs:subClassOf':
                subclass, superclass = edge
                if subclass not in subclass_map:
                    subclass_map[subclass] = set()
                subclass_map[subclass].add(superclass)
        
        # Find instance relationships
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') in ['rdf:type', 'instanceof']:
                instance, class_uri = edge
                
                # Find all superclasses
                superclasses = self._get_all_superclasses(class_uri, subclass_map)
                
                # Add instance relationships to superclasses
                for superclass in superclasses:
                    new_edge = (instance, superclass)
                    if new_edge not in self.kg.edges:
                        new_triples.append((
                            instance, 'rdf:type', superclass,
                            {'rule': 'subclass_inheritance', 'source_class': class_uri}
                        ))
        
        return new_triples
    
    def _rule_subproperty_inheritance(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply subproperty inheritance"""
        new_triples = []
        
        # Find subproperty relationships
        subprop_map = {}
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') == 'rdfs:subPropertyOf':
                subprop, superprop = edge
                if subprop not in subprop_map:
                    subprop_map[subprop] = set()
                subprop_map[subprop].add(superprop)
        
        # Apply subproperty inheritance
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            prop = edge_data.get('type')
            
            if prop in subprop_map:
                subject, obj = edge
                
                # Add relationships with superproperties
                superprops = self._get_all_superproperties(prop, subprop_map)
                for superprop in superprops:
                    new_edge = (subject, obj)
                    if new_edge not in self.kg.edges or self.kg.edges[new_edge].get('type') != superprop:
                        new_triples.append((
                            subject, superprop, obj,
                            {'rule': 'subproperty_inheritance', 'source_property': prop}
                        ))
        
        return new_triples
    
    def _rule_domain_range_inference(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply domain and range inference rules"""
        new_triples = []
        
        # Collect property domains and ranges
        property_domains = {}
        property_ranges = {}
        
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') == 'rdfs:domain':
                prop, domain_class = edge
                property_domains[prop] = domain_class
            elif edge_data.get('type') == 'rdfs:range':
                prop, range_class = edge
                property_ranges[prop] = range_class
        
        # Apply domain/range inference
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            prop = edge_data.get('type')
            subject, obj = edge
            
            # Domain inference: if P(a,b) and P domain C, then a rdf:type C
            if prop in property_domains:
                domain_class = property_domains[prop]
                type_edge = (subject, domain_class)
                if type_edge not in self.kg.edges:
                    new_triples.append((
                        subject, 'rdf:type', domain_class,
                        {'rule': 'domain_inference', 'source_property': prop}
                    ))
            
            # Range inference: if P(a,b) and P range C, then b rdf:type C
            if prop in property_ranges:
                range_class = property_ranges[prop]
                type_edge = (obj, range_class)
                if type_edge not in self.kg.edges:
                    new_triples.append((
                        obj, 'rdf:type', range_class,
                        {'rule': 'range_inference', 'source_property': prop}
                    ))
        
        return new_triples
    
    def _rule_functional_property(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply functional property constraints"""
        new_triples = []
        
        # Find functional properties
        functional_props = set()
        for node in self.kg.nodes:
            node_data = self.kg.nodes[node]
            characteristics = node_data.get('characteristics', [])
            if 'functional' in characteristics:
                functional_props.add(node)
        
        # Check functional property constraints
        # For functional properties, if P(a,b) and P(a,c), then b = c
        for prop in functional_props:
            prop_edges = defaultdict(list)
            
            # Group edges by subject
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == prop:
                    subject, obj = edge
                    prop_edges[subject].append(obj)
            
            # Add equivalence relationships for multiple objects
            for subject, objects in prop_edges.items():
                if len(objects) > 1:
                    for obj1, obj2 in itertools.combinations(objects, 2):
                        equiv_edge1 = (obj1, obj2)
                        equiv_edge2 = (obj2, obj1)
                        
                        if equiv_edge1 not in self.kg.edges:
                            new_triples.append((
                                obj1, 'owl:sameAs', obj2,
                                {'rule': 'functional_property', 'property': prop, 'subject': subject}
                            ))
        
        return new_triples
    
    def _rule_inverse_functional_property(self) -> List[Tuple[str, str, str, Dict]]:
        """Apply inverse functional property constraints"""
        new_triples = []
        
        # Find inverse functional properties
        inverse_functional_props = set()
        for node in self.kg.nodes:
            node_data = self.kg.nodes[node]
            characteristics = node_data.get('characteristics', [])
            if 'inverse_functional' in characteristics:
                inverse_functional_props.add(node)
        
        # Check inverse functional property constraints
        # For inverse functional properties, if P(a,c) and P(b,c), then a = b
        for prop in inverse_functional_props:
            prop_edges = defaultdict(list)
            
            # Group edges by object
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == prop:
                    subject, obj = edge
                    prop_edges[obj].append(subject)
            
            # Add equivalence relationships for multiple subjects
            for obj, subjects in prop_edges.items():
                if len(subjects) > 1:
                    for subj1, subj2 in itertools.combinations(subjects, 2):
                        equiv_edge = (subj1, subj2)
                        
                        if equiv_edge not in self.kg.edges:
                            new_triples.append((
                                subj1, 'owl:sameAs', subj2,
                                {'rule': 'inverse_functional_property', 'property': prop, 'object': obj}
                            ))
        
        return new_triples
    
    def _get_all_superclasses(self, class_uri: str, subclass_map: Dict[str, Set[str]]) -> Set[str]:
        """Get all superclasses for a given class (transitive closure)"""
        superclasses = set()
        visited = set()
        queue = deque([class_uri])
        
        while queue:
            current_class = queue.popleft()
            if current_class in visited:
                continue
            
            visited.add(current_class)
            
            if current_class in subclass_map:
                for superclass in subclass_map[current_class]:
                    if superclass not in superclasses:
                        superclasses.add(superclass)
                        queue.append(superclass)
        
        return superclasses
    
    def _get_all_superproperties(self, prop_uri: str, subprop_map: Dict[str, Set[str]]) -> Set[str]:
        """Get all superproperties for a given property (transitive closure)"""
        superprops = set()
        visited = set()
        queue = deque([prop_uri])
        
        while queue:
            current_prop = queue.popleft()
            if current_prop in visited:
                continue
            
            visited.add(current_prop)
            
            if current_prop in subprop_map:
                for superprop in subprop_map[current_prop]:
                    if superprop not in superprops:
                        superprops.add(superprop)
                        queue.append(superprop)
        
        return superprops
    
    @performance_monitor("kg_check_consistency")
    def check_consistency(self) -> Dict[str, Any]:
        """
        Check logical consistency of the knowledge graph
        
        Returns
        -------
        Dict[str, Any]
            Consistency check results including conflicts and warnings
        """
        try:
            consistency_result = {
                'consistent': True,
                'conflicts': [],
                'warnings': [],
                'checked_constraints': []
            }
            
            # Check for basic conflicts
            self._check_type_conflicts(consistency_result)
            self._check_property_conflicts(consistency_result)
            self._check_cardinality_conflicts(consistency_result)
            self._check_disjoint_classes(consistency_result)
            
            return consistency_result
            
        except Exception as e:
            self.logger.error(f"Error checking consistency: {e}")
            raise KnowledgeGraphError(f"Failed to check consistency: {e}")
    
    def _check_type_conflicts(self, result: Dict[str, Any]):
        """Check for type conflicts"""
        result['checked_constraints'].append('type_conflicts')
        
        # Find entities with multiple incompatible types
        entity_types = defaultdict(set)
        
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') in ['rdf:type', 'instanceof']:
                entity, entity_type = edge
                entity_types[entity].add(entity_type)
        
        # Check for disjoint classes
        disjoint_pairs = self._get_disjoint_classes()
        
        for entity, types in entity_types.items():
            for type1, type2 in itertools.combinations(types, 2):
                if (type1, type2) in disjoint_pairs or (type2, type1) in disjoint_pairs:
                    result['conflicts'].append({
                        'type': 'disjoint_types',
                        'entity': entity,
                        'conflicting_types': [type1, type2],
                        'description': f"Entity {entity} has conflicting types: {type1} and {type2}"
                    })
                    result['consistent'] = False
    
    def _check_property_conflicts(self, result: Dict[str, Any]):
        """Check for property-related conflicts"""
        result['checked_constraints'].append('property_conflicts')
        
        # Check functional property violations
        functional_props = self._get_functional_properties()
        
        for prop in functional_props:
            subject_objects = defaultdict(set)
            
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == prop:
                    subject, obj = edge
                    subject_objects[subject].add(obj)
            
            for subject, objects in subject_objects.items():
                if len(objects) > 1:
                    # Check if objects are equivalent
                    equivalent_objects = self._find_equivalent_objects(objects)
                    if len(equivalent_objects) > 1:
                        result['conflicts'].append({
                            'type': 'functional_property_violation',
                            'property': prop,
                            'subject': subject,
                            'conflicting_objects': list(objects),
                            'description': f"Functional property {prop} has multiple distinct values for {subject}"
                        })
                        result['consistent'] = False
    
    def _check_cardinality_conflicts(self, result: Dict[str, Any]):
        """Check for cardinality constraint violations"""
        result['checked_constraints'].append('cardinality_conflicts')
        
        # This would check against defined cardinality restrictions
        # Implementation depends on how cardinality constraints are stored
        pass
    
    def _check_disjoint_classes(self, result: Dict[str, Any]):
        """Check for disjoint class violations"""
        result['checked_constraints'].append('disjoint_classes')
        
        # Find explicitly disjoint classes
        disjoint_pairs = self._get_disjoint_classes()
        
        # Check if any entity belongs to disjoint classes
        for entity, types in self._get_entity_types().items():
            for type1, type2 in itertools.combinations(types, 2):
                if (type1, type2) in disjoint_pairs:
                    result['conflicts'].append({
                        'type': 'disjoint_class_violation',
                        'entity': entity,
                        'disjoint_classes': [type1, type2],
                        'description': f"Entity {entity} belongs to disjoint classes: {type1} and {type2}"
                    })
                    result['consistent'] = False
    
    def _get_disjoint_classes(self) -> Set[Tuple[str, str]]:
        """Get pairs of disjoint classes"""
        disjoint_pairs = set()
        
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') == 'owl:disjointWith':
                class1, class2 = edge
                disjoint_pairs.add((class1, class2))
                disjoint_pairs.add((class2, class1))  # Symmetric
        
        return disjoint_pairs
    
    def _get_functional_properties(self) -> Set[str]:
        """Get set of functional properties"""
        functional_props = set()
        
        for node in self.kg.nodes:
            node_data = self.kg.nodes[node]
            characteristics = node_data.get('characteristics', [])
            if 'functional' in characteristics:
                functional_props.add(node)
        
        return functional_props
    
    def _get_entity_types(self) -> Dict[str, Set[str]]:
        """Get mapping of entities to their types"""
        entity_types = defaultdict(set)
        
        for edge in self.kg.edges:
            edge_data = self.kg.edges[edge]
            if edge_data.get('type') in ['rdf:type', 'instanceof']:
                entity, entity_type = edge
                entity_types[entity].add(entity_type)
        
        return entity_types
    
    def _find_equivalent_objects(self, objects: Set[str]) -> Set[Set[str]]:
        """Find equivalent objects using sameAs relationships"""
        # Simple implementation - could be enhanced with proper equivalence classes
        equivalent_groups: List[Set[str]] = []
        remaining_objects = set(objects)
        
        while remaining_objects:
            current_obj = remaining_objects.pop()
            equivalent_group = {current_obj}
            
            # Find all objects equivalent to current_obj
            for other_obj in list(remaining_objects):
                if self._are_equivalent(current_obj, other_obj):
                    equivalent_group.add(other_obj)
                    remaining_objects.remove(other_obj)
            
            equivalent_groups.append(equivalent_group)
        
        return set(equivalent_groups)
    
    def _are_equivalent(self, obj1: str, obj2: str) -> bool:
        """Check if two objects are equivalent via sameAs relationships"""
        # Check direct equivalence
        same_as_edge1 = (obj1, obj2)
        same_as_edge2 = (obj2, obj1)
        
        for edge in [same_as_edge1, same_as_edge2]:
            if edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == 'owl:sameAs':
                    return True
        
        return False
    
    def add_custom_rule(self, rule_name: str, rule_function: Callable[[], List[Tuple[str, str, str, Dict]]]):
        """
        Add a custom inference rule
        
        Parameters
        ----------
        rule_name : str
            Name of the custom rule
        rule_function : Callable
            Function that returns list of new triples: (subject, predicate, object, metadata)
        """
        self.custom_rules[rule_name] = rule_function
        self.logger.info(f"Added custom rule: {rule_name}")
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """
        Remove a custom inference rule
        
        Parameters
        ----------
        rule_name : str
            Name of the rule to remove
            
        Returns
        -------
        bool
            True if rule was removed, False if not found
        """
        if rule_name in self.custom_rules:
            del self.custom_rules[rule_name]
            self.logger.info(f"Removed custom rule: {rule_name}")
            return True
        return False
    
    def compute_transitive_closure(self, property_uri: str) -> Set[Tuple[str, str]]:
        """
        Compute transitive closure for a specific property
        
        Parameters
        ----------
        property_uri : str
            URI of the property to compute transitive closure for
            
        Returns
        -------
        Set[Tuple[str, str]]
            Set of (subject, object) pairs in the transitive closure
        """
        try:
            # Get all direct relationships for this property
            direct_relations = set()
            for edge in self.kg.edges:
                edge_data = self.kg.edges[edge]
                if edge_data.get('type') == property_uri:
                    direct_relations.add(edge)
            
            # Compute transitive closure using Floyd-Warshall-like algorithm
            closure = set(direct_relations)
            
            # Keep adding new relations until no more can be added
            changed = True
            while changed:
                changed = False
                new_relations = set()
                
                for (a, b) in closure:
                    for (c, d) in closure:
                        if b == c:  # Can connect a->b and b->d to get a->d
                            new_relation = (a, d)
                            if new_relation not in closure:
                                new_relations.add(new_relation)
                                changed = True
                
                closure.update(new_relations)
            
            return closure
            
        except Exception as e:
            self.logger.error(f"Error computing transitive closure: {e}")
            raise KnowledgeGraphError(f"Failed to compute transitive closure: {e}")
    
    def explain_inference(self, subject: str, predicate: str, obj: str) -> Dict[str, Any]:
        """
        Explain how a specific triple was inferred
        
        Parameters
        ----------
        subject : str
            Subject of the triple
        predicate : str
            Predicate of the triple
        obj : str
            Object of the triple
            
        Returns
        -------
        Dict[str, Any]
            Explanation of the inference chain
        """
        try:
            edge = (subject, obj)
            if edge not in self.kg.edges:
                return {'explained': False, 'reason': 'Triple not found in knowledge graph'}
            
            edge_data = self.kg.edges[edge]
            
            # Check if it's an inferred triple
            if not edge_data.get('inferred', False):
                return {
                    'explained': True,
                    'type': 'asserted',
                    'reason': 'Triple was explicitly asserted, not inferred'
                }
            
            # Extract inference information
            rule = edge_data.get('rule', 'unknown')
            explanation = {
                'explained': True,
                'type': 'inferred',
                'rule': rule,
                'predicate': edge_data.get('type'),
                'metadata': {k: v for k, v in edge_data.items() if k not in ['type', 'inferred', 'rule']}
            }
            
            # Add rule-specific explanation
            if rule == 'transitivity':
                source_edges = edge_data.get('source_edges', [])
                explanation['chain'] = [
                    f"{source_edges[0][0]} -> {source_edges[0][1]}",
                    f"{source_edges[1][0]} -> {source_edges[1][1]}",
                    f"Therefore: {subject} -> {obj}"
                ]
            elif rule == 'subclass_inheritance':
                source_class = edge_data.get('source_class')
                explanation['chain'] = [
                    f"{subject} instanceof {source_class}",
                    f"{source_class} subClassOf {obj}",
                    f"Therefore: {subject} instanceof {obj}"
                ]
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining inference: {e}")
            return {'explained': False, 'reason': f'Error during explanation: {e}'}