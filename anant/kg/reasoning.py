"""
Reasoning and Inference Engine
=============================

Advanced reasoning capabilities for knowledge graphs including path reasoning,
rule-based inference, and logical deduction. Domain-agnostic design.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
import time

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..algorithms.sampling import SmartSampler

logger = logging.getLogger(__name__)


@dataclass
class ReasoningRule:
    """Represents a reasoning rule"""
    rule_id: str
    premise: List[Tuple[str, str, str]]  # (subject, predicate, object) patterns
    conclusion: Tuple[str, str, str]     # (subject, predicate, object) pattern
    confidence: float
    rule_type: str  # 'transitive', 'symmetric', 'inverse', 'custom'


@dataclass
class InferenceResult:
    """Result of an inference operation"""
    subject: str
    predicate: str
    object: str
    confidence: float
    supporting_evidence: List[Tuple[str, str, str]]
    inference_method: str
    reasoning_depth: int


@dataclass
class PathResult:
    """Result of path reasoning"""
    path: List[str]
    path_length: int
    confidence: float
    relationships: List[str]
    semantic_meaning: Optional[str]


class PathReasoner:
    """
    Advanced path reasoning and graph traversal with semantic understanding
    
    Features:
    - Multi-hop path discovery
    - Semantic path interpretation
    - Constraint-based path finding
    - Path ranking by relevance
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize path reasoner
        
        Args:
            knowledge_graph: KnowledgeGraph instance
        """
        self.kg = knowledge_graph
        
        # Configuration
        self.config = {
            'max_path_length': 5,
            'max_paths_per_query': 100,
            'path_confidence_threshold': 0.3,
            'use_sampling': True,
            'sampling_threshold': 5000,
            'semantic_weight': 0.7,
            'structural_weight': 0.3
        }
        
        # Path caches
        self._path_cache = {}
        self._shortest_path_cache = {}
        self._semantic_cache = {}
        
        logger.info("Path Reasoner initialized")
    
    @performance_monitor("path_discovery")
    def find_paths(self, 
                   start_entity: str,
                   end_entity: str,
                   max_length: Optional[int] = None,
                   relationship_constraints: Optional[List[str]] = None,
                   entity_type_constraints: Optional[List[str]] = None) -> List[PathResult]:
        """
        Find all paths between two entities
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_length: Maximum path length
            relationship_constraints: Required relationship types
            entity_type_constraints: Required entity types in path
            
        Returns:
            List of discovered paths with metadata
        """
        
        if max_length is None:
            max_length = self.config['max_path_length']
        
        # Check cache first
        cache_key = f"{start_entity}|{end_entity}|{max_length}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        logger.info(f"Finding paths from {start_entity} to {end_entity} (max length: {max_length})")
        
        if start_entity not in self.kg.nodes or end_entity not in self.kg.nodes:
            return []
        
        paths = []
        
        # Use BFS with path tracking
        queue = deque([(start_entity, [start_entity], [], 1.0)])  # (current, path, relationships, confidence)
        visited_paths = set()
        
        while queue and len(paths) < self.config['max_paths_per_query']:
            current_entity, path, relationships, confidence = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            if current_entity == end_entity and len(path) > 1:
                # Found a path
                semantic_meaning = self._interpret_path_semantics(path, relationships)
                
                path_result = PathResult(
                    path=path,
                    path_length=len(path) - 1,
                    confidence=confidence,
                    relationships=relationships,
                    semantic_meaning=semantic_meaning
                )
                paths.append(path_result)
                continue
            
            # Explore neighbors
            neighbors = self._get_entity_neighbors_with_relationships(current_entity)
            
            for neighbor, relationship in neighbors:
                if neighbor not in path:  # Avoid cycles
                    
                    # Check constraints
                    if relationship_constraints and relationship not in relationship_constraints:
                        continue
                    
                    if entity_type_constraints:
                        neighbor_type = self.kg.get_entity_type(neighbor)
                        if neighbor_type not in entity_type_constraints:
                            continue
                    
                    new_path = path + [neighbor]
                    new_relationships = relationships + [relationship]
                    
                    # Calculate path confidence (decreases with length)
                    new_confidence = confidence * self._calculate_relationship_confidence(relationship)
                    
                    path_key = tuple(new_path)
                    if path_key not in visited_paths and new_confidence >= self.config['path_confidence_threshold']:
                        visited_paths.add(path_key)
                        queue.append((neighbor, new_path, new_relationships, new_confidence))
        
        # Sort paths by relevance
        paths.sort(key=lambda p: (p.confidence, -p.path_length), reverse=True)
        
        # Cache results
        self._path_cache[cache_key] = paths
        
        logger.info(f"Found {len(paths)} paths")
        
        return paths
    
    def _get_entity_neighbors_with_relationships(self, entity: str) -> List[Tuple[str, str]]:
        """Get neighbors with their connecting relationships"""
        
        neighbors_with_rels = []
        
        if entity in self.kg.nodes:
            incident_edges = self.kg.incidences.get_node_edges(entity)
            
            for edge in incident_edges:
                edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                
                # Extract relationship type
                relationship = self._extract_relationship_from_edge(edge, edge_nodes)
                
                for neighbor in edge_nodes:
                    if neighbor != entity:
                        neighbors_with_rels.append((neighbor, relationship))
        
        return neighbors_with_rels
    
    def _extract_relationship_from_edge(self, edge: str, edge_nodes: List[str]) -> str:
        """Extract relationship type from edge"""
        
        # Look for relationship indicators in edge nodes
        for node in edge_nodes:
            if any(indicator in node.lower() for indicator in ['has', 'is', 'relates', 'connects']):
                return self._normalize_relationship_name(node)
        
        # Use edge ID or relationship type
        relationship_type = self.kg._extract_relationship_type(edge, edge_nodes)
        return relationship_type or "connected_to"
    
    def _normalize_relationship_name(self, relationship: str) -> str:
        """Normalize relationship name"""
        
        # Extract meaningful part from URI
        if '/' in relationship or '#' in relationship:
            separator = '#' if '#' in relationship else '/'
            relationship = relationship.split(separator)[-1]
        
        # Remove common prefixes
        prefixes = ['has', 'is', 'get', 'set']
        for prefix in prefixes:
            if relationship.lower().startswith(prefix.lower()):
                relationship = relationship[len(prefix):].lstrip('_')
                break
        
        return relationship
    
    def _calculate_relationship_confidence(self, relationship: str) -> float:
        """Calculate confidence weight for a relationship"""
        
        # Different relationship types have different reliability
        high_confidence_rels = ['subClassOf', 'instanceOf', 'sameAs', 'equivalentTo']
        medium_confidence_rels = ['relatedTo', 'associatedWith', 'partOf']
        
        relationship_lower = relationship.lower()
        
        for rel in high_confidence_rels:
            if rel.lower() in relationship_lower:
                return 0.9
        
        for rel in medium_confidence_rels:
            if rel.lower() in relationship_lower:
                return 0.7
        
        return 0.5  # Default confidence
    
    def _interpret_path_semantics(self, path: List[str], relationships: List[str]) -> Optional[str]:
        """Interpret the semantic meaning of a path"""
        
        if not relationships:
            return None
        
        # Simple semantic interpretation patterns
        if len(relationships) == 1:
            return f"Direct {relationships[0]} relationship"
        
        # Check for common patterns
        rel_string = " -> ".join(relationships)
        
        # Inheritance chain
        if all('subclass' in rel.lower() or 'inherit' in rel.lower() for rel in relationships):
            return f"Inheritance hierarchy ({len(relationships)} levels)"
        
        # Part-whole relationship
        if any('part' in rel.lower() or 'contain' in rel.lower() for rel in relationships):
            return "Part-whole relationship chain"
        
        # Association chain
        if any('associat' in rel.lower() or 'relat' in rel.lower() for rel in relationships):
            return "Association relationship chain"
        
        return f"Complex relationship: {rel_string}"
    
    @performance_monitor("shortest_path_finding")
    def find_shortest_paths(self, start_entity: str, end_entity: str, 
                           max_paths: int = 5) -> List[PathResult]:
        """
        Find shortest paths between entities using optimized algorithm
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_paths: Maximum number of paths to return
            
        Returns:
            List of shortest paths
        """
        
        cache_key = f"shortest|{start_entity}|{end_entity}|{max_paths}"
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        
        if start_entity not in self.kg.nodes or end_entity not in self.kg.nodes:
            return []
        
        # Modified BFS for shortest paths
        queue = deque([(start_entity, [start_entity], [], 1.0, 0)])  # Added distance
        visited_distances = {start_entity: 0}
        shortest_paths = []
        min_distance = float('inf')
        
        while queue and len(shortest_paths) < max_paths:
            current, path, relationships, confidence, distance = queue.popleft()
            
            # Skip if we've found shorter paths and this is longer
            if distance > min_distance:
                break
            
            if current == end_entity and distance > 0:
                # Found target
                if distance < min_distance:
                    min_distance = distance
                    shortest_paths = []  # Reset to keep only shortest
                
                if distance == min_distance:
                    semantic_meaning = self._interpret_path_semantics(path, relationships)
                    
                    path_result = PathResult(
                        path=path,
                        path_length=distance,
                        confidence=confidence,
                        relationships=relationships,
                        semantic_meaning=semantic_meaning
                    )
                    shortest_paths.append(path_result)
                
                continue
            
            # Explore neighbors if within reasonable distance
            if distance < self.config['max_path_length']:
                neighbors = self._get_entity_neighbors_with_relationships(current)
                
                for neighbor, relationship in neighbors:
                    new_distance = distance + 1
                    
                    # Only visit if we haven't been here with shorter distance
                    if (neighbor not in visited_distances or 
                        visited_distances[neighbor] >= new_distance):
                        
                        visited_distances[neighbor] = new_distance
                        
                        new_path = path + [neighbor]
                        new_relationships = relationships + [relationship]
                        new_confidence = confidence * self._calculate_relationship_confidence(relationship)
                        
                        queue.append((neighbor, new_path, new_relationships, new_confidence, new_distance))
        
        # Cache results
        self._shortest_path_cache[cache_key] = shortest_paths
        
        logger.info(f"Found {len(shortest_paths)} shortest paths of length {min_distance}")
        
        return shortest_paths
    
    def analyze_path_patterns(self, entity_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze common path patterns between entity pairs
        
        Args:
            entity_pairs: List of (start, end) entity pairs
            
        Returns:
            Analysis of common path patterns
        """
        
        logger.info(f"Analyzing path patterns for {len(entity_pairs)} entity pairs")
        
        all_paths = []
        
        for start, end in entity_pairs[:50]:  # Limit for performance
            paths = self.find_paths(start, end, max_length=3)
            all_paths.extend(paths)
        
        # Analyze patterns
        relationship_sequences = []
        path_lengths = []
        
        for path in all_paths:
            relationship_sequences.append(tuple(path.relationships))
            path_lengths.append(path.path_length)
        
        # Count common patterns
        from collections import Counter
        sequence_counts = Counter(relationship_sequences)
        length_counts = Counter(path_lengths)
        
        analysis = {
            'total_paths_analyzed': len(all_paths),
            'common_relationship_patterns': dict(sequence_counts.most_common(10)),
            'path_length_distribution': dict(length_counts),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'unique_patterns': len(sequence_counts),
            'most_common_relationships': self._analyze_relationship_frequency(all_paths)
        }
        
        return analysis
    
    def _analyze_relationship_frequency(self, paths: List[PathResult]) -> Dict[str, int]:
        """Analyze frequency of individual relationships"""
        
        from collections import Counter
        all_relationships = []
        
        for path in paths:
            all_relationships.extend(path.relationships)
        
        rel_counts = Counter(all_relationships)
        return dict(rel_counts.most_common(10))


class InferenceEngine:
    """
    Rule-based inference engine for knowledge graphs
    
    Features:
    - Transitive relationship inference
    - Symmetric relationship handling
    - Custom rule application
    - Confidence propagation
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize inference engine
        
        Args:
            knowledge_graph: KnowledgeGraph instance
        """
        self.kg = knowledge_graph
        
        # Built-in reasoning rules
        self.rules = []
        self._initialize_default_rules()
        
        # Configuration
        self.config = {
            'max_inference_depth': 3,
            'min_confidence_threshold': 0.5,
            'max_inferences_per_iteration': 1000,
            'enable_transitive_closure': True,
            'enable_symmetric_closure': True
        }
        
        # Inference caches
        self._inference_cache = {}
        self._transitive_cache = {}
        
        logger.info("Inference Engine initialized")
    
    def _initialize_default_rules(self):
        """Initialize common reasoning rules"""
        
        # Transitive rules
        self.add_rule(ReasoningRule(
            rule_id="subclass_transitivity",
            premise=[("?X", "subClassOf", "?Y"), ("?Y", "subClassOf", "?Z")],
            conclusion=("?X", "subClassOf", "?Z"),
            confidence=0.9,
            rule_type="transitive"
        ))
        
        self.add_rule(ReasoningRule(
            rule_id="partof_transitivity",
            premise=[("?X", "partOf", "?Y"), ("?Y", "partOf", "?Z")],
            conclusion=("?X", "partOf", "?Z"),
            confidence=0.8,
            rule_type="transitive"
        ))
        
        # Symmetric rules
        self.add_rule(ReasoningRule(
            rule_id="sameas_symmetry",
            premise=[("?X", "sameAs", "?Y")],
            conclusion=("?Y", "sameAs", "?X"),
            confidence=1.0,
            rule_type="symmetric"
        ))
        
        # Inverse rules
        self.add_rule(ReasoningRule(
            rule_id="contains_inverse_partof",
            premise=[("?X", "partOf", "?Y")],
            conclusion=("?Y", "contains", "?X"),
            confidence=0.9,
            rule_type="inverse"
        ))
    
    def add_rule(self, rule: ReasoningRule):
        """Add a custom reasoning rule"""
        self.rules.append(rule)
        logger.info(f"Added reasoning rule: {rule.rule_id}")
    
    @performance_monitor("inference_application")
    def apply_inference_rules(self, max_iterations: int = 5) -> List[InferenceResult]:
        """
        Apply all reasoning rules to infer new knowledge
        
        Args:
            max_iterations: Maximum number of inference iterations
            
        Returns:
            List of inferred triples
        """
        
        logger.info(f"Applying inference rules (max {max_iterations} iterations)")
        
        all_inferences = []
        
        for iteration in range(max_iterations):
            iteration_inferences = []
            
            logger.info(f"Inference iteration {iteration + 1}")
            
            for rule in self.rules:
                rule_inferences = self._apply_single_rule(rule)
                iteration_inferences.extend(rule_inferences)
                
                if len(iteration_inferences) >= self.config['max_inferences_per_iteration']:
                    break
            
            if not iteration_inferences:
                logger.info(f"No new inferences in iteration {iteration + 1}, stopping")
                break
            
            # Add inferences to knowledge graph (conceptually)
            all_inferences.extend(iteration_inferences)
            
            logger.info(f"Iteration {iteration + 1}: {len(iteration_inferences)} new inferences")
        
        logger.info(f"Total inferences: {len(all_inferences)}")
        
        return all_inferences
    
    def _apply_single_rule(self, rule: ReasoningRule) -> List[InferenceResult]:
        """Apply a single reasoning rule"""
        
        inferences = []
        
        # Find all instantiations of the rule premise
        premise_matches = self._find_premise_matches(rule.premise)
        
        for match in premise_matches:
            # Apply rule to generate conclusion
            conclusion_triple = self._instantiate_conclusion(rule.conclusion, match)
            
            if conclusion_triple and self._is_novel_inference(conclusion_triple):
                
                # Calculate confidence
                confidence = rule.confidence * self._calculate_evidence_confidence(match)
                
                if confidence >= self.config['min_confidence_threshold']:
                    
                    inference = InferenceResult(
                        subject=conclusion_triple[0],
                        predicate=conclusion_triple[1],
                        object=conclusion_triple[2],
                        confidence=confidence,
                        supporting_evidence=[(m['subject'], m['predicate'], m['object']) for m in match.values()],
                        inference_method=rule.rule_id,
                        reasoning_depth=1  # Could be calculated based on inference chain
                    )
                    
                    inferences.append(inference)
        
        return inferences
    
    def _find_premise_matches(self, premise: List[Tuple[str, str, str]]) -> List[Dict[str, Dict[str, str]]]:
        """Find all matches for rule premises in the knowledge graph"""
        
        matches = []
        
        # For each premise pattern, find matching triples
        if len(premise) == 1:
            # Single premise pattern
            pattern_matches = self._find_pattern_matches(premise[0])
            matches = [{"premise_0": match} for match in pattern_matches]
            
        elif len(premise) == 2:
            # Two premise patterns - find compatible matches
            pattern1_matches = self._find_pattern_matches(premise[0])
            pattern2_matches = self._find_pattern_matches(premise[1])
            
            # Join on shared variables
            for match1 in pattern1_matches:
                for match2 in pattern2_matches:
                    if self._matches_are_compatible(premise[0], match1, premise[1], match2):
                        matches.append({
                            "premise_0": match1,
                            "premise_1": match2
                        })
        
        return matches
    
    def _find_pattern_matches(self, pattern: Tuple[str, str, str]) -> List[Dict[str, str]]:
        """Find matches for a single triple pattern"""
        
        subject_pattern, predicate_pattern, object_pattern = pattern
        matches = []
        
        # Convert to query engine patterns and search
        if hasattr(self.kg, 'query'):
            # Use the semantic query engine if available
            
            # Create a simple pattern dict
            query_pattern = {}
            if not subject_pattern.startswith('?'):
                query_pattern['subject'] = subject_pattern
            if not predicate_pattern.startswith('?'):
                query_pattern['predicate'] = predicate_pattern
            if not object_pattern.startswith('?'):
                query_pattern['object'] = object_pattern
            
            # Add variables
            variables = set()
            if subject_pattern.startswith('?'):
                variables.add(subject_pattern[1:])
            if predicate_pattern.startswith('?'):
                variables.add(predicate_pattern[1:])
            if object_pattern.startswith('?'):
                variables.add(object_pattern[1:])
            
            query_pattern['variables'] = variables
            
            try:
                bindings = self.kg.query.pattern_match(query_pattern, limit=100)
                
                for binding in bindings:
                    match = {
                        'subject': binding.get('subject', subject_pattern),
                        'predicate': binding.get('predicate', predicate_pattern),
                        'object': binding.get('object', object_pattern)
                    }
                    matches.append(match)
                    
            except Exception as e:
                logger.warning(f"Pattern matching failed: {str(e)}")
        
        return matches
    
    def _matches_are_compatible(self, pattern1: Tuple[str, str, str], match1: Dict[str, str],
                               pattern2: Tuple[str, str, str], match2: Dict[str, str]) -> bool:
        """Check if two pattern matches are compatible (share variable bindings)"""
        
        # Extract variable mappings
        vars1 = self._extract_variable_mapping(pattern1, match1)
        vars2 = self._extract_variable_mapping(pattern2, match2)
        
        # Check for consistent variable bindings
        for var, value1 in vars1.items():
            if var in vars2:
                if vars2[var] != value1:
                    return False
        
        return True
    
    def _extract_variable_mapping(self, pattern: Tuple[str, str, str], 
                                 match: Dict[str, str]) -> Dict[str, str]:
        """Extract variable to value mappings from a pattern match"""
        
        mapping = {}
        
        if pattern[0].startswith('?'):
            mapping[pattern[0]] = match['subject']
        if pattern[1].startswith('?'):
            mapping[pattern[1]] = match['predicate']
        if pattern[2].startswith('?'):
            mapping[pattern[2]] = match['object']
        
        return mapping
    
    def _instantiate_conclusion(self, conclusion_pattern: Tuple[str, str, str], 
                               premise_matches: Dict[str, Dict[str, str]]) -> Optional[Tuple[str, str, str]]:
        """Instantiate conclusion pattern with variable bindings"""
        
        # Collect all variable bindings
        all_bindings = {}
        for premise_key, match in premise_matches.items():
            premise_index = int(premise_key.split('_')[1])
            # Would need to access the actual premise pattern here
            # Simplified for this implementation
            pass
        
        # For now, return a simplified instantiation
        # In full implementation, would properly substitute variables
        return conclusion_pattern
    
    def _is_novel_inference(self, triple: Tuple[str, str, str]) -> bool:
        """Check if an inference is novel (not already in the graph)"""
        
        subject, predicate, obj = triple
        
        # Check if relationship already exists
        if subject in self.kg.nodes and obj in self.kg.nodes:
            # Simplified check - in full implementation would check actual relationships
            return True
        
        return False
    
    def _calculate_evidence_confidence(self, evidence: Dict[str, Dict[str, str]]) -> float:
        """Calculate confidence based on supporting evidence"""
        
        # Simple confidence calculation based on number of evidence pieces
        base_confidence = 0.8
        evidence_count = len(evidence)
        
        # More evidence increases confidence, but with diminishing returns
        confidence_boost = min(0.2, evidence_count * 0.05)
        
        return min(1.0, base_confidence + confidence_boost)
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        
        rule_stats = {}
        for rule in self.rules:
            rule_stats[rule.rule_id] = {
                'rule_type': rule.rule_type,
                'confidence': rule.confidence,
                'premise_length': len(rule.premise)
            }
        
        return {
            'total_rules': len(self.rules),
            'rule_types': list(set(rule.rule_type for rule in self.rules)),
            'rule_details': rule_stats,
            'configuration': self.config,
            'cache_sizes': {
                'inference_cache': len(self._inference_cache),
                'transitive_cache': len(self._transitive_cache)
            }
        }