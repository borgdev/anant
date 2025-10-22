"""
Cross-Level Operations for Hierarchical Knowledge Graph
=====================================================

This module handles operations that span multiple hierarchy levels,
including cross-level relationship management, pattern detection,
and hierarchical influence propagation.

Key Features:
- Cross-level relationship creation and management
- Pattern detection across hierarchical boundaries
- Influence propagation between levels
- Cross-level consistency validation
- Hierarchical pattern matching and alignment
- Multi-level aggregation and rollup operations
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class CrossLevelOperations:
    """
    Handles operations that span multiple levels of the hierarchy.
    
    This class manages the complex interactions between different hierarchical
    levels, ensuring consistency and enabling analysis that crosses level boundaries.
    
    Features:
    - Cross-level relationship management
    - Hierarchical pattern detection and matching
    - Multi-level influence propagation
    - Cross-level consistency checking
    - Pattern alignment across levels
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize cross-level operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
    
    # =====================================================================
    # CROSS-LEVEL RELATIONSHIP MANAGEMENT
    # =====================================================================
    
    def add_cross_level_relationship(self,
                                   source_entity: str,
                                   target_entity: str,
                                   relationship_type: str,
                                   properties: Optional[Dict[str, Any]] = None,
                                   validate_hierarchy: bool = True) -> bool:
        """
        Add a relationship that spans across hierarchy levels.
        
        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID  
            relationship_type: Type of cross-level relationship
            properties: Additional relationship properties
            validate_hierarchy: Whether to validate hierarchical constraints
            
        Returns:
            Success status
        """
        # Get entity levels
        source_level = self.hkg.entity_levels.get(source_entity)
        target_level = self.hkg.entity_levels.get(target_entity)
        
        if not source_level or not target_level:
            logger.warning(f"Entities must be assigned to levels: {source_entity} -> {source_level}, {target_entity} -> {target_level}")
            return False
        
        if source_level == target_level:
            logger.warning(f"Use regular add_relationship for same-level entities: {source_level}")
            return False
        
        # Validate hierarchical constraints if requested
        if validate_hierarchy and not self._validate_cross_level_relationship(
            source_entity, target_entity, source_level, target_level, relationship_type
        ):
            return False
        
        # Create relationship
        relationship = {
            'id': f"cross_{source_entity}_{target_entity}_{len(self.hkg.cross_level_relationships)}",
            'source_entity': source_entity,
            'target_entity': target_entity,
            'source_level': source_level,
            'target_level': target_level,
            'relationship_type': relationship_type,
            'properties': properties or {},
            'created_at': datetime.now().isoformat(),
            'level_distance': abs(self.hkg.level_order.get(source_level, 0) - self.hkg.level_order.get(target_level, 0))
        }
        
        self.hkg.cross_level_relationships.append(relationship)
        logger.info(f"Added cross-level relationship: {source_entity} ({source_level}) -> {target_entity} ({target_level})")
        
        return True
    
    def _validate_cross_level_relationship(self,
                                         source_entity: str,
                                         target_entity: str,
                                         source_level: str,
                                         target_level: str,
                                         relationship_type: str) -> bool:
        """Validate that cross-level relationship is hierarchically valid."""
        source_order = self.hkg.level_order.get(source_level, 0)
        target_order = self.hkg.level_order.get(target_level, 0)
        
        # Check relationship type constraints
        if relationship_type in ['parent_of', 'contains', 'manages']:
            # These relationships should go from higher to lower levels
            if source_order >= target_order:
                logger.warning(f"Hierarchical constraint violated: {relationship_type} should go from higher to lower level")
                return False
        elif relationship_type in ['child_of', 'contained_by', 'managed_by']:
            # These relationships should go from lower to higher levels
            if source_order <= target_order:
                logger.warning(f"Hierarchical constraint violated: {relationship_type} should go from lower to higher level")
                return False
        
        # Check for cycles (simplified check)
        if self._would_create_cycle(source_entity, target_entity):
            logger.warning(f"Cross-level relationship would create a cycle")
            return False
        
        return True
    
    def _would_create_cycle(self, source_entity: str, target_entity: str) -> bool:
        """Check if adding relationship would create a cycle."""
        # Simple cycle detection - check if target can reach source
        visited = set()
        queue = deque([target_entity])
        
        while queue:
            current = queue.popleft()
            if current == source_entity:
                return True
            
            if current in visited:
                continue
            visited.add(current)
            
            # Follow cross-level relationships
            for rel in self.hkg.cross_level_relationships:
                if rel['source_entity'] == current:
                    queue.append(rel['target_entity'])
        
        return False
    
    def remove_cross_level_relationship(self,
                                      relationship_id: Optional[str] = None,
                                      source_entity: Optional[str] = None,
                                      target_entity: Optional[str] = None) -> bool:
        """
        Remove cross-level relationships by ID or entity pair.
        
        Args:
            relationship_id: Specific relationship ID to remove
            source_entity: Source entity for relationship to remove
            target_entity: Target entity for relationship to remove
            
        Returns:
            Success status
        """
        initial_count = len(self.hkg.cross_level_relationships)
        
        if relationship_id:
            self.hkg.cross_level_relationships = [
                rel for rel in self.hkg.cross_level_relationships
                if rel.get('id') != relationship_id
            ]
        elif source_entity and target_entity:
            self.hkg.cross_level_relationships = [
                rel for rel in self.hkg.cross_level_relationships
                if not (rel.get('source_entity') == source_entity and rel.get('target_entity') == target_entity)
            ]
        else:
            return False
        
        removed_count = initial_count - len(self.hkg.cross_level_relationships)
        logger.info(f"Removed {removed_count} cross-level relationships")
        
        return removed_count > 0
    
    def get_cross_level_relationships(self,
                                    entity_id: Optional[str] = None,
                                    level_id: Optional[str] = None,
                                    relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get cross-level relationships with filtering options.
        
        Args:
            entity_id: Filter by specific entity (source or target)
            level_id: Filter by specific level (source or target)
            relationship_type: Filter by relationship type
            
        Returns:
            List of matching cross-level relationships
        """
        relationships = self.hkg.cross_level_relationships
        
        if entity_id:
            relationships = [
                rel for rel in relationships
                if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id
            ]
        
        if level_id:
            relationships = [
                rel for rel in relationships
                if rel.get('source_level') == level_id or rel.get('target_level') == level_id
            ]
        
        if relationship_type:
            relationships = [
                rel for rel in relationships
                if rel.get('relationship_type') == relationship_type
            ]
        
        return relationships
    
    # =====================================================================
    # PATTERN DETECTION AND MATCHING
    # =====================================================================
    
    def detect_cross_level_patterns(self,
                                  pattern_types: Optional[List[str]] = None,
                                  min_pattern_size: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect patterns that span across hierarchical levels.
        
        Args:
            pattern_types: Types of patterns to detect
            min_pattern_size: Minimum size for valid patterns
            
        Returns:
            Dictionary of detected patterns by type
        """
        if pattern_types is None:
            pattern_types = ['hierarchical_cascade', 'cross_level_cluster', 'level_bridge', 'inheritance_chain']
        
        detected_patterns = {}
        
        for pattern_type in pattern_types:
            if pattern_type == 'hierarchical_cascade':
                patterns = self._detect_hierarchical_cascades(min_pattern_size)
            elif pattern_type == 'cross_level_cluster':
                patterns = self._detect_cross_level_clusters(min_pattern_size)
            elif pattern_type == 'level_bridge':
                patterns = self._detect_level_bridges()
            elif pattern_type == 'inheritance_chain':
                patterns = self._detect_inheritance_chains(min_pattern_size)
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
                patterns = []
            
            detected_patterns[pattern_type] = patterns
        
        return detected_patterns
    
    def _detect_hierarchical_cascades(self, min_size: int) -> List[Dict[str, Any]]:
        """Detect cascading relationships across multiple levels."""
        cascades = []
        
        # Find chains of relationships that span multiple levels
        for start_rel in self.hkg.cross_level_relationships:
            cascade = [start_rel]
            current_entity = start_rel['target_entity']
            
            # Follow the cascade down
            while True:
                next_rel = None
                for rel in self.hkg.cross_level_relationships:
                    if (rel['source_entity'] == current_entity and 
                        rel not in cascade and
                        self.hkg.level_order.get(rel['target_level'], 0) > 
                        self.hkg.level_order.get(rel['source_level'], 0)):
                        next_rel = rel
                        break
                
                if next_rel:
                    cascade.append(next_rel)
                    current_entity = next_rel['target_entity']
                else:
                    break
            
            # Check if cascade meets minimum size
            if len(cascade) >= min_size:
                cascades.append({
                    'pattern_type': 'hierarchical_cascade',
                    'relationships': cascade,
                    'levels_spanned': len(set(rel['source_level'] for rel in cascade) | 
                                        set(rel['target_level'] for rel in cascade)),
                    'entities_involved': list(set(rel['source_entity'] for rel in cascade) |
                                            set(rel['target_entity'] for rel in cascade)),
                    'cascade_depth': len(cascade)
                })
        
        return cascades
    
    def _detect_cross_level_clusters(self, min_size: int) -> List[Dict[str, Any]]:
        """Detect clusters of entities connected across multiple levels."""
        clusters = []
        
        # Build cross-level connectivity graph
        cross_level_graph = defaultdict(set)
        for rel in self.hkg.cross_level_relationships:
            source = rel['source_entity']
            target = rel['target_entity']
            cross_level_graph[source].add(target)
            cross_level_graph[target].add(source)
        
        # Find connected components in cross-level graph
        visited = set()
        for entity in cross_level_graph:
            if entity not in visited:
                cluster = []
                queue = deque([entity])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        
                        for neighbor in cross_level_graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if len(cluster) >= min_size:
                    # Get levels involved in cluster
                    levels_involved = set(self.hkg.entity_levels.get(e, 'unknown') for e in cluster)
                    
                    clusters.append({
                        'pattern_type': 'cross_level_cluster',
                        'entities': cluster,
                        'size': len(cluster),
                        'levels_involved': list(levels_involved),
                        'level_count': len(levels_involved)
                    })
        
        return clusters
    
    def _detect_level_bridges(self) -> List[Dict[str, Any]]:
        """Detect entities that serve as bridges between levels."""
        bridges = []
        
        # Count cross-level connections for each entity
        entity_connections = defaultdict(lambda: {'levels': set(), 'relationship_count': 0})
        
        for rel in self.hkg.cross_level_relationships:
            source = rel['source_entity']
            target = rel['target_entity']
            
            entity_connections[source]['levels'].add(rel['target_level'])
            entity_connections[source]['relationship_count'] += 1
            
            entity_connections[target]['levels'].add(rel['source_level'])
            entity_connections[target]['relationship_count'] += 1
        
        # Find entities connected to multiple levels
        for entity, connections in entity_connections.items():
            if len(connections['levels']) >= 2:  # Connected to at least 2 different levels
                bridges.append({
                    'pattern_type': 'level_bridge',
                    'entity': entity,
                    'entity_level': self.hkg.entity_levels.get(entity, 'unknown'),
                    'connected_levels': list(connections['levels']),
                    'cross_level_connections': connections['relationship_count'],
                    'bridge_degree': len(connections['levels'])
                })
        
        # Sort by bridge degree (most connected first)
        bridges.sort(key=lambda x: x['bridge_degree'], reverse=True)
        
        return bridges
    
    def _detect_inheritance_chains(self, min_size: int) -> List[Dict[str, Any]]:
        """Detect inheritance-like chains across levels."""
        inheritance_chains = []
        
        # Look for inheritance-type relationships
        inheritance_types = ['inherits_from', 'extends', 'specializes', 'is_a', 'parent_of', 'child_of']
        
        for relationship_type in inheritance_types:
            inheritance_rels = [
                rel for rel in self.hkg.cross_level_relationships
                if rel['relationship_type'] == relationship_type
            ]
            
            # Build directed graph for this relationship type
            inheritance_graph = defaultdict(list)
            for rel in inheritance_rels:
                inheritance_graph[rel['source_entity']].append(rel['target_entity'])
            
            # Find chains
            for start_entity in inheritance_graph:
                chain = [start_entity]
                current = start_entity
                
                while current in inheritance_graph and inheritance_graph[current]:
                    next_entity = inheritance_graph[current][0]  # Take first child
                    if next_entity not in chain:  # Avoid cycles
                        chain.append(next_entity)
                        current = next_entity
                    else:
                        break
                
                if len(chain) >= min_size:
                    # Get level information for chain
                    chain_levels = [self.hkg.entity_levels.get(e, 'unknown') for e in chain]
                    level_orders = [self.hkg.level_order.get(level, 0) for level in chain_levels]
                    
                    inheritance_chains.append({
                        'pattern_type': 'inheritance_chain',
                        'relationship_type': relationship_type,
                        'chain': chain,
                        'chain_length': len(chain),
                        'levels': chain_levels,
                        'level_orders': level_orders,
                        'spans_levels': len(set(chain_levels)) > 1
                    })
        
        return inheritance_chains
    
    # =====================================================================
    # INFLUENCE PROPAGATION
    # =====================================================================
    
    def propagate_influence(self,
                          source_entities: Union[str, List[str]],
                          influence_type: str = "activation",
                          propagation_rules: Optional[Dict[str, Any]] = None,
                          max_propagation_steps: int = 10) -> Dict[str, Any]:
        """
        Propagate influence across hierarchical levels.
        
        Args:
            source_entities: Starting points for influence propagation
            influence_type: Type of influence being propagated
            propagation_rules: Rules governing how influence spreads
            max_propagation_steps: Maximum steps for propagation
            
        Returns:
            Results of influence propagation including affected entities
        """
        if isinstance(source_entities, str):
            source_entities = [source_entities]
        
        # Default propagation rules
        default_rules = {
            'decay_factor': 0.8,  # Influence decays by 20% at each step
            'level_weight': 1.0,  # Weight for same-level propagation
            'cross_level_weight': 0.6,  # Weight for cross-level propagation
            'min_influence_threshold': 0.1  # Minimum influence to continue propagation
        }
        
        rules = {**default_rules, **(propagation_rules or {})}
        
        # Initialize influence tracking
        influence_scores = {}
        for entity in source_entities:
            influence_scores[entity] = 1.0
        
        propagation_history = []
        current_wave = set(source_entities)
        
        for step in range(max_propagation_steps):
            if not current_wave:
                break
            
            next_wave = set()
            step_influences = {}
            
            for source_entity in current_wave:
                source_influence = influence_scores.get(source_entity, 0)
                if source_influence < rules['min_influence_threshold']:
                    continue
                
                # Propagate to connected entities
                connected_entities = self._get_connected_entities(source_entity)
                
                for target_entity, connection_type in connected_entities:
                    # Calculate influence transfer
                    if connection_type == 'same_level':
                        transfer_weight = rules['level_weight']
                    else:  # cross_level
                        transfer_weight = rules['cross_level_weight']
                    
                    transferred_influence = (source_influence * 
                                           rules['decay_factor'] * 
                                           transfer_weight)
                    
                    if transferred_influence >= rules['min_influence_threshold']:
                        if target_entity not in influence_scores:
                            influence_scores[target_entity] = 0
                        
                        # Accumulate influence (don't override)
                        influence_scores[target_entity] = max(
                            influence_scores[target_entity],
                            transferred_influence
                        )
                        
                        step_influences[target_entity] = transferred_influence
                        next_wave.add(target_entity)
            
            # Record propagation step
            propagation_history.append({
                'step': step,
                'propagating_from': list(current_wave),
                'newly_influenced': step_influences,
                'total_influenced_entities': len(influence_scores)
            })
            
            current_wave = next_wave
        
        # Analyze propagation results
        results = {
            'influence_type': influence_type,
            'source_entities': source_entities,
            'propagation_rules': rules,
            'total_steps': len(propagation_history),
            'influence_scores': influence_scores,
            'propagation_history': propagation_history,
            'affected_entities_by_level': self._group_entities_by_level(influence_scores.keys()),
            'summary': {
                'total_affected_entities': len(influence_scores),
                'max_influence': max(influence_scores.values()) if influence_scores else 0,
                'avg_influence': sum(influence_scores.values()) / len(influence_scores) if influence_scores else 0,
                'levels_affected': len(self._group_entities_by_level(influence_scores.keys()))
            }
        }
        
        return results
    
    def _get_connected_entities(self, entity_id: str) -> List[Tuple[str, str]]:
        """Get entities connected to the given entity with connection type."""
        connected = []
        
        # Same-level connections
        entity_level = self.hkg.entity_levels.get(entity_id)
        if entity_level and entity_level in self.hkg.level_graphs:
            level_graph = self.hkg.level_graphs[entity_level]
            if hasattr(level_graph, 'get_entity_relationships'):
                relationships = level_graph.get_entity_relationships(entity_id)
                for rel in relationships:
                    target = rel.get('target_entity')
                    source = rel.get('source_entity')
                    
                    other_entity = target if source == entity_id else source
                    if other_entity:
                        connected.append((other_entity, 'same_level'))
        
        # Cross-level connections
        for rel in self.hkg.cross_level_relationships:
            if rel['source_entity'] == entity_id:
                connected.append((rel['target_entity'], 'cross_level'))
            elif rel['target_entity'] == entity_id:
                connected.append((rel['source_entity'], 'cross_level'))
        
        return connected
    
    def _group_entities_by_level(self, entity_ids) -> Dict[str, List[str]]:
        """Group entities by their hierarchical level."""
        grouped = defaultdict(list)
        
        for entity_id in entity_ids:
            level_id = self.hkg.entity_levels.get(entity_id, 'unassigned')
            grouped[level_id].append(entity_id)
        
        return dict(grouped)
    
    # =====================================================================
    # CROSS-LEVEL AGGREGATION
    # =====================================================================
    
    def aggregate_across_levels(self,
                              aggregation_type: str,
                              source_level: str,
                              target_level: str,
                              properties_to_aggregate: List[str]) -> Dict[str, Any]:
        """
        Aggregate properties from source level to target level.
        
        Args:
            aggregation_type: Type of aggregation (sum, count, avg, max, min)
            source_level: Level to aggregate from
            target_level: Level to aggregate to
            properties_to_aggregate: Properties to include in aggregation
            
        Returns:
            Aggregation results
        """
        # Get entities at both levels
        source_entities = self.hkg.hierarchy_ops.get_entities_at_level(source_level)
        target_entities = self.hkg.hierarchy_ops.get_entities_at_level(target_level)
        
        aggregation_results = {}
        
        for target_entity in target_entities:
            # Find source entities connected to this target
            connected_sources = []
            for rel in self.hkg.cross_level_relationships:
                if (rel['target_entity'] == target_entity and 
                    rel['source_level'] == source_level):
                    connected_sources.append(rel['source_entity'])
                elif (rel['source_entity'] == target_entity and 
                      rel['target_level'] == source_level):
                    connected_sources.append(rel['target_entity'])
            
            # Aggregate properties from connected sources
            target_aggregation = {}
            for prop in properties_to_aggregate:
                values = []
                
                for source_entity in connected_sources:
                    entity_data = self.hkg.knowledge_graph.get_entity(source_entity)
                    if entity_data and 'properties' in entity_data:
                        prop_value = entity_data['properties'].get(prop)
                        if prop_value is not None:
                            try:
                                # Try to convert to numeric
                                numeric_value = float(prop_value)
                                values.append(numeric_value)
                            except (ValueError, TypeError):
                                # Handle non-numeric values
                                if aggregation_type == 'count':
                                    values.append(1)
                
                # Apply aggregation function
                if values:
                    if aggregation_type == 'sum':
                        target_aggregation[prop] = sum(values)
                    elif aggregation_type == 'count':
                        target_aggregation[prop] = len(values)
                    elif aggregation_type == 'avg':
                        target_aggregation[prop] = sum(values) / len(values)
                    elif aggregation_type == 'max':
                        target_aggregation[prop] = max(values)
                    elif aggregation_type == 'min':
                        target_aggregation[prop] = min(values)
                    else:
                        target_aggregation[prop] = values  # Return all values
                else:
                    target_aggregation[prop] = None
            
            aggregation_results[target_entity] = {
                'aggregated_properties': target_aggregation,
                'source_entity_count': len(connected_sources),
                'source_entities': connected_sources
            }
        
        return {
            'aggregation_type': aggregation_type,
            'source_level': source_level,
            'target_level': target_level,
            'properties_aggregated': properties_to_aggregate,
            'results': aggregation_results,
            'summary': {
                'target_entities_processed': len(target_entities),
                'successful_aggregations': len([r for r in aggregation_results.values() 
                                              if r['source_entity_count'] > 0])
            }
        }
    
    # =====================================================================
    # CONSISTENCY VALIDATION
    # =====================================================================
    
    def validate_cross_level_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency of cross-level relationships and hierarchical structure.
        
        Returns:
            Validation results including any inconsistencies found
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'is_consistent': True,
            'issues_found': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for orphaned relationships
        orphaned_relationships = []
        for rel in self.hkg.cross_level_relationships:
            source_exists = self.hkg.knowledge_graph.has_entity(rel['source_entity'])
            target_exists = self.hkg.knowledge_graph.has_entity(rel['target_entity'])
            
            if not source_exists or not target_exists:
                orphaned_relationships.append({
                    'relationship': rel,
                    'issue': f"Missing entity: source_exists={source_exists}, target_exists={target_exists}"
                })
        
        if orphaned_relationships:
            validation_results['is_consistent'] = False
            validation_results['issues_found'].extend(orphaned_relationships)
        
        # Check for level consistency
        level_inconsistencies = []
        for rel in self.hkg.cross_level_relationships:
            source_actual_level = self.hkg.entity_levels.get(rel['source_entity'])
            target_actual_level = self.hkg.entity_levels.get(rel['target_entity'])
            
            if (source_actual_level != rel.get('source_level') or 
                target_actual_level != rel.get('target_level')):
                level_inconsistencies.append({
                    'relationship': rel,
                    'issue': f"Level mismatch: expected {rel.get('source_level')}->{rel.get('target_level')}, "
                            f"actual {source_actual_level}->{target_actual_level}"
                })
        
        if level_inconsistencies:
            validation_results['is_consistent'] = False
            validation_results['issues_found'].extend(level_inconsistencies)
        
        # Check for circular dependencies
        cycles = self._detect_hierarchical_cycles()
        if cycles:
            validation_results['is_consistent'] = False
            validation_results['issues_found'].append({
                'type': 'circular_dependencies',
                'cycles': cycles
            })
        
        # Statistics
        validation_results['statistics'] = {
            'total_cross_level_relationships': len(self.hkg.cross_level_relationships),
            'orphaned_relationships': len(orphaned_relationships),
            'level_inconsistencies': len(level_inconsistencies),
            'circular_dependencies': len(cycles)
        }
        
        return validation_results
    
    def _detect_hierarchical_cycles(self) -> List[List[str]]:
        """Detect cycles in the hierarchical structure."""
        cycles = []
        
        # Build directed graph from cross-level relationships
        graph = defaultdict(list)
        for rel in self.hkg.cross_level_relationships:
            graph[rel['source_entity']].append(rel['target_entity'])
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_detection(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if dfs_cycle_detection(neighbor, path + [neighbor]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs_cycle_detection(node, [node])
        
        return cycles