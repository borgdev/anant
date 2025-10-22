"""
Hierarchical Core Operations for Hierarchical Knowledge Graph
===========================================================

This module handles core CRUD operations, batch processing, transaction support,
and fundamental data management for hierarchical knowledge graphs.

Key Features:
- Basic CRUD operations with hierarchical context
- Batch operations for efficient bulk processing
- Transaction support and rollback capabilities
- Data integrity and validation
- Entity lifecycle management
- Relationship consistency maintenance
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set, ContextManager
from collections import defaultdict
from datetime import datetime
import logging
import uuid
import copy
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TransactionContext:
    """Context manager for hierarchical knowledge graph transactions."""
    
    def __init__(self, hkg):
        self.hkg = hkg
        self.operations = []
        self.rollback_data = {}
        self.committed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, rollback
            self.rollback()
        elif not self.committed:
            # No explicit commit, rollback
            self.rollback()
    
    def commit(self):
        """Commit all operations in the transaction."""
        self.committed = True
    
    def rollback(self):
        """Rollback all operations in the transaction."""
        for operation in reversed(self.operations):
            self._rollback_operation(operation)
    
    def _rollback_operation(self, operation):
        """Rollback a single operation."""
        op_type = operation['type']
        
        if op_type == 'add_entity':
            self.hkg.remove_entity(operation['entity_id'])
        elif op_type == 'update_entity':
            if operation['entity_id'] in self.rollback_data:
                # Restore original entity data
                original_data = self.rollback_data[operation['entity_id']]
                self.hkg.knowledge_graph.update_entity(operation['entity_id'], original_data)
        elif op_type == 'remove_entity':
            # Restore removed entity
            if operation['entity_id'] in self.rollback_data:
                entity_data = self.rollback_data[operation['entity_id']]
                self.hkg.add_entity(operation['entity_id'], entity_data['properties'])
        elif op_type == 'add_relationship':
            self.hkg.knowledge_graph.remove_relationship(operation['relationship_id'])
        elif op_type == 'remove_relationship':
            if operation['relationship_id'] in self.rollback_data:
                rel_data = self.rollback_data[operation['relationship_id']]
                self.hkg.add_relationship(
                    rel_data['source_entity'],
                    rel_data['target_entity'],
                    rel_data['relationship_type'],
                    rel_data.get('properties')
                )


class HierarchicalCoreOperations:
    """
    Handles core CRUD operations and data management for hierarchical knowledge graphs.
    
    This class provides fundamental operations for entity and relationship management
    with hierarchical context, transaction support, and batch processing capabilities.
    
    Features:
    - CRUD operations with hierarchical validation
    - Batch processing for performance
    - Transaction support with rollback
    - Data integrity enforcement
    - Efficient bulk operations
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize hierarchical core operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
    
    # =====================================================================
    # BASIC CRUD OPERATIONS
    # =====================================================================
    
    def create_entity(self,
                     entity_id: str,
                     entity_type: str,
                     properties: Dict[str, Any],
                     level_id: Optional[str] = None,
                     validate: bool = True,
                     transaction: Optional[TransactionContext] = None) -> bool:
        """
        Create a new entity with hierarchical context.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Entity properties
            level_id: Level to assign entity to
            validate: Whether to perform validation
            transaction: Optional transaction context
            
        Returns:
            Success status
        """
        try:
            # Validate entity doesn't already exist
            if validate and self.hkg.has_node(entity_id):
                logger.warning(f"Entity {entity_id} already exists")
                return False
            
            # Validate level if specified
            if level_id and level_id not in self.hkg.levels:
                logger.warning(f"Level {level_id} does not exist")
                return False
            
            # Create entity in main knowledge graph
            success = self.hkg.knowledge_graph.add_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                properties=properties
            )
            
            if success:
                # Assign to level if specified
                if level_id:
                    self.hkg.entity_levels[entity_id] = level_id
                    
                    # Add to level-specific graph if it exists
                    if level_id in self.hkg.level_graphs:
                        self.hkg.level_graphs[level_id].add_entity(
                            entity_id, entity_type, properties
                        )
                
                # Record operation for transaction
                if transaction:
                    transaction.operations.append({
                        'type': 'add_entity',
                        'entity_id': entity_id,
                        'entity_type': entity_type,
                        'level_id': level_id
                    })
                
                logger.debug(f"Created entity {entity_id} in level {level_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create entity {entity_id}: {e}")
            return False
    
    def read_entity(self,
                   entity_id: str,
                   include_level_info: bool = True,
                   include_relationships: bool = False) -> Optional[Dict[str, Any]]:
        """
        Read an entity with hierarchical context.
        
        Args:
            entity_id: ID of entity to read
            include_level_info: Include hierarchical level information
            include_relationships: Include relationship information
            
        Returns:
            Entity data with hierarchical context or None
        """
        try:
            # Get base entity data
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if not entity_data:
                return None
            
            # Add hierarchical context
            if include_level_info:
                level_id = self.hkg.entity_levels.get(entity_id)
                entity_data['hierarchical_info'] = {
                    'level_id': level_id,
                    'level_name': self.hkg.levels.get(level_id, {}).get('name') if level_id else None,
                    'level_order': self.hkg.level_order.get(level_id) if level_id else None
                }
            
            # Add relationship information
            if include_relationships:
                # Same-level relationships
                same_level_rels = []
                all_relationships = self.hkg.knowledge_graph.get_all_relationships()
                for rel in all_relationships:
                    if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                        same_level_rels.append(rel)
                
                # Cross-level relationships
                cross_level_rels = []
                for rel in self.hkg.cross_level_relationships:
                    if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                        cross_level_rels.append(rel)
                
                entity_data['relationships'] = {
                    'same_level': same_level_rels,
                    'cross_level': cross_level_rels,
                    'total_count': len(same_level_rels) + len(cross_level_rels)
                }
            
            return entity_data
            
        except Exception as e:
            logger.error(f"Failed to read entity {entity_id}: {e}")
            return None
    
    def update_entity(self,
                     entity_id: str,
                     properties: Optional[Dict[str, Any]] = None,
                     entity_type: Optional[str] = None,
                     new_level_id: Optional[str] = None,
                     merge_properties: bool = True,
                     transaction: Optional[TransactionContext] = None) -> bool:
        """
        Update an entity with hierarchical context.
        
        Args:
            entity_id: ID of entity to update
            properties: New or updated properties
            entity_type: New entity type
            new_level_id: New level to move entity to
            merge_properties: Whether to merge with existing properties
            transaction: Optional transaction context
            
        Returns:
            Success status
        """
        try:
            # Check entity exists
            if not self.hkg.has_node(entity_id):
                logger.warning(f"Entity {entity_id} does not exist")
                return False
            
            # Store original data for rollback
            if transaction:
                original_data = self.hkg.knowledge_graph.get_entity(entity_id)
                transaction.rollback_data[entity_id] = original_data
            
            # Update properties if provided
            if properties is not None:
                existing_entity = self.hkg.knowledge_graph.get_entity(entity_id)
                if existing_entity:
                    if merge_properties:
                        updated_properties = existing_entity.get('properties', {}).copy()
                        updated_properties.update(properties)
                    else:
                        updated_properties = properties
                    
                    # Update in main graph
                    self.hkg.knowledge_graph.update_entity_properties(entity_id, updated_properties)
                    
                    # Update in level graph if entity is in a level
                    current_level = self.hkg.entity_levels.get(entity_id)
                    if current_level and current_level in self.hkg.level_graphs:
                        self.hkg.level_graphs[current_level].update_entity_properties(
                            entity_id, updated_properties
                        )
            
            # Update entity type if provided
            if entity_type is not None:
                self.hkg.knowledge_graph.update_entity_type(entity_id, entity_type)
            
            # Move to new level if specified
            if new_level_id is not None:
                if new_level_id not in self.hkg.levels:
                    logger.warning(f"Target level {new_level_id} does not exist")
                    return False
                
                # Remove from current level
                current_level = self.hkg.entity_levels.get(entity_id)
                if current_level and current_level in self.hkg.level_graphs:
                    self.hkg.level_graphs[current_level].remove_entity(entity_id)
                
                # Add to new level
                self.hkg.entity_levels[entity_id] = new_level_id
                if new_level_id in self.hkg.level_graphs:
                    entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
                    if entity_data:
                        self.hkg.level_graphs[new_level_id].add_entity(
                            entity_id,
                            entity_data.get('type', 'entity'),
                            entity_data.get('properties', {})
                        )
            
            # Record operation for transaction
            if transaction:
                transaction.operations.append({
                    'type': 'update_entity',
                    'entity_id': entity_id,
                    'properties': properties,
                    'entity_type': entity_type,
                    'new_level_id': new_level_id
                })
            
            logger.debug(f"Updated entity {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entity {entity_id}: {e}")
            return False
    
    def delete_entity(self,
                     entity_id: str,
                     cascade: bool = False,
                     transaction: Optional[TransactionContext] = None) -> bool:
        """
        Delete an entity with hierarchical cleanup.
        
        Args:
            entity_id: ID of entity to delete
            cascade: Whether to delete related entities
            transaction: Optional transaction context
            
        Returns:
            Success status
        """
        try:
            # Check entity exists
            if not self.hkg.has_node(entity_id):
                logger.warning(f"Entity {entity_id} does not exist")
                return False
            
            # Store original data for rollback
            if transaction:
                entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
                transaction.rollback_data[entity_id] = entity_data
            
            # Handle cascade deletion
            if cascade:
                # Get related entities through navigation
                if hasattr(self.hkg, 'navigation_ops'):
                    children = self.hkg.navigation_ops.get_children(entity_id)
                    for child_id in children or []:
                        self.delete_entity(child_id, cascade=True, transaction=transaction)
            
            # Remove from level tracking
            current_level = self.hkg.entity_levels.pop(entity_id, None)
            
            # Remove from level-specific graph
            if current_level and current_level in self.hkg.level_graphs:
                self.hkg.level_graphs[current_level].remove_entity(entity_id)
            
            # Remove cross-level relationships
            self.hkg.cross_level_relationships = [
                rel for rel in self.hkg.cross_level_relationships
                if rel.get('source_entity') != entity_id and rel.get('target_entity') != entity_id
            ]
            
            # Remove from main knowledge graph
            success = self.hkg.knowledge_graph.remove_entity(entity_id)
            
            # Record operation for transaction
            if transaction:
                transaction.operations.append({
                    'type': 'remove_entity',
                    'entity_id': entity_id,
                    'cascade': cascade
                })
            
            logger.debug(f"Deleted entity {entity_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_id}: {e}")
            return False
    
    # =====================================================================
    # RELATIONSHIP OPERATIONS
    # =====================================================================
    
    def create_relationship(self,
                          source_entity: str,
                          target_entity: str,
                          relationship_type: str,
                          properties: Optional[Dict[str, Any]] = None,
                          cross_level: bool = False,
                          transaction: Optional[TransactionContext] = None) -> Optional[str]:
        """
        Create a relationship with hierarchical awareness.
        
        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            relationship_type: Type of relationship
            properties: Relationship properties
            cross_level: Whether this is a cross-level relationship
            transaction: Optional transaction context
            
        Returns:
            Relationship ID if successful, None otherwise
        """
        try:
            # Validate entities exist
            if not self.hkg.has_node(source_entity):
                logger.warning(f"Source entity {source_entity} does not exist")
                return None
            
            if not self.hkg.has_node(target_entity):
                logger.warning(f"Target entity {target_entity} does not exist")
                return None
            
            # Generate relationship ID
            relationship_id = str(uuid.uuid4())
            
            # Determine if cross-level based on entity levels
            source_level = self.hkg.entity_levels.get(source_entity)
            target_level = self.hkg.entity_levels.get(target_entity)
            
            if not cross_level and source_level != target_level:
                cross_level = True
                logger.info(f"Auto-detected cross-level relationship: {source_level} -> {target_level}")
            
            if cross_level:
                # Create cross-level relationship
                success = self.hkg.add_cross_level_relationship(
                    source_entity, target_entity, relationship_type, properties
                )
            else:
                # Create same-level relationship
                success = self.hkg.add_relationship(
                    source_entity, target_entity, relationship_type, properties
                )
            
            if success:
                # Record operation for transaction
                if transaction:
                    transaction.operations.append({
                        'type': 'add_relationship',
                        'relationship_id': relationship_id,
                        'source_entity': source_entity,
                        'target_entity': target_entity,
                        'cross_level': cross_level
                    })
                
                logger.debug(f"Created {'cross-level' if cross_level else 'same-level'} relationship: {source_entity} -> {target_entity}")
                return relationship_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create relationship {source_entity} -> {target_entity}: {e}")
            return None
    
    # =====================================================================
    # BATCH OPERATIONS
    # =====================================================================
    
    def batch_create_entities(self,
                            entities: List[Dict[str, Any]],
                            continue_on_error: bool = True,
                            use_transaction: bool = True) -> Dict[str, Any]:
        """
        Create multiple entities in a batch operation.
        
        Args:
            entities: List of entity specifications
            continue_on_error: Whether to continue if individual entities fail
            use_transaction: Whether to wrap in transaction
            
        Returns:
            Results dictionary with success/failure counts
        """
        results = {
            'total': len(entities),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        operation_func = self._batch_create_entities_impl
        
        if use_transaction:
            with self.transaction() as tx:
                operation_func(entities, continue_on_error, results, tx)
                if results['failed'] == 0:
                    tx.commit()
        else:
            operation_func(entities, continue_on_error, results, None)
        
        return results
    
    def _batch_create_entities_impl(self,
                                   entities: List[Dict[str, Any]],
                                   continue_on_error: bool,
                                   results: Dict[str, Any],
                                   transaction: Optional[TransactionContext]):
        """Implementation of batch entity creation."""
        for entity_spec in entities:
            try:
                entity_id = entity_spec.get('entity_id')
                entity_type = entity_spec.get('entity_type', 'entity')
                properties = entity_spec.get('properties', {})
                level_id = entity_spec.get('level_id')
                
                if not entity_id:
                    raise ValueError("entity_id is required")
                
                success = self.create_entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    properties=properties,
                    level_id=level_id,
                    transaction=transaction
                )
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to create entity {entity_id}")
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error creating entity: {e}")
                
                if not continue_on_error:
                    break
    
    def batch_update_entities(self,
                            updates: List[Dict[str, Any]],
                            continue_on_error: bool = True,
                            use_transaction: bool = True) -> Dict[str, Any]:
        """
        Update multiple entities in a batch operation.
        
        Args:
            updates: List of update specifications
            continue_on_error: Whether to continue if individual updates fail
            use_transaction: Whether to wrap in transaction
            
        Returns:
            Results dictionary with success/failure counts
        """
        results = {
            'total': len(updates),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        operation_func = self._batch_update_entities_impl
        
        if use_transaction:
            with self.transaction() as tx:
                operation_func(updates, continue_on_error, results, tx)
                if results['failed'] == 0:
                    tx.commit()
        else:
            operation_func(updates, continue_on_error, results, None)
        
        return results
    
    def _batch_update_entities_impl(self,
                                   updates: List[Dict[str, Any]],
                                   continue_on_error: bool,
                                   results: Dict[str, Any],
                                   transaction: Optional[TransactionContext]):
        """Implementation of batch entity updates."""
        for update_spec in updates:
            try:
                entity_id = update_spec.get('entity_id')
                if not entity_id:
                    raise ValueError("entity_id is required")
                
                success = self.update_entity(
                    entity_id=entity_id,
                    properties=update_spec.get('properties'),
                    entity_type=update_spec.get('entity_type'),
                    new_level_id=update_spec.get('new_level_id'),
                    merge_properties=update_spec.get('merge_properties', True),
                    transaction=transaction
                )
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to update entity {entity_id}")
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error updating entity: {e}")
                
                if not continue_on_error:
                    break
    
    def batch_delete_entities(self,
                            entity_ids: List[str],
                            cascade: bool = False,
                            continue_on_error: bool = True,
                            use_transaction: bool = True) -> Dict[str, Any]:
        """
        Delete multiple entities in a batch operation.
        
        Args:
            entity_ids: List of entity IDs to delete
            cascade: Whether to cascade deletes
            continue_on_error: Whether to continue if individual deletes fail
            use_transaction: Whether to wrap in transaction
            
        Returns:
            Results dictionary with success/failure counts
        """
        results = {
            'total': len(entity_ids),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        operation_func = self._batch_delete_entities_impl
        
        if use_transaction:
            with self.transaction() as tx:
                operation_func(entity_ids, cascade, continue_on_error, results, tx)
                if results['failed'] == 0:
                    tx.commit()
        else:
            operation_func(entity_ids, cascade, continue_on_error, results, None)
        
        return results
    
    def _batch_delete_entities_impl(self,
                                   entity_ids: List[str],
                                   cascade: bool,
                                   continue_on_error: bool,
                                   results: Dict[str, Any],
                                   transaction: Optional[TransactionContext]):
        """Implementation of batch entity deletion."""
        for entity_id in entity_ids:
            try:
                success = self.delete_entity(
                    entity_id=entity_id,
                    cascade=cascade,
                    transaction=transaction
                )
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to delete entity {entity_id}")
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error deleting entity {entity_id}: {e}")
                
                if not continue_on_error:
                    break
    
    # =====================================================================
    # TRANSACTION SUPPORT
    # =====================================================================
    
    @contextmanager
    def transaction(self) -> ContextManager[TransactionContext]:
        """
        Create a transaction context for atomic operations.
        
        Usage:
            with core_ops.transaction() as tx:
                core_ops.create_entity(..., transaction=tx)
                core_ops.update_entity(..., transaction=tx)
                tx.commit()  # Explicit commit required
        
        Returns:
            Transaction context manager
        """
        tx = TransactionContext(self.hkg)
        try:
            yield tx
        except Exception:
            tx.rollback()
            raise
    
    # =====================================================================
    # DATA INTEGRITY AND VALIDATION
    # =====================================================================
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of hierarchical data.
        
        Returns:
            Validation report with any issues found
        """
        issues = {
            'orphaned_entities': [],
            'missing_level_assignments': [],
            'invalid_cross_level_relationships': [],
            'inconsistent_level_graphs': []
        }
        
        # Check for entities without level assignments
        all_entities = set(self.hkg.nodes())
        assigned_entities = set(self.hkg.entity_levels.keys())
        orphaned = all_entities - assigned_entities
        issues['orphaned_entities'] = list(orphaned)
        
        # Check for invalid level assignments
        for entity_id, level_id in self.hkg.entity_levels.items():
            if level_id not in self.hkg.levels:
                issues['missing_level_assignments'].append({
                    'entity_id': entity_id,
                    'invalid_level': level_id
                })
        
        # Validate cross-level relationships
        for rel in self.hkg.cross_level_relationships:
            source_entity = rel.get('source_entity')
            target_entity = rel.get('target_entity')
            
            if not self.hkg.has_node(source_entity) or not self.hkg.has_node(target_entity):
                issues['invalid_cross_level_relationships'].append({
                    'relationship': rel,
                    'issue': 'Referenced entities do not exist'
                })
        
        # Check consistency between main graph and level graphs
        for level_id, level_graph in self.hkg.level_graphs.items():
            level_entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            
            if hasattr(level_graph, 'get_all_entities'):
                graph_entities = set(level_graph.get_all_entities())
                expected_entities = set(level_entities)
                
                if graph_entities != expected_entities:
                    issues['inconsistent_level_graphs'].append({
                        'level_id': level_id,
                        'expected': list(expected_entities),
                        'actual': list(graph_entities),
                        'missing': list(expected_entities - graph_entities),
                        'extra': list(graph_entities - expected_entities)
                    })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_issues': sum(len(issue_list) for issue_list in issues.values()),
            'issues': issues,
            'is_valid': all(len(issue_list) == 0 for issue_list in issues.values())
        }
    
    def repair_data_integrity(self, validation_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Attempt to repair data integrity issues.
        
        Args:
            validation_report: Optional pre-computed validation report
            
        Returns:
            Repair results
        """
        if validation_report is None:
            validation_report = self.validate_data_integrity()
        
        repair_results = {
            'timestamp': datetime.now().isoformat(),
            'repairs_attempted': 0,
            'repairs_successful': 0,
            'repairs_failed': 0,
            'repair_actions': []
        }
        
        issues = validation_report.get('issues', {})
        
        # Assign orphaned entities to a default level
        orphaned_entities = issues.get('orphaned_entities', [])
        if orphaned_entities:
            # Create default level if it doesn't exist
            default_level = 'unassigned'
            if default_level not in self.hkg.levels:
                self.hkg.create_level(default_level, "Unassigned Entities", level_order=-1)
            
            for entity_id in orphaned_entities:
                try:
                    self.hkg.entity_levels[entity_id] = default_level
                    repair_results['repairs_successful'] += 1
                    repair_results['repair_actions'].append(f"Assigned {entity_id} to {default_level} level")
                except Exception as e:
                    repair_results['repairs_failed'] += 1
                    repair_results['repair_actions'].append(f"Failed to assign {entity_id}: {e}")
                repair_results['repairs_attempted'] += 1
        
        # Remove invalid cross-level relationships
        invalid_relationships = issues.get('invalid_cross_level_relationships', [])
        for invalid_rel in invalid_relationships:
            try:
                self.hkg.cross_level_relationships.remove(invalid_rel['relationship'])
                repair_results['repairs_successful'] += 1
                repair_results['repair_actions'].append(f"Removed invalid cross-level relationship")
            except Exception as e:
                repair_results['repairs_failed'] += 1
                repair_results['repair_actions'].append(f"Failed to remove invalid relationship: {e}")
            repair_results['repairs_attempted'] += 1
        
        # Sync level graphs
        inconsistent_graphs = issues.get('inconsistent_level_graphs', [])
        for graph_issue in inconsistent_graphs:
            level_id = graph_issue['level_id']
            try:
                # Rebuild level graph from entity assignments
                if level_id in self.hkg.level_graphs:
                    # Clear and rebuild
                    self.hkg.level_graphs[level_id].clear()
                    
                    for entity_id in graph_issue['expected']:
                        entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
                        if entity_data:
                            self.hkg.level_graphs[level_id].add_entity(
                                entity_id,
                                entity_data.get('type', 'entity'),
                                entity_data.get('properties', {})
                            )
                
                repair_results['repairs_successful'] += 1
                repair_results['repair_actions'].append(f"Rebuilt level graph for {level_id}")
            except Exception as e:
                repair_results['repairs_failed'] += 1
                repair_results['repair_actions'].append(f"Failed to rebuild level graph {level_id}: {e}")
            repair_results['repairs_attempted'] += 1
        
        return repair_results
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive entity statistics."""
        stats = {
            'total_entities': self.hkg.num_nodes(),
            'total_relationships': self.hkg.num_edges(),
            'cross_level_relationships': len(self.hkg.cross_level_relationships),
            'entities_by_level': {},
            'entities_by_type': defaultdict(int),
            'orphaned_entities': 0
        }
        
        # Count entities by level
        for level_id in self.hkg.levels.keys():
            level_entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            stats['entities_by_level'][level_id] = len(level_entities)
        
        # Count entities by type and find orphaned entities
        all_entities = set(self.hkg.nodes())
        assigned_entities = set(self.hkg.entity_levels.keys())
        stats['orphaned_entities'] = len(all_entities - assigned_entities)
        
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if entity_data:
                entity_type = entity_data.get('type', 'unknown')
                stats['entities_by_type'][entity_type] += 1
        
        stats['entities_by_type'] = dict(stats['entities_by_type'])
        
        return stats
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage and clean up unused data."""
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'space_saved': 0,
            'performance_improvements': []
        }
        
        # Clean up empty levels
        empty_levels = []
        for level_id in list(self.hkg.levels.keys()):
            entities_in_level = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            if len(entities_in_level) == 0:
                empty_levels.append(level_id)
        
        for level_id in empty_levels:
            del self.hkg.levels[level_id]
            if level_id in self.hkg.level_graphs:
                del self.hkg.level_graphs[level_id]
            if level_id in self.hkg.level_order:
                del self.hkg.level_order[level_id]
            
            optimization_results['actions_taken'].append(f"Removed empty level: {level_id}")
        
        # Remove duplicate cross-level relationships
        seen_relationships = set()
        unique_relationships = []
        duplicates_removed = 0
        
        for rel in self.hkg.cross_level_relationships:
            rel_signature = (
                rel.get('source_entity'),
                rel.get('target_entity'),
                rel.get('relationship_type')
            )
            
            if rel_signature not in seen_relationships:
                seen_relationships.add(rel_signature)
                unique_relationships.append(rel)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            self.hkg.cross_level_relationships = unique_relationships
            optimization_results['actions_taken'].append(f"Removed {duplicates_removed} duplicate relationships")
        
        return optimization_results