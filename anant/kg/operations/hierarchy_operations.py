"""
Hierarchy Management Operations
==============================

Operations for managing hierarchical levels in the knowledge graph including:
- Level creation and management
- Entity assignment to levels
- Level metadata and ordering
- Level statistics and validation
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class HierarchyOperations:
    """
    Handles hierarchy management operations for hierarchical knowledge graphs.
    
    This class provides methods for creating and managing hierarchical levels,
    assigning entities to levels, and maintaining hierarchy metadata.
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize hierarchy operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
        self.logger = logger.getChild(self.__class__.__name__)
    
    def create_level(self, 
                    level_id: str,
                    level_name: str, 
                    level_description: str = "",
                    level_order: int = 0) -> bool:
        """
        Create a new hierarchical level.
        
        Args:
            level_id: Unique identifier for the level
            level_name: Human-readable name
            level_description: Description of what this level represents
            level_order: Order in hierarchy (0=top, 1=next down, etc.)
            
        Returns:
            Success status
        """
        if level_id in self.hkg.levels:
            self.logger.warning(f"Level {level_id} already exists")
            return False
        
        # Create level metadata
        level_metadata = {
            'level_id': level_id,
            'level_name': level_name,
            'level_description': level_description,
            'level_order': level_order,
            'created_at': datetime.now().isoformat(),
            'node_count': 0
        }
        
        # Store level metadata and create associated knowledge graph
        self.hkg.levels[level_id] = level_metadata
        self.hkg.level_order[level_id] = level_order
        
        # Import here to avoid circular imports
        from ..core import KnowledgeGraph
        self.hkg.level_graphs[level_id] = KnowledgeGraph()
        
        self.logger.info(f"Created level '{level_id}' with order {level_order}")
        return True
    
    def add_level(self, level_id: str, level_name: str, level_description: str = "", level_order: int = 0) -> bool:
        """Alias for create_level for backward compatibility."""
        return self.create_level(level_id, level_name, level_description, level_order)
    
    def add_node_to_level(self,
                           node_id: str,
                           node_type: str,
                           properties: Dict[str, Any],
                           level_id: str) -> bool:
        """
        Add a node to a specific hierarchical level.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type/category of the node
            properties: Node properties dictionary
            level_id: Target level identifier
            
        Returns:
            Success status
        """
        if level_id not in self.hkg.levels:
            self.logger.error(f"Level {level_id} does not exist")
            return False
        
        # Add node to the main knowledge graph
        node_data = properties.copy()
        node_data['node_type'] = node_type
        
        success = self.hkg.knowledge_graph.add_node(
            node_id=node_id,
            data=node_data,
            node_type=node_type
        )
        
        if success:
            # Track node-level association
            self.hkg.node_levels[node_id] = level_id
            
            # Add to level-specific graph
            level_graph = self.hkg.level_graphs[level_id]
            level_node_data = properties.copy()
            level_node_data['node_type'] = node_type
            level_graph.add_node(node_id, level_node_data, node_type=node_type)
            
            # Update level statistics
            self.hkg.levels[level_id]['node_count'] += 1
            
            self.logger.info(f"Added node '{node_id}' to level '{level_id}'")
        
        return success
    
    def get_nodes_at_level(self, level_id: str) -> List[str]:
        """
        Get all nodes at a specific level.
        
        Args:
            level_id: Level identifier
            
        Returns:
            List of node identifiers at the level
        """
        if level_id not in self.hkg.levels:
            self.logger.warning(f"Level {level_id} does not exist")
            return []
        
        return [node_id for node_id, node_level in self.hkg.node_levels.items() 
                if node_level == level_id]
    
    def get_node_level(self, node_id: str) -> Optional[str]:
        """Get the level of a specific node."""
        return self.hkg.node_levels.get(node_id)
    
    def get_level_metadata(self, level_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific level."""
        return self.hkg.levels.get(level_id)
    
    def get_all_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get all levels and their metadata."""
        return self.hkg.levels.copy()
    
    def get_ordered_levels(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get levels ordered by their hierarchy order.
        
        Returns:
            List of (level_id, metadata) tuples ordered by level_order
        """
        return sorted(
            [(level_id, metadata) for level_id, metadata in self.hkg.levels.items()],
            key=lambda x: x[1]['level_order']
        )
    
    def remove_level(self, level_id: str, force: bool = False) -> bool:
        """
        Remove a level and optionally its entities.
        
        Args:
            level_id: Level to remove
            force: If True, remove entities; if False, fail if entities exist
            
        Returns:
            Success status
        """
        if level_id not in self.hkg.levels:
            self.logger.warning(f"Level {level_id} does not exist")
            return False
        
        nodes_at_level = self.get_nodes_at_level(level_id)
        
        if nodes_at_level and not force:
            self.logger.error(f"Cannot remove level {level_id}: contains {len(nodes_at_level)} nodes")
            return False
        
        # Remove nodes if force=True
        if nodes_at_level and force:
            for node_id in nodes_at_level:
                self.hkg.core_ops.remove_node(node_id)
        
        # Remove level metadata and graph
        del self.hkg.levels[level_id]
        del self.hkg.level_order[level_id]
        del self.hkg.level_graphs[level_id]
        
        self.logger.info(f"Removed level '{level_id}'")
        return True
    
    def update_level_metadata(self, level_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update metadata for a level.
        
        Args:
            level_id: Level to update
            updates: Dictionary of metadata updates
            
        Returns:
            Success status
        """
        if level_id not in self.hkg.levels:
            self.logger.error(f"Level {level_id} does not exist")
            return False
        
        # Update metadata
        self.hkg.levels[level_id].update(updates)
        self.hkg.levels[level_id]['updated_at'] = datetime.now().isoformat()
        
        # Update level order if provided
        if 'level_order' in updates:
            self.hkg.level_order[level_id] = updates['level_order']
        
        self.logger.info(f"Updated metadata for level '{level_id}'")
        return True
    
    def reorder_levels(self, level_order_map: Dict[str, int]) -> bool:
        """
        Reorder multiple levels at once.
        
        Args:
            level_order_map: Mapping of level_id -> new_order
            
        Returns:
            Success status
        """
        # Validate all levels exist
        for level_id in level_order_map:
            if level_id not in self.hkg.levels:
                self.logger.error(f"Level {level_id} does not exist")
                return False
        
        # Update orders
        for level_id, new_order in level_order_map.items():
            self.hkg.level_order[level_id] = new_order
            self.hkg.levels[level_id]['level_order'] = new_order
            self.hkg.levels[level_id]['updated_at'] = datetime.now().isoformat()
        
        self.logger.info(f"Reordered {len(level_order_map)} levels")
        return True
    
    def validate_hierarchy_consistency(self) -> Dict[str, Any]:
        """
        Validate the consistency of the hierarchy structure.
        
        Returns:
            Validation report with issues found
        """
        issues = []
        
        # Check for duplicate level orders
        order_counts = {}
        for level_id, order in self.hkg.level_order.items():
            if order not in order_counts:
                order_counts[order] = []
            order_counts[order].append(level_id)
        
        for order, levels in order_counts.items():
            if len(levels) > 1:
                issues.append({
                    'type': 'duplicate_level_order',
                    'order': order,
                    'levels': levels
                })
        
        # Check for entities without levels
        entities_without_levels = []
        for entity_id in self.hkg.knowledge_graph.get_all_entities():
            if entity_id not in self.hkg.node_levels:
                entities_without_levels.append(entity_id)
        
        if entities_without_levels:
            issues.append({
                'type': 'entities_without_levels',
                'entities': entities_without_levels
            })
        
        # Check for missing level graphs
        missing_level_graphs = []
        for level_id in self.hkg.levels:
            if level_id not in self.hkg.level_graphs:
                missing_level_graphs.append(level_id)
        
        if missing_level_graphs:
            issues.append({
                'type': 'missing_level_graphs',
                'levels': missing_level_graphs
            })
        
        return {
            'is_valid': len(issues) == 0,
            'issue_count': len(issues),
            'issues': issues
        }
    
    def get_level_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about hierarchy levels.
        
        Returns:
            Dictionary containing hierarchy statistics
        """
        if not self.hkg.levels:
            return {'total_levels': 0}
        
        # Basic counts
        total_levels = len(self.hkg.levels)
        total_nodes_in_levels = len(self.hkg.node_levels)
        
        # Node distribution across levels
        level_node_counts = {}
        for level_id in self.hkg.levels:
            level_node_counts[level_id] = len(self.get_nodes_at_level(level_id))
        
        # Level order statistics
        orders = list(self.hkg.level_order.values())
        min_order = min(orders) if orders else 0
        max_order = max(orders) if orders else 0
        
        return {
            'total_levels': total_levels,
            'total_nodes_in_levels': total_nodes_in_levels,
            'node_distribution': level_node_counts,
            'level_depth': max_order - min_order + 1,
            'min_order': min_order,
            'max_order': max_order,
            'average_nodes_per_level': total_nodes_in_levels / total_levels if total_levels > 0 else 0
        }