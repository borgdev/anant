"""
Navigation Operations
====================

Operations for navigating hierarchical relationships including:
- Parent/child navigation  
- Ancestor/descendant traversal
- Cross-level navigation
- Path finding within hierarchy
"""

from typing import Dict, List, Optional, Any, Set
import logging
from collections import deque

logger = logging.getLogger(__name__)


class NavigationOperations:
    """
    Handles navigation operations for hierarchical knowledge graphs.
    
    This class provides methods for traversing hierarchical relationships
    and finding paths between entities across different levels.
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize navigation operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
        self.logger = logger.getChild(self.__class__.__name__)
    
    def navigate_up(self, entity_id: str) -> List[str]:
        """
        Navigate up the hierarchy from an entity.
        
        Args:
            entity_id: Starting entity identifier
            
        Returns:
            List of entities at higher levels (closer to root)
        """
        if entity_id not in self.hkg.node_levels:
            self.logger.warning(f"Entity {entity_id} not found in hierarchy")
            return []
        
        current_level = self.hkg.node_levels[entity_id]
        current_order = self.hkg.level_order.get(current_level, 0)
        
        # Find entities at higher levels (lower order numbers)
        higher_entities = []
        for entity, level in self.hkg.node_levels.items():
            level_order = self.hkg.level_order.get(level, 0)
            if level_order < current_order:
                higher_entities.append(entity)
        
        return higher_entities
    
    def navigate_down(self, entity_id: str) -> List[str]:
        """
        Navigate down the hierarchy from an entity.
        
        Args:
            entity_id: Starting entity identifier
            
        Returns:
            List of entities at lower levels (further from root)
        """
        if entity_id not in self.hkg.node_levels:
            self.logger.warning(f"Entity {entity_id} not found in hierarchy")
            return []
        
        current_level = self.hkg.node_levels[entity_id]
        current_order = self.hkg.level_order.get(current_level, 0)
        
        # Find entities at lower levels (higher order numbers)
        lower_entities = []
        for entity, level in self.hkg.node_levels.items():
            level_order = self.hkg.level_order.get(level, 0)
            if level_order > current_order:
                lower_entities.append(entity)
        
        return lower_entities
    
    def get_parent(self, entity_id: str) -> Optional[str]:
        """
        Get the direct parent of an entity in the hierarchy.
        
        This looks for explicit parent-child relationships stored in
        cross-level relationships.
        
        Args:
            entity_id: Entity to find parent for
            
        Returns:
            Parent entity ID or None
        """
        # Look for parent relationship in cross-level relationships
        for relationship in self.hkg.cross_level_relationships:
            if (relationship.get('target_entity') == entity_id and 
                relationship.get('relationship_type') == 'parent_of'):
                return relationship.get('source_entity')
            elif (relationship.get('source_entity') == entity_id and 
                  relationship.get('relationship_type') == 'child_of'):
                return relationship.get('target_entity')
        
        # If no explicit parent relationship, try to infer from levels
        current_level = self.hkg.node_levels.get(entity_id)
        if current_level is None:
            return None
        
        current_order = self.hkg.level_order.get(current_level, 0)
        
        # Look for entities one level up that might be parents
        for entity, level in self.hkg.node_levels.items():
            level_order = self.hkg.level_order.get(level, 0)
            if level_order == current_order - 1:
                # Check if there's a semantic relationship
                if self.hkg.knowledge_graph.has_relationship(entity, entity_id):
                    return entity
        
        return None
    
    def get_children(self, entity_id: str) -> List[str]:
        """
        Get the direct children of an entity in the hierarchy.
        
        Args:
            entity_id: Entity to find children for
            
        Returns:
            List of child entity IDs
        """
        children = []
        
        # Look for child relationships in cross-level relationships
        for relationship in self.hkg.cross_level_relationships:
            if (relationship.get('source_entity') == entity_id and 
                relationship.get('relationship_type') == 'parent_of'):
                children.append(relationship.get('target_entity'))
            elif (relationship.get('target_entity') == entity_id and 
                  relationship.get('relationship_type') == 'child_of'):
                children.append(relationship.get('source_entity'))
        
        # If no explicit relationships, infer from levels
        if not children:
            current_level = self.hkg.node_levels.get(entity_id)
            if current_level is None:
                return []
            
            current_order = self.hkg.level_order.get(current_level, 0)
            
            # Look for entities one level down
            for entity, level in self.hkg.node_levels.items():
                level_order = self.hkg.level_order.get(level, 0)
                if level_order == current_order + 1:
                    # Check if there's a semantic relationship
                    if self.hkg.knowledge_graph.has_relationship(entity_id, entity):
                        children.append(entity)
        
        return children
    
    def get_ancestors(self, entity_id: str) -> List[str]:
        """
        Get all ancestors of an entity (recursive parent traversal).
        
        Args:
            entity_id: Starting entity
            
        Returns:
            List of ancestor entity IDs
        """
        ancestors = []
        visited = set()
        queue = [entity_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            parent = self.get_parent(current)
            if parent and parent not in visited:
                ancestors.append(parent)
                queue.append(parent)
        
        return ancestors
    
    def get_descendants(self, entity_id: str) -> List[str]:
        """
        Get all descendants of an entity (recursive child traversal).
        
        Args:
            entity_id: Starting entity
            
        Returns:
            List of descendant entity IDs
        """
        descendants = []
        visited = set()
        queue = [entity_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            children = self.get_children(current)
            for child in children:
                if child not in visited:
                    descendants.append(child)
                    queue.append(child)
        
        return descendants
    
    def find_path_between_entities(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Find a path between two entities in the hierarchy.
        
        Args:
            source_id: Source entity
            target_id: Target entity
            
        Returns:
            Path as list of entity IDs, or None if no path exists
        """
        if source_id == target_id:
            return [source_id]
        
        # Use BFS to find shortest path
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            # Get all connected entities (parents, children, and same-level relationships)
            connected = set()
            
            # Add parent and children
            parent = self.get_parent(current)
            if parent:
                connected.add(parent)
            connected.update(self.get_children(current))
            
            # Add same-level related entities
            current_level = self.hkg.node_levels.get(current)
            if current_level and current_level in self.hkg.level_graphs:
                level_graph = self.hkg.level_graphs[current_level]
                related = level_graph.get_related_entities(current)
                connected.update(related)
            
            for neighbor in connected:
                if neighbor == target_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def get_entity_depth(self, entity_id: str) -> int:
        """
        Get the depth of an entity in the hierarchy.
        
        Args:
            entity_id: Entity to get depth for
            
        Returns:
            Depth (0 = root level, higher = deeper)
        """
        level = self.hkg.node_levels.get(entity_id)
        if level is None:
            return -1
        
        return self.hkg.level_order.get(level, 0)
    
    def get_root_entities(self) -> List[str]:
        """
        Get all entities at the root level (minimum level order).
        
        Returns:
            List of root entity IDs
        """
        if not self.hkg.level_order:
            return []
        
        min_order = min(self.hkg.level_order.values())
        root_entities = []
        
        for entity_id, level in self.hkg.node_levels.items():
            if self.hkg.level_order.get(level) == min_order:
                root_entities.append(entity_id)
        
        return root_entities
    
    def get_leaf_entities(self) -> List[str]:
        """
        Get all entities at leaf levels (entities with no children).
        
        Returns:
            List of leaf entity IDs
        """
        leaf_entities = []
        
        for entity_id in self.hkg.node_levels:
            if not self.get_children(entity_id):
                leaf_entities.append(entity_id)
        
        return leaf_entities
    
    def get_sibling_entities(self, entity_id: str) -> List[str]:
        """
        Get all sibling entities (same parent or same level).
        
        Args:
            entity_id: Entity to find siblings for
            
        Returns:
            List of sibling entity IDs
        """
        siblings = []
        
        # Method 1: Same parent
        parent = self.get_parent(entity_id)
        if parent:
            parent_children = self.get_children(parent)
            siblings.extend([child for child in parent_children if child != entity_id])
        
        # Method 2: Same level (if no parent found)
        if not siblings:
            entity_level = self.hkg.node_levels.get(entity_id)
            if entity_level:
                level_entities = self.hkg.hierarchy_ops.get_entities_at_level(entity_level)
                siblings.extend([e for e in level_entities if e != entity_id])
        
        return list(set(siblings))  # Remove duplicates
    
    def get_subtree(self, root_entity_id: str) -> Dict[str, Any]:
        """
        Get the complete subtree rooted at a given entity.
        
        Args:
            root_entity_id: Root of the subtree
            
        Returns:
            Tree structure as nested dictionary
        """
        def build_subtree(entity_id: str, visited: Set[str]) -> Dict[str, Any]:
            if entity_id in visited:
                return {'entity_id': entity_id, 'circular_reference': True}
            
            visited.add(entity_id)
            
            children = self.get_children(entity_id)
            subtree = {
                'entity_id': entity_id,
                'level': self.hkg.node_levels.get(entity_id, 'unknown'),
                'depth': self.get_entity_depth(entity_id),
                'children': [build_subtree(child, visited.copy()) for child in children]
            }
            
            return subtree
        
        return build_subtree(root_entity_id, set())
    
    def get_entity_hierarchy_position(self, entity_id: str) -> Dict[str, Any]:
        """
        Get comprehensive position information for an entity in the hierarchy.
        
        Args:
            entity_id: Entity to analyze
            
        Returns:
            Dictionary with position information
        """
        return {
            'entity_id': entity_id,
            'level': self.hkg.node_levels.get(entity_id),
            'depth': self.get_entity_depth(entity_id),
            'parent': self.get_parent(entity_id),
            'children': self.get_children(entity_id),
            'siblings': self.get_sibling_entities(entity_id),
            'ancestors': self.get_ancestors(entity_id),
            'descendants': self.get_descendants(entity_id),
            'is_root': entity_id in self.get_root_entities(),
            'is_leaf': entity_id in self.get_leaf_entities()
        }