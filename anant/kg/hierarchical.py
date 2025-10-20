"""
Hierarchical Knowledge Graph - Integration Layer
===============================================

Combines ANANT's KnowledgeGraph (semantic hypergraph) with Metagraph's
hierarchical capabilities to provide multi-level knowledge representation
for complex domains.

This integration gives you:
- Semantic reasoning from KnowledgeGraph
- Hierarchical navigation from Metagraph
- Multi-level graph structures for complex domains
- Enterprise-grade metadata management
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging

from .core import KnowledgeGraph
from ..classes.hypergraph import Hypergraph

logger = logging.getLogger(__name__)


class HierarchicalKnowledgeGraph:
    """
    Multi-level knowledge graph with hierarchical organization.
    
    Combines semantic hypergraph capabilities with hierarchical navigation
    to represent complex domains with multiple levels of abstraction.
    
    Architecture:
    - Base Layer: KnowledgeGraph for semantic reasoning
    - Hierarchy Layer: Multi-level organization of knowledge domains
    - Integration Layer: Cross-level relationships and navigation
    
    Use Cases:
    - Enterprise knowledge modeling with organizational hierarchies
    - Domain-specific knowledge with multiple abstraction levels
    - Complex system modeling with sub-systems and components
    - Research knowledge organization with fields, topics, subtopics
    """
    
    def __init__(self, 
                 name: str = "HierarchicalKG",
                 enable_semantic_reasoning: bool = True,
                 enable_temporal_tracking: bool = False):
        """
        Initialize hierarchical knowledge graph.
        
        Args:
            name: Name identifier for this knowledge graph
            enable_semantic_reasoning: Enable semantic reasoning capabilities
            enable_temporal_tracking: Enable temporal relationship tracking
        """
        
        self.name = name
        self.enable_semantic_reasoning = enable_semantic_reasoning
        self.enable_temporal_tracking = enable_temporal_tracking
        
        # Core knowledge graph for semantic reasoning
        self.knowledge_graph = KnowledgeGraph()
        
        # Hierarchical structure management
        self.levels = {}  # level_id -> level_metadata
        self.level_graphs = {}  # level_id -> KnowledgeGraph
        self.cross_level_relationships = []  # relationships across levels
        
        # Metadata for hierarchy management
        self.entity_levels = {}  # entity_id -> level_id
        self.level_order = {}  # level_id -> order (0=top, 1=next, etc.)
        
        logger.info(f"Initialized HierarchicalKnowledgeGraph: {name}")
    
    # ===== HIERARCHY MANAGEMENT =====
    
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
        
        if level_id in self.levels:
            logger.warning(f"Level {level_id} already exists")
            return False
        
        # Create level metadata
        self.levels[level_id] = {
            "level_id": level_id,
            "level_name": level_name,
            "level_description": level_description,
            "level_order": level_order,
            "entity_count": 0,
            "created_at": str(datetime.now())
        }
        
        # Create knowledge graph for this level
        self.level_graphs[level_id] = KnowledgeGraph()
        self.level_order[level_id] = level_order
        
        logger.info(f"Created level: {level_name} (order: {level_order})")
        return True
    
    def add_entity_to_level(self,
                          entity_id: str,
                          level_id: str,
                          entity_type: str = "Entity",
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an entity to a specific hierarchical level.
        
        Args:
            entity_id: Unique entity identifier
            level_id: Target level for the entity
            entity_type: Type/category of the entity
            properties: Additional entity properties
            
        Returns:
            Success status
        """
        
        if level_id not in self.levels:
            logger.error(f"Level {level_id} does not exist")
            return False
        
        # Add to main knowledge graph
        self.knowledge_graph.add_node(entity_id)
        
        # Add to level-specific graph
        level_kg = self.level_graphs[level_id]
        level_kg.add_node(entity_id)
        
        # Track level assignment
        self.entity_levels[entity_id] = level_id
        self.levels[level_id]["entity_count"] += 1
        
        logger.info(f"Added entity {entity_id} to level {level_id}")
        return True
    
    def add_relationship(self,
                        source_entity: str,
                        target_entity: str,
                        relationship_type: str,
                        properties: Optional[Dict[str, Any]] = None,
                        cross_level: bool = False) -> bool:
        """
        Add a relationship between entities.
        
        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID  
            relationship_type: Type of relationship
            properties: Additional relationship properties
            cross_level: Whether this is a cross-level relationship
            
        Returns:
            Success status
        """
        
        # Add to main knowledge graph
        relationship_id = f"{source_entity}_{relationship_type}_{target_entity}"
        
        # Ensure nodes exist
        self.knowledge_graph.add_node(source_entity)
        self.knowledge_graph.add_node(target_entity)
        
        # Add edge
        success = True
        try:
            self.knowledge_graph.add_edge(relationship_id, [source_entity, target_entity], properties=properties)
        except Exception as e:
            logger.warning(f"Could not add edge: {e}")
            success = False
        
        if not success:
            return False
        
        # Handle cross-level relationships
        source_level = self.entity_levels.get(source_entity)
        target_level = self.entity_levels.get(target_entity)
        
        if source_level != target_level:
            cross_level = True
        
        if cross_level:
            self.cross_level_relationships.append({
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relationship_type": relationship_type,
                "source_level": source_level,
                "target_level": target_level,
                "properties": properties or {}
            })
        else:
            # Add to level-specific graph if same level
            if source_level and source_level in self.level_graphs:
                level_kg = self.level_graphs[source_level]
                level_kg.add_node(source_entity)
                level_kg.add_node(target_entity)
                level_kg.add_edge(relationship_id, [source_entity, target_entity], properties=properties)
        
        logger.info(f"Added relationship: {source_entity} --{relationship_type}--> {target_entity}")
        return True
    
    # ===== HIERARCHICAL NAVIGATION =====
    
    def get_entities_at_level(self, level_id: str) -> List[str]:
        """Get all entities at a specific level."""
        if level_id not in self.level_graphs:
            return []
        
        level_kg = self.level_graphs[level_id]
        return list(level_kg.nodes)
    
    def get_entity_level(self, entity_id: str) -> Optional[str]:
        """Get the level of an entity."""
        return self.entity_levels.get(entity_id)
    
    def get_level_metadata(self, level_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a level."""
        return self.levels.get(level_id)
    
    def get_cross_level_relationships(self, 
                                    from_level: Optional[str] = None,
                                    to_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cross-level relationships, optionally filtered by levels."""
        
        relationships = self.cross_level_relationships
        
        if from_level:
            relationships = [r for r in relationships if r["source_level"] == from_level]
        
        if to_level:
            relationships = [r for r in relationships if r["target_level"] == to_level]
        
        return relationships
    
    def navigate_up(self, entity_id: str) -> List[str]:
        """Find entities at higher levels related to this entity."""
        entity_level = self.get_entity_level(entity_id)
        if not entity_level:
            return []
        
        current_order = self.level_order[entity_level]
        higher_levels = [
            level_id for level_id, order in self.level_order.items() 
            if order < current_order
        ]
        
        related_entities = []
        for rel in self.cross_level_relationships:
            if (rel["source_entity"] == entity_id and 
                rel["target_level"] in higher_levels):
                related_entities.append(rel["target_entity"])
            elif (rel["target_entity"] == entity_id and 
                  rel["source_level"] in higher_levels):
                related_entities.append(rel["source_entity"])
        
        return related_entities
    
    def navigate_down(self, entity_id: str) -> List[str]:
        """Find entities at lower levels related to this entity."""
        entity_level = self.get_entity_level(entity_id)
        if not entity_level:
            return []
        
        current_order = self.level_order[entity_level]
        lower_levels = [
            level_id for level_id, order in self.level_order.items() 
            if order > current_order
        ]
        
        related_entities = []
        for rel in self.cross_level_relationships:
            if (rel["source_entity"] == entity_id and 
                rel["target_level"] in lower_levels):
                related_entities.append(rel["target_entity"])
            elif (rel["target_entity"] == entity_id and 
                  rel["source_level"] in lower_levels):
                related_entities.append(rel["source_entity"])
        
        return related_entities
    
    # ===== SEMANTIC OPERATIONS =====
    
    def semantic_search(self, 
                       query: str,
                       level_filter: Optional[List[str]] = None,
                       entity_types: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Perform semantic search across levels.
        
        Args:
            query: Search query
            level_filter: Limit search to specific levels
            entity_types: Filter by entity types
            **kwargs: Additional search parameters
            
        Returns:
            Search results with level information
        """
        
        # Search in main knowledge graph
        results = self.knowledge_graph.semantic_search(
            entity_type=entity_types[0] if entity_types else None,
            pattern={"query": query},
            **kwargs
        )
        
        # Add level information to results
        if "entities" in results:
            for entity_data in results["entities"]:
                entity_id = entity_data.get("entity")
                if entity_id:
                    entity_data["level"] = self.get_entity_level(entity_id)
                    entity_data["level_metadata"] = self.get_level_metadata(
                        entity_data["level"]
                    )
        
        # Filter by levels if specified
        if level_filter and "entities" in results:
            filtered_entities = [
                entity for entity in results["entities"]
                if entity.get("level") in level_filter
            ]
            results["entities"] = filtered_entities
            results["total_results"] = len(filtered_entities)
        
        return results
    
    def get_level_subgraph(self, level_id: str) -> Optional[KnowledgeGraph]:
        """Get the knowledge graph for a specific level."""
        return self.level_graphs.get(level_id)
    
    def merge_levels(self, level_ids: List[str]) -> KnowledgeGraph:
        """Create a merged knowledge graph from multiple levels."""
        merged_kg = KnowledgeGraph()
        
        for level_id in level_ids:
            if level_id in self.level_graphs:
                level_kg = self.level_graphs[level_id]
                merged_kg = merged_kg.merge_knowledge_graphs(level_kg)
        
        return merged_kg
    
    # ===== STATISTICS AND ANALYSIS =====
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hierarchy."""
        
        stats = {
            "total_levels": len(self.levels),
            "total_entities": len(self.entity_levels),
            "total_relationships": len(self.knowledge_graph.edges),
            "cross_level_relationships": len(self.cross_level_relationships),
            "levels": {}
        }
        
        # Level-specific statistics
        for level_id, level_data in self.levels.items():
            level_kg = self.level_graphs[level_id]
            
            stats["levels"][level_id] = {
                "name": level_data["level_name"],
                "order": level_data["level_order"],
                "entity_count": level_data["entity_count"],
                "internal_relationships": len(level_kg.edges),
                "entity_types": len(level_kg.get_entity_types()) if hasattr(level_kg, 'get_entity_types') else 0
            }
        
        return stats
    
    def analyze_cross_level_connectivity(self) -> Dict[str, Any]:
        """Analyze connectivity patterns across levels."""
        
        connectivity = {
            "level_pairs": {},
            "most_connected_levels": [],
            "isolated_levels": [],
            "relationship_types": {}
        }
        
        # Count relationships between level pairs
        for rel in self.cross_level_relationships:
            source_level = rel["source_level"]
            target_level = rel["target_level"]
            rel_type = rel["relationship_type"]
            
            # Count level pairs
            pair_key = f"{source_level}→{target_level}"
            connectivity["level_pairs"][pair_key] = connectivity["level_pairs"].get(pair_key, 0) + 1
            
            # Count relationship types
            connectivity["relationship_types"][rel_type] = connectivity["relationship_types"].get(rel_type, 0) + 1
        
        # Find most connected levels
        level_connections = {}
        for rel in self.cross_level_relationships:
            source_level = rel["source_level"] 
            target_level = rel["target_level"]
            
            level_connections[source_level] = level_connections.get(source_level, 0) + 1
            level_connections[target_level] = level_connections.get(target_level, 0) + 1
        
        # Sort by connection count
        sorted_levels = sorted(level_connections.items(), key=lambda x: x[1], reverse=True)
        connectivity["most_connected_levels"] = sorted_levels[:5]
        
        # Find isolated levels (no cross-level connections)
        connected_levels = set(level_connections.keys())
        all_levels = set(self.levels.keys())
        connectivity["isolated_levels"] = list(all_levels - connected_levels)
        
        return connectivity
    
    def get_parent(self, entity_id: str) -> Optional[str]:
        """
        Get the direct parent entity in the hierarchy.
        
        Args:
            entity_id: Entity to find parent for
            
        Returns:
            Parent entity ID or None if no parent or entity not found
        """
        
        if entity_id not in self.entity_levels:
            logger.warning(f"Entity {entity_id} not found in hierarchy")
            return None
        
        # Find entities one level up that are directly connected
        entity_level = self.entity_levels[entity_id]
        current_order = self.level_order.get(entity_level, 0)
        
        # Look for parent level (one level up)
        parent_level = None
        for level_id, order in self.level_order.items():
            if order == current_order - 1:
                parent_level = level_id
                break
        
        if not parent_level:
            return None  # Already at top level
        
        # Find direct parent relationships
        for rel in self.cross_level_relationships:
            # Check if this entity is the target (child) in a parent-child relationship
            if (rel["target_entity"] == entity_id and 
                rel["source_level"] == parent_level and
                rel["relationship_type"] in ["has_child", "contains", "parent_of"]):
                return rel["source_entity"]
            
            # Also check reverse relationships
            if (rel["source_entity"] == entity_id and 
                rel["target_level"] == parent_level and
                rel["relationship_type"] in ["child_of", "part_of", "belongs_to"]):
                return rel["target_entity"]
        
        # If no explicit parent relationship found, return first entity at parent level
        # that has any relationship with this entity
        for rel in self.cross_level_relationships:
            if rel["target_entity"] == entity_id and rel["source_level"] == parent_level:
                return rel["source_entity"]
            elif rel["source_entity"] == entity_id and rel["target_level"] == parent_level:
                return rel["target_entity"]
        
        return None
    
    def get_children(self, entity_id: str) -> List[str]:
        """
        Get all direct child entities in the hierarchy.
        
        Args:
            entity_id: Entity to find children for
            
        Returns:
            List of child entity IDs
        """
        
        if entity_id not in self.entity_levels:
            logger.warning(f"Entity {entity_id} not found in hierarchy")
            return []
        
        children = []
        entity_level = self.entity_levels[entity_id]
        current_order = self.level_order.get(entity_level, 0)
        
        # Look for child level (one level down)
        child_level = None
        for level_id, order in self.level_order.items():
            if order == current_order + 1:
                child_level = level_id
                break
        
        if not child_level:
            return []  # Already at bottom level or no child level
        
        # Find direct child relationships
        for rel in self.cross_level_relationships:
            # Check if this entity is the source (parent) in a parent-child relationship
            if (rel["source_entity"] == entity_id and 
                rel["target_level"] == child_level and
                rel["relationship_type"] in ["has_child", "contains", "parent_of"]):
                children.append(rel["target_entity"])
            
            # Also check reverse relationships
            elif (rel["target_entity"] == entity_id and 
                  rel["source_level"] == child_level and
                  rel["relationship_type"] in ["child_of", "part_of", "belongs_to"]):
                children.append(rel["source_entity"])
        
        # If no explicit child relationships, find all entities at child level 
        # that have any relationship with this entity
        if not children:
            for rel in self.cross_level_relationships:
                if rel["source_entity"] == entity_id and rel["target_level"] == child_level:
                    if rel["target_entity"] not in children:
                        children.append(rel["target_entity"])
                elif rel["target_entity"] == entity_id and rel["source_level"] == child_level:
                    if rel["source_entity"] not in children:
                        children.append(rel["source_entity"])
        
        return children
    
    def get_ancestors(self, entity_id: str) -> List[str]:
        """
        Get all ancestor entities in the hierarchy (all entities above this one).
        
        Args:
            entity_id: Entity to find ancestors for
            
        Returns:
            List of ancestor entity IDs, ordered from immediate parent to root
        """
        
        ancestors = []
        current_entity = entity_id
        visited = set()  # Prevent infinite loops
        
        while current_entity and current_entity not in visited:
            visited.add(current_entity)
            parent = self.get_parent(current_entity)
            
            if parent and parent != current_entity:
                ancestors.append(parent)
                current_entity = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, entity_id: str) -> List[str]:
        """
        Get all descendant entities in the hierarchy (all entities below this one).
        
        Args:
            entity_id: Entity to find descendants for
            
        Returns:
            List of descendant entity IDs
        """
        
        descendants = []
        visited = set()
        queue = [entity_id]
        
        while queue:
            current_entity = queue.pop(0)
            
            if current_entity in visited:
                continue
            
            visited.add(current_entity)
            children = self.get_children(current_entity)
            
            for child in children:
                if child not in descendants and child != entity_id:
                    descendants.append(child)
                    if child not in visited:
                        queue.append(child)
        
        return descendants
    
    def max_depth(self) -> int:
        """
        Calculate the maximum depth of the hierarchy.
        
        Returns:
            Maximum depth (number of levels from root to deepest leaf)
        """
        
        if not self.level_order:
            return 0
        
        # Maximum depth is the highest level order plus 1
        return max(self.level_order.values()) + 1
    
    def avg_branching_factor(self) -> float:
        """
        Calculate the average branching factor of the hierarchy.
        
        Returns:
            Average number of children per non-leaf entity
        """
        
        if not self.entity_levels:
            return 0.0
        
        total_children = 0
        entities_with_children = 0
        
        # Count children for each entity
        for entity_id in self.entity_levels.keys():
            children = self.get_children(entity_id)
            if children:
                total_children += len(children)
                entities_with_children += 1
        
        if entities_with_children == 0:
            return 0.0
        
        return total_children / entities_with_children
    
    # ===== CORE GRAPH INTERFACE METHODS =====
    
    def add_node(self, node_id: str, properties: Optional[Dict] = None, level_id: Optional[str] = None):
        """Add a node to the hierarchical knowledge graph"""
        if level_id:
            entity_type = properties.get('entity_type', 'Entity') if properties else 'Entity'
            self.add_entity_to_level(node_id, level_id, entity_type, properties)
        else:
            # Add to the main knowledge graph
            self.knowledge_graph.add_node(node_id, properties)
    
    def add_edge(self, edge_id: str, node_list: List[str], properties: Optional[Dict] = None, level_id: Optional[str] = None):
        """Add an edge to the hierarchical knowledge graph"""
        if level_id and level_id in self.level_graphs:
            # Add to specific level
            self.level_graphs[level_id].add_edge(edge_id, node_list, properties=properties)
        else:
            # Add to main knowledge graph
            self.knowledge_graph.add_edge(edge_id, node_list, properties=properties)
    
    def add_entity(self, entity_id: str, properties: Dict[str, Any], level_id: Optional[str] = None) -> bool:
        """Add an entity (alias for add_entity_to_level with fallback)"""
        if level_id:
            entity_type = properties.get('entity_type', 'Entity')
            return self.add_entity_to_level(entity_id, level_id, entity_type, properties)
        else:
            # Add to main knowledge graph
            self.knowledge_graph.add_entity(entity_id, properties)
            return True
    
    def add_level(self, level_id: str, level_name: str, level_description: str = "", level_order: int = 0) -> bool:
        """Add a new hierarchical level (alias for create_level)"""
        return self.create_level(level_id, level_name, level_description, level_order)
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from all levels"""
        success = False
        
        # Remove from specific level if tracked
        if entity_id in self.entity_levels:
            level_id = self.entity_levels[entity_id]
            if level_id in self.level_graphs:
                try:
                    self.level_graphs[level_id].remove_node(entity_id)
                    success = True
                except:
                    pass
            del self.entity_levels[entity_id]
        
        # Remove from main knowledge graph
        try:
            self.knowledge_graph.remove_node(entity_id)
            success = True
        except:
            pass
        
        return success
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node (alias for remove_entity)"""
        return self.remove_entity(node_id)
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from all levels"""
        success = False
        
        # Remove from all level graphs
        for level_graph in self.level_graphs.values():
            try:
                level_graph.remove_edge(edge_id)
                success = True
            except:
                pass
        
        # Remove from main knowledge graph
        try:
            self.knowledge_graph.remove_edge(edge_id)
            success = True
        except:
            pass
        
        return success
    
    @property
    def nodes(self) -> set:
        """Get all nodes across all levels"""
        all_nodes = set(self.knowledge_graph.nodes) if hasattr(self.knowledge_graph, 'nodes') else set()
        
        for level_graph in self.level_graphs.values():
            if hasattr(level_graph, 'nodes'):
                all_nodes.update(level_graph.nodes)
        
        return all_nodes
    
    @property
    def edges(self) -> set:
        """Get all edges across all levels"""
        all_edges = set(self.knowledge_graph.edges) if hasattr(self.knowledge_graph, 'edges') else set()
        
        for level_graph in self.level_graphs.values():
            if hasattr(level_graph, 'edges'):
                all_edges.update(level_graph.edges)
        
        return all_edges
    
    @property
    def num_nodes(self) -> int:
        """Get total number of nodes"""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """Get total number of edges"""
        return len(self.edges)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in any level"""
        return node_id in self.nodes
    
    def has_edge(self, edge_id: str) -> bool:
        """Check if edge exists in any level"""
        return edge_id in self.edges
    
    def clear(self):
        """Clear all data from the hierarchical knowledge graph"""
        self.levels.clear()
        self.level_graphs.clear()
        self.cross_level_relationships.clear()
        self.entity_levels.clear()
        self.level_order.clear()
        
        # Clear main knowledge graph
        if hasattr(self.knowledge_graph, 'clear'):
            self.knowledge_graph.clear()
    
    def copy(self) -> 'HierarchicalKnowledgeGraph':
        """Create a copy of the hierarchical knowledge graph"""
        new_hkg = HierarchicalKnowledgeGraph(
            name=f"copy_{self.name}",
            enable_semantic_reasoning=self.enable_semantic_reasoning,
            enable_temporal_tracking=self.enable_temporal_tracking
        )
        
        # Copy levels
        for level_id, level_meta in self.levels.items():
            new_hkg.levels[level_id] = level_meta.copy()
        
        # Copy level graphs
        for level_id, level_graph in self.level_graphs.items():
            if hasattr(level_graph, 'copy'):
                new_hkg.level_graphs[level_id] = level_graph.copy()
        
        # Copy other metadata
        new_hkg.cross_level_relationships = self.cross_level_relationships.copy()
        new_hkg.entity_levels = self.entity_levels.copy()
        new_hkg.level_order = self.level_order.copy()
        
        # Copy main knowledge graph
        if hasattr(self.knowledge_graph, 'copy'):
            new_hkg.knowledge_graph = self.knowledge_graph.copy()
        
        return new_hkg
    
    def add_cross_level_relationship(self, 
                                   relationship_id: str,
                                   source_entity: str, 
                                   target_entity: str,
                                   relationship_type: str,
                                   properties: Optional[Dict] = None) -> bool:
        """
        Add a relationship that crosses hierarchical levels
        
        Args:
            relationship_id: Unique identifier for relationship
            source_entity: Source entity ID
            target_entity: Target entity ID  
            relationship_type: Type of cross-level relationship
            properties: Additional properties
            
        Returns:
            Success status
        """
        try:
            cross_level_rel = {
                'id': relationship_id,
                'source': source_entity,
                'target': target_entity,
                'type': relationship_type,
                'properties': properties or {},
                'created_at': datetime.now()
            }
            
            self.cross_level_relationships.append(cross_level_rel)
            
            # Also add as edge to main knowledge graph
            edge_properties = {
                'relationship_type': relationship_type,
                'cross_level': True
            }
            edge_properties.update(properties or {})
            
            self.knowledge_graph.add_edge(
                relationship_id, 
                [source_entity, target_entity],
                properties=edge_properties
            )
            
            logger.info(f"Added cross-level relationship {relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add cross-level relationship: {e}")
            return False
    
    # ===== MATRIX AND ANALYSIS METHODS =====
    
    def adjacency_matrix(self):
        """Get adjacency matrix for the entire hierarchical graph"""
        # Delegate to the main knowledge graph
        if hasattr(self.knowledge_graph, 'adjacency_matrix'):
            return self.knowledge_graph.adjacency_matrix()
        else:
            import numpy as np
            nodes = list(self.nodes)
            n = len(nodes)
            return np.zeros((n, n)) if nodes else np.array([])
    
    def connected_components(self):
        """Find connected components across all levels"""
        if hasattr(self.knowledge_graph, 'connected_components'):
            return self.knowledge_graph.connected_components()
        else:
            # Simple implementation using DFS
            visited = set()
            components = []
            
            for node in self.nodes:
                if node not in visited:
                    component = set()
                    stack = [node]
                    
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            
                            # Find neighbors across all levels
                            neighbors = set()
                            for level_graph in self.level_graphs.values():
                                if hasattr(level_graph, 'neighbors') and current in level_graph.nodes:
                                    neighbors.update(level_graph.neighbors(current))
                            
                            for neighbor in neighbors:
                                if neighbor not in visited:
                                    stack.append(neighbor)
                    
                    if component:
                        components.append(component)
            
            return components
    
    def diameter(self):
        """Get the diameter of the hierarchical graph"""
        if hasattr(self.knowledge_graph, 'diameter'):
            return self.knowledge_graph.diameter()
        else:
            # Simple estimation - find longest shortest path
            nodes = list(self.nodes)
            if len(nodes) < 2:
                return 0
            
            max_distance = 0
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    distance = self._shortest_path_distance(node1, node2)
                    max_distance = max(max_distance, distance)
            
            return max_distance
    
    def _shortest_path_distance(self, start, end):
        """Helper method to find shortest path distance between two nodes"""
        if start == end:
            return 0
        
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if current == end:
                return distance
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Find neighbors across all levels
            neighbors = set()
            for level_graph in self.level_graphs.values():
                if hasattr(level_graph, 'neighbors') and current in level_graph.nodes:
                    neighbors.update(level_graph.neighbors(current))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return float('inf')  # No path found
    
    # ===== ALGORITHM METHODS =====
    
    def all_shortest_paths(self, source, target):
        """Find all shortest paths between two nodes"""
        if hasattr(self.knowledge_graph, 'all_shortest_paths'):
            return self.knowledge_graph.all_shortest_paths(source, target)
        else:
            # Simple BFS-based implementation
            if source not in self.nodes or target not in self.nodes:
                return []
            
            if source == target:
                return [[source]]
            
            paths = []
            queue = [(source, [source])]
            visited = {source: 0}
            shortest_distance = None
            
            while queue:
                current, path = queue.pop(0)
                
                if shortest_distance is not None and len(path) > shortest_distance:
                    break
                
                # Find neighbors
                neighbors = set()
                for level_graph in self.level_graphs.values():
                    if hasattr(level_graph, 'neighbors') and current in level_graph.nodes:
                        neighbors.update(level_graph.neighbors(current))
                
                for neighbor in neighbors:
                    new_path = path + [neighbor]
                    
                    if neighbor == target:
                        if shortest_distance is None:
                            shortest_distance = len(new_path)
                        if len(new_path) == shortest_distance:
                            paths.append(new_path)
                    elif neighbor not in visited or visited[neighbor] >= len(new_path):
                        visited[neighbor] = len(new_path)
                        queue.append((neighbor, new_path))
            
            return paths
    
    def betweenness_centrality(self):
        """Calculate betweenness centrality for all nodes"""
        if hasattr(self.knowledge_graph, 'betweenness_centrality'):
            return self.knowledge_graph.betweenness_centrality()
        else:
            # Simplified betweenness centrality calculation
            centrality = {node: 0.0 for node in self.nodes}
            nodes = list(self.nodes)
            
            for i, source in enumerate(nodes):
                for target in nodes[i+1:]:
                    paths = self.all_shortest_paths(source, target)
                    if paths:
                        path_count = len(paths)
                        for path in paths:
                            for node in path[1:-1]:  # Exclude source and target
                                centrality[node] += 1.0 / path_count
            
            # Normalize
            n = len(nodes)
            if n > 2:
                normalization = 2.0 / ((n - 1) * (n - 2))
                for node in centrality:
                    centrality[node] *= normalization
            
            return centrality
    
    def closeness_centrality(self):
        """Calculate closeness centrality for all nodes"""
        if hasattr(self.knowledge_graph, 'closeness_centrality'):
            return self.knowledge_graph.closeness_centrality()
        else:
            centrality = {}
            nodes = list(self.nodes)
            
            for node in nodes:
                distances = []
                for other_node in nodes:
                    if node != other_node:
                        distance = self._shortest_path_distance(node, other_node)
                        if distance != float('inf'):
                            distances.append(distance)
                
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    centrality[node] = 1.0 / avg_distance if avg_distance > 0 else 0.0
                else:
                    centrality[node] = 0.0
            
            return centrality
    
    def clustering_coefficient(self):
        """Calculate clustering coefficient for all nodes"""
        if hasattr(self.knowledge_graph, 'clustering_coefficient'):
            return self.knowledge_graph.clustering_coefficient()
        else:
            coefficients = {}
            
            for node in self.nodes:
                # Find neighbors across all levels
                neighbors = set()
                for level_graph in self.level_graphs.values():
                    if hasattr(level_graph, 'neighbors') and node in level_graph.nodes:
                        neighbors.update(level_graph.neighbors(node))
                
                neighbors = list(neighbors)
                n_neighbors = len(neighbors)
                
                if n_neighbors < 2:
                    coefficients[node] = 0.0
                else:
                    # Count edges between neighbors
                    edges_between_neighbors = 0
                    for i, neighbor1 in enumerate(neighbors):
                        for neighbor2 in neighbors[i+1:]:
                            if self._are_neighbors(neighbor1, neighbor2):
                                edges_between_neighbors += 1
                    
                    possible_edges = n_neighbors * (n_neighbors - 1) / 2
                    coefficients[node] = edges_between_neighbors / possible_edges if possible_edges > 0 else 0.0
            
            return coefficients
    
    def _are_neighbors(self, node1, node2):
        """Check if two nodes are neighbors in any level"""
        for level_graph in self.level_graphs.values():
            if (hasattr(level_graph, 'neighbors') and 
                node1 in level_graph.nodes and 
                node2 in level_graph.neighbors(node1)):
                return True
        return False
    
    def community_detection(self, algorithm='simple'):
        """Detect communities in the hierarchical graph"""
        if hasattr(self.knowledge_graph, 'community_detection'):
            return self.knowledge_graph.community_detection(algorithm)
        else:
            # Simple community detection using connected components
            components = self.connected_components()
            communities = {}
            
            for i, component in enumerate(components):
                for node in component:
                    communities[node] = i
            
            return communities
    
    def spectral_clustering(self, n_clusters=2):
        """Perform spectral clustering on the hierarchical graph"""
        if hasattr(self.knowledge_graph, 'spectral_clustering'):
            return self.knowledge_graph.spectral_clustering(n_clusters)
        else:
            # Simplified spectral clustering using adjacency matrix
            adj_matrix = self.adjacency_matrix()
            
            if adj_matrix.size == 0:
                return {}
            
            try:
                import numpy as np
                from sklearn.cluster import SpectralClustering
                
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
                labels = clustering.fit_predict(adj_matrix)
                
                nodes = list(self.nodes)
                return {nodes[i]: int(labels[i]) for i in range(len(nodes))}
            except ImportError:
                # Fallback to simple clustering
                return self.community_detection()
    
    # ===== HIERARCHY-SPECIFIC METHODS =====
    
    def aggregate_across_levels(self, aggregation_type='sum', property_name=None):
        """Aggregate data across hierarchical levels"""
        try:
            results = {}
            
            if aggregation_type == 'count':
                # Count entities per level
                for level_id in self.levels:
                    results[level_id] = len(self.get_entities_at_level(level_id))
            
            elif aggregation_type == 'sum' and property_name:
                # Sum numeric property across levels
                for level_id in self.levels:
                    total = 0
                    entities = self.get_entities_at_level(level_id)
                    
                    for entity in entities:
                        if level_id in self.level_graphs:
                            props = self.level_graphs[level_id].properties.get_node_properties(entity)
                            if props and property_name in props:
                                try:
                                    total += float(props[property_name])
                                except (ValueError, TypeError):
                                    pass
                    
                    results[level_id] = total
            
            return results
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {}
    
    def aggregate_query(self, query_func, level_ids=None):
        """Apply a query function across specified levels"""
        try:
            if level_ids is None:
                level_ids = list(self.levels.keys())
            
            results = {}
            for level_id in level_ids:
                if level_id in self.level_graphs:
                    try:
                        level_result = query_func(self.level_graphs[level_id])
                        results[level_id] = level_result
                    except Exception as e:
                        logger.warning(f"Query failed for level {level_id}: {e}")
                        results[level_id] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Aggregate query failed: {e}")
            return {}
    
    def detect_hierarchy_anomalies(self):
        """Detect anomalies in the hierarchical structure"""
        anomalies = {
            'orphaned_entities': [],
            'circular_references': [],
            'missing_levels': [],
            'empty_levels': []
        }
        
        try:
            # Find orphaned entities (entities without proper level assignment)
            for entity_id in self.nodes:
                if entity_id not in self.entity_levels:
                    anomalies['orphaned_entities'].append(entity_id)
            
            # Find empty levels
            for level_id in self.levels:
                entities = self.get_entities_at_level(level_id)
                if not entities:
                    anomalies['empty_levels'].append(level_id)
            
            # Check for circular references in cross-level relationships
            visited = set()
            rec_stack = set()
            
            def has_cycle(entity):
                if entity in rec_stack:
                    return True
                if entity in visited:
                    return False
                
                visited.add(entity)
                rec_stack.add(entity)
                
                # Check cross-level relationships
                for rel in self.cross_level_relationships:
                    if rel['source'] == entity:
                        if has_cycle(rel['target']):
                            return True
                
                rec_stack.remove(entity)
                return False
            
            for entity in self.nodes:
                if entity not in visited:
                    if has_cycle(entity):
                        anomalies['circular_references'].append(entity)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def find_cross_level_patterns(self):
        """Find patterns in cross-level relationships"""
        patterns = {
            'common_relationship_types': {},
            'level_connectivity': {},
            'entity_bridges': []
        }
        
        try:
            # Analyze relationship types
            for rel in self.cross_level_relationships:
                rel_type = rel.get('type', 'unknown')
                patterns['common_relationship_types'][rel_type] = (
                    patterns['common_relationship_types'].get(rel_type, 0) + 1
                )
            
            # Analyze level connectivity
            for rel in self.cross_level_relationships:
                source_level = self.entity_levels.get(rel['source'])
                target_level = self.entity_levels.get(rel['target'])
                
                if source_level and target_level:
                    level_pair = f"{source_level}->{target_level}"
                    patterns['level_connectivity'][level_pair] = (
                        patterns['level_connectivity'].get(level_pair, 0) + 1
                    )
            
            # Find entities that bridge multiple levels
            entity_connections = {}
            for rel in self.cross_level_relationships:
                source = rel['source']
                target = rel['target']
                
                entity_connections.setdefault(source, set()).add(target)
                entity_connections.setdefault(target, set()).add(source)
            
            for entity, connections in entity_connections.items():
                if len(connections) >= 3:  # Entity connected to 3+ other entities
                    patterns['entity_bridges'].append({
                        'entity': entity,
                        'connections': len(connections),
                        'level': self.entity_levels.get(entity)
                    })
        
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
        
        return patterns
    
    def generate_hierarchy_layout(self, layout_type='tree'):
        """Generate layout coordinates for hierarchy visualization"""
        layout = {}
        
        try:
            if layout_type == 'tree':
                # Tree-like layout with levels as vertical layers
                level_order = sorted(self.level_order.items(), key=lambda x: x[1])
                
                y_spacing = 100
                x_spacing = 150
                
                for i, (level_id, order) in enumerate(level_order):
                    entities = self.get_entities_at_level(level_id)
                    y_pos = order * y_spacing
                    
                    for j, entity in enumerate(entities):
                        x_pos = (j - len(entities)/2) * x_spacing
                        layout[entity] = (x_pos, y_pos)
            
            elif layout_type == 'circular':
                # Circular layout with levels as concentric circles
                import math
                
                level_order = sorted(self.level_order.items(), key=lambda x: x[1])
                max_radius = 200
                
                for i, (level_id, order) in enumerate(level_order):
                    entities = self.get_entities_at_level(level_id)
                    radius = (order + 1) * (max_radius / len(level_order))
                    
                    for j, entity in enumerate(entities):
                        if len(entities) > 1:
                            angle = 2 * math.pi * j / len(entities)
                        else:
                            angle = 0
                        
                        x_pos = radius * math.cos(angle)
                        y_pos = radius * math.sin(angle)
                        layout[entity] = (x_pos, y_pos)
        
        except Exception as e:
            logger.error(f"Layout generation failed: {e}")
        
        return layout
    
    def hierarchy_balance(self):
        """Calculate balance metrics for the hierarchy"""
        try:
            metrics = {
                'level_distribution': {},
                'depth_balance': 0.0,
                'width_balance': 0.0,
                'total_balance_score': 0.0
            }
            
            # Level distribution
            for level_id in self.levels:
                entity_count = len(self.get_entities_at_level(level_id))
                metrics['level_distribution'][level_id] = entity_count
            
            # Calculate balance scores
            entity_counts = list(metrics['level_distribution'].values())
            if entity_counts:
                mean_count = sum(entity_counts) / len(entity_counts)
                variance = sum((count - mean_count) ** 2 for count in entity_counts) / len(entity_counts)
                
                # Lower variance = better balance
                metrics['width_balance'] = 1.0 / (1.0 + variance / max(mean_count, 1))
                
                # Depth balance (prefer not too deep, not too shallow)
                num_levels = len(self.levels)
                optimal_levels = max(2, min(5, int(sum(entity_counts) ** 0.5)))
                depth_deviation = abs(num_levels - optimal_levels)
                metrics['depth_balance'] = 1.0 / (1.0 + depth_deviation)
                
                # Overall balance score
                metrics['total_balance_score'] = (metrics['width_balance'] + metrics['depth_balance']) / 2
            
            return metrics
            
        except Exception as e:
            logger.error(f"Balance calculation failed: {e}")
            return {'total_balance_score': 0.0}
    
    def hierarchy_metrics(self):
        """Get comprehensive hierarchy metrics"""
        try:
            metrics = {
                'structure': self.get_hierarchy_statistics(),
                'balance': self.hierarchy_balance(),
                'anomalies': self.detect_hierarchy_anomalies(),
                'patterns': self.find_cross_level_patterns(),
                'connectivity': self.analyze_cross_level_connectivity()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {}
    
    # ===== I/O CONVERSION METHODS =====
    
    def to_json(self):
        """Convert hierarchical knowledge graph to JSON"""
        if hasattr(self.knowledge_graph, 'to_json'):
            return self.knowledge_graph.to_json()
        else:
            import json
            
            data = {
                'name': self.name,
                'levels': self.levels,
                'entity_levels': self.entity_levels,
                'level_order': self.level_order,
                'cross_level_relationships': [
                    {**rel, 'created_at': rel['created_at'].isoformat() if isinstance(rel.get('created_at'), datetime) else str(rel.get('created_at', ''))}
                    for rel in self.cross_level_relationships
                ],
                'nodes': list(self.nodes),
                'edges': list(self.edges)
            }
            
            return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_data):
        """Create hierarchical knowledge graph from JSON"""
        import json
        
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        hkg = cls(name=data.get('name', 'unnamed'))
        
        # Restore levels
        hkg.levels = data.get('levels', {})
        hkg.entity_levels = data.get('entity_levels', {})
        hkg.level_order = data.get('level_order', {})
        
        # Restore cross-level relationships
        for rel_data in data.get('cross_level_relationships', []):
            rel = rel_data.copy()
            if 'created_at' in rel:
                try:
                    rel['created_at'] = datetime.fromisoformat(rel['created_at'])
                except:
                    pass
            hkg.cross_level_relationships.append(rel)
        
        return hkg
    
    def to_networkx(self):
        """Convert to NetworkX graph"""
        if hasattr(self.knowledge_graph, 'to_networkx'):
            return self.knowledge_graph.to_networkx()
        else:
            try:
                import networkx as nx
                G = nx.Graph()
                
                # Add nodes with level information
                for node in self.nodes:
                    level = self.entity_levels.get(node, 'unknown')
                    G.add_node(node, level=level)
                
                # Add edges from all level graphs
                for level_id, level_graph in self.level_graphs.items():
                    if hasattr(level_graph, 'edges'):
                        for edge in level_graph.edges:
                            edge_nodes = level_graph.incidences.get_edge_nodes(edge)
                            if len(edge_nodes) >= 2:
                                G.add_edge(edge_nodes[0], edge_nodes[1], 
                                         edge_id=edge, level=level_id)
                
                return G
                
            except ImportError:
                raise ImportError("NetworkX is required for this operation")
    
    @classmethod
    def from_networkx(cls, nx_graph):
        """Create from NetworkX graph"""
        hkg = cls()
        
        try:
            # Extract levels from node attributes
            levels_found = set()
            for node, data in nx_graph.nodes(data=True):
                level = data.get('level', 'default')
                levels_found.add(level)
            
            # Create levels
            for i, level in enumerate(levels_found):
                hkg.create_level(level, level, f"Level {level}", i)
            
            # Add nodes to appropriate levels
            for node, data in nx_graph.nodes(data=True):
                level = data.get('level', 'default')
                entity_type = data.get('entity_type', 'Entity')
                hkg.add_entity_to_level(node, level, entity_type, data)
            
            # Add edges
            for i, (u, v, data) in enumerate(nx_graph.edges(data=True)):
                edge_id = data.get('edge_id', f'edge_{i}')
                level = data.get('level', 'default')
                
                if level in hkg.level_graphs:
                    hkg.level_graphs[level].add_edge(edge_id, [u, v], properties=data)
            
            return hkg
            
        except Exception as e:
            logger.error(f"NetworkX import failed: {e}")
            return hkg
    
    def to_gexf(self, filename=None):
        """Export to GEXF format"""
        try:
            nx_graph = self.to_networkx()
            
            if filename:
                import networkx as nx
                nx.write_gexf(nx_graph, filename)
                return filename
            else:
                # Return GEXF string
                import io
                import networkx as nx
                
                buffer = io.StringIO()
                nx.write_gexf(nx_graph, buffer)
                return buffer.getvalue()
                
        except ImportError:
            raise ImportError("NetworkX is required for GEXF export")
        except Exception as e:
            logger.error(f"GEXF export failed: {e}")
            return ""
    
    @classmethod
    def from_gexf(cls, filename_or_data):
        """Import from GEXF format"""
        try:
            import networkx as nx
            
            if filename_or_data.endswith('.gexf') or '/' in filename_or_data:
                # Assume it's a filename
                nx_graph = nx.read_gexf(filename_or_data)
            else:
                # Assume it's GEXF data
                import io
                buffer = io.StringIO(filename_or_data)
                nx_graph = nx.read_gexf(buffer)
            
            return cls.from_networkx(nx_graph)
            
        except ImportError:
            raise ImportError("NetworkX is required for GEXF import")
        except Exception as e:
            logger.error(f"GEXF import failed: {e}")
            return cls()
    
    def to_graphml(self, filename=None):
        """Export to GraphML format"""
        try:
            nx_graph = self.to_networkx()
            
            if filename:
                import networkx as nx
                nx.write_graphml(nx_graph, filename)
                return filename
            else:
                # Return GraphML string
                import io
                import networkx as nx
                
                buffer = io.StringIO()
                nx.write_graphml(nx_graph, buffer)
                return buffer.getvalue()
                
        except ImportError:
            raise ImportError("NetworkX is required for GraphML export")
        except Exception as e:
            logger.error(f"GraphML export failed: {e}")
            return ""
    
    @classmethod
    def from_graphml(cls, filename_or_data):
        """Import from GraphML format"""
        try:
            import networkx as nx
            
            if filename_or_data.endswith('.graphml') or '/' in filename_or_data:
                # Assume it's a filename
                nx_graph = nx.read_graphml(filename_or_data)
            else:
                # Assume it's GraphML data
                import io
                buffer = io.StringIO(filename_or_data)
                nx_graph = nx.read_graphml(buffer)
            
            return cls.from_networkx(nx_graph)
            
        except ImportError:
            raise ImportError("NetworkX is required for GraphML import")
        except Exception as e:
            logger.error(f"GraphML import failed: {e}")
            return cls()

    def __repr__(self) -> str:
        stats = self.get_hierarchy_statistics()
        return (f"HierarchicalKnowledgeGraph(name='{self.name}', "
                f"levels={stats['total_levels']}, "
                f"entities={stats['total_entities']}, "
                f"relationships={stats['total_relationships']})")


# ===== CONVENIENCE FUNCTIONS =====

def create_domain_hierarchy(domain_name: str, 
                          levels_config: List[Dict[str, Any]]) -> HierarchicalKnowledgeGraph:
    """
    Create a hierarchical knowledge graph for a specific domain.
    
    Args:
        domain_name: Name of the domain
        levels_config: List of level configurations with 'id', 'name', 'description', 'order'
        
    Returns:
        Configured HierarchicalKnowledgeGraph
        
    Example:
        >>> levels = [
        ...     {"id": "enterprise", "name": "Enterprise", "description": "Organization level", "order": 0},
        ...     {"id": "division", "name": "Division", "description": "Business divisions", "order": 1},
        ...     {"id": "department", "name": "Department", "description": "Departments", "order": 2},
        ...     {"id": "team", "name": "Team", "description": "Work teams", "order": 3}
        ... ]
        >>> hkg = create_domain_hierarchy("Corporate Structure", levels)
    """
    
    hkg = HierarchicalKnowledgeGraph(name=f"{domain_name}_Hierarchy")
    
    for level_config in levels_config:
        hkg.create_level(
            level_id=level_config["id"],
            level_name=level_config["name"],
            level_description=level_config.get("description", ""),
            level_order=level_config["order"]
        )
    
    return hkg


def create_research_hierarchy() -> HierarchicalKnowledgeGraph:
    """Create a pre-configured hierarchy for research knowledge organization."""
    
    levels = [
        {"id": "field", "name": "Research Field", "description": "Major research fields", "order": 0},
        {"id": "area", "name": "Research Area", "description": "Specific research areas", "order": 1},
        {"id": "topic", "name": "Research Topic", "description": "Research topics", "order": 2},
        {"id": "paper", "name": "Research Paper", "description": "Individual papers", "order": 3},
        {"id": "concept", "name": "Concept", "description": "Research concepts and methods", "order": 4}
    ]
    
    return create_domain_hierarchy("Research Knowledge", levels)


def create_enterprise_hierarchy() -> HierarchicalKnowledgeGraph:
    """Create a pre-configured hierarchy for enterprise knowledge management."""
    
    levels = [
        {"id": "enterprise", "name": "Enterprise", "description": "Organization level", "order": 0},
        {"id": "business_unit", "name": "Business Unit", "description": "Business units/divisions", "order": 1},
        {"id": "data_domain", "name": "Data Domain", "description": "Logical data domains", "order": 2},
        {"id": "dataset", "name": "Dataset", "description": "Individual datasets", "order": 3},
        {"id": "schema", "name": "Schema", "description": "Data schemas and tables", "order": 4},
        {"id": "field", "name": "Field", "description": "Individual data fields", "order": 5}
    ]
    
    return create_domain_hierarchy("Enterprise Data", levels)