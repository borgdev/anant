"""
Hierarchical Store for Metagraph - Phase 1
==========================================

Implements hierarchical navigation and storage using Polars+Parquet backend.
Supports multi-level hierarchies with efficient parent-child relationships.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import orjson
import uuid


class HierarchicalStore:
    """
    Polars+Parquet-based hierarchical data storage for metagraph levels.
    
    Provides efficient storage and querying of hierarchical relationships
    with support for multiple hierarchy types and fast navigation.
    """
    
    def __init__(self, 
                 storage_path: Path,
                 compression: str = "zstd"):
        """
        Initialize hierarchical store with Polars+Parquet backend.
        
        Parameters
        ----------
        storage_path : Path
            Directory for storing hierarchical data
        compression : str
            Compression algorithm for Parquet files
        """
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        
        # Initialize storage structure
        self._setup_storage_structure()
        
        # Load existing data
        self._load_existing_data()
    
    def _setup_storage_structure(self) -> None:
        """Create directory structure for hierarchical storage."""
        
        directories = [
            "levels",        # Level definitions and entity assignments
            "relationships", # Parent-child relationships
            "hierarchies",   # Multiple hierarchy support
            "navigation"     # Pre-computed navigation paths
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def _load_existing_data(self) -> None:
        """Load existing hierarchical data from storage."""
        
        # Load levels
        levels_file = self.storage_path / "levels" / "levels.parquet"
        if levels_file.exists():
            self.levels_df = pl.read_parquet(levels_file)
        else:
            self.levels_df = pl.DataFrame({
                "level_name": [],
                "level_order": [],
                "description": [],
                "created_at": [],
                "updated_at": []
            }, schema={
                "level_name": pl.Utf8,
                "level_order": pl.Int32,
                "description": pl.Utf8,
                "created_at": pl.Datetime,
                "updated_at": pl.Datetime
            })
        
        # Load entity assignments
        entities_file = self.storage_path / "levels" / "entities.parquet"
        if entities_file.exists():
            self.entities_df = pl.read_parquet(entities_file)
        else:
            self.entities_df = pl.DataFrame({
                "entity_id": [],
                "level_name": [],
                "entity_data": [],  # JSON string
                "created_at": [],
                "updated_at": []
            }, schema={
                "entity_id": pl.Utf8,
                "level_name": pl.Utf8,
                "entity_data": pl.Utf8,
                "created_at": pl.Datetime,
                "updated_at": pl.Datetime
            })
        
        # Load relationships
        relationships_file = self.storage_path / "relationships" / "parent_child.parquet"
        if relationships_file.exists():
            self.relationships_df = pl.read_parquet(relationships_file)
        else:
            self.relationships_df = pl.DataFrame({
                "relationship_id": [],
                "parent_id": [],
                "child_id": [],
                "relationship_type": [],
                "properties": [],  # JSON string
                "created_at": [],
                "updated_at": []
            }, schema={
                "relationship_id": pl.Utf8,
                "parent_id": pl.Utf8,
                "child_id": pl.Utf8,
                "relationship_type": pl.Utf8,
                "properties": pl.Utf8,
                "created_at": pl.Datetime,
                "updated_at": pl.Datetime
            })
    
    def create_level(self, 
                    level_name: str, 
                    level_data: Dict[str, Any],
                    level_order: Optional[int] = None) -> None:
        """
        Create a new hierarchical level.
        
        Parameters
        ----------
        level_name : str
            Name of the hierarchical level
        level_data : Dict
            Level configuration and metadata
        level_order : int, optional
            Order in hierarchy (lower = higher in hierarchy)
        """
        
        # Determine level order if not provided
        if level_order is None:
            max_order = 0
            if self.levels_df.height > 0:
                max_order = self.levels_df["level_order"].max()
            level_order = max_order + 1
        
        # Create level record
        level_record = pl.DataFrame([{
            "level_name": level_name,
            "level_order": level_order,
            "description": level_data.get("description", ""),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }])
        
        # Add to levels DataFrame
        self.levels_df = pl.concat([self.levels_df, level_record])
        
        # Save to storage
        self._save_levels()
    
    def add_entity_to_level(self, 
                           entity_id: str,
                           level_name: str,
                           entity_data: Dict[str, Any]) -> None:
        """
        Add entity to a hierarchical level.
        
        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        level_name : str
            Name of the level to add entity to
        entity_data : Dict
            Entity data and properties
        """
        
        # Create entity record
        entity_record = pl.DataFrame([{
            "entity_id": entity_id,
            "level_name": level_name,
            "entity_data": orjson.dumps(entity_data).decode(),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }])
        
        # Add to entities DataFrame
        self.entities_df = pl.concat([self.entities_df, entity_record])
        
        # Save to storage
        self._save_entities()
    
    def add_parent_child_relationship(self,
                                    parent_id: str,
                                    child_id: str,
                                    relationship_type: str = "contains",
                                    properties: Optional[Dict] = None) -> None:
        """
        Add parent-child relationship between entities.
        
        Parameters
        ----------
        parent_id : str
            Parent entity ID
        child_id : str
            Child entity ID
        relationship_type : str
            Type of parent-child relationship
        properties : Dict, optional
            Additional relationship properties
        """
        
        relationship_id = str(uuid.uuid4())
        
        # Create relationship record
        relationship_record = pl.DataFrame([{
            "relationship_id": relationship_id,
            "parent_id": parent_id,
            "child_id": child_id,
            "relationship_type": relationship_type,
            "properties": orjson.dumps(properties or {}).decode(),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }])
        
        # Add to relationships DataFrame
        self.relationships_df = pl.concat([self.relationships_df, relationship_record])
        
        # Save to storage
        self._save_relationships()
    
    def get_level(self, level_name: str) -> Optional[pl.DataFrame]:
        """Get all entities at a specific level."""
        
        if self.entities_df.height == 0:
            return None
            
        level_entities = self.entities_df.filter(
            pl.col("level_name") == level_name
        )
        
        if level_entities.height == 0:
            return None
            
        return level_entities
    
    def get_parent(self, entity_id: str) -> Optional[str]:
        """Get parent entity ID."""
        
        if self.relationships_df.height == 0:
            return None
            
        parent_rel = self.relationships_df.filter(
            pl.col("child_id") == entity_id
        )
        
        if parent_rel.height == 0:
            return None
            
        return parent_rel["parent_id"][0]
    
    def get_children(self, entity_id: str) -> List[str]:
        """Get child entity IDs."""
        
        if self.relationships_df.height == 0:
            return []
            
        children_rels = self.relationships_df.filter(
            pl.col("parent_id") == entity_id
        )
        
        if children_rels.height == 0:
            return []
            
        return children_rels["child_id"].to_list()
    
    def get_ancestors(self, entity_id: str) -> List[str]:
        """Get all ancestor entity IDs up the hierarchy."""
        
        ancestors = []
        current_id = entity_id
        
        while True:
            parent_id = self.get_parent(current_id)
            if parent_id is None:
                break
            ancestors.append(parent_id)
            current_id = parent_id
            
            # Prevent infinite loops
            if len(ancestors) > 100:
                break
        
        return ancestors
    
    def get_descendants(self, entity_id: str) -> List[str]:
        """Get all descendant entity IDs down the hierarchy."""
        
        descendants = []
        to_process = [entity_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            if current_id in processed:
                continue
                
            processed.add(current_id)
            children = self.get_children(current_id)
            
            for child in children:
                if child not in processed:
                    descendants.append(child)
                    to_process.append(child)
        
        return descendants
    
    def find_path(self, from_entity: str, to_entity: str) -> Optional[List[str]]:
        """Find navigation path between two entities."""
        
        # Simple BFS implementation
        queue = [(from_entity, [from_entity])]
        visited = {from_entity}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == to_entity:
                return path
            
            # Check children
            for child in self.get_children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
            
            # Check parent
            parent = self.get_parent(current)
            if parent and parent not in visited:
                visited.add(parent)
                queue.append((parent, path + [parent]))
        
        return None  # No path found
    
    def get_level_count(self) -> int:
        """Get number of defined levels."""
        return self.levels_df.height
    
    def get_entity_count(self) -> int:
        """Get total number of entities."""
        return self.entities_df.height
    
    def add_relationships(self, relationship_type: str, relationships: List[Dict]) -> None:
        """Add multiple relationships of a specific type."""
        
        for rel in relationships:
            self.add_parent_child_relationship(
                parent_id=rel["parent"],
                child_id=rel["child"],
                relationship_type=relationship_type,
                properties=rel.get("properties")
            )
    
    def _save_levels(self) -> None:
        """Save levels to Parquet file."""
        levels_file = self.storage_path / "levels" / "levels.parquet"
        self.levels_df.write_parquet(levels_file, compression=self.compression)
    
    def _save_entities(self) -> None:
        """Save entities to Parquet file."""
        entities_file = self.storage_path / "levels" / "entities.parquet"
        self.entities_df.write_parquet(entities_file, compression=self.compression)
    
    def _save_relationships(self) -> None:
        """Save relationships to Parquet file."""
        relationships_file = self.storage_path / "relationships" / "parent_child.parquet"
        self.relationships_df.write_parquet(relationships_file, compression=self.compression)
    
    def save_state(self) -> None:
        """Save complete hierarchical store state."""
        self._save_levels()
        self._save_entities()
        self._save_relationships()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchical store statistics."""
        
        return {
            "levels_count": self.get_level_count(),
            "entities_count": self.get_entity_count(),
            "relationships_count": self.relationships_df.height,
            "max_depth": self._calculate_max_depth(),
            "storage_path": str(self.storage_path)
        }
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum hierarchy depth."""
        
        if self.entities_df.height == 0:
            return 0
        
        max_depth = 0
        
        # Find root entities (entities with no parents)
        all_children = set(self.relationships_df["child_id"].to_list()) if self.relationships_df.height > 0 else set()
        all_entities = set(self.entities_df["entity_id"].to_list())
        root_entities = all_entities - all_children
        
        for root in root_entities:
            depth = self._calculate_entity_depth(root, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_entity_depth(self, entity_id: str, visited: Set[str]) -> int:
        """Calculate depth of entity in hierarchy."""
        
        if entity_id in visited:
            return 0  # Prevent cycles
        
        visited.add(entity_id)
        children = self.get_children(entity_id)
        
        if not children:
            return 1
        
        max_child_depth = 0
        for child in children:
            child_depth = self._calculate_entity_depth(child, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth