"""
Entity Operations Module
=======================

Core entity management operations for the Metagraph system including:
- Entity creation, update, deletion
- Entity retrieval and searching  
- Relationship management
- Basic CRUD operations with proper exception handling
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from datetime import datetime
import orjson
import uuid
import logging

from ....exceptions import (
    MetagraphError, EntityError, RelationshipError, ValidationError,
    require_not_none, require_valid_string, require_valid_dict, handle_exception
)

logger = logging.getLogger(__name__)


class EntityOperations:
    """
    Handles all entity-related operations for the Metagraph.
    
    This class provides methods for creating, updating, deleting, and querying
    entities with proper validation, error handling, and logging.
    """
    
    def __init__(self, storage_path: str, metadata_store, hierarchical_store):
        """
        Initialize entity operations.
        
        Args:
            storage_path: Path to store entity data
            metadata_store: Reference to metadata storage system
            hierarchical_store: Reference to hierarchical storage system
        """
        self.storage_path = Path(storage_path)
        self.metadata_store = metadata_store
        self.hierarchical_store = hierarchical_store
        self.logger = logger.getChild(self.__class__.__name__)
        
    def create_entity(self,
                     entity_id: str,
                     entity_type: str, 
                     properties: Dict[str, Any],
                     parent_id: Optional[str] = None,
                     validate_schema: bool = True) -> bool:
        """
        Create a new entity with proper validation and error handling.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type/category of the entity
            properties: Dictionary of entity properties
            parent_id: Optional parent entity for hierarchical relationships
            validate_schema: Whether to validate against schema
            
        Returns:
            True if entity was created successfully
            
        Raises:
            EntityError: If entity creation fails
            ValidationError: If validation fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            entity_type = require_valid_string(entity_type, "entity_type")
            properties = require_valid_dict(properties, "properties")
            
            # Check if entity already exists
            if self._entity_exists(entity_id):
                raise EntityError(
                    f"Entity '{entity_id}' already exists",
                    error_code="ENTITY_ALREADY_EXISTS",
                    context={"entity_id": entity_id, "entity_type": entity_type}
                )
            
            # Validate parent if provided
            if parent_id and not self._entity_exists(parent_id):
                raise EntityError(
                    f"Parent entity '{parent_id}' does not exist",
                    error_code="PARENT_ENTITY_NOT_FOUND",
                    context={"entity_id": entity_id, "parent_id": parent_id}
                )
            
            # Create entity data structure
            entity_data = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "properties": properties,
                "parent_id": parent_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": 1
            }
            
            # Store in metadata system
            if not self.metadata_store.store_entity(entity_data):
                raise EntityError(
                    f"Failed to store entity metadata for '{entity_id}'",
                    error_code="METADATA_STORE_FAILED",
                    context={"entity_id": entity_id, "entity_type": entity_type}
                )
            
            # Store in hierarchical system if parent specified
            if parent_id:
                if not self.hierarchical_store.add_child_relationship(parent_id, entity_id):
                    # Rollback metadata store
                    self.metadata_store.delete_entity(entity_id)
                    raise EntityError(
                        f"Failed to create hierarchical relationship for '{entity_id}'",
                        error_code="HIERARCHICAL_STORE_FAILED",
                        context={"entity_id": entity_id, "parent_id": parent_id}
                    )
            
            self.logger.info(
                "Entity created successfully",
                extra={
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "has_parent": parent_id is not None
                }
            )
            
            return True
            
        except (EntityError, ValidationError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception(f"creating entity '{entity_id}'", e, {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "parent_id": parent_id
            })
    
    def update_entity(self,
                     entity_id: str,
                     properties: Dict[str, Any],
                     merge_properties: bool = True,
                     create_version: bool = True) -> bool:
        """
        Update an existing entity with proper validation and versioning.
        
        Args:
            entity_id: ID of entity to update
            properties: New or updated properties
            merge_properties: Whether to merge with existing properties
            create_version: Whether to create a new version
            
        Returns:
            True if entity was updated successfully
            
        Raises:
            EntityError: If entity update fails
            ValidationError: If validation fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            properties = require_valid_dict(properties, "properties")
            
            # Check if entity exists
            existing_entity = self.get_entity(entity_id)
            if not existing_entity:
                raise EntityError(
                    f"Entity '{entity_id}' not found",
                    error_code="ENTITY_NOT_FOUND",
                    context={"entity_id": entity_id}
                )
            
            # Prepare updated data
            updated_properties = existing_entity["properties"].copy() if merge_properties else {}
            updated_properties.update(properties)
            
            updated_entity = existing_entity.copy()
            updated_entity.update({
                "properties": updated_properties,
                "updated_at": datetime.now().isoformat(),
                "version": existing_entity.get("version", 1) + (1 if create_version else 0)
            })
            
            # Update in metadata store
            if not self.metadata_store.update_entity(entity_id, updated_entity):
                raise EntityError(
                    f"Failed to update entity metadata for '{entity_id}'",
                    error_code="METADATA_UPDATE_FAILED",
                    context={"entity_id": entity_id}
                )
            
            self.logger.info(
                "Entity updated successfully",
                extra={
                    "entity_id": entity_id,
                    "properties_updated": len(properties),
                    "version": updated_entity["version"]
                }
            )
            
            return True
            
        except (EntityError, ValidationError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception(f"updating entity '{entity_id}'", e, {
                "entity_id": entity_id,
                "properties_count": len(properties) if properties else 0
            })
    
    def delete_entity(self,
                     entity_id: str,
                     delete_children: bool = False,
                     create_backup: bool = True) -> bool:
        """
        Delete an entity with proper cleanup and validation.
        
        Args:
            entity_id: ID of entity to delete
            delete_children: Whether to delete child entities
            create_backup: Whether to create backup before deletion
            
        Returns:
            True if entity was deleted successfully
            
        Raises:
            EntityError: If entity deletion fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            
            # Check if entity exists
            entity = self.get_entity(entity_id)
            if not entity:
                raise EntityError(
                    f"Entity '{entity_id}' not found",
                    error_code="ENTITY_NOT_FOUND",
                    context={"entity_id": entity_id}
                )
            
            # Check for children if not deleting them
            children = self.hierarchical_store.get_children(entity_id)
            if children and not delete_children:
                raise EntityError(
                    f"Entity '{entity_id}' has children and delete_children=False",
                    error_code="ENTITY_HAS_CHILDREN",
                    context={"entity_id": entity_id, "children_count": len(children)}
                )
            
            # Create backup if requested
            if create_backup:
                backup_path = self.storage_path / "backups" / f"{entity_id}_{datetime.now().isoformat()}.json"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_path, 'wb') as f:
                    f.write(orjson.dumps(entity))
            
            # Delete children first if requested
            if delete_children and children:
                for child_id in children:
                    self.delete_entity(child_id, delete_children=True, create_backup=create_backup)
            
            # Remove from hierarchical store
            self.hierarchical_store.remove_entity(entity_id)
            
            # Remove from metadata store
            if not self.metadata_store.delete_entity(entity_id):
                raise EntityError(
                    f"Failed to delete entity metadata for '{entity_id}'",
                    error_code="METADATA_DELETE_FAILED",
                    context={"entity_id": entity_id}
                )
            
            self.logger.info(
                "Entity deleted successfully",
                extra={
                    "entity_id": entity_id,
                    "children_deleted": len(children) if delete_children else 0,
                    "backup_created": create_backup
                }
            )
            
            return True
            
        except (EntityError, ValidationError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception(f"deleting entity '{entity_id}'", e, {
                "entity_id": entity_id,
                "delete_children": delete_children
            })
    
    def get_entity(self, entity_id: str, include_relationships: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity with optional relationship information.
        
        Args:
            entity_id: ID of entity to retrieve
            include_relationships: Whether to include relationship data
            
        Returns:
            Entity data dictionary or None if not found
            
        Raises:
            EntityError: If retrieval fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            
            # Get base entity data
            entity = self.metadata_store.get_entity(entity_id)
            if not entity:
                return None
            
            # Add relationship information if requested
            if include_relationships:
                entity["children"] = self.hierarchical_store.get_children(entity_id)
                entity["parent"] = self.hierarchical_store.get_parent(entity_id)
                entity["relationships"] = self._get_entity_relationships(entity_id)
            
            return entity
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception(f"retrieving entity '{entity_id}'", e, {
                "entity_id": entity_id,
                "include_relationships": include_relationships
            })
    
    def search_entities(self,
                       query: Optional[str] = None,
                       entity_type: Optional[str] = None,
                       property_filters: Optional[Dict[str, Any]] = None,
                       limit: int = 100,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search entities with various filters and pagination.
        
        Args:
            query: Text query to search in entity properties
            entity_type: Filter by entity type
            property_filters: Filters for specific properties
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching entities
            
        Raises:
            EntityError: If search fails
        """
        try:
            # Validate inputs
            if limit <= 0:
                raise ValidationError(
                    "Limit must be positive",
                    error_code="INVALID_LIMIT",
                    context={"limit": limit}
                )
            
            if offset < 0:
                raise ValidationError(
                    "Offset must be non-negative",
                    error_code="INVALID_OFFSET",
                    context={"offset": offset}
                )
            
            # Perform search using metadata store
            results = self.metadata_store.search_entities(
                query=query,
                entity_type=entity_type,
                property_filters=property_filters,
                limit=limit,
                offset=offset
            )
            
            self.logger.debug(
                "Entity search completed",
                extra={
                    "query": query,
                    "entity_type": entity_type,
                    "results_count": len(results),
                    "limit": limit,
                    "offset": offset
                }
            )
            
            return results
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception("searching entities", e, {
                "query": query,
                "entity_type": entity_type,
                "limit": limit,
                "offset": offset
            })
    
    def add_relationship(self,
                        source_entity_id: str,
                        target_entity_id: str,
                        relationship_type: str,
                        properties: Optional[Dict[str, Any]] = None,
                        strength: float = 1.0) -> str:
        """
        Add a relationship between two entities.
        
        Args:
            source_entity_id: ID of source entity
            target_entity_id: ID of target entity
            relationship_type: Type of relationship
            properties: Optional relationship properties
            strength: Relationship strength (0.0 to 1.0)
            
        Returns:
            Relationship ID
            
        Raises:
            RelationshipError: If relationship creation fails
        """
        try:
            # Validate inputs
            source_entity_id = require_valid_string(source_entity_id, "source_entity_id")
            target_entity_id = require_valid_string(target_entity_id, "target_entity_id")
            relationship_type = require_valid_string(relationship_type, "relationship_type")
            
            if not 0.0 <= strength <= 1.0:
                raise ValidationError(
                    "Strength must be between 0.0 and 1.0",
                    error_code="INVALID_STRENGTH",
                    context={"strength": strength}
                )
            
            # Check if entities exist
            if not self._entity_exists(source_entity_id):
                raise RelationshipError(
                    f"Source entity '{source_entity_id}' not found",
                    error_code="SOURCE_ENTITY_NOT_FOUND",
                    context={"source_entity_id": source_entity_id}
                )
            
            if not self._entity_exists(target_entity_id):
                raise RelationshipError(
                    f"Target entity '{target_entity_id}' not found",
                    error_code="TARGET_ENTITY_NOT_FOUND",
                    context={"target_entity_id": target_entity_id}
                )
            
            # Create relationship
            relationship_id = str(uuid.uuid4())
            relationship_data = {
                "relationship_id": relationship_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relationship_type": relationship_type,
                "properties": properties or {},
                "strength": strength,
                "created_at": datetime.now().isoformat()
            }
            
            # Store relationship
            if not self.metadata_store.store_relationship(relationship_data):
                raise RelationshipError(
                    f"Failed to store relationship",
                    error_code="RELATIONSHIP_STORE_FAILED",
                    context={"relationship_id": relationship_id}
                )
            
            self.logger.info(
                "Relationship created successfully",
                extra={
                    "relationship_id": relationship_id,
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                    "relationship_type": relationship_type
                }
            )
            
            return relationship_id
            
        except (RelationshipError, ValidationError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception("creating relationship", e, {
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relationship_type": relationship_type
            })
    
    def get_related_entities(self,
                           entity_id: str,
                           relationship_type: Optional[str] = None,
                           direction: Literal['incoming', 'outgoing', 'both'] = 'both',
                           max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_id: ID of the source entity
            relationship_type: Optional filter by relationship type
            direction: Direction of relationships to follow
            max_depth: Maximum depth to traverse
            
        Returns:
            List of related entities
            
        Raises:
            EntityError: If retrieval fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            
            if max_depth <= 0:
                raise ValidationError(
                    "Max depth must be positive",
                    error_code="INVALID_MAX_DEPTH",
                    context={"max_depth": max_depth}
                )
            
            # Check if entity exists
            if not self._entity_exists(entity_id):
                raise EntityError(
                    f"Entity '{entity_id}' not found",
                    error_code="ENTITY_NOT_FOUND",
                    context={"entity_id": entity_id}
                )
            
            # Get related entities
            related_entities = self.metadata_store.get_related_entities(
                entity_id=entity_id,
                relationship_type=relationship_type,
                direction=direction,
                max_depth=max_depth
            )
            
            return related_entities
            
        except (EntityError, ValidationError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise handle_exception(f"getting related entities for '{entity_id}'", e, {
                "entity_id": entity_id,
                "relationship_type": relationship_type,
                "direction": direction
            })
    
    def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        try:
            return self.metadata_store.entity_exists(entity_id)
        except Exception as e:
            self.logger.warning(f"Error checking entity existence: {e}")
            return False
    
    def _get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        try:
            return self.metadata_store.get_entity_relationships(entity_id)
        except Exception as e:
            self.logger.warning(f"Error getting entity relationships: {e}")
            return []