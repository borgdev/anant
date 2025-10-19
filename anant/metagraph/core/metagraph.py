"""
Core Metagraph Implementation - Phase 1
=======================================

Enterprise-grade knowledge graph with hierarchical navigation, semantic relationships,
temporal tracking, and governance policies. Built on Polars+Parquet for performance.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from datetime import datetime
import orjson
import uuid

# Import our layers
from .hierarchical_store import HierarchicalStore
from .metadata_store import MetadataStore
from ..semantic.semantic_layer import SemanticLayer
from ..temporal.temporal_layer import TemporalLayer  
from ..governance.policy_layer import PolicyEngine


class Metagraph:
    """
    Enterprise Metagraph - Advanced knowledge management system.
    
    Combines hierarchical data organization, semantic relationships, temporal tracking,
    and governance policies in a high-performance Polars+Parquet architecture.
    
    Features:
    - Hierarchical entity organization with parent-child relationships
    - Rich metadata storage with schema validation and querying
    - Semantic embeddings and relationship discovery
    - Temporal event tracking and pattern analysis
    - Governance policies with access control and compliance
    - High-performance analytics with Polars backend
    - Enterprise-grade data management with ZSTD compression
    """
    
    def __init__(self,
                 storage_path: str = "./metagraph_data",
                 embedding_dimension: int = 768,
                 compression: str = "zstd",
                 retention_days: int = 365):
        """
        Initialize Enterprise Metagraph.
        
        Args:
            storage_path: Base path for all metagraph data
            embedding_dimension: Dimension for semantic embeddings
            compression: Parquet compression algorithm
            retention_days: Temporal data retention period
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core layers
        self.hierarchical = HierarchicalStore(
            storage_path=str(self.storage_path / "hierarchical"),
            compression=compression
        )
        
        self.metadata = MetadataStore(
            storage_path=str(self.storage_path / "metadata"),
            compression=compression
        )
        
        self.semantic = SemanticLayer(
            storage_path=str(self.storage_path / "semantic"),
            embedding_dimension=embedding_dimension,
            compression=compression
        )
        
        self.temporal = TemporalLayer(
            storage_path=str(self.storage_path / "temporal"),
            compression=compression,
            retention_days=retention_days
        )
        
        self.governance = PolicyEngine(
            storage_path=str(self.storage_path / "governance"),
            compression=compression
        )
        
        # Cross-layer integration state
        self._entity_registry: Dict[str, Dict[str, Any]] = {}
        
    def create_entity(self,
                     entity_id: str,
                     entity_type: str,
                     properties: Dict[str, Any],
                     level: int = 0,
                     parent_id: Optional[str] = None,
                     classification: Literal["public", "internal", "confidential", "restricted", "top_secret"] = "internal",
                     created_by: str = "system") -> bool:
        """
        Create a new entity across all layers.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type/category of entity
            properties: Entity properties and attributes
            level: Hierarchical level (0 = root)
            parent_id: Parent entity for hierarchical organization
            classification: Data classification level
            created_by: Entity creator
            
        Returns:
            Success status
        """
        try:
            # 1. Add to hierarchical store
            self.hierarchical.assign_entity_to_level(entity_id, level, parent_id, properties)
            
            # 2. Add metadata
            self.metadata.add_entity_metadata(entity_id, entity_type, properties, created_by)
            
            # 3. Record temporal event
            self.temporal.record_event(
                entity_id=entity_id,
                operation="create",
                details={"entity_type": entity_type, "level": level, "parent_id": parent_id},
                user_id=created_by
            )
            
            # 4. Apply data classification
            self.governance.classify_resource(
                resource_id=entity_id,
                classification=classification,
                classified_by=created_by,
                classification_method="automatic"
            )
            
            # 5. Update entity registry
            self._entity_registry[entity_id] = {
                "entity_type": entity_type,
                "level": level,
                "parent_id": parent_id,
                "classification": classification,
                "created_at": datetime.now(),
                "created_by": created_by
            }
            
            return True
            
        except Exception as e:
            # Rollback on failure
            self._rollback_entity_creation(entity_id)
            raise e
    
    def update_entity(self,
                     entity_id: str,
                     properties: Dict[str, Any],
                     updated_by: str = "system") -> bool:
        """
        Update entity properties across layers.
        
        Args:
            entity_id: Entity to update
            properties: New/updated properties
            updated_by: User making the update
            
        Returns:
            Success status
        """
        # Check access permission
        access_result = self.governance.check_access(
            user_id=updated_by,
            resource_id=entity_id,
            access_level="write"
        )
        
        if access_result["decision"] != "granted":
            raise PermissionError(f"Access denied: {access_result}")
        
        # Get current state for change tracking
        current_metadata = self.metadata.get_entity_metadata(entity_id)
        
        # Update metadata
        updated = self.metadata.update_entity_metadata(
            entity_id=entity_id,
            properties=properties,
            updated_by=updated_by
        )
        
        if updated:
            # Record temporal event with change details
            change_summary = self._calculate_change_summary(current_metadata, properties)
            
            self.temporal.record_event(
                entity_id=entity_id,
                operation="update",
                details={"changes": change_summary, "properties_count": len(properties)},
                user_id=updated_by
            )
            
            # Create temporal snapshot
            new_metadata = self.metadata.get_entity_metadata(entity_id)
            self.temporal.create_snapshot(
                entity_id=entity_id,
                state=new_metadata,
                change_summary=change_summary,
                created_by=updated_by
            )
        
        return updated
    
    def delete_entity(self,
                     entity_id: str,
                     deleted_by: str = "system",
                     cascade: bool = False) -> bool:
        """
        Delete entity from all layers.
        
        Args:
            entity_id: Entity to delete
            deleted_by: User performing deletion
            cascade: Whether to delete child entities
            
        Returns:
            Success status
        """
        # Check access permission
        access_result = self.governance.check_access(
            user_id=deleted_by,
            resource_id=entity_id,
            access_level="delete"
        )
        
        if access_result["decision"] != "granted":
            raise PermissionError(f"Delete access denied: {access_result}")
        
        # Get children if cascade delete
        children = []
        if cascade:
            children = self.hierarchical.get_entities_at_level_with_parent(
                level=self.hierarchical.get_entity_level(entity_id) + 1,
                parent_id=entity_id
            )
        
        # Record deletion event
        self.temporal.record_event(
            entity_id=entity_id,
            operation="delete",
            details={"cascade": cascade, "children_count": len(children)},
            user_id=deleted_by
        )
        
        # Delete from all layers
        success = True
        
        # Delete children first if cascading
        if cascade:
            for child_data in children:
                child_id = child_data["entity_id"]
                success &= self.delete_entity(child_id, deleted_by, cascade=False)
        
        # Delete from hierarchical store
        success &= self.hierarchical.remove_entity_from_level(entity_id)
        
        # Mark metadata as deleted (soft delete for audit trail)
        success &= self.metadata.soft_delete_entity(entity_id, deleted_by)
        
        # Remove from entity registry
        if entity_id in self._entity_registry:
            del self._entity_registry[entity_id]
        
        return success
    
    def get_entity(self, entity_id: str, include_relationships: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive entity information from all layers.
        
        Args:
            entity_id: Entity identifier
            include_relationships: Include semantic relationships
            
        Returns:
            Complete entity information or None
        """
        # Get base metadata
        metadata = self.metadata.get_metadata(entity_id)
        if not metadata:
            return None
        
        # Get hierarchical information
        level = self.hierarchical.get_entity_level(entity_id)
        parent_id = self.hierarchical.get_parent(entity_id)
        children = self.hierarchical.get_children(entity_id)
        
        # Build comprehensive entity view
        entity_info = {
            "entity_id": entity_id,
            "metadata": metadata,
            "hierarchical": {
                "level": level,
                "parent_id": parent_id,
                "children": children,
                "path": self.hierarchical.get_path_to_root(entity_id)
            },
            "temporal": {
                "timeline": self.temporal.get_entity_timeline(entity_id, limit=10),
                "latest_snapshot": self.temporal.get_snapshot(entity_id)
            }
        }
        
        # Add relationships if requested
        if include_relationships:
            entity_info["semantic"] = self.semantic.get_semantic_context(entity_id)
        
        # Add governance information
        entity_info["governance"] = {
            "classification": self._get_entity_classification(entity_id),
            "compliance_status": self._get_compliance_status(entity_id)
        }
        
        return entity_info
    
    def search_entities(self,
                       query: str,
                       entity_types: Optional[List[str]] = None,
                       levels: Optional[List[int]] = None,
                       date_range: Optional[Tuple[datetime, datetime]] = None,
                       classification_levels: Optional[List[str]] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search entities across all layers with comprehensive filtering.
        
        Args:
            query: Search query text
            entity_types: Filter by entity types
            levels: Filter by hierarchical levels
            date_range: Filter by creation/update date range
            classification_levels: Filter by classification levels
            limit: Maximum results to return
            
        Returns:
            List of matching entities with scores
        """
        # Start with metadata search
        metadata_results = self.metadata.search_metadata(
            query=query,
            entity_types=entity_types,
            date_range=date_range,
            limit=limit * 2  # Get more for further filtering
        )
        
        # Apply additional filters
        filtered_results = []
        
        for result in metadata_results:
            entity_id = result["entity_id"]
            
            # Filter by hierarchical level
            if levels is not None:
                entity_level = self.hierarchical.get_entity_level(entity_id)
                if entity_level not in levels:
                    continue
            
            # Filter by classification
            if classification_levels is not None:
                classification = self._get_entity_classification(entity_id)
                if classification not in classification_levels:
                    continue
            
            # Add comprehensive information
            entity_info = self.get_entity(entity_id, include_relationships=False)
            if entity_info:
                entity_info["search_score"] = result.get("score", 1.0)
                filtered_results.append(entity_info)
            
            # Limit results
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    def add_relationship(self,
                        source_id: str,
                        target_id: str,
                        relationship_type: str,
                        strength: float = 1.0,
                        metadata: Optional[Dict[str, Any]] = None,
                        created_by: str = "system") -> str:
        """
        Add semantic relationship between entities.
        
        Args:
            source_id: Source entity
            target_id: Target entity
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            metadata: Additional relationship metadata
            created_by: User creating relationship
            
        Returns:
            relationship_id: Unique identifier
        """
        relationship_id = self.semantic.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata
        )
        
        # Record temporal events for both entities
        for entity_id in [source_id, target_id]:
            self.temporal.record_event(
                entity_id=entity_id,
                operation="update",
                details={
                    "relationship_added": relationship_id,
                    "related_entity": target_id if entity_id == source_id else source_id,
                    "relationship_type": relationship_type
                },
                user_id=created_by
            )
        
        return relationship_id
    
    def get_related_entities(self,
                           entity_id: str,
                           max_depth: int = 2,
                           min_strength: float = 0.1) -> Dict[str, Any]:
        """
        Get entities related through semantic relationships and hierarchy.
        
        Args:
            entity_id: Starting entity
            max_depth: Maximum relationship depth to traverse
            min_strength: Minimum relationship strength threshold
            
        Returns:
            Related entities with relationship information
        """
        related = {
            "entity_id": entity_id,
            "semantic_relationships": [],
            "hierarchical_relationships": {},
            "similar_entities": []
        }
        
        # Get semantic relationships
        relationships = self.semantic.get_relationships(
            entity_id=entity_id,
            min_strength=min_strength
        )
        related["semantic_relationships"] = [
            {
                "target_id": rel.target_id,
                "relationship_type": rel.relationship_type,
                "strength": rel.strength,
                "context": rel.context
            }
            for rel in relationships
        ]
        
        # Get hierarchical relationships
        related["hierarchical_relationships"] = {
            "parent": self.hierarchical.get_parent(entity_id),
            "children": self.hierarchical.get_children(entity_id),
            "siblings": self.hierarchical.get_siblings(entity_id),
            "ancestors": self.hierarchical.get_path_to_root(entity_id)[:-1]  # Exclude self
        }
        
        # Get similar entities based on embeddings
        similar = self.semantic.find_similar_entities(
            entity_id=entity_id,
            top_k=10,
            min_similarity=min_strength
        )
        related["similar_entities"] = [
            {"entity_id": eid, "similarity": sim}
            for eid, sim in similar
        ]
        
        return related
    
    def analyze_temporal_patterns(self,
                                entity_ids: Optional[List[str]] = None,
                                days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze temporal patterns across entities.
        
        Args:
            entity_ids: Specific entities to analyze (None for all)
            days_back: Number of days to analyze
            
        Returns:
            Temporal analysis results
        """
        from datetime import timedelta
        
        patterns = self.temporal.analyze_temporal_patterns(
            entity_ids=entity_ids,
            time_window=timedelta(days=days_back),
            min_occurrences=3
        )
        
        return {
            "analysis_period_days": days_back,
            "entities_analyzed": len(entity_ids) if entity_ids else "all",
            "patterns_discovered": len(patterns),
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "entities": p.entities,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "description": p.description,
                    "metadata": p.metadata
                }
                for p in patterns
            ]
        }
    
    def create_policy(self,
                     name: str,
                     policy_type: str,
                     rules: Dict[str, Any],
                     created_by: str,
                     description: str = "",
                     enabled: bool = True) -> str:
        """
        Create governance policy.
        
        Args:
            name: Policy name
            policy_type: Type of policy (access, compliance, etc.)
            rules: Policy rules and conditions
            created_by: Policy creator
            description: Policy description
            enabled: Whether policy is active
            
        Returns:
            policy_id: Unique identifier
        """
        return self.governance.create_policy(
            name=name,
            policy_type=policy_type,
            description=description,
            rules=rules,
            created_by=created_by,
            enabled=enabled
        )
    
    def check_entity_access(self,
                          user_id: str,
                          entity_id: str,
                          access_level: str = "read") -> Dict[str, Any]:
        """
        Check user access to entity.
        
        Args:
            user_id: User identifier
            entity_id: Entity identifier
            access_level: Required access level
            
        Returns:
            Access decision with details
        """
        return self.governance.check_access(
            user_id=user_id,
            resource_id=entity_id,
            access_level=access_level
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all layers."""
        return {
            "hierarchical": self.hierarchical.get_stats(),
            "metadata": self.metadata.get_stats(),
            "semantic": self.semantic.get_stats(),
            "temporal": self.temporal.get_stats(),
            "governance": self.governance.get_stats(),
            "cross_layer": {
                "total_entities": len(self._entity_registry),
                "storage_path": str(self.storage_path),
                "layers_initialized": 5
            }
        }
    
    def save_all(self):
        """Save all layers to persistent storage."""
        self.hierarchical.save()
        self.metadata.save()
        self.semantic.save()
        self.temporal.save()
        self.governance.save()
    
    def cleanup(self, days_to_keep: int = 365):
        """Clean up old data across layers."""
        # Clean temporal data
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        self.temporal.cleanup_old_data(cutoff_date)
        
        # Save after cleanup
        self.save_all()
    
    # Helper methods
    
    def _rollback_entity_creation(self, entity_id: str):
        """Rollback failed entity creation."""
        try:
            self.hierarchical.remove_entity(entity_id)
            self.metadata.delete_metadata(entity_id, "system")
            if entity_id in self._entity_registry:
                del self._entity_registry[entity_id]
        except:
            pass  # Best effort cleanup
    
    def _calculate_change_summary(self,
                                 old_metadata: Optional[Dict[str, Any]],
                                 new_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary of changes between old and new metadata."""
        if not old_metadata:
            return {"type": "creation", "properties_added": list(new_properties.keys())}
        
        old_props = old_metadata.get("properties", {})
        
        added = []
        modified = []
        
        for key, value in new_properties.items():
            if key not in old_props:
                added.append(key)
            elif old_props[key] != value:
                modified.append(key)
        
        return {
            "type": "update",
            "properties_added": added,
            "properties_modified": modified,
            "total_changes": len(added) + len(modified)
        }
    
    def _get_entity_classification(self, entity_id: str) -> str:
        """Get entity classification level."""
        classification_df = self.governance.classifications_df.filter(
            pl.col("resource_id") == entity_id
        )
        
        if classification_df.height > 0:
            return classification_df.row(0, named=True)["classification"]
        
        return "unknown"
    
    def _get_compliance_status(self, entity_id: str) -> Dict[str, Any]:
        """Get entity compliance status."""
        compliance_df = self.governance.compliance_df.filter(
            pl.col("resource_id") == entity_id
        )
        
        if compliance_df.height > 0:
            latest = compliance_df.sort("last_checked", descending=True).limit(1)
            row = latest.row(0, named=True)
            return {
                "framework": row["framework"],
                "status": row["status"],
                "last_checked": row["last_checked"]
            }
        
        return {"status": "unknown"}

import polars as pl
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import orjson

from .hierarchical_store import HierarchicalStore
from .metadata_store import MetadataStore
from ..semantic.semantic_layer import SemanticLayer
from ..temporal.temporal_layer import TemporalLayer
from ..governance.policy_layer import PolicyEngine


class Metagraph:
    """
    Advanced Metagraph with hierarchical knowledge modeling and Polars+Parquet storage.
    
    A metagraph where nodes and edges can represent entire subgraphs, enabling
    sophisticated enterprise data modeling with semantic intelligence and temporal tracking.
    """
    
    def __init__(self,
                 storage_path: Optional[Path] = None,
                 hierarchical_data: Optional[Dict] = None,
                 enable_semantics: bool = True,
                 enable_temporal: bool = True,
                 enable_governance: bool = True,
                 **kwargs):
        """
        Initialize Metagraph with Polars+Parquet backend.
        
        Parameters
        ----------
        storage_path : Path, optional
            Directory for Polars+Parquet metadata storage
        hierarchical_data : Dict, optional
            Initial hierarchical data structure
        enable_semantics : bool
            Enable semantic layer and LLM integration
        enable_temporal : bool
            Enable temporal tracking and analytics
        enable_governance : bool
            Enable policy and governance layer
        """
        
        # Core identification
        self.metagraph_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Storage configuration
        self.storage_path = Path(storage_path) if storage_path else Path(f"./metagraph_data_{self.metagraph_id[:8]}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core stores with Polars+Parquet backend
        self.hierarchical_store = HierarchicalStore(
            storage_path=self.storage_path / "hierarchical",
            compression="zstd"  # High compression for metadata
        )
        
        self.metadata_store = MetadataStore(
            storage_path=self.storage_path / "metadata",
            compression="zstd"
        )
        
        # Initialize layers based on configuration
        self.semantic_layer = None
        self.temporal_layer = None
        self.policy_layer = None
        
        if enable_semantics:
            self.semantic_layer = SemanticLayer(
                metadata_store=self.metadata_store,
                storage_path=self.storage_path / "semantic"
            )
            
        if enable_temporal:
            self.temporal_layer = TemporalLayer(
                metadata_store=self.metadata_store,
                storage_path=self.storage_path / "temporal"
            )
            
        if enable_governance:
            self.policy_layer = PolicyEngine(
                metadata_store=self.metadata_store,
                storage_path=self.storage_path / "governance"
            )
        
        # Load initial data if provided
        if hierarchical_data:
            self._initialize_from_data(hierarchical_data)
            
        # Store metagraph configuration
        self._save_configuration(kwargs)
    
    def _initialize_from_data(self, hierarchical_data: Dict) -> None:
        """Initialize metagraph from hierarchical data structure."""
        
        # Process hierarchical structure
        if "levels" in hierarchical_data:
            for level_name, level_data in hierarchical_data["levels"].items():
                self.hierarchical_store.create_level(level_name, level_data)
        
        # Process relationships
        if "relationships" in hierarchical_data:
            for rel_type, relationships in hierarchical_data["relationships"].items():
                self.hierarchical_store.add_relationships(rel_type, relationships)
        
        # Process metadata
        if "metadata" in hierarchical_data:
            for entity_id, metadata in hierarchical_data["metadata"].items():
                self.metadata_store.store_entity_metadata(
                    entity_id=entity_id,
                    metadata=metadata,
                    entity_type=metadata.get("type", "unknown")
                )
    
    def _save_configuration(self, config: Dict[str, Any]) -> None:
        """Save metagraph configuration to Parquet."""
        
        config_data = {
            "metagraph_id": self.metagraph_id,
            "created_at": self.created_at,
            "storage_path": str(self.storage_path),
            "enabled_layers": {
                "semantic": self.semantic_layer is not None,
                "temporal": self.temporal_layer is not None,
                "governance": self.policy_layer is not None
            },
            "user_config": config
        }
        
        config_df = pl.DataFrame([{
            "config_id": self.metagraph_id,
            "config_data": orjson.dumps(config_data).decode(),
            "created_at": self.created_at,
            "updated_at": self.created_at
        }])
        
        config_path = self.storage_path / "metagraph_config.parquet"
        config_df.write_parquet(config_path, compression="zstd")
    
    # ===== HIERARCHICAL NAVIGATION =====
    
    def get_level(self, level_name: str) -> Optional[pl.DataFrame]:
        """Get all entities at a specific hierarchical level."""
        return self.hierarchical_store.get_level(level_name)
    
    def get_parent(self, entity_id: str) -> Optional[str]:
        """Get parent entity ID in hierarchy."""
        return self.hierarchical_store.get_parent(entity_id)
    
    def get_children(self, entity_id: str) -> List[str]:
        """Get child entity IDs in hierarchy."""
        return self.hierarchical_store.get_children(entity_id)
    
    def get_ancestors(self, entity_id: str) -> List[str]:
        """Get all ancestor entity IDs up the hierarchy."""
        return self.hierarchical_store.get_ancestors(entity_id)
    
    def get_descendants(self, entity_id: str) -> List[str]:
        """Get all descendant entity IDs down the hierarchy."""
        return self.hierarchical_store.get_descendants(entity_id)
    
    def navigate_path(self, from_entity: str, to_entity: str) -> Optional[List[str]]:
        """Find navigation path between two entities."""
        return self.hierarchical_store.find_path(from_entity, to_entity)
    
    # ===== BASIC SEMANTIC RELATIONSHIPS =====
    
    def add_semantic_relationship(self, 
                                from_entity: str, 
                                to_entity: str, 
                                relationship_type: str,
                                properties: Optional[Dict] = None) -> None:
        """Add semantic relationship between entities."""
        
        relationship_data = {
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relationship_type": relationship_type,
            "properties": properties or {},
            "created_at": datetime.now()
        }
        
        # Store in metadata store
        rel_id = f"rel_{uuid.uuid4()}"
        self.metadata_store.store_entity_metadata(
            entity_id=rel_id,
            metadata=relationship_data,
            entity_type="semantic_relationship"
        )
        
        # Add to semantic layer if enabled
        if self.semantic_layer:
            self.semantic_layer.add_relationship(
                from_entity, to_entity, relationship_type, properties
            )
    
    def get_semantic_relationships(self, 
                                 entity_id: str, 
                                 relationship_type: Optional[str] = None) -> pl.DataFrame:
        """Get semantic relationships for an entity."""
        
        if self.semantic_layer:
            return self.semantic_layer.get_relationships(entity_id, relationship_type)
        else:
            # Basic query from metadata store
            return self.metadata_store.query_entities(
                filters={"from_entity": entity_id},
                entity_type="semantic_relationship"
            )
    
    # ===== INITIAL TEMPORAL TRACKING =====
    
    def track_entity_change(self, 
                          entity_id: str, 
                          change_type: str,
                          old_value: Any = None,
                          new_value: Any = None,
                          metadata: Optional[Dict] = None) -> None:
        """Track temporal changes to entities."""
        
        if self.temporal_layer:
            self.temporal_layer.track_change(
                entity_id, change_type, old_value, new_value, metadata
            )
    
    def get_entity_history(self, entity_id: str) -> pl.DataFrame:
        """Get temporal history for an entity."""
        
        if self.temporal_layer:
            return self.temporal_layer.get_entity_history(entity_id)
        else:
            return pl.DataFrame()  # Empty if temporal tracking disabled
    
    def get_entity_at_time(self, entity_id: str, timestamp: datetime) -> Optional[Dict]:
        """Get entity state at specific time."""
        
        if self.temporal_layer:
            return self.temporal_layer.get_entity_at_time(entity_id, timestamp)
        else:
            return None
    
    # ===== METADATA OPERATIONS =====
    
    def store_metadata(self, 
                      entity_id: str, 
                      metadata: Dict[str, Any],
                      entity_type: str = "unknown") -> None:
        """Store metadata for an entity using Polars+Parquet."""
        
        self.metadata_store.store_entity_metadata(entity_id, metadata, entity_type)
        
        # Track the change if temporal layer is enabled
        self.track_entity_change(
            entity_id, "metadata_update", 
            metadata={"entity_type": entity_type}
        )
    
    def get_metadata(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for an entity."""
        return self.metadata_store.get_entity_metadata(entity_id)
    
    def query_entities(self, 
                      filters: Dict[str, Any],
                      entity_type: Optional[str] = None) -> pl.DataFrame:
        """Query entities based on metadata filters."""
        return self.metadata_store.query_entities(filters, entity_type)
    
    # ===== UTILITY METHODS =====
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of metagraph."""
        
        summary = {
            "metagraph_id": self.metagraph_id,
            "created_at": self.created_at,
            "storage_path": str(self.storage_path),
            "enabled_layers": {
                "semantic": self.semantic_layer is not None,
                "temporal": self.temporal_layer is not None,
                "governance": self.policy_layer is not None
            },
            "statistics": {
                "levels": self.hierarchical_store.get_level_count(),
                "entities": self.metadata_store.get_entity_count(),
                "relationships": self.metadata_store.get_entity_count("semantic_relationship")
            }
        }
        
        if self.semantic_layer:
            summary["semantic_stats"] = self.semantic_layer.get_statistics()
            
        if self.temporal_layer:
            summary["temporal_stats"] = self.temporal_layer.get_statistics()
            
        if self.policy_layer:
            summary["governance_stats"] = self.policy_layer.get_statistics()
        
        return summary
    
    def save_state(self) -> None:
        """Save complete metagraph state to Polars+Parquet files."""
        
        # Save individual components
        self.hierarchical_store.save_state()
        self.metadata_store.save_state()
        
        if self.semantic_layer:
            self.semantic_layer.save_state()
            
        if self.temporal_layer:
            self.temporal_layer.save_state()
            
        if self.policy_layer:
            self.policy_layer.save_state()
    
    @classmethod
    def load_from_storage(cls, storage_path: Path) -> 'Metagraph':
        """Load metagraph from existing Polars+Parquet storage."""
        
        config_path = storage_path / "metagraph_config.parquet"
        if not config_path.exists():
            raise FileNotFoundError(f"Metagraph config not found at {config_path}")
        
        config_df = pl.read_parquet(config_path)
        config_data = orjson.loads(config_df["config_data"][0])
        
        # Recreate metagraph with saved configuration
        enabled_layers = config_data["enabled_layers"]
        
        return cls(
            storage_path=storage_path,
            enable_semantics=enabled_layers["semantic"],
            enable_temporal=enabled_layers["temporal"],
            enable_governance=enabled_layers["governance"],
            **config_data["user_config"]
        )
    
    def __repr__(self) -> str:
        return f"Metagraph(id={self.metagraph_id[:8]}, storage={self.storage_path})"