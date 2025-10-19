"""
Metadata Store for Metagraph - Phase 1
=====================================

Polars+Parquet-based metadata storage with rich querying capabilities.
Optimized for enterprise metadata management with ZSTD compression.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
import orjson
import uuid

# Type alias for Polars compression
ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]


class MetadataStore:
    """
    Polars+Parquet-based metadata storage for metagraph entities.
    
    Provides efficient storage, retrieval, and querying of metadata
    with support for complex filters and schema evolution.
    """
    
    def __init__(self, 
                 storage_path: str = "./metagraph_metadata",
                 compression: ParquetCompression = "zstd"):
        """
        Initialize metadata store with Polars+Parquet backend.
        
        Parameters
        ----------
        storage_path : Path
            Directory for storing metadata
        compression : ParquetCompression
            Compression algorithm for Parquet files
        """
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression: ParquetCompression = compression
        
        # Initialize storage structure
        self._setup_storage_structure()
        
        # Load existing data
        self._load_existing_data()
    
    def _setup_storage_structure(self) -> None:
        """Create directory structure for metadata storage."""
        
        directories = [
            "entities",      # Entity metadata by type
            "schemas",       # Schema definitions and validation
            "indexes",       # Search indexes and views
            "versions"       # Versioned metadata
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def _load_existing_data(self) -> None:
        """Load existing metadata from storage."""
        
        # Load main metadata table
        metadata_file = self.storage_path / "entities" / "metadata.parquet"
        if metadata_file.exists():
            self.metadata_df = pl.read_parquet(metadata_file)
        else:
            self.metadata_df = pl.DataFrame({
                "entity_id": [],
                "entity_type": [],
                "metadata_json": [],  # JSON string of metadata
                "created_at": [],
                "updated_at": [],
                "version": []
            }, schema={
                "entity_id": pl.Utf8,
                "entity_type": pl.Utf8,
                "metadata_json": pl.Utf8,
                "created_at": pl.Datetime,
                "updated_at": pl.Datetime,
                "version": pl.Int32
            })
        
        # Load schema registry
        schema_file = self.storage_path / "schemas" / "registry.parquet"
        if schema_file.exists():
            self.schema_df = pl.read_parquet(schema_file)
        else:
            self.schema_df = pl.DataFrame({
                "entity_type": [],
                "schema_version": [],
                "schema_definition": [],  # JSON string
                "created_at": [],
                "is_active": []
            }, schema={
                "entity_type": pl.Utf8,
                "schema_version": pl.Int32,
                "schema_definition": pl.Utf8,
                "created_at": pl.Datetime,
                "is_active": pl.Boolean
            })
    
    def store_entity_metadata(self,
                            entity_id: str,
                            metadata: Dict[str, Any],
                            entity_type: str = "unknown") -> None:
        """
        Store metadata for an entity.
        
        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        metadata : Dict
            Metadata to store
        entity_type : str
            Type of entity for organization
        """
        
        # Check if entity already exists
        existing = self.metadata_df.filter(pl.col("entity_id") == entity_id)
        
        if existing.height > 0:
            # Update existing entity
            current_version = existing["version"][0]
            new_version = current_version + 1
            
            # Remove old record
            self.metadata_df = self.metadata_df.filter(pl.col("entity_id") != entity_id)
        else:
            # New entity
            new_version = 1
        
        # Create new record
        new_record = pl.DataFrame([{
            "entity_id": entity_id,
            "entity_type": entity_type,
            "metadata_json": orjson.dumps(metadata).decode(),
            "created_at": existing["created_at"][0] if existing.height > 0 else datetime.now(),
            "updated_at": datetime.now(),
            "version": new_version
        }])
        
        # Add to metadata DataFrame
        self.metadata_df = pl.concat([self.metadata_df, new_record])
        
        # Save to storage
        self._save_metadata()
        
        # Validate against schema if exists
        self._validate_metadata(entity_type, metadata)
    
    def get_entity_metadata(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity identifier
            
        Returns
        -------
        Dict or None
            Entity metadata or None if not found
        """
        
        if self.metadata_df.height == 0:
            return None
        
        entity_record = self.metadata_df.filter(pl.col("entity_id") == entity_id)
        
        if entity_record.height == 0:
            return None
        
        metadata_json = entity_record["metadata_json"][0]
        return orjson.loads(metadata_json)
    
    def query_entities(self,
                      filters: Dict[str, Any],
                      entity_type: Optional[str] = None) -> pl.DataFrame:
        """
        Query entities based on metadata filters.
        
        Parameters
        ----------
        filters : Dict
            Filters to apply to metadata
        entity_type : str, optional
            Filter by entity type
            
        Returns
        -------
        pl.DataFrame
            Matching entities
        """
        
        # Start with all entities or filter by type
        if entity_type:
            result_df = self.metadata_df.filter(pl.col("entity_type") == entity_type)
        else:
            result_df = self.metadata_df
        
        if result_df.height == 0:
            return result_df
        
        # Apply metadata filters
        for filter_key, filter_value in filters.items():
            if isinstance(filter_value, dict):
                # Handle complex filters (e.g., {"contains": "value"})
                if "contains" in filter_value:
                    # JSON contains filter
                    result_df = result_df.filter(
                        pl.col("metadata_json").str.contains(str(filter_value["contains"]))
                    )
                elif "equals" in filter_value:
                    # JSON equals filter (exact match)
                    search_pattern = f'"{filter_key}":"{filter_value["equals"]}"'
                    result_df = result_df.filter(
                        pl.col("metadata_json").str.contains(search_pattern)
                    )
            else:
                # Simple string contains filter
                search_pattern = f'"{filter_key}":"{filter_value}"'
                result_df = result_df.filter(
                    pl.col("metadata_json").str.contains(search_pattern)
                )
        
        return result_df
    
    def get_entities_by_type(self, entity_type: str) -> pl.DataFrame:
        """Get all entities of a specific type."""
        
        return self.metadata_df.filter(pl.col("entity_type") == entity_type)
    
    def get_entity_types(self) -> List[str]:
        """Get list of all entity types."""
        
        if self.metadata_df.height == 0:
            return []
        
        return self.metadata_df["entity_type"].unique().to_list()
    
    def get_entity_count(self, entity_type: Optional[str] = None) -> int:
        """Get count of entities, optionally filtered by type."""
        
        if entity_type:
            return self.metadata_df.filter(pl.col("entity_type") == entity_type).height
        else:
            return self.metadata_df.height
    
    def register_schema(self,
                       entity_type: str,
                       schema_definition: Dict[str, Any]) -> None:
        """
        Register a schema for an entity type.
        
        Parameters
        ----------
        entity_type : str
            Type of entity
        schema_definition : Dict
            JSON schema definition
        """
        
        # Deactivate previous schemas for this type
        self.schema_df = self.schema_df.with_columns([
            pl.when(pl.col("entity_type") == entity_type)
            .then(pl.lit(False))
            .otherwise(pl.col("is_active"))
            .alias("is_active")
        ])
        
        # Determine next version - simplified approach
        existing_versions = self.schema_df.filter(pl.col("entity_type") == entity_type)
        if existing_versions.height > 0:
            # Get the maximum version and add 1
            version_list = existing_versions["schema_version"].to_list()
            next_version = max(version_list) + 1 if version_list else 1
        else:
            next_version = 1
        
        # Add new schema
        new_schema = pl.DataFrame([{
            "entity_type": entity_type,
            "schema_version": next_version,
            "schema_definition": orjson.dumps(schema_definition).decode(),
            "created_at": datetime.now(),
            "is_active": True
        }])
        
        self.schema_df = pl.concat([self.schema_df, new_schema])
        
        # Save schema registry
        self._save_schemas()
    
    def _validate_metadata(self, entity_type: str, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against registered schema."""
        
        # Get active schema for entity type
        active_schema = self.schema_df.filter(
            (pl.col("entity_type") == entity_type) & 
            (pl.col("is_active") == True)
        )
        
        if active_schema.height == 0:
            return True  # No schema to validate against
        
        # Basic validation (can be extended with jsonschema library)
        schema_def = orjson.loads(active_schema["schema_definition"][0])
        
        # Simple required fields validation
        if "required" in schema_def:
            for required_field in schema_def["required"]:
                if required_field not in metadata:
                    raise ValueError(f"Required field '{required_field}' missing for entity type '{entity_type}'")
        
        return True
    
    def bulk_store_metadata(self, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Bulk store multiple metadata records efficiently.
        
        Parameters
        ----------
        metadata_list : List[Dict]
            List of metadata records with keys: entity_id, metadata, entity_type
        """
        
        if not metadata_list:
            return
        
        # Prepare bulk data
        bulk_records = []
        for record in metadata_list:
            bulk_records.append({
                "entity_id": record["entity_id"],
                "entity_type": record.get("entity_type", "unknown"),
                "metadata_json": orjson.dumps(record["metadata"]).decode(),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "version": 1  # Simplified for bulk insert
            })
        
        # Create DataFrame and append
        bulk_df = pl.DataFrame(bulk_records)
        self.metadata_df = pl.concat([self.metadata_df, bulk_df])
        
        # Save to storage
        self._save_metadata()
    
    def search_metadata_content(self, search_term: str) -> pl.DataFrame:
        """
        Full-text search across all metadata content.
        
        Parameters
        ----------
        search_term : str
            Term to search for
            
        Returns
        -------
        pl.DataFrame
            Entities containing the search term
        """
        
        return self.metadata_df.filter(
            pl.col("metadata_json").str.contains(search_term)
        )
    
    def get_entity_versions(self, entity_id: str) -> pl.DataFrame:
        """Get all versions of an entity's metadata."""
        
        # This would require a separate versions table in a full implementation
        # For now, return current version
        return self.metadata_df.filter(pl.col("entity_id") == entity_id)
    
    def _save_metadata(self) -> None:
        """Save metadata to Parquet file."""
        metadata_file = self.storage_path / "entities" / "metadata.parquet"
        self.metadata_df.write_parquet(metadata_file, compression=self.compression)
    
    def _save_schemas(self) -> None:
        """Save schema registry to Parquet file."""
        schema_file = self.storage_path / "schemas" / "registry.parquet"
        self.schema_df.write_parquet(schema_file, compression=self.compression)
    
    def save_state(self) -> None:
        """Save complete metadata store state."""
        self._save_metadata()
        self._save_schemas()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata store statistics."""
        
        stats = {
            "total_entities": self.get_entity_count(),
            "entity_types": len(self.get_entity_types()),
            "schemas_registered": self.schema_df.height,
            "storage_path": str(self.storage_path)
        }
        
        # Entity type breakdown
        type_breakdown = {}
        for entity_type in self.get_entity_types():
            type_breakdown[entity_type] = self.get_entity_count(entity_type)
        
        stats["type_breakdown"] = type_breakdown
        
        return stats
    
    def optimize_storage(self) -> None:
        """Optimize storage by removing old versions and compacting files."""
        
        # Remove duplicate entity_ids, keeping only the latest version
        latest_versions = (
            self.metadata_df
            .group_by("entity_id")
            .agg(pl.col("version").max().alias("max_version"))
        )
        
        self.metadata_df = self.metadata_df.join(
            latest_versions,
            on="entity_id"
        ).filter(
            pl.col("version") == pl.col("max_version")
        ).drop("max_version")
        
        # Save optimized data
        self._save_metadata()
    
    def export_metadata(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """Export metadata for backup or migration."""
        
        if entity_type:
            export_df = self.get_entities_by_type(entity_type)
        else:
            export_df = self.metadata_df
        
        export_data = {
            "metadata": export_df.to_dicts(),
            "schemas": self.schema_df.to_dicts(),
            "export_timestamp": datetime.now().isoformat(),
            "total_records": export_df.height
        }
        
        return export_data
    
    def import_metadata(self, import_data: Dict[str, Any]) -> None:
        """Import metadata from backup or migration."""
        
        # Import metadata records
        if "metadata" in import_data:
            metadata_records = import_data["metadata"]
            import_df = pl.DataFrame(metadata_records)
            self.metadata_df = pl.concat([self.metadata_df, import_df])
        
        # Import schemas
        if "schemas" in import_data:
            schema_records = import_data["schemas"]
            if schema_records:
                import_schema_df = pl.DataFrame(schema_records)
                self.schema_df = pl.concat([self.schema_df, import_schema_df])
        
        # Save imported data
        self.save_state()