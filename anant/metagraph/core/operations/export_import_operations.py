"""
Export/Import Operations Module
===============================

Data export and import operations including:
- Multiple format support (JSON, CSV, Parquet, Graph formats)
- Bulk import/export capabilities
- Data transformation and validation
- Schema mapping and migration
- Performance optimization for large datasets
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Literal, IO
from datetime import datetime
import json
import csv
import uuid
import logging
import zipfile
import tempfile
from io import StringIO, BytesIO

from ....exceptions import (
    ValidationError, GraphError, handle_exception, 
    require_not_none, require_valid_string, require_valid_dict
)

# Define specific exceptions for this module
class AnantImportError(GraphError):
    """Exception raised during import operations."""
    pass

class ExportError(GraphError):
    """Exception raised during export operations."""
    pass

logger = logging.getLogger(__name__)


class ExportImportOperations:
    """
    Handles data export and import operations for the Metagraph.
    
    Provides multiple format support, bulk operations, validation,
    and performance optimization with proper error handling.
    """
    
    def __init__(self, storage_path: str, metadata_store, chunk_size: int = 10000):
        """
        Initialize export/import operations.
        
        Args:
            storage_path: Path to store export/import data
            metadata_store: Reference to metadata storage system
            chunk_size: Size of chunks for processing large datasets
        """
        self.storage_path = Path(storage_path)
        self.metadata_store = metadata_store
        self.chunk_size = chunk_size
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Supported formats
        self.supported_export_formats = [
            "json", "csv", "parquet", "jsonl", "graphml", "gexf", "pickle", "yaml"
        ]
        self.supported_import_formats = [
            "json", "csv", "parquet", "jsonl", "graphml", "gexf", "pickle", "yaml"
        ]
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def export_entities(self,
                       output_path: str,
                       format_type: str = "json",
                       entity_types: Optional[List[str]] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       include_relationships: bool = True,
                       compress: bool = False) -> Dict[str, Any]:
        """
        Export entities to file.
        
        Args:
            output_path: Path to write exported data
            format_type: Export format (json, csv, parquet, etc.)
            entity_types: Optional filter for specific entity types
            filters: Optional additional filters
            include_relationships: Whether to include relationship data
            compress: Whether to compress the output
            
        Returns:
            Export summary and statistics
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            output_path = require_valid_string(output_path, "output_path")
            if format_type not in self.supported_export_formats:
                raise ValidationError(
                    f"Unsupported export format: {format_type}",
                    error_code="UNSUPPORTED_EXPORT_FORMAT",
                    context={"format_type": format_type, "supported": self.supported_export_formats}
                )
            
            # Get entities to export
            all_entities = self.metadata_store.get_all_entities()
            
            # Apply filters
            entities_to_export = self._filter_entities(all_entities, entity_types, filters)
            
            # Get relationships if requested
            relationships = []
            if include_relationships:
                relationships = self._get_entity_relationships(entities_to_export)
            
            # Prepare export data
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "format_version": "1.0",
                    "entity_count": len(entities_to_export),
                    "relationship_count": len(relationships),
                    "entity_types": list(set(e.get("entity_type", "unknown") for e in entities_to_export)),
                    "filters_applied": {
                        "entity_types": entity_types,
                        "custom_filters": filters is not None
                    }
                },
                "entities": entities_to_export,
                "relationships": relationships
            }
            
            # Export based on format
            output_file_path = Path(output_path)
            if format_type == "json":
                self._export_json(export_data, output_file_path, compress)
            elif format_type == "csv":
                self._export_csv(export_data, output_file_path, compress)
            elif format_type == "parquet":
                self._export_parquet(export_data, output_file_path)
            elif format_type == "jsonl":
                self._export_jsonl(export_data, output_file_path, compress)
            elif format_type == "graphml":
                self._export_graphml(export_data, output_file_path)
            elif format_type == "gexf":
                self._export_gexf(export_data, output_file_path)
            elif format_type == "pickle":
                self._export_pickle(export_data, output_file_path, compress)
            elif format_type == "yaml":
                self._export_yaml(export_data, output_file_path, compress)
            
            # Create export summary
            summary = {
                "export_timestamp": export_data["metadata"]["export_timestamp"],
                "output_path": str(output_file_path),
                "format": format_type,
                "compressed": compress,
                "entities_exported": len(entities_to_export),
                "relationships_exported": len(relationships),
                "file_size_bytes": output_file_path.stat().st_size if output_file_path.exists() else 0,
                "entity_types_exported": export_data["metadata"]["entity_types"]
            }
            
            self.logger.info(
                "Export completed successfully",
                extra={
                    "output_path": output_path,
                    "format": format_type,
                    "entities_exported": len(entities_to_export),
                    "file_size": summary["file_size_bytes"]
                }
            )
            
            return summary
            
        except (ExportError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"exporting entities to '{output_path}'", e, {
                "output_path": output_path,
                "format_type": format_type,
                "entity_types": entity_types
            })
    
    def import_entities(self,
                       input_path: str,
                       format_type: Optional[str] = None,
                       validation_mode: str = "strict",
                       update_existing: bool = False,
                       batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Import entities from file.
        
        Args:
            input_path: Path to file to import
            format_type: Import format (auto-detected if None)
            validation_mode: How strictly to validate (strict, lenient, skip)
            update_existing: Whether to update existing entities
            batch_size: Size of batches for processing (uses default if None)
            
        Returns:
            Import summary and statistics
            
        Raises:
            AnantImportError: If import fails
        """
        try:
            # Validate inputs
            input_path = require_valid_string(input_path, "input_path")
            input_file_path = Path(input_path)
            
            if not input_file_path.exists():
                raise AnantImportError(
                    f"Input file does not exist: {input_path}",
                    error_code="FILE_NOT_FOUND",
                    context={"input_path": input_path}
                )
            
            # Auto-detect format if not specified
            if format_type is None:
                format_type = self._detect_format(input_file_path)
            
            if format_type not in self.supported_import_formats:
                raise ValidationError(
                    f"Unsupported import format: {format_type}",
                    error_code="UNSUPPORTED_IMPORT_FORMAT",
                    context={"format_type": format_type, "supported": self.supported_import_formats}
                )
            
            if validation_mode not in ["strict", "lenient", "skip"]:
                raise ValidationError(
                    f"Invalid validation mode: {validation_mode}",
                    error_code="INVALID_VALIDATION_MODE",
                    context={"validation_mode": validation_mode}
                )
            
            # Import based on format
            import_data = None
            if format_type == "json":
                import_data = self._import_json(input_file_path)
            elif format_type == "csv":
                import_data = self._import_csv(input_file_path)
            elif format_type == "parquet":
                import_data = self._import_parquet(input_file_path)
            elif format_type == "jsonl":
                import_data = self._import_jsonl(input_file_path)
            elif format_type == "graphml":
                import_data = self._import_graphml(input_file_path)
            elif format_type == "gexf":
                import_data = self._import_gexf(input_file_path)
            elif format_type == "pickle":
                import_data = self._import_pickle(input_file_path)
            elif format_type == "yaml":
                import_data = self._import_yaml(input_file_path)
            
            if not import_data:
                raise AnantImportError(
                    f"Failed to parse import data from {input_path}",
                    error_code="PARSE_FAILED",
                    context={"input_path": input_path, "format": format_type}
                )
            
            # Process imported data
            entities = import_data.get("entities", [])
            relationships = import_data.get("relationships", [])
            
            # Validate data
            validation_results = self._validate_import_data(entities, relationships, validation_mode)
            
            # Process entities in batches
            batch_size = batch_size or self.chunk_size
            import_summary = {
                "import_timestamp": datetime.now().isoformat(),
                "input_path": str(input_file_path),
                "format": format_type,
                "validation_mode": validation_mode,
                "entities_processed": 0,
                "entities_created": 0,
                "entities_updated": 0,
                "entities_failed": 0,
                "relationships_processed": 0,
                "relationships_created": 0,
                "relationships_failed": 0,
                "validation_errors": validation_results["errors"],
                "validation_warnings": validation_results["warnings"]
            }
            
            # Import entities in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                batch_results = self._process_entity_batch(batch, update_existing, validation_mode)
                
                import_summary["entities_processed"] += batch_results["processed"]
                import_summary["entities_created"] += batch_results["created"]
                import_summary["entities_updated"] += batch_results["updated"]
                import_summary["entities_failed"] += batch_results["failed"]
            
            # Import relationships
            if relationships:
                rel_results = self._process_relationships(relationships, validation_mode)
                import_summary["relationships_processed"] = rel_results["processed"]
                import_summary["relationships_created"] = rel_results["created"]
                import_summary["relationships_failed"] = rel_results["failed"]
            
            self.logger.info(
                "Import completed successfully",
                extra={
                    "input_path": input_path,
                    "format": format_type,
                    "entities_created": import_summary["entities_created"],
                    "entities_updated": import_summary["entities_updated"],
                    "entities_failed": import_summary["entities_failed"]
                }
            )
            
            return import_summary
            
        except (AnantImportError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"importing entities from '{input_path}'", e, {
                "input_path": input_path,
                "format_type": format_type
            })
    
    def bulk_export(self,
                   entity_collections: Dict[str, Dict[str, Any]],
                   output_directory: str,
                   format_type: str = "json",
                   parallel: bool = False) -> Dict[str, Any]:
        """
        Export multiple entity collections in bulk.
        
        Args:
            entity_collections: Dict mapping collection names to export configs
            output_directory: Directory to write exports
            format_type: Export format
            parallel: Whether to export in parallel
            
        Returns:
            Bulk export summary
            
        Raises:
            ExportError: If bulk export fails
        """
        try:
            # Validate inputs
            entity_collections = require_valid_dict(entity_collections, "entity_collections")
            output_directory = require_valid_string(output_directory, "output_directory")
            
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            bulk_summary = {
                "export_timestamp": datetime.now().isoformat(),
                "output_directory": str(output_dir),
                "format": format_type,
                "parallel": parallel,
                "collections_processed": 0,
                "collections_succeeded": 0,
                "collections_failed": 0,
                "total_entities_exported": 0,
                "collection_results": {}
            }
            
            # Process each collection
            for collection_name, config in entity_collections.items():
                try:
                    bulk_summary["collections_processed"] += 1
                    
                    # Prepare output path
                    output_file = output_dir / f"{collection_name}.{format_type}"
                    
                    # Export collection
                    export_result = self.export_entities(
                        output_path=str(output_file),
                        format_type=format_type,
                        entity_types=config.get("entity_types"),
                        filters=config.get("filters"),
                        include_relationships=config.get("include_relationships", True),
                        compress=config.get("compress", False)
                    )
                    
                    bulk_summary["collections_succeeded"] += 1
                    bulk_summary["total_entities_exported"] += export_result["entities_exported"]
                    bulk_summary["collection_results"][collection_name] = {
                        "status": "success",
                        "entities_exported": export_result["entities_exported"],
                        "file_path": str(output_file),
                        "file_size": export_result["file_size_bytes"]
                    }
                    
                except Exception as e:
                    bulk_summary["collections_failed"] += 1
                    bulk_summary["collection_results"][collection_name] = {
                        "status": "failed",
                        "error": str(e),
                        "entities_exported": 0
                    }
                    
                    if not parallel:  # Re-raise in sequential mode
                        raise
            
            return bulk_summary
            
        except (ExportError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception("performing bulk export", e, {
                "collections": list(entity_collections.keys()),
                "output_directory": output_directory
            })
    
    def transform_data(self,
                      data: Dict[str, Any],
                      transformation_rules: Dict[str, Any],
                      validate_output: bool = True) -> Dict[str, Any]:
        """
        Transform imported/exported data using transformation rules.
        
        Args:
            data: Data to transform
            transformation_rules: Rules for transformation
            validate_output: Whether to validate transformed data
            
        Returns:
            Transformed data
            
        Raises:
            ValidationError: If transformation fails
        """
        try:
            # Validate inputs
            data = require_valid_dict(data, "data")
            transformation_rules = require_valid_dict(transformation_rules, "transformation_rules")
            
            transformed_data = {
                "metadata": data.get("metadata", {}),
                "entities": [],
                "relationships": data.get("relationships", [])
            }
            
            # Transform entities
            entity_rules = transformation_rules.get("entity_transformations", {})
            for entity in data.get("entities", []):
                transformed_entity = self._transform_entity(entity, entity_rules)
                if transformed_entity:
                    transformed_data["entities"].append(transformed_entity)
            
            # Transform relationships
            relationship_rules = transformation_rules.get("relationship_transformations", {})
            transformed_relationships = []
            for relationship in data.get("relationships", []):
                transformed_rel = self._transform_relationship(relationship, relationship_rules)
                if transformed_rel:
                    transformed_relationships.append(transformed_rel)
            
            transformed_data["relationships"] = transformed_relationships
            
            # Validate output if requested
            if validate_output:
                validation_results = self._validate_import_data(
                    transformed_data["entities"],
                    transformed_data["relationships"],
                    "strict"
                )
                
                if validation_results["errors"]:
                    raise ValidationError(
                        "Transformed data failed validation",
                        error_code="TRANSFORMATION_VALIDATION_FAILED",
                        context={
                            "errors": validation_results["errors"][:5],  # First 5 errors
                            "total_errors": len(validation_results["errors"])
                        }
                    )
            
            return transformed_data
            
        except ValidationError:
            raise
        except Exception as e:
            raise handle_exception("transforming data", e, {
                "entity_count": len(data.get("entities", [])),
                "relationship_count": len(data.get("relationships", []))
            })
    
    def _filter_entities(self, entities: List[Dict[str, Any]], 
                        entity_types: Optional[List[str]], 
                        filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter entities based on criteria."""
        filtered = entities
        
        # Filter by entity types
        if entity_types:
            filtered = [e for e in filtered if e.get("entity_type") in entity_types]
        
        # Apply custom filters
        if filters:
            for filter_key, filter_value in filters.items():
                if filter_key == "created_after":
                    cutoff_date = datetime.fromisoformat(filter_value)
                    filtered = [e for e in filtered 
                              if datetime.fromisoformat(e.get("created_at", "1970-01-01")) > cutoff_date]
                elif filter_key == "has_property":
                    filtered = [e for e in filtered 
                              if filter_value in e.get("properties", {})]
                elif filter_key == "property_value":
                    prop_name, prop_value = filter_value
                    filtered = [e for e in filtered 
                              if e.get("properties", {}).get(prop_name) == prop_value]
        
        return filtered
    
    def _get_entity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get relationships for entities."""
        entity_ids = {e["entity_id"] for e in entities}
        all_relationships = self.metadata_store.get_all_relationships()
        
        # Filter relationships that involve the exported entities
        relevant_relationships = []
        for rel in all_relationships:
            if rel.get("source_entity_id") in entity_ids or rel.get("target_entity_id") in entity_ids:
                relevant_relationships.append(rel)
        
        return relevant_relationships
    
    def _export_json(self, data: Dict[str, Any], output_path: Path, compress: bool):
        """Export data as JSON."""
        if compress:
            with zipfile.ZipFile(f"{output_path}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                with zf.open(output_path.name, 'w') as f:
                    f.write(json.dumps(data, indent=2, default=str).encode())
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, data: Dict[str, Any], output_path: Path, compress: bool):
        """Export entities as CSV (flattened)."""
        entities = data.get("entities", [])
        if not entities:
            return
        
        # Flatten entities for CSV
        flattened = []
        for entity in entities:
            flat_entity = {
                "entity_id": entity["entity_id"],
                "entity_type": entity.get("entity_type", ""),
                "created_at": entity.get("created_at", ""),
                "updated_at": entity.get("updated_at", "")
            }
            
            # Flatten properties
            properties = entity.get("properties", {})
            for prop_key, prop_value in properties.items():
                flat_entity[f"prop_{prop_key}"] = str(prop_value) if prop_value is not None else ""
            
            flattened.append(flat_entity)
        
        # Write CSV
        if flattened:
            fieldnames = list(flattened[0].keys())
            
            if compress:
                with zipfile.ZipFile(f"{output_path}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                    with zf.open(output_path.name, 'w') as f:
                        csv_content = StringIO()
                        writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened)
                        f.write(csv_content.getvalue().encode())
            else:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened)
    
    def _export_parquet(self, data: Dict[str, Any], output_path: Path):
        """Export entities as Parquet."""
        entities = data.get("entities", [])
        if not entities:
            return
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(entities)
        df.write_parquet(output_path)
    
    def _export_jsonl(self, data: Dict[str, Any], output_path: Path, compress: bool):
        """Export entities as JSON Lines."""
        entities = data.get("entities", [])
        
        if compress:
            with zipfile.ZipFile(f"{output_path}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                with zf.open(output_path.name, 'w') as f:
                    for entity in entities:
                        f.write((json.dumps(entity, default=str) + '\n').encode())
        else:
            with open(output_path, 'w') as f:
                for entity in entities:
                    f.write(json.dumps(entity, default=str) + '\n')
    
    def _export_pickle(self, data: Dict[str, Any], output_path: Path, compress: bool):
        """Export data as Pickle."""
        import pickle
        
        if compress:
            with zipfile.ZipFile(f"{output_path}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                # Write pickle data to a temporary buffer first
                pickle_data = pickle.dumps(data)
                zf.writestr(output_path.name, pickle_data)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    
    def _export_yaml(self, data: Dict[str, Any], output_path: Path, compress: bool):
        """Export data as YAML."""
        import yaml
        
        if compress:
            with zipfile.ZipFile(f"{output_path}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                with zf.open(output_path.name, 'w') as f:
                    f.write(yaml.dump(data, default_flow_style=False).encode())
        else:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
    
    def _export_graphml(self, data: Dict[str, Any], output_path: Path):
        """Export as GraphML format."""
        # Simplified GraphML export - in practice would use networkx or similar
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        graphml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        graphml_content += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        graphml_content += '  <graph id="G" edgedefault="directed">\n'
        
        # Add nodes (entities)
        for entity in entities:
            graphml_content += f'    <node id="{entity["entity_id"]}">\n'
            graphml_content += f'      <data key="type">{entity.get("entity_type", "")}</data>\n'
            graphml_content += '    </node>\n'
        
        # Add edges (relationships)
        for rel in relationships:
            source = rel.get("source_entity_id", "")
            target = rel.get("target_entity_id", "")
            rel_type = rel.get("relationship_type", "")
            graphml_content += f'    <edge source="{source}" target="{target}">\n'
            graphml_content += f'      <data key="type">{rel_type}</data>\n'
            graphml_content += '    </edge>\n'
        
        graphml_content += '  </graph>\n'
        graphml_content += '</graphml>\n'
        
        with open(output_path, 'w') as f:
            f.write(graphml_content)
    
    def _export_gexf(self, data: Dict[str, Any], output_path: Path):
        """Export as GEXF format."""
        # Simplified GEXF export
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        gexf_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        gexf_content += '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n'
        gexf_content += '  <graph mode="static" defaultedgetype="directed">\n'
        gexf_content += '    <nodes>\n'
        
        # Add nodes
        for entity in entities:
            label = entity.get("properties", {}).get("name", entity["entity_id"])
            gexf_content += f'      <node id="{entity["entity_id"]}" label="{label}"/>\n'
        
        gexf_content += '    </nodes>\n'
        gexf_content += '    <edges>\n'
        
        # Add edges
        for i, rel in enumerate(relationships):
            source = rel.get("source_entity_id", "")
            target = rel.get("target_entity_id", "")
            gexf_content += f'      <edge id="{i}" source="{source}" target="{target}"/>\n'
        
        gexf_content += '    </edges>\n'
        gexf_content += '  </graph>\n'
        gexf_content += '</gexf>\n'
        
        with open(output_path, 'w') as f:
            f.write(gexf_content)
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect file format from extension."""
        extension = file_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".csv": "csv",
            ".parquet": "parquet",
            ".jsonl": "jsonl",
            ".graphml": "graphml",
            ".gexf": "gexf",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".yaml": "yaml",
            ".yml": "yaml"
        }
        
        return format_map.get(extension, "json")
    
    def _import_json(self, file_path: Path) -> Dict[str, Any]:
        """Import from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _import_csv(self, file_path: Path) -> Dict[str, Any]:
        """Import from CSV file."""
        df = pl.read_csv(file_path)
        
        # Convert back to entity format
        entities = []
        for row in df.to_dicts():
            entity = {
                "entity_id": row.get("entity_id", str(uuid.uuid4())),
                "entity_type": row.get("entity_type", "unknown"),
                "created_at": row.get("created_at", datetime.now().isoformat()),
                "updated_at": row.get("updated_at", datetime.now().isoformat()),
                "properties": {}
            }
            
            # Extract properties
            for key, value in row.items():
                if key.startswith("prop_") and value:
                    prop_name = key[5:]  # Remove "prop_" prefix
                    entity["properties"][prop_name] = value
            
            entities.append(entity)
        
        return {"entities": entities, "relationships": []}
    
    def _import_parquet(self, file_path: Path) -> Dict[str, Any]:
        """Import from Parquet file."""
        df = pl.read_parquet(file_path)
        entities = df.to_dicts()
        return {"entities": entities, "relationships": []}
    
    def _import_jsonl(self, file_path: Path) -> Dict[str, Any]:
        """Import from JSON Lines file."""
        entities = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    entities.append(json.loads(line))
        
        return {"entities": entities, "relationships": []}
    
    def _import_pickle(self, file_path: Path) -> Dict[str, Any]:
        """Import from Pickle file."""
        import pickle
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _import_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Import from YAML file."""
        import yaml
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _import_graphml(self, file_path: Path) -> Dict[str, Any]:
        """Import from GraphML file."""
        # Simplified GraphML import - would use networkx in practice
        entities = []
        relationships = []
        
        # Basic XML parsing for demo
        with open(file_path, 'r') as f:
            content = f.read()
            
        # This is a very basic implementation
        # In practice, would use proper XML parsing
        return {"entities": entities, "relationships": relationships}
    
    def _import_gexf(self, file_path: Path) -> Dict[str, Any]:
        """Import from GEXF file."""
        # Simplified GEXF import
        entities = []
        relationships = []
        
        # Basic XML parsing for demo
        with open(file_path, 'r') as f:
            content = f.read()
        
        return {"entities": entities, "relationships": relationships}
    
    def _validate_import_data(self, entities: List[Dict[str, Any]], 
                             relationships: List[Dict[str, Any]], 
                             validation_mode: str) -> Dict[str, Any]:
        """Validate imported data."""
        errors = []
        warnings = []
        
        # Validate entities
        for i, entity in enumerate(entities):
            if "entity_id" not in entity:
                error_msg = f"Entity {i}: Missing required field 'entity_id'"
                if validation_mode == "strict":
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
            
            if "entity_type" not in entity and validation_mode in ["strict", "lenient"]:
                warning_msg = f"Entity {entity.get('entity_id', i)}: Missing 'entity_type'"
                warnings.append(warning_msg)
        
        # Validate relationships
        entity_ids = {e.get("entity_id") for e in entities if "entity_id" in e}
        for i, rel in enumerate(relationships):
            source_id = rel.get("source_entity_id")
            target_id = rel.get("target_entity_id")
            
            if not source_id or not target_id:
                error_msg = f"Relationship {i}: Missing source or target entity ID"
                if validation_mode == "strict":
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
            
            # Check if referenced entities exist
            if source_id and source_id not in entity_ids:
                warning_msg = f"Relationship {i}: Source entity {source_id} not found in import"
                warnings.append(warning_msg)
            
            if target_id and target_id not in entity_ids:
                warning_msg = f"Relationship {i}: Target entity {target_id} not found in import"
                warnings.append(warning_msg)
        
        return {"errors": errors, "warnings": warnings}
    
    def _process_entity_batch(self, entities: List[Dict[str, Any]], 
                             update_existing: bool, validation_mode: str) -> Dict[str, int]:
        """Process a batch of entities."""
        results = {"processed": 0, "created": 0, "updated": 0, "failed": 0}
        
        for entity in entities:
            try:
                results["processed"] += 1
                entity_id = entity["entity_id"]
                
                # Check if entity exists
                existing_entity = self.metadata_store.get_entity(entity_id)
                
                if existing_entity and update_existing:
                    # Update existing entity
                    self.metadata_store.update_entity(entity_id, entity)
                    results["updated"] += 1
                elif not existing_entity:
                    # Create new entity
                    self.metadata_store.create_entity(entity)
                    results["created"] += 1
                
            except Exception as e:
                results["failed"] += 1
                if validation_mode == "strict":
                    raise
        
        return results
    
    def _process_relationships(self, relationships: List[Dict[str, Any]], 
                              validation_mode: str) -> Dict[str, int]:
        """Process relationships."""
        results = {"processed": 0, "created": 0, "failed": 0}
        
        for relationship in relationships:
            try:
                results["processed"] += 1
                self.metadata_store.create_relationship(relationship)
                results["created"] += 1
            except Exception as e:
                results["failed"] += 1
                if validation_mode == "strict":
                    raise
        
        return results
    
    def _transform_entity(self, entity: Dict[str, Any], 
                         transformation_rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a single entity using rules."""
        if not transformation_rules:
            return entity
        
        transformed = entity.copy()
        
        # Apply field mappings
        field_mappings = transformation_rules.get("field_mappings", {})
        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)
        
        # Apply value transformations
        value_transformations = transformation_rules.get("value_transformations", {})
        for field, transform_func in value_transformations.items():
            if field in transformed:
                # Simple transformations
                if transform_func == "uppercase":
                    transformed[field] = str(transformed[field]).upper()
                elif transform_func == "lowercase":
                    transformed[field] = str(transformed[field]).lower()
        
        # Filter conditions
        filter_conditions = transformation_rules.get("filter_conditions", {})
        for field, condition in filter_conditions.items():
            if field in transformed:
                value = transformed[field]
                if condition.get("exclude_if_empty") and not value:
                    return None
                if condition.get("exclude_if_equals") and value == condition["exclude_if_equals"]:
                    return None
        
        return transformed
    
    def _transform_relationship(self, relationship: Dict[str, Any], 
                               transformation_rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a single relationship using rules."""
        if not transformation_rules:
            return relationship
        
        transformed = relationship.copy()
        
        # Apply similar transformations as entities
        field_mappings = transformation_rules.get("field_mappings", {})
        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)
        
        return transformed