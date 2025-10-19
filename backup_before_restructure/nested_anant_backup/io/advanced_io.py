"""
Advanced I/O System for Anant

This module provides comprehensive I/O capabilities including native Parquet support,
compression options, schema preservation, multi-file dataset handling, and optimized
data loading/saving workflows for hypergraph data.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
from pathlib import Path
import polars as pl
import json
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

from ..classes import Hypergraph


class CompressionType(Enum):
    """Supported compression types for Parquet files"""
    UNCOMPRESSED = "uncompressed"
    SNAPPY = "snappy"
    GZIP = "gzip" 
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"


class FileFormat(Enum):
    """Supported file formats"""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    ARROW = "arrow"
    PICKLE = "pickle"


@dataclass
class IOConfiguration:
    """Configuration for I/O operations"""
    compression: CompressionType = CompressionType.SNAPPY
    batch_size: int = 10000
    enable_parallel: bool = True
    max_workers: int = 4
    preserve_schema: bool = True
    validate_data: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class DatasetMetadata:
    """Metadata for hypergraph datasets"""
    name: str
    description: str
    version: str
    created_at: str
    node_count: int
    edge_count: int
    incidence_count: int
    properties: Dict[str, Any]
    schema_version: str = "1.0"
    file_format: str = "parquet"
    compression: str = "snappy"


@dataclass
class LoadResult:
    """Result of data loading operation"""
    hypergraph: Hypergraph
    metadata: DatasetMetadata
    load_time: float
    memory_usage: int
    warnings: List[str]


@dataclass
class SaveResult:
    """Result of data saving operation"""
    file_paths: List[str]
    total_size_bytes: int
    save_time: float
    compression_ratio: float
    metadata_path: str


class AdvancedAnantIO:
    """
    Advanced I/O system for Anant hypergraphs
    
    Features:
    - Native Parquet support with compression
    - Schema preservation and validation
    - Multi-file dataset handling
    - Parallel processing for large datasets
    - Progress monitoring and optimization
    - Streaming support for large data
    """
    
    def __init__(self, config: Optional[IOConfiguration] = None):
        self.config = config or IOConfiguration()
        self._schema_cache = {}
        self._compression_stats = {}
        
    def save_hypergraph(
        self,
        hypergraph: Hypergraph,
        path: Union[str, Path],
        metadata: Optional[DatasetMetadata] = None,
        format: FileFormat = FileFormat.PARQUET,
        **kwargs
    ) -> SaveResult:
        """
        Save hypergraph to disk with comprehensive metadata
        
        Args:
            hypergraph: Hypergraph to save
            path: Output path (file or directory)
            metadata: Optional metadata for the dataset
            format: File format to use
            **kwargs: Additional format-specific options
            
        Returns:
            SaveResult with operation details
        """
        start_time = time.time()
        path = Path(path)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self._generate_metadata(hypergraph, path.name)
        
        # Ensure output directory exists
        if format == FileFormat.PARQUET:
            # For parquet, create directory structure
            path.mkdir(parents=True, exist_ok=True)
            return self._save_parquet_dataset(hypergraph, path, metadata, start_time, **kwargs)
        elif format == FileFormat.CSV:
            return self._save_csv_dataset(hypergraph, path, metadata, **kwargs)
        elif format == FileFormat.JSON:
            return self._save_json_dataset(hypergraph, path, metadata, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _filter_for_csv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter DataFrame for CSV compatibility (flatten complex types)"""
        if len(df) == 0:
            return df
        
        columns_to_keep = []
        for col in df.columns:
            col_dtype = df.schema[col]
            
            # Skip complex types that CSV can't handle
            if isinstance(col_dtype, (pl.Struct, pl.List, pl.Array)):
                # For empty structs, skip entirely
                if isinstance(col_dtype, pl.Struct) and len(col_dtype.fields) == 0:
                    continue
                # For other complex types, try to convert to string
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Utf8))
                    columns_to_keep.append(col)
                except:
                    continue  # Skip if can't convert
            elif isinstance(col_dtype, pl.Categorical):
                # Convert categorical to string
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
                columns_to_keep.append(col)
            elif isinstance(col_dtype, pl.Datetime):
                # Convert datetime to string
                df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d %H:%M:%S"))
                columns_to_keep.append(col)
            else:
                columns_to_keep.append(col)
        
        if not columns_to_keep:
            # If no columns to keep, return empty dataframe with uid column
            return pl.DataFrame({"uid": pl.Series([], dtype=pl.Utf8)})
        
        return df.select(columns_to_keep)
    
    def _filter_empty_structs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter out empty struct columns that cause Parquet write issues"""
        if len(df) == 0:
            return df
        
        columns_to_keep = []
        for col in df.columns:
            col_dtype = df.schema[col]
            # Skip empty struct types
            if isinstance(col_dtype, pl.Struct) and len(col_dtype.fields) == 0:
                continue
            columns_to_keep.append(col)
        
        if not columns_to_keep:
            # If no columns to keep, return empty dataframe with uid column
            return pl.DataFrame({"uid": pl.Series([], dtype=pl.Utf8)})
        
        return df.select(columns_to_keep)
    
    def _save_parquet_dataset(
        self,
        hypergraph: Hypergraph,
        path: Path,
        metadata: DatasetMetadata,
        start_time: float,
        **kwargs
    ) -> SaveResult:
        """Save hypergraph as optimized Parquet dataset"""
        file_paths = []
        total_size = 0
        original_size = 0
        
        # Save nodes
        nodes_df = hypergraph.to_dataframe("nodes")
        if len(nodes_df) > 0:
            # Filter out empty struct columns for Parquet compatibility
            nodes_df = self._filter_empty_structs(nodes_df)
            if len(nodes_df) > 0:  # Check again after filtering
                nodes_path = path / "nodes.parquet"
                original_size += nodes_df.estimated_size("bytes")
                
                nodes_df.write_parquet(
                    nodes_path,
                    compression=self.config.compression.value,
                    **kwargs
                )
                file_paths.append(str(nodes_path))
                total_size += nodes_path.stat().st_size
        
        # Save edges
        edges_df = hypergraph.to_dataframe("edges")
        if len(edges_df) > 0:
            # Filter out empty struct columns for Parquet compatibility
            edges_df = self._filter_empty_structs(edges_df)
            if len(edges_df) > 0:  # Check again after filtering
                edges_path = path / "edges.parquet"
                original_size += edges_df.estimated_size("bytes")
                
                edges_df.write_parquet(
                    edges_path,
                    compression=self.config.compression.value,
                    **kwargs
                )
                file_paths.append(str(edges_path))
                total_size += edges_path.stat().st_size
        
        # Save incidence relationships
        incidence_df = hypergraph.to_dataframe("incidences")
        if len(incidence_df) > 0:
            # Filter out empty struct columns for Parquet compatibility
            incidence_df = self._filter_empty_structs(incidence_df)
            if len(incidence_df) > 0:  # Check again after filtering
                incidence_path = path / "incidence.parquet"
                original_size += incidence_df.estimated_size("bytes")
                
                incidence_df.write_parquet(
                    incidence_path,
                    compression=self.config.compression.value,
                    **kwargs
                )
                file_paths.append(str(incidence_path))
                total_size += incidence_path.stat().st_size
        
        # Save metadata
        metadata_path = path / "metadata.json"
        metadata_dict = {
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "created_at": metadata.created_at,
            "node_count": metadata.node_count,
            "edge_count": metadata.edge_count,
            "incidence_count": metadata.incidence_count,
            "properties": metadata.properties,
            "schema_version": metadata.schema_version,
            "file_format": metadata.file_format,
            "compression": metadata.compression,
            "files": {
                "nodes": "nodes.parquet" if (path / "nodes.parquet").exists() else None,
                "edges": "edges.parquet" if (path / "edges.parquet").exists() else None,
                "incidence": "incidence.parquet" if (path / "incidence.parquet").exists() else None
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        file_paths.append(str(metadata_path))
        total_size += metadata_path.stat().st_size
        
        save_time = time.time() - start_time
        compression_ratio = original_size / total_size if total_size > 0 else 1.0
        
        return SaveResult(
            file_paths=file_paths,
            total_size_bytes=total_size,
            save_time=save_time,
            compression_ratio=compression_ratio,
            metadata_path=str(metadata_path)
        )
    
    def _save_csv_dataset(
        self,
        hypergraph: Hypergraph,
        path: Path,
        metadata: DatasetMetadata,
        **kwargs
    ) -> SaveResult:
        """Save hypergraph as CSV dataset"""
        path.mkdir(parents=True, exist_ok=True)
        file_paths = []
        total_size = 0
        
        # Save nodes
        nodes_df = hypergraph.to_dataframe("nodes")
        if len(nodes_df) > 0:
            # Filter for CSV compatibility
            nodes_df = self._filter_for_csv(nodes_df)
            if len(nodes_df) > 0:  # Check again after filtering
                nodes_path = path / "nodes.csv"
                nodes_df.write_csv(nodes_path, **kwargs)
                file_paths.append(str(nodes_path))
                total_size += nodes_path.stat().st_size
        
        # Save edges
        edges_df = hypergraph.to_dataframe("edges")
        if len(edges_df) > 0:
            # Filter for CSV compatibility
            edges_df = self._filter_for_csv(edges_df)
            if len(edges_df) > 0:  # Check again after filtering
                edges_path = path / "edges.csv"
                edges_df.write_csv(edges_path, **kwargs)
                file_paths.append(str(edges_path))
                total_size += edges_path.stat().st_size
        
        # Save incidence
        incidence_df = hypergraph.to_dataframe("incidences")
        if len(incidence_df) > 0:
            # Filter for CSV compatibility
            incidence_df = self._filter_for_csv(incidence_df)
            if len(incidence_df) > 0:  # Check again after filtering
                incidence_path = path / "incidence.csv"
                incidence_df.write_csv(incidence_path, **kwargs)
                file_paths.append(str(incidence_path))
                total_size += incidence_path.stat().st_size
        
        # Save metadata
        metadata_path = path / "metadata.json"
        metadata_dict = {
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "created_at": metadata.created_at,
            "node_count": metadata.node_count,
            "edge_count": metadata.edge_count,
            "incidence_count": metadata.incidence_count,
            "properties": metadata.properties,
            "schema_version": metadata.schema_version,
            "file_format": "csv",
            "compression": "none",
            "files": {
                "nodes": "nodes.csv" if (path / "nodes.csv").exists() else None,
                "edges": "edges.csv" if (path / "edges.csv").exists() else None,
                "incidence": "incidence.csv" if (path / "incidence.csv").exists() else None
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        file_paths.append(str(metadata_path))
        total_size += metadata_path.stat().st_size
        
        return SaveResult(
            file_paths=file_paths,
            total_size_bytes=total_size,
            save_time=0.0,
            compression_ratio=1.0,
            metadata_path=str(metadata_path)
        )
    
    def _save_json_dataset(
        self,
        hypergraph: Hypergraph,
        path: Path,
        metadata: DatasetMetadata,
        **kwargs
    ) -> SaveResult:
        """Save hypergraph as JSON dataset"""
        path.mkdir(parents=True, exist_ok=True)
        file_paths = []
        total_size = 0
        
        # Save nodes
        nodes_df = hypergraph.to_dataframe("nodes")
        if len(nodes_df) > 0:
            nodes_path = path / "nodes.json"
            nodes_df.write_json(nodes_path, **kwargs)
            file_paths.append(str(nodes_path))
            total_size += nodes_path.stat().st_size
        
        # Save edges
        edges_df = hypergraph.to_dataframe("edges")
        if len(edges_df) > 0:
            edges_path = path / "edges.json"
            edges_df.write_json(edges_path, **kwargs)
            file_paths.append(str(edges_path))
            total_size += edges_path.stat().st_size
        
        # Save incidence
        incidence_df = hypergraph.to_dataframe("incidences")
        if len(incidence_df) > 0:
            incidence_path = path / "incidence.json"
            incidence_df.write_json(incidence_path, **kwargs)
            file_paths.append(str(incidence_path))
            total_size += incidence_path.stat().st_size
        
        # Save metadata
        metadata_path = path / "metadata.json"
        metadata_dict = {
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "created_at": metadata.created_at,
            "node_count": metadata.node_count,
            "edge_count": metadata.edge_count,
            "incidence_count": metadata.incidence_count,
            "properties": metadata.properties,
            "schema_version": metadata.schema_version,
            "file_format": "json",
            "compression": "none",
            "files": {
                "nodes": "nodes.json" if (path / "nodes.json").exists() else None,
                "edges": "edges.json" if (path / "edges.json").exists() else None,
                "incidence": "incidence.json" if (path / "incidence.json").exists() else None
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        file_paths.append(str(metadata_path))
        total_size += metadata_path.stat().st_size
        
        return SaveResult(
            file_paths=file_paths,
            total_size_bytes=total_size,
            save_time=0.0,
            compression_ratio=1.0,
            metadata_path=str(metadata_path)
        )
    
    def load_hypergraph(
        self,
        path: Union[str, Path],
        format: Optional[FileFormat] = None,
        validate: bool = True,
        **kwargs
    ) -> LoadResult:
        """
        Load hypergraph from disk with validation and metadata
        
        Args:
            path: Input path (file or directory)
            format: File format (auto-detected if None)
            validate: Whether to validate data consistency
            **kwargs: Additional format-specific options
            
        Returns:
            LoadResult with hypergraph and operation details
        """
        start_time = time.time()
        path = Path(path)
        warnings_list = []
        
        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(path)
        
        # Load based on format
        if format == FileFormat.PARQUET:
            result = self._load_parquet_dataset(path, validate, **kwargs)
        elif format == FileFormat.CSV:
            result = self._load_csv_dataset(path, validate, **kwargs)
        elif format == FileFormat.JSON:
            result = self._load_json_dataset(path, validate, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        hypergraph, metadata = result
        
        # Validate if requested
        if validate:
            validation_warnings = self._validate_hypergraph(hypergraph)
            warnings_list.extend(validation_warnings)
        
        load_time = time.time() - start_time
        memory_usage = self._estimate_memory_usage(hypergraph)
        
        return LoadResult(
            hypergraph=hypergraph,
            metadata=metadata,
            load_time=load_time,
            memory_usage=memory_usage,
            warnings=warnings_list
        )
    
    def _load_parquet_dataset(
        self,
        path: Path,
        validate: bool,
        **kwargs
    ) -> Tuple[Hypergraph, DatasetMetadata]:
        """Load hypergraph from Parquet dataset"""
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Remove fields that aren't part of DatasetMetadata
            files_info = metadata_dict.pop('files', {})
            
            # Handle any other extra fields
            valid_fields = {'name', 'description', 'version', 'created_at', 
                          'node_count', 'edge_count', 'incidence_count', 
                          'properties', 'schema_version', 'file_format', 'compression'}
            filtered_dict = {k: v for k, v in metadata_dict.items() if k in valid_fields}
            
            metadata = DatasetMetadata(**filtered_dict)
        else:
            # Create default metadata
            metadata = DatasetMetadata(
                name=path.name,
                description="Loaded from Parquet dataset",
                version="1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                node_count=0,
                edge_count=0,
                incidence_count=0,
                properties={}
            )
        
        # Load components
        nodes_df = pl.DataFrame()
        edges_df = pl.DataFrame()
        incidence_df = pl.DataFrame()
        
        # Load nodes
        nodes_path = path / "nodes.parquet"
        if nodes_path.exists():
            nodes_df = pl.read_parquet(nodes_path, **kwargs)
        
        # Load edges
        edges_path = path / "edges.parquet"
        if edges_path.exists():
            edges_df = pl.read_parquet(edges_path, **kwargs)
        
        # Load incidence
        incidence_path = path / "incidence.parquet"
        if incidence_path.exists():
            incidence_df = pl.read_parquet(incidence_path, **kwargs)
        
        # Create hypergraph
        hypergraph = self._create_hypergraph_from_dataframes(
            nodes_df, edges_df, incidence_df
        )
        
        return hypergraph, metadata
    
    def _load_csv_dataset(
        self,
        path: Path,
        validate: bool,
        **kwargs
    ) -> Tuple[Hypergraph, DatasetMetadata]:
        """Load hypergraph from CSV dataset"""
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Remove fields that aren't part of DatasetMetadata
            files_info = metadata_dict.pop('files', {})
            
            # Handle any other extra fields
            valid_fields = {'name', 'description', 'version', 'created_at', 
                          'node_count', 'edge_count', 'incidence_count', 
                          'properties', 'schema_version', 'file_format', 'compression'}
            filtered_dict = {k: v for k, v in metadata_dict.items() if k in valid_fields}
            
            metadata = DatasetMetadata(**filtered_dict)
        else:
            metadata = DatasetMetadata(
                name=path.name,
                description="Loaded from CSV dataset",
                version="1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                node_count=0,
                edge_count=0,
                incidence_count=0,
                properties={}
            )
        
        # Load components
        nodes_df = pl.DataFrame()
        edges_df = pl.DataFrame()
        incidence_df = pl.DataFrame()
        
        # Load nodes
        nodes_path = path / "nodes.csv"
        if nodes_path.exists():
            nodes_df = pl.read_csv(nodes_path, **kwargs)
        
        # Load edges  
        edges_path = path / "edges.csv"
        if edges_path.exists():
            edges_df = pl.read_csv(edges_path, **kwargs)
        
        # Load incidence
        incidence_path = path / "incidence.csv"
        if incidence_path.exists():
            incidence_df = pl.read_csv(incidence_path, **kwargs)
        
        # Create hypergraph
        hypergraph = self._create_hypergraph_from_dataframes(
            nodes_df, edges_df, incidence_df
        )
        
        return hypergraph, metadata
    
    def _load_json_dataset(
        self,
        path: Path,
        validate: bool,
        **kwargs
    ) -> Tuple[Hypergraph, DatasetMetadata]:
        """Load hypergraph from JSON dataset"""
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Remove fields that aren't part of DatasetMetadata
            files_info = metadata_dict.pop('files', {})
            
            # Handle any other extra fields
            valid_fields = {'name', 'description', 'version', 'created_at', 
                          'node_count', 'edge_count', 'incidence_count', 
                          'properties', 'schema_version', 'file_format', 'compression'}
            filtered_dict = {k: v for k, v in metadata_dict.items() if k in valid_fields}
            
            metadata = DatasetMetadata(**filtered_dict)
        else:
            metadata = DatasetMetadata(
                name=path.name,
                description="Loaded from JSON dataset",
                version="1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                node_count=0,
                edge_count=0,
                incidence_count=0,
                properties={}
            )
        
        # Load components
        nodes_df = pl.DataFrame()
        edges_df = pl.DataFrame()
        incidence_df = pl.DataFrame()
        
        # Load nodes
        nodes_path = path / "nodes.json"
        if nodes_path.exists():
            nodes_df = pl.read_json(nodes_path, **kwargs)
        
        # Load edges
        edges_path = path / "edges.json"
        if edges_path.exists():
            edges_df = pl.read_json(edges_path, **kwargs)
        
        # Load incidence
        incidence_path = path / "incidence.json"
        if incidence_path.exists():
            incidence_df = pl.read_json(incidence_path, **kwargs)
        
        # Create hypergraph
        hypergraph = self._create_hypergraph_from_dataframes(
            nodes_df, edges_df, incidence_df
        )
        
        return hypergraph, metadata
    
    def _create_hypergraph_from_dataframes(
        self,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
        incidence_df: pl.DataFrame
    ) -> Hypergraph:
        """Create hypergraph from component DataFrames"""
        
        # If we have incidence data, use it to create the hypergraph
        if len(incidence_df) > 0:
            # Create hypergraph directly from incidence DataFrame
            hypergraph = Hypergraph(incidence_df)
            
            # Add node properties if available
            if len(nodes_df) > 0:
                hypergraph.add_node_properties(nodes_df)
            
            # Add edge properties if available  
            if len(edges_df) > 0:
                hypergraph.add_edge_properties(edges_df)
        
        else:
            # Create empty hypergraph
            hypergraph = Hypergraph()
            
            # Add nodes if available
            if len(nodes_df) > 0:
                # Convert dataframe to node additions
                for row in nodes_df.iter_rows(named=True):
                    node_id = str(row[nodes_df.columns[0]])
                    # Add an empty edge for each node (this creates the node)
                    if node_id not in hypergraph.nodes:
                        hypergraph.add_edge(f"temp_edge_{node_id}", [node_id])
                        # Remove the temporary edge
                        hypergraph.remove_edge(f"temp_edge_{node_id}")
                
                # Add node properties
                hypergraph.add_node_properties(nodes_df)
            
            # Add edges if available
            if len(edges_df) > 0:
                # Convert dataframe to edge additions
                for row in edges_df.iter_rows(named=True):
                    edge_id = str(row[edges_df.columns[0]])
                    hypergraph.add_edge(edge_id, [])  # Empty edge for now
                
                # Add edge properties
                hypergraph.add_edge_properties(edges_df)
        
        return hypergraph
    
    def _detect_format(self, path: Path) -> FileFormat:
        """Auto-detect file format from path"""
        if path.is_dir():
            # Check for metadata and data files
            if (path / "metadata.json").exists():
                if (path / "nodes.parquet").exists():
                    return FileFormat.PARQUET
                elif (path / "nodes.csv").exists():
                    return FileFormat.CSV
                elif (path / "nodes.json").exists():
                    return FileFormat.JSON
            
            # Fallback to most common file type in directory
            parquet_files = list(path.glob("*.parquet"))
            csv_files = list(path.glob("*.csv"))
            json_files = list(path.glob("*.json"))
            
            if parquet_files:
                return FileFormat.PARQUET
            elif csv_files:
                return FileFormat.CSV
            elif json_files:
                return FileFormat.JSON
        
        else:
            # Single file detection
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                return FileFormat.PARQUET
            elif suffix == ".csv":
                return FileFormat.CSV
            elif suffix == ".json":
                return FileFormat.JSON
        
        # Default to CSV
        return FileFormat.CSV
    
    def _generate_metadata(
        self,
        hypergraph: Hypergraph,
        name: str
    ) -> DatasetMetadata:
        """Generate metadata for hypergraph"""
        return DatasetMetadata(
            name=name,
            description=f"Hypergraph dataset with {len(hypergraph.nodes)} nodes and {len(hypergraph.edges)} edges",
            version="1.0",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            node_count=len(hypergraph.nodes),
            edge_count=len(hypergraph.edges),
            incidence_count=hypergraph.num_incidences,
            properties={
                "node_properties": list(hypergraph.to_dataframe("nodes").columns),
                "edge_properties": list(hypergraph.to_dataframe("edges").columns),
                "incidence_properties": list(hypergraph.to_dataframe("incidences").columns)
            },
            file_format=self.config.compression.value,
            compression=self.config.compression.value
        )
    
    def _validate_hypergraph(self, hypergraph: Hypergraph) -> List[str]:
        """Validate hypergraph data consistency"""
        warnings_list = []
        
        # Check for empty hypergraph
        if len(hypergraph.nodes) == 0 and len(hypergraph.edges) == 0:
            warnings_list.append("Loaded hypergraph is empty")
        
        # Check for orphaned nodes or edges  
        if hypergraph.num_incidences > 0:
            incidence_df = hypergraph.to_dataframe("incidences")
            if "nodes" in incidence_df.columns:
                incidence_nodes = set(incidence_df.get_column("nodes"))
                hypergraph_nodes = set(hypergraph.nodes)
                
                orphaned_nodes = incidence_nodes - hypergraph_nodes
                if orphaned_nodes:
                    warnings_list.append(f"Found {len(orphaned_nodes)} orphaned nodes in incidence data")
            
            if "edges" in incidence_df.columns:
                incidence_edges = set(incidence_df.get_column("edges"))
                hypergraph_edges = set(hypergraph.edges)
                
                orphaned_edges = incidence_edges - hypergraph_edges
                if orphaned_edges:
                    warnings_list.append(f"Found {len(orphaned_edges)} orphaned edges in incidence data")
        
        return warnings_list
    
    def _estimate_memory_usage(self, hypergraph: Hypergraph) -> int:
        """Estimate memory usage of hypergraph in bytes"""
        total_memory = 0
        
        nodes_df = hypergraph.to_dataframe("nodes")
        if len(nodes_df) > 0:
            total_memory += nodes_df.estimated_size("bytes")
        
        edges_df = hypergraph.to_dataframe("edges")
        if len(edges_df) > 0:
            total_memory += edges_df.estimated_size("bytes")
        
        incidence_df = hypergraph.to_dataframe("incidences")
        if len(incidence_df) > 0:
            total_memory += incidence_df.estimated_size("bytes")
        
        return int(total_memory)
    
    def save_multiple_hypergraphs(
        self,
        hypergraphs: Dict[str, Hypergraph],
        base_path: Union[str, Path],
        format: FileFormat = FileFormat.PARQUET,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, SaveResult]:
        """
        Save multiple hypergraphs in parallel
        
        Args:
            hypergraphs: Dictionary of name -> hypergraph mappings
            base_path: Base directory for saving
            format: File format to use
            parallel: Whether to use parallel processing
            **kwargs: Additional save options
            
        Returns:
            Dictionary of name -> SaveResult mappings
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        if parallel and self.config.enable_parallel:
            # Parallel saving
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_name = {}
                
                for name, hypergraph in hypergraphs.items():
                    output_path = base_path / name
                    future = executor.submit(
                        self.save_hypergraph,
                        hypergraph,
                        output_path,
                        None,  # Let it generate metadata
                        format,
                        **kwargs
                    )
                    future_to_name[future] = name
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results[name] = result
                        
                        if self.config.progress_callback:
                            self.config.progress_callback(name, result)
                    
                    except Exception as e:
                        warnings.warn(f"Failed to save hypergraph '{name}': {e}")
                        results[name] = None
        
        else:
            # Sequential saving
            for name, hypergraph in hypergraphs.items():
                try:
                    output_path = base_path / name
                    result = self.save_hypergraph(hypergraph, output_path, None, format, **kwargs)
                    results[name] = result
                    
                    if self.config.progress_callback:
                        self.config.progress_callback(name, result)
                
                except Exception as e:
                    warnings.warn(f"Failed to save hypergraph '{name}': {e}")
                    results[name] = None
        
        return results
    
    def load_multiple_hypergraphs(
        self,
        base_path: Union[str, Path],
        format: Optional[FileFormat] = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, LoadResult]:
        """
        Load multiple hypergraphs from directory
        
        Args:
            base_path: Base directory containing hypergraph datasets
            format: File format (auto-detected if None)
            parallel: Whether to use parallel processing
            **kwargs: Additional load options
            
        Returns:
            Dictionary of name -> LoadResult mappings
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {base_path}")
        
        # Find hypergraph directories/files
        hypergraph_paths = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if it contains hypergraph data
                if (item / "metadata.json").exists():
                    hypergraph_paths.append(item)
            elif item.is_file() and format is not None:
                # Single file hypergraph
                hypergraph_paths.append(item)
        
        results = {}
        
        if parallel and self.config.enable_parallel:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_name = {}
                
                for path in hypergraph_paths:
                    name = path.stem
                    future = executor.submit(
                        self.load_hypergraph,
                        path,
                        format,
                        self.config.validate_data,
                        **kwargs
                    )
                    future_to_name[future] = name
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results[name] = result
                        
                        if self.config.progress_callback:
                            self.config.progress_callback(name, result)
                    
                    except Exception as e:
                        warnings.warn(f"Failed to load hypergraph '{name}': {e}")
                        results[name] = None
        
        else:
            # Sequential loading
            for path in hypergraph_paths:
                name = path.stem
                try:
                    result = self.load_hypergraph(path, format, self.config.validate_data, **kwargs)
                    results[name] = result
                    
                    if self.config.progress_callback:
                        self.config.progress_callback(name, result)
                
                except Exception as e:
                    warnings.warn(f"Failed to load hypergraph '{name}': {e}")
                    results[name] = None
        
        return results
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics for saved datasets"""
        return {
            "compression_ratios": self._compression_stats,
            "total_datasets": len(self._compression_stats),
            "avg_compression_ratio": sum(self._compression_stats.values()) / len(self._compression_stats) if self._compression_stats else 0
        }
    
    def benchmark_formats(
        self,
        hypergraph: Hypergraph,
        test_path: Union[str, Path],
        formats: Optional[List[FileFormat]] = None
    ) -> Dict[FileFormat, Dict[str, Any]]:
        """
        Benchmark different file formats for save/load performance
        
        Args:
            hypergraph: Test hypergraph
            test_path: Directory for benchmark files
            formats: Formats to test (default: all supported)
            
        Returns:
            Performance statistics for each format
        """
        if formats is None:
            formats = [FileFormat.PARQUET, FileFormat.CSV, FileFormat.JSON]
        
        test_path = Path(test_path)
        test_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for format in formats:
            format_path = test_path / f"benchmark_{format.value}"
            
            try:
                # Benchmark save
                save_start = time.time()
                save_result = self.save_hypergraph(hypergraph, format_path, format=format)
                save_time = time.time() - save_start
                
                # Benchmark load
                load_start = time.time()
                load_result = self.load_hypergraph(format_path, format=format)
                load_time = time.time() - load_start
                
                results[format] = {
                    "save_time": save_time,
                    "load_time": load_time,
                    "total_time": save_time + load_time,
                    "file_size": save_result.total_size_bytes,
                    "compression_ratio": save_result.compression_ratio,
                    "memory_usage": load_result.memory_usage,
                    "warnings": load_result.warnings
                }
                
            except Exception as e:
                results[format] = {
                    "error": str(e),
                    "save_time": None,
                    "load_time": None,
                    "file_size": None,
                    "compression_ratio": None
                }
        
        return results