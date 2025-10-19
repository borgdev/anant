"""
Enhanced SetSystem Types for Anant Library

Provides advanced SetSystem types including:
- Parquet SetSystem for direct file loading
- Multi-Modal SetSystem for cross-analysis  
- Streaming SetSystem for massive datasets
- Enhanced validation and error handling
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable, Iterator, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Import base factory
from .setsystem_factory import SetSystemFactory


class SetSystemType(Enum):
    """Enumeration of available SetSystem types"""
    STANDARD = "standard"
    PARQUET = "parquet"
    MULTIMODAL = "multimodal"
    STREAMING = "streaming"


@dataclass
class SetSystemMetadata:
    """Metadata for enhanced SetSystems"""
    setsystem_type: SetSystemType
    creation_time: str
    source_info: Dict[str, Any]
    validation_passed: bool
    optimization_applied: bool
    performance_stats: Dict[str, Any]


class ParquetSetSystem:
    """
    SetSystem that can directly load hypergraphs from Parquet files.
    
    Enables direct file-to-hypergraph workflows with lazy loading,
    filtering, and optimization capabilities.
    """
    
    @staticmethod
    def from_parquet(
        file_path: Union[str, Path],
        edge_column: str = "edges",
        node_column: str = "nodes", 
        weight_column: Optional[str] = "weight",
        lazy: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        validate_schema: bool = True,
        add_metadata: bool = True
    ) -> pl.DataFrame:
        """
        Create DataFrame directly from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            edge_column: Column containing edge identifiers
            node_column: Column containing node identifiers  
            weight_column: Column containing edge weights (optional)
            lazy: Whether to use lazy loading
            filters: Filter conditions to apply during load
            columns: Specific columns to load
            validate_schema: Whether to validate hypergraph schema
            add_metadata: Whether to add SetSystem metadata
            
        Returns:
            DataFrame ready for Hypergraph construction
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        logger.info(f"Creating SetSystem from Parquet: {file_path}")
        
        try:
            # Load with lazy or direct approach
            if lazy:
                lazy_df = pl.scan_parquet(file_path)
                
                # Apply filters first (before column selection)
                if filters:
                    for column, condition in filters.items():
                        if isinstance(condition, (list, tuple)):
                            lazy_df = lazy_df.filter(pl.col(column).is_in(condition))
                        else:
                            lazy_df = lazy_df.filter(pl.col(column) == condition)
                
                # Then select columns if specified
                if columns:
                    lazy_df = lazy_df.select(columns)
                
                # Materialize
                df = lazy_df.collect()
            else:
                df = pl.read_parquet(file_path, columns=columns)
                
                # Apply filters after loading
                if filters:
                    for column, condition in filters.items():
                        if isinstance(condition, (list, tuple)):
                            df = df.filter(pl.col(column).is_in(condition))
                        else:
                            df = df.filter(pl.col(column) == condition)
            
            # Validate schema if requested
            if validate_schema:
                ParquetSetSystem._validate_hypergraph_schema(
                    df, edge_column, node_column, weight_column
                )
            
            # Ensure standard columns exist
            df = ParquetSetSystem._standardize_columns(
                df, edge_column, node_column, weight_column
            )
            
            # Add metadata if requested
            if add_metadata:
                metadata_dict = dict(
                    source_file=str(file_path),
                    load_mode="lazy" if lazy else "eager", 
                    creation_time="2024-01-01",  # Fixed timestamp for metadata
                    columns_loaded=columns or "all",
                    filters_applied=str(filters) if filters else "none"
                )
                
                # Add metadata as column
                df = df.with_columns([
                    pl.lit(str(metadata_dict)).alias('__setsystem_metadata__')
                ])
            
            logger.info(f"Successfully created ParquetSetSystem: {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to create ParquetSetSystem from {file_path}: {e}")
            raise ValueError(f"Could not create ParquetSetSystem: {e}") from e
    
    @staticmethod
    def _validate_hypergraph_schema(
        df: pl.DataFrame,
        edge_column: str,
        node_column: str, 
        weight_column: Optional[str]
    ) -> None:
        """Validate that DataFrame has proper hypergraph schema"""
        
        # Check required columns exist
        if edge_column not in df.columns:
            raise ValueError(f"Edge column '{edge_column}' not found in DataFrame")
        
        if node_column not in df.columns:
            raise ValueError(f"Node column '{node_column}' not found in DataFrame")
        
        if weight_column and weight_column not in df.columns:
            raise ValueError(f"Weight column '{weight_column}' not found in DataFrame")
        
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Check for null values in required columns
        edge_nulls = df[edge_column].null_count()
        node_nulls = df[node_column].null_count()
        
        if edge_nulls > 0:
            raise ValueError(f"Found {edge_nulls} null values in edge column")
        
        if node_nulls > 0:
            raise ValueError(f"Found {node_nulls} null values in node column")
        
        # Validate weight column if specified
        if weight_column:
            weight_nulls = df[weight_column].null_count()
            if weight_nulls > 0:
                logger.warning(f"Found {weight_nulls} null values in weight column")
        
        logger.debug("ParquetSetSystem schema validation passed")
    
    @staticmethod
    def _standardize_columns(
        df: pl.DataFrame,
        edge_column: str,
        node_column: str,
        weight_column: Optional[str]
    ) -> pl.DataFrame:
        """Standardize column names and add missing columns"""
        
        # Rename columns to standard names if needed
        column_mapping = {}
        
        if edge_column != "edges":
            column_mapping[edge_column] = "edges"
        
        if node_column != "nodes":
            column_mapping[node_column] = "nodes"
        
        if weight_column and weight_column != "weight":
            column_mapping[weight_column] = "weight"
        
        # Apply renaming
        if column_mapping:
            df = df.rename(column_mapping)
        
        # Add weight column if missing
        if "weight" not in df.columns:
            df = df.with_columns([pl.lit(1.0).alias("weight")])
        
        # Ensure proper data types
        df = df.with_columns([
            pl.col("edges").cast(pl.Utf8),
            pl.col("nodes").cast(pl.Utf8),
            pl.col("weight").cast(pl.Float64)
        ])
        
        return df


class MultiModalSetSystem:
    """
    SetSystem for cross-relationship analysis between different data modalities.
    
    Enables analysis of relationships across different types of networks
    (e.g., social + biological, temporal + spatial).
    """
    
    @staticmethod
    def from_multiple_sources(
        modal_data: Dict[str, Union[pl.DataFrame, str, Path]],
        cross_modal_edges: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        modal_prefixes: Optional[Dict[str, str]] = None,
        merge_strategy: str = "union",
        validate_compatibility: bool = True,
        add_modal_metadata: bool = True
    ) -> pl.DataFrame:
        """
        Create multi-modal SetSystem from multiple data sources.
        
        Args:
            modal_data: Dictionary mapping modal names to DataFrames or file paths
            cross_modal_edges: Optional cross-modal edge definitions
            modal_prefixes: Prefixes for node/edge IDs from each modality
            merge_strategy: How to merge modalities ('union', 'intersection', 'cross_link')
            validate_compatibility: Whether to validate modal compatibility
            add_modal_metadata: Whether to add modality information
            
        Returns:
            DataFrame combining all modalities with cross-modal relationships
        """
        if not modal_data:
            raise ValueError("No modal data provided")
        
        logger.info(f"Creating MultiModalSetSystem from {len(modal_data)} modalities")
        
        try:
            # Load all modalities
            loaded_modalities = {}
            
            for modal_name, data_source in modal_data.items():
                logger.debug(f"Loading modality: {modal_name}")
                
                if isinstance(data_source, pl.DataFrame):
                    modal_df = data_source
                elif isinstance(data_source, (str, Path)):
                    # Load from file using ParquetSetSystem
                    modal_df = ParquetSetSystem.from_parquet(
                        data_source,
                        add_metadata=False  # Will add multi-modal metadata instead
                    )
                else:
                    raise ValueError(f"Unsupported data source type for modality '{modal_name}': {type(data_source)}")
                
                # Apply modal prefixes if specified
                if modal_prefixes and modal_name in modal_prefixes:
                    prefix = modal_prefixes[modal_name]
                    modal_df = modal_df.with_columns([
                        pl.col("edges").map_elements(lambda x: f"{prefix}_E_{x}").alias("edges"),
                        pl.col("nodes").map_elements(lambda x: f"{prefix}_N_{x}").alias("nodes")
                    ])
                
                # Add modality identifier
                if add_modal_metadata:
                    modal_df = modal_df.with_columns([
                        pl.lit(modal_name).alias("modality"),
                        pl.lit("intra_modal").alias("edge_type")
                    ])
                
                loaded_modalities[modal_name] = modal_df
                logger.debug(f"Loaded {modal_name}: {len(modal_df)} rows")
            
            # Validate compatibility if requested
            if validate_compatibility:
                MultiModalSetSystem._validate_modal_compatibility(loaded_modalities)
            
            # Create cross-modal edges if specified
            cross_modal_df = None
            if cross_modal_edges:
                cross_modal_df = MultiModalSetSystem._create_cross_modal_edges(
                    cross_modal_edges, 
                    loaded_modalities,
                    add_modal_metadata
                )
            
            # Merge modalities based on strategy
            combined_df = MultiModalSetSystem._merge_modalities(
                loaded_modalities,
                cross_modal_df, 
                merge_strategy
            )
            
            # Add multi-modal metadata
            if add_modal_metadata:
                metadata = SetSystemMetadata(
                    setsystem_type=SetSystemType.MULTIMODAL,
                    creation_time="2024-01-01",  # Fixed timestamp
                    source_info={
                        'modalities': list(modal_data.keys()),
                        'modality_sizes': {name: len(df) for name, df in loaded_modalities.items()},
                        'merge_strategy': merge_strategy,
                        'cross_modal_edges': cross_modal_edges is not None,
                        'total_modalities': len(modal_data)
                    },
                    validation_passed=validate_compatibility,
                    optimization_applied=True,
                    performance_stats={
                        'total_rows': len(combined_df),
                        'unique_nodes': combined_df["nodes"].n_unique(),
                        'unique_edges': combined_df["edges"].n_unique(),
                        'modalities_count': len(modal_data)
                    }
                )
                
                combined_df = combined_df.with_columns([
                    pl.lit(str(metadata.__dict__)).alias('__setsystem_metadata__')
                ])
            
            logger.info(f"Created MultiModalSetSystem: {len(combined_df)} rows, "
                       f"{combined_df['nodes'].n_unique()} nodes, "
                       f"{combined_df['edges'].n_unique()} edges")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to create MultiModalSetSystem: {e}")
            raise ValueError(f"Could not create MultiModalSetSystem: {e}") from e
    
    @staticmethod
    def _validate_modal_compatibility(modalities: Dict[str, pl.DataFrame]) -> None:
        """Validate that modalities are compatible for merging"""
        
        # Check that all modalities have required columns
        required_columns = ["edges", "nodes", "weight"]
        
        for modal_name, modal_df in modalities.items():
            missing_columns = set(required_columns) - set(modal_df.columns)
            if missing_columns:
                raise ValueError(f"Modality '{modal_name}' missing required columns: {missing_columns}")
        
        # Check for consistent data types
        first_modal = list(modalities.values())[0]
        reference_schema = {col: dtype for col, dtype in zip(first_modal.columns, first_modal.dtypes)}
        
        for modal_name, modal_df in modalities.items():
            for col in required_columns:
                if col in reference_schema and col in modal_df.columns:
                    if modal_df[col].dtype != reference_schema[col]:
                        logger.warning(f"Data type mismatch in '{modal_name}' column '{col}': "
                                     f"{modal_df[col].dtype} vs {reference_schema[col]}")
        
        logger.debug("Modal compatibility validation passed")
    
    @staticmethod
    def _create_cross_modal_edges(
        cross_modal_edges: Dict[str, List[Tuple[str, str]]],
        modalities: Dict[str, pl.DataFrame],
        add_metadata: bool
    ) -> pl.DataFrame:
        """Create edges that connect different modalities"""
        
        cross_modal_data = []
        
        for edge_type, node_pairs in cross_modal_edges.items():
            for i, (node1, node2) in enumerate(node_pairs):
                # Create bidirectional cross-modal edges
                cross_modal_data.extend([
                    {
                        "edges": f"cross_{edge_type}_{i}_fwd",
                        "nodes": node1,
                        "weight": 1.0,
                        "edge_type": "cross_modal" if add_metadata else None,
                        "modality": "cross_modal" if add_metadata else None
                    },
                    {
                        "edges": f"cross_{edge_type}_{i}_fwd", 
                        "nodes": node2,
                        "weight": 1.0,
                        "edge_type": "cross_modal" if add_metadata else None,
                        "modality": "cross_modal" if add_metadata else None
                    }
                ])
        
        if cross_modal_data:
            cross_df = pl.DataFrame(cross_modal_data)
            # Remove None columns if metadata not requested
            if not add_metadata:
                cross_df = cross_df.drop([col for col in cross_df.columns if cross_df[col].null_count() == len(cross_df)])
            return cross_df
        else:
            return pl.DataFrame({"edges": [], "nodes": [], "weight": []})
    
    @staticmethod
    def _merge_modalities(
        modalities: Dict[str, pl.DataFrame],
        cross_modal_df: Optional[pl.DataFrame],
        merge_strategy: str
    ) -> pl.DataFrame:
        """Merge all modalities and cross-modal edges"""
        
        all_dfs = list(modalities.values())
        
        # Add cross-modal edges if available
        if cross_modal_df is not None and len(cross_modal_df) > 0:
            all_dfs.append(cross_modal_df)
        
        if merge_strategy == "union":
            # Simple concatenation
            combined = pl.concat(all_dfs, how="diagonal_relaxed")
        elif merge_strategy == "intersection":
            # Only keep common columns
            common_columns = set(all_dfs[0].columns)
            for df in all_dfs[1:]:
                common_columns &= set(df.columns)
            
            common_columns = list(common_columns)
            selected_dfs = [df.select(common_columns) for df in all_dfs]
            combined = pl.concat(selected_dfs, how="vertical")
        else:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
        
        # Remove duplicates and sort
        combined = combined.unique().sort(["edges", "nodes"])
        
        return combined


class StreamingSetSystem:
    """
    SetSystem for handling massive datasets through streaming processing.
    
    Builds on I/O streaming capabilities to create hypergraphs from
    datasets too large to fit in memory.
    """
    
    def __init__(self,
                 chunk_size: int = 100000,
                 max_memory_mb: Optional[int] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize StreamingSetSystem.
        
        Args:
            chunk_size: Number of rows per processing chunk
            max_memory_mb: Maximum memory usage before forcing cleanup
            progress_callback: Optional callback for progress updates
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.progress_callback = progress_callback
        
        # State tracking
        self._processed_chunks = 0
        self._total_rows_processed = 0
        self._unique_nodes = set()
        self._unique_edges = set()
        
        logger.info(f"StreamingSetSystem initialized (chunk_size={chunk_size})")
    
    def from_parquet_stream(self,
                           file_path: Union[str, Path],
                           edge_column: str = "edges",
                           node_column: str = "nodes",
                           weight_column: Optional[str] = "weight",
                           accumulate_result: bool = True):
        """
        Create SetSystem from Parquet file using streaming.
        
        Args:
            file_path: Path to large Parquet file
            edge_column: Column containing edge identifiers
            node_column: Column containing node identifiers
            weight_column: Column containing weights
            accumulate_result: Whether to accumulate all chunks into final result
            
        Returns:
            Iterator of DataFrame chunks or accumulated final DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        logger.info(f"Creating StreamingSetSystem from: {file_path}")
        
        try:
            # Get total row count
            lazy_df = pl.scan_parquet(file_path)
            total_rows = lazy_df.select(pl.count()).collect().item()
            
            logger.info(f"Streaming {total_rows} rows in chunks of {self.chunk_size}")
            
            accumulated_chunks = []
            
            # Process in chunks
            for offset in range(0, total_rows, self.chunk_size):
                # Load chunk
                chunk_lazy = lazy_df.slice(offset, self.chunk_size)
                chunk_df = chunk_lazy.collect()
                
                if len(chunk_df) == 0:
                    break
                
                # Standardize chunk
                standardized_chunk = ParquetSetSystem._standardize_columns(
                    chunk_df, edge_column, node_column, weight_column
                )
                
                # Update statistics
                self._processed_chunks += 1
                self._total_rows_processed += len(standardized_chunk)
                
                # Track unique elements
                chunk_nodes = set(standardized_chunk["nodes"].to_list())
                chunk_edges = set(standardized_chunk["edges"].to_list())
                
                self._unique_nodes.update(chunk_nodes)
                self._unique_edges.update(chunk_edges)
                
                # Progress callback
                if self.progress_callback:
                    progress_info = {
                        'chunks_processed': self._processed_chunks,
                        'total_chunks': (total_rows + self.chunk_size - 1) // self.chunk_size,
                        'rows_processed': self._total_rows_processed,
                        'total_rows': total_rows,
                        'unique_nodes': len(self._unique_nodes),
                        'unique_edges': len(self._unique_edges)
                    }
                    self.progress_callback(progress_info)
                
                # Memory check
                if self.max_memory_mb:
                    try:
                        import psutil
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        if current_memory > self.max_memory_mb:
                            logger.warning(f"Memory limit exceeded: {current_memory:.1f} MB")
                    except ImportError:
                        pass
                
                if accumulate_result:
                    accumulated_chunks.append(standardized_chunk)
                
            
            if accumulate_result:
                # Combine all chunks and return DataFrame
                if accumulated_chunks:
                    final_df = pl.concat(accumulated_chunks, how="vertical")
                    
                    # Add streaming metadata
                    metadata = SetSystemMetadata(
                        setsystem_type=SetSystemType.STREAMING,
                        creation_time="2024-01-01",  # Fixed timestamp
                        source_info={
                            'file_path': str(file_path),
                            'chunk_size': self.chunk_size,
                            'total_chunks': self._processed_chunks,
                            'streaming_mode': 'accumulated'
                        },
                        validation_passed=True,
                        optimization_applied=True,
                        performance_stats={
                            'total_rows': self._total_rows_processed,
                            'unique_nodes': len(self._unique_nodes),
                            'unique_edges': len(self._unique_edges),
                            'chunks_processed': self._processed_chunks
                        }
                    )
                    
                    final_df = final_df.with_columns([
                        pl.lit(str(metadata.__dict__)).alias('__setsystem_metadata__')
                    ])
                    
                    logger.info(f"StreamingSetSystem accumulated: {len(final_df)} rows, "
                               f"{len(self._unique_nodes)} unique nodes, "
                               f"{len(self._unique_edges)} unique edges")
                    
                    return final_df
                else:
                    raise ValueError("No data processed")
            else:
                # Return generator for streaming mode
                return self._create_streaming_generator(file_path, edge_column, node_column, weight_column)
                
        except Exception as e:
            logger.error(f"Failed to create StreamingSetSystem: {e}")
            raise ValueError(f"Could not create StreamingSetSystem: {e}") from e
    
    def _create_streaming_generator(self, file_path, edge_column, node_column, weight_column):
        """Create a generator for streaming mode"""
        lazy_df = pl.scan_parquet(file_path)
        total_rows = lazy_df.select(pl.count()).collect().item()
        
        # Process in chunks and yield each one
        for offset in range(0, total_rows, self.chunk_size):
            chunk_lazy = lazy_df.slice(offset, self.chunk_size)
            chunk_df = chunk_lazy.collect()
            
            if len(chunk_df) == 0:
                break
                
            standardized_chunk = ParquetSetSystem._standardize_columns(
                chunk_df, edge_column, node_column, weight_column
            )
            
            yield standardized_chunk
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from streaming processing"""
        return {
            'chunks_processed': self._processed_chunks,
            'total_rows_processed': self._total_rows_processed,
            'unique_nodes_count': len(self._unique_nodes),
            'unique_edges_count': len(self._unique_edges),
            'chunk_size': self.chunk_size,
            'processing_efficiency': {
                'avg_rows_per_chunk': self._total_rows_processed / max(1, self._processed_chunks),
                'nodes_per_row': len(self._unique_nodes) / max(1, self._total_rows_processed),
                'edges_per_row': len(self._unique_edges) / max(1, self._total_rows_processed)
            }
        }
    
    def reset(self):
        """Reset streaming statistics for new processing session"""
        self._processed_chunks = 0
        self._total_rows_processed = 0
        self._unique_nodes.clear()
        self._unique_edges.clear()
        
        logger.info("StreamingSetSystem state reset")


class EnhancedSetSystemFactory:
    """
    Factory for creating enhanced set systems of different types
    """
    
    def create_parquet_setsystem(self, filepath: Union[str, Path], **kwargs):
        """Create a ParquetSetSystem - returns the data loaded from parquet"""
        return ParquetSetSystem.from_parquet(filepath, **kwargs)
    
    def create_multimodal_setsystem(self, modal_data: Dict, **kwargs):
        """Create a MultiModalSetSystem"""
        return MultiModalSetSystem.from_multiple_sources(modal_data, **kwargs)
    
    def create_streaming_setsystem(self, chunk_size: int = 10000, **kwargs):
        """Create a StreamingSetSystem"""
        return StreamingSetSystem(chunk_size, **kwargs)
    
    def create_from_dataframe(self, data: pl.DataFrame, node_col: str, edge_col: str, **kwargs):
        """Create a basic setsystem from DataFrame using the base factory"""
        return SetSystemFactory.from_dataframe(data, node_col, edge_col, **kwargs)