"""
SetSystem Factory Methods for anant library

Enhanced factory methods for creating Polars DataFrames from various input formats
with optimizations, validation, and type safety.
"""

import polars as pl
import numpy as np
import json
from typing import Any, Dict, List, Optional, Union, Iterable, Callable, Iterator
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SetSystemFactory:
    """
    Factory class for creating Polars DataFrames from various input formats
    
    Supports multiple input types with enhanced validation, optimization,
    and performance features for hypergraph construction.
    """
    
    @staticmethod
    def from_iterable_of_iterables(
        iterables: Iterable[Iterable],
        edge_id_prefix: str = "edge_",
        validate_hashable: bool = True,
        add_metadata: bool = True,
        default_weight: float = 1.0,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from iterable of iterables
        
        Parameters
        ----------
        iterables : Iterable[Iterable]
            Nested iterable structure where each sub-iterable represents an edge
        edge_id_prefix : str
            Prefix for auto-generated edge IDs
        validate_hashable : bool
            Whether to validate that all elements are hashable
        add_metadata : bool
            Whether to add creation metadata
        default_weight : float
            Default weight for edges
            
        Returns
        -------
        pl.DataFrame
            Optimized Polars DataFrame with edge-node relationships
        """
        edges = []
        nodes = []
        weights = []
        edge_sizes = []
        creation_time = datetime.now()
        
        for edge_idx, node_list in enumerate(iterables):
            edge_id = f"{edge_id_prefix}{edge_idx}"
            node_list = list(node_list)
            edge_size = len(node_list)
            
            # Validate hashable elements if requested
            if validate_hashable:
                for node in node_list:
                    if not isinstance(node, (str, int, float, tuple, frozenset)):
                        raise ValueError(f"Non-hashable element found: {node} (type: {type(node)})")
            
            # Add edge-node relationships
            for node in node_list:
                edges.append(edge_id)
                nodes.append(str(node))
                weights.append(default_weight)
                edge_sizes.append(edge_size)
        
        base_data = {
            "edges": edges,
            "nodes": nodes,
            "weight": weights,
            "edge_size": edge_sizes
        }
        
        if add_metadata:
            base_data.update({
                "created_at": [creation_time] * len(edges),
                "source_type": ["iterable"] * len(edges)
            })
        
        return SetSystemFactory._optimize_dataframe(pl.DataFrame(base_data))
    
    @staticmethod
    def from_dict_of_iterables(
        dict_data: Dict[str, Iterable],
        validate_hashable: bool = True,
        preserve_order: bool = True,
        default_weight: float = 1.0,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from dictionary of iterables
        
        Parameters
        ----------
        dict_data : Dict[str, Iterable]
            Dictionary mapping edge IDs to node iterables
        validate_hashable : bool
            Whether to validate that all elements are hashable
        preserve_order : bool
            Whether to preserve insertion order
        default_weight : float
            Default weight for edges
            
        Returns
        -------
        pl.DataFrame
            Optimized Polars DataFrame with edge-node relationships
        """
        edges = []
        nodes = []
        weights = []
        edge_sizes = []
        creation_time = datetime.now()
        
        # Process edges in order
        edge_keys = list(dict_data.keys()) if preserve_order else sorted(dict_data.keys())
        
        for edge_id in edge_keys:
            node_list = list(dict_data[edge_id])
            edge_size = len(node_list)
            
            # Validate hashable if requested
            if validate_hashable:
                for node in node_list:
                    if not isinstance(node, (str, int, float, tuple, frozenset)):
                        raise ValueError(f"Non-hashable element found: {node}")
            
            # Add relationships
            for node in node_list:
                edges.append(str(edge_id))
                nodes.append(str(node))
                weights.append(default_weight)
                edge_sizes.append(edge_size)
        
        data = {
            "edges": edges,
            "nodes": nodes,
            "weight": weights,
            "edge_size": edge_sizes,
            "created_at": [creation_time] * len(edges),
            "source_type": ["dict_iterables"] * len(edges)
        }
        
        return SetSystemFactory._optimize_dataframe(pl.DataFrame(data))
    
    @staticmethod
    def from_dict_of_dicts(
        nested_dict: Dict[str, Dict[str, Union[Iterable, Dict]]],
        cell_property_handling: str = "struct",  # "struct", "json", "separate"
        default_weight: float = 1.0,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from dictionary of dictionaries with rich cell properties
        
        Parameters
        ----------
        nested_dict : Dict[str, Dict[str, Union[Iterable, Dict]]]
            Nested dictionary with edge->node->properties structure
        cell_property_handling : str
            How to handle cell properties: "struct", "json", or "separate"
        default_weight : float
            Default weight value
            
        Returns
        -------
        pl.DataFrame
            DataFrame with cell properties properly structured
        """
        edges = []
        nodes = []
        weights = []
        cell_properties = []
        property_keys = set()
        creation_time = datetime.now()
        
        # First pass: collect all property keys
        for edge_id, edge_data in nested_dict.items():
            for node, node_data in edge_data.items():
                if isinstance(node_data, dict):
                    property_keys.update(node_data.keys())
        
        # Second pass: build structured data
        for edge_id, edge_data in nested_dict.items():
            for node, node_data in edge_data.items():
                edges.append(str(edge_id))
                nodes.append(str(node))
                
                if isinstance(node_data, dict):
                    # Extract weight if present
                    weight = node_data.get('weight', default_weight)
                    weights.append(float(weight))
                    
                    # Handle cell properties
                    props = {k: v for k, v in node_data.items() if k != 'weight'}
                    
                    if cell_property_handling == "json":
                        cell_properties.append(json.dumps(props))
                    else:
                        cell_properties.append(props)
                else:
                    weights.append(default_weight)
                    cell_properties.append({} if cell_property_handling != "json" else "{}")
        
        base_df = pl.DataFrame({
            "edges": edges,
            "nodes": nodes,
            "weight": weights,
            "created_at": [creation_time] * len(edges),
            "source_type": ["dict_dicts"] * len(edges)
        })
        
        # Add cell properties based on handling strategy
        if cell_property_handling == "struct":
            base_df = base_df.with_columns([
                pl.lit(cell_properties).alias("cell_properties")
            ])
        elif cell_property_handling == "json":
            base_df = base_df.with_columns([
                pl.lit(cell_properties).alias("cell_properties_json")
            ])
        else:  # separate columns
            for prop_key in property_keys:
                if prop_key != 'weight':
                    prop_values = [props.get(prop_key) if isinstance(props, dict) else None 
                                 for props in cell_properties]
                    base_df = base_df.with_columns([
                        pl.lit(prop_values).alias(f"cell_{prop_key}")
                    ])
        
        return SetSystemFactory._optimize_dataframe(base_df)
    
    @staticmethod
    def from_dataframe(
        df: Union[pl.DataFrame, Any],  # Any to handle pandas DataFrame
        edge_col: Union[str, int] = 0,
        node_col: Union[str, int] = 1,
        weight_col: Optional[Union[str, int]] = None,
        validate_schema: bool = True,
        optimize_memory: bool = True,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from existing DataFrame with auto-conversion
        
        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            Input DataFrame with edge-node data
        edge_col : str or int
            Column containing edge IDs
        node_col : str or int
            Column containing node IDs
        weight_col : str or int, optional
            Column containing weights
        validate_schema : bool
            Whether to validate and optimize schema
        optimize_memory : bool
            Whether to apply memory optimizations
            
        Returns
        -------
        pl.DataFrame
            Optimized Polars DataFrame
        """
        # Convert pandas to polars if needed
        if hasattr(df, 'to_pandas'):  # pandas DataFrame
            try:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    df = pl.from_pandas(df)
            except ImportError:
                pass
        elif not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        
        # Resolve column references
        def resolve_column(col_ref):
            if isinstance(col_ref, int):
                return df.columns[col_ref]
            return col_ref
        
        edge_col_name = resolve_column(edge_col)
        node_col_name = resolve_column(node_col)
        
        # Build base DataFrame
        try:
            result_df = df.select([
                pl.col(edge_col_name).cast(pl.Utf8).alias("edges"),
                pl.col(node_col_name).cast(pl.Utf8).alias("nodes")
            ])
        except Exception:
            # Fallback: direct column access
            edges_series = df.get_column(edge_col_name).cast(pl.Utf8)
            nodes_series = df.get_column(node_col_name).cast(pl.Utf8)
            result_df = pl.DataFrame({
                "edges": edges_series,
                "nodes": nodes_series
            })
        
        # Add weight column
        if weight_col is not None:
            weight_col_name = resolve_column(weight_col)
            result_df = result_df.with_columns([
                pl.col(weight_col_name).cast(pl.Float64).alias("weight")
            ])
        else:
            result_df = result_df.with_columns([
                pl.lit(kwargs.get('default_weight', 1.0)).alias("weight")
            ])
        
        # Add metadata
        result_df = result_df.with_columns([
            pl.lit(datetime.now()).alias("created_at"),
            pl.lit("dataframe").alias("source_type")
        ])
        
        # Add other columns from original DataFrame
        other_cols = [col for col in df.columns 
                     if col not in [edge_col_name, node_col_name, 
                                   resolve_column(weight_col) if weight_col else None]]
        
        if other_cols:
            result_df = result_df.with_columns([
                pl.col(col) for col in other_cols
            ])
        
        return SetSystemFactory._optimize_dataframe(result_df) if optimize_memory else result_df
    
    @staticmethod
    def from_numpy_array(
        array: np.ndarray,
        edge_id_prefix: str = "edge_",
        validate_shape: bool = True,
        add_metadata: bool = True,
        default_weight: float = 1.0,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from NumPy array
        
        Parameters
        ----------
        array : np.ndarray
            N x 2 array of edge-node pairs
        edge_id_prefix : str
            Prefix for generated edge IDs
        validate_shape : bool
            Whether to validate array shape
        add_metadata : bool
            Whether to add creation metadata
        default_weight : float
            Default weight value
            
        Returns
        -------
        pl.DataFrame
            Polars DataFrame with proper typing
        """
        if validate_shape and (array.ndim != 2 or array.shape[1] != 2):
            raise ValueError(f"Array must be N x 2, got shape {array.shape}")
        
        # Convert to string representation for consistent typing
        edges = [f"{edge_id_prefix}{row[0]}" for row in array]
        nodes = [str(row[1]) for row in array]
        
        base_data = {
            "edges": edges,
            "nodes": nodes,
            "weight": [default_weight] * len(edges)
        }
        
        if add_metadata:
            creation_time = datetime.now()
            base_data.update({
                "created_at": [creation_time] * len(edges),
                "source_type": ["numpy"] * len(edges),
                "original_edge_id": [row[0] for row in array],
                "original_node_id": [row[1] for row in array]
            })
        
        return SetSystemFactory._optimize_dataframe(pl.DataFrame(base_data))
    
    @staticmethod
    def from_parquet(
        parquet_path: Union[str, Path],
        edge_col: str = "edges",
        node_col: str = "nodes",
        lazy_loading: bool = True,
        filters: Optional[List] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame directly from parquet file
        
        Parameters
        ----------
        parquet_path : str or Path
            Path to parquet file containing edge-node data
        edge_col : str
            Column name for edges
        node_col : str
            Column name for nodes
        lazy_loading : bool
            Whether to use lazy loading
        filters : List, optional
            Polars filters to apply during loading
            
        Returns
        -------
        pl.DataFrame
            Loaded and validated DataFrame
        """
        if lazy_loading:
            df = pl.scan_parquet(parquet_path)
            if filters:
                for filter_expr in filters:
                    df = df.filter(filter_expr)
            df = df.collect()
        else:
            df = pl.read_parquet(parquet_path)
            if filters:
                for filter_expr in filters:
                    df = df.filter(filter_expr)
        
        # Validate required columns
        if edge_col not in df.columns or node_col not in df.columns:
            raise ValueError(f"Parquet file must contain columns: {edge_col}, {node_col}")
        
        # Standardize column names if needed
        if edge_col != "edges" or node_col != "nodes":
            df = df.rename({edge_col: "edges", node_col: "nodes"})
        
        # Add metadata if missing
        if "created_at" not in df.columns:
            df = df.with_columns(pl.lit(datetime.now()).alias("created_at"))
        if "source_type" not in df.columns:
            df = df.with_columns(pl.lit("parquet").alias("source_type"))
        
        return SetSystemFactory._optimize_dataframe(df)
    
    @staticmethod
    def from_multimodal(
        modal_data: Dict[str, pl.DataFrame],
        edge_prefix_map: Optional[Dict[str, str]] = None,
        merge_strategy: str = "union",  # "union", "intersection"
        **kwargs
    ) -> pl.DataFrame:
        """
        Create DataFrame from multi-modal data sources
        
        Parameters
        ----------
        modal_data : Dict[str, pl.DataFrame]
            Dictionary mapping modality names to DataFrames
        edge_prefix_map : Dict[str, str], optional
            Map modality names to edge prefixes
        merge_strategy : str
            How to combine multiple modalities
            
        Returns
        -------
        pl.DataFrame
            Combined multi-modal DataFrame
        """
        combined_dfs = []
        
        for modality, df in modal_data.items():
            # Ensure proper schema
            if "edges" not in df.columns or "nodes" not in df.columns:
                raise ValueError(f"Modal data '{modality}' missing required columns")
            
            # Add modality information
            prefix = edge_prefix_map.get(modality, modality) if edge_prefix_map else modality
            
            modal_df = df.with_columns([
                pl.concat_str([pl.lit(f"{prefix}_"), pl.col("edges")]).alias("edges"),
                pl.lit(modality).alias("modality"),
                pl.lit(datetime.now()).alias("created_at"),
                pl.lit("multimodal").alias("source_type")
            ])
            
            combined_dfs.append(modal_df)
        
        if merge_strategy == "union":
            result = pl.concat(combined_dfs, how="diagonal")
        else:  # intersection - find common nodes
            common_nodes = None
            for df in combined_dfs:
                nodes = set(df["nodes"].unique().to_list())
                common_nodes = nodes if common_nodes is None else common_nodes.intersection(nodes)
            
            if common_nodes:
                filtered_dfs = [df.filter(pl.col("nodes").is_in(list(common_nodes))) 
                               for df in combined_dfs]
                result = pl.concat(filtered_dfs, how="diagonal")
            else:
                # No common nodes, return empty DataFrame
                result = pl.DataFrame({
                    "edges": [], "nodes": [], "weight": [], "modality": [],
                    "created_at": [], "source_type": []
                })
        
        return SetSystemFactory._optimize_dataframe(result)
    
    @staticmethod
    def create_streaming_setsystem(
        data_source: Union[str, Iterator],
        chunk_size: int = 10000,
        processing_func: Optional[Callable] = None
    ) -> 'StreamingSetSystem':
        """
        Create streaming SetSystem for very large datasets
        
        Parameters
        ----------
        data_source : str or Iterator
            Data source (file path or iterator)
        chunk_size : int
            Processing chunk size
        processing_func : Callable, optional
            Function to process each chunk
            
        Returns
        -------
        StreamingSetSystem
            Streaming data processor
        """
        return StreamingSetSystem(data_source, chunk_size, processing_func)
    
    @staticmethod
    def _optimize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        """Apply memory and performance optimizations"""
        optimizations = []
        
        if df.height == 0:
            # Return empty DataFrame as-is to avoid division by zero
            return df
            
        # Optimize string columns using categorical if beneficial
        for col in ["edges", "nodes"]:
            if col in df.columns:
                unique_ratio = df[col].n_unique() / df.height
                if unique_ratio < 0.5:  # Less than 50% unique -> use categorical
                    optimizations.append(pl.col(col).cast(pl.Categorical).alias(col))
                else:
                    optimizations.append(pl.col(col))
        
        # Optimize numeric columns
        if "weight" in df.columns:
            try:
                weight_min = df["weight"].min()
                weight_max = df["weight"].max()
                if (weight_min is not None and weight_max is not None and
                    isinstance(weight_min, (int, float)) and isinstance(weight_max, (int, float)) and
                    weight_min >= -3.4e38 and weight_max <= 3.4e38):
                    optimizations.append(pl.col("weight").cast(pl.Float32).alias("weight"))
                else:
                    optimizations.append(pl.col("weight"))
            except Exception:
                optimizations.append(pl.col("weight"))
        
        # Keep other columns as-is
        other_cols = [col for col in df.columns if col not in ["edges", "nodes", "weight"]]
        optimizations.extend([pl.col(col) for col in other_cols])
        
        return df.select(optimizations)


class StreamingSetSystem:
    """
    Streaming SetSystem for memory-efficient processing of large datasets
    """
    
    def __init__(
        self,
        data_source: Union[str, Iterator],
        chunk_size: int = 10000,
        processing_func: Optional[Callable] = None
    ):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.processing_func = processing_func or (lambda x: x)
        self._total_processed = 0
    
    def __iter__(self):
        if isinstance(self.data_source, str):
            # File-based streaming
            try:
                # Read in chunks using batched approach
                lazy_df = pl.scan_parquet(self.data_source)
                total_rows = lazy_df.select(pl.len()).collect().item()
                
                for offset in range(0, total_rows, self.chunk_size):
                    chunk = lazy_df.slice(offset, self.chunk_size).collect()
                    processed_chunk = self.processing_func(chunk)
                    self._total_processed += chunk.height
                    yield processed_chunk
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                raise
        else:
            # Iterator-based streaming
            for chunk in self.data_source:
                processed_chunk = self.processing_func(chunk)
                self._total_processed += len(chunk) if hasattr(chunk, '__len__') else 1
                yield processed_chunk
    
    def to_dataframe(self, max_rows: Optional[int] = None) -> pl.DataFrame:
        """Materialize stream to DataFrame with optional row limit"""
        chunks = []
        total_rows = 0
        
        for chunk in self:
            chunks.append(chunk)
            total_rows += chunk.height if hasattr(chunk, 'height') else len(chunk)
            
            if max_rows and total_rows >= max_rows:
                break
        
        return pl.concat(chunks, how="diagonal") if chunks else pl.DataFrame()
    
    @property
    def total_processed(self) -> int:
        """Get total number of items processed"""
        return self._total_processed