"""
Working Advanced I/O Implementation for Anant Library

This module provides production-grade I/O capabilities that actually work
with the existing Anant infrastructure.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Iterator
import logging
import time

logger = logging.getLogger(__name__)

# Import Anant classes - use relative imports to avoid circular issues
try:
    from ..classes.hypergraph import Hypergraph
    from ..optimization import PerformanceOptimizer
except ImportError:
    # Fallback for direct usage
    import sys
    sys.path.append('/home/amansingh/dev/ai/anant/anant')
    from anant.classes.hypergraph import Hypergraph
    from anant.optimization import PerformanceOptimizer


class WorkingAnantIO:
    """
    Working implementation of advanced I/O operations for Hypergraph objects.
    
    This class provides production-grade I/O capabilities that work with
    the existing Anant infrastructure and eliminate CSV conversion overhead.
    """
    
    SUPPORTED_COMPRESSIONS = ['snappy', 'gzip', 'lz4', 'zstd', 'uncompressed']
    
    @staticmethod
    def save_hypergraph(
        hypergraph: Hypergraph,
        path: Union[str, Path], 
        format: str = "parquet",
        compression: str = "snappy",
        include_metadata: bool = True
    ) -> None:
        """
        Save a Hypergraph with advanced options.
        
        Args:
            hypergraph: The Hypergraph to save
            path: Output path for the file
            format: File format ('parquet' or 'csv')
            compression: Compression format for Parquet
            include_metadata: Whether to include hypergraph metadata
        """
        path = Path(path)
        
        logger.info(f"Saving Hypergraph to {path} (format={format}, compression={compression})")
        
        try:
            # Get the underlying DataFrame 
            df = hypergraph.incidence_df
            
            # Add metadata if requested
            if include_metadata:
                metadata_dict = {
                    'num_nodes': hypergraph.num_nodes,
                    'num_edges': hypergraph.num_edges,
                    'anant_version': '0.1.0',
                }
                
                # Add optimizer statistics if available
                if hasattr(hypergraph, '_optimizer') and hypergraph._optimizer:
                    try:
                        stats = hypergraph._optimizer.get_performance_stats()
                        metadata_dict.update({
                            'optimizer_cache_hits': stats.get('cache_hits', 0),
                            'optimizer_query_count': stats.get('query_count', 0)
                        })
                    except:
                        pass  # Skip if optimizer stats not available
                
                # Add metadata as a column (will be filtered out on load)
                df = df.with_columns([
                    pl.lit(str(metadata_dict)).alias('__anant_metadata__')
                ])
            
            # Save based on format
            if format.lower() == "parquet":
                compression_arg = None if compression == "uncompressed" else compression
                df.write_parquet(path, compression=compression_arg, use_pyarrow=True)
            elif format.lower() == "csv":
                df.write_csv(path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Successfully saved Hypergraph ({hypergraph.num_nodes} nodes, "
                       f"{hypergraph.num_edges} edges) to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save Hypergraph to {path}: {e}")
            raise IOError(f"Could not save Hypergraph: {e}") from e
    
    @staticmethod  
    def load_hypergraph(
        path: Union[str, Path],
        format: str = "parquet",
        lazy: bool = True,
        columns: Optional[List[str]] = None,
        with_optimizer: bool = True
    ) -> Hypergraph:
        """
        Load a Hypergraph with advanced options.
        
        Args:
            path: Path to the file
            format: File format ('parquet' or 'csv')
            lazy: Whether to use lazy loading (memory efficient)
            columns: Specific columns to load (None for all)
            with_optimizer: Whether to attach performance optimizer
            
        Returns:
            Hypergraph object loaded from file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        logger.info(f"Loading Hypergraph from {path} (format={format}, lazy={lazy})")
        
        try:
            # Load based on format and lazy option
            if format.lower() == "parquet":
                if lazy:
                    lazy_df = pl.scan_parquet(path)
                    if columns:
                        lazy_df = lazy_df.select(columns)
                    df = lazy_df.collect()
                else:
                    df = pl.read_parquet(path, columns=columns)
            elif format.lower() == "csv":
                if lazy:
                    lazy_df = pl.scan_csv(path)
                    if columns:
                        lazy_df = lazy_df.select(columns)
                    df = lazy_df.collect()
                else:
                    df = pl.read_csv(path, columns=columns)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Extract and remove metadata if present
            metadata = None
            if '__anant_metadata__' in df.columns:
                try:
                    metadata_str = df['__anant_metadata__'][0]
                    metadata = eval(metadata_str)  # Safe since we control the format
                    df = df.drop('__anant_metadata__')
                except Exception:
                    logger.warning("Could not parse metadata from file")
            
            # Create Hypergraph
            hypergraph = Hypergraph(df)
            
            # Attach performance optimizer if requested
            if with_optimizer:
                try:
                    optimizer = PerformanceOptimizer(hypergraph)
                    hypergraph._optimizer = optimizer
                    
                    # Restore optimizer statistics if available
                    if metadata:
                        stats = {
                            'cache_hits': metadata.get('optimizer_cache_hits', 0),
                            'query_count': metadata.get('optimizer_query_count', 0)
                        }
                        optimizer._stats.update(stats)
                except Exception:
                    logger.warning("Could not attach performance optimizer")
            
            # Log metadata info if available
            if metadata:
                logger.info(f"Loaded Hypergraph with metadata: {metadata.get('num_nodes')} nodes, "
                           f"{metadata.get('num_edges')} edges")
            else:
                logger.info(f"Loaded Hypergraph: {hypergraph.num_nodes} nodes, {hypergraph.num_edges} edges")
            
            return hypergraph
            
        except Exception as e:
            logger.error(f"Failed to load Hypergraph from {path}: {e}")
            raise ValueError(f"Could not load Hypergraph: {e}") from e
    
    @staticmethod
    def stream_dataset(
        path: Union[str, Path],
        chunk_size: int = 50000,
        format: str = "parquet"
    ) -> Iterator[Hypergraph]:
        """
        Stream a large dataset in chunks to handle datasets larger than memory.
        
        Args:
            path: Path to the dataset file
            chunk_size: Number of rows per chunk
            format: File format ('parquet' or 'csv')
            
        Yields:
            Hypergraph objects for each chunk of the dataset
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        logger.info(f"Streaming dataset from {path} with chunk_size={chunk_size}")
        
        try:
            # Get total row count first
            if format.lower() == "parquet":
                lazy_df = pl.scan_parquet(path)
            elif format.lower() == "csv":
                lazy_df = pl.scan_csv(path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            total_rows = lazy_df.select(pl.count()).collect().item()
            
            logger.info(f"Dataset has {total_rows} rows, processing in {chunk_size}-row chunks")
            
            # Process in chunks
            for offset in range(0, total_rows, chunk_size):
                chunk_df = lazy_df.slice(offset, chunk_size).collect()
                
                if len(chunk_df) == 0:
                    break
                
                # Remove metadata column if present
                if '__anant_metadata__' in chunk_df.columns:
                    chunk_df = chunk_df.drop('__anant_metadata__')
                
                # Create Hypergraph from chunk  
                chunk_hg = Hypergraph(chunk_df)
                
                logger.debug(f"Yielding chunk {offset//chunk_size + 1}: "
                            f"{chunk_hg.num_nodes} nodes, {chunk_hg.num_edges} edges")
                
                yield chunk_hg
                
        except Exception as e:
            logger.error(f"Failed to stream dataset from {path}: {e}")
            raise ValueError(f"Could not stream dataset: {e}") from e
    
    @staticmethod
    def merge_datasets(
        paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        format: str = "parquet",
        merge_strategy: str = "union"
    ) -> Hypergraph:
        """
        Load and merge multiple files into a single Hypergraph.
        
        Args:
            paths: List of paths to files
            output_path: Path to save merged dataset (optional)
            format: File format ('parquet' or 'csv')
            merge_strategy: How to merge files ('union' or 'intersection')
            
        Returns:
            Merged Hypergraph object
        """
        if not paths:
            raise ValueError("No file paths provided")
        
        logger.info(f"Merging {len(paths)} files with {merge_strategy} strategy")
        
        try:
            # Load all files
            dataframes = []
            for path in paths:
                path = Path(path)
                if not path.exists():
                    logger.warning(f"File not found, skipping: {path}")
                    continue
                
                if format.lower() == "parquet":
                    df = pl.read_parquet(path)
                elif format.lower() == "csv":
                    df = pl.read_csv(path)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                # Remove metadata columns
                if '__anant_metadata__' in df.columns:
                    df = df.drop('__anant_metadata__')
                
                dataframes.append(df)
                logger.debug(f"Loaded {path}: {len(df)} rows")
            
            if len(dataframes) == 0:
                raise ValueError("No valid files found to merge")
            
            if len(dataframes) == 1:
                merged_df = dataframes[0]
            else:
                # Merge based on strategy
                if merge_strategy == "union":
                    merged_df = pl.concat(dataframes, how="diagonal_relaxed")
                elif merge_strategy == "intersection":
                    # Find common columns
                    common_columns = set(dataframes[0].columns)
                    for df in dataframes[1:]:
                        common_columns &= set(df.columns)
                    common_columns = list(common_columns)
                    
                    logger.info(f"Using common columns: {common_columns}")
                    selected_dfs = [df.select(common_columns) for df in dataframes]
                    merged_df = pl.concat(selected_dfs, how="vertical")
                else:
                    raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
                
                # Remove duplicates
                merged_df = merged_df.unique()
            
            # Create merged Hypergraph
            merged_hg = Hypergraph(merged_df)
            
            # Save merged result if output path provided
            if output_path:
                WorkingAnantIO.save_hypergraph(merged_hg, output_path, format=format)
                logger.info(f"Saved merged dataset to {output_path}")
            
            logger.info(f"Merged dataset: {merged_hg.num_nodes} nodes, {merged_hg.num_edges} edges")
            
            return merged_hg
            
        except Exception as e:
            logger.error(f"Failed to merge datasets: {e}")
            raise ValueError(f"Could not merge datasets: {e}") from e


# Convenience functions
def save_hypergraph_parquet(hypergraph: Hypergraph, path: Union[str, Path], **kwargs):
    """Convenience function to save Hypergraph to Parquet."""
    return WorkingAnantIO.save_hypergraph(hypergraph, path, format="parquet", **kwargs)


def load_hypergraph_parquet(path: Union[str, Path], **kwargs):
    """Convenience function to load Hypergraph from Parquet.""" 
    return WorkingAnantIO.load_hypergraph(path, format="parquet", **kwargs)


def stream_parquet_dataset(path: Union[str, Path], chunk_size: int = 50000):
    """Convenience function to stream Parquet dataset."""
    return WorkingAnantIO.stream_dataset(path, chunk_size=chunk_size, format="parquet")