"""
Lazy Loading Capabilities for Anant Library

Provides memory-efficient lazy loading for large datasets using Polars' lazy API.
Enables working with datasets larger than available memory through deferred computation.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable
import logging

from ..classes.hypergraph import Hypergraph
from ..optimization import PerformanceOptimizer

logger = logging.getLogger(__name__)


class LazyHypergraph:
    """
    A lazy-evaluated Hypergraph that defers computation until results are needed.
    
    This enables working with datasets larger than memory by only computing
    results when explicitly requested through .collect() or similar operations.
    """
    
    def __init__(self, lazy_frame: pl.LazyFrame, source_path: Optional[Union[str, Path]] = None):
        """
        Initialize LazyHypergraph with a Polars LazyFrame.
        
        Args:
            lazy_frame: Polars LazyFrame containing the hypergraph data
            source_path: Optional source file path for reference
        """
        self._lazy_frame = lazy_frame
        self._source_path = Path(source_path) if source_path else None
        self._optimizer = None
        
        logger.debug(f"Created LazyHypergraph from {source_path or 'LazyFrame'}")
    
    def collect(self, with_optimizer: bool = True) -> Hypergraph:
        """
        Materialize the lazy computation into a concrete Hypergraph.
        
        Args:
            with_optimizer: Whether to attach performance optimizer to result
            
        Returns:
            Materialized Hypergraph object
        """
        logger.info("Collecting LazyHypergraph into concrete Hypergraph")
        
        try:
            # Execute the lazy computation
            df = self._lazy_frame.collect()
            
            # Create Hypergraph
            hypergraph = Hypergraph(df)
            
            # Attach optimizer if requested
            if with_optimizer:
                optimizer = PerformanceOptimizer(hypergraph)
                hypergraph._optimizer = optimizer
            
            logger.info(f"Materialized LazyHypergraph: {hypergraph.num_nodes} nodes, "
                       f"{hypergraph.num_edges} edges")
            
            return hypergraph
            
        except Exception as e:
            logger.error(f"Failed to collect LazyHypergraph: {e}")
            raise RuntimeError(f"Could not materialize LazyHypergraph: {e}") from e
    
    def select(self, *columns: str) -> 'LazyHypergraph':
        """
        Lazily select specific columns.
        
        Args:
            *columns: Column names to select
            
        Returns:
            New LazyHypergraph with column selection applied
        """
        new_lazy_frame = self._lazy_frame.select(*columns)
        return LazyHypergraph(new_lazy_frame, self._source_path)
    
    def filter(self, predicate: pl.Expr) -> 'LazyHypergraph':
        """
        Lazily filter rows based on a predicate.
        
        Args:
            predicate: Polars expression for filtering
            
        Returns:
            New LazyHypergraph with filter applied
        """
        new_lazy_frame = self._lazy_frame.filter(predicate)
        return LazyHypergraph(new_lazy_frame, self._source_path)
    
    def with_columns(self, *exprs: pl.Expr) -> 'LazyHypergraph':
        """
        Lazily add or modify columns.
        
        Args:
            *exprs: Polars expressions for new columns
            
        Returns:
            New LazyHypergraph with column operations applied
        """
        new_lazy_frame = self._lazy_frame.with_columns(*exprs)
        return LazyHypergraph(new_lazy_frame, self._source_path)
    
    def slice(self, offset: int, length: int) -> 'LazyHypergraph':
        """
        Lazily slice a portion of the data.
        
        Args:
            offset: Starting row index
            length: Number of rows to include
            
        Returns:
            New LazyHypergraph with slice applied
        """
        new_lazy_frame = self._lazy_frame.slice(offset, length)
        return LazyHypergraph(new_lazy_frame, self._source_path)
    
    def head(self, n: int = 5) -> Hypergraph:
        """
        Materialize and return the first n rows as a Hypergraph.
        
        Args:
            n: Number of rows to return
            
        Returns:
            Small Hypergraph with first n rows
        """
        logger.debug(f"Getting head({n}) from LazyHypergraph")
        
        df = self._lazy_frame.head(n).collect()
        return Hypergraph(df)
    
    def tail(self, n: int = 5) -> Hypergraph:
        """
        Materialize and return the last n rows as a Hypergraph.
        
        Args:
            n: Number of rows to return
            
        Returns:
            Small Hypergraph with last n rows
        """
        logger.debug(f"Getting tail({n}) from LazyHypergraph")
        
        df = self._lazy_frame.tail(n).collect()
        return Hypergraph(df)
    
    def describe(self) -> pl.DataFrame:
        """
        Get summary statistics without fully materializing the data.
        
        Returns:
            DataFrame with summary statistics
        """
        logger.debug("Computing summary statistics for LazyHypergraph")
        
        return self._lazy_frame.describe().collect()
    
    def count_rows(self) -> int:
        """
        Count rows without materializing the full dataset.
        
        Returns:
            Total number of rows
        """
        return self._lazy_frame.select(pl.count()).collect().item()
    
    def count_unique_nodes(self, node_column: str = "nodes") -> int:
        """
        Count unique nodes without materializing the full dataset.
        
        Args:
            node_column: Column name containing node identifiers
            
        Returns:
            Number of unique nodes
        """
        return self._lazy_frame.select(pl.col(node_column).n_unique()).collect().item()
    
    def count_unique_edges(self, edge_column: str = "edges") -> int:
        """
        Count unique edges without materializing the full dataset.
        
        Args:
            edge_column: Column name containing edge identifiers
            
        Returns:
            Number of unique edges
        """
        return self._lazy_frame.select(pl.col(edge_column).n_unique()).collect().item()
    
    @property
    def columns(self) -> List[str]:
        """Get column names without materializing data."""
        return self._lazy_frame.columns
    
    @property 
    def source_path(self) -> Optional[Path]:
        """Get the source file path if available."""
        return self._source_path


class LazyLoader:
    """
    Factory class for creating LazyHypergraph objects from various sources.
    
    Provides convenient methods for lazy loading from files and other sources
    while maintaining memory efficiency.
    """
    
    @staticmethod
    def from_parquet(
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        row_count_limit: Optional[int] = None
    ) -> LazyHypergraph:
        """
        Create LazyHypergraph from Parquet file with lazy loading.
        
        Args:
            path: Path to Parquet file
            columns: Specific columns to load (None for all)
            filters: Filter conditions to apply during scan
            row_count_limit: Maximum number of rows to consider
            
        Returns:
            LazyHypergraph object for deferred computation
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        
        logger.info(f"Creating LazyHypergraph from {path}")
        
        try:
            # Create lazy frame
            lazy_df = pl.scan_parquet(path)
            
            # Apply column selection if specified
            if columns:
                lazy_df = lazy_df.select(columns)
            
            # Apply filters if specified
            if filters:
                for column, condition in filters.items():
                    if isinstance(condition, (list, tuple)):
                        lazy_df = lazy_df.filter(pl.col(column).is_in(condition))
                    else:
                        lazy_df = lazy_df.filter(pl.col(column) == condition)
            
            # Apply row limit if specified
            if row_count_limit:
                lazy_df = lazy_df.head(row_count_limit)
            
            return LazyHypergraph(lazy_df, path)
            
        except Exception as e:
            logger.error(f"Failed to create LazyHypergraph from {path}: {e}")
            raise ValueError(f"Could not create LazyHypergraph: {e}") from e
    
    @staticmethod
    def from_csv(
        path: Union[str, Path], 
        separator: str = ",",
        columns: Optional[List[str]] = None,
        **scan_kwargs
    ) -> LazyHypergraph:
        """
        Create LazyHypergraph from CSV file with lazy loading.
        
        Args:
            path: Path to CSV file  
            separator: Column separator character
            columns: Specific columns to load (None for all)
            **scan_kwargs: Additional arguments for pl.scan_csv
            
        Returns:
            LazyHypergraph object for deferred computation
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        logger.info(f"Creating LazyHypergraph from CSV: {path}")
        
        try:
            # Create lazy frame from CSV
            lazy_df = pl.scan_csv(path, separator=separator, **scan_kwargs)
            
            # Apply column selection if specified
            if columns:
                lazy_df = lazy_df.select(columns)
            
            return LazyHypergraph(lazy_df, path)
            
        except Exception as e:
            logger.error(f"Failed to create LazyHypergraph from CSV {path}: {e}")
            raise ValueError(f"Could not create LazyHypergraph from CSV: {e}") from e
    
    @staticmethod
    def from_multiple_files(
        paths: List[Union[str, Path]],
        file_format: str = "parquet",
        **load_kwargs
    ) -> LazyHypergraph:
        """
        Create LazyHypergraph from multiple files with lazy concatenation.
        
        Args:
            paths: List of file paths
            file_format: Format of files ('parquet' or 'csv')
            **load_kwargs: Additional arguments for file scanning
            
        Returns:
            LazyHypergraph with concatenated data from all files
        """
        if not paths:
            raise ValueError("No file paths provided")
        
        logger.info(f"Creating LazyHypergraph from {len(paths)} {file_format} files")
        
        try:
            lazy_frames = []
            
            for path in paths:
                path = Path(path)
                if not path.exists():
                    logger.warning(f"File not found, skipping: {path}")
                    continue
                
                if file_format.lower() == "parquet":
                    lazy_df = pl.scan_parquet(path, **load_kwargs)
                elif file_format.lower() == "csv":
                    lazy_df = pl.scan_csv(path, **load_kwargs)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                lazy_frames.append(lazy_df)
            
            if not lazy_frames:
                raise ValueError("No valid files found to load")
            
            # Concatenate all lazy frames
            if len(lazy_frames) == 1:
                combined_lazy_df = lazy_frames[0]
            else:
                combined_lazy_df = pl.concat(lazy_frames, how="diagonal_relaxed")
            
            return LazyHypergraph(combined_lazy_df)
            
        except Exception as e:
            logger.error(f"Failed to create LazyHypergraph from multiple files: {e}")
            raise ValueError(f"Could not create LazyHypergraph from multiple files: {e}") from e
    
    @staticmethod
    def chain_operations(
        lazy_hg: LazyHypergraph,
        operations: List[Callable[[LazyHypergraph], LazyHypergraph]]
    ) -> LazyHypergraph:
        """
        Chain multiple lazy operations together.
        
        Args:
            lazy_hg: Starting LazyHypergraph
            operations: List of functions that take and return LazyHypergraph
            
        Returns:
            LazyHypergraph with all operations chained
        """
        result = lazy_hg
        
        for operation in operations:
            result = operation(result)
        
        return result