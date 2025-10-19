"""
Streaming I/O Capabilities for Anant Library

Provides streaming data processing for datasets larger than memory,
enabling real-time processing and incremental analytics on massive datasets.
"""

import polars as pl
from pathlib import Path
from typing import Union, Iterator, Optional, Dict, Any, Callable, List
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class StreamingStats:
    """Statistics for streaming operations."""
    total_chunks: int = 0
    processed_chunks: int = 0
    total_rows: int = 0
    processed_rows: int = 0
    start_time: float = 0.0
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0


class StreamingDatasetReader:
    """
    High-performance streaming reader for large datasets.
    
    Processes datasets in configurable chunks to handle files larger than memory
    while maintaining memory efficiency and providing progress tracking.
    """
    
    def __init__(self, 
                 chunk_size: int = 50000,
                 memory_limit_mb: Optional[int] = None,
                 progress_callback: Optional[Callable[[StreamingStats], None]] = None):
        """
        Initialize streaming reader.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Optional memory limit for safety
            progress_callback: Optional callback for progress updates
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.progress_callback = progress_callback
        self.stats = StreamingStats()
        
        logger.info(f"StreamingDatasetReader initialized with chunk_size={chunk_size}")
    
    def stream_parquet(self, 
                      path: Union[str, Path],
                      columns: Optional[List[str]] = None,
                      filters: Optional[Dict[str, Any]] = None) -> Iterator[pl.DataFrame]:
        """
        Stream a Parquet file in chunks.
        
        Args:
            path: Path to Parquet file
            columns: Specific columns to read
            filters: Filter conditions to apply
            
        Yields:
            DataFrame chunks from the file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        
        logger.info(f"Streaming Parquet file: {path}")
        
        try:
            # Initialize stats
            self.stats.start_time = time.time()
            
            # Get total row count for progress tracking
            lazy_df = pl.scan_parquet(path)
            if filters:
                for column, condition in filters.items():
                    if isinstance(condition, (list, tuple)):
                        lazy_df = lazy_df.filter(pl.col(column).is_in(condition))
                    else:
                        lazy_df = lazy_df.filter(pl.col(column) == condition)
            
            self.stats.total_rows = lazy_df.select(pl.count()).collect().item()
            self.stats.total_chunks = (self.stats.total_rows + self.chunk_size - 1) // self.chunk_size
            
            logger.info(f"Dataset has {self.stats.total_rows} rows, "
                       f"will process in {self.stats.total_chunks} chunks")
            
            # Stream chunks
            for offset in range(0, self.stats.total_rows, self.chunk_size):
                # Apply slice to lazy frame
                chunk_lazy = lazy_df.slice(offset, self.chunk_size)
                
                # Apply column selection if specified
                if columns:
                    chunk_lazy = chunk_lazy.select(columns)
                
                # Collect chunk
                chunk_df = chunk_lazy.collect()
                
                if len(chunk_df) == 0:
                    break
                
                # Update stats
                self.stats.processed_chunks += 1
                self.stats.processed_rows += len(chunk_df)
                
                # Memory monitoring
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.stats.current_memory_mb = memory_mb
                    self.stats.peak_memory_mb = max(self.stats.peak_memory_mb, memory_mb)
                    
                    # Check memory limit
                    if self.memory_limit_mb and memory_mb > self.memory_limit_mb:
                        logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds limit "
                                     f"({self.memory_limit_mb} MB)")
                        
                except ImportError:
                    # psutil not available, skip memory monitoring
                    pass
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(self.stats)
                
                logger.debug(f"Yielding chunk {self.stats.processed_chunks}/{self.stats.total_chunks}: "
                            f"{len(chunk_df)} rows")
                
                yield chunk_df
                
        except Exception as e:
            logger.error(f"Failed to stream Parquet file {path}: {e}")
            raise RuntimeError(f"Streaming failed: {e}") from e
    
    def stream_csv(self,
                   path: Union[str, Path],
                   separator: str = ",",
                   columns: Optional[List[str]] = None,
                   **scan_kwargs) -> Iterator[pl.DataFrame]:
        """
        Stream a CSV file in chunks.
        
        Args:
            path: Path to CSV file
            separator: Column separator
            columns: Specific columns to read
            **scan_kwargs: Additional arguments for CSV scanning
            
        Yields:
            DataFrame chunks from the file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        logger.info(f"Streaming CSV file: {path}")
        
        try:
            # Initialize stats
            self.stats.start_time = time.time()
            
            # Get total row count
            lazy_df = pl.scan_csv(path, separator=separator, **scan_kwargs)
            self.stats.total_rows = lazy_df.select(pl.count()).collect().item()
            self.stats.total_chunks = (self.stats.total_rows + self.chunk_size - 1) // self.chunk_size
            
            logger.info(f"CSV has {self.stats.total_rows} rows, "
                       f"will process in {self.stats.total_chunks} chunks")
            
            # Stream chunks
            for offset in range(0, self.stats.total_rows, self.chunk_size):
                chunk_lazy = lazy_df.slice(offset, self.chunk_size)
                
                if columns:
                    chunk_lazy = chunk_lazy.select(columns)
                
                chunk_df = chunk_lazy.collect()
                
                if len(chunk_df) == 0:
                    break
                
                # Update stats
                self.stats.processed_chunks += 1
                self.stats.processed_rows += len(chunk_df)
                
                yield chunk_df
                
        except Exception as e:
            logger.error(f"Failed to stream CSV file {path}: {e}")
            raise RuntimeError(f"CSV streaming failed: {e}") from e
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current streaming progress information."""
        elapsed_time = time.time() - self.stats.start_time
        
        progress_pct = 0.0
        if self.stats.total_chunks > 0:
            progress_pct = (self.stats.processed_chunks / self.stats.total_chunks) * 100
        
        rows_per_sec = 0.0
        if elapsed_time > 0:
            rows_per_sec = self.stats.processed_rows / elapsed_time
        
        return {
            'progress_percentage': progress_pct,
            'processed_chunks': self.stats.processed_chunks,
            'total_chunks': self.stats.total_chunks,
            'processed_rows': self.stats.processed_rows,
            'total_rows': self.stats.total_rows,
            'elapsed_time_sec': elapsed_time,
            'rows_per_second': rows_per_sec,
            'current_memory_mb': self.stats.current_memory_mb,
            'peak_memory_mb': self.stats.peak_memory_mb
        }


class ChunkedHypergraphProcessor:
    """
    Process streaming data chunks and maintain hypergraph state.
    
    Enables incremental processing of large datasets while maintaining
    hypergraph properties and enabling real-time analytics.
    """
    
    def __init__(self, 
                 accumulate_results: bool = True,
                 max_memory_mb: Optional[int] = None):
        """
        Initialize chunked processor.
        
        Args:
            accumulate_results: Whether to accumulate results across chunks
            max_memory_mb: Maximum memory usage before forcing cleanup
        """
        self.accumulate_results = accumulate_results
        self.max_memory_mb = max_memory_mb
        
        # State tracking
        self._accumulated_df = None
        self._node_counts = {}
        self._edge_counts = {}
        self._processed_chunks = 0
        
        logger.info(f"ChunkedHypergraphProcessor initialized (accumulate={accumulate_results})")
    
    def process_chunk(self, 
                     chunk_df: pl.DataFrame,
                     node_column: str = "nodes",
                     edge_column: str = "edges") -> Dict[str, Any]:
        """
        Process a single chunk and update state.
        
        Args:
            chunk_df: DataFrame chunk to process
            node_column: Column containing node identifiers
            edge_column: Column containing edge identifiers
            
        Returns:
            Processing results for this chunk
        """
        self._processed_chunks += 1
        
        logger.debug(f"Processing chunk {self._processed_chunks}: {len(chunk_df)} rows")
        
        try:
            # Basic statistics for this chunk
            chunk_stats = {
                'chunk_id': self._processed_chunks,
                'chunk_rows': len(chunk_df),
                'unique_nodes': chunk_df[node_column].n_unique(),
                'unique_edges': chunk_df[edge_column].n_unique(),
            }
            
            # Update accumulated counts if enabled
            if self.accumulate_results:
                # Update node counts
                node_counts = chunk_df.group_by(node_column).len().to_dict()
                for node, count in zip(node_counts[node_column], node_counts['len']):
                    self._node_counts[node] = self._node_counts.get(node, 0) + count
                
                # Update edge counts  
                edge_counts = chunk_df.group_by(edge_column).len().to_dict()
                for edge, count in zip(edge_counts[edge_column], edge_counts['len']):
                    self._edge_counts[edge] = self._edge_counts.get(edge, 0) + count
                
                # Accumulate DataFrame if memory allows
                if self._should_accumulate_df():
                    if self._accumulated_df is None:
                        self._accumulated_df = chunk_df
                    else:
                        self._accumulated_df = pl.concat([self._accumulated_df, chunk_df])
                    
                    chunk_stats['accumulated_rows'] = len(self._accumulated_df)
            
            # Add global statistics if available
            if self.accumulate_results:
                chunk_stats.update({
                    'total_unique_nodes': len(self._node_counts),
                    'total_unique_edges': len(self._edge_counts),
                    'total_chunks_processed': self._processed_chunks,
                })
            
            return chunk_stats
            
        except Exception as e:
            logger.error(f"Failed to process chunk {self._processed_chunks}: {e}")
            raise RuntimeError(f"Chunk processing failed: {e}") from e
    
    def _should_accumulate_df(self) -> bool:
        """Check if we should continue accumulating DataFrame based on memory."""
        if not self.max_memory_mb:
            return True
        
        try:
            import psutil
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            
            if current_memory_mb > self.max_memory_mb:
                logger.warning(f"Memory limit reached ({current_memory_mb:.1f} MB), "
                              f"stopping DataFrame accumulation")
                return False
        
        except ImportError:
            # psutil not available, assume we can accumulate
            pass
        
        return True
    
    def get_accumulated_hypergraph(self):
        """
        Get accumulated hypergraph if available.
        
        Returns:
            Hypergraph created from accumulated data, or None if not available
        """
        if self._accumulated_df is None:
            logger.warning("No accumulated DataFrame available")
            return None
        
        try:
            # Import here to avoid circular imports
            from ..classes.hypergraph import Hypergraph
            from ..optimization import PerformanceOptimizer
            
            # Create hypergraph
            hg = Hypergraph(self._accumulated_df)
            
            # Attach optimizer
            optimizer = PerformanceOptimizer(hg)
            hg._optimizer = optimizer
            
            logger.info(f"Created accumulated Hypergraph: {hg.num_nodes} nodes, "
                       f"{hg.num_edges} edges")
            
            return hg
            
        except Exception as e:
            logger.error(f"Failed to create accumulated Hypergraph: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from processing."""
        stats = {
            'chunks_processed': self._processed_chunks,
            'total_unique_nodes': len(self._node_counts),
            'total_unique_edges': len(self._edge_counts),
            'has_accumulated_df': self._accumulated_df is not None,
        }
        
        if self._accumulated_df is not None:
            stats['accumulated_rows'] = len(self._accumulated_df)
        
        # Top nodes/edges by frequency
        if self._node_counts:
            sorted_nodes = sorted(self._node_counts.items(), key=lambda x: x[1], reverse=True)
            stats['top_nodes'] = sorted_nodes[:10]
        
        if self._edge_counts:
            sorted_edges = sorted(self._edge_counts.items(), key=lambda x: x[1], reverse=True)  
            stats['top_edges'] = sorted_edges[:10]
        
        return stats
    
    def reset(self):
        """Reset processor state for new streaming session."""
        self._accumulated_df = None
        self._node_counts.clear()
        self._edge_counts.clear()
        self._processed_chunks = 0
        
        logger.info("ChunkedHypergraphProcessor state reset")