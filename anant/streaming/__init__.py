"""
Streaming Data Processing for Hypergraphs

This module provides comprehensive streaming capabilities for hypergraph processing,
leveraging the Performance Optimization Engine for efficient real-time analysis.
Supports incremental updates, streaming analytics, and memory-efficient operations.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Iterator
from dataclasses import dataclass
from queue import Queue
import threading
import time
from datetime import datetime
from abc import ABC, abstractmethod

from ..classes.hypergraph import Hypergraph
from ..optimization import PerformanceOptimizer, MemoryMonitor
from ..analysis.centrality import degree_centrality, s_centrality
from ..analysis.clustering import modularity_clustering
from ..analysis.temporal import TemporalSnapshot, TemporalHypergraph


@dataclass
class StreamingUpdate:
    """Represents a single streaming update to a hypergraph"""
    timestamp: Union[int, float, str]
    operation: str  # 'add_edge', 'remove_edge', 'add_node', 'remove_node', 'modify_edge'
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class StreamProcessor(ABC):
    """Abstract base class for streaming processors"""
    
    @abstractmethod
    def process_update(self, update: StreamingUpdate, hg: Hypergraph) -> Any:
        """Process a single streaming update"""
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get current processing results"""
        pass


class IncrementalCentralityProcessor(StreamProcessor):
    """
    Incrementally compute centrality measures as hypergraph changes.
    
    Maintains running centrality scores with efficient updates.
    """
    
    def __init__(self, measures: List[str] = ['degree', 's_centrality']):
        """
        Initialize incremental centrality processor.
        
        Args:
            measures: List of centrality measures to track
        """
        self.measures = measures
        self.centrality_scores = {}
        self.update_count = 0
        
        # Initialize tracking for each measure
        for measure in measures:
            self.centrality_scores[measure] = {}
    
    def process_update(self, update: StreamingUpdate, hg: Hypergraph) -> Dict[str, Any]:
        """Process streaming update and update centrality scores"""
        self.update_count += 1
        
        # For efficiency, recompute centralities periodically or on significant changes
        should_recompute = (
            self.update_count % 10 == 0 or  # Every 10 updates
            update.operation in ['add_edge', 'remove_edge']  # Structural changes
        )
        
        if should_recompute:
            self._recompute_centralities(hg)
        
        return {
            'timestamp': update.timestamp,
            'centrality_scores': self.centrality_scores.copy(),
            'update_count': self.update_count
        }
    
    def _recompute_centralities(self, hg: Hypergraph):
        """Recompute all centrality measures"""
        if 'degree' in self.measures:
            degree_cents = degree_centrality(hg, normalized=True)
            self.centrality_scores['degree'] = degree_cents['nodes']
        
        if 's_centrality' in self.measures:
            s_cents = s_centrality(hg, s=1, normalized=True)
            self.centrality_scores['s_centrality'] = s_cents
    
    def get_results(self) -> Dict[str, Any]:
        """Get current centrality results"""
        return {
            'centrality_scores': self.centrality_scores,
            'update_count': self.update_count,
            'measures': self.measures
        }


class StreamingClusteringProcessor(StreamProcessor):
    """
    Incrementally update community structure as hypergraph evolves.
    
    Maintains community assignments with periodic recomputation.
    """
    
    def __init__(self, recompute_interval: int = 20):
        """
        Initialize streaming clustering processor.
        
        Args:
            recompute_interval: How often to recompute communities
        """
        self.communities = {}
        self.recompute_interval = recompute_interval
        self.update_count = 0
        self.last_modularity = 0.0
    
    def process_update(self, update: StreamingUpdate, hg: Hypergraph) -> Dict[str, Any]:
        """Process streaming update and update community structure"""
        self.update_count += 1
        
        # Recompute communities periodically
        if self.update_count % self.recompute_interval == 0:
            self._recompute_communities(hg)
        
        return {
            'timestamp': update.timestamp,
            'communities': self.communities.copy(),
            'modularity': self.last_modularity,
            'update_count': self.update_count
        }
    
    def _recompute_communities(self, hg: Hypergraph):
        """Recompute community structure"""
        try:
            self.communities = modularity_clustering(hg)
            
            # Calculate current modularity
            from ..analysis.clustering import community_quality_metrics
            quality = community_quality_metrics(hg, self.communities)
            self.last_modularity = quality['modularity']
            
        except Exception:
            # Keep previous communities if computation fails
            pass
    
    def get_results(self) -> Dict[str, Any]:
        """Get current clustering results"""
        return {
            'communities': self.communities,
            'modularity': self.last_modularity,
            'update_count': self.update_count
        }


class StreamingHypergraph:
    """
    Hypergraph with streaming capabilities and real-time processing.
    
    Integrates with Performance Optimization Engine for efficient operations.
    """
    
    def __init__(self, 
                 initial_hg: Optional[Hypergraph] = None,
                 enable_optimization: bool = True,
                 buffer_size: int = 1000):
        """
        Initialize streaming hypergraph.
        
        Args:
            initial_hg: Initial hypergraph state
            enable_optimization: Whether to use performance optimization
            buffer_size: Size of update buffer
        """
        self.hypergraph = initial_hg or Hypergraph(pl.DataFrame({"edges": [], "nodes": []}))
        self.update_buffer = Queue(maxsize=buffer_size)
        self.processors = []
        
        # Performance optimization
        if enable_optimization:
            from ..optimization import OptimizationConfig
            config = OptimizationConfig()
            self.optimizer = PerformanceOptimizer(config)
            self.memory_monitor = MemoryMonitor()
        else:
            self.optimizer = None
            self.memory_monitor = None
        
        # Streaming state
        self.processing_thread = None
        self.is_processing = False
        self.processed_updates = 0
        self.processing_stats = {
            'total_updates': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    def add_processor(self, processor: StreamProcessor):
        """Add a streaming processor"""
        self.processors.append(processor)
    
    def start_processing(self, process_interval: float = 0.1):
        """Start background processing of updates"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(process_interval,),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop background processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self, interval: float):
        """Main processing loop for streaming updates"""
        while self.is_processing:
            try:
                # Process pending updates
                updates_processed = 0
                start_time = time.time()
                
                while not self.update_buffer.empty() and updates_processed < 100:
                    update = self.update_buffer.get_nowait()
                    self._apply_update(update)
                    updates_processed += 1
                
                if updates_processed > 0:
                    processing_time = time.time() - start_time
                    self.processing_stats['total_updates'] += updates_processed
                    self.processing_stats['processing_time'] += processing_time
                
                time.sleep(interval)
                
            except Exception as e:
                self.processing_stats['errors'] += 1
                print(f"Error in processing loop: {e}")
    
    def _apply_update(self, update: StreamingUpdate):
        """Apply a single update to the hypergraph"""
        try:
            # Apply the update to the hypergraph
            if update.operation == 'add_edge':
                self._add_edge_from_update(update)
            elif update.operation == 'remove_edge':
                self._remove_edge_from_update(update)
            elif update.operation == 'modify_edge':
                self._modify_edge_from_update(update)
            
            # Process with all registered processors
            for processor in self.processors:
                processor.process_update(update, self.hypergraph)
            
            self.processed_updates += 1
            
        except Exception as e:
            print(f"Error applying update: {e}")
            self.processing_stats['errors'] += 1
    
    def _add_edge_from_update(self, update: StreamingUpdate):
        """Add edge from streaming update"""
        edge_id = update.data.get('edge_id')
        nodes = update.data.get('nodes', [])
        weight = update.data.get('weight', 1.0)
        
        if edge_id and nodes:
            # Create new incidence data with proper schema
            new_incidences = [
                {"edges": edge_id, "nodes": node, "weight": weight}
                for node in nodes
            ]
            
            # Add to existing hypergraph
            new_df = pl.DataFrame(new_incidences)
            
            # Get the current timestamp
            current_time = datetime.now()
            
            # Ensure proper data types match existing data and add created_at column
            existing_edges_dtype = self.hypergraph._incidence_store.data["edges"].dtype
            existing_nodes_dtype = self.hypergraph._incidence_store.data["nodes"].dtype
            
            # Handle categorical types properly by first converting to string, then casting
            if existing_edges_dtype == pl.Categorical:
                # For categorical, we need to ensure the new values can be added to the category
                new_df = new_df.with_columns([
                    pl.col("edges").cast(pl.Utf8),
                    pl.col("nodes").cast(pl.Utf8), 
                    pl.col("weight").cast(pl.Float64),
                    pl.lit(current_time).alias("created_at")
                ])
                # Convert existing data to string temporarily for concatenation
                existing_df = self.hypergraph._incidence_store.data.with_columns([
                    pl.col("edges").cast(pl.Utf8),
                    pl.col("nodes").cast(pl.Utf8)
                ])
                combined_df = pl.concat([existing_df, new_df])
                # Convert back to categorical after concatenation
                combined_df = combined_df.with_columns([
                    pl.col("edges").cast(pl.Categorical),
                    pl.col("nodes").cast(pl.Categorical)
                ])
            else:
                # For non-categorical types, use direct casting
                new_df = new_df.with_columns([
                    pl.col("edges").cast(existing_edges_dtype),
                    pl.col("nodes").cast(existing_nodes_dtype), 
                    pl.col("weight").cast(pl.Float64),
                    pl.lit(current_time).alias("created_at")
                ])
                combined_df = pl.concat([self.hypergraph._incidence_store.data, new_df])
            
            # Update hypergraph
            self.hypergraph = Hypergraph(combined_df)
    
    def _remove_edge_from_update(self, update: StreamingUpdate):
        """Remove edge from streaming update"""
        edge_id = update.data.get('edge_id')
        
        if edge_id:
            # Filter out the edge
            filtered_df = self.hypergraph._incidence_store.data.filter(
                pl.col("edges") != edge_id
            )
            
            # Update hypergraph
            self.hypergraph = Hypergraph(filtered_df)
    
    def _modify_edge_from_update(self, update: StreamingUpdate):
        """Modify edge from streaming update"""
        edge_id = update.data.get('edge_id')
        new_nodes = update.data.get('nodes', [])
        weight = update.data.get('weight', 1.0)
        
        if edge_id and new_nodes:
            # Remove old edge and add new one
            self._remove_edge_from_update(update)
            add_update = StreamingUpdate(
                timestamp=update.timestamp,
                operation='add_edge',
                data={'edge_id': edge_id, 'nodes': new_nodes, 'weight': weight}
            )
            self._add_edge_from_update(add_update)
    
    def add_update(self, update: StreamingUpdate) -> bool:
        """
        Add streaming update to processing queue.
        
        Returns:
            True if update was added, False if buffer is full
        """
        try:
            self.update_buffer.put_nowait(update)
            return True
        except:
            return False
    
    def add_edge_update(self, 
                       timestamp: Union[int, float, str],
                       edge_id: str, 
                       nodes: List[str],
                       weight: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience method to add edge update"""
        update = StreamingUpdate(
            timestamp=timestamp,
            operation='add_edge',
            data={'edge_id': edge_id, 'nodes': nodes, 'weight': weight},
            metadata=metadata
        )
        return self.add_update(update)
    
    def remove_edge_update(self,
                          timestamp: Union[int, float, str], 
                          edge_id: str,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience method to remove edge update"""
        update = StreamingUpdate(
            timestamp=timestamp,
            operation='remove_edge',
            data={'edge_id': edge_id},
            metadata=metadata
        )
        return self.add_update(update)
    
    def get_processor_results(self) -> Dict[str, Any]:
        """Get results from all processors"""
        results = {}
        for i, processor in enumerate(self.processors):
            results[f'processor_{i}'] = processor.get_results()
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming processing statistics"""
        stats = self.processing_stats.copy()
        stats.update({
            'processed_updates': self.processed_updates,
            'buffer_size': self.update_buffer.qsize(),
            'is_processing': self.is_processing,
            'num_processors': len(self.processors)
        })
        
        if self.memory_monitor:
            stats['memory_usage_mb'] = self.memory_monitor.get_usage_mb()
            stats['memory_delta_mb'] = self.memory_monitor.get_delta_mb()
        
        return stats
    
    @property
    def current_hypergraph(self) -> Hypergraph:
        """Get current hypergraph state"""
        return self.hypergraph


class StreamingAnalytics:
    """
    Real-time analytics engine for streaming hypergraphs.
    
    Provides continuous analysis capabilities with configurable metrics.
    """
    
    def __init__(self, streaming_hg: StreamingHypergraph):
        """Initialize streaming analytics engine"""
        self.streaming_hg = streaming_hg
        self.analytics_history = []
        self.enabled_analytics = set()
        
        # Available analytics
        self.available_analytics = {
            'size_metrics': self._compute_size_metrics,
            'density_metrics': self._compute_density_metrics,
            'centrality_stats': self._compute_centrality_stats,
            'community_stats': self._compute_community_stats
        }
    
    def enable_analytics(self, analytics: List[str]):
        """Enable specific analytics computations"""
        for analytic in analytics:
            if analytic in self.available_analytics:
                self.enabled_analytics.add(analytic)
    
    def compute_analytics(self, timestamp: Union[int, float, str]) -> Dict[str, Any]:
        """Compute enabled analytics for current hypergraph state"""
        hg = self.streaming_hg.current_hypergraph
        results: Dict[str, Any] = {'timestamp': timestamp}
        
        for analytic in self.enabled_analytics:
            if analytic in self.available_analytics:
                try:
                    results[analytic] = self.available_analytics[analytic](hg)
                except Exception as e:
                    results[analytic] = {'error': str(e)}
        
        self.analytics_history.append(results)
        return results
    
    def _compute_size_metrics(self, hg: Hypergraph) -> Dict[str, float]:
        """Compute basic size metrics"""
        return {
            'num_nodes': float(hg.num_nodes),
            'num_edges': float(hg.num_edges),
            'num_incidences': float(hg.num_incidences)
        }
    
    def _compute_density_metrics(self, hg: Hypergraph) -> Dict[str, float]:
        """Compute density-related metrics"""
        if hg.num_edges == 0 or hg.num_nodes == 0:
            return {'density': 0.0, 'avg_edge_size': 0.0, 'avg_node_degree': 0.0}
        
        return {
            'density': float(hg.num_incidences / (hg.num_nodes * hg.num_edges)),
            'avg_edge_size': float(hg.num_incidences / hg.num_edges),
            'avg_node_degree': float(hg.num_incidences / hg.num_nodes)
        }
    
    def _compute_centrality_stats(self, hg: Hypergraph) -> Dict[str, float]:
        """Compute centrality statistics"""
        if hg.num_nodes == 0:
            return {'max_degree': 0.0, 'avg_degree': 0.0, 'degree_std': 0.0}
        
        try:
            degree_cents = degree_centrality(hg, normalized=True)['nodes']
            degrees = list(degree_cents.values())
            
            return {
                'max_degree': float(max(degrees)),
                'avg_degree': float(np.mean(degrees)),
                'degree_std': float(np.std(degrees))
            }
        except Exception:
            return {'max_degree': 0.0, 'avg_degree': 0.0, 'degree_std': 0.0}
    
    def _compute_community_stats(self, hg: Hypergraph) -> Dict[str, float]:
        """Compute community structure statistics"""
        if hg.num_nodes < 2:
            return {'num_communities': 0.0, 'modularity': 0.0}
        
        try:
            communities = modularity_clustering(hg)
            from ..analysis.clustering import community_quality_metrics
            quality = community_quality_metrics(hg, communities)
            
            return {
                'num_communities': float(quality['n_communities']),
                'modularity': float(quality['modularity'])
            }
        except Exception:
            return {'num_communities': 0.0, 'modularity': 0.0}
    
    def get_analytics_history(self) -> List[Dict[str, Any]]:
        """Get complete analytics history"""
        return self.analytics_history
    
    def get_recent_analytics(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent analytics results"""
        return self.analytics_history[-n:]


class StreamingDataIngestion:
    """
    Real-time data ingestion for streaming hypergraphs.
    
    Supports various data sources and formats with buffering and error handling.
    """
    
    def __init__(self, 
                 streaming_hg: StreamingHypergraph,
                 buffer_size: int = 10000,
                 batch_size: int = 100):
        """
        Initialize streaming data ingestion.
        
        Args:
            streaming_hg: Target streaming hypergraph
            buffer_size: Size of ingestion buffer
            batch_size: Number of updates to process in each batch
        """
        self.streaming_hg = streaming_hg
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ingestion_buffer = Queue(maxsize=buffer_size)
        
        # Statistics
        self.ingestion_stats = {
            'total_ingested': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'ingestion_rate': 0.0
        }
        
        # Processing control
        self.is_ingesting = False
        self.ingestion_thread = None
    
    def start_ingestion(self, process_interval: float = 0.05):
        """Start background data ingestion processing"""
        if self.is_ingesting:
            return
        
        self.is_ingesting = True
        self.ingestion_thread = threading.Thread(
            target=self._ingestion_loop,
            args=(process_interval,),
            daemon=True
        )
        self.ingestion_thread.start()
    
    def stop_ingestion(self):
        """Stop background data ingestion"""
        self.is_ingesting = False
        if self.ingestion_thread:
            self.ingestion_thread.join()
    
    def _ingestion_loop(self, interval: float):
        """Main ingestion processing loop"""
        while self.is_ingesting:
            try:
                processed = 0
                start_time = time.time()
                
                # Process batch of updates
                while not self.ingestion_buffer.empty() and processed < self.batch_size:
                    raw_data = self.ingestion_buffer.get_nowait()
                    
                    # Convert raw data to streaming update
                    update = self._parse_raw_data(raw_data)
                    
                    if update:
                        success = self.streaming_hg.add_update(update)
                        if success:
                            self.ingestion_stats['successful_updates'] += 1
                        else:
                            self.ingestion_stats['failed_updates'] += 1
                    
                    processed += 1
                
                # Update statistics
                if processed > 0:
                    elapsed = time.time() - start_time
                    self.ingestion_stats['total_ingested'] += processed
                    self.ingestion_stats['ingestion_rate'] = processed / elapsed
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in ingestion loop: {e}")
                self.ingestion_stats['failed_updates'] += 1
    
    def _parse_raw_data(self, raw_data: Dict[str, Any]) -> Optional[StreamingUpdate]:
        """Parse raw data into StreamingUpdate"""
        try:
            return StreamingUpdate(
                timestamp=raw_data.get('timestamp', time.time()),
                operation=raw_data.get('operation', 'add_edge'),
                data=raw_data.get('data', {}),
                metadata=raw_data.get('metadata', {})
            )
        except Exception:
            return None
    
    def ingest_data(self, raw_data: Dict[str, Any]) -> bool:
        """
        Ingest raw data for processing.
        
        Returns:
            True if data was buffered, False if buffer is full
        """
        try:
            self.ingestion_buffer.put_nowait(raw_data)
            return True
        except:
            return False
    
    def ingest_edge_data(self, 
                        timestamp: Union[int, float, str],
                        edge_id: str,
                        nodes: List[str],
                        operation: str = 'add_edge') -> bool:
        """Convenience method for ingesting edge data"""
        raw_data = {
            'timestamp': timestamp,
            'operation': operation,
            'data': {'edge_id': edge_id, 'nodes': nodes}
        }
        return self.ingest_data(raw_data)
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        stats = self.ingestion_stats.copy()
        stats.update({
            'buffer_size': self.ingestion_buffer.qsize(),
            'is_ingesting': self.is_ingesting
        })
        return stats


def stream_from_temporal_hypergraph(
    temporal_hg: TemporalHypergraph,
    streaming_hg: StreamingHypergraph,
    replay_speed: float = 1.0
) -> None:
    """
    Stream data from a temporal hypergraph to a streaming hypergraph.
    
    Useful for replaying historical data or testing streaming algorithms.
    
    Args:
        temporal_hg: Source temporal hypergraph
        streaming_hg: Target streaming hypergraph  
        replay_speed: Speed multiplier for replay (1.0 = real-time)
    """
    temporal_hg.sort_snapshots()
    
    if len(temporal_hg.snapshots) < 2:
        return
    
    # Calculate time differences between snapshots
    prev_snapshot = temporal_hg.snapshots[0]
    
    for current_snapshot in temporal_hg.snapshots[1:]:
        # Calculate edges to add/remove between snapshots
        prev_edges = set(prev_snapshot.hypergraph.edges)
        curr_edges = set(current_snapshot.hypergraph.edges)
        
        # Add new edges
        new_edges = curr_edges - prev_edges
        for edge_id in new_edges:
            nodes = (
                current_snapshot.hypergraph._incidence_store.data
                .filter(pl.col("edges") == edge_id)
                .select("nodes")
                .to_series()
                .to_list()
            )
            
            streaming_hg.add_edge_update(
                timestamp=current_snapshot.timestamp,
                edge_id=edge_id,
                nodes=nodes
            )
        
        # Remove old edges  
        removed_edges = prev_edges - curr_edges
        for edge_id in removed_edges:
            streaming_hg.remove_edge_update(
                timestamp=current_snapshot.timestamp,
                edge_id=edge_id
            )
        
        prev_snapshot = current_snapshot