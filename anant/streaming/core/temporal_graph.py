"""
Temporal Graph Framework
=======================

Time-aware graph data structure with versioning, snapshots, and temporal queries.
Supports time-travel analysis, temporal analytics, and historical data exploration.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterator
import bisect

import polars as pl
import numpy as np

from ...classes.hypergraph import Hypergraph
from ...classes.incidence_store import IncidenceStore

logger = logging.getLogger(__name__)


class TemporalScope(Enum):
    """Temporal analysis scope."""
    SNAPSHOT = "snapshot"          # Single point in time
    INTERVAL = "interval"          # Time range
    SLIDING_WINDOW = "sliding_window"  # Moving time window
    ALL_TIME = "all_time"         # Entire history


class VersioningStrategy(Enum):
    """Graph versioning strategies."""
    FULL_SNAPSHOT = "full_snapshot"    # Store complete graph at each version
    DELTA_ONLY = "delta_only"          # Store only changes between versions
    HYBRID = "hybrid"                  # Combine snapshots with deltas
    COW = "copy_on_write"              # Copy-on-write optimization


@dataclass
class TimeRange:
    """Represents a time range for temporal queries."""
    start: datetime
    end: datetime
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start time must be before end time")
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this range."""
        return self.start <= timestamp <= self.end
    
    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return not (self.end < other.start or self.start > other.end)
    
    def duration(self) -> timedelta:
        """Get duration of this time range."""
        return self.end - self.start
    
    def split(self, num_parts: int) -> List["TimeRange"]:
        """Split time range into equal parts."""
        if num_parts <= 0:
            raise ValueError("Number of parts must be positive")
        
        duration = self.duration()
        part_duration = duration / num_parts
        
        ranges = []
        current_start = self.start
        
        for i in range(num_parts):
            current_end = current_start + part_duration
            if i == num_parts - 1:  # Last part should end exactly at end time
                current_end = self.end
            
            ranges.append(TimeRange(current_start, current_end))
            current_start = current_end
        
        return ranges


@dataclass
class GraphSnapshot:
    """Represents a graph state at a specific point in time."""
    timestamp: datetime
    graph: Hypergraph
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    
    def __post_init__(self):
        if self.size_bytes is None:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Estimate memory size of this snapshot."""
        # Rough estimation based on data structure sizes
        incidence_size = len(self.graph.incidences.data) * 100  # Rough bytes per row
        metadata_size = len(str(self.metadata).encode())
        return incidence_size + metadata_size
    
    def get_nodes_at_time(self) -> Set[str]:
        """Get all nodes that existed at this timestamp."""
        return set(self.graph.nodes)
    
    def get_edges_at_time(self) -> Set[str]:
        """Get all edges that existed at this timestamp."""
        return set(self.graph.edges)


@dataclass
class GraphDelta:
    """Represents changes between two graph states."""
    from_version: int
    to_version: int
    timestamp: datetime
    added_nodes: Set[str] = field(default_factory=set)
    removed_nodes: Set[str] = field(default_factory=set)
    added_edges: Dict[str, List[str]] = field(default_factory=dict)  # edge_id -> nodes
    removed_edges: Set[str] = field(default_factory=set)
    modified_edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # edge_id -> changes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if this delta contains no changes."""
        return (not self.added_nodes and not self.removed_nodes and 
                not self.added_edges and not self.removed_edges and 
                not self.modified_edges)
    
    def get_affected_nodes(self) -> Set[str]:
        """Get all nodes affected by this delta."""
        affected = set()
        affected.update(self.added_nodes)
        affected.update(self.removed_nodes)
        
        # Add nodes from edge changes
        for nodes in self.added_edges.values():
            affected.update(nodes)
        
        return affected
    
    def invert(self) -> "GraphDelta":
        """Create inverse delta to undo these changes."""
        return GraphDelta(
            from_version=self.to_version,
            to_version=self.from_version,
            timestamp=self.timestamp,
            added_nodes=self.removed_nodes.copy(),
            removed_nodes=self.added_nodes.copy(),
            added_edges={},  # Edge modifications would need more complex inversion
            removed_edges=set(self.added_edges.keys()),
            modified_edges={},  # Complex to invert
            metadata={"inverted": True, "original_delta": str(self.from_version)}
        )


class TemporalIndex:
    """Efficient index for temporal graph queries."""
    
    def __init__(self):
        self.timestamp_index: List[Tuple[datetime, int]] = []  # (timestamp, version)
        self.node_timeline: Dict[str, List[Tuple[datetime, str]]] = {}  # node -> [(time, action)]
        self.edge_timeline: Dict[str, List[Tuple[datetime, str]]] = {}  # edge -> [(time, action)]
        
    def add_snapshot(self, snapshot: GraphSnapshot):
        """Add snapshot to temporal index."""
        # Add to timestamp index
        bisect.insort(self.timestamp_index, (snapshot.timestamp, snapshot.version))
        
        # Update node timeline
        for node in snapshot.graph.nodes:
            if node not in self.node_timeline:
                self.node_timeline[node] = []
            self.node_timeline[node].append((snapshot.timestamp, "exists"))
        
        # Update edge timeline
        for edge in snapshot.graph.edges:
            if edge not in self.edge_timeline:
                self.edge_timeline[edge] = []
            self.edge_timeline[edge].append((snapshot.timestamp, "exists"))
    
    def find_version_at_time(self, timestamp: datetime) -> Optional[int]:
        """Find the latest version at or before the given timestamp."""
        # Binary search for the latest version <= timestamp
        index = bisect.bisect_right(self.timestamp_index, (timestamp, float('inf'))) - 1
        
        if index >= 0:
            return self.timestamp_index[index][1]
        return None
    
    def find_versions_in_range(self, time_range: TimeRange) -> List[int]:
        """Find all versions within the given time range."""
        start_index = bisect.bisect_left(self.timestamp_index, (time_range.start, 0))
        end_index = bisect.bisect_right(self.timestamp_index, (time_range.end, float('inf')))
        
        return [version for _, version in self.timestamp_index[start_index:end_index]]
    
    def get_node_existence_periods(self, node_id: str) -> List[TimeRange]:
        """Get time periods when a node existed."""
        if node_id not in self.node_timeline:
            return []
        
        # Simple implementation - assumes continuous existence
        timeline = self.node_timeline[node_id]
        if not timeline:
            return []
        
        # Find continuous periods of existence
        periods = []
        start_time = timeline[0][0]
        
        # For now, assume node exists from first appearance to latest timestamp
        if len(self.timestamp_index) > 0:
            end_time = self.timestamp_index[-1][0]
            periods.append(TimeRange(start_time, end_time))
        
        return periods


class TemporalGraph:
    """
    Time-aware graph that maintains historical versions and supports temporal queries.
    """
    
    def __init__(self, 
                 versioning_strategy: VersioningStrategy = VersioningStrategy.HYBRID,
                 max_snapshots: int = 1000,
                 snapshot_interval: timedelta = timedelta(hours=1)):
        """
        Initialize temporal graph.
        
        Args:
            versioning_strategy: How to store graph versions
            max_snapshots: Maximum number of snapshots to keep
            snapshot_interval: Minimum time between automatic snapshots
        """
        self.versioning_strategy = versioning_strategy
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        
        # Storage
        self.snapshots: List[GraphSnapshot] = []
        self.deltas: List[GraphDelta] = []
        self.current_version = 0
        self.current_graph: Optional[Hypergraph] = None
        
        # Indexing
        self.temporal_index = TemporalIndex()
        
        # State tracking
        self.last_snapshot_time: Optional[datetime] = None
        self.total_size_bytes = 0
        
        logger.info(f"Initialized TemporalGraph with {versioning_strategy.value} versioning")
    
    def add_snapshot(self, graph: Hypergraph, timestamp: Optional[datetime] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new graph snapshot.
        
        Args:
            graph: Graph to snapshot
            timestamp: Snapshot timestamp (defaults to now)
            metadata: Additional metadata
            
        Returns:
            Version number of the snapshot
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metadata = metadata or {}
        self.current_version += 1
        
        # Create snapshot
        snapshot = GraphSnapshot(
            timestamp=timestamp,
            graph=graph,
            version=self.current_version,
            metadata=metadata
        )
        
        # Add to storage
        self.snapshots.append(snapshot)
        self.temporal_index.add_snapshot(snapshot)
        self.current_graph = graph
        self.last_snapshot_time = timestamp
        self.total_size_bytes += snapshot.size_bytes or 0
        
        # Manage storage limits
        self._manage_storage()
        
        logger.debug(f"Added snapshot version {self.current_version} at {timestamp}")
        return self.current_version
    
    def get_snapshot_at_time(self, timestamp: datetime) -> Optional[GraphSnapshot]:
        """Get the graph snapshot at or before the specified time."""
        version = self.temporal_index.find_version_at_time(timestamp)
        if version is None:
            return None
        
        return self.get_snapshot_by_version(version)
    
    def get_snapshot_by_version(self, version: int) -> Optional[GraphSnapshot]:
        """Get snapshot by version number."""
        for snapshot in self.snapshots:
            if snapshot.version == version:
                return snapshot
        return None
    
    def get_snapshots_in_range(self, time_range: TimeRange) -> List[GraphSnapshot]:
        """Get all snapshots within the specified time range."""
        versions = self.temporal_index.find_versions_in_range(time_range)
        snapshots = []
        
        for version in versions:
            snapshot = self.get_snapshot_by_version(version)
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots
    
    def query_temporal(self, 
                      scope: TemporalScope,
                      timestamp: Optional[datetime] = None,
                      time_range: Optional[TimeRange] = None,
                      window_size: Optional[timedelta] = None) -> List[GraphSnapshot]:
        """
        Perform temporal query on the graph.
        
        Args:
            scope: Type of temporal query
            timestamp: Specific timestamp (for SNAPSHOT scope)
            time_range: Time range (for INTERVAL scope)
            window_size: Window size (for SLIDING_WINDOW scope)
            
        Returns:
            List of relevant snapshots
        """
        if scope == TemporalScope.SNAPSHOT:
            if timestamp is None:
                raise ValueError("Timestamp required for SNAPSHOT scope")
            snapshot = self.get_snapshot_at_time(timestamp)
            return [snapshot] if snapshot else []
        
        elif scope == TemporalScope.INTERVAL:
            if time_range is None:
                raise ValueError("Time range required for INTERVAL scope")
            return self.get_snapshots_in_range(time_range)
        
        elif scope == TemporalScope.SLIDING_WINDOW:
            if timestamp is None or window_size is None:
                raise ValueError("Timestamp and window size required for SLIDING_WINDOW scope")
            
            window_start = timestamp - window_size
            window_range = TimeRange(window_start, timestamp)
            return self.get_snapshots_in_range(window_range)
        
        elif scope == TemporalScope.ALL_TIME:
            return self.snapshots.copy()
        
        else:
            raise ValueError(f"Unsupported temporal scope: {scope}")
    
    def compute_node_evolution(self, node_id: str) -> Dict[str, Any]:
        """Compute how a node evolved over time."""
        evolution = {
            "node_id": node_id,
            "existence_periods": self.temporal_index.get_node_existence_periods(node_id),
            "degree_evolution": [],
            "centrality_evolution": [],
            "first_appearance": None,
            "last_appearance": None
        }
        
        # Analyze node across snapshots
        for snapshot in self.snapshots:
            if node_id in snapshot.graph.nodes:
                if evolution["first_appearance"] is None:
                    evolution["first_appearance"] = snapshot.timestamp
                evolution["last_appearance"] = snapshot.timestamp
                
                # Calculate degree at this snapshot
                node_edges = snapshot.graph.incidences.data.filter(
                    pl.col("nodes") == node_id
                ).select("edges").unique()
                degree = len(node_edges)
                
                evolution["degree_evolution"].append({
                    "timestamp": snapshot.timestamp,
                    "degree": degree
                })
        
        return evolution
    
    def compute_graph_evolution_metrics(self, time_range: Optional[TimeRange] = None) -> Dict[str, Any]:
        """Compute evolution metrics for the entire graph."""
        snapshots = self.snapshots
        if time_range:
            snapshots = self.get_snapshots_in_range(time_range)
        
        if not snapshots:
            return {"error": "No snapshots in specified range"}
        
        metrics = {
            "time_range": {
                "start": snapshots[0].timestamp.isoformat(),
                "end": snapshots[-1].timestamp.isoformat(),
                "duration_hours": (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 3600
            },
            "size_evolution": [],
            "density_evolution": [],
            "growth_rate": {},
            "volatility": {}
        }
        
        prev_snapshot = None
        node_changes = []
        edge_changes = []
        
        for snapshot in snapshots:
            # Size metrics
            metrics["size_evolution"].append({
                "timestamp": snapshot.timestamp.isoformat(),
                "num_nodes": snapshot.graph.num_nodes,
                "num_edges": snapshot.graph.num_edges,
                "num_incidences": snapshot.graph.num_incidences
            })
            
            # Density metrics
            if snapshot.graph.num_nodes > 0 and snapshot.graph.num_edges > 0:
                density = snapshot.graph.num_incidences / (snapshot.graph.num_nodes * snapshot.graph.num_edges)
            else:
                density = 0.0
            
            metrics["density_evolution"].append({
                "timestamp": snapshot.timestamp.isoformat(),
                "density": density
            })
            
            # Calculate changes from previous snapshot
            if prev_snapshot:
                node_change = snapshot.graph.num_nodes - prev_snapshot.graph.num_nodes
                edge_change = snapshot.graph.num_edges - prev_snapshot.graph.num_edges
                
                node_changes.append(node_change)
                edge_changes.append(edge_change)
            
            prev_snapshot = snapshot
        
        # Calculate growth rates and volatility
        if node_changes:
            metrics["growth_rate"]["avg_nodes_per_snapshot"] = np.mean(node_changes)
            metrics["growth_rate"]["avg_edges_per_snapshot"] = np.mean(edge_changes)
            
            metrics["volatility"]["node_change_std"] = np.std(node_changes)
            metrics["volatility"]["edge_change_std"] = np.std(edge_changes)
        
        return metrics
    
    def find_temporal_patterns(self, pattern_type: str = "periodic") -> List[Dict[str, Any]]:
        """Find temporal patterns in graph evolution."""
        patterns = []
        
        if pattern_type == "periodic":
            # Look for periodic changes in graph size
            if len(self.snapshots) < 3:
                return patterns
            
            # Extract time series of graph sizes
            timestamps = [s.timestamp for s in self.snapshots]
            node_counts = [s.graph.num_nodes for s in self.snapshots]
            edge_counts = [s.graph.num_edges for s in self.snapshots]
            
            # Simple pattern detection (could be enhanced with FFT, etc.)
            # Look for repeating patterns in differences
            node_diffs = np.diff(node_counts)
            edge_diffs = np.diff(edge_counts)
            
            # Find potential periods
            for period in range(2, min(10, len(node_diffs) // 2)):
                correlation = 0
                valid_comparisons = 0
                
                for i in range(len(node_diffs) - period):
                    if i + period < len(node_diffs):
                        correlation += abs(node_diffs[i] - node_diffs[i + period])
                        valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_diff = correlation / valid_comparisons
                    if avg_diff < 2:  # Threshold for pattern detection
                        patterns.append({
                            "type": "periodic_nodes",
                            "period": period,
                            "confidence": 1.0 / (avg_diff + 1),
                            "description": f"Periodic pattern in node count with period {period}"
                        })
        
        return patterns
    
    def create_temporal_view(self, 
                           time_range: TimeRange,
                           aggregation: str = "union") -> Hypergraph:
        """
        Create a temporal view of the graph over a time range.
        
        Args:
            time_range: Time range for the view
            aggregation: How to aggregate ('union', 'intersection', 'latest')
            
        Returns:
            Aggregated hypergraph for the time range
        """
        snapshots = self.get_snapshots_in_range(time_range)
        
        if not snapshots:
            # Return empty graph
            return Hypergraph(pl.DataFrame({"edges": [], "nodes": [], "weight": []}))
        
        if aggregation == "latest":
            return snapshots[-1].graph
        
        elif aggregation == "union":
            # Combine all edges and nodes from all snapshots
            all_incidences = []
            
            for snapshot in snapshots:
                snapshot_data = snapshot.graph.incidences.data
                # Add timestamp to track when each incidence was present
                timestamped_data = snapshot_data.with_columns([
                    pl.lit(snapshot.timestamp).alias("snapshot_timestamp")
                ])
                all_incidences.append(timestamped_data)
            
            if all_incidences:
                combined_data = pl.concat(all_incidences)
                # Remove duplicates (keep latest occurrence)
                deduplicated = combined_data.unique(subset=["edges", "nodes"], keep="last")
                return Hypergraph(deduplicated.drop("snapshot_timestamp"))
            else:
                return Hypergraph(pl.DataFrame({"edges": [], "nodes": [], "weight": []}))
        
        elif aggregation == "intersection":
            # Only include edges that appear in ALL snapshots
            if len(snapshots) == 1:
                return snapshots[0].graph
            
            # Find edges common to all snapshots
            edge_sets = [set(snapshot.graph.edges) for snapshot in snapshots]
            common_edges = set.intersection(*edge_sets)
            
            if not common_edges:
                return Hypergraph(pl.DataFrame({"edges": [], "nodes": [], "weight": []}))
            
            # Use the latest snapshot and filter to common edges
            latest_graph = snapshots[-1].graph
            filtered_data = latest_graph.incidences.data.filter(
                pl.col("edges").is_in(list(common_edges))
            )
            
            return Hypergraph(filtered_data)
        
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    def _manage_storage(self):
        """Manage storage to stay within limits."""
        # Remove old snapshots if we exceed the limit
        while len(self.snapshots) > self.max_snapshots:
            removed_snapshot = self.snapshots.pop(0)
            self.total_size_bytes -= removed_snapshot.size_bytes or 0
            logger.debug(f"Removed old snapshot version {removed_snapshot.version}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "num_snapshots": len(self.snapshots),
            "num_deltas": len(self.deltas),
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
            "current_version": self.current_version,
            "versioning_strategy": self.versioning_strategy.value,
            "oldest_snapshot": self.snapshots[0].timestamp.isoformat() if self.snapshots else None,
            "newest_snapshot": self.snapshots[-1].timestamp.isoformat() if self.snapshots else None
        }
    
    def export_temporal_data(self, format: str = "json") -> Dict[str, Any]:
        """Export temporal graph data for analysis or backup."""
        if format == "json":
            return {
                "metadata": {
                    "versioning_strategy": self.versioning_strategy.value,
                    "current_version": self.current_version,
                    "total_snapshots": len(self.snapshots),
                    "export_timestamp": datetime.now().isoformat()
                },
                "snapshots": [
                    {
                        "version": snapshot.version,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "num_nodes": snapshot.graph.num_nodes,
                        "num_edges": snapshot.graph.num_edges,
                        "metadata": snapshot.metadata
                    }
                    for snapshot in self.snapshots
                ],
                "storage_stats": self.get_storage_stats()
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Factory functions
def create_temporal_graph(strategy: str = "hybrid", **kwargs) -> TemporalGraph:
    """Create a temporal graph with specified strategy."""
    return TemporalGraph(
        versioning_strategy=VersioningStrategy(strategy),
        **kwargs
    )