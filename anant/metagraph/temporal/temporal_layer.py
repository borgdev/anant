"""
Temporal Layer for Metagraph - Phase 1
======================================

Manages temporal aspects of data, tracking changes over time and providing
temporal analytics for enterprise knowledge management.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Literal, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import orjson
import uuid

# Type aliases
TimeUnit = Literal["second", "minute", "hour", "day", "week", "month", "year"]
TemporalOperation = Literal["create", "update", "delete", "access", "transform"]
AggregationFunction = Literal["count", "sum", "avg", "min", "max", "first", "last"]
ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]


@dataclass
class TemporalEvent:
    """Represents a temporal event in the knowledge graph."""
    event_id: str
    entity_id: str
    operation: TemporalOperation
    timestamp: datetime
    details: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]


@dataclass
class TemporalPattern:
    """Represents a discovered temporal pattern."""
    pattern_id: str
    pattern_type: str
    entities: List[str]
    frequency: str
    confidence: float
    description: str
    metadata: Dict[str, Any]


class TemporalLayer:
    """
    Enterprise temporal layer for time-series analysis and temporal patterns.
    
    Tracks entity changes, access patterns, and temporal relationships
    with high-performance Polars+Parquet backend.
    """
    
    def __init__(self, 
                 storage_path: str = "./metagraph_temporal",
                 compression: ParquetCompression = "zstd",
                 retention_days: int = 365):
        """
        Initialize temporal layer.
        
        Args:
            storage_path: Path for storing temporal data
            compression: Parquet compression algorithm
            retention_days: Days to retain temporal data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression: ParquetCompression = compression
        self.retention_days = retention_days
        
        # File paths
        self.events_file = self.storage_path / "events.parquet"
        self.patterns_file = self.storage_path / "patterns.parquet"
        self.snapshots_file = self.storage_path / "snapshots.parquet"
        self.lineage_file = self.storage_path / "lineage.parquet"
        
        # Initialize data structures
        self._init_events()
        self._init_patterns()
        self._init_snapshots()
        self._init_lineage()
        
        # Temporal caches
        self._pattern_cache: Dict[str, List[TemporalPattern]] = {}
        self._timeline_cache: Dict[str, List[TemporalEvent]] = {}
    
    def _init_events(self):
        """Initialize events DataFrame."""
        if self.events_file.exists():
            self.events_df = pl.read_parquet(self.events_file)
        else:
            self.events_df = pl.DataFrame({
                "event_id": pl.Series([], dtype=pl.Utf8),
                "entity_id": pl.Series([], dtype=pl.Utf8),
                "operation": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "details": pl.Series([], dtype=pl.Utf8),  # JSON
                "user_id": pl.Series([], dtype=pl.Utf8),
                "session_id": pl.Series([], dtype=pl.Utf8),
                "source_system": pl.Series([], dtype=pl.Utf8),
                "batch_id": pl.Series([], dtype=pl.Utf8),
                "correlation_id": pl.Series([], dtype=pl.Utf8),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                "tags": pl.Series([], dtype=pl.List(pl.Utf8)),
                "duration_ms": pl.Series([], dtype=pl.Int64),
                "status": pl.Series([], dtype=pl.Utf8)
            })
    
    def _init_patterns(self):
        """Initialize patterns DataFrame."""
        if self.patterns_file.exists():
            self.patterns_df = pl.read_parquet(self.patterns_file)
        else:
            self.patterns_df = pl.DataFrame({
                "pattern_id": pl.Series([], dtype=pl.Utf8),
                "pattern_type": pl.Series([], dtype=pl.Utf8),
                "entities": pl.Series([], dtype=pl.List(pl.Utf8)),
                "frequency": pl.Series([], dtype=pl.Utf8),
                "confidence": pl.Series([], dtype=pl.Float64),
                "description": pl.Series([], dtype=pl.Utf8),
                "first_observed": pl.Series([], dtype=pl.Datetime),
                "last_observed": pl.Series([], dtype=pl.Datetime),
                "occurrence_count": pl.Series([], dtype=pl.Int64),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                "status": pl.Series([], dtype=pl.Utf8),
                "created_at": pl.Series([], dtype=pl.Datetime),
                "updated_at": pl.Series([], dtype=pl.Datetime)
            })
    
    def _init_snapshots(self):
        """Initialize snapshots DataFrame."""
        if self.snapshots_file.exists():
            self.snapshots_df = pl.read_parquet(self.snapshots_file)
        else:
            self.snapshots_df = pl.DataFrame({
                "snapshot_id": pl.Series([], dtype=pl.Utf8),
                "entity_id": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "snapshot_type": pl.Series([], dtype=pl.Utf8),
                "state": pl.Series([], dtype=pl.Utf8),  # JSON
                "version": pl.Series([], dtype=pl.Int64),
                "change_summary": pl.Series([], dtype=pl.Utf8),  # JSON
                "created_by": pl.Series([], dtype=pl.Utf8),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                "size_bytes": pl.Series([], dtype=pl.Int64),
                "compression_ratio": pl.Series([], dtype=pl.Float64)
            })
    
    def _init_lineage(self):
        """Initialize lineage DataFrame."""
        if self.lineage_file.exists():
            self.lineage_df = pl.read_parquet(self.lineage_file)
        else:
            self.lineage_df = pl.DataFrame({
                "lineage_id": pl.Series([], dtype=pl.Utf8),
                "source_entity_id": pl.Series([], dtype=pl.Utf8),
                "target_entity_id": pl.Series([], dtype=pl.Utf8),
                "operation": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "transformation_details": pl.Series([], dtype=pl.Utf8),  # JSON
                "confidence": pl.Series([], dtype=pl.Float64),
                "lineage_type": pl.Series([], dtype=pl.Utf8),
                "processing_system": pl.Series([], dtype=pl.Utf8),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                "validated": pl.Series([], dtype=pl.Boolean),
                "created_at": pl.Series([], dtype=pl.Datetime)
            })
    
    def record_event(self,
                    entity_id: str,
                    operation: TemporalOperation,
                    details: Optional[Dict[str, Any]] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    source_system: str = "unknown",
                    batch_id: Optional[str] = None,
                    correlation_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    tags: Optional[List[str]] = None,
                    duration_ms: Optional[int] = None) -> str:
        """
        Record a temporal event.
        
        Args:
            entity_id: Entity identifier
            operation: Type of operation performed
            details: Operation details
            user_id: User who performed the operation
            session_id: Session identifier
            source_system: System that generated the event
            batch_id: Batch processing identifier
            correlation_id: Correlation identifier for related events
            metadata: Additional metadata
            tags: Classification tags
            duration_ms: Operation duration in milliseconds
            
        Returns:
            event_id: Unique identifier for the event
        """
        event_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_event = pl.DataFrame({
            "event_id": [event_id],
            "entity_id": [entity_id],
            "operation": [operation],
            "timestamp": [now],
            "details": [orjson.dumps(details or {}).decode()],
            "user_id": [user_id],
            "session_id": [session_id],
            "source_system": [source_system],
            "batch_id": [batch_id],
            "correlation_id": [correlation_id],
            "metadata": [orjson.dumps(metadata or {}).decode()],
            "tags": [tags or []],
            "duration_ms": [duration_ms],
            "status": ["completed"]
        })
        
        self.events_df = pl.concat([self.events_df, new_event])
        
        # Clear relevant caches
        if entity_id in self._timeline_cache:
            del self._timeline_cache[entity_id]
        
        return event_id
    
    def get_entity_timeline(self,
                           entity_id: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           operations: Optional[List[TemporalOperation]] = None,
                           limit: Optional[int] = None) -> List[TemporalEvent]:
        """
        Get temporal timeline for an entity.
        
        Args:
            entity_id: Entity identifier
            start_time: Start of time range
            end_time: End of time range
            operations: Filter by operation types
            limit: Maximum number of events
            
        Returns:
            List of temporal events
        """
        cache_key = f"{entity_id}_{start_time}_{end_time}_{operations}_{limit}"
        if cache_key in self._timeline_cache:
            return self._timeline_cache[cache_key]
        
        query = self.events_df.filter(pl.col("entity_id") == entity_id)
        
        if start_time:
            query = query.filter(pl.col("timestamp") >= start_time)
        
        if end_time:
            query = query.filter(pl.col("timestamp") <= end_time)
        
        if operations:
            query = query.filter(pl.col("operation").is_in(operations))
        
        query = query.sort("timestamp", descending=True)
        
        if limit:
            query = query.limit(limit)
        
        events = []
        for row in query.iter_rows(named=True):
            events.append(TemporalEvent(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                operation=row["operation"],
                timestamp=row["timestamp"],
                details=orjson.loads(row["details"]),
                user_id=row["user_id"],
                session_id=row["session_id"]
            ))
        
        self._timeline_cache[cache_key] = events
        return events
    
    def analyze_temporal_patterns(self,
                                 entity_ids: Optional[List[str]] = None,
                                 time_window: timedelta = timedelta(days=30),
                                 min_occurrences: int = 3,
                                 pattern_types: Optional[List[str]] = None) -> List[TemporalPattern]:
        """
        Analyze temporal patterns in entity events.
        
        Args:
            entity_ids: Entities to analyze (None for all)
            time_window: Time window for pattern detection
            min_occurrences: Minimum pattern occurrences
            pattern_types: Types of patterns to detect
            
        Returns:
            List of discovered temporal patterns
        """
        end_time = datetime.now()
        start_time = end_time - time_window
        
        query = self.events_df.filter(
            (pl.col("timestamp") >= start_time) &
            (pl.col("timestamp") <= end_time)
        )
        
        if entity_ids:
            query = query.filter(pl.col("entity_id").is_in(entity_ids))
        
        patterns = []
        
        # Pattern 1: Periodic access patterns
        if not pattern_types or "periodic_access" in pattern_types:
            patterns.extend(self._detect_periodic_patterns(query, min_occurrences))
        
        # Pattern 2: Burst patterns
        if not pattern_types or "burst" in pattern_types:
            patterns.extend(self._detect_burst_patterns(query, min_occurrences))
        
        # Pattern 3: Sequential patterns
        if not pattern_types or "sequential" in pattern_types:
            patterns.extend(self._detect_sequential_patterns(query, min_occurrences))
        
        return patterns
    
    def _detect_periodic_patterns(self, events_df: pl.DataFrame, min_occurrences: int) -> List[TemporalPattern]:
        """Detect periodic access patterns."""
        patterns = []
        
        # Group by entity and operation - simplified approach without list aggregation
        grouped = events_df.group_by(["entity_id", "operation"]).agg([
            pl.col("timestamp").count().alias("count"),
            pl.col("timestamp").min().alias("first_time"),
            pl.col("timestamp").max().alias("last_time")
        ]).filter(pl.col("count") >= min_occurrences)
        
        for row in grouped.iter_rows(named=True):
            # Get individual timestamps for this entity/operation combination
            entity_operation_events = events_df.filter(
                (pl.col("entity_id") == row["entity_id"]) & 
                (pl.col("operation") == row["operation"])
            ).sort("timestamp")
            
            timestamps = entity_operation_events["timestamp"].to_list()
            intervals = [
                (timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            
            if len(intervals) < 2:
                continue
            
            # Check for regular intervals
            avg_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            
            if std_interval / avg_interval < 0.3:  # Regular pattern
                pattern_id = str(uuid.uuid4())
                frequency = self._interval_to_frequency(avg_interval)
                
                patterns.append(TemporalPattern(
                    pattern_id=pattern_id,
                    pattern_type="periodic_access",
                    entities=[row["entity_id"]],
                    frequency=frequency,
                    confidence=float(1.0 - (std_interval / avg_interval)),
                    description=f"Regular {row['operation']} operations every {frequency}",
                    metadata={
                        "operation": row["operation"],
                        "avg_interval_seconds": avg_interval,
                        "occurrences": row["count"]
                    }
                ))
        
        return patterns
    
    def _detect_burst_patterns(self, events_df: pl.DataFrame, min_occurrences: int) -> List[TemporalPattern]:
        """Detect burst activity patterns."""
        patterns = []
        
        # Define burst as events clustered in short time windows
        burst_window = timedelta(minutes=5)
        
        # Group events by entity
        for entity_id in events_df["entity_id"].unique():
            entity_events = events_df.filter(pl.col("entity_id") == entity_id).sort("timestamp")
            
            if entity_events.height < min_occurrences:
                continue
            
            timestamps = entity_events["timestamp"].to_list()
            burst_groups = []
            current_burst = [timestamps[0]]
            
            for i in range(1, len(timestamps)):
                if timestamps[i] - timestamps[i-1] <= burst_window:
                    current_burst.append(timestamps[i])
                else:
                    if len(current_burst) >= min_occurrences:
                        burst_groups.append(current_burst)
                    current_burst = [timestamps[i]]
            
            if len(current_burst) >= min_occurrences:
                burst_groups.append(current_burst)
            
            for burst in burst_groups:
                pattern_id = str(uuid.uuid4())
                duration = (burst[-1] - burst[0]).total_seconds()
                rate = len(burst) / (duration / 60)  # events per minute
                
                patterns.append(TemporalPattern(
                    pattern_id=pattern_id,
                    pattern_type="burst",
                    entities=[entity_id],
                    frequency=f"{rate:.1f} events/min",
                    confidence=0.9,
                    description=f"Burst of {len(burst)} events in {duration:.1f} seconds",
                    metadata={
                        "burst_start": burst[0].isoformat(),
                        "burst_end": burst[-1].isoformat(),
                        "event_count": len(burst),
                        "duration_seconds": duration,
                        "rate_per_minute": rate
                    }
                ))
        
        return patterns
    
    def _detect_sequential_patterns(self, events_df: pl.DataFrame, min_occurrences: int) -> List[TemporalPattern]:
        """Detect sequential operation patterns."""
        patterns = []
        
        # Look for common operation sequences
        sequence_window = timedelta(hours=1)
        
        # Group by session or correlation ID
        for group_col in ["session_id", "correlation_id"]:
            if group_col in events_df.columns:
                grouped = events_df.filter(pl.col(group_col).is_not_null()).group_by(group_col)
                
                for group_data in grouped:
                    group_id, group_df = group_data
                    if group_df.height < 2:
                        continue
                    
                    # Sort by timestamp and extract operation sequence
                    sorted_ops = group_df.sort("timestamp")["operation"].to_list()
                    
                    if len(sorted_ops) >= min_occurrences:
                        sequence_str = " → ".join(sorted_ops)
                        pattern_id = str(uuid.uuid4())
                        
                        patterns.append(TemporalPattern(
                            pattern_id=pattern_id,
                            pattern_type="sequential",
                            entities=group_df["entity_id"].unique().to_list(),
                            frequency="session-based",
                            confidence=0.8,
                            description=f"Sequential operations: {sequence_str}",
                            metadata={
                                "sequence": sorted_ops,
                                "group_type": group_col,
                                "group_id": group_id,
                                "entity_count": len(group_df["entity_id"].unique())
                            }
                        ))
        
        return patterns
    
    def _interval_to_frequency(self, interval_seconds: float) -> str:
        """Convert interval in seconds to human-readable frequency."""
        if interval_seconds < 60:
            return f"{interval_seconds:.0f} seconds"
        elif interval_seconds < 3600:
            return f"{interval_seconds/60:.0f} minutes"
        elif interval_seconds < 86400:
            return f"{interval_seconds/3600:.1f} hours"
        else:
            return f"{interval_seconds/86400:.1f} days"
    
    def create_snapshot(self,
                       entity_id: str,
                       state: Dict[str, Any],
                       snapshot_type: str = "state",
                       change_summary: Optional[Dict[str, Any]] = None,
                       created_by: str = "system",
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a temporal snapshot of entity state.
        
        Args:
            entity_id: Entity identifier
            state: Current state of the entity
            snapshot_type: Type of snapshot
            change_summary: Summary of changes since last snapshot
            created_by: Creator identifier
            metadata: Additional metadata
            
        Returns:
            snapshot_id: Unique identifier for the snapshot
        """
        snapshot_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Determine version number
        existing_snapshots = self.snapshots_df.filter(pl.col("entity_id") == entity_id)
        if existing_snapshots.height > 0:
            version_list = existing_snapshots["version"].to_list()
            version = max(version_list) + 1 if version_list else 1
        else:
            version = 1
        
        # Calculate state size
        state_json = orjson.dumps(state)
        size_bytes = len(state_json)
        
        new_snapshot = pl.DataFrame({
            "snapshot_id": [snapshot_id],
            "entity_id": [entity_id],
            "timestamp": [now],
            "snapshot_type": [snapshot_type],
            "state": [state_json.decode()],
            "version": [version],
            "change_summary": [orjson.dumps(change_summary or {}).decode()],
            "created_by": [created_by],
            "metadata": [orjson.dumps(metadata or {}).decode()],
            "size_bytes": [size_bytes],
            "compression_ratio": [1.0]  # Could implement compression later
        })
        
        self.snapshots_df = pl.concat([self.snapshots_df, new_snapshot])
        return snapshot_id
    
    def get_snapshot(self,
                    entity_id: str,
                    timestamp: Optional[datetime] = None,
                    version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get entity snapshot at specific time or version.
        
        Args:
            entity_id: Entity identifier
            timestamp: Get snapshot at or before this time
            version: Get specific version
            
        Returns:
            Entity state or None if not found
        """
        query = self.snapshots_df.filter(pl.col("entity_id") == entity_id)
        
        if version is not None:
            query = query.filter(pl.col("version") == version)
        elif timestamp is not None:
            query = query.filter(pl.col("timestamp") <= timestamp)
        
        if query.height == 0:
            return None
        
        # Get the most recent snapshot
        snapshot = query.sort("timestamp", descending=True).limit(1)
        
        if snapshot.height == 0:
            return None
        
        row = snapshot.row(0, named=True)
        return orjson.loads(row["state"])
    
    def record_lineage(self,
                      source_entity_id: str,
                      target_entity_id: str,
                      operation: str,
                      transformation_details: Optional[Dict[str, Any]] = None,
                      confidence: float = 1.0,
                      lineage_type: str = "transformation",
                      processing_system: str = "unknown",
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record data lineage relationship.
        
        Args:
            source_entity_id: Source entity
            target_entity_id: Target entity
            operation: Transformation operation
            transformation_details: Details of the transformation
            confidence: Confidence in lineage (0.0 to 1.0)
            lineage_type: Type of lineage relationship
            processing_system: System that performed the transformation
            metadata: Additional metadata
            
        Returns:
            lineage_id: Unique identifier for the lineage record
        """
        lineage_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_lineage = pl.DataFrame({
            "lineage_id": [lineage_id],
            "source_entity_id": [source_entity_id],
            "target_entity_id": [target_entity_id],
            "operation": [operation],
            "timestamp": [now],
            "transformation_details": [orjson.dumps(transformation_details or {}).decode()],
            "confidence": [max(0.0, min(1.0, confidence))],
            "lineage_type": [lineage_type],
            "processing_system": [processing_system],
            "metadata": [orjson.dumps(metadata or {}).decode()],
            "validated": [False],
            "created_at": [now]
        })
        
        self.lineage_df = pl.concat([self.lineage_df, new_lineage])
        return lineage_id
    
    def get_lineage_chain(self,
                         entity_id: str,
                         direction: Literal["upstream", "downstream", "both"] = "both",
                         max_depth: int = 10) -> Dict[str, Any]:
        """
        Get complete lineage chain for an entity.
        
        Args:
            entity_id: Entity to trace lineage for
            direction: Direction to trace lineage
            max_depth: Maximum depth to traverse
            
        Returns:
            Lineage chain with upstream and downstream relationships
        """
        upstream_chain = []
        downstream_chain = []
        
        if direction in ["upstream", "both"]:
            upstream_chain = self._trace_lineage(entity_id, "upstream", max_depth)
        
        if direction in ["downstream", "both"]:
            downstream_chain = self._trace_lineage(entity_id, "downstream", max_depth)
        
        return {
            "entity_id": entity_id,
            "upstream": upstream_chain,
            "downstream": downstream_chain,
            "total_upstream": len(upstream_chain),
            "total_downstream": len(downstream_chain)
        }
    
    def _trace_lineage(self, entity_id: str, direction: str, max_depth: int, visited: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Recursively trace lineage in specified direction."""
        if visited is None:
            visited = set()
        
        if entity_id in visited or max_depth <= 0:
            return []
        
        visited.add(entity_id)
        lineage_chain = []
        
        if direction == "upstream":
            # Find entities that this entity was derived from
            related = self.lineage_df.filter(pl.col("target_entity_id") == entity_id)
            next_entities = related["source_entity_id"].unique().to_list()
        else:  # downstream
            # Find entities derived from this entity
            related = self.lineage_df.filter(pl.col("source_entity_id") == entity_id)
            next_entities = related["target_entity_id"].unique().to_list()
        
        for row in related.iter_rows(named=True):
            lineage_info = {
                "lineage_id": row["lineage_id"],
                "source_entity_id": row["source_entity_id"],
                "target_entity_id": row["target_entity_id"],
                "operation": row["operation"],
                "timestamp": row["timestamp"],
                "confidence": row["confidence"],
                "lineage_type": row["lineage_type"],
                "processing_system": row["processing_system"],
                "transformation_details": orjson.loads(row["transformation_details"])
            }
            lineage_chain.append(lineage_info)
        
        # Recursively trace further
        for next_entity in next_entities:
            if next_entity not in visited:
                sub_chain = self._trace_lineage(next_entity, direction, max_depth - 1, visited.copy())
                lineage_chain.extend(sub_chain)
        
        return lineage_chain
    
    def cleanup_old_data(self, cutoff_date: Optional[datetime] = None):
        """
        Clean up old temporal data based on retention policy.
        
        Args:
            cutoff_date: Delete data older than this date (default: retention_days ago)
        """
        if cutoff_date is None:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up events
        self.events_df = self.events_df.filter(pl.col("timestamp") >= cutoff_date)
        
        # Clean up snapshots (keep at least one per entity)
        entities_with_snapshots = self.snapshots_df["entity_id"].unique()
        snapshots_to_keep = []
        
        for entity_id in entities_with_snapshots:
            entity_snapshots = self.snapshots_df.filter(pl.col("entity_id") == entity_id)
            
            # Keep snapshots after cutoff date
            recent_snapshots = entity_snapshots.filter(pl.col("timestamp") >= cutoff_date)
            
            # If no recent snapshots, keep the most recent one
            if recent_snapshots.height == 0:
                recent_snapshots = entity_snapshots.sort("timestamp", descending=True).limit(1)
            
            snapshots_to_keep.append(recent_snapshots)
        
        if snapshots_to_keep:
            self.snapshots_df = pl.concat(snapshots_to_keep)
        else:
            self.snapshots_df = self.snapshots_df.filter(pl.col("timestamp") >= cutoff_date)
    
    def save(self):
        """Save all temporal data to Parquet files."""
        self.events_df.write_parquet(self.events_file, compression=self.compression)
        self.patterns_df.write_parquet(self.patterns_file, compression=self.compression)
        self.snapshots_df.write_parquet(self.snapshots_file, compression=self.compression)
        self.lineage_df.write_parquet(self.lineage_file, compression=self.compression)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get temporal layer statistics."""
        return {
            "events_count": self.events_df.height,
            "patterns_count": self.patterns_df.height,
            "snapshots_count": self.snapshots_df.height,
            "lineage_records_count": self.lineage_df.height,
            "unique_entities_with_events": self.events_df["entity_id"].n_unique(),
            "unique_entities_with_snapshots": self.snapshots_df["entity_id"].n_unique(),
            "date_range": {
                "earliest_event": self.events_df["timestamp"].min() if self.events_df.height > 0 else None,
                "latest_event": self.events_df["timestamp"].max() if self.events_df.height > 0 else None
            },
            "operation_types": self.events_df["operation"].unique().to_list(),
            "storage_size_mb": sum([
                self.events_file.stat().st_size if self.events_file.exists() else 0,
                self.patterns_file.stat().st_size if self.patterns_file.exists() else 0,
                self.snapshots_file.stat().st_size if self.snapshots_file.exists() else 0,
                self.lineage_file.stat().st_size if self.lineage_file.exists() else 0
            ]) / (1024 * 1024)
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for temporal data."""
        return self.get_stats()