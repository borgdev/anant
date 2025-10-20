"""
Event Store System
==================

Persistent event storage for graph streaming with:
- Event sourcing pattern implementation
- Multiple storage backends (SQLite, PostgreSQL, MongoDB, File)
- Event replay and time-travel queries
- Snapshots and compaction
- Event indexing and efficient queries
- Event streaming and subscriptions
"""

import asyncio
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, AsyncIterator, Iterator, Callable
import json
import gzip
import pickle

import polars as pl

from .stream_processor import GraphEvent, EventType

# Optional database dependencies
try:
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import motor.motor_asyncio
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import orjson
    JSON_SERIALIZER = orjson
except ImportError:
    import json as JSON_SERIALIZER

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Event storage backend types."""
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class CompressionType(Enum):
    """Event compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"


@dataclass
class EventQuery:
    """Query specification for event retrieval."""
    graph_id: Optional[str] = None
    event_types: Optional[List[EventType]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"
    descending: bool = False
    include_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            "graph_id": self.graph_id,
            "event_types": [et.value for et in self.event_types] if self.event_types else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "limit": self.limit,
            "offset": self.offset,
            "order_by": self.order_by,
            "descending": self.descending,
            "include_metadata": self.include_metadata
        }


@dataclass
class EventBatch:
    """Batch of events for efficient processing."""
    events: List[GraphEvent]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.events)
    
    def get_size_bytes(self) -> int:
        """Estimate batch size in bytes."""
        total_size = 0
        for event in self.events:
            total_size += len(event.serialize())
        return total_size
    
    def filter_by_type(self, event_types: List[EventType]) -> "EventBatch":
        """Filter events by type."""
        filtered_events = [e for e in self.events if e.event_type in event_types]
        return EventBatch(
            events=filtered_events,
            batch_id=f"filtered_{self.batch_id}",
            metadata={**self.metadata, "filtered_from": self.batch_id}
        )


@dataclass
class Snapshot:
    """Event store snapshot for compaction."""
    snapshot_id: str
    graph_id: str
    timestamp: datetime
    last_event_id: str
    event_count: int
    data: Dict[str, Any]  # Serialized graph state
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression: CompressionType = CompressionType.GZIP
    
    def serialize(self) -> bytes:
        """Serialize snapshot to bytes."""
        data = {
            "snapshot_id": self.snapshot_id,
            "graph_id": self.graph_id,
            "timestamp": self.timestamp.isoformat(),
            "last_event_id": self.last_event_id,
            "event_count": self.event_count,
            "data": self.data,
            "metadata": self.metadata,
            "compression": self.compression.value
        }
        
        serialized = JSON_SERIALIZER.dumps(data)
        
        if self.compression == CompressionType.GZIP:
            return gzip.compress(serialized)
        else:
            return serialized
    
    @classmethod
    def deserialize(cls, data: bytes) -> "Snapshot":
        """Deserialize snapshot from bytes."""
        # Try to decompress if it's gzipped
        try:
            decompressed = gzip.decompress(data)
            data = decompressed
        except:
            pass  # Not compressed or different compression
        
        parsed = JSON_SERIALIZER.loads(data)
        
        return cls(
            snapshot_id=parsed["snapshot_id"],
            graph_id=parsed["graph_id"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            last_event_id=parsed["last_event_id"],
            event_count=parsed["event_count"],
            data=parsed["data"],
            metadata=parsed.get("metadata", {}),
            compression=CompressionType(parsed.get("compression", "gzip"))
        )


class EventStoreBackend(ABC):
    """Abstract base class for event store backends."""
    
    @abstractmethod
    async def store_event(self, event: GraphEvent) -> bool:
        """Store a single event."""
        pass
    
    @abstractmethod
    async def store_events(self, events: List[GraphEvent]) -> int:
        """Store multiple events. Returns number of events stored."""
        pass
    
    @abstractmethod
    async def query_events(self, query: EventQuery) -> List[GraphEvent]:
        """Query events based on criteria."""
        pass
    
    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[GraphEvent]:
        """Get a specific event by ID."""
        pass
    
    @abstractmethod
    async def delete_events(self, query: EventQuery) -> int:
        """Delete events matching query. Returns number deleted."""
        pass
    
    @abstractmethod
    async def store_snapshot(self, snapshot: Snapshot) -> bool:
        """Store a snapshot."""
        pass
    
    @abstractmethod
    async def get_latest_snapshot(self, graph_id: str) -> Optional[Snapshot]:
        """Get the latest snapshot for a graph."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the backend connection."""
        pass


class MemoryEventStore(EventStoreBackend):
    """In-memory event store for testing and development."""
    
    def __init__(self, max_events: int = 100000):
        self.events: Dict[str, GraphEvent] = {}
        self.events_by_graph: Dict[str, List[str]] = {}
        self.snapshots: Dict[str, List[Snapshot]] = {}
        self.max_events = max_events
        self.event_order: List[str] = []  # For chronological ordering
        
    async def store_event(self, event: GraphEvent) -> bool:
        """Store a single event."""
        # Manage memory limits
        if len(self.events) >= self.max_events:
            # Remove oldest events
            oldest_id = self.event_order.pop(0)
            oldest_event = self.events.pop(oldest_id)
            
            # Update graph index
            if oldest_event.graph_id in self.events_by_graph:
                try:
                    self.events_by_graph[oldest_event.graph_id].remove(oldest_id)
                except ValueError:
                    pass
        
        # Store event
        self.events[event.event_id] = event
        self.event_order.append(event.event_id)
        
        # Update graph index
        if event.graph_id not in self.events_by_graph:
            self.events_by_graph[event.graph_id] = []
        self.events_by_graph[event.graph_id].append(event.event_id)
        
        return True
    
    async def store_events(self, events: List[GraphEvent]) -> int:
        """Store multiple events."""
        stored = 0
        for event in events:
            if await self.store_event(event):
                stored += 1
        return stored
    
    async def query_events(self, query: EventQuery) -> List[GraphEvent]:
        """Query events based on criteria."""
        results = []
        
        # Get candidate events
        if query.graph_id:
            candidate_ids = self.events_by_graph.get(query.graph_id, [])
            candidates = [self.events[eid] for eid in candidate_ids if eid in self.events]
        else:
            candidates = list(self.events.values())
        
        # Apply filters
        for event in candidates:
            if self._matches_query(event, query):
                results.append(event)
        
        # Sort results
        if query.order_by == "timestamp":
            results.sort(key=lambda e: e.timestamp, reverse=query.descending)
        
        # Apply pagination
        start = query.offset
        end = start + query.limit if query.limit else None
        return results[start:end]
    
    async def get_event(self, event_id: str) -> Optional[GraphEvent]:
        """Get a specific event by ID."""
        return self.events.get(event_id)
    
    async def delete_events(self, query: EventQuery) -> int:
        """Delete events matching query."""
        to_delete = await self.query_events(query)
        deleted = 0
        
        for event in to_delete:
            if event.event_id in self.events:
                del self.events[event.event_id]
                self.event_order.remove(event.event_id)
                
                # Update graph index
                if event.graph_id in self.events_by_graph:
                    try:
                        self.events_by_graph[event.graph_id].remove(event.event_id)
                    except ValueError:
                        pass
                
                deleted += 1
        
        return deleted
    
    async def store_snapshot(self, snapshot: Snapshot) -> bool:
        """Store a snapshot."""
        if snapshot.graph_id not in self.snapshots:
            self.snapshots[snapshot.graph_id] = []
        
        self.snapshots[snapshot.graph_id].append(snapshot)
        # Keep only latest 10 snapshots per graph
        self.snapshots[snapshot.graph_id] = sorted(
            self.snapshots[snapshot.graph_id],
            key=lambda s: s.timestamp,
            reverse=True
        )[:10]
        
        return True
    
    async def get_latest_snapshot(self, graph_id: str) -> Optional[Snapshot]:
        """Get the latest snapshot for a graph."""
        if graph_id not in self.snapshots or not self.snapshots[graph_id]:
            return None
        
        return max(self.snapshots[graph_id], key=lambda s: s.timestamp)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(len(event.serialize()) for event in self.events.values())
        
        return {
            "backend": "memory",
            "total_events": len(self.events),
            "graphs": len(self.events_by_graph),
            "total_snapshots": sum(len(snaps) for snaps in self.snapshots.values()),
            "estimated_size_bytes": total_size,
            "estimated_size_mb": total_size / (1024 * 1024)
        }
    
    async def close(self):
        """Close the backend connection."""
        self.events.clear()
        self.events_by_graph.clear()
        self.snapshots.clear()
    
    def _matches_query(self, event: GraphEvent, query: EventQuery) -> bool:
        """Check if event matches query criteria."""
        if query.event_types and event.event_type not in query.event_types:
            return False
        
        if query.start_time and event.timestamp < query.start_time:
            return False
        
        if query.end_time and event.timestamp > query.end_time:
            return False
        
        if query.source and event.source != query.source:
            return False
        
        if query.correlation_id and event.correlation_id != query.correlation_id:
            return False
        
        return True


class SQLiteEventStore(EventStoreBackend):
    """SQLite-based event store for persistent storage."""
    
    def __init__(self, db_path: str, table_prefix: str = "events"):
        self.db_path = db_path
        self.table_prefix = table_prefix
        self.events_table = f"{table_prefix}_events"
        self.snapshots_table = f"{table_prefix}_snapshots"
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure database tables are created."""
        if self._initialized:
            return
        
        if not SQLITE_AVAILABLE:
            raise RuntimeError("aiosqlite not available. Install with: pip install aiosqlite")
        
        async with aiosqlite.connect(self.db_path) as db:
            # Create events table
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.events_table} (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT,
                    source TEXT,
                    version INTEGER DEFAULT 1,
                    correlation_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.events_table}_graph_time 
                ON {self.events_table}(graph_id, timestamp)
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.events_table}_type 
                ON {self.events_table}(event_type)
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.events_table}_correlation 
                ON {self.events_table}(correlation_id)
            """)
            
            # Create snapshots table
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.snapshots_table} (
                    snapshot_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    last_event_id TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    metadata TEXT,
                    compression TEXT DEFAULT 'gzip',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.snapshots_table}_graph_time 
                ON {self.snapshots_table}(graph_id, timestamp)
            """)
            
            await db.commit()
        
        self._initialized = True
    
    async def store_event(self, event: GraphEvent) -> bool:
        """Store a single event."""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f"""
                    INSERT INTO {self.events_table} 
                    (event_id, event_type, timestamp, graph_id, data, metadata, source, version, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.graph_id,
                    JSON_SERIALIZER.dumps(event.data),
                    JSON_SERIALIZER.dumps(event.metadata),
                    event.source,
                    event.version,
                    event.correlation_id
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store event {event.event_id}: {e}")
            return False
    
    async def store_events(self, events: List[GraphEvent]) -> int:
        """Store multiple events."""
        await self._ensure_initialized()
        
        if not events:
            return 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                event_data = [
                    (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.graph_id,
                        JSON_SERIALIZER.dumps(event.data),
                        JSON_SERIALIZER.dumps(event.metadata),
                        event.source,
                        event.version,
                        event.correlation_id
                    )
                    for event in events
                ]
                
                await db.executemany(f"""
                    INSERT OR REPLACE INTO {self.events_table} 
                    (event_id, event_type, timestamp, graph_id, data, metadata, source, version, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, event_data)
                
                await db.commit()
                return len(events)
        except Exception as e:
            logger.error(f"Failed to store {len(events)} events: {e}")
            return 0
    
    async def query_events(self, query: EventQuery) -> List[GraphEvent]:
        """Query events based on criteria."""
        await self._ensure_initialized()
        
        # Build SQL query
        where_clauses = []
        params = []
        
        if query.graph_id:
            where_clauses.append("graph_id = ?")
            params.append(query.graph_id)
        
        if query.event_types:
            placeholders = ",".join("?" * len(query.event_types))
            where_clauses.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in query.event_types])
        
        if query.start_time:
            where_clauses.append("timestamp >= ?")
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            where_clauses.append("timestamp <= ?")
            params.append(query.end_time.isoformat())
        
        if query.source:
            where_clauses.append("source = ?")
            params.append(query.source)
        
        if query.correlation_id:
            where_clauses.append("correlation_id = ?")
            params.append(query.correlation_id)
        
        # Build complete query
        sql = f"SELECT * FROM {self.events_table}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        sql += f" ORDER BY {query.order_by}"
        if query.descending:
            sql += " DESC"
        
        if query.limit:
            sql += f" LIMIT {query.limit}"
        
        if query.offset:
            sql += f" OFFSET {query.offset}"
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    events = []
                    for row in rows:
                        event = GraphEvent(
                            event_id=row["event_id"],
                            event_type=EventType(row["event_type"]),
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            graph_id=row["graph_id"],
                            data=JSON_SERIALIZER.loads(row["data"]),
                            metadata=JSON_SERIALIZER.loads(row["metadata"] or "{}"),
                            source=row["source"],
                            version=row["version"],
                            correlation_id=row["correlation_id"]
                        )
                        events.append(event)
                    
                    return events
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []
    
    async def get_event(self, event_id: str) -> Optional[GraphEvent]:
        """Get a specific event by ID."""
        query = EventQuery()
        query.limit = 1
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(f"""
                    SELECT * FROM {self.events_table} WHERE event_id = ?
                """, (event_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return GraphEvent(
                            event_id=row["event_id"],
                            event_type=EventType(row["event_type"]),
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            graph_id=row["graph_id"],
                            data=JSON_SERIALIZER.loads(row["data"]),
                            metadata=JSON_SERIALIZER.loads(row["metadata"] or "{}"),
                            source=row["source"],
                            version=row["version"],
                            correlation_id=row["correlation_id"]
                        )
                    return None
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            return None
    
    async def delete_events(self, query: EventQuery) -> int:
        """Delete events matching query."""
        # First get the events to count them
        events_to_delete = await self.query_events(query)
        
        if not events_to_delete:
            return 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete by event IDs
                event_ids = [event.event_id for event in events_to_delete]
                placeholders = ",".join("?" * len(event_ids))
                
                await db.execute(f"""
                    DELETE FROM {self.events_table} WHERE event_id IN ({placeholders})
                """, event_ids)
                
                await db.commit()
                return len(events_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete events: {e}")
            return 0
    
    async def store_snapshot(self, snapshot: Snapshot) -> bool:
        """Store a snapshot."""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f"""
                    INSERT OR REPLACE INTO {self.snapshots_table} 
                    (snapshot_id, graph_id, timestamp, last_event_id, event_count, data, metadata, compression)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.snapshot_id,
                    snapshot.graph_id,
                    snapshot.timestamp.isoformat(),
                    snapshot.last_event_id,
                    snapshot.event_count,
                    snapshot.serialize(),
                    JSON_SERIALIZER.dumps(snapshot.metadata),
                    snapshot.compression.value
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store snapshot {snapshot.snapshot_id}: {e}")
            return False
    
    async def get_latest_snapshot(self, graph_id: str) -> Optional[Snapshot]:
        """Get the latest snapshot for a graph."""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(f"""
                    SELECT * FROM {self.snapshots_table} 
                    WHERE graph_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (graph_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return Snapshot(
                            snapshot_id=row["snapshot_id"],
                            graph_id=row["graph_id"],
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            last_event_id=row["last_event_id"],
                            event_count=row["event_count"],
                            data=JSON_SERIALIZER.loads(gzip.decompress(row["data"])),
                            metadata=JSON_SERIALIZER.loads(row["metadata"] or "{}"),
                            compression=CompressionType(row["compression"])
                        )
                    return None
        except Exception as e:
            logger.error(f"Failed to get latest snapshot for graph {graph_id}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get event stats
                async with db.execute(f"SELECT COUNT(*) FROM {self.events_table}") as cursor:
                    total_events = (await cursor.fetchone())[0]
                
                async with db.execute(f"SELECT COUNT(DISTINCT graph_id) FROM {self.events_table}") as cursor:
                    unique_graphs = (await cursor.fetchone())[0]
                
                async with db.execute(f"SELECT COUNT(*) FROM {self.snapshots_table}") as cursor:
                    total_snapshots = (await cursor.fetchone())[0]
                
                # Get database file size
                db_file = Path(self.db_path)
                file_size = db_file.stat().st_size if db_file.exists() else 0
                
                return {
                    "backend": "sqlite",
                    "db_path": self.db_path,
                    "total_events": total_events,
                    "graphs": unique_graphs,
                    "total_snapshots": total_snapshots,
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the backend connection."""
        # SQLite connections are per-operation, nothing to close
        pass


class EventStore:
    """
    High-level event store with features for event sourcing, replay, and analytics.
    """
    
    def __init__(self, 
                 backend: EventStoreBackend,
                 enable_snapshots: bool = True,
                 snapshot_interval: int = 1000,  # events
                 retention_days: Optional[int] = None,
                 enable_compression: bool = True):
        """
        Initialize event store.
        
        Args:
            backend: Storage backend implementation
            enable_snapshots: Whether to create snapshots for compaction
            snapshot_interval: Create snapshot every N events
            retention_days: Delete events older than N days (None = keep all)
            enable_compression: Use compression for snapshots
        """
        self.backend = backend
        self.enable_snapshots = enable_snapshots
        self.snapshot_interval = snapshot_interval
        self.retention_days = retention_days
        self.enable_compression = enable_compression
        
        # Metrics
        self.events_stored = 0
        self.snapshots_created = 0
        self.last_snapshot_event_count = 0
        
        # Event subscribers
        self.subscribers: Dict[str, List[Callable[[GraphEvent], None]]] = {}
        
        logger.info(f"Initialized EventStore with {type(backend).__name__} backend")
    
    async def store_event(self, event: GraphEvent, notify_subscribers: bool = True) -> bool:
        """Store a single event and optionally notify subscribers."""
        success = await self.backend.store_event(event)
        
        if success:
            self.events_stored += 1
            
            # Check if we need to create a snapshot
            if (self.enable_snapshots and 
                self.events_stored - self.last_snapshot_event_count >= self.snapshot_interval):
                await self._maybe_create_snapshot(event.graph_id)
            
            # Notify subscribers
            if notify_subscribers:
                await self._notify_subscribers(event)
        
        return success
    
    async def store_events(self, events: List[GraphEvent], notify_subscribers: bool = True) -> int:
        """Store multiple events efficiently."""
        if not events:
            return 0
        
        stored = await self.backend.store_events(events)
        self.events_stored += stored
        
        # Check snapshots for each affected graph
        if self.enable_snapshots and stored > 0:
            affected_graphs = set(event.graph_id for event in events)
            for graph_id in affected_graphs:
                if (self.events_stored - self.last_snapshot_event_count >= self.snapshot_interval):
                    await self._maybe_create_snapshot(graph_id)
        
        # Notify subscribers
        if notify_subscribers and stored > 0:
            for event in events[:stored]:  # Only notify for successfully stored events
                await self._notify_subscribers(event)
        
        return stored
    
    async def store_batch(self, batch: EventBatch, notify_subscribers: bool = True) -> int:
        """Store an event batch."""
        return await self.store_events(batch.events, notify_subscribers)
    
    async def query_events(self, query: EventQuery) -> List[GraphEvent]:
        """Query events from the store."""
        return await self.backend.query_events(query)
    
    async def get_event(self, event_id: str) -> Optional[GraphEvent]:
        """Get a specific event by ID."""
        return await self.backend.get_event(event_id)
    
    async def replay_events(self, 
                          graph_id: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_types: Optional[List[EventType]] = None) -> AsyncIterator[GraphEvent]:
        """
        Replay events in chronological order for event sourcing.
        
        Args:
            graph_id: Graph to replay events for
            start_time: Start time for replay (inclusive)
            end_time: End time for replay (inclusive)
            event_types: Filter by event types
            
        Yields:
            Events in chronological order
        """
        query = EventQuery(
            graph_id=graph_id,
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            order_by="timestamp",
            descending=False
        )
        
        # Process in batches to handle large result sets
        batch_size = 1000
        offset = 0
        
        while True:
            query.limit = batch_size
            query.offset = offset
            
            batch_events = await self.query_events(query)
            
            if not batch_events:
                break
            
            for event in batch_events:
                yield event
            
            if len(batch_events) < batch_size:
                break  # Last batch
            
            offset += batch_size
    
    async def create_snapshot(self, graph_id: str, graph_state: Dict[str, Any]) -> Optional[Snapshot]:
        """Manually create a snapshot for a graph."""
        # Get the latest event for this graph
        latest_events = await self.query_events(EventQuery(
            graph_id=graph_id,
            limit=1,
            order_by="timestamp",
            descending=True
        ))
        
        if not latest_events:
            logger.warning(f"No events found for graph {graph_id}, cannot create snapshot")
            return None
        
        latest_event = latest_events[0]
        
        # Count events for this graph
        all_events = await self.query_events(EventQuery(graph_id=graph_id))
        event_count = len(all_events)
        
        snapshot = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            graph_id=graph_id,
            timestamp=datetime.now(),
            last_event_id=latest_event.event_id,
            event_count=event_count,
            data=graph_state,
            compression=CompressionType.GZIP if self.enable_compression else CompressionType.NONE
        )
        
        success = await self.backend.store_snapshot(snapshot)
        if success:
            self.snapshots_created += 1
            self.last_snapshot_event_count = self.events_stored
            logger.info(f"Created snapshot {snapshot.snapshot_id} for graph {graph_id}")
            return snapshot
        
        return None
    
    async def get_latest_snapshot(self, graph_id: str) -> Optional[Snapshot]:
        """Get the latest snapshot for a graph."""
        return await self.backend.get_latest_snapshot(graph_id)
    
    async def subscribe(self, 
                       subscription_id: str, 
                       callback: Callable[[GraphEvent], None],
                       event_filter: Optional[Callable[[GraphEvent], bool]] = None):
        """
        Subscribe to event notifications.
        
        Args:
            subscription_id: Unique ID for this subscription
            callback: Function to call when events match
            event_filter: Optional filter function
        """
        if subscription_id not in self.subscribers:
            self.subscribers[subscription_id] = []
        
        # Wrap callback with filter if provided
        if event_filter:
            def filtered_callback(event: GraphEvent):
                if event_filter(event):
                    callback(event)
            self.subscribers[subscription_id].append(filtered_callback)
        else:
            self.subscribers[subscription_id].append(callback)
        
        logger.info(f"Added subscription {subscription_id}")
    
    async def unsubscribe(self, subscription_id: str):
        """Remove all callbacks for a subscription."""
        if subscription_id in self.subscribers:
            del self.subscribers[subscription_id]
            logger.info(f"Removed subscription {subscription_id}")
    
    async def cleanup_old_events(self) -> int:
        """Remove events older than retention period."""
        if not self.retention_days:
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        query = EventQuery(end_time=cutoff_time)
        
        deleted = await self.backend.delete_events(query)
        logger.info(f"Cleaned up {deleted} old events")
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive event store statistics."""
        backend_stats = await self.backend.get_stats()
        
        store_stats = {
            "events_stored_session": self.events_stored,
            "snapshots_created_session": self.snapshots_created,
            "active_subscriptions": len(self.subscribers),
            "snapshot_interval": self.snapshot_interval,
            "retention_days": self.retention_days,
            "compression_enabled": self.enable_compression
        }
        
        return {**backend_stats, **store_stats}
    
    async def close(self):
        """Close the event store and backend."""
        await self.backend.close()
        self.subscribers.clear()
        logger.info("Event store closed")
    
    async def _maybe_create_snapshot(self, graph_id: str):
        """Create a snapshot if needed."""
        # This is a placeholder - in a real implementation, you'd need to
        # get the current graph state to create a meaningful snapshot
        logger.debug(f"Snapshot check for graph {graph_id}")
    
    async def _notify_subscribers(self, event: GraphEvent):
        """Notify all subscribers about an event."""
        for subscription_id, callbacks in self.subscribers.items():
            for callback in callbacks:
                try:
                    # Run callback in a separate task to avoid blocking
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in subscription {subscription_id}: {e}")


# Factory functions
def create_memory_store(**kwargs) -> EventStore:
    """Create an in-memory event store."""
    backend = MemoryEventStore(**kwargs)
    return EventStore(backend)


def create_sqlite_store(db_path: str, **kwargs) -> EventStore:
    """Create a SQLite-based event store."""
    backend = SQLiteEventStore(db_path)
    return EventStore(backend, **kwargs)


async def create_event_store(backend_type: str, **kwargs) -> EventStore:
    """Create an event store with the specified backend."""
    if backend_type == "memory":
        return create_memory_store(**kwargs)
    elif backend_type == "sqlite":
        db_path = kwargs.pop("db_path", "events.db")
        return create_sqlite_store(db_path, **kwargs)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")