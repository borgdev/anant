"""
Streaming Integration
====================

Comprehensive streaming framework that integrates:
- Real-time stream processing (GraphStreamProcessor)
- Time-aware graph storage (TemporalGraph)
- Event sourcing and persistence (EventStore)
- Stream analytics and monitoring
- WebSocket support for real-time clients
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, AsyncIterator, Union
import json

import polars as pl

from .core.stream_processor import (
    GraphStreamProcessor, GraphEvent, EventType, ConflictResolution,
    StreamConfig, StreamMetrics
)
from .core.temporal_graph import TemporalGraph, TemporalScope, TimeRange, GraphSnapshot
from .core.event_store import EventStore, EventQuery, create_event_store

# Optional WebSocket dependencies
try:
    import websockets
    import websockets.server
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming operation modes."""
    REAL_TIME = "real_time"          # Process events immediately
    BATCH = "batch"                  # Process in batches
    HYBRID = "hybrid"                # Adaptive batching based on load
    REPLAY = "replay"                # Historical event replay


@dataclass
class StreamingConfig:
    """Configuration for the streaming framework."""
    # Processing
    mode: StreamingMode = StreamingMode.HYBRID
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_concurrent_streams: int = 10
    
    # Storage
    enable_persistence: bool = True
    event_store_backend: str = "sqlite"
    event_store_path: str = "streaming_events.db"
    retention_days: Optional[int] = 30
    
    # Temporal features
    enable_temporal_analysis: bool = True
    snapshot_interval: int = 1000
    max_snapshots: int = 1000
    
    # WebSocket server
    enable_websocket: bool = True
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    max_websocket_connections: int = 100
    
    # Performance
    enable_compression: bool = True
    enable_metrics: bool = True
    metrics_interval_seconds: int = 60
    
    # Stream processing
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    enable_deduplication: bool = True
    deduplication_window_seconds: int = 300


@dataclass
class StreamingStats:
    """Real-time streaming statistics."""
    events_processed: int = 0
    events_per_second: float = 0.0
    bytes_processed: int = 0
    active_streams: int = 0
    active_websockets: int = 0
    temporal_snapshots: int = 0
    storage_size_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_event_time: Optional[datetime] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "events_processed": self.events_processed,
            "events_per_second": self.events_per_second,
            "bytes_processed": self.bytes_processed,
            "active_streams": self.active_streams,
            "active_websockets": self.active_websockets,
            "temporal_snapshots": self.temporal_snapshots,
            "storage_size_mb": self.storage_size_mb,
            "uptime_seconds": self.uptime_seconds,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "error_count": self.error_count
        }


class WebSocketHandler:
    """Handles WebSocket connections for real-time streaming."""
    
    def __init__(self, streaming_framework: "StreamingFramework"):
        self.framework = streaming_framework
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Dict[str, Any]] = {}  # connection_id -> subscription_info
        self.server = None
        
    async def start_server(self, host: str, port: int):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSocket support not available. Install with: pip install websockets")
            return
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        self.server = await websockets.server.serve(
            self.handle_connection,
            host,
            port,
            max_size=10**6,  # 1MB max message size
            max_queue=100,   # Max queued messages
            compression=None  # Disable compression for simplicity
        )
        
        logger.info(f"WebSocket server started on {host}:{port}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    async def handle_connection(self, websocket, path):
        """Handle a new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        self.connections.add(websocket)
        
        logger.info(f"New WebSocket connection: {connection_id}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
                "supported_commands": ["subscribe", "unsubscribe", "query", "stats"]
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, connection_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.connections.discard(websocket)
            if connection_id in self.subscriptions:
                del self.subscriptions[connection_id]
    
    async def handle_message(self, websocket, connection_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        command = data.get("type")
        
        if command == "subscribe":
            # Subscribe to events
            graph_id = data.get("graph_id")
            event_types = data.get("event_types", [])
            
            self.subscriptions[connection_id] = {
                "graph_id": graph_id,
                "event_types": [EventType(et) for et in event_types] if event_types else None,
                "websocket": websocket
            }
            
            await websocket.send(json.dumps({
                "type": "subscribed",
                "graph_id": graph_id,
                "event_types": event_types
            }))
        
        elif command == "unsubscribe":
            if connection_id in self.subscriptions:
                del self.subscriptions[connection_id]
            
            await websocket.send(json.dumps({
                "type": "unsubscribed"
            }))
        
        elif command == "query":
            # Query historical events
            query_data = data.get("query", {})
            query = EventQuery(
                graph_id=query_data.get("graph_id"),
                event_types=[EventType(et) for et in query_data.get("event_types", [])] if query_data.get("event_types") else None,
                start_time=datetime.fromisoformat(query_data["start_time"]) if query_data.get("start_time") else None,
                end_time=datetime.fromisoformat(query_data["end_time"]) if query_data.get("end_time") else None,
                limit=query_data.get("limit", 100)
            )
            
            events = await self.framework.event_store.query_events(query)
            
            await websocket.send(json.dumps({
                "type": "query_result",
                "events": [event.to_dict() for event in events],
                "count": len(events)
            }))
        
        elif command == "stats":
            # Get streaming statistics
            stats = await self.framework.get_stats()
            
            await websocket.send(json.dumps({
                "type": "stats",
                "data": stats.to_dict()
            }))
        
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown command: {command}"
            }))
    
    async def broadcast_event(self, event: GraphEvent):
        """Broadcast event to subscribed WebSocket clients."""
        if not self.connections:
            return
        
        # Find matching subscriptions
        for connection_id, subscription in self.subscriptions.items():
            if self._matches_subscription(event, subscription):
                try:
                    websocket = subscription["websocket"]
                    if websocket in self.connections:
                        await websocket.send(json.dumps({
                            "type": "event",
                            "event": event.to_dict()
                        }))
                except Exception as e:
                    logger.error(f"Failed to send event to WebSocket {connection_id}: {e}")
    
    def _matches_subscription(self, event: GraphEvent, subscription: Dict[str, Any]) -> bool:
        """Check if event matches WebSocket subscription."""
        # Check graph ID
        if subscription.get("graph_id") and event.graph_id != subscription["graph_id"]:
            return False
        
        # Check event types
        if subscription.get("event_types") and event.event_type not in subscription["event_types"]:
            return False
        
        return True


class StreamingFramework:
    """
    Comprehensive streaming framework that integrates all streaming components.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the streaming framework."""
        self.config = config or StreamingConfig()
        
        # Core components
        self.stream_processor: Optional[GraphStreamProcessor] = None
        self.temporal_graph: Optional[TemporalGraph] = None
        self.event_store: Optional[EventStore] = None
        self.websocket_handler: Optional[WebSocketHandler] = None
        
        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.stats = StreamingStats()
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_callbacks: List[Callable[[GraphEvent], None]] = []
        
        logger.info("Initialized StreamingFramework")
    
    async def start(self):
        """Start the streaming framework."""
        if self.is_running:
            logger.warning("Streaming framework already running")
            return
        
        logger.info("Starting streaming framework...")
        self.start_time = datetime.now()
        
        try:
            # Initialize event store
            if self.config.enable_persistence:
                self.event_store = await create_event_store(
                    self.config.event_store_backend,
                    db_path=self.config.event_store_path,
                    retention_days=self.config.retention_days,
                    enable_compression=self.config.enable_compression
                )
                logger.info("Event store initialized")
            
            # Initialize temporal graph
            if self.config.enable_temporal_analysis:
                from .core.temporal_graph import VersioningStrategy
                self.temporal_graph = TemporalGraph(
                    versioning_strategy=VersioningStrategy.HYBRID,
                    max_snapshots=self.config.max_snapshots,
                    snapshot_interval=timedelta(hours=1)
                )
                logger.info("Temporal graph initialized")
            
            # Initialize stream processor
            stream_config = StreamConfig(
                conflict_resolution=self.config.conflict_resolution,
                enable_deduplication=self.config.enable_deduplication,
                deduplication_window_seconds=self.config.deduplication_window_seconds,
                enable_metrics=self.config.enable_metrics,
                metrics_interval_seconds=self.config.metrics_interval_seconds
            )
            
            self.stream_processor = GraphStreamProcessor(stream_config)
            await self.stream_processor.start()
            logger.info("Stream processor initialized")
            
            # Initialize WebSocket handler
            if self.config.enable_websocket:
                self.websocket_handler = WebSocketHandler(self)
                await self.websocket_handler.start_server(
                    self.config.websocket_host,
                    self.config.websocket_port
                )
                logger.info("WebSocket server initialized")
            
            # Start background tasks
            self.active_tasks.add(asyncio.create_task(self._event_processing_loop()))
            if self.config.enable_metrics:
                self.active_tasks.add(asyncio.create_task(self._metrics_loop()))
            
            # Subscribe to stream processor events
            if self.stream_processor and self.event_store:
                await self.stream_processor.subscribe_to_events(self._handle_stream_event)
            
            self.is_running = True
            logger.info("Streaming framework started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start streaming framework: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the streaming framework."""
        if not self.is_running:
            return
        
        logger.info("Stopping streaming framework...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.active_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()
        
        # Stop components
        if self.websocket_handler:
            await self.websocket_handler.stop_server()
        
        if self.stream_processor:
            await self.stream_processor.stop()
        
        if self.event_store:
            await self.event_store.close()
        
        logger.info("Streaming framework stopped")
    
    async def process_event(self, event: GraphEvent) -> bool:
        """Process a single event through the streaming pipeline."""
        if not self.is_running:
            logger.warning("Framework not running, dropping event")
            return False
        
        try:
            # Add to processing queue
            await self.event_queue.put(event)
            return True
        except Exception as e:
            logger.error(f"Failed to queue event: {e}")
            self.stats.error_count += 1
            return False
    
    async def process_events(self, events: List[GraphEvent]) -> int:
        """Process multiple events efficiently."""
        if not self.is_running:
            logger.warning("Framework not running, dropping events")
            return 0
        
        processed = 0
        for event in events:
            if await self.process_event(event):
                processed += 1
        
        return processed
    
    async def query_events(self, query: EventQuery) -> List[GraphEvent]:
        """Query historical events."""
        if not self.event_store:
            logger.warning("Event store not available")
            return []
        
        return await self.event_store.query_events(query)
    
    async def get_temporal_view(self, 
                               graph_id: str,
                               time_range: TimeRange,
                               aggregation: str = "union"):
        """Get a temporal view of the graph."""
        if not self.temporal_graph:
            logger.warning("Temporal graph not available")
            return None
        
        return self.temporal_graph.create_temporal_view(time_range, aggregation)
    
    async def replay_events(self, 
                           graph_id: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> AsyncIterator[GraphEvent]:
        """Replay historical events."""
        if not self.event_store:
            logger.warning("Event store not available")
            return
        
        async for event in self.event_store.replay_events(graph_id, start_time, end_time):
            yield event
    
    def add_event_callback(self, callback: Callable[[GraphEvent], None]):
        """Add a callback for processed events."""
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[GraphEvent], None]):
        """Remove an event callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    async def get_stats(self) -> StreamingStats:
        """Get current streaming statistics."""
        # Update basic stats
        if self.start_time:
            self.stats.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Get component stats
        if self.event_store:
            store_stats = await self.event_store.get_stats()
            self.stats.storage_size_mb = store_stats.get("file_size_mb", 0.0)
        
        if self.temporal_graph:
            temporal_stats = self.temporal_graph.get_storage_stats()
            self.stats.temporal_snapshots = temporal_stats.get("num_snapshots", 0)
        
        if self.websocket_handler:
            self.stats.active_websockets = len(self.websocket_handler.connections)
        
        return self.stats
    
    async def _event_processing_loop(self):
        """Main event processing loop."""
        logger.info("Started event processing loop")
        
        batch = []
        last_batch_time = time.time()
        
        try:
            while self.is_running:
                try:
                    # Get event with timeout
                    timeout = max(0.1, (self.config.batch_timeout_ms - 
                                      (time.time() - last_batch_time) * 1000) / 1000)
                    
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=timeout
                    )
                    
                    batch.append(event)
                    
                    # Process batch if conditions are met
                    should_process = (
                        len(batch) >= self.config.batch_size or
                        (time.time() - last_batch_time) * 1000 >= self.config.batch_timeout_ms or
                        self.config.mode == StreamingMode.REAL_TIME
                    )
                    
                    if should_process and batch:
                        await self._process_event_batch(batch)
                        batch.clear()
                        last_batch_time = time.time()
                
                except asyncio.TimeoutError:
                    # Process any pending events in batch
                    if batch:
                        await self._process_event_batch(batch)
                        batch.clear()
                        last_batch_time = time.time()
                
                except Exception as e:
                    logger.error(f"Error in event processing loop: {e}")
                    self.stats.error_count += 1
                    await asyncio.sleep(0.1)  # Brief pause before retrying
        
        except asyncio.CancelledError:
            logger.info("Event processing loop cancelled")
        finally:
            # Process any remaining events
            if batch:
                await self._process_event_batch(batch)
    
    async def _process_event_batch(self, events: List[GraphEvent]):
        """Process a batch of events."""
        if not events:
            return
        
        try:
            # Store in event store
            if self.event_store:
                stored = await self.event_store.store_events(events, notify_subscribers=False)
                logger.debug(f"Stored {stored}/{len(events)} events")
            
            # Add to temporal graph
            if self.temporal_graph:
                for event in events:
                    # This would need graph reconstruction logic in a real implementation
                    pass
            
            # Process through stream processor
            if self.stream_processor:
                for event in events:
                    await self.stream_processor.process_event(event)
            
            # Notify WebSocket clients
            if self.websocket_handler:
                for event in events:
                    await self.websocket_handler.broadcast_event(event)
            
            # Call event callbacks
            for event in events:
                for callback in self.event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
            
            # Update stats
            self.stats.events_processed += len(events)
            self.stats.bytes_processed += sum(len(event.serialize()) for event in events)
            if events:
                self.stats.last_event_time = max(event.timestamp for event in events)
        
        except Exception as e:
            logger.error(f"Failed to process event batch: {e}")
            self.stats.error_count += 1
    
    async def _handle_stream_event(self, event: GraphEvent):
        """Handle events from stream processor."""
        await self.process_event(event)
    
    async def _metrics_loop(self):
        """Background metrics collection loop."""
        logger.info("Started metrics collection loop")
        
        last_event_count = 0
        last_time = time.time()
        
        try:
            while self.is_running:
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
                # Calculate events per second
                current_time = time.time()
                current_events = self.stats.events_processed
                
                if current_time > last_time:
                    self.stats.events_per_second = (
                        (current_events - last_event_count) / (current_time - last_time)
                    )
                
                last_event_count = current_events
                last_time = current_time
                
                # Log metrics
                stats = await self.get_stats()
                logger.info(f"Streaming metrics: {stats.events_per_second:.1f} events/sec, "
                          f"{stats.active_streams} streams, {stats.storage_size_mb:.1f}MB storage")
        
        except asyncio.CancelledError:
            logger.info("Metrics loop cancelled")
        except Exception as e:
            logger.error(f"Error in metrics loop: {e}")
    
    @asynccontextmanager
    async def streaming_session(self):
        """Context manager for streaming session."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# Factory functions
async def create_streaming_framework(config: Optional[StreamingConfig] = None) -> StreamingFramework:
    """Create and configure a streaming framework."""
    framework = StreamingFramework(config)
    return framework


async def create_real_time_streaming(
    backend: str = "memory",
    enable_websocket: bool = True,
    **kwargs
) -> StreamingFramework:
    """Create a real-time streaming setup."""
    config = StreamingConfig(
        mode=StreamingMode.REAL_TIME,
        event_store_backend=backend,
        enable_websocket=enable_websocket,
        batch_size=1,  # Process immediately
        **kwargs
    )
    
    return await create_streaming_framework(config)


async def create_batch_streaming(
    batch_size: int = 1000,
    batch_timeout_ms: int = 5000,
    **kwargs
) -> StreamingFramework:
    """Create a batch streaming setup."""
    config = StreamingConfig(
        mode=StreamingMode.BATCH,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        **kwargs
    )
    
    return await create_streaming_framework(config)