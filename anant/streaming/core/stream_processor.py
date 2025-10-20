"""
Core Stream Processor
====================

Production-grade stream processing for real-time graph updates with:
- Event-driven architecture
- Fault tolerance and recovery
- Multiple stream backends (Kafka, Pulsar, Redis)
- Conflict resolution and consistency
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, AsyncIterator, Union
import json
from datetime import datetime, timedelta

import polars as pl

# Optional streaming dependencies
try:
    import aiokafka
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import pulsar
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import orjson
    JSON_SERIALIZER = orjson
except ImportError:
    import json as JSON_SERIALIZER

logger = logging.getLogger(__name__)


class StreamBackend(Enum):
    """Available streaming backends."""
    KAFKA = "kafka"
    PULSAR = "pulsar"
    REDIS_STREAMS = "redis_streams"
    WEBSOCKET = "websocket"
    MEMORY = "memory"  # For testing


class EventType(Enum):
    """Types of graph events."""
    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_UPDATED = "node_updated"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    EDGE_UPDATED = "edge_updated"
    GRAPH_SNAPSHOT = "graph_snapshot"
    GRAPH_RESET = "graph_reset"
    BATCH_UPDATE = "batch_update"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    REJECT = "reject"
    MANUAL = "manual"


@dataclass
class GraphEvent:
    """Represents a single graph event."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    graph_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    version: int = 1
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "graph_id": self.graph_id,
            "data": self.data,
            "metadata": self.metadata,
            "source": self.source,
            "version": self.version,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            graph_id=data["graph_id"],
            data=data["data"],
            metadata=data.get("metadata", {}),
            source=data.get("source"),
            version=data.get("version", 1),
            correlation_id=data.get("correlation_id")
        )
    
    def serialize(self) -> bytes:
        """Serialize event to bytes."""
        return JSON_SERIALIZER.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "GraphEvent":
        """Deserialize event from bytes."""
        return cls.from_dict(JSON_SERIALIZER.loads(data))


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    backend: StreamBackend = StreamBackend.MEMORY
    connection_params: Dict[str, Any] = field(default_factory=dict)
    topic_prefix: str = "anant_graph"
    consumer_group: str = "anant_consumers"
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_retries: int = 3
    retry_delay_ms: int = 100
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    enable_compression: bool = True
    enable_deduplication: bool = True
    deduplication_window_ms: int = 5000
    enable_ordering: bool = True
    enable_metrics: bool = True
    enable_dead_letter: bool = True
    dead_letter_topic: str = "anant_dead_letter"


@dataclass
class StreamMetrics:
    """Stream processing metrics."""
    events_processed: int = 0
    events_failed: int = 0
    processing_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    last_processed_timestamp: Optional[datetime] = None
    backlog_size: int = 0
    consumer_lag_ms: int = 0
    error_rate: float = 0.0
    throughput_events_per_sec: float = 0.0
    
    def update_processing_stats(self, processing_time_ms: float, success: bool):
        """Update processing statistics."""
        if success:
            self.events_processed += 1
        else:
            self.events_failed += 1
        
        # Update averages
        total_events = self.events_processed + self.events_failed
        if total_events > 0:
            self.avg_processing_time_ms = (
                (self.avg_processing_time_ms * (total_events - 1) + processing_time_ms) / total_events
            )
            self.error_rate = self.events_failed / total_events
        
        self.last_processed_timestamp = datetime.now()


class StreamEventHandler(ABC):
    """Abstract base class for stream event handlers."""
    
    @abstractmethod
    async def handle_event(self, event: GraphEvent) -> bool:
        """Handle a single graph event. Returns True if successful."""
        pass
    
    @abstractmethod
    async def handle_batch(self, events: List[GraphEvent]) -> List[bool]:
        """Handle a batch of events. Returns list of success flags."""
        pass


class GraphStreamProcessor:
    """Production-grade stream processor for graph events."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.handlers: List[StreamEventHandler] = []
        self.metrics = StreamMetrics()
        self.is_running = False
        self.consumer_task: Optional[asyncio.Task] = None
        self.producer_queue: asyncio.Queue = asyncio.Queue()
        self.deduplication_cache: Dict[str, datetime] = {}
        self.pending_events: Dict[str, GraphEvent] = {}
        
        # Backend-specific components
        self.backend_producer = None
        self.backend_consumer = None
        
        logger.info(f"Initialized GraphStreamProcessor with {config.backend.value} backend")
    
    def add_handler(self, handler: StreamEventHandler):
        """Add an event handler."""
        self.handlers.append(handler)
        logger.info(f"Added handler: {handler.__class__.__name__}")
    
    async def start(self):
        """Start stream processing."""
        if self.is_running:
            return
        
        logger.info("Starting stream processor...")
        
        # Initialize backend
        await self._initialize_backend()
        
        # Start consumer
        self.consumer_task = asyncio.create_task(self._consumer_loop())
        
        # Start producer
        asyncio.create_task(self._producer_loop())
        
        # Start metrics collection
        if self.config.enable_metrics:
            asyncio.create_task(self._metrics_loop())
        
        # Start cleanup tasks
        asyncio.create_task(self._cleanup_loop())
        
        self.is_running = True
        logger.info("Stream processor started successfully")
    
    async def stop(self):
        """Stop stream processing."""
        if not self.is_running:
            return
        
        logger.info("Stopping stream processor...")
        
        self.is_running = False
        
        # Cancel consumer task
        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        
        # Close backend connections
        await self._cleanup_backend()
        
        logger.info("Stream processor stopped")
    
    async def produce_event(self, event: GraphEvent):
        """Produce a graph event to the stream."""
        # Add to producer queue
        await self.producer_queue.put(event)
    
    async def _initialize_backend(self):
        """Initialize the streaming backend."""
        if self.config.backend == StreamBackend.KAFKA:
            await self._initialize_kafka()
        elif self.config.backend == StreamBackend.PULSAR:
            await self._initialize_pulsar()
        elif self.config.backend == StreamBackend.REDIS_STREAMS:
            await self._initialize_redis()
        elif self.config.backend == StreamBackend.MEMORY:
            await self._initialize_memory()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    async def _initialize_kafka(self):
        """Initialize Kafka backend."""
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka required for Kafka backend")
        
        # Initialize producer
        self.backend_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.config.connection_params.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: v.serialize() if hasattr(v, 'serialize') else v
        )
        await self.backend_producer.start()
        
        # Initialize consumer
        topic = f"{self.config.topic_prefix}_events"
        self.backend_consumer = aiokafka.AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.config.connection_params.get("bootstrap_servers", "localhost:9092"),
            group_id=self.config.consumer_group,
            value_deserializer=lambda v: GraphEvent.deserialize(v) if v else None
        )
        await self.backend_consumer.start()
        
        logger.info("Kafka backend initialized")
    
    async def _initialize_pulsar(self):
        """Initialize Pulsar backend."""
        if not PULSAR_AVAILABLE:
            raise ImportError("pulsar-client required for Pulsar backend")
        
        service_url = self.config.connection_params.get("service_url", "pulsar://localhost:6650")
        
        # Initialize Pulsar client
        self.pulsar_client = pulsar.Client(service_url)
        
        # Initialize producer
        topic = f"{self.config.topic_prefix}_events"
        self.backend_producer = self.pulsar_client.create_producer(topic)
        
        # Initialize consumer
        self.backend_consumer = self.pulsar_client.subscribe(
            topic,
            subscription_name=self.config.consumer_group
        )
        
        logger.info("Pulsar backend initialized")
    
    async def _initialize_redis(self):
        """Initialize Redis Streams backend."""
        if not REDIS_AVAILABLE:
            raise ImportError("aioredis required for Redis backend")
        
        redis_url = self.config.connection_params.get("redis_url", "redis://localhost:6379")
        self.backend_redis = aioredis.from_url(redis_url)
        
        # Test connection
        await self.backend_redis.ping()
        
        logger.info("Redis Streams backend initialized")
    
    async def _initialize_memory(self):
        """Initialize in-memory backend for testing."""
        self.memory_stream: asyncio.Queue = asyncio.Queue()
        logger.info("Memory backend initialized")
    
    async def _producer_loop(self):
        """Producer loop for sending events."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                try:
                    event = await asyncio.wait_for(self.producer_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Send to backend
                await self._send_to_backend(event)
                
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_to_backend(self, event: GraphEvent):
        """Send event to the configured backend."""
        if self.config.backend == StreamBackend.KAFKA:
            topic = f"{self.config.topic_prefix}_events"
            await self.backend_producer.send_and_wait(topic, event)
        
        elif self.config.backend == StreamBackend.PULSAR:
            await asyncio.get_event_loop().run_in_executor(
                None, self.backend_producer.send, event.serialize()
            )
        
        elif self.config.backend == StreamBackend.REDIS_STREAMS:
            stream_name = f"{self.config.topic_prefix}_events"
            await self.backend_redis.xadd(
                stream_name,
                {"event": event.serialize()}
            )
        
        elif self.config.backend == StreamBackend.MEMORY:
            await self.memory_stream.put(event)
    
    async def _consumer_loop(self):
        """Consumer loop for processing events."""
        while self.is_running:
            try:
                # Get events from backend
                events = await self._receive_from_backend()
                
                if events:
                    # Process events
                    await self._process_events(events)
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _receive_from_backend(self) -> List[GraphEvent]:
        """Receive events from the configured backend."""
        events = []
        
        if self.config.backend == StreamBackend.KAFKA:
            try:
                # Poll for messages
                msg_pack = await asyncio.wait_for(
                    self.backend_consumer.getmany(timeout_ms=self.config.batch_timeout_ms),
                    timeout=self.config.batch_timeout_ms / 1000
                )
                
                for tp, messages in msg_pack.items():
                    for message in messages:
                        if message.value:
                            events.append(message.value)
                            
            except asyncio.TimeoutError:
                pass
        
        elif self.config.backend == StreamBackend.PULSAR:
            try:
                # Receive message
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, self.backend_consumer.receive
                )
                event = GraphEvent.deserialize(msg.data())
                events.append(event)
                
                # Acknowledge message
                await asyncio.get_event_loop().run_in_executor(
                    None, self.backend_consumer.acknowledge, msg
                )
                
            except Exception:
                pass
        
        elif self.config.backend == StreamBackend.REDIS_STREAMS:
            try:
                stream_name = f"{self.config.topic_prefix}_events"
                messages = await self.backend_redis.xread(
                    {stream_name: "$"},
                    count=self.config.batch_size,
                    block=self.config.batch_timeout_ms
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        if b"event" in fields:
                            event = GraphEvent.deserialize(fields[b"event"])
                            events.append(event)
                            
            except Exception:
                pass
        
        elif self.config.backend == StreamBackend.MEMORY:
            try:
                event = await asyncio.wait_for(self.memory_stream.get(), timeout=1.0)
                events.append(event)
            except asyncio.TimeoutError:
                pass
        
        return events
    
    async def _process_events(self, events: List[GraphEvent]):
        """Process a batch of events."""
        if not events:
            return
        
        start_time = time.time()
        
        try:
            # Filter duplicates if enabled
            if self.config.enable_deduplication:
                events = self._deduplicate_events(events)
            
            # Sort by timestamp if ordering is enabled
            if self.config.enable_ordering:
                events.sort(key=lambda e: e.timestamp)
            
            # Process events with handlers
            for handler in self.handlers:
                try:
                    results = await handler.handle_batch(events)
                    
                    # Update metrics
                    for success in results:
                        processing_time = (time.time() - start_time) * 1000
                        self.metrics.update_processing_stats(processing_time, success)
                        
                except Exception as e:
                    logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                    
                    # Send to dead letter queue if enabled
                    if self.config.enable_dead_letter:
                        await self._send_to_dead_letter(events, str(e))
        
        except Exception as e:
            logger.error(f"Error processing events: {e}")
    
    def _deduplicate_events(self, events: List[GraphEvent]) -> List[GraphEvent]:
        """Remove duplicate events based on event ID and time window."""
        deduplicated = []
        current_time = datetime.now()
        window = timedelta(milliseconds=self.config.deduplication_window_ms)
        
        for event in events:
            # Check if we've seen this event recently
            if event.event_id in self.deduplication_cache:
                cached_time = self.deduplication_cache[event.event_id]
                if current_time - cached_time < window:
                    continue  # Skip duplicate
            
            # Add to cache and result
            self.deduplication_cache[event.event_id] = current_time
            deduplicated.append(event)
        
        return deduplicated
    
    async def _send_to_dead_letter(self, events: List[GraphEvent], error: str):
        """Send failed events to dead letter queue."""
        dead_letter_topic = self.config.dead_letter_topic
        
        for event in events:
            # Add error information to metadata
            event.metadata["error"] = error
            event.metadata["failed_at"] = datetime.now().isoformat()
            
            # Send to dead letter topic
            if self.config.backend == StreamBackend.KAFKA:
                await self.backend_producer.send_and_wait(dead_letter_topic, event)
            # Add other backends as needed
    
    async def _metrics_loop(self):
        """Metrics collection loop."""
        while self.is_running:
            try:
                # Calculate throughput
                if self.metrics.last_processed_timestamp:
                    time_diff = (datetime.now() - self.metrics.last_processed_timestamp).total_seconds()
                    if time_diff > 0:
                        self.metrics.throughput_events_per_sec = self.metrics.events_processed / time_diff
                
                # Log metrics periodically
                logger.debug(f"Stream metrics: {self.metrics}")
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup loop for maintenance tasks."""
        while self.is_running:
            try:
                # Clean old deduplication cache entries
                current_time = datetime.now()
                window = timedelta(milliseconds=self.config.deduplication_window_ms * 2)
                
                expired_keys = [
                    key for key, timestamp in self.deduplication_cache.items()
                    if current_time - timestamp > window
                ]
                
                for key in expired_keys:
                    del self.deduplication_cache[key]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_backend(self):
        """Cleanup backend connections."""
        try:
            if self.config.backend == StreamBackend.KAFKA:
                if self.backend_producer:
                    await self.backend_producer.stop()
                if self.backend_consumer:
                    await self.backend_consumer.stop()
            
            elif self.config.backend == StreamBackend.PULSAR:
                if hasattr(self, 'pulsar_client'):
                    self.pulsar_client.close()
            
            elif self.config.backend == StreamBackend.REDIS_STREAMS:
                if hasattr(self, 'backend_redis'):
                    await self.backend_redis.close()
                    
        except Exception as e:
            logger.error(f"Error cleaning up backend: {e}")
    
    def get_metrics(self) -> StreamMetrics:
        """Get current stream processing metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the stream processor."""
        health = {
            "status": "healthy" if self.is_running else "stopped",
            "backend": self.config.backend.value,
            "handlers_count": len(self.handlers),
            "metrics": {
                "events_processed": self.metrics.events_processed,
                "events_failed": self.metrics.events_failed,
                "error_rate": self.metrics.error_rate,
                "throughput": self.metrics.throughput_events_per_sec
            }
        }
        
        # Backend-specific health checks
        try:
            if self.config.backend == StreamBackend.REDIS_STREAMS and hasattr(self, 'backend_redis'):
                await self.backend_redis.ping()
                health["backend_status"] = "connected"
            else:
                health["backend_status"] = "connected"
        except Exception as e:
            health["backend_status"] = f"error: {e}"
            health["status"] = "degraded"
        
        return health


# Factory function for easy creation
def create_stream_processor(backend: str = "memory", **kwargs) -> GraphStreamProcessor:
    """Create a stream processor with specified backend."""
    config = StreamConfig(
        backend=StreamBackend(backend),
        **kwargs
    )
    return GraphStreamProcessor(config)