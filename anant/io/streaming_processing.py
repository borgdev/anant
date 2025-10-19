"""
Streaming Data Processing for Anant Library

Provides real-time hypergraph updates and streaming capabilities:
- Real-time edge/node addition and removal
- Streaming data ingestion from various sources
- Event-driven hypergraph updates
- Temporal hypergraph evolution tracking
- Memory-efficient streaming algorithms
- WebSocket and message queue integration
- Batch processing for high-throughput scenarios

This module enables dynamic hypergraph analysis with live data streams
and supports real-time applications requiring immediate updates.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import polars as pl
from enum import Enum
import threading
import queue

from ..classes.hypergraph import Hypergraph


class StreamEventType(Enum):
    """Types of streaming events"""
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    UPDATE_WEIGHT = "update_weight"
    BATCH_UPDATE = "batch_update"
    SNAPSHOT = "snapshot"


@dataclass
class StreamEvent:
    """Represents a streaming event for hypergraph updates"""
    event_type: StreamEventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create from dictionary"""
        return cls(
            event_type=StreamEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data.get("metadata")
        )


@dataclass
class StreamingConfig:
    """Configuration for streaming operations"""
    buffer_size: int = 1000
    batch_size: int = 100
    flush_interval: float = 1.0  # seconds
    enable_temporal_tracking: bool = True
    max_history_size: int = 10000
    enable_compression: bool = False
    checkpoint_interval: int = 1000
    enable_metrics: bool = True
    memory_threshold_mb: int = 500


class StreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.is_running = False
    
    @abstractmethod
    async def start(self) -> None:
        """Start the stream processor"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the stream processor"""
        pass
    
    @abstractmethod
    async def process_event(self, event: StreamEvent) -> None:
        """Process a single stream event"""
        pass


class HypergraphStreamProcessor(StreamProcessor):
    """Real-time hypergraph stream processor"""
    
    def __init__(self, hg: Hypergraph, config: Optional[StreamingConfig] = None):
        super().__init__(config or StreamingConfig())
        self.hypergraph = hg
        self.event_buffer = deque(maxlen=self.config.buffer_size)
        self.event_history = deque(maxlen=self.config.max_history_size)
        self.subscribers = []
        self.metrics = {
            "events_processed": 0,
            "nodes_added": 0,
            "nodes_removed": 0,
            "edges_added": 0,
            "edges_removed": 0,
            "last_update": None,
            "processing_rate": 0.0
        }
        self._last_flush = time.time()
        self._processing_task = None
        self._checkpoint_counter = 0
    
    async def start(self) -> None:
        """Start the stream processor"""
        if not self.is_running:
            self.is_running = True
            self._processing_task = asyncio.create_task(self._process_loop())
    
    async def stop(self) -> None:
        """Stop the stream processor"""
        if self.is_running:
            self.is_running = False
            if self._processing_task:
                await self._processing_task
                self._processing_task = None
    
    async def add_event(self, event: StreamEvent) -> None:
        """Add event to processing buffer"""
        self.event_buffer.append(event)
        
        # Immediate processing for critical events
        if event.event_type in [StreamEventType.SNAPSHOT]:
            await self.process_event(event)
    
    async def process_event(self, event: StreamEvent) -> None:
        """Process a single stream event"""
        try:
            if event.event_type == StreamEventType.ADD_NODE:
                await self._handle_add_node(event)
            elif event.event_type == StreamEventType.REMOVE_NODE:
                await self._handle_remove_node(event)
            elif event.event_type == StreamEventType.ADD_EDGE:
                await self._handle_add_edge(event)
            elif event.event_type == StreamEventType.REMOVE_EDGE:
                await self._handle_remove_edge(event)
            elif event.event_type == StreamEventType.UPDATE_WEIGHT:
                await self._handle_update_weight(event)
            elif event.event_type == StreamEventType.BATCH_UPDATE:
                await self._handle_batch_update(event)
            elif event.event_type == StreamEventType.SNAPSHOT:
                await self._handle_snapshot(event)
            
            # Update metrics
            self.metrics["events_processed"] += 1
            self.metrics["last_update"] = datetime.now()
            
            # Store in history if enabled
            if self.config.enable_temporal_tracking:
                self.event_history.append(event)
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            # Checkpoint if needed
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= self.config.checkpoint_interval:
                await self._create_checkpoint()
                self._checkpoint_counter = 0
                
        except Exception as e:
            print(f"Error processing event: {e}")
    
    async def _handle_add_node(self, event: StreamEvent) -> None:
        """Handle add node event"""
        node_id = event.data.get("node_id")
        properties = event.data.get("properties", {})
        
        if node_id and node_id not in self.hypergraph.nodes:
            # Note: Hypergraph nodes are added implicitly when edges are added
            # For explicit node addition, we could extend the Hypergraph class
            self.metrics["nodes_added"] += 1
    
    async def _handle_remove_node(self, event: StreamEvent) -> None:
        """Handle remove node event"""
        node_id = event.data.get("node_id")
        
        if node_id and node_id in self.hypergraph.nodes:
            # Remove all edges containing this node
            edges_to_remove = []
            for edge_id in self.hypergraph.edges:
                edge_nodes = self.hypergraph.incidences.data.filter(
                    pl.col("edge_id") == edge_id
                )["node_id"].to_list()
                if node_id in edge_nodes:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                self.hypergraph.remove_edge(edge_id)
            
            self.metrics["nodes_removed"] += 1
    
    async def _handle_add_edge(self, event: StreamEvent) -> None:
        """Handle add edge event"""
        edge_id = event.data.get("edge_id")
        nodes = event.data.get("nodes", [])
        weight = event.data.get("weight", 1.0)
        properties = event.data.get("properties", {})
        
        if edge_id and nodes:
            self.hypergraph.add_edge(edge_id, nodes, weight=weight)
            self.metrics["edges_added"] += 1
    
    async def _handle_remove_edge(self, event: StreamEvent) -> None:
        """Handle remove edge event"""
        edge_id = event.data.get("edge_id")
        
        if edge_id and edge_id in self.hypergraph.edges:
            self.hypergraph.remove_edge(edge_id)
            self.metrics["edges_removed"] += 1
    
    async def _handle_update_weight(self, event: StreamEvent) -> None:
        """Handle weight update event"""
        edge_id = event.data.get("edge_id")
        new_weight = event.data.get("weight")
        
        if edge_id and new_weight is not None and edge_id in self.hypergraph.edges:
            # Get edge nodes
            edge_nodes = self.hypergraph.incidences.data.filter(
                pl.col("edge_id") == edge_id
            )["node_id"].to_list()
            
            # Remove and re-add with new weight
            self.hypergraph.remove_edge(edge_id)
            self.hypergraph.add_edge(edge_id, edge_nodes, weight=new_weight)
    
    async def _handle_batch_update(self, event: StreamEvent) -> None:
        """Handle batch update event"""
        updates = event.data.get("updates", [])
        
        for update in updates:
            sub_event = StreamEvent(
                event_type=StreamEventType(update["event_type"]),
                timestamp=event.timestamp,
                data=update["data"],
                metadata=update.get("metadata")
            )
            await self.process_event(sub_event)
    
    async def _handle_snapshot(self, event: StreamEvent) -> None:
        """Handle snapshot event (full hypergraph replacement)"""
        snapshot_data = event.data.get("hypergraph_data")
        
        if snapshot_data:
            # Clear current hypergraph
            for edge_id in list(self.hypergraph.edges):
                self.hypergraph.remove_edge(edge_id)
            
            # Rebuild from snapshot
            incidences = snapshot_data.get("incidences", [])
            edge_groups = defaultdict(list)
            edge_weights = {}
            
            for inc in incidences:
                edge_id = inc["edge_id"]
                node_id = inc["node_id"]
                weight = inc.get("weight", 1.0)
                
                edge_groups[edge_id].append(node_id)
                edge_weights[edge_id] = weight
            
            for edge_id, nodes in edge_groups.items():
                weight = edge_weights.get(edge_id, 1.0)
                self.hypergraph.add_edge(edge_id, nodes, weight=weight)
    
    async def _process_loop(self) -> None:
        """Main processing loop"""
        while self.is_running:
            try:
                # Process buffered events
                events_to_process = []
                while self.event_buffer and len(events_to_process) < self.config.batch_size:
                    events_to_process.append(self.event_buffer.popleft())
                
                # Process events
                start_time = time.time()
                for event in events_to_process:
                    await self.process_event(event)
                
                # Update processing rate
                if events_to_process:
                    processing_time = time.time() - start_time
                    self.metrics["processing_rate"] = len(events_to_process) / processing_time
                
                # Check if flush interval elapsed
                current_time = time.time()
                if current_time - self._last_flush >= self.config.flush_interval:
                    await self._flush_updates()
                    self._last_flush = current_time
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _flush_updates(self) -> None:
        """Flush any pending updates"""
        # This could be used for batched database writes, etc.
        pass
    
    async def _create_checkpoint(self) -> None:
        """Create a checkpoint of the current state"""
        if self.config.enable_temporal_tracking:
            checkpoint_event = StreamEvent(
                event_type=StreamEventType.SNAPSHOT,
                data={
                    "hypergraph_data": {
                        "incidences": [
                            row for row in self.hypergraph.incidences.data.iter_rows(named=True)
                        ]
                    },
                    "metrics": self.metrics.copy()
                },
                metadata={"checkpoint": True}
            )
            self.event_history.append(checkpoint_event)
    
    async def _notify_subscribers(self, event: StreamEvent) -> None:
        """Notify all subscribers of the event"""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event, self.hypergraph)
                else:
                    subscriber(event, self.hypergraph)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable[[StreamEvent, Hypergraph], None]) -> None:
        """Subscribe to hypergraph updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[StreamEvent, Hypergraph], None]) -> None:
        """Unsubscribe from hypergraph updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return self.metrics.copy()
    
    def get_history(self, start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None) -> List[StreamEvent]:
        """Get event history within time range"""
        if not self.config.enable_temporal_tracking:
            return []
        
        history = list(self.event_history)
        
        if start_time:
            history = [e for e in history if e.timestamp >= start_time]
        
        if end_time:
            history = [e for e in history if e.timestamp <= end_time]
        
        return history


class WebSocketStreamer:
    """WebSocket-based streaming for real-time updates"""
    
    def __init__(self, processor: HypergraphStreamProcessor, port: int = 8765):
        self.processor = processor
        self.port = port
        self.server = None
        self.connected_clients = set()
    
    async def start_server(self) -> None:
        """Start WebSocket server"""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required for WebSocket streaming. Install with: pip install websockets")
        
        self.server = await websockets.serve(
            self._handle_client,
            "localhost",
            self.port
        )
        
        # Subscribe to processor updates
        self.processor.subscribe(self._broadcast_update)
        
        print(f"WebSocket server started on ws://localhost:{self.port}")
    
    async def stop_server(self) -> None:
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        
        # Unsubscribe from processor
        self.processor.unsubscribe(self._broadcast_update)
    
    async def _handle_client(self, websocket, path):
        """Handle new WebSocket client"""
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                # Parse incoming message as stream event
                try:
                    event_data = json.loads(message)
                    event = StreamEvent.from_dict(event_data)
                    await self.processor.add_event(event)
                except Exception as e:
                    await websocket.send(json.dumps({
                        "error": f"Invalid event format: {e}"
                    }))
        except Exception as e:
            print(f"Client disconnected: {e}")
        finally:
            self.connected_clients.remove(websocket)
    
    async def _broadcast_update(self, event: StreamEvent, hg: Hypergraph) -> None:
        """Broadcast update to all connected clients"""
        if self.connected_clients:
            message = json.dumps({
                "event": event.to_dict(),
                "hypergraph_stats": {
                    "num_nodes": hg.num_nodes,
                    "num_edges": hg.num_edges
                }
            })
            
            disconnected_clients = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except Exception:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients


class MessageQueueStreamer:
    """Message queue-based streaming (Redis, RabbitMQ, etc.)"""
    
    def __init__(self, processor: HypergraphStreamProcessor, 
                 queue_type: str = "redis", **connection_params):
        self.processor = processor
        self.queue_type = queue_type
        self.connection_params = connection_params
        self.connection = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start message queue consumer"""
        if self.queue_type == "redis":
            await self._start_redis()
        elif self.queue_type == "rabbitmq":
            await self._start_rabbitmq()
        else:
            raise ValueError(f"Unsupported queue type: {self.queue_type}")
        
        self.is_running = True
    
    async def stop(self) -> None:
        """Stop message queue consumer"""
        self.is_running = False
        if self.connection:
            if self.queue_type == "redis":
                await self.connection.close()
            elif self.queue_type == "rabbitmq":
                await self.connection.close()
    
    async def _start_redis(self) -> None:
        """Start Redis consumer"""
        try:
            import aioredis
        except ImportError:
            raise ImportError("aioredis required for Redis streaming. Install with: pip install aioredis")
        
        self.connection = await aioredis.create_redis_pool(
            f"redis://{self.connection_params.get('host', 'localhost')}:{self.connection_params.get('port', 6379)}"
        )
        
        # Start consuming task
        asyncio.create_task(self._redis_consumer())
    
    async def _redis_consumer(self) -> None:
        """Redis message consumer"""
        channel_name = self.connection_params.get('channel', 'hypergraph_events')
        
        while self.is_running:
            try:
                # Use BLPOP for blocking pop
                result = await self.connection.blpop(channel_name, timeout=1)
                if result:
                    _, message = result
                    event_data = json.loads(message.decode('utf-8'))
                    event = StreamEvent.from_dict(event_data)
                    await self.processor.add_event(event)
            except Exception as e:
                print(f"Redis consumer error: {e}")
                await asyncio.sleep(1)
    
    async def _start_rabbitmq(self) -> None:
        """Start RabbitMQ consumer"""
        try:
            import aio_pika
        except ImportError:
            raise ImportError("aio-pika required for RabbitMQ streaming. Install with: pip install aio-pika")
        
        connection_url = f"amqp://{self.connection_params.get('username', 'guest')}:{self.connection_params.get('password', 'guest')}@{self.connection_params.get('host', 'localhost')}:{self.connection_params.get('port', 5672)}/"
        
        self.connection = await aio_pika.connect_robust(connection_url)
        channel = await self.connection.channel()
        
        queue_name = self.connection_params.get('queue', 'hypergraph_events')
        queue = await channel.declare_queue(queue_name, durable=True)
        
        # Start consuming
        await queue.consume(self._rabbitmq_callback)
    
    async def _rabbitmq_callback(self, message) -> None:
        """RabbitMQ message callback"""
        try:
            async with message.process():
                event_data = json.loads(message.body.decode('utf-8'))
                event = StreamEvent.from_dict(event_data)
                await self.processor.add_event(event)
        except Exception as e:
            print(f"RabbitMQ callback error: {e}")


class FileStreamMonitor:
    """Monitor file changes for streaming updates"""
    
    def __init__(self, processor: HypergraphStreamProcessor, 
                 file_path: str, format_type: str = "json"):
        self.processor = processor
        self.file_path = file_path
        self.format_type = format_type
        self.is_monitoring = False
        self._monitor_task = None
        self._last_modified = None
    
    async def start_monitoring(self) -> None:
        """Start file monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop file monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self._monitor_task:
                await self._monitor_task
                self._monitor_task = None
    
    async def _monitor_loop(self) -> None:
        """File monitoring loop"""
        from pathlib import Path
        
        file_path = Path(self.file_path)
        
        while self.is_monitoring:
            try:
                if file_path.exists():
                    current_modified = file_path.stat().st_mtime
                    
                    if self._last_modified is None:
                        self._last_modified = current_modified
                    elif current_modified > self._last_modified:
                        # File was modified, process it
                        await self._process_file_update()
                        self._last_modified = current_modified
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                print(f"File monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_file_update(self) -> None:
        """Process file update"""
        try:
            if self.format_type == "json":
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # List of events
                    for event_data in data:
                        event = StreamEvent.from_dict(event_data)
                        await self.processor.add_event(event)
                else:
                    # Single event
                    event = StreamEvent.from_dict(data)
                    await self.processor.add_event(event)
            
        except Exception as e:
            print(f"Error processing file update: {e}")


# Convenience functions
async def create_streaming_hypergraph(initial_hg: Optional[Hypergraph] = None,
                                     config: Optional[StreamingConfig] = None) -> HypergraphStreamProcessor:
    """Create a streaming hypergraph processor"""
    hg = initial_hg or Hypergraph()
    processor = HypergraphStreamProcessor(hg, config)
    await processor.start()
    return processor


def create_websocket_streamer(processor: HypergraphStreamProcessor, 
                             port: int = 8765) -> WebSocketStreamer:
    """Create WebSocket streamer"""
    return WebSocketStreamer(processor, port)


def create_message_queue_streamer(processor: HypergraphStreamProcessor,
                                 queue_type: str = "redis",
                                 **connection_params) -> MessageQueueStreamer:
    """Create message queue streamer"""
    return MessageQueueStreamer(processor, queue_type, **connection_params)


def create_file_monitor(processor: HypergraphStreamProcessor,
                       file_path: str, format_type: str = "json") -> FileStreamMonitor:
    """Create file stream monitor"""
    return FileStreamMonitor(processor, file_path, format_type)