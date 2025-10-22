"""
Streaming Integration for LCG
==============================

Integrates LayeredContextualGraph with Anant's streaming infrastructure
for real-time event-driven layer synchronization.

Uses:
- anant.streaming.core.stream_processor.GraphStreamProcessor
- anant.streaming.core.event_store.EventStore
- anant.streaming.integration.StreamingFramework
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from anant.streaming import (
        GraphStreamProcessor,
        GraphEvent,
        EventType,
        StreamConfig,
        EventStore,
        create_memory_store,
        StreamingFramework,
        create_real_time_streaming
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logging.warning("Anant streaming not available")

from ..core import LayeredContextualGraph, Layer, Context, SuperpositionState

logger = logging.getLogger(__name__)


class LayerEventType:
    """Event types specific to LCG operations"""
    LAYER_ADDED = "layer_added"
    LAYER_REMOVED = "layer_removed"
    LAYER_UPDATED = "layer_updated"
    CONTEXT_ADDED = "context_added"
    CONTEXT_UPDATED = "context_updated"
    SUPERPOSITION_CREATED = "superposition_created"
    SUPERPOSITION_COLLAPSED = "superposition_collapsed"
    ENTITIES_ENTANGLED = "entities_entangled"
    CROSS_LAYER_QUERY = "cross_layer_query"
    HIERARCHY_PROPAGATED = "hierarchy_propagated"


@dataclass
class LayerEvent:
    """Event specific to layer operations"""
    event_id: str
    event_type: str
    timestamp: datetime
    lcg_name: str
    layer_name: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_graph_event(self, graph_id: str) -> 'GraphEvent':
        """Convert to Anant GraphEvent for processing"""
        if not STREAMING_AVAILABLE:
            raise RuntimeError("Streaming not available")
        
        # Map LCG events to graph events
        event_type_map = {
            LayerEventType.LAYER_ADDED: EventType.NODE_ADDED,
            LayerEventType.LAYER_REMOVED: EventType.NODE_REMOVED,
            LayerEventType.SUPERPOSITION_CREATED: EventType.NODE_UPDATED,
            LayerEventType.SUPERPOSITION_COLLAPSED: EventType.NODE_UPDATED,
        }
        
        graph_event_type = event_type_map.get(self.event_type, EventType.NODE_UPDATED)
        
        return GraphEvent(
            event_id=self.event_id,
            event_type=graph_event_type,
            timestamp=self.timestamp,
            graph_id=graph_id,
            data={
                'lcg_event_type': self.event_type,
                'layer_name': self.layer_name,
                **self.data
            },
            metadata=self.metadata
        )


class LayerEventAdapter:
    """
    Adapter that connects LCG operations to Anant's streaming infrastructure.
    
    Emits events when:
    - Layers are added/removed
    - Superpositions are created/collapsed
    - Entities are entangled
    - Cross-layer queries execute
    """
    
    def __init__(
        self,
        lcg: LayeredContextualGraph,
        event_store: Optional['EventStore'] = None,
        enable_async: bool = True
    ):
        self.lcg = lcg
        self.lcg_name = lcg.name
        self.event_store = event_store or (create_memory_store() if STREAMING_AVAILABLE else None)
        self.enable_async = enable_async
        self.listeners: List[Callable] = []
        
        # Track events
        self.event_count = 0
        self.events_by_type: Dict[str, int] = {}
        
        logger.info(f"LayerEventAdapter initialized for '{lcg.name}'")
    
    def emit_event(self, event_type: str, layer_name: Optional[str], data: Dict[str, Any]):
        """Emit a layer event"""
        event = LayerEvent(
            event_id=f"{self.lcg_name}_{self.event_count}_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=datetime.now(),
            lcg_name=self.lcg_name,
            layer_name=layer_name,
            data=data
        )
        
        self.event_count += 1
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1
        
        # Store event (EventStore is async, so we skip for sync context)
        # In production with async support, use: await self.event_store.store_event(graph_event)
        # For now, we just track events in memory via the adapter itself
        
        # Notify listeners (synchronous only)
        for listener in self.listeners:
            try:
                # Only call synchronous listeners
                if not asyncio.iscoroutinefunction(listener):
                    listener(event)
            except Exception as e:
                logger.error(f"Listener error: {e}")
        
        logger.debug(f"Emitted event: {event_type} for layer={layer_name}")
    
    def on_layer_added(self, layer_name: str, layer: Layer):
        """Called when a layer is added"""
        self.emit_event(
            LayerEventType.LAYER_ADDED,
            layer_name,
            {
                'layer_type': layer.layer_type.value,
                'level': layer.level,
                'parent_layer': layer.parent_layer,
                'weight': layer.weight
            }
        )
    
    def on_layer_removed(self, layer_name: str):
        """Called when a layer is removed"""
        self.emit_event(
            LayerEventType.LAYER_REMOVED,
            layer_name,
            {'removed': True}
        )
    
    def on_superposition_created(self, entity_id: str, layer_states: Dict[str, Any]):
        """Called when a superposition is created"""
        self.emit_event(
            LayerEventType.SUPERPOSITION_CREATED,
            None,
            {
                'entity_id': entity_id,
                'layers': list(layer_states.keys()),
                'num_states': len(layer_states)
            }
        )
    
    def on_superposition_collapsed(self, entity_id: str, collapsed_state: str, layer: str):
        """Called when a superposition collapses"""
        self.emit_event(
            LayerEventType.SUPERPOSITION_COLLAPSED,
            layer,
            {
                'entity_id': entity_id,
                'collapsed_to': collapsed_state,
                'layer': layer
            }
        )
    
    def on_entities_entangled(self, entity1: str, entity2: str):
        """Called when entities are entangled"""
        self.emit_event(
            LayerEventType.ENTITIES_ENTANGLED,
            None,
            {
                'entity1': entity1,
                'entity2': entity2
            }
        )
    
    def subscribe(self, listener: Callable):
        """Subscribe to layer events"""
        self.listeners.append(listener)
        logger.info(f"Subscribed listener: {listener.__name__}")
    
    def unsubscribe(self, listener: Callable):
        """Unsubscribe from layer events"""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics"""
        return {
            'total_events': self.event_count,
            'events_by_type': self.events_by_type,
            'num_listeners': len(self.listeners),
            'event_store_enabled': self.event_store is not None
        }


class SuperpositionEventListener:
    """
    Listens to superposition events and propagates changes across layers.
    
    When an entity's state changes in one layer, this listener can:
    - Update dependent layers
    - Maintain consistency across superpositions
    - Trigger cascading updates
    """
    
    def __init__(
        self,
        lcg: LayeredContextualGraph,
        propagation_strategy: str = "immediate"  # immediate, batched, lazy
    ):
        self.lcg = lcg
        self.propagation_strategy = propagation_strategy
        self.pending_updates: List[Dict] = []
        
    async def handle_event(self, event: LayerEvent):
        """Handle incoming layer event"""
        if event.event_type == LayerEventType.SUPERPOSITION_COLLAPSED:
            await self._handle_collapse(event)
        elif event.event_type == LayerEventType.LAYER_UPDATED:
            await self._handle_layer_update(event)
    
    async def _handle_collapse(self, event: LayerEvent):
        """Handle superposition collapse"""
        entity_id = event.data.get('entity_id')
        collapsed_state = event.data.get('collapsed_to')
        layer_name = event.data.get('layer')
        
        if not entity_id or entity_id not in self.lcg.superposition_states:
            return
        
        # Propagate collapse to dependent layers
        superpos = self.lcg.superposition_states[entity_id]
        
        # Find dependent layers (child layers)
        if layer_name and layer_name in self.lcg.layers:
            layer = self.lcg.layers[layer_name]
            for child_layer_name in layer.child_layers:
                # Update child layer states based on parent collapse
                logger.info(f"Propagating collapse to child layer: {child_layer_name}")
                # Trigger child layer update
                if child_layer_name in superpos.layer_states:
                    pass  # Add custom propagation logic here
    
    async def _handle_layer_update(self, event: LayerEvent):
        """Handle layer update"""
        layer_name = event.layer_name
        if layer_name and layer_name in self.lcg.layers:
            # Invalidate caches
            self.lcg._layer_cache.clear()


class StreamingLayeredGraph(LayeredContextualGraph):
    """
    LayeredContextualGraph with integrated streaming capabilities.
    
    Automatically emits events for all operations and supports real-time
    subscription to layer changes.
    
    Examples:
        >>> slcg = StreamingLayeredGraph(name="streaming_kg", quantum_enabled=True)
        >>> 
        >>> # Subscribe to events
        >>> def on_layer_change(event):
        ...     print(f"Layer changed: {event.event_type}")
        >>> 
        >>> slcg.event_adapter.subscribe(on_layer_change)
        >>> 
        >>> # Operations automatically emit events
        >>> slcg.add_layer("physical", physical_hg, LayerType.PHYSICAL, level=0)
        >>> # Event emitted: LAYER_ADDED
    """
    
    def __init__(
        self,
        name: str = "streaming_layered_graph",
        quantum_enabled: bool = True,
        stream_backend: str = "memory",
        enable_event_store: bool = False,  # Disabled by default to avoid async issues
        **kwargs
    ):
        super().__init__(name=name, quantum_enabled=quantum_enabled, **kwargs)
        
        # Initialize streaming components (event_store disabled for sync contexts)
        self.event_store = None  # Can be enabled later with async support
        
        self.event_adapter = LayerEventAdapter(
            lcg=self,
            event_store=None,  # Don't use EventStore in sync context
            enable_async=False  # Sync-only for now
        )
        
        self.listener = SuperpositionEventListener(
            lcg=self,
            propagation_strategy="immediate"
        )
        
        # Note: listener.handle_event is async, so we don't subscribe it in sync mode
        # In async context, use: self.event_adapter.subscribe(self.listener.handle_event)
        
        logger.info(f"StreamingLayeredGraph initialized: {name}")
    
    def add_layer(self, name: str, hypergraph: Any, *args, **kwargs):
        """Add layer with event emission"""
        super().add_layer(name, hypergraph, *args, **kwargs)
        
        # Emit event
        layer = self.layers[name]
        self.event_adapter.on_layer_added(name, layer)
    
    def remove_layer(self, name: str) -> bool:
        """Remove layer with event emission"""
        result = super().remove_layer(name)
        
        if result:
            self.event_adapter.on_layer_removed(name)
        
        return result
    
    def create_superposition(self, entity_id: str, layer_states=None, quantum_states=None):
        """Create superposition with event emission"""
        superpos = super().create_superposition(entity_id, layer_states, quantum_states)
        
        self.event_adapter.on_superposition_created(entity_id, layer_states or {})
        
        return superpos
    
    def observe(self, entity_id: str, layer=None, context=None, collapse_quantum=True):
        """Observe with event emission on collapse"""
        result = super().observe(entity_id, layer, context, collapse_quantum)
        
        if collapse_quantum and entity_id in self.superposition_states:
            superpos = self.superposition_states[entity_id]
            if superpos.quantum_state and superpos.quantum_state.collapsed:
                self.event_adapter.on_superposition_collapsed(
                    entity_id,
                    superpos.quantum_state.collapsed_state,
                    layer or "unknown"
                )
        
        return result
    
    def entangle_entities(self, entity_id1: str, entity_id2: str, correlation_strength: float = 1.0) -> bool:
        """Entangle entities with event emission"""
        result = super().entangle_entities(entity_id1, entity_id2, correlation_strength)
        
        if result:
            self.event_adapter.on_entities_entangled(entity_id1, entity_id2)
        
        return result
    
    def subscribe_to_entity(self, entity_id: str, callback: Callable):
        """
        Subscribe to changes for a specific entity.
        
        Callback is called whenever the entity's superposition changes.
        """
        def entity_filter(event: LayerEvent):
            if 'entity_id' in event.data and event.data['entity_id'] == entity_id:
                callback(event)
        
        self.event_adapter.subscribe(entity_filter)
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        stats = self.event_adapter.get_stats()
        # EventStore doesn't have synchronous len(), skip for now
        return stats


def enable_streaming(lcg: LayeredContextualGraph, stream_backend: str = "memory") -> LayerEventAdapter:
    """
    Enable streaming on an existing LayeredContextualGraph.
    
    Args:
        lcg: Existing LayeredContextualGraph
        stream_backend: Streaming backend to use
        
    Returns:
        LayerEventAdapter instance
    """
    # Create adapter without EventStore (sync mode)
    adapter = LayerEventAdapter(lcg=lcg, event_store=None, enable_async=False)
    
    logger.info(f"Enabled streaming for LCG: {lcg.name}")
    return adapter
