"""
Layered Contextual Graph Implementation
=======================================

Quantum-inspired multi-layered graph with contextual superposition.
Extends Anant's core Hypergraph class.

Concepts:
- Layers: Hierarchical levels of abstraction
- Context: Situational state that influences interpretation
- Superposition: Multiple simultaneous states (quantum-inspired)
- Quantum-ready: Prepared for future quantum database integration
"""

from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum
import logging
import sys
from pathlib import Path
import uuid
import json

# Add Anant to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import polars as pl
    import numpy as np
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

# Import Anant core Hypergraph
try:
    from anant.classes.hypergraph.core.hypergraph import Hypergraph as AnantHypergraph
    ANANT_AVAILABLE = True
except ImportError:
    ANANT_AVAILABLE = False
    logging.warning("Anant core library not available. Using standalone mode.")
    class AnantHypergraph:
        """Placeholder when Anant is not available"""
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'hypergraph')

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of layers in the contextual graph"""
    PHYSICAL = "physical"          # Base physical/data layer
    LOGICAL = "logical"            # Logical/semantic layer
    CONCEPTUAL = "conceptual"      # Abstract conceptual layer
    TEMPORAL = "temporal"          # Time-based layer
    SPATIAL = "spatial"            # Spatial/geometric layer
    SEMANTIC = "semantic"          # Semantic meaning layer
    CONTEXT = "context"            # Pure context layer
    QUANTUM = "quantum"            # Quantum state layer


class ContextType(Enum):
    """Types of contextual information"""
    TEMPORAL = "temporal"          # Time-based context
    SPATIAL = "spatial"            # Location-based context
    SOCIAL = "social"              # Social/relational context
    DOMAIN = "domain"              # Domain-specific context
    USER = "user"                  # User-specific context
    ENVIRONMENTAL = "environmental" # Environmental context
    SITUATIONAL = "situational"    # Situational context


@dataclass
class QuantumState:
    """
    Quantum-inspired state representation.
    
    Represents a superposition of multiple possible states with probabilities.
    Quantum-ready for future quantum database integration.
    """
    state_id: str
    states: Dict[str, float] = field(default_factory=dict)  # state_name -> probability
    collapsed: bool = False
    collapsed_state: Optional[str] = None
    entangled_with: Set[str] = field(default_factory=set)
    measurement_history: List[Dict[str, Any]] = field(default_factory=list)
    coherence: float = 1.0  # Quantum coherence (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_state(self, state_name: str, probability: float):
        """Add a state to the superposition"""
        if self.collapsed:
            raise ValueError("Cannot add state to collapsed quantum state")
        self.states[state_name] = probability
        self._normalize()
    
    def _normalize(self):
        """Normalize probabilities to sum to 1"""
        total = sum(self.states.values())
        if total > 0:
            self.states = {k: v/total for k, v in self.states.items()}
    
    def collapse(self, measurement_context: Optional[Dict] = None) -> str:
        """
        Collapse the superposition to a single state.
        
        Quantum measurement causes wave function collapse.
        """
        if self.collapsed:
            return self.collapsed_state
        
        # Weighted random choice based on probabilities
        states = list(self.states.keys())
        probabilities = list(self.states.values())
        
        collapsed_state = np.random.choice(states, p=probabilities)
        
        self.collapsed = True
        self.collapsed_state = collapsed_state
        self.coherence = 0.0
        
        # Record measurement
        self.measurement_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': collapsed_state,
            'context': measurement_context or {}
        })
        
        return collapsed_state
    
    def get_dominant_state(self) -> Tuple[str, float]:
        """Get the state with highest probability"""
        if self.collapsed:
            return self.collapsed_state, 1.0
        if not self.states:
            return None, 0.0
        return max(self.states.items(), key=lambda x: x[1])
    
    def entangle(self, other_state_id: str):
        """Create quantum entanglement with another state"""
        self.entangled_with.add(other_state_id)


@dataclass
class SuperpositionState:
    """
    Represents an entity or relationship in superposition across layers.
    
    In quantum mechanics, superposition allows a system to be in multiple states
    simultaneously until observed. Here, entities exist in multiple layers/contexts
    at once until queried.
    """
    entity_id: str
    layer_states: Dict[str, Any] = field(default_factory=dict)  # layer_name -> state
    context_states: Dict[str, Any] = field(default_factory=dict)  # context -> state
    quantum_state: Optional[QuantumState] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_layer_state(self, layer_name: str, state: Any):
        """Add state for a specific layer"""
        self.layer_states[layer_name] = state
    
    def add_context_state(self, context_name: str, state: Any):
        """Add state for a specific context"""
        self.context_states[context_name] = state
    
    def observe(self, layer: Optional[str] = None, context: Optional[str] = None) -> Any:
        """
        Observe the entity in a specific layer/context.
        
        Quantum observation: collapses superposition to specific state.
        """
        if layer and layer in self.layer_states:
            return self.layer_states[layer]
        if context and context in self.context_states:
            return self.context_states[context]
        
        # If quantum state exists, collapse it
        if self.quantum_state and not self.quantum_state.collapsed:
            return self.quantum_state.collapse({'layer': layer, 'context': context})
        
        return None


@dataclass
class Context:
    """
    Contextual information that influences graph interpretation.
    
    Context provides situational awareness for graph queries and reasoning.
    """
    name: str
    context_type: ContextType
    parameters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    priority: int = 0  # Higher priority contexts override lower
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    spatial_range: Optional[Dict[str, Any]] = None
    applicable_layers: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_applicable(self, layer_name: str, timestamp: Optional[datetime] = None) -> bool:
        """Check if context is applicable to given layer and time"""
        if not self.active:
            return False
        
        if self.applicable_layers and layer_name not in self.applicable_layers:
            return False
        
        if self.temporal_range and timestamp:
            start, end = self.temporal_range
            if not (start <= timestamp <= end):
                return False
        
        return True


@dataclass
class Layer:
    """
    A layer in the contextual graph hierarchy.
    
    Layers represent different levels of abstraction or different dimensional views
    of the same underlying data.
    """
    name: str
    layer_type: LayerType
    hypergraph: Any  # Anant Hypergraph instance
    level: int = 0  # Hierarchical level (0=bottom/physical)
    parent_layer: Optional[str] = None
    child_layers: Set[str] = field(default_factory=set)
    contexts: Dict[str, Context] = field(default_factory=dict)
    weight: float = 1.0
    active: bool = True
    quantum_enabled: bool = False  # Future quantum DB integration
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_context(self, context: Context):
        """Add context to this layer"""
        self.contexts[context.name] = context
    
    def get_active_contexts(self, timestamp: Optional[datetime] = None) -> List[Context]:
        """Get all active contexts for this layer"""
        return [
            ctx for ctx in self.contexts.values()
            if ctx.is_applicable(self.name, timestamp)
        ]


class LayeredContextualGraph(AnantHypergraph):
    """
    Quantum-Inspired Layered Contextual Graph
    
    Extends Anant's core Hypergraph with:
    - Multiple hierarchical layers
    - Contextual awareness
    - Quantum superposition of states
    - Cross-layer reasoning
    - Quantum-ready architecture for future DB integration
    
    Architecture:
    - Each layer is an Anant Hypergraph
    - Entities exist in superposition across layers
    - Contexts influence interpretation
    - Quantum states allow multiple simultaneous possibilities
    
    Examples:
        >>> lcg = LayeredContextualGraph(name="knowledge_graph")
        >>> lcg.add_layer("physical", physical_hg, LayerType.PHYSICAL, level=0)
        >>> lcg.add_layer("semantic", semantic_hg, LayerType.SEMANTIC, level=1)
        >>> lcg.add_context("temporal", Context(...))
        >>> result = lcg.query_with_context(entity_id, context="temporal")
    """
    
    def __init__(
        self,
        name: str = "layered_contextual_graph",
        setsystem: Optional[Union[dict, Any]] = None,
        data: Optional[pl.DataFrame] = None,
        properties: Optional[dict] = None,
        quantum_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize Layered Contextual Graph.
        
        Args:
            name: Name identifier
            setsystem: Optional underlying set system (for base Hypergraph)
            data: Optional DataFrame data (for base Hypergraph)
            properties: Optional properties dict (for base Hypergraph)
            quantum_enabled: Enable quantum superposition features
            **kwargs: Additional arguments passed to base Hypergraph
        """
        # Initialize base Anant Hypergraph
        if ANANT_AVAILABLE:
            super().__init__(
                setsystem=setsystem,
                data=data,
                properties=properties,
                name=name,
                **kwargs
            )
        else:
            self.name = name
        
        # Layered contextual graph specific attributes
        self.layers: Dict[str, Layer] = {}  # layer_name -> Layer
        self.layer_hierarchy: Dict[int, Set[str]] = defaultdict(set)  # level -> layer_names
        self.contexts: Dict[str, Context] = {}  # context_name -> Context
        self.superposition_states: Dict[str, SuperpositionState] = {}  # entity_id -> state
        self.quantum_states: Dict[str, QuantumState] = {}  # state_id -> QuantumState
        
        # Configuration
        self.quantum_enabled = quantum_enabled
        self.max_layers = kwargs.get('max_layers', 10)
        self.coherence_threshold = kwargs.get('coherence_threshold', 0.5)
        
        # Caching
        self._layer_cache: Dict[str, Any] = {}
        self._context_cache: Dict[str, Any] = {}
        
        logger.info(f"Initialized LayeredContextualGraph: {name} (Quantum: {quantum_enabled}, Anant: {ANANT_AVAILABLE})")
    
    def add_layer(
        self,
        name: str,
        hypergraph: Any,
        layer_type: LayerType,
        level: int = 0,
        parent_layer: Optional[str] = None,
        weight: float = 1.0,
        quantum_enabled: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a layer to the contextual graph.
        
        Args:
            name: Layer name
            hypergraph: Anant Hypergraph instance for this layer
            layer_type: Type of layer
            level: Hierarchical level (0=bottom)
            parent_layer: Name of parent layer (if any)
            weight: Layer importance weight
            quantum_enabled: Enable quantum features for this layer
            metadata: Additional metadata
        """
        if name in self.layers:
            raise ValueError(f"Layer '{name}' already exists")
        
        layer = Layer(
            name=name,
            layer_type=layer_type,
            hypergraph=hypergraph,
            level=level,
            parent_layer=parent_layer,
            weight=weight,
            quantum_enabled=quantum_enabled or self.quantum_enabled,
            metadata=metadata or {}
        )
        
        self.layers[name] = layer
        self.layer_hierarchy[level].add(name)
        
        # Update parent-child relationships
        if parent_layer and parent_layer in self.layers:
            self.layers[parent_layer].child_layers.add(name)
        
        # Invalidate cache
        self._layer_cache.clear()
        
        logger.info(f"Added layer '{name}' (type={layer_type.value}, level={level})")
    
    def remove_layer(self, name: str) -> bool:
        """Remove a layer from the graph"""
        if name not in self.layers:
            return False
        
        layer = self.layers[name]
        
        # Update hierarchy
        self.layer_hierarchy[layer.level].discard(name)
        
        # Update parent
        if layer.parent_layer and layer.parent_layer in self.layers:
            self.layers[layer.parent_layer].child_layers.discard(name)
        
        # Remove from layers
        del self.layers[name]
        
        # Invalidate cache
        self._layer_cache.clear()
        
        logger.info(f"Removed layer '{name}'")
        return True
    
    def add_context(
        self,
        name: str,
        context_type: ContextType,
        parameters: Optional[Dict[str, Any]] = None,
        applicable_layers: Optional[Set[str]] = None,
        priority: int = 0,
        temporal_range: Optional[Tuple[datetime, datetime]] = None
    ) -> None:
        """
        Add a context to the graph.
        
        Contexts provide situational awareness and influence how the graph
        is interpreted and queried.
        """
        context = Context(
            name=name,
            context_type=context_type,
            parameters=parameters or {},
            priority=priority,
            temporal_range=temporal_range,
            applicable_layers=applicable_layers or set()
        )
        
        self.contexts[name] = context
        
        # Add to applicable layers
        for layer_name in (applicable_layers or self.layers.keys()):
            if layer_name in self.layers:
                self.layers[layer_name].add_context(context)
        
        logger.info(f"Added context '{name}' (type={context_type.value})")
    
    def create_superposition(
        self,
        entity_id: str,
        layer_states: Optional[Dict[str, Any]] = None,
        quantum_states: Optional[Dict[str, float]] = None
    ) -> SuperpositionState:
        """
        Create a superposition state for an entity.
        
        The entity exists simultaneously in multiple layers/states until observed.
        This is quantum-inspired: the entity's true state is indeterminate until
        measurement (query).
        """
        superposition = SuperpositionState(
            entity_id=entity_id,
            layer_states=layer_states or {}
        )
        
        # Create quantum state if provided
        if quantum_states and self.quantum_enabled:
            quantum_state = QuantumState(
                state_id=f"qs_{entity_id}_{uuid.uuid4().hex[:8]}",
                states=quantum_states
            )
            superposition.quantum_state = quantum_state
            self.quantum_states[quantum_state.state_id] = quantum_state
        
        self.superposition_states[entity_id] = superposition
        
        logger.debug(f"Created superposition for entity '{entity_id}'")
        return superposition
    
    def observe(
        self,
        entity_id: str,
        layer: Optional[str] = None,
        context: Optional[str] = None,
        collapse_quantum: bool = True
    ) -> Any:
        """
        Observe an entity in a specific layer/context.
        
        Quantum observation: If entity is in superposition, this collapses
        it to a definite state based on the observation context.
        
        Args:
            entity_id: Entity to observe
            layer: Layer to observe in
            context: Context for observation
            collapse_quantum: Whether to collapse quantum superposition
            
        Returns:
            Observed state of the entity
        """
        if entity_id not in self.superposition_states:
            return None
        
        superposition = self.superposition_states[entity_id]
        
        # Apply context if specified
        active_context = None
        if context and context in self.contexts:
            active_context = self.contexts[context]
        
        # Observe in specific layer
        if layer:
            return superposition.observe(layer=layer, context=context)
        
        # If no layer specified, get dominant quantum state
        if superposition.quantum_state and collapse_quantum:
            return superposition.quantum_state.collapse({'context': context})
        
        return superposition.layer_states
    
    def query_across_layers(
        self,
        entity_id: str,
        layers: Optional[List[str]] = None,
        context: Optional[str] = None,
        aggregate: str = "union"
    ) -> Dict[str, Any]:
        """
        Query an entity across multiple layers.
        
        Args:
            entity_id: Entity to query
            layers: Layers to query (None = all layers)
            context: Context for query
            aggregate: How to combine results (union, intersection, weighted)
            
        Returns:
            Aggregated results across layers
        """
        if layers is None:
            layers = list(self.layers.keys())
        
        results = {}
        
        for layer_name in layers:
            if layer_name not in self.layers:
                continue
            
            layer = self.layers[layer_name]
            
            # Check if context is applicable
            if context:
                ctx = self.contexts.get(context)
                if ctx and not ctx.is_applicable(layer_name):
                    continue
            
            # Query in this layer
            state = self.observe(entity_id, layer=layer_name, context=context)
            if state is not None:
                results[layer_name] = {
                    'state': state,
                    'layer_type': layer.layer_type.value,
                    'level': layer.level,
                    'weight': layer.weight
                }
        
        return results
    
    def get_layer_hierarchy(self) -> Dict[int, List[str]]:
        """Get layers organized by hierarchical level"""
        return {
            level: sorted(layers)
            for level, layers in self.layer_hierarchy.items()
        }
    
    def propagate_up(
        self,
        entity_id: str,
        from_layer: str,
        to_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Propagate entity information up the layer hierarchy.
        
        Bottom-up reasoning: physical → logical → conceptual
        """
        if from_layer not in self.layers:
            return {}
        
        start_layer = self.layers[from_layer]
        start_level = start_layer.level
        
        if to_level is None:
            to_level = max(self.layer_hierarchy.keys())
        
        results = {}
        
        # Propagate through each level
        for level in range(start_level + 1, to_level + 1):
            if level not in self.layer_hierarchy:
                continue
            
            for layer_name in self.layer_hierarchy[level]:
                layer = self.layers[layer_name]
                
                # Check if this layer is a child of our path
                if layer.parent_layer == from_layer or level == start_level + 1:
                    state = self.observe(entity_id, layer=layer_name)
                    if state:
                        results[layer_name] = state
                        from_layer = layer_name  # Continue from here
        
        return results
    
    def propagate_down(
        self,
        entity_id: str,
        from_layer: str,
        to_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Propagate entity information down the layer hierarchy.
        
        Top-down reasoning: conceptual → logical → physical
        """
        if from_layer not in self.layers:
            return {}
        
        start_layer = self.layers[from_layer]
        start_level = start_layer.level
        
        if to_level is None:
            to_level = 0
        
        results = {}
        
        # Propagate through each level
        for level in range(start_level - 1, to_level - 1, -1):
            if level not in self.layer_hierarchy:
                continue
            
            for layer_name in self.layer_hierarchy[level]:
                layer = self.layers[layer_name]
                
                # Check if this is a parent-child relationship
                if layer_name in start_layer.child_layers or from_layer in layer.child_layers:
                    state = self.observe(entity_id, layer=layer_name)
                    if state:
                        results[layer_name] = state
        
        return results
    
    def entangle_entities(
        self,
        entity_id1: str,
        entity_id2: str,
        correlation_strength: float = 1.0
    ) -> bool:
        """
        Create quantum entanglement between two entities.
        
        Entangled entities have correlated states: observing one affects the other.
        This is quantum-ready for future quantum database integration.
        """
        if not self.quantum_enabled:
            logger.warning("Quantum features not enabled")
            return False
        
        # Ensure both entities have quantum states
        for eid in [entity_id1, entity_id2]:
            if eid not in self.superposition_states:
                self.create_superposition(eid)
            
            superpos = self.superposition_states[eid]
            if not superpos.quantum_state:
                superpos.quantum_state = QuantumState(
                    state_id=f"qs_{eid}_{uuid.uuid4().hex[:8]}",
                    metadata={'correlation_strength': correlation_strength}
                )
        
        # Create entanglement
        qs1 = self.superposition_states[entity_id1].quantum_state
        qs2 = self.superposition_states[entity_id2].quantum_state
        
        qs1.entangle(qs2.state_id)
        qs2.entangle(qs1.state_id)
        
        logger.info(f"Entangled entities '{entity_id1}' and '{entity_id2}'")
        return True
    
    def get_quantum_coherence(self, entity_id: str) -> float:
        """
        Get quantum coherence for an entity.
        
        Coherence measures how "quantum" the state is (1.0 = fully quantum, 0.0 = classical).
        Quantum-ready metric for future quantum DB.
        """
        if entity_id not in self.superposition_states:
            return 0.0
        
        superpos = self.superposition_states[entity_id]
        if not superpos.quantum_state:
            return 0.0
        
        return superpos.quantum_state.coherence
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of the layered contextual graph"""
        return {
            'name': self.name,
            'num_layers': len(self.layers),
            'num_contexts': len(self.contexts),
            'num_superpositions': len(self.superposition_states),
            'num_quantum_states': len(self.quantum_states),
            'quantum_enabled': self.quantum_enabled,
            'layers_by_level': {
                level: len(layers)
                for level, layers in self.layer_hierarchy.items()
            },
            'layer_types': {
                layer_type.value: sum(
                    1 for layer in self.layers.values()
                    if layer.layer_type == layer_type
                )
                for layer_type in LayerType
            },
            'context_types': {
                ctx_type.value: sum(
                    1 for ctx in self.contexts.values()
                    if ctx.context_type == ctx_type
                )
                for ctx_type in ContextType
            },
            'anant_integrated': ANANT_AVAILABLE
        }
    
    def __repr__(self) -> str:
        return (f"LayeredContextualGraph(name='{self.name}', "
                f"layers={len(self.layers)}, contexts={len(self.contexts)}, "
                f"quantum={self.quantum_enabled})")
