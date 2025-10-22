"""
Advanced Reasoning Integration for LCG
======================================

Integrates reasoning capabilities with LayeredContextualGraph:
- Rule-based inference across layers
- Probabilistic reasoning with Bayesian networks
- Contradiction detection and resolution
- Hierarchical reasoning (bottom-up and top-down)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..core import LayeredContextualGraph, Layer, Context, SuperpositionState, LayerType

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of inference rules"""
    FORWARD = "forward"          # If A then B
    BACKWARD = "backward"        # To prove B, prove A
    BIDIRECTIONAL = "bidirectional"  # A if and only if B
    PROBABILISTIC = "probabilistic"  # A implies B with probability p


@dataclass
class InferenceRule:
    """
    A rule for cross-layer inference.
    
    Examples:
        - If entity exists in physical layer, it may exist in semantic layer
        - If superposition collapses in layer A, update layer B
    """
    rule_id: str
    rule_type: RuleType
    source_layer: str
    target_layer: str
    condition: Callable[[Any], bool]  # Condition to check
    action: Callable[[Any], Any]      # Action to perform
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    
    def evaluate(self, entity_data: Any) -> bool:
        """Evaluate if rule condition is met"""
        if not self.active:
            return False
        try:
            return self.condition(entity_data)
        except Exception as e:
            logger.error(f"Rule evaluation error: {e}")
            return False
    
    def apply(self, entity_data: Any) -> Any:
        """Apply rule action"""
        if not self.active:
            return None
        try:
            return self.action(entity_data)
        except Exception as e:
            logger.error(f"Rule application error: {e}")
            return None


@dataclass
class Contradiction:
    """Represents a detected contradiction"""
    contradiction_id: str
    entity_id: str
    layer1: str
    layer2: str
    state1: Any
    state2: Any
    severity: float  # 0-1, how severe is the contradiction
    description: str
    detected_at: Any = field(default_factory=lambda: __import__('datetime').datetime.now())


class ReasoningEngine:
    """
    Reasoning engine for LayeredContextualGraph.
    
    Capabilities:
    - Rule-based inference
    - Contradiction detection
    - Belief propagation
    - Hierarchical reasoning
    """
    
    def __init__(self, lcg: LayeredContextualGraph):
        self.lcg = lcg
        self.rules: Dict[str, InferenceRule] = {}
        self.contradictions: List[Contradiction] = []
        
        # Reasoning cache
        self.inferred_facts: Dict[str, Set[Any]] = {}
        self.belief_network: Dict[str, Dict[str, float]] = {}  # entity -> {state: probability}
        
        logger.info(f"ReasoningEngine initialized for '{lcg.name}'")
    
    def add_rule(self, rule: InferenceRule):
        """Add inference rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.rule_id} ({rule.source_layer} -> {rule.target_layer})")
    
    def infer(
        self,
        entity_id: str,
        from_layer: str,
        to_layer: str,
        max_steps: int = 10
    ) -> List[Any]:
        """
        Perform inference from source layer to target layer.
        
        Returns inferred facts about entity in target layer.
        """
        if entity_id not in self.lcg.superposition_states:
            return []
        
        superpos = self.lcg.superposition_states[entity_id]
        
        # Get entity state in source layer
        if from_layer not in superpos.layer_states:
            return []
        
        source_state = superpos.layer_states[from_layer]
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.rules.values()
            if rule.source_layer == from_layer and rule.target_layer == to_layer
        ]
        
        inferred = []
        
        for rule in applicable_rules:
            if rule.evaluate(source_state):
                result = rule.apply(source_state)
                if result is not None:
                    inferred.append({
                        'fact': result,
                        'rule': rule.rule_id,
                        'confidence': rule.confidence
                    })
        
        return inferred
    
    def propagate_beliefs(
        self,
        entity_id: str,
        evidence: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Propagate beliefs across layers using Bayesian updating.
        
        Args:
            entity_id: Entity to update beliefs for
            evidence: Observed evidence {layer: state}
            
        Returns:
            Updated belief distribution {state: probability}
        """
        if entity_id not in self.lcg.superposition_states:
            return {}
        
        superpos = self.lcg.superposition_states[entity_id]
        
        # Initialize with quantum state probabilities
        if superpos.quantum_state:
            beliefs = dict(superpos.quantum_state.states)
        else:
            beliefs = {}
        
        # Update based on evidence
        for layer_name, observed_state in evidence.items():
            # Find rules connecting this layer to quantum states
            for rule in self.rules.values():
                if rule.source_layer == layer_name and rule.rule_type == RuleType.PROBABILISTIC:
                    # Apply Bayesian update (simplified)
                    for state in beliefs:
                        if rule.evaluate({'state': state, 'evidence': observed_state}):
                            beliefs[state] *= rule.confidence
        
        # Normalize
        total = sum(beliefs.values())
        if total > 0:
            beliefs = {k: v/total for k, v in beliefs.items()}
        
        # Store in belief network
        self.belief_network[entity_id] = beliefs
        
        return beliefs
    
    def detect_contradictions(
        self,
        entity_id: Optional[str] = None,
        severity_threshold: float = 0.5
    ) -> List[Contradiction]:
        """
        Detect contradictions in entity states across layers.
        
        A contradiction occurs when an entity has incompatible states
        in different layers according to defined rules.
        """
        contradictions = []
        
        entities_to_check = [entity_id] if entity_id else self.lcg.superposition_states.keys()
        
        for eid in entities_to_check:
            if eid not in self.lcg.superposition_states:
                continue
            
            superpos = self.lcg.superposition_states[eid]
            layer_states = superpos.layer_states
            
            # Check all pairs of layers
            layers = list(layer_states.keys())
            for i, layer1 in enumerate(layers):
                for layer2 in layers[i+1:]:
                    state1 = layer_states[layer1]
                    state2 = layer_states[layer2]
                    
                    # Check for contradictions using rules
                    contradiction = self._check_contradiction(
                        eid, layer1, layer2, state1, state2
                    )
                    
                    if contradiction and contradiction.severity >= severity_threshold:
                        contradictions.append(contradiction)
        
        self.contradictions.extend(contradictions)
        return contradictions
    
    def _check_contradiction(
        self,
        entity_id: str,
        layer1: str,
        layer2: str,
        state1: Any,
        state2: Any
    ) -> Optional[Contradiction]:
        """Check if two states contradict each other"""
        # Simple heuristic: check if states are incompatible
        # In a real implementation, this would use domain-specific rules
        
        if state1 == state2:
            return None  # No contradiction if states are the same
        
        # Check if there are rules that say these states are incompatible
        for rule in self.rules.values():
            if (rule.source_layer == layer1 and rule.target_layer == layer2) or \
               (rule.source_layer == layer2 and rule.target_layer == layer1):
                # Evaluate if states are contradictory
                try:
                    if rule.evaluate({'state1': state1, 'state2': state2}):
                        # States satisfy rule, no contradiction
                        return None
                except:
                    pass
        
        # If no rules validate compatibility, flag as potential contradiction
        severity = 0.7  # Default severity
        
        return Contradiction(
            contradiction_id=f"contra_{entity_id}_{layer1}_{layer2}",
            entity_id=entity_id,
            layer1=layer1,
            layer2=layer2,
            state1=state1,
            state2=state2,
            severity=severity,
            description=f"Incompatible states in {layer1} and {layer2}"
        )
    
    def resolve_contradiction(
        self,
        contradiction: Contradiction,
        strategy: str = "priority"
    ) -> bool:
        """
        Resolve a detected contradiction.
        
        Strategies:
            - priority: Keep state from higher-priority layer
            - merge: Attempt to merge states
            - collapse: Force quantum collapse to resolve
        """
        entity_id = contradiction.entity_id
        
        if entity_id not in self.lcg.superposition_states:
            return False
        
        superpos = self.lcg.superposition_states[entity_id]
        layer1 = contradiction.layer1
        layer2 = contradiction.layer2
        
        if strategy == "priority":
            # Use layer hierarchy - keep state from higher level
            level1 = self.lcg.layers[layer1].level if layer1 in self.lcg.layers else 0
            level2 = self.lcg.layers[layer2].level if layer2 in self.lcg.layers else 0
            
            if level1 > level2:
                # Layer1 has priority, remove layer2 state
                if layer2 in superpos.layer_states:
                    del superpos.layer_states[layer2]
            else:
                # Layer2 has priority, remove layer1 state
                if layer1 in superpos.layer_states:
                    del superpos.layer_states[layer1]
            
            logger.info(f"Resolved contradiction for {entity_id} using priority strategy")
            return True
        
        elif strategy == "collapse":
            # Force quantum collapse to resolve
            if superpos.quantum_state and not superpos.quantum_state.collapsed:
                self.lcg.observe(entity_id, collapse_quantum=True)
                logger.info(f"Resolved contradiction for {entity_id} by collapsing")
                return True
        
        return False
    
    def hierarchical_inference(
        self,
        entity_id: str,
        direction: str = "up"  # "up" or "down"
    ) -> Dict[str, Any]:
        """
        Perform hierarchical inference (bottom-up or top-down).
        
        Bottom-up: Infer higher-level concepts from lower-level data
        Top-down: Infer lower-level details from higher-level concepts
        """
        if direction == "up":
            return self._bottom_up_inference(entity_id)
        else:
            return self._top_down_inference(entity_id)
    
    def _bottom_up_inference(self, entity_id: str) -> Dict[str, Any]:
        """Infer higher-level concepts from lower-level data"""
        inferred = {}
        
        # Get layers sorted by level
        sorted_layers = sorted(
            self.lcg.layers.items(),
            key=lambda x: x[1].level
        )
        
        # Start from lowest level
        for i, (layer_name, layer) in enumerate(sorted_layers[:-1]):
            next_layer_name = sorted_layers[i+1][0]
            
            # Infer from current to next layer
            results = self.infer(entity_id, layer_name, next_layer_name)
            if results:
                inferred[f"{layer_name}_to_{next_layer_name}"] = results
        
        return inferred
    
    def _top_down_inference(self, entity_id: str) -> Dict[str, Any]:
        """Infer lower-level details from higher-level concepts"""
        inferred = {}
        
        # Get layers sorted by level (reversed for top-down)
        sorted_layers = sorted(
            self.lcg.layers.items(),
            key=lambda x: x[1].level,
            reverse=True
        )
        
        # Start from highest level
        for i, (layer_name, layer) in enumerate(sorted_layers[:-1]):
            next_layer_name = sorted_layers[i+1][0]
            
            # Infer from current to next layer
            results = self.infer(entity_id, layer_name, next_layer_name)
            if results:
                inferred[f"{layer_name}_to_{next_layer_name}"] = results
        
        return inferred


class ReasoningLayeredGraph(LayeredContextualGraph):
    """
    LayeredContextualGraph with advanced reasoning capabilities.
    
    Automatically performs inference and detects contradictions.
    """
    
    def __init__(
        self,
        name: str = "reasoning_layered_graph",
        quantum_enabled: bool = True,
        auto_detect_contradictions: bool = True,
        **kwargs
    ):
        super().__init__(name=name, quantum_enabled=quantum_enabled, **kwargs)
        
        self.reasoning_engine = ReasoningEngine(self)
        self.auto_detect_contradictions = auto_detect_contradictions
        
        logger.info(f"ReasoningLayeredGraph initialized: {name}")
    
    def add_inference_rule(
        self,
        rule_id: str,
        source_layer: str,
        target_layer: str,
        condition: Callable,
        action: Callable,
        rule_type: RuleType = RuleType.FORWARD,
        confidence: float = 1.0
    ):
        """Add an inference rule"""
        rule = InferenceRule(
            rule_id=rule_id,
            rule_type=rule_type,
            source_layer=source_layer,
            target_layer=target_layer,
            condition=condition,
            action=action,
            confidence=confidence
        )
        
        self.reasoning_engine.add_rule(rule)
    
    def infer_cross_layer(
        self,
        entity_id: str,
        from_layer: str,
        to_layer: str
    ) -> List[Any]:
        """Perform cross-layer inference"""
        return self.reasoning_engine.infer(entity_id, from_layer, to_layer)
    
    def check_consistency(self, entity_id: Optional[str] = None) -> List[Contradiction]:
        """Check for contradictions"""
        return self.reasoning_engine.detect_contradictions(entity_id)
    
    def create_superposition(self, entity_id: str, layer_states=None, quantum_states=None):
        """Create superposition with automatic contradiction detection"""
        superpos = super().create_superposition(entity_id, layer_states, quantum_states)
        
        # Auto-detect contradictions if enabled
        if self.auto_detect_contradictions and layer_states:
            contradictions = self.reasoning_engine.detect_contradictions(entity_id)
            if contradictions:
                logger.warning(f"Detected {len(contradictions)} contradictions for {entity_id}")
        
        return superpos


def enable_reasoning(lcg: LayeredContextualGraph, auto_detect: bool = True) -> ReasoningLayeredGraph:
    """
    Convert existing LayeredContextualGraph to ReasoningLayeredGraph.
    """
    reasoning_lcg = ReasoningLayeredGraph(
        name=lcg.name,
        quantum_enabled=lcg.quantum_enabled,
        auto_detect_contradictions=auto_detect
    )
    
    # Copy state
    reasoning_lcg.layers = lcg.layers
    reasoning_lcg.layer_hierarchy = lcg.layer_hierarchy
    reasoning_lcg.contexts = lcg.contexts
    reasoning_lcg.superposition_states = lcg.superposition_states
    reasoning_lcg.quantum_states = lcg.quantum_states
    
    logger.info(f"Enabled reasoning for LCG: {lcg.name}")
    return reasoning_lcg
