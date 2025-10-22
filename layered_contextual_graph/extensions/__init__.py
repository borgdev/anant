"""
LCG Extensions Module
=====================

Extensions that integrate LCG with Anant's core capabilities:
- Streaming & event-driven (using anant.streaming)
- Machine learning (embeddings, similarity, clustering)
- Advanced reasoning (inference, contradiction detection)
"""

from .streaming_integration import (
    LayerEventAdapter,
    SuperpositionEventListener,
    enable_streaming,
    StreamingLayeredGraph
)

from .ml_integration import (
    EmbeddingLayer,
    EntityEmbedding,
    MLLayeredGraph,
    enable_ml
)

from .reasoning_integration import (
    InferenceRule,
    RuleType,
    ReasoningEngine,
    Contradiction,
    ReasoningLayeredGraph,
    enable_reasoning
)

__all__ = [
    # Streaming
    'LayerEventAdapter',
    'SuperpositionEventListener',
    'enable_streaming',
    'StreamingLayeredGraph',
    
    # ML
    'EmbeddingLayer',
    'EntityEmbedding',
    'MLLayeredGraph',
    'enable_ml',
    
    # Reasoning
    'InferenceRule',
    'RuleType',
    'ReasoningEngine',
    'Contradiction',
    'ReasoningLayeredGraph',
    'enable_reasoning',
]
