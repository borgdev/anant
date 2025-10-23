"""
Layered Contextual Graph Module
===============================

Advanced layered contextual graph analysis with quantum-inspired operations.

This module provides:
- Multi-layer contextual graph structures
- Quantum-inspired graph operations
- Advanced analytics and reasoning
- Production-ready streaming capabilities
- Extended graph functionalities

Core Features:
- Revolutionary layered graph architecture
- Context-aware graph operations
- Quantum algorithms for graph analysis
- Real-time streaming and processing
- Advanced ML integration
"""

# Core layered contextual graph components
from .core import LayeredContextualGraph, LayerType, ContextType

# Extended modules - lazy import to avoid heavy dependencies
def _get_extensions():
    """Lazy import of extensions module"""
    try:
        from . import extensions
        return extensions
    except ImportError as e:
        print(f"Warning: Layered contextual graph extensions not available: {e}")
        return None

def _get_quantum():
    """Lazy import of quantum module"""
    try:
        from . import quantum
        return quantum
    except ImportError as e:
        print(f"Warning: Quantum module not available: {e}")
        return None

def _get_analytics():
    """Lazy import of analytics module"""
    try:
        from . import analytics
        return analytics
    except ImportError as e:
        print(f"Warning: Analytics module not available: {e}")
        return None

def _get_production():
    """Lazy import of production module"""
    try:
        from . import production
        return production
    except ImportError as e:
        print(f"Warning: Production module not available: {e}")
        return None

def __getattr__(name):
    """Provide lazy access to sub-modules"""
    if name == 'extensions':
        return _get_extensions()
    elif name == 'quantum':
        return _get_quantum()
    elif name == 'analytics':
        return _get_analytics()
    elif name == 'production':
        return _get_production()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'LayeredContextualGraph',
    'LayerType',
    'ContextType',
    'extensions',
    'quantum',
    'analytics',
    'production',
]