"""
Factory module for anant library

Factory methods for creating hypergraph data structures from various input formats.

Enhanced Capabilities (NEW):
- Direct Parquet file loading with lazy evaluation
- Multi-modal SetSystem creation and cross-analysis
- Streaming processing for massive datasets
- Enhanced validation with multiple levels
- Automatic optimization and performance monitoring
"""

from .setsystem_factory import SetSystemFactory

# Enhanced SetSystem capabilities
from .enhanced_setsystems import (
    ParquetSetSystem,
    MultiModalSetSystem, 
    StreamingSetSystem,
    SetSystemType
)

from .enhanced_validation import (
    EnhancedSetSystemValidator,
    ValidationLevel,
    ValidationResult
)

from .enhanced_integration import (
    EnhancedSetSystemFactory,
    create_parquet_setsystem,
    create_multimodal_setsystem,
    create_streaming_setsystem,
    get_enhanced_factory
)

__all__ = [
    # Standard factory
    "SetSystemFactory",
    
    # Enhanced SetSystem types
    "ParquetSetSystem",
    "MultiModalSetSystem", 
    "StreamingSetSystem",
    "SetSystemType",
    
    # Enhanced validation
    "EnhancedSetSystemValidator",
    "ValidationLevel",
    "ValidationResult",
    
    # Enhanced factory and convenience functions
    "EnhancedSetSystemFactory",
    "create_parquet_setsystem",
    "create_multimodal_setsystem", 
    "create_streaming_setsystem",
    "get_enhanced_factory"
]