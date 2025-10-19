"""
I/O operations for anant library

Provides comprehensive I/O functionality including native Parquet support,
compression options, schema preservation, multi-file dataset handling, and
optimized data loading/saving workflows for hypergraph data.
"""

from .parquet_io import AnantIO, save_hypergraph, load_hypergraph
from .advanced_io import (
    AdvancedAnantIO,
    CompressionType,
    FileFormat,
    IOConfiguration,
    DatasetMetadata,
    LoadResult,
    SaveResult
)

# Legacy compatibility
__all__ = [
    # Legacy AnantIO
    "AnantIO",
    "save_hypergraph", 
    "load_hypergraph",
    # Advanced I/O system
    "AdvancedAnantIO",
    "CompressionType",
    "FileFormat", 
    "IOConfiguration",
    "DatasetMetadata",
    "LoadResult",
    "SaveResult"
]

# Default instance for convenience
advanced_io = AdvancedAnantIO()