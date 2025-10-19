"""
Anant Hypergraph Library

A powerful, modern hypergraph analysis library built for performance and scalability.
Provides enhanced functionality beyond traditional graph libraries with support for:
- Advanced hypergraph operations and analysis
- Enhanced SetSystem implementations (Parquet, MultiModal, Streaming)
- Advanced property management with correlation analysis
- Comprehensive validation and debugging tools
- Jupyter integration with interactive widgets
- High-performance data processing with Polars

Version: 0.1.0
Author: Anant Development Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Anant Development Team"
__license__ = "MIT"

# Core classes
from .classes.hypergraph import Hypergraph
from .classes.incidence_store import IncidenceStore
from .classes.property_store import PropertyStore
from .classes.advanced_properties import AdvancedPropertyStore

# Enhanced SetSystems  
from .factory.enhanced_setsystems import (
    EnhancedSetSystemFactory,
    ParquetSetSystem,
    MultiModalSetSystem, 
    StreamingSetSystem
)

# I/O and utilities
from .io.parquet_io import AnantIO, save_hypergraph, load_hypergraph
from .io import (
    # Enhanced file formats
    quick_import, quick_export, enhanced_formats,
    # Database connectivity
    create_sqlite_manager, create_postgresql_manager, create_mongodb_manager,
    # Streaming processing  
    create_streaming_hypergraph, StreamEvent, StreamEventType,
    # Data transformation
    quick_clean, quick_quality_check, quick_convert
)
from .utils.decorators import performance_monitor, cache_result
from .utils.extras import setup_polars_config

# Advanced Analysis Algorithms
try:
    from . import algorithms
    _HAS_ALGORITHMS = True
except ImportError as e:
    print(f"Warning: Advanced algorithms not available: {e}")
    _HAS_ALGORITHMS = False

# Analysis modules
try:
    from . import analysis
    _HAS_ANALYSIS = True
except ImportError as e:
    print(f"Warning: Analysis modules not available: {e}")
    _HAS_ANALYSIS = False

# Governance modules
try:
    from . import governance
    _HAS_GOVERNANCE = True
except ImportError as e:
    print(f"Warning: Governance modules not available: {e}")
    _HAS_GOVERNANCE = False

# Development tools
from .debugging_tools import (
    debug_hypergraph,
    start_profiling,
    stop_profiling,
    PerformanceProfiler,
    DataIntegrityValidator
)

# Jupyter integration (conditional import) - temporarily disabled
# try:
#     from .jupyter_integration import (
#         setup_jupyter_integration,
#         explore_hypergraph,
#         JupyterIntegration,
#         HypergraphExplorer
#     )
#     JUPYTER_AVAILABLE = True
# except ImportError:
#     JUPYTER_AVAILABLE = False
JUPYTER_AVAILABLE = False

# Version check for dependencies
def check_dependencies():
    """Check if all required dependencies are available"""
    
    deps = {
        'polars': '>=0.18.0',
        'numpy': '>=1.20.0', 
        'pyarrow': '>=10.0.0',
        'psutil': '>=5.8.0'
    }
    
    missing_deps = []
    
    for dep, version in deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep}{version}")
    
    if missing_deps:
        print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join([f'"{dep}"' for dep in missing_deps]))
        return False
    
    return True

# Auto-setup
def setup():
    """Setup Anant library for optimal performance"""
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("❌ Dependency check failed")
        return False
    
    # Setup Polars configuration
    try:
        setup_polars_config()
        print("✅ Polars configuration optimized")
    except Exception as e:
        print(f"⚠️ Polars setup warning: {e}")
    
    # Setup Jupyter integration if available (temporarily disabled)
    # if JUPYTER_AVAILABLE:
    #     try:
    #         setup_jupyter_integration()
    #         print("✅ Jupyter integration enabled")
    #     except Exception as e:
    #         print(f"⚠️ Jupyter setup warning: {e}")
    
    print("🚀 Anant library ready!")
    return True

# Convenience imports
__all__ = [
    # Core classes
    'Hypergraph',
    'IncidenceStore', 
    'PropertyStore',
    'AdvancedPropertyStore',
    
    # Enhanced SetSystems
    'EnhancedSetSystemFactory',
    'ParquetSetSystem',
    'MultiModalSetSystem',
    'StreamingSetSystem',
    
    # I/O functionality
    'AnantIO',
    'save_hypergraph',
    'load_hypergraph',
    'quick_import',
    'quick_export',
    'enhanced_formats',
    'create_sqlite_manager',
    'create_postgresql_manager',
    'create_mongodb_manager',
    'create_streaming_hypergraph',
    'StreamEvent',
    'StreamEventType',
    'quick_clean',
    'quick_quality_check',
    'quick_convert',
    
    # Utilities
    'performance_monitor',
    'cache_result', 
    'setup_polars_config',
    
    # Development tools
    'debug_hypergraph',
    'start_profiling',
    'stop_profiling',
    'PerformanceProfiler',
    'DataIntegrityValidator',
    
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Modules (if available)
    'algorithms',
    'analysis', 
    'governance',
]# Add Jupyter exports if available (temporarily disabled)
# if JUPYTER_AVAILABLE:
#     __all__.extend([
#         "setup_jupyter_integration",
#         "explore_hypergraph", 
#         "JupyterIntegration",
#         "HypergraphExplorer"
#     ])

# Library information
def info():
    """Display library information"""
    print(f"""
🔗 Anant Hypergraph Library v{__version__}
==========================================

📦 Core Features:
  ✓ Advanced Hypergraph Operations
  ✓ Enhanced SetSystem Types (Parquet, MultiModal, Streaming)
  ✓ Advanced Property Management
  ✓ High-Performance Data Processing
  ✓ Comprehensive Validation Framework
  ✓ Development & Debugging Tools
  ✓ Advanced I/O & Integration (JSON, CSV, GraphML, HDF5, Databases)
  ✓ Real-time Streaming & WebSocket Support
  ✓ Data Quality Assessment & ETL Pipelines

🔧 Optional Features:
  {'✓' if JUPYTER_AVAILABLE else '✗'} Jupyter Integration & Widgets
  
📚 Documentation: https://github.com/anant-dev/anant
📧 Support: anant-dev@example.com
🐛 Issues: https://github.com/anant-dev/anant/issues
    """)