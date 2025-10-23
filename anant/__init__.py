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

# Enhanced SetSystems - make lazy to avoid heavy imports
def _get_enhanced_setsystems():
    """Lazy import of enhanced setsystems"""
    from .factory.enhanced_setsystems import (
        EnhancedSetSystemFactory,
        ParquetSetSystem,
        MultiModalSetSystem, 
        StreamingSetSystem
    )
    return {
        'EnhancedSetSystemFactory': EnhancedSetSystemFactory,
        'ParquetSetSystem': ParquetSetSystem,
        'MultiModalSetSystem': MultiModalSetSystem,
        'StreamingSetSystem': StreamingSetSystem
    }

# I/O and utilities - make lazy to avoid psutil, database drivers, etc.
def _get_io_modules():
    """Lazy import of I/O modules"""
    from .io.parquet_io import AnantIO, save_hypergraph, load_hypergraph
    from .io import (
        quick_import, quick_export, enhanced_formats,
        create_sqlite_manager, create_postgresql_manager, create_mongodb_manager,
        create_streaming_hypergraph, StreamEvent, StreamEventType,
        quick_clean, quick_quality_check, quick_convert
    )
    return {
        'AnantIO': AnantIO,
        'save_hypergraph': save_hypergraph,
        'load_hypergraph': load_hypergraph,
        'quick_import': quick_import,
        'quick_export': quick_export,
        'enhanced_formats': enhanced_formats,
        'create_sqlite_manager': create_sqlite_manager,
        'create_postgresql_manager': create_postgresql_manager,
        'create_mongodb_manager': create_mongodb_manager,
        'create_streaming_hypergraph': create_streaming_hypergraph,
        'StreamEvent': StreamEvent,
        'StreamEventType': StreamEventType,
        'quick_clean': quick_clean,
        'quick_quality_check': quick_quality_check,
        'quick_convert': quick_convert
    }

# Only import lightweight utilities
from .utils.decorators import performance_monitor, cache_result
from .utils.extras import setup_polars_config

# Create lazy I/O functions
def AnantIO(*args, **kwargs):
    """Lazy AnantIO wrapper"""
    io_modules = _get_io_modules()
    return io_modules['AnantIO'](*args, **kwargs)

def save_hypergraph(*args, **kwargs):
    """Lazy save_hypergraph wrapper"""
    io_modules = _get_io_modules()
    return io_modules['save_hypergraph'](*args, **kwargs)

def load_hypergraph(*args, **kwargs):
    """Lazy load_hypergraph wrapper"""
    io_modules = _get_io_modules()
    return io_modules['load_hypergraph'](*args, **kwargs)

# More lazy I/O wrappers...
def quick_import(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['quick_import'](*args, **kwargs)

def quick_export(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['quick_export'](*args, **kwargs)

def enhanced_formats(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['enhanced_formats'](*args, **kwargs)

def create_sqlite_manager(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['create_sqlite_manager'](*args, **kwargs)

def create_postgresql_manager(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['create_postgresql_manager'](*args, **kwargs)

def create_mongodb_manager(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['create_mongodb_manager'](*args, **kwargs)

def create_streaming_hypergraph(*args, **kwargs):
    io_modules = _get_io_modules()
    return io_modules['create_streaming_hypergraph'](*args, **kwargs)

# Lazy enhanced setsystem classes
class EnhancedSetSystemFactory:
    """Lazy wrapper for EnhancedSetSystemFactory"""
    def __new__(cls, *args, **kwargs):
        setsystems = _get_enhanced_setsystems()
        return setsystems['EnhancedSetSystemFactory'](*args, **kwargs)

class ParquetSetSystem:
    """Lazy wrapper for ParquetSetSystem"""
    def __new__(cls, *args, **kwargs):
        setsystems = _get_enhanced_setsystems()
        return setsystems['ParquetSetSystem'](*args, **kwargs)

class MultiModalSetSystem:
    """Lazy wrapper for MultiModalSetSystem"""
    def __new__(cls, *args, **kwargs):
        setsystems = _get_enhanced_setsystems()
        return setsystems['MultiModalSetSystem'](*args, **kwargs)

class StreamingSetSystem:
    """Lazy wrapper for StreamingSetSystem"""
    def __new__(cls, *args, **kwargs):
        setsystems = _get_enhanced_setsystems()
        return setsystems['StreamingSetSystem'](*args, **kwargs)

# LAZY IMPORTS FOR PERFORMANCE OPTIMIZATION
# Advanced Analysis Algorithms - Lazy loaded to avoid 2000+ module cascade
_HAS_ALGORITHMS = True
_HAS_ANALYSIS = True  
_HAS_GOVERNANCE = True

def _get_algorithms():
    """Lazy import of algorithms module"""
    try:
        from . import algorithms
        return algorithms
    except ImportError as e:
        print(f"Warning: Advanced algorithms not available: {e}")
        return None

def _get_analysis():
    """Lazy import of analysis module"""
    try:
        from . import analysis
        return analysis
    except ImportError as e:
        print(f"Warning: Analysis modules not available: {e}")
        return None

def _get_governance():
    """Lazy import of governance module"""
    try:
        from . import governance
        return governance
    except ImportError as e:
        print(f"Warning: Governance modules not available: {e}")
        return None

def _get_debugging_tools():
    """Lazy import of debugging tools"""
    from .debugging_tools import (
        debug_hypergraph,
        start_profiling,
        stop_profiling,
        PerformanceProfiler,
        DataIntegrityValidator
    )
    return {
        'debug_hypergraph': debug_hypergraph,
        'start_profiling': start_profiling,
        'stop_profiling': stop_profiling,
        'PerformanceProfiler': PerformanceProfiler,
        'DataIntegrityValidator': DataIntegrityValidator
    }

# Create lazy accessors
class LazyModuleAccessor:
    """Provides lazy access to heavy modules"""
    
    @property
    def algorithms(self):
        return _get_algorithms()
    
    @property  
    def analysis(self):
        return _get_analysis()
        
    @property
    def governance(self):
        return _get_governance()

# Global lazy accessor
_lazy = LazyModuleAccessor()

# Make debugging tools available on-demand
def debug_hypergraph(*args, **kwargs):
    """Debug hypergraph - lazy loaded"""
    tools = _get_debugging_tools()
    return tools['debug_hypergraph'](*args, **kwargs)

def start_profiling(*args, **kwargs):
    """Start profiling - lazy loaded"""
    tools = _get_debugging_tools()
    return tools['start_profiling'](*args, **kwargs)

def stop_profiling(*args, **kwargs):
    """Stop profiling - lazy loaded"""
    tools = _get_debugging_tools()
    return tools['stop_profiling'](*args, **kwargs)

class LazyPerformanceProfiler:
    """Lazy wrapper for PerformanceProfiler"""
    def __new__(cls, *args, **kwargs):
        tools = _get_debugging_tools()
        return tools['PerformanceProfiler'](*args, **kwargs)

class LazyDataIntegrityValidator:
    """Lazy wrapper for DataIntegrityValidator"""
    def __new__(cls, *args, **kwargs):
        tools = _get_debugging_tools()
        return tools['DataIntegrityValidator'](*args, **kwargs)

# Assign to module level for direct access
PerformanceProfiler = LazyPerformanceProfiler
DataIntegrityValidator = LazyDataIntegrityValidator

# Expose lazy modules at module level
algorithms = property(lambda self: _lazy.algorithms)
analysis = property(lambda self: _lazy.analysis) 
governance = property(lambda self: _lazy.governance)

# Create module-level properties for lazy access
def __getattr__(name):
    """Provide lazy access to heavy modules and I/O functions"""
    if name == 'algorithms':
        return _get_algorithms()
    elif name == 'analysis':
        return _get_analysis()
    elif name == 'governance':
        return _get_governance()
    elif name == 'geometry':
        # Lazy import of geometry module
        try:
            import anant.geometry as geometry_module
            return geometry_module
        except ImportError as e:
            print(f"Warning: Geometry module not available: {e}")
            return None
    elif name == 'layered_contextual_graph':
        # Lazy import of layered_contextual_graph module
        try:
            import anant.layered_contextual_graph as lcg_module
            return lcg_module
        except ImportError as e:
            print(f"Warning: Layered Contextual Graph module not available: {e}")
            return None
    elif name in ['StreamEvent', 'StreamEventType', 'quick_clean', 'quick_quality_check', 'quick_convert']:
        # Lazy I/O module access
        io_modules = _get_io_modules()
        return io_modules[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
    'geometry',
    'layered_contextual_graph',
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