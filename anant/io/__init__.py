"""
Advanced I/O and Integration Module for Anant Library

This module provides comprehensive I/O capabilities and data integration tools:

Enhanced File Format Support:
    - JSON with schema validation
    - CSV with flexible column mapping
    - GraphML for visualization tools
    - GML (Graph Modeling Language)
    - GEXF (Graph Exchange XML Format)
    - HDF5 for high-performance storage
    - Parquet for columnar data

Database Connectivity:
    - SQLite for lightweight embedded storage
    - PostgreSQL for production systems
    - MongoDB for document-based storage
    - Generic database interface
    - Schema management and migrations
    - Optimized batch operations

Streaming Data Processing:
    - Real-time hypergraph updates
    - Event-driven architecture
    - WebSocket streaming
    - Message queue integration (Redis, RabbitMQ)
    - File monitoring for live updates
    - Temporal hypergraph evolution tracking

Data Transformation Utilities:
    - Data quality assessment
    - Cleaning and preprocessing pipelines
    - Format conversion strategies
    - Feature engineering for hypergraphs
    - ETL (Extract, Transform, Load) pipelines
    - Statistical preprocessing and normalization

Key Features:
- Automatic format detection
- Schema validation and migration
- Memory-efficient streaming
- Comprehensive data quality reports
- Flexible transformation pipelines
- Real-time hypergraph updates
- Production-ready database integration
"""

# Legacy I/O imports
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

# New Advanced I/O imports
from .enhanced_formats import (
    EnhancedFileFormats,
    ImportExportConfig,
    export_hypergraph as export_enhanced,
    import_hypergraph as import_enhanced
)

from .database_connectivity import (
    DatabaseConfig,
    DatabaseConnector,
    SQLiteConnector,
    PostgreSQLConnector,
    MongoDBConnector,
    DatabaseManager,
    create_sqlite_manager,
    create_postgresql_manager,
    create_mongodb_manager
)

from .streaming_processing import (
    StreamEvent,
    StreamEventType,
    StreamingConfig,
    HypergraphStreamProcessor,
    WebSocketStreamer,
    MessageQueueStreamer,
    FileStreamMonitor,
    create_streaming_hypergraph,
    create_websocket_streamer,
    create_message_queue_streamer,
    create_file_monitor
)

from .data_transformation import (
    DataQualityIssue,
    DataQualityReport,
    TransformationConfig,
    DataQualityAssessor,
    DataCleaner,
    HypergraphConverter,
    FeatureEngineer,
    ETLPipeline,
    assess_data_quality,
    clean_data,
    convert_to_hypergraph,
    create_etl_pipeline
)


# Complete exports
__all__ = [
    # Legacy I/O
    "AnantIO",
    "save_hypergraph", 
    "load_hypergraph",
    "AdvancedAnantIO",
    "CompressionType",
    "FileFormat", 
    "IOConfiguration",
    "DatasetMetadata",
    "LoadResult",
    "SaveResult",
    
    # Enhanced file formats
    "EnhancedFileFormats",
    "ImportExportConfig", 
    "export_enhanced",
    "import_enhanced",
    
    # Database connectivity
    "DatabaseConfig",
    "DatabaseConnector",
    "SQLiteConnector", 
    "PostgreSQLConnector",
    "MongoDBConnector",
    "DatabaseManager",
    "create_sqlite_manager",
    "create_postgresql_manager", 
    "create_mongodb_manager",
    
    # Streaming processing
    "StreamEvent",
    "StreamEventType",
    "StreamingConfig",
    "HypergraphStreamProcessor",
    "WebSocketStreamer",
    "MessageQueueStreamer", 
    "FileStreamMonitor",
    "create_streaming_hypergraph",
    "create_websocket_streamer",
    "create_message_queue_streamer",
    "create_file_monitor",
    
    # Data transformation
    "DataQualityIssue",
    "DataQualityReport",
    "TransformationConfig",
    "DataQualityAssessor",
    "DataCleaner",
    "HypergraphConverter",
    "FeatureEngineer", 
    "ETLPipeline",
    "assess_data_quality",
    "clean_data",
    "convert_to_hypergraph",
    "create_etl_pipeline",
    
    # Quick access functions
    "quick_import",
    "quick_export",
    "quick_clean",
    "quick_quality_check",
    "quick_convert"
]


# Version information
__version__ = "1.0.0"
__author__ = "Anant Development Team"


# Quick access functions for common use cases
def quick_import(filepath, format_type=None, **kwargs):
    """
    Quick import function for common file formats
    
    Args:
        filepath: Path to the file
        format_type: Override auto-detection ('json', 'csv', 'graphml', etc.)
        **kwargs: Format-specific arguments
    
    Returns:
        Hypergraph instance
    
    Examples:
        >>> hg = quick_import('data.json')
        >>> hg = quick_import('data.csv', format_type='csv', format_type='incidences')
        >>> hg = quick_import('data.h5', format_type='hdf5')
    """
    return import_enhanced(filepath, format_type, **kwargs)


def quick_export(hg, filepath, format_type=None, **kwargs):
    """
    Quick export function for common file formats
    
    Args:
        hg: Hypergraph to export
        filepath: Output file path
        format_type: Override auto-detection
        **kwargs: Format-specific arguments
    
    Examples:
        >>> quick_export(hg, 'output.json')
        >>> quick_export(hg, 'output.csv', format_type='csv', format_type='edgelist')
        >>> quick_export(hg, 'output.graphml')
    """
    export_enhanced(hg, filepath, format_type, **kwargs)


def quick_clean(data, **kwargs):
    """
    Quick data cleaning with sensible defaults
    
    Args:
        data: DataFrame or file path
        **kwargs: Configuration overrides
    
    Returns:
        Cleaned DataFrame
    
    Examples:
        >>> clean_df = quick_clean('data.csv')
        >>> clean_df = quick_clean(df, handle_missing_values='fill')
    """
    return clean_data(data, **kwargs)


def quick_quality_check(data):
    """
    Quick data quality assessment
    
    Args:
        data: DataFrame or file path
    
    Returns:
        DataQualityReport
    
    Examples:
        >>> report = quick_quality_check('data.csv')
        >>> print(f"Quality score: {report.overall_score}")
    """
    return assess_data_quality(data)


def quick_convert(data, strategy='transaction', **kwargs):
    """
    Quick conversion to hypergraph
    
    Args:
        data: DataFrame or file path
        strategy: Conversion strategy ('transaction', 'bipartite', etc.)
        **kwargs: Strategy-specific arguments
    
    Returns:
        Hypergraph instance
    
    Examples:
        >>> hg = quick_convert('transactions.csv', strategy='transaction')
        >>> hg = quick_convert(df, strategy='bipartite', source_col='user', target_col='item')
    """
    return convert_to_hypergraph(data, strategy, **kwargs)


# Default instances for convenience
advanced_io = AdvancedAnantIO()
enhanced_formats = EnhancedFileFormats()


# Integration examples and tutorials
EXAMPLES = {
    'basic_io': """
# Basic I/O Operations
from anant.io import quick_import, quick_export

# Import from various formats
hg_json = quick_import('hypergraph.json')
hg_csv = quick_import('data.csv', format_type='incidences')
hg_graphml = quick_import('network.graphml')

# Export to various formats  
quick_export(hg_json, 'output.json')
quick_export(hg_json, 'output.csv', format_type='edgelist')
quick_export(hg_json, 'output.h5')
""",

    'database_integration': """
# Database Integration
from anant.io import create_sqlite_manager
import asyncio

async def database_example():
    # Create database manager
    db = create_sqlite_manager('hypergraphs.db')
    
    async with db:
        # Save hypergraph
        await db.save(hg, 'my_hypergraph', metadata={'version': 1})
        
        # Load hypergraph
        loaded_hg = await db.load('my_hypergraph')
        
        # List all hypergraphs
        hypergraphs = await db.list()

asyncio.run(database_example())
""",

    'streaming_processing': """
# Streaming Data Processing
from anant.io import create_streaming_hypergraph, StreamEvent, StreamEventType
import asyncio

async def streaming_example():
    # Create streaming processor
    processor = await create_streaming_hypergraph()
    
    # Add streaming events
    event = StreamEvent(
        event_type=StreamEventType.ADD_EDGE,
        data={'edge_id': 'e1', 'nodes': ['n1', 'n2', 'n3']}
    )
    await processor.add_event(event)
    
    # Subscribe to updates
    def on_update(event, hg):
        print(f"Hypergraph updated: {hg.num_nodes} nodes, {hg.num_edges} edges")
    
    processor.subscribe(on_update)

asyncio.run(streaming_example())
""",

    'data_transformation': """
# Data Transformation Pipeline
from anant.io import quick_quality_check, quick_clean, quick_convert

# Assess data quality
report = quick_quality_check('raw_data.csv')
print(f"Data quality score: {report.overall_score}")

# Clean data
clean_df = quick_clean('raw_data.csv', 
                      handle_missing_values='fill',
                      remove_duplicates=True)

# Convert to hypergraph
hg = quick_convert(clean_df, 
                  strategy='transaction',
                  transaction_col='transaction_id',
                  item_col='product_id')
""",

    'etl_pipeline': """
# ETL Pipeline Example
from anant.io import create_etl_pipeline, DataCleaner, HypergraphConverter
import polars as pl

# Create ETL pipeline
pipeline = create_etl_pipeline()

# Add extractors
pipeline.add_extractor('csv', lambda path: pl.read_csv(path))
pipeline.add_extractor('json', lambda path: pl.read_json(path))

# Add transformers  
pipeline.add_transformer(DataCleaner())

# Add loaders
pipeline.add_loader('hypergraph', lambda data: HypergraphConverter().convert(data, 'transaction'))

# Run pipeline
hg = pipeline.run('csv', 'hypergraph', 
                 extract_params={'path': 'data.csv'})
"""
}


def get_example(example_name: str) -> str:
    """
    Get code example for specific functionality
    
    Args:
        example_name: Name of the example ('basic_io', 'database_integration', etc.)
    
    Returns:
        Example code as string
    """
    return EXAMPLES.get(example_name, "Example not found. Available examples: " + ", ".join(EXAMPLES.keys()))


def list_examples() -> list:
    """List all available examples"""
    return list(EXAMPLES.keys())


# Module initialization
def _check_optional_dependencies():
    """Check for optional dependencies and provide helpful messages"""
    optional_deps = {
        'h5py': 'HDF5 support',
        'websockets': 'WebSocket streaming',
        'aioredis': 'Redis streaming',
        'aio_pika': 'RabbitMQ streaming', 
        'asyncpg': 'PostgreSQL support',
        'motor': 'MongoDB support'
    }
    
    missing_deps = []
    for dep, feature in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep} (for {feature})")
    
    if missing_deps:
        print(f"Optional dependencies not found: {', '.join(missing_deps)}")
        print("Install them as needed for additional functionality.")


# Check dependencies on import (commented out to avoid noise)
# _check_optional_dependencies()