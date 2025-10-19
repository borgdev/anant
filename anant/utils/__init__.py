"""
Utility modules for the Anant library
"""

from .decorators import performance_monitor, cache_result
from .extras import setup_polars_config, create_empty_dataframe, validate_hypergraph_data

# Advanced utility modules
try:
    from .property_types import PropertyType, PropertyTypeManager
    from .property_analysis import (
        PropertyAnalysisFramework, 
        CorrelationType, 
        AnomalyType,
        CorrelationResult,
        AnomalyDetectionResult,
        PropertyDistribution
    )
    from .weight_analyzer import (
        WeightAnalyzer,
        WeightNormalizationType,
        WeightDistributionType,
        WeightStatistics,
        WeightDistribution,
        WeightCluster
    )
    from .incidence_patterns import (
        IncidencePatternAnalyzer,
        PatternType,
        MotifSize,
        IncidenceMotif,
        PatternStatistics,
        TopologicalFeatures
    )
    from .benchmarks import PerformanceBenchmark
    
    HAS_ADVANCED_UTILS = True
except ImportError as e:
    print(f"Warning: Some advanced utilities not available: {e}")
    HAS_ADVANCED_UTILS = False

__all__ = [
    'performance_monitor',
    'cache_result', 
    'setup_polars_config',
    'create_empty_dataframe',
    'validate_hypergraph_data'
]

# Add advanced utilities if available
if HAS_ADVANCED_UTILS:
    __all__.extend([
        'PropertyType', 'PropertyTypeManager',
        'PropertyAnalysisFramework', 'CorrelationType', 'AnomalyType',
        'CorrelationResult', 'AnomalyDetectionResult', 'PropertyDistribution',
        'WeightAnalyzer', 'WeightNormalizationType', 'WeightDistributionType',
        'WeightStatistics', 'WeightDistribution', 'WeightCluster',
        'IncidencePatternAnalyzer', 'PatternType', 'MotifSize',
        'IncidenceMotif', 'PatternStatistics', 'TopologicalFeatures',
        'PerformanceBenchmark'
    ])