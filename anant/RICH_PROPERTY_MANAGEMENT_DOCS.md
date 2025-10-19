# Rich Property Management System Documentation

## Overview

The Rich Property Management System is a comprehensive suite of advanced tools for analyzing, managing, and optimizing hypergraph properties. This system extends the core Anant library with sophisticated capabilities for property type detection, analysis frameworks, weight management, and incidence pattern recognition.

## System Architecture

The Rich Property Management System consists of four main components:

1. **PropertyTypeManager** - Automatic property type detection and optimization
2. **PropertyAnalysisFramework** - Comprehensive property analysis and correlation discovery
3. **WeightAnalyzer** - Specialized weight analysis and normalization tools
4. **IncidencePatternAnalyzer** - Structural pattern detection and motif analysis

## Component Documentation

### 1. PropertyTypeManager

The PropertyTypeManager provides automatic detection and optimization of property types in hypergraph data.

#### Features
- **Automatic Type Detection**: Detects categorical, numerical, temporal, text, vector, and JSON property types
- **Storage Optimization**: Optimizes memory usage through intelligent type conversion
- **Batch Processing**: Analyzes entire DataFrames efficiently
- **Memory Analysis**: Provides detailed memory usage reports and optimization recommendations

#### Supported Property Types
```python
class PropertyType(Enum):
    CATEGORICAL = "categorical"    # Limited unique string values
    NUMERICAL = "numerical"       # Integer or floating-point numbers
    TEMPORAL = "temporal"         # Date/time values
    TEXT = "text"                # Free-form text content
    VECTOR = "vector"            # Lists of numerical values
    JSON = "json"                # Structured JSON data
    UNKNOWN = "unknown"          # Unable to determine type
```

#### Usage Examples

```python
from anant.utils import PropertyTypeManager, PropertyType

# Initialize the manager
manager = PropertyTypeManager()

# Detect property types for a single column
prop_type = manager.detect_property_type(df['column_name'])

# Detect types for all columns
type_map = manager.detect_all_property_types(df)

# Optimize storage for a specific column
optimized_df = manager.optimize_column_storage(df, 'column_name')

# Optimize entire DataFrame
optimized_df = manager.optimize_dataframe_storage(df)

# Analyze memory usage
memory_analysis = manager.analyze_memory_usage(df)
```

#### Storage Optimizations
- **Categorical**: Converts string columns to `pl.Categorical` type
- **Numerical**: Optimizes integer/float precision based on value ranges
- **Sparse Data**: Identifies sparsity patterns for potential compression
- **Type Conversion**: Suggests optimal data types for memory efficiency

### 2. PropertyAnalysisFramework

The PropertyAnalysisFramework provides comprehensive statistical analysis and relationship discovery for hypergraph properties.

#### Features
- **Correlation Analysis**: Multiple correlation methods (Pearson, Spearman, Kendall, Mutual Information)
- **Anomaly Detection**: Statistical outliers, isolation forest, distribution-based detection
- **Distribution Analysis**: Statistical characterization of property distributions
- **Relationship Discovery**: Automated discovery of property relationships and dependencies
- **Significance Testing**: Statistical significance assessment for correlations

#### Correlation Types
```python
class CorrelationType(Enum):
    PEARSON = "pearson"           # Linear correlation
    SPEARMAN = "spearman"         # Rank correlation
    KENDALL = "kendall"           # Tau correlation
    MUTUAL_INFO = "mutual_information"  # Information-theoretic correlation
```

#### Anomaly Detection Methods
```python
class AnomalyType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"    # IQR-based outliers
    ISOLATION_FOREST = "isolation_forest"          # ML-based isolation
    DISTRIBUTION_ANOMALY = "distribution_anomaly"  # Distribution fit anomalies
    PATTERN_ANOMALY = "pattern_anomaly"            # Pattern-based anomalies
```

#### Usage Examples

```python
from anant.utils import PropertyAnalysisFramework, CorrelationType, AnomalyType

# Initialize the framework
framework = PropertyAnalysisFramework()

# Analyze correlations between properties
correlations = framework.analyze_property_correlations(
    df,
    properties=['prop1', 'prop2', 'prop3'],
    correlation_types=[CorrelationType.PEARSON, CorrelationType.SPEARMAN],
    min_correlation=0.3
)

# Detect anomalies in a property
anomalies = framework.detect_property_anomalies(
    df,
    'property_name',
    anomaly_types=[AnomalyType.STATISTICAL_OUTLIER, AnomalyType.ISOLATION_FOREST]
)

# Analyze property distribution
distribution = framework.analyze_property_distribution(df, 'property_name')

# Discover relationships for a target property
relationships = framework.discover_property_relationships(
    df,
    'target_property',
    min_correlation=0.3
)

# Generate comprehensive analysis report
report = framework.generate_analysis_report(df, properties=['prop1', 'prop2'])
```

### 3. WeightAnalyzer

The WeightAnalyzer specializes in analyzing and optimizing weight properties in hypergraphs.

#### Features
- **Weight Statistics**: Comprehensive statistical analysis of weight distributions
- **Distribution Detection**: Automatic detection of weight distribution types
- **Normalization Methods**: Multiple weight normalization techniques
- **Weight Clustering**: Cluster entities based on weight patterns
- **Correlation Analysis**: Analyze relationships between different weight properties
- **Storage Optimization**: Optimize weight storage through compression and type optimization

#### Weight Distribution Types
```python
class WeightDistributionType(Enum):
    UNIFORM = "uniform"           # Uniform distribution
    NORMAL = "normal"             # Normal/Gaussian distribution
    EXPONENTIAL = "exponential"  # Exponential distribution
    POWER_LAW = "power_law"      # Power law distribution
    LOG_NORMAL = "log_normal"    # Log-normal distribution
    UNKNOWN = "unknown"          # Unknown distribution
```

#### Normalization Methods
```python
class WeightNormalizationType(Enum):
    MIN_MAX = "min_max"          # Min-max scaling to [0,1]
    Z_SCORE = "z_score"          # Z-score standardization
    UNIT_VECTOR = "unit_vector"  # L2 normalization
    SOFTMAX = "softmax"          # Softmax normalization
    RANK = "rank"                # Rank-based normalization
    QUANTILE = "quantile"        # Quantile normalization
```

#### Usage Examples

```python
from anant.utils import WeightAnalyzer, WeightNormalizationType, WeightDistributionType

# Initialize the analyzer
analyzer = WeightAnalyzer()

# Analyze weight statistics
stats = analyzer.analyze_weight_statistics(
    df,
    ['weight_col1', 'weight_col2'],
    entity_type='node'
)

# Detect weight distribution
distribution = analyzer.detect_weight_distribution(df, 'weight_column')

# Normalize weights
normalized_df = analyzer.normalize_weights(
    df,
    ['weight_col1', 'weight_col2'],
    WeightNormalizationType.MIN_MAX
)

# Cluster by weight patterns
clusters = analyzer.cluster_by_weights(
    df,
    ['weight_col1', 'weight_col2'],
    n_clusters=5,
    clustering_method='kmeans'
)

# Analyze weight correlations
correlations = analyzer.analyze_weight_correlations(
    df,
    ['weight_col1', 'weight_col2', 'weight_col3'],
    min_correlation=0.3
)

# Optimize weight storage
optimization = analyzer.optimize_weight_storage(
    df,
    ['weight_col1', 'weight_col2'],
    compression_threshold=0.1
)
```

### 4. IncidencePatternAnalyzer

The IncidencePatternAnalyzer detects and analyzes structural patterns in hypergraph incidence relationships.

#### Features
- **Motif Detection**: Detect recurring structural motifs in incidence data
- **Pattern Classification**: Classify patterns into types (star, chain, clique, hub, etc.)
- **Topological Analysis**: Compute topological features of incidence structures
- **Anomaly Detection**: Identify anomalous structural patterns
- **Pattern Statistics**: Statistical analysis of detected patterns

#### Pattern Types
```python
class PatternType(Enum):
    STAR = "star"                # One edge connecting multiple nodes
    CHAIN = "chain"              # Linear sequence of connections
    CYCLE = "cycle"              # Circular connection pattern
    CLIQUE = "clique"           # All nodes connected to same edges
    BIPARTITE = "bipartite"     # Two distinct node sets
    TREE = "tree"               # Hierarchical pattern
    MESH = "mesh"               # Grid-like pattern
    HUB = "hub"                 # High-degree central node
    BRIDGE = "bridge"           # Edge connecting components
    COMMUNITY = "community"     # Dense substructure
```

#### Usage Examples

```python
from anant.utils import IncidencePatternAnalyzer, PatternType, MotifSize

# Initialize the analyzer
analyzer = IncidencePatternAnalyzer()

# Detect incidence motifs
motifs = analyzer.detect_incidence_motifs(
    incidence_df,
    node_col='node',
    edge_col='edge',
    min_frequency=2
)

# Analyze pattern statistics
pattern_stats = analyzer.analyze_pattern_statistics(motifs)

# Compute topological features
topo_features = analyzer.compute_topological_features(
    incidence_df,
    node_col='node',
    edge_col='edge'
)

# Detect anomalous patterns
anomalies = analyzer.detect_anomalous_patterns(
    motifs,
    significance_threshold=0.05
)

# Generate comprehensive pattern report
report = analyzer.generate_pattern_report(
    incidence_df,
    node_col='node',
    edge_col='edge'
)
```

## Integration with Core Anant Library

The Rich Property Management System seamlessly integrates with the core Anant library components:

### Integration with Hypergraph Class

```python
from anant import Hypergraph
from anant.utils import PropertyTypeManager, PropertyAnalysisFramework

# Create hypergraph with rich property analysis
hg = Hypergraph.from_incidence_dataframe(incidence_df)

# Analyze node properties
property_manager = PropertyTypeManager()
analysis_framework = PropertyAnalysisFramework()

# Optimize node property storage
optimized_nodes = property_manager.optimize_dataframe_storage(hg.nodes.dataframe)

# Analyze node property correlations
node_correlations = analysis_framework.analyze_property_correlations(
    hg.nodes.dataframe,
    min_correlation=0.3
)

# Update hypergraph with optimized properties
hg.nodes.dataframe = optimized_nodes
```

### Integration with PropertyStore

```python
from anant.classes import PropertyStore
from anant.utils import WeightAnalyzer

# Create property store with weight analysis
property_store = PropertyStore()
weight_analyzer = WeightAnalyzer()

# Add properties and analyze weights
property_store.add_property('node_weight', node_weights)
property_store.add_property('edge_weight', edge_weights)

# Analyze weight statistics
weight_stats = weight_analyzer.analyze_weight_statistics(
    property_store.dataframe,
    ['node_weight', 'edge_weight']
)

# Normalize weights
normalized_df = weight_analyzer.normalize_weights(
    property_store.dataframe,
    ['node_weight', 'edge_weight'],
    WeightNormalizationType.Z_SCORE
)
```

## Performance Characteristics

### Memory Optimization
- **Categorical Optimization**: 50-90% memory reduction for string categories
- **Numerical Precision**: 25-75% reduction through appropriate integer/float types
- **Sparse Data**: Up to 80% compression for sparse weight matrices
- **Type Detection**: O(n) time complexity for n data points

### Analysis Performance
- **Correlation Analysis**: O(k²) for k properties
- **Anomaly Detection**: O(n log n) for statistical methods, O(n) for isolation forest
- **Pattern Detection**: O(e × n) for e edges and n nodes
- **Distribution Fitting**: O(n log n) for most distribution tests

### Scalability
- **Property Detection**: Handles millions of data points efficiently
- **Pattern Analysis**: Optimized for hypergraphs with thousands of nodes/edges
- **Memory Analysis**: Constant memory overhead regardless of data size
- **Batch Processing**: Efficient parallel processing for multiple properties

## Advanced Usage Patterns

### Multi-Modal Analysis Pipeline

```python
from anant.utils import (
    PropertyTypeManager, PropertyAnalysisFramework,
    WeightAnalyzer, IncidencePatternAnalyzer
)

def comprehensive_hypergraph_analysis(hypergraph):
    """Complete analysis pipeline for hypergraph properties"""
    
    # 1. Property type detection and optimization
    property_manager = PropertyTypeManager()
    
    # Optimize node properties
    node_types = property_manager.detect_all_property_types(hypergraph.nodes.dataframe)
    optimized_nodes = property_manager.optimize_dataframe_storage(hypergraph.nodes.dataframe)
    
    # Optimize edge properties
    edge_types = property_manager.detect_all_property_types(hypergraph.edges.dataframe)
    optimized_edges = property_manager.optimize_dataframe_storage(hypergraph.edges.dataframe)
    
    # 2. Statistical analysis
    analysis_framework = PropertyAnalysisFramework()
    
    # Analyze node property correlations
    node_correlations = analysis_framework.analyze_property_correlations(optimized_nodes)
    edge_correlations = analysis_framework.analyze_property_correlations(optimized_edges)
    
    # Detect anomalies
    node_anomalies = {}
    for prop in optimized_nodes.columns:
        if node_types[prop] == PropertyType.NUMERICAL:
            anomalies = analysis_framework.detect_property_anomalies(optimized_nodes, prop)
            if anomalies:
                node_anomalies[prop] = anomalies
    
    # 3. Weight analysis
    weight_analyzer = WeightAnalyzer()
    
    # Identify weight columns
    weight_columns = [col for col, ptype in {**node_types, **edge_types}.items() 
                     if ptype == PropertyType.NUMERICAL and 'weight' in col.lower()]
    
    if weight_columns:
        # Combine all weight data
        all_weights = combine_weight_data(optimized_nodes, optimized_edges, weight_columns)
        
        # Analyze weight distributions
        weight_stats = weight_analyzer.analyze_weight_statistics(all_weights, weight_columns)
        weight_correlations = weight_analyzer.analyze_weight_correlations(all_weights, weight_columns)
        
        # Cluster by weight patterns
        weight_clusters = weight_analyzer.cluster_by_weights(all_weights, weight_columns, n_clusters=5)
    
    # 4. Structural pattern analysis
    pattern_analyzer = IncidencePatternAnalyzer()
    
    # Get incidence data
    incidence_df = hypergraph.incidence_store.dataframe
    
    # Detect motifs and patterns
    motifs = pattern_analyzer.detect_incidence_motifs(incidence_df)
    pattern_stats = pattern_analyzer.analyze_pattern_statistics(motifs)
    topo_features = pattern_analyzer.compute_topological_features(incidence_df)
    
    # Generate comprehensive report
    return {
        'property_optimization': {
            'node_types': node_types,
            'edge_types': edge_types,
            'memory_savings': calculate_memory_savings(hypergraph.nodes.dataframe, optimized_nodes)
        },
        'statistical_analysis': {
            'node_correlations': node_correlations,
            'edge_correlations': edge_correlations,
            'node_anomalies': node_anomalies
        },
        'weight_analysis': {
            'weight_statistics': weight_stats,
            'weight_correlations': weight_correlations,
            'weight_clusters': weight_clusters
        } if weight_columns else None,
        'pattern_analysis': {
            'motifs': motifs,
            'pattern_statistics': pattern_stats,
            'topological_features': topo_features
        }
    }
```

### Custom Property Type Extensions

```python
from anant.utils import PropertyTypeManager, PropertyType
from enum import Enum

class ExtendedPropertyType(Enum):
    """Extended property types for domain-specific data"""
    GEOGRAPHIC = "geographic"     # GPS coordinates, addresses
    CURRENCY = "currency"         # Monetary values
    IDENTIFIER = "identifier"     # Unique identifiers
    RATING = "rating"            # Rating scales (1-5, 1-10, etc.)

class CustomPropertyManager(PropertyTypeManager):
    """Extended property manager with domain-specific types"""
    
    def detect_property_type(self, sample: pl.Series) -> ExtendedPropertyType:
        """Extended type detection with custom types"""
        
        # First try base detection
        base_type = super().detect_property_type(sample)
        
        if base_type == PropertyType.TEXT:
            # Check for geographic data
            if self._is_geographic(sample):
                return ExtendedPropertyType.GEOGRAPHIC
            
            # Check for identifiers
            if self._is_identifier(sample):
                return ExtendedPropertyType.IDENTIFIER
        
        elif base_type == PropertyType.NUMERICAL:
            # Check for currency
            if self._is_currency(sample):
                return ExtendedPropertyType.CURRENCY
            
            # Check for ratings
            if self._is_rating(sample):
                return ExtendedPropertyType.RATING
        
        return base_type
    
    def _is_geographic(self, sample: pl.Series) -> bool:
        """Detect geographic data patterns"""
        values = sample.head(100).to_list()
        geo_patterns = [
            r'^\d+\.\d+,\s*-?\d+\.\d+$',  # lat,lon coordinates
            r'^\d{5}(-\d{4})?$',          # ZIP codes
            r'^[A-Z]{2}\s\d{5}$'          # State ZIP
        ]
        
        import re
        geo_count = 0
        for value in values:
            if isinstance(value, str):
                for pattern in geo_patterns:
                    if re.match(pattern, value):
                        geo_count += 1
                        break
        
        return geo_count / len(values) > 0.7
    
    def _is_currency(self, sample: pl.Series) -> bool:
        """Detect currency values"""
        values = sample.drop_nulls().to_numpy()
        
        # Check if values are in typical currency ranges and precision
        if len(values) == 0:
            return False
        
        # Currency typically has 2 decimal places when converted to cents
        cent_values = values * 100
        is_integer = np.allclose(cent_values, np.round(cent_values))
        
        # Currency values are typically positive and in reasonable ranges
        is_positive = np.all(values >= 0)
        reasonable_range = np.all(values < 1000000)  # Less than $1M
        
        return is_integer and is_positive and reasonable_range
    
    def _is_rating(self, sample: pl.Series) -> bool:
        """Detect rating scales"""
        values = sample.drop_nulls().to_numpy()
        
        if len(values) == 0:
            return False
        
        unique_values = np.unique(values)
        
        # Check for common rating scales
        common_scales = [
            set(range(1, 6)),    # 1-5 scale
            set(range(1, 11)),   # 1-10 scale
            set(range(0, 6)),    # 0-5 scale
            set(range(0, 11))    # 0-10 scale
        ]
        
        value_set = set(unique_values)
        
        for scale in common_scales:
            if value_set.issubset(scale) and len(value_set) >= 3:
                return True
        
        return False
```

## Best Practices

### 1. Property Type Detection
- **Sample Size**: Use sufficient sample sizes for accurate type detection
- **Data Quality**: Clean data before type detection for better accuracy
- **Custom Types**: Extend base types for domain-specific requirements
- **Validation**: Validate detected types with domain knowledge

### 2. Performance Optimization
- **Batch Processing**: Process multiple properties together when possible
- **Memory Management**: Monitor memory usage during large-scale analysis
- **Caching**: Cache results for expensive computations
- **Parallel Processing**: Use parallel processing for independent analyses

### 3. Statistical Analysis
- **Significance Testing**: Always check statistical significance of correlations
- **Multiple Testing**: Apply corrections for multiple hypothesis testing
- **Sample Size**: Ensure adequate sample sizes for statistical validity
- **Assumptions**: Validate statistical assumptions before applying tests

### 4. Pattern Analysis
- **Frequency Thresholds**: Set appropriate minimum frequency thresholds for pattern detection
- **Complexity Limits**: Limit pattern complexity to avoid combinatorial explosion
- **Validation**: Validate detected patterns with domain expertise
- **Significance**: Focus on statistically significant patterns

## Error Handling and Debugging

### Common Issues and Solutions

1. **Memory Errors with Large DataFrames**
   ```python
   # Solution: Process in chunks
   def analyze_large_dataframe(df, chunk_size=10000):
       results = []
       for i in range(0, len(df), chunk_size):
           chunk = df.slice(i, chunk_size)
           result = analyze_chunk(chunk)
           results.append(result)
       return combine_results(results)
   ```

2. **Type Detection Failures**
   ```python
   # Solution: Handle edge cases
   try:
       prop_type = manager.detect_property_type(column)
   except Exception as e:
       logger.warning(f"Type detection failed for {column}: {e}")
       prop_type = PropertyType.UNKNOWN
   ```

3. **Statistical Analysis Errors**
   ```python
   # Solution: Validate data before analysis
   def safe_correlation_analysis(df, properties):
       # Check for sufficient data
       if len(df) < 10:
           return []
       
       # Filter numerical properties
       valid_props = []
       for prop in properties:
           if prop in df.columns and df[prop].dtype.is_numeric():
               valid_props.append(prop)
       
       if len(valid_props) < 2:
           return []
       
       return framework.analyze_property_correlations(df, valid_props)
   ```

4. **Pattern Detection Issues**
   ```python
   # Solution: Handle sparse or disconnected graphs
   def robust_pattern_detection(incidence_df):
       # Check for minimum connectivity
       if len(incidence_df) < 10:
           return []
       
       # Validate incidence structure
       nodes = set(incidence_df['node'])
       edges = set(incidence_df['edge'])
       
       if len(nodes) < 3 or len(edges) < 2:
           return []
       
       return analyzer.detect_incidence_motifs(incidence_df)
   ```

## Testing and Validation

The Rich Property Management System includes comprehensive test suites:

- **Unit Tests**: Test individual component functionality
- **Integration Tests**: Test component interactions
- **Performance Tests**: Validate performance characteristics
- **Edge Case Tests**: Handle boundary conditions and edge cases

### Running Tests

```bash
# Run all Rich Property Management tests
python test_rich_property_management.py

# Run specific component tests
python -m pytest test_rich_property_management.py::TestPropertyTypeManager
python -m pytest test_rich_property_management.py::TestPropertyAnalysisFramework
python -m pytest test_rich_property_management.py::TestWeightAnalyzer
python -m pytest test_rich_property_management.py::TestIncidencePatternAnalyzer
```

## Future Extensions

### Planned Enhancements
1. **Machine Learning Integration**: Advanced ML-based property analysis
2. **Temporal Analysis**: Time-series property analysis capabilities
3. **Graph Neural Networks**: Integration with GNN frameworks
4. **Distributed Computing**: Support for distributed hypergraph analysis
5. **Interactive Visualization**: Rich interactive property exploration tools

### Extension Points
- **Custom Property Types**: Add domain-specific property types
- **Analysis Algorithms**: Implement custom analysis algorithms
- **Optimization Strategies**: Add custom storage optimization strategies
- **Pattern Types**: Define custom structural pattern types

## Conclusion

The Rich Property Management System provides a comprehensive foundation for advanced hypergraph property analysis. By combining automatic type detection, statistical analysis, weight management, and pattern recognition, it enables sophisticated analysis workflows that were previously difficult to implement.

The system's modular design allows for easy extension and customization while maintaining high performance and reliability. Whether you're analyzing small research datasets or large-scale production hypergraphs, the Rich Property Management System provides the tools needed for deep property insights.