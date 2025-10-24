# Enhanced Data Analysis Capabilities for "anant"

**Document Version**: 1.0  
**Date**: October 17, 2025  
**Focus**: Dataset Analysis with Properties, Weights, and Incidences

## Overview

This document outlines enhanced capabilities for the "anant" library to support comprehensive dataset analysis stored as parquet files, with rich property management and analytical features using weights and incidences.

## Table of Contents

1. [Enhanced Property Storage](#enhanced-property-storage)
2. [Parquet-Based Dataset Integration](#parquet-based-dataset-integration)
3. [Weight and Incidence Analysis](#weight-and-incidence-analysis)
4. [Data Analysis Workflows](#data-analysis-workflows)
5. [Implementation Examples](#implementation-examples)
6. [Performance Optimizations](#performance-optimizations)

## Enhanced Property Storage

### Multi-Level Property System

The enhanced "anant" system will support a hierarchical property storage system that goes beyond simple ID storage:

#### 1. Node Properties
```python
# Rich node property storage
node_properties = {
    "categorical": ["type", "category", "group"],
    "numerical": ["score", "value", "importance"],
    "temporal": ["created_at", "updated_at", "event_time"],
    "textual": ["description", "name", "label"],
    "vector": ["embedding", "features"],
    "metadata": ["source", "confidence", "version"]
}
```

#### 2. Edge Properties  
```python
# Rich edge property storage
edge_properties = {
    "relationship": ["relation_type", "strength", "direction"],
    "numerical": ["weight", "cost", "distance", "similarity"],
    "temporal": ["start_time", "end_time", "duration"],
    "categorical": ["status", "priority", "label"],
    "contextual": ["context", "environment", "conditions"]
}
```

#### 3. Incidence Properties
```python
# Properties specific to node-edge relationships
incidence_properties = {
    "participation": ["role", "influence", "contribution"],
    "weights": ["edge_weight", "node_weight", "combined_weight"],
    "temporal": ["join_time", "leave_time", "duration"],
    "context": ["interaction_type", "strength", "frequency"]
}
```

### Enhanced PropertyStore Implementation

```python
import polars as pl
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

class EnhancedPropertyStore:
    """
    Advanced property storage system for anant with rich data type support
    """
    
    def __init__(self, 
                 level: int,  # 0=edges, 1=nodes, 2=incidences
                 schema: Optional[Dict[str, pl.DataType]] = None,
                 default_properties: Optional[Dict[str, Any]] = None):
        
        self.level = level
        self.schema = schema or self._get_default_schema()
        self.default_properties = default_properties or {}
        
        # Initialize with proper schema
        self._data = pl.DataFrame(schema=self.schema)
        
        # Property type tracking
        self.categorical_props = set()
        self.numerical_props = set()
        self.temporal_props = set()
        self.vector_props = set()
        self.text_props = set()
    
    def _get_default_schema(self) -> Dict[str, pl.DataType]:
        """Define default schema based on level"""
        base_schema = {
            "weight": pl.Float64,
            "misc_properties": pl.Struct([]),
            "created_at": pl.Datetime,
            "updated_at": pl.Datetime
        }
        
        if self.level == 2:  # Incidences
            base_schema.update({
                "edge_id": pl.Utf8,
                "node_id": pl.Utf8,
                "incidence_weight": pl.Float64,
                "role": pl.Utf8
            })
        else:  # Edges or Nodes
            base_schema.update({
                "uid": pl.Utf8,
                "label": pl.Utf8,
                "type": pl.Utf8
            })
        
        return base_schema
    
    def add_property_column(self, 
                           name: str, 
                           dtype: pl.DataType, 
                           default_value: Any = None,
                           property_type: str = "misc") -> None:
        """
        Add a new property column with proper type handling
        
        Parameters
        ----------
        name : str
            Property name
        dtype : pl.DataType
            Polars data type
        default_value : Any
            Default value for existing rows
        property_type : str
            Type category: 'categorical', 'numerical', 'temporal', 'vector', 'text'
        """
        if name not in self._data.columns:
            # Add column with default value
            self._data = self._data.with_columns(
                pl.lit(default_value).cast(dtype).alias(name)
            )
            
            # Update schema
            self.schema[name] = dtype
            
            # Track property type
            getattr(self, f"{property_type}_props").add(name)
    
    def bulk_set_properties(self, 
                           properties_df: pl.DataFrame,
                           join_on: str = "uid") -> None:
        """
        Efficiently bulk update properties from external DataFrame
        
        Parameters
        ---------- 
        properties_df : pl.DataFrame
            DataFrame containing properties to merge
        join_on : str
            Column to join on
        """
        # Ensure compatible schemas
        for col in properties_df.columns:
            if col != join_on and col not in self.schema:
                # Auto-detect type and add column
                dtype = properties_df[col].dtype
                self.add_property_column(col, dtype)
        
        # Perform efficient join
        self._data = self._data.join(
            properties_df,
            on=join_on,
            how="left",
            suffix="_new"
        ).select([
            # Use new values where available, otherwise keep existing
            pl.when(pl.col(f"{col}_new").is_not_null())
            .then(pl.col(f"{col}_new"))
            .otherwise(pl.col(col))
            .alias(col) if f"{col}_new" in self._data.columns else pl.col(col)
            for col in self.schema.keys()
        ])
    
    def get_property_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of stored properties"""
        summary = {
            "total_objects": self._data.height,
            "property_columns": len(self.schema),
            "property_types": {
                "categorical": list(self.categorical_props),
                "numerical": list(self.numerical_props), 
                "temporal": list(self.temporal_props),
                "vector": list(self.vector_props),
                "text": list(self.text_props)
            },
            "memory_usage": self._data.estimated_size("mb"),
            "null_counts": {
                col: self._data[col].null_count() 
                for col in self._data.columns
            }
        }
        return summary

class DatasetHypergraph:
    """
    Enhanced hypergraph designed for dataset analysis with rich properties
    """
    
    def __init__(self, 
                 incidence_data: Optional[pl.DataFrame] = None,
                 node_properties: Optional[pl.DataFrame] = None,
                 edge_properties: Optional[pl.DataFrame] = None,
                 incidence_properties: Optional[pl.DataFrame] = None):
        
        # Initialize enhanced property stores
        self.nodes = EnhancedPropertyStore(level=1)
        self.edges = EnhancedPropertyStore(level=0) 
        self.incidences = EnhancedPropertyStore(level=2)
        
        # Load data if provided
        if incidence_data is not None:
            self._load_incidence_data(incidence_data)
        
        if node_properties is not None:
            self.nodes.bulk_set_properties(node_properties)
            
        if edge_properties is not None:
            self.edges.bulk_set_properties(edge_properties)
            
        if incidence_properties is not None:
            self.incidences.bulk_set_properties(incidence_properties)
    
    def _load_incidence_data(self, data: pl.DataFrame) -> None:
        """Load incidence relationships with automatic property detection"""
        required_cols = ["edge_id", "node_id"]
        
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Incidence data must contain columns: {required_cols}")
        
        # Add weight if not present
        if "weight" not in data.columns:
            data = data.with_columns(pl.lit(1.0).alias("weight"))
        
        # Store incidence data
        self.incidences._data = data
        
        # Extract unique nodes and edges
        unique_nodes = data.select("node_id").unique()
        unique_edges = data.select("edge_id").unique()
        
        # Initialize node and edge stores with IDs
        self.nodes._data = unique_nodes.rename({"node_id": "uid"}).with_columns([
            pl.lit(1.0).alias("weight"),
            pl.lit({}).alias("misc_properties"),
            pl.lit(datetime.now()).alias("created_at"),
            pl.lit(datetime.now()).alias("updated_at")
        ])
        
        self.edges._data = unique_edges.rename({"edge_id": "uid"}).with_columns([
            pl.lit(1.0).alias("weight"), 
            pl.lit({}).alias("misc_properties"),
            pl.lit(datetime.now()).alias("created_at"),
            pl.lit(datetime.now()).alias("updated_at")
        ])
```

## Parquet-Based Dataset Integration

### Dataset Loading and Analysis Pipeline

```python
class DatasetAnalyzer:
    """
    Comprehensive dataset analysis using hypergraph structures
    """
    
    def __init__(self, hypergraph: DatasetHypergraph):
        self.hg = hypergraph
        
    @classmethod
    def from_parquet_dataset(cls, 
                           dataset_path: str,
                           incidence_file: str = "incidences.parquet",
                           node_props_file: str = "nodes.parquet", 
                           edge_props_file: str = "edges.parquet",
                           lazy_loading: bool = True) -> 'DatasetAnalyzer':
        """
        Load complete dataset from parquet files
        
        Parameters
        ----------
        dataset_path : str
            Base path containing parquet files
        incidence_file : str
            Incidence relationships file
        node_props_file : str
            Node properties file
        edge_props_file : str
            Edge properties file  
        lazy_loading : bool
            Whether to use lazy loading for large datasets
        """
        import os
        
        # Load incidence data (required)
        incidence_path = os.path.join(dataset_path, incidence_file)
        if lazy_loading:
            incidences = pl.scan_parquet(incidence_path).collect()
        else:
            incidences = pl.read_parquet(incidence_path)
        
        # Load property files if they exist
        node_props = None
        edge_props = None
        
        node_path = os.path.join(dataset_path, node_props_file)
        if os.path.exists(node_path):
            if lazy_loading:
                node_props = pl.scan_parquet(node_path).collect()
            else:
                node_props = pl.read_parquet(node_path)
        
        edge_path = os.path.join(dataset_path, edge_props_file)
        if os.path.exists(edge_path):
            if lazy_loading:
                edge_props = pl.scan_parquet(edge_path).collect()
            else:
                edge_props = pl.read_parquet(edge_path)
        
        # Create hypergraph
        hg = DatasetHypergraph(
            incidence_data=incidences,
            node_properties=node_props,
            edge_properties=edge_props
        )
        
        return cls(hg)
    
    def analyze_weight_distributions(self) -> Dict[str, Any]:
        """
        Analyze weight distributions across nodes, edges, and incidences
        """
        analysis = {}
        
        # Node weight analysis
        node_weights = self.hg.nodes._data.select("weight")
        analysis["node_weights"] = {
            "count": node_weights.height,
            "mean": node_weights.mean().item(),
            "std": node_weights.std().item(),
            "min": node_weights.min().item(),
            "max": node_weights.max().item(),
            "distribution": node_weights.to_series().value_counts().to_dict()
        }
        
        # Edge weight analysis  
        edge_weights = self.hg.edges._data.select("weight")
        analysis["edge_weights"] = {
            "count": edge_weights.height,
            "mean": edge_weights.mean().item(),
            "std": edge_weights.std().item(), 
            "min": edge_weights.min().item(),
            "max": edge_weights.max().item(),
            "distribution": edge_weights.to_series().value_counts().to_dict()
        }
        
        # Incidence weight analysis
        if "weight" in self.hg.incidences._data.columns:
            inc_weights = self.hg.incidences._data.select("weight") 
            analysis["incidence_weights"] = {
                "count": inc_weights.height,
                "mean": inc_weights.mean().item(),
                "std": inc_weights.std().item(),
                "min": inc_weights.min().item(), 
                "max": inc_weights.max().item(),
                "distribution": inc_weights.to_series().value_counts().to_dict()
            }
        
        return analysis
    
    def analyze_incidence_patterns(self) -> Dict[str, Any]:
        """
        Analyze incidence patterns and relationships
        """
        incidence_data = self.hg.incidences._data
        
        # Node degree analysis
        node_degrees = (
            incidence_data
            .group_by("node_id")
            .agg([
                pl.count().alias("degree"),
                pl.col("weight").sum().alias("total_weight"),
                pl.col("weight").mean().alias("avg_weight")
            ])
        )
        
        # Edge size analysis  
        edge_sizes = (
            incidence_data
            .group_by("edge_id")
            .agg([
                pl.count().alias("size"), 
                pl.col("weight").sum().alias("total_weight"),
                pl.col("weight").mean().alias("avg_weight")
            ])
        )
        
        analysis = {
            "node_degree_stats": {
                "mean_degree": node_degrees["degree"].mean(),
                "max_degree": node_degrees["degree"].max(),
                "min_degree": node_degrees["degree"].min(),
                "degree_distribution": node_degrees["degree"].value_counts().to_dict()
            },
            "edge_size_stats": {
                "mean_size": edge_sizes["size"].mean(),
                "max_size": edge_sizes["size"].max(), 
                "min_size": edge_sizes["size"].min(),
                "size_distribution": edge_sizes["size"].value_counts().to_dict()
            },
            "incidence_count": incidence_data.height,
            "unique_nodes": incidence_data["node_id"].n_unique(),
            "unique_edges": incidence_data["edge_id"].n_unique()
        }
        
        return analysis
    
    def weighted_centrality_analysis(self) -> Dict[str, pl.DataFrame]:
        """
        Compute weighted centrality measures using incidence weights
        """
        incidence_data = self.hg.incidences._data
        
        # Weighted degree centrality
        weighted_degree = (
            incidence_data
            .group_by("node_id") 
            .agg([
                pl.col("weight").sum().alias("weighted_degree"),
                pl.count().alias("unweighted_degree"),
                pl.col("weight").mean().alias("avg_edge_weight")
            ])
            .with_columns([
                # Normalize by max weighted degree
                (pl.col("weighted_degree") / pl.col("weighted_degree").max()).alias("normalized_weighted_degree")
            ])
        )
        
        # Edge-based centrality (importance of edges)
        edge_centrality = (
            incidence_data
            .group_by("edge_id")
            .agg([
                pl.col("weight").sum().alias("total_edge_weight"),
                pl.count().alias("edge_size"), 
                pl.col("weight").mean().alias("avg_node_weight")
            ])
            .with_columns([
                # Edge importance score
                (pl.col("total_edge_weight") * pl.col("edge_size")).alias("edge_importance"),
                (pl.col("total_edge_weight") / pl.col("total_edge_weight").max()).alias("normalized_edge_weight")
            ])
        )
        
        # Weighted betweenness approximation via edge participation
        node_edge_participation = (
            incidence_data
            .join(edge_centrality.select(["edge_id", "edge_importance"]), on="edge_id")
            .group_by("node_id")
            .agg([
                pl.col("edge_importance").sum().alias("total_edge_importance"),
                pl.col("edge_importance").count().alias("edges_participated"),
                pl.col("edge_importance").mean().alias("avg_edge_importance")
            ])
        )
        
        return {
            "weighted_degree": weighted_degree,
            "edge_centrality": edge_centrality, 
            "node_edge_participation": node_edge_participation
        }
    
    def property_correlation_analysis(self, 
                                    numerical_props: List[str],
                                    level: str = "nodes") -> pl.DataFrame:
        """
        Analyze correlations between numerical properties
        
        Parameters
        ----------
        numerical_props : List[str]
            List of numerical property names to analyze
        level : str
            Level to analyze: 'nodes', 'edges', or 'incidences'
        """
        if level == "nodes":
            data = self.hg.nodes._data
        elif level == "edges": 
            data = self.hg.edges._data
        elif level == "incidences":
            data = self.hg.incidences._data
        else:
            raise ValueError("Level must be 'nodes', 'edges', or 'incidences'")
        
        # Select numerical properties
        numeric_data = data.select(numerical_props)
        
        # Compute correlation matrix using Polars
        correlations = []
        for i, prop1 in enumerate(numerical_props):
            for j, prop2 in enumerate(numerical_props):
                if i <= j:  # Only compute upper triangle
                    corr = numeric_data.select([
                        pl.corr(prop1, prop2).alias("correlation")
                    ]).item()
                    
                    correlations.append({
                        "property1": prop1,
                        "property2": prop2, 
                        "correlation": corr
                    })
        
        return pl.DataFrame(correlations)
    
    def temporal_analysis(self, 
                         time_column: str = "created_at",
                         level: str = "incidences",
                         time_window: str = "1d") -> Dict[str, Any]:
        """
        Analyze temporal patterns in the hypergraph
        
        Parameters
        ----------
        time_column : str
            Column containing temporal information
        level : str
            Level to analyze: 'nodes', 'edges', or 'incidences'  
        time_window : str
            Aggregation window (e.g., '1d', '1h', '1w')
        """
        if level == "nodes":
            data = self.hg.nodes._data
        elif level == "edges":
            data = self.hg.edges._data 
        elif level == "incidences":
            data = self.hg.incidences._data
        else:
            raise ValueError("Level must be 'nodes', 'edges', or 'incidences'")
        
        if time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found in {level} data")
        
        # Temporal aggregation
        temporal_stats = (
            data
            .with_columns([
                pl.col(time_column).dt.truncate(time_window).alias("time_window")
            ])
            .group_by("time_window")
            .agg([
                pl.count().alias("count"),
                pl.col("weight").sum().alias("total_weight"),
                pl.col("weight").mean().alias("avg_weight")
            ])
            .sort("time_window")
        )
        
        return {
            "temporal_stats": temporal_stats,
            "time_range": {
                "start": data[time_column].min(),
                "end": data[time_column].max(),
                "span": data[time_column].max() - data[time_column].min()
            },
            "activity_periods": temporal_stats.filter(pl.col("count") > 0).height
        }

# Usage Examples
def example_dataset_analysis():
    """
    Example of comprehensive dataset analysis workflow
    """
    
    # Load dataset from parquet files
    analyzer = DatasetAnalyzer.from_parquet_dataset(
        dataset_path="/path/to/dataset/",
        lazy_loading=True
    )
    
    # Basic structural analysis
    print("=== Structural Analysis ===")
    incidence_analysis = analyzer.analyze_incidence_patterns()
    print(f"Nodes: {incidence_analysis['unique_nodes']}")
    print(f"Edges: {incidence_analysis['unique_edges']}")
    print(f"Incidences: {incidence_analysis['incidence_count']}")
    
    # Weight distribution analysis
    print("\n=== Weight Analysis ===")
    weight_analysis = analyzer.analyze_weight_distributions()
    print(f"Mean node weight: {weight_analysis['node_weights']['mean']:.3f}")
    print(f"Mean edge weight: {weight_analysis['edge_weights']['mean']:.3f}")
    
    # Centrality analysis using weights
    print("\n=== Centrality Analysis ===")
    centrality = analyzer.weighted_centrality_analysis()
    
    # Top nodes by weighted degree
    top_nodes = (
        centrality["weighted_degree"]
        .sort("weighted_degree", descending=True)
        .head(10)
    )
    print("Top 10 nodes by weighted degree:")
    print(top_nodes)
    
    # Property correlation analysis
    print("\n=== Property Correlation ===")
    numerical_props = ["weight", "degree", "importance_score"]
    correlations = analyzer.property_correlation_analysis(numerical_props)
    print(correlations)
    
    # Temporal analysis
    print("\n=== Temporal Analysis ===")
    temporal = analyzer.temporal_analysis(time_window="1d")
    print(f"Analysis span: {temporal['time_range']['span']}")
    print(f"Active periods: {temporal['activity_periods']}")
    
    return analyzer

if __name__ == "__main__":
    # Run example analysis
    analyzer = example_dataset_analysis()
```

## Weight and Incidence Analysis Applications

### 1. Social Network Analysis
```python
def social_network_analysis(analyzer: DatasetAnalyzer):
    """Social network analysis using weighted hypergraphs"""
    
    # Find influential users (high weighted degree)
    influential_users = (
        analyzer.weighted_centrality_analysis()["weighted_degree"]
        .filter(pl.col("normalized_weighted_degree") > 0.8)
        .sort("weighted_degree", descending=True)
    )
    
    # Analyze group dynamics (edge sizes and weights)
    group_analysis = (
        analyzer.hg.incidences._data
        .group_by("edge_id")
        .agg([
            pl.count().alias("group_size"),
            pl.col("weight").sum().alias("total_engagement"),
            pl.col("weight").var().alias("engagement_variance")
        ])
        .with_columns([
            # Classify group types
            pl.when(pl.col("group_size") < 5).then("small_group")
            .when(pl.col("group_size") < 20).then("medium_group") 
            .otherwise("large_group").alias("group_type")
        ])
    )
    
    return {
        "influential_users": influential_users,
        "group_analysis": group_analysis
    }

def recommendation_analysis(analyzer: DatasetAnalyzer):
    """Product recommendation using hypergraph weights"""
    
    # User-product interactions with purchase weights
    user_product_strength = (
        analyzer.hg.incidences._data
        .filter(pl.col("edge_id").str.starts_with("product_"))
        .group_by(["node_id", "edge_id"])
        .agg([
            pl.col("weight").sum().alias("interaction_strength"),
            pl.count().alias("interaction_frequency")
        ])
    )
    
    # Find similar users based on product interactions
    user_similarity = (
        user_product_strength
        .join(user_product_strength, on="edge_id", suffix="_other")
        .filter(pl.col("node_id") != pl.col("node_id_other"))
        .group_by(["node_id", "node_id_other"])
        .agg([
            pl.corr("interaction_strength", "interaction_strength_other").alias("similarity")
        ])
        .filter(pl.col("similarity") > 0.5)
    )
    
    return {
        "user_product_strength": user_product_strength,
        "user_similarity": user_similarity
    }
```

### 2. Performance Optimizations

```python
class OptimizedDatasetAnalyzer(DatasetAnalyzer):
    """Performance-optimized version for large datasets"""
    
    def __init__(self, hypergraph: DatasetHypergraph, enable_caching: bool = True):
        super().__init__(hypergraph)
        self.enable_caching = enable_caching
        self._cache = {}
    
    def cached_analysis(self, analysis_name: str, analysis_func, *args, **kwargs):
        """Cache expensive analysis results"""
        if not self.enable_caching:
            return analysis_func(*args, **kwargs)
        
        cache_key = f"{analysis_name}_{hash(str(args) + str(kwargs))}"
        
        if cache_key not in self._cache:
            self._cache[cache_key] = analysis_func(*args, **kwargs)
        
        return self._cache[cache_key]
    
    def streaming_weight_analysis(self, chunk_size: int = 10000) -> Dict[str, Any]:
        """Analyze weights using streaming for memory efficiency"""
        
        # Process incidences in chunks
        total_weight = 0
        count = 0
        min_weight = float('inf')
        max_weight = float('-inf')
        
        incidence_data = self.hg.incidences._data
        
        for i in range(0, incidence_data.height, chunk_size):
            chunk = incidence_data.slice(i, chunk_size)
            chunk_weights = chunk.select("weight").to_series()
            
            total_weight += chunk_weights.sum()
            count += chunk_weights.len()
            min_weight = min(min_weight, chunk_weights.min())
            max_weight = max(max_weight, chunk_weights.max())
        
        return {
            "total_weight": total_weight,
            "count": count,
            "mean_weight": total_weight / count if count > 0 else 0,
            "min_weight": min_weight,
            "max_weight": max_weight
        }
    
    def parallel_centrality_computation(self, n_threads: int = 4) -> Dict[str, pl.DataFrame]:
        """Compute centrality measures using parallel processing"""
        
        # Use Polars' built-in parallelism
        with pl.Config(n_threads=n_threads):
            return self.weighted_centrality_analysis()
```

This enhanced system provides:

1. **Rich Property Storage**: Support for categorical, numerical, temporal, vector, and text properties
2. **Efficient Parquet Integration**: Native loading and analysis of parquet datasets
3. **Advanced Weight Analysis**: Comprehensive weight distribution and centrality analysis
4. **Incidence Pattern Analysis**: Deep insights into node-edge relationships
5. **Performance Optimization**: Streaming, caching, and parallel processing capabilities
6. **Real-world Applications**: Social network analysis, recommendation systems, and more

The system is designed to handle large-scale datasets efficiently while providing rich analytical capabilities for understanding complex relationships in your data.
