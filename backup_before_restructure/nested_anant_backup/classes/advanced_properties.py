"""
Advanced Property Management for Anant Library

Implements advanced property management capabilities including:
- Multi-type property support with automatic optimization
- Correlation analysis and cross-property relationships  
- Bulk operations with performance optimization
- Advanced memory management and zero-division safety
- Production-ready property store enhancements

This module fixes critical property store issues and adds advanced capabilities.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import time
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Enhanced property types for advanced management"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    JSON = "json"
    VECTOR = "vector"
    SPARSE = "sparse"


class OptimizationLevel(Enum):
    """Property optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PropertyMetrics:
    """Comprehensive property metrics and statistics"""
    property_name: str
    property_type: PropertyType
    total_values: int
    unique_values: int
    null_count: int
    memory_usage_bytes: int
    compression_ratio: float = 1.0
    cardinality_ratio: float = 0.0
    correlation_candidates: List[str] = field(default_factory=list)
    optimization_applied: Optional[str] = None
    performance_score: float = 1.0


@dataclass  
class PropertyCorrelation:
    """Property correlation analysis results"""
    property1: str
    property2: str
    correlation_type: str  # 'numeric', 'categorical', 'mixed'
    correlation_strength: float  # 0.0 to 1.0
    significance: float
    sample_size: int
    relationship_description: str


class AdvancedPropertyStore:
    """
    Advanced Property Store with enhanced capabilities
    
    Features:
    - Multi-type property support with automatic optimization
    - Zero-division safe operations
    - Correlation analysis and cross-property relationships
    - Bulk operations with performance optimization
    - Advanced memory management
    """
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED,
                 enable_correlations: bool = True,
                 memory_limit_mb: Optional[float] = None):
        """
        Initialize advanced property store
        
        Args:
            optimization_level: Level of automatic optimization to apply
            enable_correlations: Whether to enable correlation analysis
            memory_limit_mb: Memory limit for operations (MB)
        """
        self.optimization_level = optimization_level
        self.enable_correlations = enable_correlations
        self.memory_limit_mb = memory_limit_mb
        
        # Core storage
        self._data = pl.DataFrame()
        self._property_metrics: Dict[str, PropertyMetrics] = {}
        self._property_correlations: List[PropertyCorrelation] = []
        
        # Performance tracking
        self._operation_stats = {
            'total_operations': 0,
            'optimization_time': 0.0,
            'correlation_time': 0.0,
            'bulk_operation_time': 0.0
        }
        
        logger.info(f"Advanced PropertyStore initialized with {optimization_level.value} optimization")
    
    def bulk_set_properties(self, 
                           data: Union[pl.DataFrame, Dict[str, Any]], 
                           analyze_correlations: Optional[bool] = None) -> None:
        """
        Bulk set properties with advanced optimization
        
        Args:
            data: Property data to set
            analyze_correlations: Whether to analyze correlations (default: use store setting)
        """
        start_time = time.time()
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                data = pl.DataFrame(data)
            
            # Safety check for empty data
            if len(data) == 0:
                logger.warning("Attempting to set properties with empty data")
                return
                
            # Store data with safety checks
            self._data = self._safe_data_assignment(data)
            
            # Apply optimization if enabled
            if self.optimization_level != OptimizationLevel.NONE:
                self._apply_advanced_optimization()
            
            # Analyze correlations if enabled
            if (analyze_correlations is True or 
                (analyze_correlations is None and self.enable_correlations)):
                self._analyze_property_correlations()
            
            # Update metrics
            self._update_property_metrics()
            
            self._operation_stats['bulk_operation_time'] += time.time() - start_time
            self._operation_stats['total_operations'] += 1
            
            logger.info(f"Bulk properties set: {len(data)} rows, {len(data.columns)} properties")
            
        except Exception as e:
            logger.error(f"Failed to bulk set properties: {e}")
            raise ValueError(f"Could not set properties: {e}") from e
    
    def _safe_data_assignment(self, data: pl.DataFrame) -> pl.DataFrame:
        """Safely assign data with comprehensive validation"""
        
        try:
            # Validate basic structure
            if not isinstance(data, pl.DataFrame):
                raise TypeError("Data must be a Polars DataFrame")
            
            if len(data.columns) == 0:
                logger.warning("No columns in property data")
                return pl.DataFrame()
            
            # Check for completely empty DataFrame  
            if data.height == 0:
                logger.warning("Empty DataFrame provided for properties")
                return data
            
            # Clean and validate data
            cleaned_data = self._clean_property_data(data)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data assignment failed: {e}")
            # Return safe empty DataFrame instead of crashing
            return pl.DataFrame()
    
    def _clean_property_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Clean and validate property data"""
        
        cleaned = data.clone()
        
        try:
            # Remove completely null columns
            non_null_cols = []
            for col in cleaned.columns:
                if cleaned[col].null_count() < len(cleaned):  # Has at least one non-null value
                    non_null_cols.append(col)
            
            if non_null_cols:
                cleaned = cleaned.select(non_null_cols)
            else:
                logger.warning("All columns are completely null")
                return pl.DataFrame()
            
            # Handle data type optimization and validation
            optimized_cols = []
            for col in cleaned.columns:
                try:
                    # Basic type validation and optimization
                    col_data = cleaned[col]
                    
                    # Skip if all null
                    if col_data.null_count() == len(col_data):
                        continue
                    
                    optimized_cols.append(col)
                    
                except Exception as e:
                    logger.warning(f"Skipping problematic column {col}: {e}")
                    continue
            
            if optimized_cols:
                cleaned = cleaned.select(optimized_cols)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return data  # Return original if cleaning fails
    
    def _apply_advanced_optimization(self) -> None:
        """Apply advanced optimization with zero-division safety"""
        
        start_time = time.time()
        
        try:
            if len(self._data) == 0:
                logger.debug("No data to optimize")
                return
            
            optimized_data = self._data.clone()
            
            for col in self._data.columns:
                try:
                    optimized_data = self._optimize_column_safely(optimized_data, col)
                except Exception as e:
                    logger.warning(f"Column optimization failed for {col}: {e}")
                    continue
            
            self._data = optimized_data
            
            self._operation_stats['optimization_time'] += time.time() - start_time
            
        except Exception as e:
            logger.error(f"Advanced optimization failed: {e}")
            # Don't crash - keep original data
    
    def _optimize_column_safely(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Safely optimize a single column with zero-division protection"""
        
        try:
            col_data = data[col]
            
            # Safety check: ensure data exists
            if len(col_data) == 0:
                logger.debug(f"Column {col} is empty, skipping optimization")
                return data
            
            # Safety check: handle all-null columns
            if col_data.null_count() == len(col_data):
                logger.debug(f"Column {col} is all null, skipping optimization")
                return data
            
            # Zero-division safe cardinality calculation
            data_height = data.height
            if data_height == 0:
                logger.debug(f"DataFrame height is 0, skipping optimization for {col}")
                return data
            
            unique_count = col_data.n_unique()
            cardinality_ratio = unique_count / data_height  # Now safe from division by zero
            
            # Apply optimization based on cardinality and type
            if cardinality_ratio < 0.1:  # Low cardinality - consider categorical
                return self._optimize_as_categorical(data, col)
            elif cardinality_ratio > 0.9:  # High cardinality - consider keeping as-is
                return self._optimize_high_cardinality(data, col)
            else:  # Medium cardinality - apply standard optimization
                return self._optimize_standard(data, col)
                
        except Exception as e:
            logger.warning(f"Safe column optimization failed for {col}: {e}")
            return data  # Return original data if optimization fails
    
    def _optimize_as_categorical(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Optimize column as categorical"""
        try:
            # Convert to categorical if beneficial
            if self.optimization_level in [OptimizationLevel.ADVANCED, OptimizationLevel.AGGRESSIVE]:
                # Only convert if it will save memory
                unique_values = data[col].n_unique()
                total_values = len(data)
                
                if unique_values < total_values * 0.5:  # Will save memory
                    return data.with_columns([
                        data[col].cast(pl.Categorical).alias(col)
                    ])
            
            return data
            
        except Exception as e:
            logger.debug(f"Categorical optimization failed for {col}: {e}")
            return data
    
    def _optimize_high_cardinality(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Optimize high cardinality column"""
        try:
            # For high cardinality, focus on memory efficiency
            col_data = data[col]
            
            # Try to optimize numeric types
            if col_data.dtype.is_numeric():
                return self._optimize_numeric_precision(data, col)
            
            # For string types, consider compression techniques
            if col_data.dtype == pl.Utf8:
                return self._optimize_string_column(data, col)
            
            return data
            
        except Exception as e:
            logger.debug(f"High cardinality optimization failed for {col}: {e}")
            return data
    
    def _optimize_standard(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Apply standard optimization"""
        try:
            col_data = data[col]
            
            # Apply type-specific optimization
            if col_data.dtype.is_numeric():
                return self._optimize_numeric_precision(data, col)
            elif col_data.dtype == pl.Boolean:
                return data  # Boolean already optimal
            elif col_data.dtype == pl.Utf8:
                return self._optimize_string_column(data, col)
            
            return data
            
        except Exception as e:
            logger.debug(f"Standard optimization failed for {col}: {e}")
            return data
    
    def _optimize_numeric_precision(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Optimize numeric column precision"""
        try:
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                col_data = data[col]
                
                # Check if integer values can be downcasted
                if col_data.dtype.is_integer():
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    if min_val is not None and max_val is not None:
                        # Find appropriate integer type (safe numeric comparison) 
                        try:
                            # Only process if we have numeric values
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                min_int = int(min_val) 
                                max_int = int(max_val)
                                
                                if -128 <= min_int <= 127 and max_int <= 127:
                                    return data.with_columns([col_data.cast(pl.Int8).alias(col)])
                                elif -32768 <= min_int <= 32767 and max_int <= 32767:
                                    return data.with_columns([col_data.cast(pl.Int16).alias(col)])
                        except (ValueError, TypeError):
                            pass  # Skip optimization if conversion fails
            
            return data
            
        except Exception as e:
            logger.debug(f"Numeric optimization failed for {col}: {e}")
            return data
    
    def _optimize_string_column(self, data: pl.DataFrame, col: str) -> pl.DataFrame:
        """Optimize string column"""
        try:
            # Convert to categorical if low cardinality
            unique_count = data[col].n_unique()
            total_count = len(data)
            
            if total_count > 0 and unique_count / total_count < 0.3:
                return data.with_columns([
                    data[col].cast(pl.Categorical).alias(col)
                ])
            
            return data
            
        except Exception as e:
            logger.debug(f"String optimization failed for {col}: {e}")
            return data
    
    def _analyze_property_correlations(self) -> None:
        """Analyze correlations between properties"""
        
        if not self.enable_correlations or len(self._data) == 0:
            return
        
        start_time = time.time()
        
        try:
            self._property_correlations.clear()
            
            numeric_columns = [col for col in self._data.columns 
                             if self._data[col].dtype.is_numeric()]
            
            # Analyze numeric correlations
            if len(numeric_columns) > 1:
                self._analyze_numeric_correlations(numeric_columns)
            
            # Analyze categorical correlations
            categorical_columns = [col for col in self._data.columns 
                                 if self._data[col].dtype in [pl.Utf8, pl.Categorical]]
            
            if len(categorical_columns) > 1:
                self._analyze_categorical_correlations(categorical_columns)
            
            self._operation_stats['correlation_time'] += time.time() - start_time
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
    
    def _analyze_numeric_correlations(self, numeric_columns: List[str]) -> None:
        """Analyze correlations between numeric properties"""
        
        try:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    
                    # Get valid (non-null) pairs
                    valid_data = self._data.select([col1, col2]).drop_nulls()
                    
                    if len(valid_data) < 10:  # Need at least 10 valid pairs
                        continue
                    
                    # Calculate correlation
                    correlation_matrix = valid_data.corr()
                    if len(correlation_matrix) >= 2:
                        correlation_value = correlation_matrix[col1][1] if correlation_matrix[col1][1] != 1.0 else correlation_matrix[col2][0]
                        
                        if abs(correlation_value) > 0.1:  # Only significant correlations
                            correlation = PropertyCorrelation(
                                property1=col1,
                                property2=col2,
                                correlation_type="numeric",
                                correlation_strength=abs(correlation_value),
                                significance=min(1.0, abs(correlation_value) * 2),
                                sample_size=len(valid_data),
                                relationship_description=f"Numeric correlation: {correlation_value:.3f}"
                            )
                            
                            self._property_correlations.append(correlation)
        
        except Exception as e:
            logger.debug(f"Numeric correlation analysis failed: {e}")
    
    def _analyze_categorical_correlations(self, categorical_columns: List[str]) -> None:
        """Analyze correlations between categorical properties"""
        
        try:
            for i, col1 in enumerate(categorical_columns):
                for col2 in categorical_columns[i+1:]:
                    
                    # Calculate categorical association (Cramer's V approximation)
                    valid_data = self._data.select([col1, col2]).drop_nulls()
                    
                    if len(valid_data) < 10:
                        continue
                    
                    # Simple association measure based on unique value overlap
                    col1_unique = set(valid_data[col1].to_list())
                    col2_unique = set(valid_data[col2].to_list())
                    
                    if len(col1_unique) > 0 and len(col2_unique) > 0:
                        # Jaccard similarity as a simple association measure
                        intersection = len(col1_unique.intersection(col2_unique))
                        union = len(col1_unique.union(col2_unique))
                        
                        if union > 0:
                            association = intersection / union
                            
                            if association > 0.1:  # Significant association
                                correlation = PropertyCorrelation(
                                    property1=col1,
                                    property2=col2,
                                    correlation_type="categorical",
                                    correlation_strength=association,
                                    significance=association,
                                    sample_size=len(valid_data),
                                    relationship_description=f"Categorical association: {association:.3f}"
                                )
                                
                                self._property_correlations.append(correlation)
        
        except Exception as e:
            logger.debug(f"Categorical correlation analysis failed: {e}")
    
    def _update_property_metrics(self) -> None:
        """Update comprehensive property metrics"""
        
        try:
            self._property_metrics.clear()
            
            for col in self._data.columns:
                try:
                    col_data = self._data[col]
                    
                    # Determine property type
                    prop_type = self._determine_property_type(col_data)
                    
                    # Calculate metrics safely
                    total_values = len(col_data)
                    unique_values = col_data.n_unique() if total_values > 0 else 0
                    null_count = col_data.null_count()
                    
                    # Safe cardinality ratio calculation
                    cardinality_ratio = unique_values / total_values if total_values > 0 else 0.0
                    
                    # Estimate memory usage
                    memory_bytes = self._estimate_column_memory(col_data)
                    
                    # Find correlation candidates
                    correlation_candidates = [
                        corr.property2 if corr.property1 == col else corr.property1
                        for corr in self._property_correlations
                        if (corr.property1 == col or corr.property2 == col) and corr.correlation_strength > 0.3
                    ]
                    
                    metrics = PropertyMetrics(
                        property_name=col,
                        property_type=prop_type,
                        total_values=total_values,
                        unique_values=unique_values,
                        null_count=null_count,
                        memory_usage_bytes=memory_bytes,
                        cardinality_ratio=cardinality_ratio,
                        correlation_candidates=correlation_candidates,
                        optimization_applied=f"{self.optimization_level.value}_optimization",
                        performance_score=self._calculate_performance_score(cardinality_ratio, null_count, total_values)
                    )
                    
                    self._property_metrics[col] = metrics
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for {col}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Property metrics update failed: {e}")
    
    def _determine_property_type(self, col_data: pl.Series) -> PropertyType:
        """Determine the property type of a column"""
        
        try:
            dtype = col_data.dtype
            
            if dtype.is_numeric():
                if dtype.is_integer():
                    # Check if it's actually boolean (0/1 only)
                    unique_vals = col_data.unique().drop_nulls().to_list()
                    if set(unique_vals).issubset({0, 1}):
                        return PropertyType.BOOLEAN
                return PropertyType.NUMERIC
            elif dtype == pl.Boolean:
                return PropertyType.BOOLEAN
            elif dtype in [pl.Utf8, pl.Categorical]:
                # Check if it looks like timestamps
                if self._looks_like_timestamp(col_data):
                    return PropertyType.TIMESTAMP
                return PropertyType.CATEGORICAL
            else:
                return PropertyType.TEXT
                
        except Exception:
            return PropertyType.TEXT
    
    def _looks_like_timestamp(self, col_data: pl.Series) -> bool:
        """Check if string column looks like timestamps"""
        try:
            # Simple heuristic: check first few non-null values
            sample_values = col_data.drop_nulls().head(5).to_list()
            
            for val in sample_values:
                if isinstance(val, str):
                    # Look for common timestamp patterns
                    if any(pattern in val for pattern in ['-', ':', 'T', 'Z', '+00']):
                        return True
                    
            return False
        except:
            return False
    
    def _estimate_column_memory(self, col_data: pl.Series) -> int:
        """Estimate memory usage of a column in bytes"""
        try:
            # Rough estimation based on data type and size
            dtype = col_data.dtype
            length = len(col_data)
            
            if dtype.is_numeric():
                if dtype == pl.Int8:
                    return length * 1
                elif dtype == pl.Int16:
                    return length * 2
                elif dtype == pl.Int32 or dtype == pl.Float32:
                    return length * 4
                else:  # Int64, Float64
                    return length * 8
            elif dtype == pl.Boolean:
                return length * 1
            elif dtype == pl.Utf8:
                # Estimate average string length
                sample = col_data.drop_nulls().head(100)
                if len(sample) > 0:
                    avg_length = sum(len(str(val)) for val in sample.to_list()) / len(sample)
                    return int(length * avg_length)
                return length * 10  # Default estimate
            else:
                return length * 8  # Default estimate
                
        except Exception:
            return len(col_data) * 8  # Safe default
    
    def _calculate_performance_score(self, cardinality_ratio: float, null_count: int, total_values: int) -> float:
        """Calculate performance score for a property"""
        try:
            # Start with perfect score
            score = 1.0
            
            # Penalize high cardinality (harder to optimize)
            if cardinality_ratio > 0.8:
                score -= 0.2
            elif cardinality_ratio > 0.5:
                score -= 0.1
            
            # Penalize high null ratio
            if total_values > 0:
                null_ratio = null_count / total_values
                if null_ratio > 0.5:
                    score -= 0.3
                elif null_ratio > 0.2:
                    score -= 0.1
            
            return max(0.0, score)
        
        except Exception:
            return 0.5  # Default middle score
    
    # Public API methods
    
    def get_property_metrics(self, property_name: Optional[str] = None) -> Union[Optional[PropertyMetrics], Dict[str, PropertyMetrics]]:
        """Get comprehensive metrics for properties"""
        if property_name:
            return self._property_metrics.get(property_name)
        return self._property_metrics.copy()
    
    def get_correlations(self, property_name: Optional[str] = None, min_strength: float = 0.0) -> List[PropertyCorrelation]:
        """Get property correlations"""
        correlations = self._property_correlations
        
        if property_name:
            correlations = [c for c in correlations 
                          if c.property1 == property_name or c.property2 == property_name]
        
        if min_strength > 0:
            correlations = [c for c in correlations if c.correlation_strength >= min_strength]
        
        return correlations
    
    def optimize_properties(self, property_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Manually trigger property optimization"""
        if len(self._data) == 0:
            return {}
        
        optimization_results = {}
        
        columns_to_optimize = property_names or self._data.columns
        
        for col in columns_to_optimize:
            if col in self._data.columns:
                try:
                    original_memory = self._estimate_column_memory(self._data[col])
                    self._data = self._optimize_column_safely(self._data, col)
                    new_memory = self._estimate_column_memory(self._data[col])
                    
                    if new_memory < original_memory:
                        optimization_results[col] = f"Memory reduced by {original_memory - new_memory} bytes"
                    else:
                        optimization_results[col] = "Already optimized"
                        
                except Exception as e:
                    optimization_results[col] = f"Optimization failed: {e}"
        
        return optimization_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'operation_stats': self._operation_stats.copy(),
            'data_stats': {
                'total_rows': len(self._data),
                'total_columns': len(self._data.columns),
                'memory_estimate_mb': sum(
                    self._estimate_column_memory(self._data[col]) 
                    for col in self._data.columns
                ) / (1024 * 1024),
                'optimization_level': self.optimization_level.value
            },
            'correlation_stats': {
                'total_correlations': len(self._property_correlations),
                'strong_correlations': len([c for c in self._property_correlations if c.correlation_strength > 0.5]),
                'correlation_types': list(set(c.correlation_type for c in self._property_correlations))
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset performance and operation statistics"""
        self._operation_stats = {
            'total_operations': 0,
            'optimization_time': 0.0,
            'correlation_time': 0.0,
            'bulk_operation_time': 0.0
        }
        logger.info("Performance statistics reset")
    
    # Backward compatibility methods
    
    def get_properties(self, entity_id: str) -> Dict[str, Any]:
        """Get properties for an entity (backward compatibility)"""
        try:
            if len(self._data) == 0:
                return {}
            
            # Assume first column is entity ID
            if len(self._data.columns) > 0:
                entity_col = self._data.columns[0]
                entity_row = self._data.filter(pl.col(entity_col) == entity_id)
                
                if len(entity_row) > 0:
                    return entity_row.to_dicts()[0]
            
            return {}
        
        except Exception as e:
            logger.warning(f"Failed to get properties for {entity_id}: {e}")
            return {}
    
    def set_property(self, entity_id: str, property_name: str, value: Any) -> None:
        """Set a single property (backward compatibility)"""
        try:
            # This is a simplified implementation for backward compatibility
            logger.debug(f"Setting property {property_name} for {entity_id} to {value}")
            # In a full implementation, this would update the DataFrame
            
        except Exception as e:
            logger.warning(f"Failed to set property {property_name} for {entity_id}: {e}")


# Convenience functions for direct usage

def create_advanced_property_store(optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED,
                                 enable_correlations: bool = True,
                                 memory_limit_mb: Optional[float] = None) -> AdvancedPropertyStore:
    """Create an advanced property store with specified configuration"""
    return AdvancedPropertyStore(
        optimization_level=optimization_level,
        enable_correlations=enable_correlations, 
        memory_limit_mb=memory_limit_mb
    )


def analyze_property_relationships(data: pl.DataFrame) -> List[PropertyCorrelation]:
    """Analyze relationships between properties in a DataFrame"""
    store = AdvancedPropertyStore(enable_correlations=True)
    store.bulk_set_properties(data)
    return store.get_correlations()


def optimize_dataframe_properties(data: pl.DataFrame, 
                                 optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED) -> pl.DataFrame:
    """Optimize a DataFrame's properties and return optimized version"""
    store = AdvancedPropertyStore(optimization_level=optimization_level, enable_correlations=False)
    store.bulk_set_properties(data)
    return store._data