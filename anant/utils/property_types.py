"""
Property Type Management System

Advanced property type detection, validation, and optimization for the anant library.
Supports categorical, numerical, temporal, text, vector, and JSON property types
with automatic optimization and analysis capabilities.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
import json
import re
from enum import Enum


class PropertyType(Enum):
    """Enumeration of supported property types"""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical" 
    TEMPORAL = "temporal"
    TEXT = "text"
    VECTOR = "vector"
    JSON = "json"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class PropertyTypeManager:
    """
    Manage different property types with appropriate optimizations
    
    This class automatically detects property types and applies appropriate
    storage optimizations and analysis capabilities.
    """
    
    SUPPORTED_TYPES = {
        PropertyType.CATEGORICAL: [pl.Categorical, pl.Utf8],
        PropertyType.NUMERICAL: [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64],
        PropertyType.TEMPORAL: [pl.Date, pl.Datetime, pl.Duration, pl.Time],
        PropertyType.TEXT: [pl.Utf8],
        PropertyType.VECTOR: [pl.List(pl.Float32), pl.List(pl.Float64)],
        PropertyType.JSON: [pl.Utf8],
        PropertyType.BOOLEAN: [pl.Boolean],
    }
    
    def __init__(self):
        self.type_cache = {}
        self.optimization_stats = {}
    
    def detect_property_type(self, column: pl.Series, sample_size: int = 1000) -> PropertyType:
        """
        Automatically detect and categorize property types
        
        Args:
            column: Polars Series to analyze
            sample_size: Number of samples to use for detection
            
        Returns:
            Detected PropertyType
        """
        # Use cache if available
        cache_key = f"{column.name}_{column.dtype}"
        if cache_key in self.type_cache:
            return self.type_cache[cache_key]
        
        # Sample data for analysis
        sample = column.head(min(sample_size, column.len()))
        non_null_sample = sample.drop_nulls()
        
        if non_null_sample.len() == 0:
            return PropertyType.UNKNOWN
        
        detected_type = self._analyze_column_content(non_null_sample)
        self.type_cache[cache_key] = detected_type
        
        return detected_type
    
    def _analyze_column_content(self, sample: pl.Series) -> PropertyType:
        """Analyze column content to determine type"""
        
        # Check current dtype first
        if sample.dtype in [pl.Boolean]:
            return PropertyType.BOOLEAN
        
        if sample.dtype in [pl.Date, pl.Datetime, pl.Duration, pl.Time]:
            return PropertyType.TEMPORAL
        
        if sample.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            return PropertyType.NUMERICAL
        
        # For string columns, analyze content
        if sample.dtype == pl.Utf8:
            return self._analyze_string_content(sample)
        
        # For list columns, check if they're vectors
        if sample.dtype == pl.List:
            # Try to infer inner type from actual data
            try:
                first_non_null = sample.drop_nulls().first()
                if first_non_null is not None and isinstance(first_non_null, list) and len(first_non_null) > 0:
                    if all(isinstance(x, (int, float)) for x in first_non_null):
                        return PropertyType.VECTOR
            except Exception:
                pass
            return PropertyType.JSON
        
        return PropertyType.UNKNOWN
    
    def _analyze_string_content(self, sample: pl.Series) -> PropertyType:
        """Analyze string column content"""
        
        # Sample some values for analysis
        values = sample.head(100).to_list()
        
        # Check for JSON content
        json_count = 0
        for value in values:
            if self._is_json_string(value):
                json_count += 1
        
        if json_count / len(values) > 0.5:  # More than 50% are JSON
            return PropertyType.JSON
        
        # Check for temporal strings
        temporal_count = 0
        for value in values:
            if self._is_temporal_string(value):
                temporal_count += 1
        
        if temporal_count / len(values) > 0.7:  # More than 70% are temporal
            return PropertyType.TEMPORAL
        
        # Check for numerical strings
        numerical_count = 0
        for value in values:
            if self._is_numerical_string(value):
                numerical_count += 1
        
        if numerical_count / len(values) > 0.8:  # More than 80% are numerical
            return PropertyType.NUMERICAL
        
        # Check uniqueness for categorical vs text
        unique_ratio = sample.n_unique() / sample.len()
        
        if unique_ratio < 0.1:  # Very low uniqueness - likely categorical
            return PropertyType.CATEGORICAL
        elif unique_ratio > 0.8:  # High uniqueness - likely text
            return PropertyType.TEXT
        else:
            # Medium uniqueness - could be either, check average length
            avg_length = sample.str.len_chars().mean()
            if avg_length and isinstance(avg_length, (int, float)) and avg_length < 50:
                return PropertyType.CATEGORICAL
            else:
                return PropertyType.TEXT
    
    def _is_json_string(self, value: str) -> bool:
        """Check if string is valid JSON"""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def _is_temporal_string(self, value: str) -> bool:
        """Check if string represents temporal data"""
        temporal_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
        
        for pattern in temporal_patterns:
            if re.match(pattern, value):
                return True
        
        return False
    
    def _is_numerical_string(self, value: str) -> bool:
        """Check if string represents numerical data"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def optimize_column_storage(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """
        Apply type-specific storage optimizations
        
        Args:
            df: DataFrame to optimize
            column: Column name to optimize
            
        Returns:
            DataFrame with optimized column
        """
        if column not in df.columns:
            return df
        
        prop_type = self.detect_property_type(df[column])
        original_memory = df[column].estimated_size("bytes")
        
        if prop_type == PropertyType.CATEGORICAL:
            optimized_df = df.with_columns([
                pl.col(column).cast(pl.Categorical).alias(column)
            ])
        
        elif prop_type == PropertyType.NUMERICAL:
            # Try to downcast numerical types
            optimized_df = self._optimize_numerical_column(df, column)
        
        elif prop_type == PropertyType.TEMPORAL:
            # Convert temporal strings to proper temporal types
            optimized_df = self._optimize_temporal_column(df, column)
        
        elif prop_type == PropertyType.VECTOR:
            # Optimize vector storage
            optimized_df = self._optimize_vector_column(df, column)
        
        else:
            # No optimization for text/json/unknown types
            optimized_df = df
        
        # Track optimization statistics
        if column in optimized_df.columns:
            new_memory = optimized_df[column].estimated_size("bytes")
            self.optimization_stats[column] = {
                "type": prop_type.value,
                "original_memory": original_memory,
                "optimized_memory": new_memory,
                "reduction_ratio": (original_memory - new_memory) / original_memory if original_memory > 0 else 0
            }
        
        return optimized_df
    
    def _optimize_numerical_column(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Optimize numerical column by downcasting"""
        col_data = df[column]
        
        # Skip if already optimized or contains nulls that would complicate analysis
        if col_data.dtype in [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.Float32]:
            return df
        
        col_min = col_data.min()
        col_max = col_data.max()
        
        if col_min is None or col_max is None:
            return df
        
        # Ensure we're working with numeric types
        if not isinstance(col_min, (int, float)) or not isinstance(col_max, (int, float)):
            return df
        
        # Integer optimization
        if col_data.dtype in [pl.Int32, pl.Int64]:
            if col_min >= 0:
                # Unsigned integer optimization
                if col_max <= 255:
                    return df.with_columns([pl.col(column).cast(pl.UInt8)])
                elif col_max <= 65535:
                    return df.with_columns([pl.col(column).cast(pl.UInt16)])
                elif col_max <= 4294967295:
                    return df.with_columns([pl.col(column).cast(pl.UInt32)])
            else:
                # Signed integer optimization
                if col_min >= -128 and col_max <= 127:
                    return df.with_columns([pl.col(column).cast(pl.Int8)])
                elif col_min >= -32768 and col_max <= 32767:
                    return df.with_columns([pl.col(column).cast(pl.Int16)])
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    return df.with_columns([pl.col(column).cast(pl.Int32)])
        
        # Float optimization
        elif col_data.dtype == pl.Float64:
            # Check if values fit in Float32 range
            if col_min >= -3.4e38 and col_max <= 3.4e38:
                return df.with_columns([pl.col(column).cast(pl.Float32)])
        
        return df
    
    def _optimize_temporal_column(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Optimize temporal column by converting strings to proper types"""
        if df[column].dtype != pl.Utf8:
            return df
        
        # Try different temporal parsing strategies
        try:
            # Try parsing as datetime first
            return df.with_columns([
                pl.col(column).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            ])
        except:
            try:
                # Try parsing as date
                return df.with_columns([
                    pl.col(column).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                ])
            except:
                # If parsing fails, keep as string
                return df
    
    def _optimize_vector_column(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Optimize vector column storage"""
        # For now, just ensure it's using appropriate precision
        if df[column].dtype == pl.List:
            # Try to optimize list elements if they are numeric
            try:
                # Sample some values to check content
                sample_values = df[column].drop_nulls().head(10).to_list()
                if sample_values and all(isinstance(val, list) for val in sample_values):
                    # Check if all elements are float64 that could be float32
                    first_list = sample_values[0]
                    if first_list and all(isinstance(x, float) for x in first_list):
                        # For simplicity, keep as is - list optimization is complex
                        return df
            except Exception:
                pass
        
        return df
    
    def detect_all_property_types(self, df: pl.DataFrame) -> Dict[str, PropertyType]:
        """
        Detect property types for all columns in a DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to PropertyType
        """
        type_map = {}
        for column in df.columns:
            type_map[column] = self.detect_property_type(df[column])
        return type_map
    
    def analyze_memory_usage(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze memory usage of DataFrame columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory usage analysis
        """
        total_memory = 0
        column_analysis = {}
        
        for column in df.columns:
            col_memory = df[column].estimated_size("bytes")
            total_memory += col_memory
            
            col_data = df[column]
            prop_type = self.detect_property_type(col_data)
            
            column_analysis[column] = {
                "memory_bytes": col_memory,
                "property_type": prop_type.value,
                "dtype": str(col_data.dtype),
                "null_count": col_data.null_count(),
                "unique_count": len(col_data.unique()),
                "compression_ratio": 1.0  # Placeholder for actual compression analysis
            }
        
        return {
            "total_memory_bytes": total_memory,
            "columns": column_analysis,
            "optimization_opportunities": self._suggest_optimizations(df)
        }
    
    def _suggest_optimizations(self, df: pl.DataFrame) -> List[str]:
        """Suggest memory optimization opportunities"""
        suggestions = []
        
        for column in df.columns:
            col_data = df[column]
            prop_type = self.detect_property_type(col_data)
            
            if prop_type == PropertyType.CATEGORICAL:
                if col_data.dtype != pl.Categorical:
                    suggestions.append(f"Convert {column} to categorical type")
            
            elif prop_type == PropertyType.NUMERICAL:
                # Check for integer optimization
                if col_data.dtype in [pl.Float64, pl.Int64]:
                    try:
                        col_min = col_data.min()
                        col_max = col_data.max()
                        if col_min is not None and col_max is not None:
                            if isinstance(col_min, (int, float)) and isinstance(col_max, (int, float)):
                                if col_min >= 0 and col_max <= 255:
                                    suggestions.append(f"Consider UInt8 for {column}")
                                elif col_min >= -128 and col_max <= 127:
                                    suggestions.append(f"Consider Int8 for {column}")
                    except Exception:
                        pass
        
        return suggestions
    
    def optimize_dataframe_storage(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize storage for entire DataFrame
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            DataFrame with optimized column types
        """
        result_df = df.clone()
        
        for column in df.columns:
            result_df = self.optimize_column_storage(result_df, column)
        
        return result_df
        """Get optimization statistics report"""
        if not self.optimization_stats:
            return {"message": "No optimizations performed yet"}
        
        total_original = sum(stats["original_memory"] for stats in self.optimization_stats.values())
        total_optimized = sum(stats["optimized_memory"] for stats in self.optimization_stats.values())
        
        return {
            "columns_optimized": len(self.optimization_stats),
            "total_memory_saved": total_original - total_optimized,
            "total_reduction_ratio": (total_original - total_optimized) / total_original if total_original > 0 else 0,
            "by_column": self.optimization_stats.copy(),
            "type_distribution": self._get_type_distribution()
        }
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of detected types"""
        type_counts = {}
        for stats in self.optimization_stats.values():
            prop_type = stats["type"]
            type_counts[prop_type] = type_counts.get(prop_type, 0) + 1
        return type_counts