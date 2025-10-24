"""
Data Workers
===========

Ray workers for distributed data processing tasks using Polars.
"""

import ray
import time
import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional, Union
import io
import json
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@ray.remote
class DataProcessor:
    """Distributed data processor using Ray and Polars."""
    
    def __init__(self, worker_id: str = "worker"):
        self.worker_id = worker_id
        self.processed_batches = 0
        self.processing_history = []
        
    def process_csv_data(self, csv_data: str, operations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process CSV data with specified operations."""
        operations = operations or ["validate", "clean"]
        start_time = time.time()
        
        try:
            # Read CSV data
            csv_file = io.StringIO(csv_data)
            df = pl.read_csv(csv_file)
            
            results = {
                "worker_id": self.worker_id,
                "original_shape": df.shape,
                "operations_applied": operations
            }
            
            # Apply operations
            for operation in operations:
                if operation == "clean":
                    df, clean_stats = self._clean_data(df)
                    results["clean_stats"] = clean_stats
                elif operation == "validate":
                    validation_result = self._validate_data(df)
                    results["validation"] = validation_result
                elif operation == "transform":
                    df, transform_stats = self._transform_data(df)
                    results["transform_stats"] = transform_stats
                elif operation == "aggregate":
                    agg_result = self._aggregate_data(df)
                    results["aggregation"] = agg_result
            
            # Final results
            results.update({
                "final_shape": df.shape,
                "processing_time": time.time() - start_time,
                "processed_at": datetime.now().isoformat(),
                "sample_data": df.head().to_dict(as_series=False) if df.height > 0 else {}
            })
            
            self.processed_batches += 1
            return {"status": "success", "data": results}
            
        except Exception as e:
            return {
                "status": "error",
                "worker_id": self.worker_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_json_data(self, json_data: List[Dict[str, Any]], 
                         operations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process JSON data with specified operations."""
        operations = operations or ["validate", "transform"]
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pl.DataFrame(json_data)
            
            results = {
                "worker_id": self.worker_id,
                "original_records": len(json_data),
                "original_columns": df.columns,
                "operations_applied": operations
            }
            
            # Apply operations
            for operation in operations:
                if operation == "clean":
                    df, clean_stats = self._clean_data(df)
                    results["clean_stats"] = clean_stats
                elif operation == "validate":
                    validation_result = self._validate_data(df)
                    results["validation"] = validation_result
                elif operation == "transform":
                    df, transform_stats = self._transform_data(df)
                    results["transform_stats"] = transform_stats
                elif operation == "aggregate":
                    agg_result = self._aggregate_data(df)
                    results["aggregation"] = agg_result
            
            # Final results
            results.update({
                "final_records": df.height,
                "final_columns": df.columns,
                "processing_time": time.time() - start_time,
                "processed_at": datetime.now().isoformat(),
                "sample_data": df.head().to_dict(as_series=False) if df.height > 0 else {}
            })
            
            self.processed_batches += 1
            return {"status": "success", "data": results}
            
        except Exception as e:
            return {
                "status": "error",
                "worker_id": self.worker_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _validate_data(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Validate data quality using Polars."""
        validation_result = {
            "total_rows": df.height,
            "total_columns": df.width,
            "column_names": df.columns,
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Check for null values
        null_counts = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                null_counts[col] = null_count
        validation_result["null_values"] = null_counts
        
        # Check for duplicate rows
        duplicate_count = df.height - df.unique().height
        validation_result["duplicate_rows"] = duplicate_count
        
        # Get numeric column statistics
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        if numeric_columns:
            numeric_stats = {}
            for col in numeric_columns:
                try:
                    stats = {
                        "count": df[col].count(),
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max()
                    }
                    numeric_stats[col] = stats
                except Exception as e:
                    numeric_stats[col] = {"error": str(e)}
            validation_result["numeric_stats"] = numeric_stats
        
        return validation_result
    
    def _clean_data(self, df: pl.DataFrame) -> tuple:
        """Clean data by removing duplicates and handling missing values."""
        original_shape = df.shape
        
        # Remove duplicates
        df_cleaned = df.unique()
        duplicates_removed = original_shape[0] - df_cleaned.height
        
        # Handle missing values
        missing_before = sum(df_cleaned[col].null_count() for col in df_cleaned.columns)
        
        # For numeric columns, fill with median
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype.is_numeric() and df_cleaned[col].null_count() > 0:
                try:
                    median_val = df_cleaned[col].median()
                    df_cleaned = df_cleaned.with_columns(
                        pl.col(col).fill_null(median_val)
                    )
                except Exception:
                    # If median fails, use 0
                    df_cleaned = df_cleaned.with_columns(
                        pl.col(col).fill_null(0)
                    )
            elif df_cleaned[col].dtype == pl.Utf8 and df_cleaned[col].null_count() > 0:
                # For string columns, fill with "unknown"
                df_cleaned = df_cleaned.with_columns(
                    pl.col(col).fill_null("unknown")
                )
        
        missing_after = sum(df_cleaned[col].null_count() for col in df_cleaned.columns)
        
        clean_result = {
            "duplicates_removed": duplicates_removed,
            "missing_values_filled": missing_before - missing_after,
            "final_shape": df_cleaned.shape
        }
        
        return df_cleaned, clean_result
    
    def _transform_data(self, df: pl.DataFrame) -> tuple:
        """Transform data with common operations using Polars."""
        df_transformed = df.clone()
        transformations = []
        
        # Normalize numeric columns
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        for col in numeric_columns:
            try:
                std_val = df_transformed[col].std()
                if std_val and std_val > 0:  # Avoid division by zero
                    mean_val = df_transformed[col].mean()
                    df_transformed = df_transformed.with_columns(
                        ((pl.col(col) - mean_val) / std_val).alias(f"{col}_normalized")
                    )
                    transformations.append(f"normalized_{col}")
            except Exception:
                continue
        
        # Create categorical encoding for string columns
        string_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
        for col in string_columns[:3]:  # Limit to first 3 to avoid too many features
            try:
                unique_count = df_transformed[col].n_unique()
                if unique_count <= 20:  # Only encode if not too many categories
                    # Simple label encoding
                    unique_values = df_transformed[col].unique().to_list()
                    mapping = {val: idx for idx, val in enumerate(unique_values) if val is not None}
                    
                    # Create encoded column using when-then expressions
                    when_then_exprs = []
                    for k, v in mapping.items():
                        when_then_exprs.append(pl.when(pl.col(col) == k).then(v))
                    
                    if when_then_exprs:
                        # Use fold to combine all when-then expressions
                        encoded_col = pl.coalesce(when_then_exprs).alias(f"{col}_encoded")
                        df_transformed = df_transformed.with_columns(encoded_col)
                        transformations.append(f"{col}_encoded")
            except Exception:
                continue
        
        # Add row index and processing timestamp
        df_transformed = df_transformed.with_row_count("row_index")
        df_transformed = df_transformed.with_columns(
            pl.lit(datetime.now().isoformat()).alias("processed_at")
        )
        transformations.extend(["row_index", "processed_at"])
        
        transform_result = {
            "transformations_applied": transformations,
            "new_columns_created": len(transformations)
        }
        
        return df_transformed, transform_result
    
    def _aggregate_data(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Aggregate data for summary statistics using Polars."""
        agg_result = {"summary_stats": {}}
        
        # Numeric aggregations
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        if numeric_columns:
            try:
                # Get basic statistics for numeric columns
                numeric_stats = {}
                for col in numeric_columns:
                    try:
                        stats = {
                            "count": df[col].count(),
                            "mean": df[col].mean(),
                            "sum": df[col].sum(),
                            "min": df[col].min(),
                            "max": df[col].max()
                        }
                        numeric_stats[col] = stats
                    except Exception as e:
                        numeric_stats[col] = {"error": str(e)}
                agg_result["summary_stats"]["numeric"] = numeric_stats
            except Exception as e:
                agg_result["summary_stats"]["numeric"] = {"error": str(e)}
        
        # Categorical aggregations
        string_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
        if string_columns:
            cat_stats = {}
            for col in string_columns:
                try:
                    value_counts = df[col].value_counts().head(10)
                    cat_stats[col] = {
                        "unique_count": df[col].n_unique(),
                        "top_values": value_counts.to_dict(as_series=False) if value_counts.height > 0 else {}
                    }
                except Exception as e:
                    cat_stats[col] = {"error": str(e)}
            agg_result["summary_stats"]["categorical"] = cat_stats
        
        return agg_result
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker processing statistics."""
        return {
            "worker_id": self.worker_id,
            "processed_batches": self.processed_batches,
            "worker_type": "DataProcessor",
            "last_activity": datetime.now().isoformat(),
            "recent_history": self.processing_history[-5:] if self.processing_history else []
        }


@ray.remote
class DataPipelineStage:
    """Represents a stage in a data processing pipeline using Polars."""
    
    def __init__(self, stage_name: str, stage_config: Dict[str, Any]):
        self.stage_name = stage_name
        self.stage_config = stage_config
        self.executions = 0
        
    def execute(self, input_data: Union[pl.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute this pipeline stage."""
        start_time = time.time()
        self.executions += 1
        
        try:
            # Convert input to DataFrame if needed
            if isinstance(input_data, list):
                df = pl.DataFrame(input_data)
            else:
                df = input_data
            
            stage_type = self.stage_config.get("type", "passthrough")
            
            if stage_type == "filter":
                result_df = self._execute_filter(df)
            elif stage_type == "transform":
                result_df = self._execute_transform(df)
            elif stage_type == "validate":
                result_df, validation_info = self._execute_validate(df)
            elif stage_type == "aggregate":
                result_df = self._execute_aggregate(df)
            else:
                result_df = df
            
            execution_time = time.time() - start_time
            
            return {
                "status": "success",
                "stage_name": self.stage_name,
                "execution_time": round(execution_time, 4),
                "executions_count": self.executions,
                "output": result_df,
                "output_shape": result_df.shape if hasattr(result_df, 'shape') else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "stage_name": self.stage_name,
                "error": str(e),
                "executions_count": self.executions
            }
    
    def _execute_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute filter operation using Polars."""
        filter_config = self.stage_config.get("filter", {})
        field = filter_config.get("field")
        operator = filter_config.get("operator", "equals")
        value = filter_config.get("value")
        
        if not field or field not in df.columns:
            return df
        
        try:
            if operator == "equals":
                return df.filter(pl.col(field) == value)
            elif operator == "not_equals":
                return df.filter(pl.col(field) != value)
            elif operator == "greater_than":
                return df.filter(pl.col(field) > value)
            elif operator == "less_than":
                return df.filter(pl.col(field) < value)
            elif operator == "contains":
                return df.filter(pl.col(field).str.contains(str(value)))
            else:
                return df
        except Exception:
            return df
    
    def _execute_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute transform operation using Polars."""
        transform_config = self.stage_config.get("transform", {})
        operation = transform_config.get("operation", "add_field")
        
        try:
            if operation == "add_field":
                field_name = transform_config.get("field_name", "new_field")
                field_value = transform_config.get("field_value", "default")
                return df.with_columns(pl.lit(field_value).alias(field_name))
                
            elif operation == "remove_field":
                field_name = transform_config.get("field_name")
                if field_name and field_name in df.columns:
                    return df.drop(field_name)
                return df
                
            elif operation == "rename_field":
                old_name = transform_config.get("old_name")
                new_name = transform_config.get("new_name")
                if old_name and new_name and old_name in df.columns:
                    return df.rename({old_name: new_name})
                return df
        except Exception:
            pass
        
        return df
    
    def _execute_validate(self, df: pl.DataFrame) -> tuple:
        """Execute validation operation using Polars."""
        validation_config = self.stage_config.get("validation", {})
        required_fields = validation_config.get("required_fields", [])
        
        try:
            # Check for required fields
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                # Return empty DataFrame with validation info
                return pl.DataFrame(), {"missing_fields": missing_fields}
            
            # Filter out rows with null values in required fields
            valid_df = df
            for field in required_fields:
                valid_df = valid_df.filter(pl.col(field).is_not_null())
            
            return valid_df, {"validation_passed": True}
        except Exception as e:
            return df, {"validation_error": str(e)}
    
    def _execute_aggregate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute aggregation operation using Polars."""
        aggregate_config = self.stage_config.get("aggregate", {})
        group_by = aggregate_config.get("group_by")
        aggregations = aggregate_config.get("aggregations", [])
        
        if not group_by or not aggregations or group_by not in df.columns:
            return df
        
        try:
            # Build aggregation expressions
            agg_exprs = [pl.count().alias("count")]
            
            for agg in aggregations:
                field = agg.get("field")
                operation = agg.get("operation")
                
                if field and operation and field in df.columns:
                    if operation == "sum":
                        agg_exprs.append(pl.col(field).sum().alias(f"{field}_sum"))
                    elif operation == "avg":
                        agg_exprs.append(pl.col(field).mean().alias(f"{field}_avg"))
                    elif operation == "count":
                        agg_exprs.append(pl.col(field).count().alias(f"{field}_count"))
            
            return df.group_by(group_by).agg(agg_exprs)
        except Exception:
            return df


@ray.remote
class DataPipelineCoordinator:
    """Coordinates multi-stage data processing pipelines using Polars."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stages = []
        self.executions = 0
        
    def add_stage(self, stage_name: str, stage_config: Dict[str, Any]) -> str:
        """Add a stage to the pipeline."""
        stage = DataPipelineStage.remote(stage_name, stage_config)
        self.stages.append(stage)
        return f"Stage '{stage_name}' added to pipeline"
    
    def execute_pipeline(self, input_data: Union[List[Dict[str, Any]], pl.DataFrame]) -> Dict[str, Any]:
        """Execute the entire pipeline."""
        if not self.stages:
            return {"status": "error", "message": "No stages defined in pipeline"}
        
        start_time = time.time()
        self.executions += 1
        
        # Convert input to DataFrame if needed
        if isinstance(input_data, list):
            current_data = pl.DataFrame(input_data)
        else:
            current_data = input_data
            
        stage_results = []
        
        try:
            # Execute stages sequentially
            for i, stage in enumerate(self.stages):
                stage_result = ray.get(stage.execute.remote(current_data))
                stage_results.append(stage_result)
                
                if stage_result.get("status") == "error":
                    return {
                        "status": "error",
                        "pipeline_name": self.pipeline_name,
                        "failed_at_stage": i,
                        "stage_results": stage_results,
                        "error": stage_result.get("error", "Unknown error")
                    }
                
                current_data = stage_result.get("output", current_data)
            
            execution_time = time.time() - start_time
            
            # Convert final output to dict if it's a DataFrame
            final_output = current_data
            if hasattr(current_data, 'to_dict'):
                try:
                    final_output = current_data.to_dict(as_series=False)
                except Exception:
                    final_output = {"data": "serialization_error"}
            
            return {
                "status": "success",
                "pipeline_name": self.pipeline_name,
                "execution_time": round(execution_time, 4),
                "stages_executed": len(self.stages),
                "executions_count": self.executions,
                "final_output": final_output,
                "final_shape": current_data.shape if hasattr(current_data, 'shape') else None,
                "stage_results": stage_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "pipeline_name": self.pipeline_name,
                "error": str(e),
                "stage_results": stage_results
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "pipeline_name": self.pipeline_name,
            "stages_count": len(self.stages),
            "executions_count": self.executions
        }


# Utility functions
def create_data_processing_cluster(num_processors: int = 3) -> List:
    """Create a cluster of data processors."""
    processors = []
    for i in range(num_processors):
        processor = DataProcessor.remote(f"processor_{i}")
        processors.append(processor)
    return processors


def parallel_data_processing(data_chunks: List[Any], operation: str, 
                           processors: Optional[List] = None) -> Dict[str, Any]:
    """Process data chunks in parallel using Polars."""
    if not processors:
        processors = create_data_processing_cluster(3)
    
    start_time = time.time()
    
    # Distribute chunks across processors
    tasks = []
    for i, chunk in enumerate(data_chunks):
        processor = processors[i % len(processors)]
        
        if operation == "csv":
            task = processor.process_csv_data.remote(chunk, ["validate", "clean"])
        elif operation == "json":
            task = processor.process_json_data.remote(chunk, ["validate", "transform"])
        else:
            continue
        
        tasks.append(task)
    
    # Collect results
    results = ray.get(tasks)
    
    processing_time = time.time() - start_time
    
    return {
        "chunks_processed": len(data_chunks),
        "processing_time": round(processing_time, 4),
        "processors_used": len(processors),
        "results": results
    }


def create_sample_pipeline():
    """Create a sample data processing pipeline with Polars."""
    pipeline = DataPipelineCoordinator.remote("sample_polars_pipeline")
    
    # Add stages
    ray.get(pipeline.add_stage.remote("validate", {
        "type": "validate",
        "validation": {"required_fields": ["id", "name"]}
    }))
    
    ray.get(pipeline.add_stage.remote("filter", {
        "type": "filter",
        "filter": {"field": "status", "operator": "equals", "value": "active"}
    }))
    
    ray.get(pipeline.add_stage.remote("transform", {
        "type": "transform",
        "transform": {"operation": "add_field", "field_name": "processed_date", "field_value": datetime.now().isoformat()}
    }))
    
    return pipeline