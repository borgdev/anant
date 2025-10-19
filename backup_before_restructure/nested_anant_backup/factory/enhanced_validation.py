"""
Enhanced SetSystem Validation and Integration

Provides comprehensive validation, error handling, and integration
for all enhanced SetSystem types with the existing factory infrastructure.
"""

import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
import logging
from dataclasses import dataclass
from enum import Enum

from .enhanced_setsystems import (
    ParquetSetSystem, 
    MultiModalSetSystem, 
    StreamingSetSystem,
    SetSystemType,
    SetSystemMetadata
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    MINIMAL = "minimal"      # Basic existence checks
    STANDARD = "standard"    # Standard hypergraph validation
    STRICT = "strict"        # Comprehensive validation with performance checks


@dataclass
class ValidationResult:
    """Result of SetSystem validation"""
    passed: bool
    validation_level: ValidationLevel
    warnings: List[str]
    errors: List[str]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


class EnhancedSetSystemValidator:
    """
    Comprehensive validator for enhanced SetSystem types.
    
    Provides multi-level validation with performance analysis
    and optimization recommendations.
    """
    
    @staticmethod
    def validate_setsystem(
        df: pl.DataFrame,
        setsystem_type: SetSystemType,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        performance_check: bool = True
    ) -> ValidationResult:
        """
        Comprehensive validation of enhanced SetSystem DataFrame.
        
        Args:
            df: DataFrame to validate
            setsystem_type: Type of SetSystem being validated
            validation_level: Strictness level for validation
            performance_check: Whether to perform performance analysis
            
        Returns:
            Detailed validation result
        """
        warnings = []
        errors = []
        recommendations = []
        performance_metrics = {}
        
        try:
            # Basic validation
            basic_result = EnhancedSetSystemValidator._validate_basic_structure(df)
            errors.extend(basic_result['errors'])
            warnings.extend(basic_result['warnings'])
            
            # Schema validation
            schema_result = EnhancedSetSystemValidator._validate_schema(df, validation_level)
            errors.extend(schema_result['errors'])
            warnings.extend(schema_result['warnings'])
            
            # Type-specific validation
            type_result = EnhancedSetSystemValidator._validate_type_specific(df, setsystem_type)
            errors.extend(type_result['errors'])
            warnings.extend(type_result['warnings'])
            recommendations.extend(type_result['recommendations'])
            
            # Performance validation
            if performance_check:
                perf_result = EnhancedSetSystemValidator._validate_performance(df, validation_level)
                warnings.extend(perf_result['warnings'])
                recommendations.extend(perf_result['recommendations'])
                performance_metrics.update(perf_result['metrics'])
            
            # Determine overall result
            passed = len(errors) == 0
            
            if validation_level == ValidationLevel.STRICT:
                # In strict mode, warnings also cause failure
                passed = passed and len(warnings) == 0
            
            return ValidationResult(
                passed=passed,
                validation_level=validation_level,
                warnings=warnings,
                errors=errors,
                performance_metrics=performance_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return ValidationResult(
                passed=False,
                validation_level=validation_level,
                warnings=[],
                errors=[f"Validation exception: {str(e)}"],
                performance_metrics={},
                recommendations=["Review DataFrame structure and content"]
            )
    
    @staticmethod
    def _validate_basic_structure(df: pl.DataFrame) -> Dict[str, List[str]]:
        """Validate basic DataFrame structure"""
        errors = []
        warnings = []
        
        # Check if DataFrame exists and is not None
        if df is None:
            errors.append("DataFrame is None")
            return {'errors': errors, 'warnings': warnings}
        
        # Check if DataFrame is empty
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return {'errors': errors, 'warnings': warnings}
        
        # Check column count
        if len(df.columns) < 2:
            errors.append(f"DataFrame has insufficient columns: {len(df.columns)}")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            errors.append("DataFrame has duplicate column names")
        
        # Check row count reasonableness
        if len(df) > 10_000_000:
            warnings.append(f"Large DataFrame detected: {len(df)} rows may impact performance")
        
        return {'errors': errors, 'warnings': warnings}
    
    @staticmethod
    def _validate_schema(df: pl.DataFrame, validation_level: ValidationLevel) -> Dict[str, List[str]]:
        """Validate hypergraph schema requirements"""
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = ["edges", "nodes"]
        missing_required = set(required_columns) - set(df.columns)
        
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
            return {'errors': errors, 'warnings': warnings}
        
        # Check optional columns
        if "weight" not in df.columns:
            if validation_level == ValidationLevel.STRICT:
                errors.append("Weight column is required in strict mode")
            else:
                warnings.append("Weight column missing, will use default weights")
        
        # Validate data types
        try:
            edge_dtype = df["edges"].dtype
            node_dtype = df["nodes"].dtype
            
            if not edge_dtype.is_string():
                if validation_level == ValidationLevel.STRICT:
                    errors.append(f"Edge column must be string type, got: {edge_dtype}")
                else:
                    warnings.append(f"Edge column type {edge_dtype} may cause issues")
            
            if not node_dtype.is_string():
                if validation_level == ValidationLevel.STRICT:
                    errors.append(f"Node column must be string type, got: {node_dtype}")
                else:
                    warnings.append(f"Node column type {node_dtype} may cause issues")
            
            if "weight" in df.columns:
                weight_dtype = df["weight"].dtype
                if not weight_dtype.is_numeric():
                    errors.append(f"Weight column must be numeric, got: {weight_dtype}")
        
        except Exception as e:
            errors.append(f"Data type validation failed: {e}")
        
        # Check for null values
        try:
            edge_nulls = df["edges"].null_count()
            node_nulls = df["nodes"].null_count()
            
            if edge_nulls > 0:
                errors.append(f"Found {edge_nulls} null values in edges column")
            
            if node_nulls > 0:
                errors.append(f"Found {node_nulls} null values in nodes column")
            
            if "weight" in df.columns:
                weight_nulls = df["weight"].null_count()
                if weight_nulls > 0:
                    if validation_level == ValidationLevel.STRICT:
                        errors.append(f"Found {weight_nulls} null values in weights column")
                    else:
                        warnings.append(f"Found {weight_nulls} null values in weights column")
        
        except Exception as e:
            errors.append(f"Null value validation failed: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    @staticmethod
    def _validate_type_specific(df: pl.DataFrame, setsystem_type: SetSystemType) -> Dict[str, Any]:
        """Validate type-specific requirements"""
        errors = []
        warnings = []
        recommendations = []
        
        try:
            if setsystem_type == SetSystemType.PARQUET:
                # Parquet-specific validation
                if "__setsystem_metadata__" in df.columns:
                    recommendations.append("Consider removing metadata column before analysis")
                
                # Check for efficient data types
                if "edges" in df.columns and df["edges"].dtype != pl.Categorical:
                    recommendations.append("Consider using Categorical type for edges column for better performance")
                
                if "nodes" in df.columns and df["nodes"].dtype != pl.Categorical:
                    recommendations.append("Consider using Categorical type for nodes column for better performance")
            
            elif setsystem_type == SetSystemType.MULTIMODAL:
                # Multi-modal specific validation
                if "modality" not in df.columns:
                    warnings.append("Modality column missing - cannot distinguish between different modalities")
                
                if "edge_type" not in df.columns:
                    warnings.append("Edge type column missing - cannot distinguish intra vs cross-modal edges")
                
                # Check modality distribution
                if "modality" in df.columns:
                    modality_counts = df["modality"].value_counts()
                    if len(modality_counts) < 2:
                        warnings.append("Only one modality detected in multi-modal SetSystem")
                    
                    # Check for balanced modalities
                    counts = modality_counts["count"].to_list()
                    if max(counts) / min(counts) > 10:
                        recommendations.append("Modalities are highly imbalanced - consider resampling")
            
            elif setsystem_type == SetSystemType.STREAMING:
                # Streaming-specific validation
                recommendations.append("Ensure chunk size is appropriate for available memory")
                
                # Check for sorted data (better for streaming)
                if len(df) > 1000:  # Only check for larger datasets
                    try:
                        is_sorted = df["edges"].is_sorted()
                        if not is_sorted:
                            recommendations.append("Consider sorting by edges column for better streaming performance")
                    except:
                        pass  # Ignore if sorting check fails
        
        except Exception as e:
            errors.append(f"Type-specific validation failed: {e}")
        
        return {
            'errors': errors, 
            'warnings': warnings, 
            'recommendations': recommendations
        }
    
    @staticmethod
    def _validate_performance(df: pl.DataFrame, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate performance characteristics"""
        warnings = []
        recommendations = []
        metrics = {}
        
        try:
            # Basic metrics
            row_count = len(df)
            col_count = len(df.columns)
            
            metrics['row_count'] = row_count
            metrics['column_count'] = col_count
            
            # Memory usage estimation
            try:
                if hasattr(df, 'estimated_size'):
                    memory_bytes = df.estimated_size()
                    memory_mb = memory_bytes / 1024 / 1024
                    metrics['estimated_memory_mb'] = memory_mb
                    
                    if memory_mb > 1000:  # > 1GB
                        warnings.append(f"Large memory usage detected: {memory_mb:.1f} MB")
                        recommendations.append("Consider using streaming processing for large datasets")
            except:
                pass
            
            # Cardinality analysis
            if "edges" in df.columns:
                edge_cardinality = df["edges"].n_unique()
                metrics['unique_edges'] = edge_cardinality
                metrics['avg_nodes_per_edge'] = row_count / edge_cardinality if edge_cardinality > 0 else 0
            
            if "nodes" in df.columns:
                node_cardinality = df["nodes"].n_unique()
                metrics['unique_nodes'] = node_cardinality
                metrics['avg_edges_per_node'] = row_count / node_cardinality if node_cardinality > 0 else 0
            
            # Density analysis
            if "edges" in df.columns and "nodes" in df.columns:
                edge_count = df["edges"].n_unique()
                node_count = df["nodes"].n_unique()
                
                if edge_count > 0 and node_count > 0:
                    # Hypergraph density (rough approximation)
                    max_possible_incidences = edge_count * node_count
                    density = row_count / max_possible_incidences
                    metrics['hypergraph_density'] = density
                    
                    if density < 0.01:  # Very sparse
                        recommendations.append("Very sparse hypergraph - consider sparse storage optimizations")
                    elif density > 0.5:  # Very dense
                        recommendations.append("Dense hypergraph detected - algorithms may be computationally expensive")
            
            # Performance recommendations based on size
            if row_count > 1_000_000:
                recommendations.append("Large dataset - consider chunked processing or streaming algorithms")
            
            if col_count > 20:
                recommendations.append("Many columns detected - consider selecting only necessary columns for analysis")
            
            # Data type efficiency
            string_columns = [col for col in df.columns if df[col].dtype.is_string()]
            if len(string_columns) > 2:
                recommendations.append("Many string columns - consider using categorical types for repeated values")
            
        except Exception as e:
            warnings.append(f"Performance analysis failed: {e}")
        
        return {
            'warnings': warnings,
            'recommendations': recommendations,
            'metrics': metrics
        }


class EnhancedSetSystemIntegrator:
    """
    Integrator for enhanced SetSystem types with existing factory infrastructure.
    
    Provides seamless integration and backward compatibility.
    """
    
    @staticmethod
    def integrate_with_factory(factory_class: Type) -> Type:
        """
        Integrate enhanced SetSystems with existing factory class.
        
        Args:
            factory_class: Existing SetSystemFactory class to extend
            
        Returns:
            Enhanced factory class with new capabilities
        """
        
        # Add enhanced methods to factory class
        factory_class.from_parquet_enhanced = EnhancedSetSystemIntegrator._create_parquet_method()
        factory_class.from_multimodal = EnhancedSetSystemIntegrator._create_multimodal_method()
        factory_class.from_streaming = EnhancedSetSystemIntegrator._create_streaming_method()
        factory_class.validate_enhanced = EnhancedSetSystemIntegrator._create_validation_method()
        
        logger.info("Enhanced SetSystems integrated with factory")
        
        return factory_class
    
    @staticmethod
    def _create_parquet_method() -> Callable:
        """Create enhanced Parquet factory method"""
        def from_parquet_enhanced(
            file_path: Union[str, Path],
            validation_level: ValidationLevel = ValidationLevel.STANDARD,
            **kwargs
        ) -> pl.DataFrame:
            """Enhanced Parquet SetSystem creation with validation"""
            
            # Create using ParquetSetSystem
            df = ParquetSetSystem.from_parquet(file_path, **kwargs)
            
            # Validate result
            validation_result = EnhancedSetSystemValidator.validate_setsystem(
                df, SetSystemType.PARQUET, validation_level
            )
            
            if not validation_result.passed:
                error_msg = f"Parquet SetSystem validation failed: {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log warnings and recommendations
            for warning in validation_result.warnings:
                logger.warning(f"Parquet SetSystem warning: {warning}")
            
            for rec in validation_result.recommendations:
                logger.info(f"Parquet SetSystem recommendation: {rec}")
            
            return df
        
        return from_parquet_enhanced
    
    @staticmethod
    def _create_multimodal_method() -> Callable:
        """Create MultiModal factory method"""
        def from_multimodal(
            modal_data: Dict[str, Union[pl.DataFrame, str, Path]],
            validation_level: ValidationLevel = ValidationLevel.STANDARD,
            **kwargs
        ) -> pl.DataFrame:
            """MultiModal SetSystem creation with validation"""
            
            # Create using MultiModalSetSystem
            df = MultiModalSetSystem.from_multiple_sources(modal_data, **kwargs)
            
            # Validate result
            validation_result = EnhancedSetSystemValidator.validate_setsystem(
                df, SetSystemType.MULTIMODAL, validation_level
            )
            
            if not validation_result.passed:
                error_msg = f"MultiModal SetSystem validation failed: {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log insights
            for warning in validation_result.warnings:
                logger.warning(f"MultiModal SetSystem warning: {warning}")
            
            for rec in validation_result.recommendations:
                logger.info(f"MultiModal SetSystem recommendation: {rec}")
            
            return df
        
        return from_multimodal
    
    @staticmethod
    def _create_streaming_method() -> Callable:
        """Create Streaming factory method"""
        def from_streaming(
            file_path: Union[str, Path],
            validation_level: ValidationLevel = ValidationLevel.STANDARD,
            **kwargs
        ) -> pl.DataFrame:
            """Streaming SetSystem creation with validation"""
            
            # Create streaming processor
            streaming_processor = StreamingSetSystem(**kwargs)
            
            # Process with accumulation
            df = streaming_processor.from_parquet_stream(file_path, accumulate_result=True)
            
            # Validate result
            validation_result = EnhancedSetSystemValidator.validate_setsystem(
                df, SetSystemType.STREAMING, validation_level
            )
            
            if not validation_result.passed:
                error_msg = f"Streaming SetSystem validation failed: {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log performance statistics
            stats = streaming_processor.get_statistics()
            logger.info(f"Streaming SetSystem stats: {stats}")
            
            return df
        
        return from_streaming
    
    @staticmethod
    def _create_validation_method() -> Callable:
        """Create validation method for factory"""
        def validate_enhanced(
            df: pl.DataFrame,
            setsystem_type: SetSystemType = SetSystemType.STANDARD,
            validation_level: ValidationLevel = ValidationLevel.STANDARD
        ) -> ValidationResult:
            """Validate any SetSystem DataFrame"""
            
            return EnhancedSetSystemValidator.validate_setsystem(
                df, setsystem_type, validation_level
            )
        
        return validate_enhanced