"""
Enhanced SetSystem Integration 

Integrates enhanced SetSystem capabilities with the main SetSystemFactory.
Provides seamless access to:
- Parquet SetSystems for direct file loading
- Multi-Modal SetSystems for cross-analysis
- Streaming SetSystems for massive datasets
- Enhanced validation and optimization

This module enables backward compatibility while exposing new functionality.
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

from .enhanced_setsystems import (
    ParquetSetSystem, 
    MultiModalSetSystem, 
    StreamingSetSystem, 
    SetSystemType
)
from .enhanced_validation import EnhancedSetSystemValidator, ValidationLevel

import logging
logger = logging.getLogger(__name__)


class EnhancedSetSystemFactory:
    """
    Enhanced SetSystemFactory with advanced capabilities
    
    Provides:
    - Direct Parquet file loading with lazy evaluation
    - Multi-modal SetSystem creation and cross-analysis
    - Streaming processing for massive datasets
    - Enhanced validation with multiple levels
    - Automatic optimization and performance monitoring
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize enhanced factory
        
        Args:
            validation_level: Level of validation to apply to SetSystems
        """
        self.validation_level = validation_level
        self._enhanced_validator = EnhancedSetSystemValidator()
        
        logger.info(f"Enhanced SetSystemFactory initialized with {validation_level.value} validation")
    
    def from_parquet(self,
                     file_path: Union[str, Path],
                     edge_column: str = "edges",
                     node_column: str = "nodes", 
                     weight_column: Optional[str] = "weight",
                     columns: Optional[List[str]] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     lazy: bool = True,
                     validate_schema: bool = True,
                     add_metadata: bool = True) -> pl.DataFrame:
        """
        Create SetSystem directly from Parquet file
        
        Args:
            file_path: Path to Parquet file
            edge_column: Column containing edge identifiers
            node_column: Column containing node identifiers
            weight_column: Column containing weights (optional)
            columns: Specific columns to load (optional)
            filters: Filters to apply during loading
            lazy: Use lazy evaluation for performance
            validate_schema: Validate Parquet schema compatibility
            add_metadata: Add SetSystem metadata to result
            
        Returns:
            SetSystem DataFrame ready for Hypergraph creation
        """
        logger.info(f"Creating SetSystem from Parquet: {file_path}")
        
        try:
            # Create using ParquetSetSystem
            df = ParquetSetSystem.from_parquet(
                file_path=file_path,
                edge_column=edge_column,
                node_column=node_column,
                weight_column=weight_column,
                columns=columns,
                filters=filters,
                lazy=lazy,
                validate_schema=validate_schema,
                add_metadata=add_metadata
            )
            
            # Apply enhanced validation
            if self.validation_level != ValidationLevel.MINIMAL:
                validation_result = self._enhanced_validator.validate_setsystem(
                    df, SetSystemType.PARQUET, self.validation_level
                )
                
                if not validation_result.passed:
                    logger.warning(f"SetSystem validation failed: {len(validation_result.errors)} errors, "
                                 f"{len(validation_result.warnings)} warnings")
                    
                    # Log recommendations if available
                    if validation_result.recommendations:
                        for rec in validation_result.recommendations[:3]:  # Top 3
                            logger.info(f"Recommendation: {rec}")
            
            logger.info(f"Successfully created Parquet SetSystem: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to create SetSystem from Parquet: {e}")
            raise ValueError(f"Could not create SetSystem from Parquet file: {e}") from e
    
    def from_multimodal(self,
                        modal_data: Dict[str, Union[pl.DataFrame, str, Path]],
                        modal_prefixes: Optional[Dict[str, str]] = None,
                        cross_modal_edges: Optional[Dict[str, List[tuple]]] = None,
                        merge_strategy: str = "union",
                        validate_compatibility: bool = True,
                        add_modal_metadata: bool = True) -> pl.DataFrame:
        """
        Create SetSystem from multiple modalities
        
        Args:
            modal_data: Dictionary mapping modality names to DataFrames or file paths
            modal_prefixes: Prefixes to add to edge/node names for each modality
            cross_modal_edges: Cross-modal connections to add
            merge_strategy: How to merge modalities ("union" or "intersection")
            validate_compatibility: Validate modality compatibility
            add_modal_metadata: Add modality metadata to result
            
        Returns:
            Multi-modal SetSystem DataFrame
        """
        logger.info(f"Creating multi-modal SetSystem from {len(modal_data)} modalities")
        
        try:
            # Create using MultiModalSetSystem
            df = MultiModalSetSystem.from_multiple_sources(
                modal_data=modal_data,
                modal_prefixes=modal_prefixes,
                cross_modal_edges=cross_modal_edges,
                merge_strategy=merge_strategy,
                validate_compatibility=validate_compatibility,
                add_modal_metadata=add_modal_metadata
            )
            
            # Apply enhanced validation
            if self.validation_level != ValidationLevel.MINIMAL:
                validation_result = self._enhanced_validator.validate_setsystem(
                    df, SetSystemType.MULTIMODAL, self.validation_level
                )
                
                if validation_result.recommendations:
                    logger.info(f"Multi-modal recommendations: {len(validation_result.recommendations)}")
            
            logger.info(f"Successfully created multi-modal SetSystem: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to create multi-modal SetSystem: {e}")
            raise ValueError(f"Could not create multi-modal SetSystem: {e}") from e
    
    def from_streaming(self,
                       file_path: Union[str, Path],
                       chunk_size: int = 10000,
                       edge_column: str = "edges",
                       node_column: str = "nodes",
                       weight_column: Optional[str] = "weight",
                       accumulate_result: bool = True,
                       progress_callback: Optional[Callable] = None,
                       max_memory_mb: Optional[int] = None):
        """
        Create SetSystem using streaming for large datasets
        
        Args:
            file_path: Path to large Parquet file
            chunk_size: Size of processing chunks
            edge_column: Column containing edge identifiers
            node_column: Column containing node identifiers
            weight_column: Column containing weights (optional)
            accumulate_result: Whether to accumulate all chunks into final result
            progress_callback: Callback for progress updates
            max_memory_mb: Memory limit in MB
            
        Returns:
            DataFrame (if accumulate_result=True) or Iterator (if False)
        """
        logger.info(f"Creating streaming SetSystem from: {file_path}")
        
        try:
            # Create streaming processor
            streaming_processor = StreamingSetSystem(
                chunk_size=chunk_size,
                progress_callback=progress_callback,
                max_memory_mb=max_memory_mb
            )
            
            # Process the file
            result = streaming_processor.from_parquet_stream(
                file_path=file_path,
                edge_column=edge_column,
                node_column=node_column,
                weight_column=weight_column,
                accumulate_result=accumulate_result
            )
            
            if accumulate_result:
                # Apply enhanced validation for accumulated result
                if self.validation_level != ValidationLevel.MINIMAL and isinstance(result, pl.DataFrame):
                    validation_result = self._enhanced_validator.validate_setsystem(
                        result, SetSystemType.STREAMING, self.validation_level
                    )
                    
                    if validation_result.performance_metrics:
                        logger.info(f"Streaming performance: {validation_result.performance_metrics}")
                
                if isinstance(result, pl.DataFrame):
                    logger.info(f"Successfully created streaming SetSystem: {len(result)} rows")
                else:
                    logger.info("Created streaming SetSystem (unknown size)")
            else:
                logger.info("Created streaming SetSystem iterator")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create streaming SetSystem: {e}")
            raise ValueError(f"Could not create streaming SetSystem: {e}") from e
    
    def validate_setsystem(self,
                          df: pl.DataFrame,
                          setsystem_type: SetSystemType = SetSystemType.STANDARD,
                          validation_level: Optional[ValidationLevel] = None):
        """
        Validate SetSystem with enhanced validation
        
        Args:
            df: SetSystem DataFrame to validate
            setsystem_type: Type of SetSystem for validation
            validation_level: Override factory's validation level
            
        Returns:
            ValidationResult with detailed analysis
        """
        level = validation_level or self.validation_level
        
        logger.info(f"Validating SetSystem with {level.value} validation")
        
        result = self._enhanced_validator.validate_setsystem(df, setsystem_type, level)
        
        logger.info(f"Validation result: {'PASSED' if result.passed else 'FAILED'} "
                   f"({len(result.errors)} errors, {len(result.warnings)} warnings)")
        
        return result
    
    def get_factory_capabilities(self) -> Dict[str, Any]:
        """
        Get comprehensive factory capabilities and configuration
        
        Returns:
            Dictionary describing all factory capabilities
        """
        return {
            'factory_type': 'EnhancedSetSystemFactory',
            'validation_level': self.validation_level.value,
            'supported_sources': [
                'parquet_files',
                'multimodal_data', 
                'streaming_data',
                'standard_dataframes'
            ],
            'capabilities': {
                'lazy_loading': True,
                'streaming_processing': True,
                'multimodal_analysis': True,
                'enhanced_validation': True,
                'performance_optimization': True,
                'metadata_preservation': True,
                'cross_modal_edges': True,
                'progress_tracking': True
            },
            'performance_features': {
                'memory_monitoring': True,
                'chunk_processing': True,
                'polars_backend': True,
                'compression_support': True
            }
        }


# Convenience functions for direct access
def create_parquet_setsystem(file_path: Union[str, Path], **kwargs) -> pl.DataFrame:
    """Convenience function to create SetSystem from Parquet file"""
    factory = EnhancedSetSystemFactory()
    return factory.from_parquet(file_path, **kwargs)


def create_multimodal_setsystem(modal_data: Dict[str, Union[pl.DataFrame, str, Path]], **kwargs) -> pl.DataFrame:
    """Convenience function to create multi-modal SetSystem"""
    factory = EnhancedSetSystemFactory()
    return factory.from_multimodal(modal_data, **kwargs)


def create_streaming_setsystem(file_path: Union[str, Path], **kwargs):
    """Convenience function to create streaming SetSystem"""
    factory = EnhancedSetSystemFactory()
    return factory.from_streaming(file_path, **kwargs)


# Integration utilities
def get_enhanced_factory(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> EnhancedSetSystemFactory:
    """
    Get an enhanced SetSystemFactory instance
    
    Args:
        validation_level: Level of validation to apply
        
    Returns:
        Enhanced factory instance
    """
    factory = EnhancedSetSystemFactory(validation_level)
    logger.info("Created enhanced SetSystemFactory with advanced capabilities")
    return factory