"""
Utility functions and extras for the Anant library

Provides configuration helpers, data processing utilities,
and other convenience functions.
"""

import polars as pl
from typing import Dict, Any, Optional
import importlib
import logging

logger = logging.getLogger(__name__)


def safe_import(module_name: str) -> Optional[Any]:
    """
    Safely import a module, returning None if import fails.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Imported module or None if import fails
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.debug(f"Optional dependency '{module_name}' not available")
        return None


def setup_polars_config():
    """
    Setup optimal Polars configuration for Anant library
    
    Configures Polars settings for better performance and 
    compatibility with hypergraph operations.
    """
    
    # Set reasonable defaults for hypergraph processing
    pl.Config.set_tbl_rows(20)  # Show more rows in display
    pl.Config.set_tbl_cols(10)  # Show more columns
    pl.Config.set_tbl_width_chars(120)  # Wider table display
    
    # Enable string cache for better performance with categorical data
    pl.enable_string_cache()


def create_empty_dataframe() -> pl.DataFrame:
    """Create standardized empty DataFrame for hypergraph data"""
    
    return pl.DataFrame({
        'edge_id': pl.Series([], dtype=pl.Utf8),
        'node_id': pl.Series([], dtype=pl.Utf8), 
        'weight': pl.Series([], dtype=pl.Float64)
    })


def validate_hypergraph_data(data: pl.DataFrame) -> Dict[str, Any]:
    """
    Validate hypergraph data format
    
    Parameters
    ----------
    data : pl.DataFrame
        Input data to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation results with issues found
    """
    
    issues = []
    
    # Check required columns
    required_cols = {'edge_id', 'node_id'}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for null values in key columns
    if 'edge_id' in data.columns:
        null_edges = data.select(pl.col('edge_id').is_null().sum()).item()
        if null_edges > 0:
            issues.append(f"Found {null_edges} null edge_id values")
    
    if 'node_id' in data.columns:
        null_nodes = data.select(pl.col('node_id').is_null().sum()).item()
        if null_nodes > 0:
            issues.append(f"Found {null_nodes} null node_id values")
    
    # Check data types
    if 'weight' in data.columns:
        try:
            data.select(pl.col('weight').cast(pl.Float64))
        except:
            issues.append("Weight column cannot be converted to float")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'shape': data.shape,
        'columns': list(data.columns)
    }