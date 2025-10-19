"""
SetSystem Factory for Anant Library

Provides factory functions for creating different types of set systems
that can be used to build hypergraphs.
"""

import polars as pl
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class SetSystemFactory:
    """
    Factory for creating different types of set systems
    """
    
    @staticmethod
    def from_dict(edge_dict: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
        """Create set system from dictionary"""
        return edge_dict
    
    @staticmethod
    def from_dataframe(
        data: pl.DataFrame,
        node_col: str,
        edge_col: str,
        **kwargs
    ) -> Dict[Any, List[Any]]:
        """Create set system from DataFrame"""
        
        # Group by edge and collect nodes
        result = {}
        
        if data.is_empty():
            return result
        
        edge_groups = data.group_by(edge_col, maintain_order=True)
        
        for (edge_id,), group_data in edge_groups:
            node_list = group_data.select(node_col).unique().to_series().to_list()
            result[edge_id] = node_list
        
        return result
    
    @staticmethod
    def from_file(filepath: Union[str, Path], format: Optional[str] = None, **kwargs) -> Dict[Any, List[Any]]:
        """Create set system from file"""
        
        filepath = Path(filepath)
        
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
        
        if format == 'csv':
            data = pl.read_csv(filepath)
        elif format == 'parquet':
            data = pl.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Auto-detect columns
        node_col = kwargs.get('node_col')
        edge_col = kwargs.get('edge_col')
        
        if node_col is None:
            for col in ['node_id', 'node', 'vertex']:
                if col in data.columns:
                    node_col = col
                    break
        
        if edge_col is None:
            for col in ['edge_id', 'edge', 'hyperedge']:
                if col in data.columns:
                    edge_col = col
                    break
        
        if node_col is None or edge_col is None:
            raise ValueError("Could not auto-detect node and edge columns")
        
        return SetSystemFactory.from_dataframe(data, node_col, edge_col, **kwargs)