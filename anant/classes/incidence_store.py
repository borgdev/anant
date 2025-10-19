"""
Incidence Store Implementation for Anant Library

Manages the incidence relationships between nodes and edges in a hypergraph.
Uses Polars DataFrames for high-performance storage and querying.

The incidence store maintains the core data structure representing which nodes
belong to which edges, along with associated weights and properties.
"""

import polars as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json


class IncidenceStore:
    """
    High-performance storage for hypergraph incidence relationships
    
    The IncidenceStore manages the fundamental node-edge relationships that define
    a hypergraph. It uses Polars DataFrames internally for efficient operations
    on large datasets.
    
    The core data structure is a DataFrame with columns:
    - edge_id: Identifier for the hyperedge
    - node_id: Identifier for the node
    - weight: Weight of the incidence relationship (default 1.0)
    
    Additional columns can store incidence-specific properties.
    
    Parameters
    ----------
    data : pl.DataFrame, optional
        Initial incidence data. Must have 'edge_id' and 'node_id' columns.
    
    Examples
    --------
    >>> import polars as pl
    >>> from anant.classes import IncidenceStore
    
    >>> data = pl.DataFrame({
    ...     'edge_id': ['e1', 'e1', 'e2'],
    ...     'node_id': ['n1', 'n2', 'n1'],
    ...     'weight': [1.0, 0.8, 1.0]
    ... })
    >>> store = IncidenceStore(data)
    """
    
    def __init__(self, data: Optional[pl.DataFrame] = None):
        
        if data is not None:
            self._validate_data(data)
            self.data = data.clone()
        else:
            # Create empty store with proper schema
            self.data = pl.DataFrame({
                'edge_id': pl.Series([], dtype=pl.Utf8),
                'node_id': pl.Series([], dtype=pl.Utf8), 
                'weight': pl.Series([], dtype=pl.Float64)
            })
    
    @property
    def edge_column(self) -> str:
        """Column name for edge identifiers"""
        return 'edge_id'
    
    @property
    def node_column(self) -> str:
        """Column name for node identifiers"""
        return 'node_id'
    
    @property
    def weight_column(self) -> str:
        """Column name for incidence weights"""
        return 'weight'
    
    @classmethod
    def from_dict(cls, edge_dict: Dict[Any, List[Any]]) -> 'IncidenceStore':
        """
        Create IncidenceStore from edge dictionary
        
        Parameters
        ----------
        edge_dict : Dict[Any, List[Any]]
            Dictionary mapping edge IDs to lists of node IDs
            
        Returns
        -------
        IncidenceStore
            New incidence store instance
        
        Examples
        --------
        >>> edge_dict = {'e1': ['n1', 'n2'], 'e2': ['n2', 'n3']}
        >>> store = IncidenceStore.from_dict(edge_dict)
        """
        
        if not edge_dict:
            return cls()
        
        # Convert dictionary to DataFrame rows
        rows = []
        for edge_id, node_list in edge_dict.items():
            for node_id in node_list:
                rows.append({
                    'edge_id': str(edge_id),
                    'node_id': str(node_id),
                    'weight': 1.0
                })
        
        data = pl.DataFrame(rows)
        return cls(data)
    
    @classmethod
    def from_pairs(cls, pairs: List[Tuple[Any, Any]], weights: Optional[List[float]] = None) -> 'IncidenceStore':
        """
        Create IncidenceStore from list of (edge_id, node_id) pairs
        
        Parameters
        ----------
        pairs : List[Tuple[Any, Any]]
            List of (edge_id, node_id) tuples
        weights : List[float], optional
            Corresponding weights for each pair
            
        Returns
        -------
        IncidenceStore
            New incidence store instance
        """
        
        if not pairs:
            return cls()
        
        if weights is None:
            weights = [1.0] * len(pairs)
        elif len(weights) != len(pairs):
            raise ValueError("Number of weights must match number of pairs")
        
        data = pl.DataFrame({
            'edge_id': [str(pair[0]) for pair in pairs],
            'node_id': [str(pair[1]) for pair in pairs],
            'weight': weights
        })
        
        return cls(data)
    
    def _validate_data(self, data: pl.DataFrame):
        """Validate input DataFrame has required columns"""
        
        required_cols = {'edge_id', 'node_id'}
        available_cols = set(data.columns)
        
        missing_cols = required_cols - available_cols
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add weight column if missing
        if 'weight' not in data.columns:
            data = data.with_columns(pl.lit(1.0).alias('weight'))
    
    def add_incidence(self, edge_id: Any, node_id: Any, weight: float = 1.0, **properties):
        """
        Add a single incidence relationship
        
        Parameters
        ----------
        edge_id : Any
            Edge identifier
        node_id : Any 
            Node identifier
        weight : float, default 1.0
            Weight of the relationship
        **properties
            Additional properties for this incidence
        """
        
        # Create new row
        new_row = {
            'edge_id': str(edge_id),
            'node_id': str(node_id),
            'weight': weight
        }
        
        # Add any additional properties
        new_row.update(properties)
        
        # Convert to DataFrame and append
        new_data = pl.DataFrame([new_row])
        
        if self.data.is_empty():
            self.data = new_data
        else:
            # Ensure compatible schemas
            for col in new_data.columns:
                if col not in self.data.columns:
                    self.data = self.data.with_columns(pl.lit(None).alias(col))
            
            for col in self.data.columns:
                if col not in new_data.columns:
                    # Determine appropriate null value based on column type
                    if col == 'weight':
                        new_data = new_data.with_columns(pl.lit(1.0).alias(col))
                    else:
                        new_data = new_data.with_columns(pl.lit(None).alias(col))
            
            self.data = pl.concat([self.data, new_data])
    
    def remove_incidence(self, edge_id: Any, node_id: Any):
        """Remove specific incidence relationship"""
        
        self.data = self.data.filter(
            ~((pl.col('edge_id') == str(edge_id)) & (pl.col('node_id') == str(node_id)))
        )
    
    def remove_edge(self, edge_id: Any):
        """Remove all incidences for an edge"""
        
        self.data = self.data.filter(pl.col('edge_id') != str(edge_id))
    
    def remove_node(self, node_id: Any):
        """Remove all incidences for a node"""
        
        self.data = self.data.filter(pl.col('node_id') != str(node_id))
    
    def get_edge_nodes(self, edge_id: Any) -> List[str]:
        """Get all nodes incident to an edge"""
        
        if self.data.is_empty():
            return []
        
        return (
            self.data
            .filter(pl.col('edge_id') == str(edge_id))
            .select('node_id')
            .unique()
            .to_series()
            .to_list()
        )
    
    def get_node_edges(self, node_id: Any) -> List[str]:
        """Get all edges incident to a node"""
        
        if self.data.is_empty():
            return []
        
        return (
            self.data
            .filter(pl.col('node_id') == str(node_id))
            .select('edge_id')
            .unique()
            .to_series()
            .to_list()
        )
    
    def get_edge_size(self, edge_id: Any) -> int:
        """Get number of nodes in an edge"""
        
        return len(self.get_edge_nodes(edge_id))
    
    def get_node_degree(self, node_id: Any) -> int:
        """Get degree (number of incident edges) of a node"""
        
        return len(self.get_node_edges(node_id))
    
    def get_weight(self, edge_id: Any, node_id: Any) -> Optional[float]:
        """Get weight of specific incidence"""
        
        if self.data.is_empty():
            return None
        
        result = (
            self.data
            .filter(
                (pl.col('edge_id') == str(edge_id)) & 
                (pl.col('node_id') == str(node_id))
            )
            .select('weight')
        )
        
        if result.height == 0:
            return None
        
        return result.to_series().to_list()[0]
    
    def set_weight(self, edge_id: Any, node_id: Any, weight: float):
        """Set weight of specific incidence"""
        
        # Update existing incidence or add new one
        mask = (pl.col('edge_id') == str(edge_id)) & (pl.col('node_id') == str(node_id))
        
        if self.data.filter(mask).height > 0:
            # Update existing
            self.data = self.data.with_columns(
                pl.when(mask).then(weight).otherwise(pl.col('weight')).alias('weight')
            )
        else:
            # Add new incidence
            self.add_incidence(edge_id, node_id, weight)
    
    def get_all_nodes(self) -> List[str]:
        """Get all unique node IDs"""
        
        if self.data.is_empty():
            return []
        
        return self.data.select('node_id').unique().to_series().to_list()
    
    def get_all_edges(self) -> List[str]:
        """Get all unique edge IDs"""
        
        if self.data.is_empty():
            return []
        
        return self.data.select('edge_id').unique().to_series().to_list()
    
    def num_nodes(self) -> int:
        """Count unique nodes"""
        
        if self.data.is_empty():
            return 0
        
        return self.data.select('node_id').n_unique()
    
    def num_edges(self) -> int:
        """Count unique edges"""
        
        if self.data.is_empty():
            return 0
        
        return self.data.select('edge_id').n_unique()
    
    def num_incidences(self) -> int:
        """Count total incidence relationships"""
        
        return self.data.height
    
    def filter_by_weight(self, min_weight: float = 0.0, max_weight: Optional[float] = None) -> 'IncidenceStore':
        """Create filtered store based on weight range"""
        
        filtered_data = self.data.filter(pl.col('weight') >= min_weight)
        
        if max_weight is not None:
            filtered_data = filtered_data.filter(pl.col('weight') <= max_weight)
        
        return IncidenceStore(filtered_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the incidence store"""
        
        if self.data.is_empty():
            return {
                'num_nodes': 0,
                'num_edges': 0, 
                'num_incidences': 0,
                'avg_edge_size': 0.0,
                'avg_node_degree': 0.0,
                'weight_stats': {}
            }
        
        # Basic counts
        num_nodes = self.num_nodes()
        num_edges = self.num_edges()
        num_incidences = self.num_incidences()
        
        # Average edge size and node degree
        avg_edge_size = num_incidences / max(1, num_edges)
        avg_node_degree = num_incidences / max(1, num_nodes)
        
        # Weight statistics
        weight_stats = {}
        if 'weight' in self.data.columns:
            weight_series = self.data.select('weight').to_series()
            weight_stats = {
                'mean': float(weight_series.mean()),
                'std': float(weight_series.std()),
                'min': float(weight_series.min()),
                'max': float(weight_series.max()),
                'median': float(weight_series.median())
            }
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_incidences': num_incidences,
            'avg_edge_size': avg_edge_size,
            'avg_node_degree': avg_node_degree,
            'weight_stats': weight_stats
        }
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to edge dictionary representation"""
        
        if self.data.is_empty():
            return {}
        
        result = {}
        edge_groups = self.data.group_by('edge_id', maintain_order=True)
        
        for (edge_id,), group_data in edge_groups:
            node_list = group_data.select('node_id').unique().to_series().to_list()
            result[edge_id] = node_list
        
        return result
    
    def to_pairs(self) -> List[Tuple[str, str]]:
        """Convert to list of (edge_id, node_id) pairs"""
        
        if self.data.is_empty():
            return []
        
        return list(zip(
            self.data.select('edge_id').to_series().to_list(),
            self.data.select('node_id').to_series().to_list()
        ))
    
    def clone(self) -> 'IncidenceStore':
        """Create a deep copy of the store"""
        
        return IncidenceStore(self.data.clone())
    
    def is_empty(self) -> bool:
        """Check if store contains any incidences"""
        
        return self.data.is_empty()
    
    def save(self, filepath: Union[str, Path], format: str = 'parquet'):
        """Save incidence store to file"""
        
        filepath = Path(filepath)
        
        if format == 'parquet':
            self.data.write_parquet(filepath)
        elif format == 'csv':
            self.data.write_csv(filepath)
        elif format == 'json':
            edge_dict = self.to_dict()
            with open(filepath, 'w') as f:
                json.dump(edge_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: Optional[str] = None) -> 'IncidenceStore':
        """Load incidence store from file"""
        
        filepath = Path(filepath)
        
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
        
        if format == 'parquet':
            data = pl.read_parquet(filepath)
        elif format == 'csv':
            data = pl.read_csv(filepath)
        elif format == 'json':
            with open(filepath, 'r') as f:
                edge_dict = json.load(f)
            return cls.from_dict(edge_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return cls(data)
    
    def __len__(self) -> int:
        """Return number of incidences"""
        return self.num_incidences()
    
    def __str__(self) -> str:
        return f"IncidenceStore(nodes={self.num_nodes()}, edges={self.num_edges()}, incidences={self.num_incidences()})"
    
    def __repr__(self) -> str:
        return self.__str__()