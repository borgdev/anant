"""
Property Store Implementation for Anant Library

Manages properties for nodes and edges in hypergraphs.
Provides efficient storage and retrieval of metadata associated
with hypergraph elements.
"""

from typing import Dict, List, Any, Optional, Union
import polars as pl
from collections import defaultdict


class PropertyStore:
    """
    High-performance storage for node and edge properties
    
    The PropertyStore manages metadata and properties associated with 
    nodes and edges in a hypergraph. It provides efficient storage,
    retrieval, and querying capabilities for property data.
    
    Properties are stored in separate dictionaries for nodes and edges,
    allowing for fast lookup and modification operations.
    
    Parameters
    ----------
    initial_properties : dict, optional
        Initial properties to load into the store
    
    Examples
    --------
    >>> from anant.classes import PropertyStore
    
    >>> store = PropertyStore()
    >>> store.set_node_property('n1', 'color', 'red')
    >>> store.set_edge_property('e1', 'weight', 0.8)
    >>> print(store.get_node_property('n1', 'color'))
    'red'
    """
    
    def __init__(self, initial_properties: Optional[Dict] = None):
        
        # Separate storage for node and edge properties
        self.node_properties = defaultdict(dict)
        self.edge_properties = defaultdict(dict)
        
        # Load initial properties if provided
        if initial_properties:
            self._load_properties(initial_properties)
    
    def _load_properties(self, properties: Dict):
        """Load properties from dictionary"""
        
        # Try to distinguish between node and edge properties
        # This is a simple heuristic - in practice you might want more sophisticated detection
        
        for uid, props in properties.items():
            if isinstance(props, dict):
                # For now, store in both - this will be refined based on actual usage
                self.node_properties[uid].update(props)
    
    def set_node_property(self, node_id: Any, property_name: str, value: Any):
        """Set a property for a node"""
        self.node_properties[node_id][property_name] = value
    
    def set_edge_property(self, edge_id: Any, property_name: str, value: Any):
        """Set a property for an edge"""
        self.edge_properties[edge_id][property_name] = value
    
    def get_node_property(self, node_id: Any, property_name: str, default: Any = None) -> Any:
        """Get a property value for a node"""
        return self.node_properties.get(node_id, {}).get(property_name, default)
    
    def get_edge_property(self, edge_id: Any, property_name: str, default: Any = None) -> Any:
        """Get a property value for an edge"""
        return self.edge_properties.get(edge_id, {}).get(property_name, default)
    
    def get_node_properties(self, node_id: Any) -> Dict[str, Any]:
        """Get all properties for a node"""
        return dict(self.node_properties.get(node_id, {}))
    
    def get_edge_properties(self, edge_id: Any) -> Dict[str, Any]:
        """Get all properties for an edge"""
        return dict(self.edge_properties.get(edge_id, {}))
    
    def set_node_properties(self, node_id: Any, properties: Dict[str, Any]):
        """Set multiple properties for a node"""
        if node_id not in self.node_properties:
            self.node_properties[node_id] = {}
        self.node_properties[node_id].update(properties)
    
    def set_edge_properties(self, edge_id: Any, properties: Dict[str, Any]):
        """Set multiple properties for an edge"""
        if edge_id not in self.edge_properties:
            self.edge_properties[edge_id] = {}
        self.edge_properties[edge_id].update(properties)
    
    def remove_node_property(self, node_id: Any, property_name: str):
        """Remove a specific property from a node"""
        if node_id in self.node_properties:
            self.node_properties[node_id].pop(property_name, None)
    
    def remove_edge_property(self, edge_id: Any, property_name: str):
        """Remove a specific property from an edge"""
        if edge_id in self.edge_properties:
            self.edge_properties[edge_id].pop(property_name, None)
    
    def remove_node_properties(self, node_id: Any):
        """Remove all properties for a node"""
        self.node_properties.pop(node_id, None)
    
    def remove_edge_properties(self, edge_id: Any):
        """Remove all properties for an edge"""
        self.edge_properties.pop(edge_id, None)
    
    def has_node_property(self, node_id: Any, property_name: str) -> bool:
        """Check if a node has a specific property"""
        return property_name in self.node_properties.get(node_id, {})
    
    def has_edge_property(self, edge_id: Any, property_name: str) -> bool:
        """Check if an edge has a specific property"""
        return property_name in self.edge_properties.get(edge_id, {})
    
    def get_all_node_ids(self) -> List[Any]:
        """Get all node IDs that have properties"""
        return list(self.node_properties.keys())
    
    def get_all_edge_ids(self) -> List[Any]:
        """Get all edge IDs that have properties"""
        return list(self.edge_properties.keys())
    
    def get_node_property_names(self, node_id: Any) -> List[str]:
        """Get all property names for a node"""
        return list(self.node_properties.get(node_id, {}).keys())
    
    def get_edge_property_names(self, edge_id: Any) -> List[str]:
        """Get all property names for an edge"""
        return list(self.edge_properties.get(edge_id, {}).keys())
    
    def get_all_node_property_names(self) -> set:
        """Get all unique property names across all nodes"""
        all_props = set()
        for props in self.node_properties.values():
            all_props.update(props.keys())
        return all_props
    
    def get_all_edge_property_names(self) -> set:
        """Get all unique property names across all edges"""
        all_props = set()
        for props in self.edge_properties.values():
            all_props.update(props.keys())
        return all_props
    
    def filter_nodes_by_property(self, property_name: str, value: Any = None) -> List[Any]:
        """Get nodes that have a specific property (optionally with specific value)"""
        
        result = []
        for node_id, props in self.node_properties.items():
            if property_name in props:
                if value is None or props[property_name] == value:
                    result.append(node_id)
        
        return result
    
    def filter_edges_by_property(self, property_name: str, value: Any = None) -> List[Any]:
        """Get edges that have a specific property (optionally with specific value)"""
        
        result = []
        for edge_id, props in self.edge_properties.items():
            if property_name in props:
                if value is None or props[property_name] == value:
                    result.append(edge_id)
        
        return result
    
    def to_dict(self) -> Dict[str, Dict]:
        """Convert property store to dictionary representation"""
        
        return {
            'nodes': dict(self.node_properties),
            'edges': dict(self.edge_properties)
        }
    
    def to_dataframe(self, element_type: str = 'both') -> Union[pl.DataFrame, Dict[str, pl.DataFrame]]:
        """
        Convert properties to DataFrame representation
        
        Parameters
        ----------
        element_type : str, default 'both'
            Which properties to convert: 'nodes', 'edges', or 'both'
            
        Returns
        -------
        pl.DataFrame or Dict[str, pl.DataFrame]
            DataFrame(s) containing the properties
        """
        
        def _props_to_dataframe(props_dict: Dict, id_col: str) -> pl.DataFrame:
            """Convert properties dictionary to DataFrame"""
            
            if not props_dict:
                return pl.DataFrame({id_col: [], 'property': [], 'value': []})
            
            rows = []
            for uid, properties in props_dict.items():
                for prop_name, prop_value in properties.items():
                    rows.append({
                        id_col: uid,
                        'property': prop_name,
                        'value': str(prop_value)  # Convert to string for consistency
                    })
            
            return pl.DataFrame(rows)
        
        if element_type == 'nodes':
            return _props_to_dataframe(self.node_properties, 'node_id')
        elif element_type == 'edges':
            return _props_to_dataframe(self.edge_properties, 'edge_id')
        elif element_type == 'both':
            return {
                'nodes': _props_to_dataframe(self.node_properties, 'node_id'),
                'edges': _props_to_dataframe(self.edge_properties, 'edge_id')
            }
        else:
            raise ValueError("element_type must be 'nodes', 'edges', or 'both'")
    
    def update_from_dict(self, properties: Dict):
        """Update properties from dictionary"""
        
        if 'nodes' in properties:
            for node_id, props in properties['nodes'].items():
                self.set_node_properties(node_id, props)
        
        if 'edges' in properties:
            for edge_id, props in properties['edges'].items():
                self.set_edge_properties(edge_id, props)
    
    def clear(self):
        """Clear all properties"""
        self.node_properties.clear()
        self.edge_properties.clear()
    
    def copy(self) -> 'PropertyStore':
        """Create a deep copy of the property store"""
        
        new_store = PropertyStore()
        
        # Deep copy node properties
        for node_id, props in self.node_properties.items():
            new_store.node_properties[node_id] = props.copy()
        
        # Deep copy edge properties  
        for edge_id, props in self.edge_properties.items():
            new_store.edge_properties[edge_id] = props.copy()
        
        return new_store
    
    def merge(self, other: 'PropertyStore', overwrite: bool = True):
        """
        Merge another property store into this one
        
        Parameters
        ----------
        other : PropertyStore
            Another property store to merge
        overwrite : bool, default True
            Whether to overwrite existing properties
        """
        
        # Merge node properties
        for node_id, props in other.node_properties.items():
            if overwrite or node_id not in self.node_properties:
                self.set_node_properties(node_id, props)
            else:
                # Merge without overwriting existing keys
                for prop_name, prop_value in props.items():
                    if prop_name not in self.node_properties[node_id]:
                        self.set_node_property(node_id, prop_name, prop_value)
        
        # Merge edge properties
        for edge_id, props in other.edge_properties.items():
            if overwrite or edge_id not in self.edge_properties:
                self.set_edge_properties(edge_id, props)
            else:
                # Merge without overwriting existing keys
                for prop_name, prop_value in props.items():
                    if prop_name not in self.edge_properties[edge_id]:
                        self.set_edge_property(edge_id, prop_name, prop_value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the property store"""
        
        node_count = len(self.node_properties)
        edge_count = len(self.edge_properties)
        
        # Count total properties
        total_node_props = sum(len(props) for props in self.node_properties.values())
        total_edge_props = sum(len(props) for props in self.edge_properties.values())
        
        # Get unique property names
        unique_node_props = self.get_all_node_property_names()
        unique_edge_props = self.get_all_edge_property_names()
        
        return {
            'nodes_with_properties': node_count,
            'edges_with_properties': edge_count,
            'total_node_properties': total_node_props,
            'total_edge_properties': total_edge_props,
            'unique_node_property_names': len(unique_node_props),
            'unique_edge_property_names': len(unique_edge_props),
            'node_property_names': list(unique_node_props),
            'edge_property_names': list(unique_edge_props)
        }
    
    def __len__(self) -> int:
        """Return total number of elements with properties"""
        return len(self.node_properties) + len(self.edge_properties)
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"PropertyStore(nodes={stats['nodes_with_properties']}, edges={stats['edges_with_properties']}, total_props={stats['total_node_properties'] + stats['total_edge_properties']})"
    
    def __repr__(self) -> str:
        return self.__str__()