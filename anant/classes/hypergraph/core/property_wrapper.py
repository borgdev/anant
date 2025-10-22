"""
PropertyWrapper utility for Hypergraph operations

Provides wrapper functionality to make PropertyStore compatible with Parquet I/O expectations.
This bridges the gap between PropertyStore dictionary format and DataFrame format.
"""

import polars as pl
from typing import Dict, Any


class PropertyWrapper:
    """
    Wrapper to make PropertyStore compatible with Parquet I/O expectations
    
    The Parquet I/O layer expects properties to have a `.properties` attribute
    that returns a Polars DataFrame. This wrapper bridges the gap between
    the PropertyStore dictionary format and the expected DataFrame format.
    """
    
    def __init__(self, properties_dict: Dict[Any, Dict[str, Any]], property_type: str):
        """
        Initialize PropertyWrapper
        
        Parameters
        ----------
        properties_dict : Dict[Any, Dict[str, Any]]
            Dictionary mapping entity IDs to their properties
        property_type : str
            Type of property ('node' or 'edge')
        """
        if not isinstance(properties_dict, dict):
            raise TypeError("properties_dict must be a dictionary")
        if not isinstance(property_type, str):
            raise TypeError("property_type must be a string")
        if property_type not in ['node', 'edge']:
            raise ValueError("property_type must be 'node' or 'edge'")
            
        self._properties_dict = properties_dict
        self._property_type = property_type
        self._cached_df = None
    
    def __len__(self) -> int:
        """Return the number of properties"""
        return len(self._properties_dict)
    
    def __bool__(self) -> bool:
        """Return True if properties exist"""
        return len(self._properties_dict) > 0
    
    @property
    def properties(self) -> pl.DataFrame:
        """
        Convert properties to Polars DataFrame for Parquet I/O
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns: {property_type}_id, property_key, property_value
        """
        try:
            if not self._properties_dict:
                # Return empty DataFrame with correct schema
                return pl.DataFrame({
                    f'{self._property_type}_id': pl.Series([], dtype=pl.Utf8),
                    'property_key': pl.Series([], dtype=pl.Utf8),
                    'property_value': pl.Series([], dtype=pl.Utf8)
                })
            
            # Convert nested dict to flat format for DataFrame
            rows = []
            for entity_id, props in self._properties_dict.items():
                if not isinstance(props, dict):
                    continue
                    
                for key, value in props.items():
                    rows.append({
                        f'{self._property_type}_id': str(entity_id),
                        'property_key': str(key),
                        'property_value': str(value)
                    })
            
            return pl.DataFrame(rows) if rows else pl.DataFrame({
                f'{self._property_type}_id': pl.Series([], dtype=pl.Utf8),
                'property_key': pl.Series([], dtype=pl.Utf8),
                'property_value': pl.Series([], dtype=pl.Utf8)
            })
        except Exception as e:
            raise RuntimeError(f"Failed to convert properties to DataFrame: {e}")
    
    def get_property(self, entity_id: Any, key: str) -> Any:
        """
        Get a specific property value
        
        Parameters
        ----------
        entity_id : Any
            ID of the entity
        key : str
            Property key
            
        Returns
        -------
        Any
            Property value, or None if not found
        """
        return self._properties_dict.get(entity_id, {}).get(key)
    
    def set_property(self, entity_id: Any, key: str, value: Any) -> None:
        """
        Set a property value
        
        Parameters
        ----------
        entity_id : Any
            ID of the entity
        key : str
            Property key
        value : Any
            Property value
        """
        if entity_id not in self._properties_dict:
            self._properties_dict[entity_id] = {}
        self._properties_dict[entity_id][key] = value
        self._cached_df = None  # Invalidate cache
    
    def has_property(self, entity_id: Any, key: str) -> bool:
        """
        Check if entity has a specific property
        
        Parameters
        ----------
        entity_id : Any
            ID of the entity
        key : str
            Property key
            
        Returns
        -------
        bool
            True if property exists, False otherwise
        """
        return entity_id in self._properties_dict and key in self._properties_dict[entity_id]
    
    def get_all_properties(self, entity_id: Any) -> Dict[str, Any]:
        """
        Get all properties for an entity
        
        Parameters
        ----------
        entity_id : Any
            ID of the entity
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of all properties for the entity
        """
        return self._properties_dict.get(entity_id, {}).copy()
    
    def remove_property(self, entity_id: Any, key: str) -> bool:
        """
        Remove a specific property
        
        Parameters
        ----------
        entity_id : Any
            ID of the entity
        key : str
            Property key to remove
            
        Returns
        -------
        bool
            True if property was removed, False if it didn't exist
        """
        if entity_id in self._properties_dict and key in self._properties_dict[entity_id]:
            del self._properties_dict[entity_id][key]
            if not self._properties_dict[entity_id]:  # Remove entity if no properties left
                del self._properties_dict[entity_id]
            self._cached_df = None  # Invalidate cache
            return True
        return False
    
    def clear(self) -> None:
        """Clear all properties"""
        self._properties_dict.clear()
        self._cached_df = None
    
    def copy(self) -> 'PropertyWrapper':
        """
        Create a deep copy of the PropertyWrapper
        
        Returns
        -------
        PropertyWrapper
            Deep copy of this PropertyWrapper
        """
        import copy
        return PropertyWrapper(
            copy.deepcopy(self._properties_dict),
            self._property_type
        )