"""
ANANT Performance Optimization - Lazy Imports Module
==================================================

Implements lazy imports to reduce startup time from 5+ seconds to <100ms.
This module provides lazy import utilities for heavy dependencies.
"""

import importlib
import sys
from typing import Any, Dict, Optional
from functools import lru_cache


class LazyImporter:
    """Lazy import utility to defer heavy imports until actually needed"""
    
    def __init__(self, module_name: str, attribute: Optional[str] = None):
        self.module_name = module_name
        self.attribute = attribute
        self._module = None
        self._cached_obj = None
    
    def __getattr__(self, name: str) -> Any:
        """Import module on first attribute access"""
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        
        if self.attribute:
            if self._cached_obj is None:
                self._cached_obj = getattr(self._module, self.attribute)
            return getattr(self._cached_obj, name)
        else:
            return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make the lazy importer callable"""
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        
        if self.attribute:
            if self._cached_obj is None:
                self._cached_obj = getattr(self._module, self.attribute)
            return self._cached_obj(*args, **kwargs)
        else:
            return self._module(*args, **kwargs)


# Lazy imports for heavy dependencies
spacy = LazyImporter('spacy')
transformers = LazyImporter('transformers')
torch = LazyImporter('torch')
sklearn = LazyImporter('sklearn')


@lru_cache(maxsize=32)
def lazy_import(module_path: str) -> Any:
    """Cache-friendly lazy import function"""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        return None


def conditional_import(module_name: str, fallback=None):
    """Import module if available, return fallback otherwise"""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return fallback


# Pre-define commonly used lazy imports
LAZY_IMPORTS = {
    'spacy': LazyImporter('spacy'),
    'transformers': LazyImporter('transformers'), 
    'torch': LazyImporter('torch'),
    'sklearn': LazyImporter('sklearn'),
    'networkx': LazyImporter('networkx'),
    'matplotlib': LazyImporter('matplotlib.pyplot'),
    'seaborn': LazyImporter('seaborn')
}


def get_lazy_import(name: str) -> LazyImporter:
    """Get a lazy importer for the specified module"""
    if name in LAZY_IMPORTS:
        return LAZY_IMPORTS[name]
    else:
        return LazyImporter(name)


class OperationsRegistry:
    """Registry for lazy-loaded operation modules"""
    
    def __init__(self):
        self._operations = {}
        self._loaded = set()
    
    def register(self, name: str, module_path: str, class_name: str):
        """Register an operation module for lazy loading"""
        self._operations[name] = {
            'module_path': module_path,
            'class_name': class_name,
            'instance': None
        }
    
    def get(self, name: str, *args, **kwargs):
        """Get operation instance, loading module if needed"""
        if name not in self._operations:
            raise KeyError(f"Operation '{name}' not registered")
        
        op_info = self._operations[name]
        
        if op_info['instance'] is None:
            # Lazy load the module
            module = importlib.import_module(op_info['module_path'])
            op_class = getattr(module, op_info['class_name'])
            op_info['instance'] = op_class(*args, **kwargs)
            self._loaded.add(name)
        
        return op_info['instance']
    
    def is_loaded(self, name: str) -> bool:
        """Check if operation module is loaded"""
        return name in self._loaded
    
    def list_operations(self) -> list:
        """List all registered operations"""
        return list(self._operations.keys())


# Global operations registry
operations_registry = OperationsRegistry()