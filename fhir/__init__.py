"""
FHIR Module for ANANT Hierarchical Knowledge Graph
=================================================

This module provides comprehensive FHIR (Fast Healthcare Interoperability Resources) 
integration capabilities for ANANT's hierarchical knowledge graph system.

Components:
- ontology_loader: Load FHIR schema definitions from turtle files
- data_loader: Load FHIR JSON data while preserving relationships
- unified_graph_builder: Main orchestration for creating unified FHIR graphs
- graph_persistence: Save/load FHIR graphs using ANANT's parquet I/O
- test_unified_graph: Comprehensive test suite

Features:
- Single unified hierarchical knowledge graph
- Ontology-to-data mapping and alignment
- Cross-level relationships between ontology and data
- FHIR-compliant persistence and reconstruction
- Comprehensive validation and testing
"""

from .ontology_loader import FHIROntologyLoader
from .data_loader import FHIRDataLoader
from .unified_graph_builder import FHIRUnifiedGraphBuilder, build_fhir_unified_graph
from .graph_persistence import save_fhir_graph, load_fhir_graph

__all__ = [
    'FHIROntologyLoader',
    'FHIRDataLoader', 
    'FHIRUnifiedGraphBuilder',
    'build_fhir_unified_graph',
    'save_fhir_graph',
    'load_fhir_graph'
]

__version__ = "1.0.0"