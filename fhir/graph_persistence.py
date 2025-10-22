"""
FHIR Graph Persistence Module
============================

This module provides persistence functionality for FHIR hierarchical knowledge graphs
using ANANT's existing I/O infrastructure. It leverages the built-in parquet export/import
capabilities and provides FHIR-specific serialization features.

Features:
- Seamless integration with ANANT's parquet I/O
- Hierarchical knowledge graph preservation
- Metadata and relationship integrity
- Support for multiple compression formats
- Lazy loading for large datasets
- FHIR-specific reconstruction capabilities
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# ANANT imports - using existing I/O infrastructure
import anant
from anant.kg import HierarchicalKnowledgeGraph
from anant.io.parquet_io import AnantIO
from anant import save_hypergraph, load_hypergraph

logger = logging.getLogger(__name__)


class FHIRGraphPersistence:
    """
    FHIR Graph persistence using ANANT's existing I/O infrastructure.
    
    This class provides high-level methods for saving and loading FHIR
    hierarchical knowledge graphs while preserving all metadata and relationships.
    """
    
    @staticmethod
    def save_fhir_graph(
        hkg: HierarchicalKnowledgeGraph,
        output_path: Union[str, Path],
        compression: str = "snappy",
        include_fhir_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Save FHIR hierarchical knowledge graph to parquet format.
        
        Args:
            hkg: FHIR hierarchical knowledge graph
            output_path: Directory path for saving
            compression: Compression algorithm (snappy, gzip, lz4, zstd)
            include_fhir_metadata: Whether to include FHIR-specific metadata
            
        Returns:
            Dictionary with save operation results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'status': 'success',
            'output_path': str(output_path),
            'compression': compression,
            'files_created': [],
            'statistics': {},
            'timestamp': datetime.utcnow().isoformat(),
            'errors': []
        }
        
        try:
            # Save the main hierarchical knowledge graph using ANANT's I/O
            main_graph_path = output_path / "main_graph"
            
            logger.info(f"Saving main hierarchical knowledge graph to {main_graph_path}")
            
            # Use ANANT's built-in parquet save
            save_hypergraph(
                hkg.knowledge_graph,  # The underlying KnowledgeGraph
                main_graph_path,
                compression=compression,
                include_metadata=True
            )
            
            results['files_created'].append(str(main_graph_path))
            
            # Save hierarchical structure metadata
            hierarchy_metadata = {
                'graph_name': hkg.name,
                'levels': hkg.levels,
                'level_order': hkg.level_order,
                'entity_levels': hkg.entity_levels,
                'cross_level_relationships': [
                    {
                        **rel,
                        'created_at': rel['created_at'] if isinstance(rel.get('created_at'), str) 
                                     else rel['created_at'].isoformat() if rel.get('created_at') 
                                     else datetime.utcnow().isoformat()
                    }
                    for rel in hkg.cross_level_relationships
                ],
                'enable_semantic_reasoning': hkg.enable_semantic_reasoning,
                'enable_temporal_tracking': hkg.enable_temporal_tracking,
                'total_levels': len(hkg.levels),
                'total_entities': len(hkg.entity_levels),
                'total_relationships': len(hkg.cross_level_relationships)
            }
            
            hierarchy_file = output_path / "hierarchy_metadata.json"
            with open(hierarchy_file, 'w', encoding='utf-8') as f:
                json.dump(hierarchy_metadata, f, indent=2, default=str)
            
            results['files_created'].append(str(hierarchy_file))
            
            # Save level-specific graphs using ANANT's I/O
            level_graphs_saved = 0
            for level_id, level_graph in hkg.level_graphs.items():
                if level_graph and hasattr(level_graph, 'num_nodes') and level_graph.num_nodes > 0:
                    level_path = output_path / f"level_{level_id}"
                    
                    logger.info(f"Saving level graph '{level_id}' to {level_path}")
                    
                    save_hypergraph(
                        level_graph,
                        level_path,
                        compression=compression,
                        include_metadata=True
                    )
                    
                    results['files_created'].append(str(level_path))
                    level_graphs_saved += 1
            
            # Save FHIR-specific metadata if requested
            if include_fhir_metadata:
                fhir_metadata = FHIRGraphPersistence._extract_fhir_metadata(hkg)
                fhir_metadata_file = output_path / "fhir_metadata.json"
                
                with open(fhir_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(fhir_metadata, f, indent=2, default=str)
                
                results['files_created'].append(str(fhir_metadata_file))
            
            # Collect statistics
            results['statistics'] = {
                'total_nodes': hkg.num_nodes,
                'total_edges': hkg.num_edges,
                'total_levels': len(hkg.levels),
                'level_graphs_saved': level_graphs_saved,
                'cross_level_relationships': len(hkg.cross_level_relationships),
                'compression_used': compression,
                'main_graph_size': hkg.knowledge_graph.num_nodes if hasattr(hkg.knowledge_graph, 'num_nodes') else 0
            }
            
            logger.info(f"Successfully saved FHIR graph with {results['statistics']['total_nodes']} nodes "
                       f"and {results['statistics']['total_edges']} edges")
            
        except Exception as e:
            error_msg = f"Failed to save FHIR graph: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    @staticmethod
    def load_fhir_graph(
        input_path: Union[str, Path],
        lazy: bool = False,
        validate: bool = True
    ) -> Tuple[HierarchicalKnowledgeGraph, Dict[str, Any]]:
        """
        Load FHIR hierarchical knowledge graph from parquet format.
        
        Args:
            input_path: Directory path containing saved graph
            lazy: Whether to use lazy loading
            validate: Whether to validate the loaded data
            
        Returns:
            Tuple of (HierarchicalKnowledgeGraph, load_results)
        """
        input_path = Path(input_path)
        
        results = {
            'status': 'success',
            'input_path': str(input_path),
            'files_loaded': [],
            'statistics': {},
            'timestamp': datetime.utcnow().isoformat(),
            'errors': [],
            'warnings': []
        }
        
        try:
            # Load hierarchy metadata
            hierarchy_file = input_path / "hierarchy_metadata.json"
            if not hierarchy_file.exists():
                raise FileNotFoundError(f"Hierarchy metadata file not found: {hierarchy_file}")
            
            with open(hierarchy_file, 'r', encoding='utf-8') as f:
                hierarchy_metadata = json.load(f)
            
            results['files_loaded'].append(str(hierarchy_file))
            
            # Reconstruct HierarchicalKnowledgeGraph
            hkg = HierarchicalKnowledgeGraph(
                name=hierarchy_metadata.get('graph_name', 'FHIR_Graph'),
                enable_semantic_reasoning=hierarchy_metadata.get('enable_semantic_reasoning', True),
                enable_temporal_tracking=hierarchy_metadata.get('enable_temporal_tracking', False)
            )
            
            # Restore hierarchy structure
            hkg.levels = hierarchy_metadata.get('levels', {})
            hkg.level_order = hierarchy_metadata.get('level_order', {})
            hkg.entity_levels = hierarchy_metadata.get('entity_levels', {})
            
            # Load main knowledge graph using ANANT's I/O
            main_graph_path = input_path / "main_graph"
            if main_graph_path.exists():
                logger.info(f"Loading main knowledge graph from {main_graph_path}")
                
                hkg.knowledge_graph = load_hypergraph(
                    main_graph_path,
                    lazy=lazy,
                    validate_schema=validate
                )
                
                results['files_loaded'].append(str(main_graph_path))
            else:
                logger.warning(f"Main graph path not found: {main_graph_path}")
                results['warnings'].append(f"Main graph not found at {main_graph_path}")
            
            # Load level-specific graphs
            level_graphs_loaded = 0
            for level_id in hkg.levels.keys():
                level_path = input_path / f"level_{level_id}"
                
                if level_path.exists():
                    logger.info(f"Loading level graph '{level_id}' from {level_path}")
                    
                    level_graph = load_hypergraph(
                        level_path,
                        lazy=lazy,
                        validate_schema=validate
                    )
                    
                    hkg.level_graphs[level_id] = level_graph
                    results['files_loaded'].append(str(level_path))
                    level_graphs_loaded += 1
                else:
                    logger.warning(f"Level graph not found: {level_path}")
                    results['warnings'].append(f"Level graph '{level_id}' not found")
            
            # Restore cross-level relationships
            cross_level_rels = hierarchy_metadata.get('cross_level_relationships', [])
            hkg.cross_level_relationships = cross_level_rels
            
            # Load FHIR metadata if available
            fhir_metadata_file = input_path / "fhir_metadata.json"
            if fhir_metadata_file.exists():
                with open(fhir_metadata_file, 'r', encoding='utf-8') as f:
                    fhir_metadata = json.load(f)
                
                results['fhir_metadata'] = fhir_metadata
                results['files_loaded'].append(str(fhir_metadata_file))
            
            # Collect statistics
            results['statistics'] = {
                'total_nodes': hkg.num_nodes,
                'total_edges': hkg.num_edges,
                'total_levels': len(hkg.levels),
                'level_graphs_loaded': level_graphs_loaded,
                'cross_level_relationships': len(hkg.cross_level_relationships),
                'expected_levels': hierarchy_metadata.get('total_levels', 0),
                'expected_entities': hierarchy_metadata.get('total_entities', 0),
                'expected_relationships': hierarchy_metadata.get('total_relationships', 0)
            }
            
            # Validation
            if validate:
                validation_results = FHIRGraphPersistence._validate_loaded_graph(hkg, hierarchy_metadata)
                results['validation'] = validation_results
                
                if validation_results['has_errors']:
                    results['warnings'].extend(validation_results['warnings'])
            
            logger.info(f"Successfully loaded FHIR graph with {results['statistics']['total_nodes']} nodes "
                       f"and {results['statistics']['total_edges']} edges")
            
        except Exception as e:
            error_msg = f"Failed to load FHIR graph: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
            
            # Return empty graph in case of error
            hkg = HierarchicalKnowledgeGraph("Error_Graph")
        
        return hkg, results
    
    @staticmethod
    def _extract_fhir_metadata(hkg: HierarchicalKnowledgeGraph) -> Dict[str, Any]:
        """Extract FHIR-specific metadata from the knowledge graph."""
        metadata = {
            'fhir_resource_counts': {},
            'fhir_levels': {},
            'ontology_info': {},
            'data_sources': set(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Count FHIR resource types by examining node properties
        for level_id, level_graph in hkg.level_graphs.items():
            if level_graph and hasattr(level_graph, 'nodes'):
                level_resources = {}
                level_sources = set()
                
                for node in level_graph.nodes:
                    # Extract properties if available
                    if hasattr(level_graph, 'properties') and hasattr(level_graph.properties, 'get_node_properties'):
                        props = level_graph.properties.get_node_properties(node)
                        
                        if props:
                            resource_type = props.get('resource_type', props.get('type', 'Unknown'))
                            level_resources[resource_type] = level_resources.get(resource_type, 0) + 1
                            
                            if 'source_file' in props:
                                level_sources.add(props['source_file'])
                
                metadata['fhir_resource_counts'][level_id] = level_resources
                metadata['data_sources'].update(level_sources)
        
        # Convert set to list for JSON serialization
        metadata['data_sources'] = list(metadata['data_sources'])
        
        # Level information
        metadata['fhir_levels'] = {
            level_id: {
                'name': level_info.get('name', level_id),
                'description': level_info.get('description', ''),
                'entity_count': len([e for e, l in hkg.entity_levels.items() if l == level_id])
            }
            for level_id, level_info in hkg.levels.items()
        }
        
        return metadata
    
    @staticmethod
    def _validate_loaded_graph(hkg: HierarchicalKnowledgeGraph, expected_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the loaded graph against expected metadata."""
        validation = {
            'has_errors': False,
            'warnings': [],
            'checks_passed': [],
            'statistics_match': {}
        }
        
        expected_levels = expected_metadata.get('total_levels', 0)
        actual_levels = len(hkg.levels)
        
        if actual_levels != expected_levels:
            validation['warnings'].append(
                f"Level count mismatch: expected {expected_levels}, got {actual_levels}"
            )
            validation['has_errors'] = True
        else:
            validation['checks_passed'].append("Level count matches")
        
        expected_entities = expected_metadata.get('total_entities', 0)
        actual_entities = len(hkg.entity_levels)
        
        if actual_entities != expected_entities:
            validation['warnings'].append(
                f"Entity count mismatch: expected {expected_entities}, got {actual_entities}"
            )
        else:
            validation['checks_passed'].append("Entity count matches")
        
        expected_relationships = expected_metadata.get('total_relationships', 0)
        actual_relationships = len(hkg.cross_level_relationships)
        
        if actual_relationships != expected_relationships:
            validation['warnings'].append(
                f"Relationship count mismatch: expected {expected_relationships}, got {actual_relationships}"
            )
        else:
            validation['checks_passed'].append("Relationship count matches")
        
        validation['statistics_match'] = {
            'levels': actual_levels == expected_levels,
            'entities': actual_entities == expected_entities,
            'relationships': actual_relationships == expected_relationships
        }
        
        return validation
    
    @staticmethod
    def get_saved_graph_info(input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a saved FHIR graph without loading it.
        
        Args:
            input_path: Directory path containing saved graph
            
        Returns:
            Dictionary with graph information
        """
        input_path = Path(input_path)
        info = {
            'path': str(input_path),
            'exists': input_path.exists(),
            'files': [],
            'metadata': {},
            'statistics': {},
            'errors': []
        }
        
        if not input_path.exists():
            info['errors'].append(f"Path does not exist: {input_path}")
            return info
        
        try:
            # Check for required files
            required_files = ['hierarchy_metadata.json']
            optional_files = ['fhir_metadata.json', 'main_graph', 'main_graph/metadata.json']
            
            for file_name in required_files + optional_files:
                file_path = input_path / file_name
                if file_path.exists():
                    info['files'].append(file_name)
            
            # Load metadata if available
            hierarchy_file = input_path / "hierarchy_metadata.json"
            if hierarchy_file.exists():
                with open(hierarchy_file, 'r', encoding='utf-8') as f:
                    info['metadata'] = json.load(f)
                
                info['statistics'] = {
                    'total_levels': info['metadata'].get('total_levels', 0),
                    'total_entities': info['metadata'].get('total_entities', 0),
                    'total_relationships': info['metadata'].get('total_relationships', 0)
                }
            
            # Check FHIR metadata
            fhir_file = input_path / "fhir_metadata.json"
            if fhir_file.exists():
                with open(fhir_file, 'r', encoding='utf-8') as f:
                    info['fhir_metadata'] = json.load(f)
            
            # Check level graphs
            level_graphs = []
            for item in input_path.iterdir():
                if item.is_dir() and item.name.startswith('level_'):
                    level_graphs.append(item.name)
            
            info['level_graphs'] = level_graphs
            
        except Exception as e:
            info['errors'].append(f"Error reading graph info: {str(e)}")
        
        return info


# Convenience functions for easy usage

def save_fhir_graph(hkg: HierarchicalKnowledgeGraph, output_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to save FHIR hierarchical knowledge graph.
    
    Args:
        hkg: FHIR hierarchical knowledge graph
        output_path: Directory path for saving
        **kwargs: Additional arguments for save operation
        
    Returns:
        Dictionary with save operation results
    """
    return FHIRGraphPersistence.save_fhir_graph(hkg, output_path, **kwargs)


def load_fhir_graph(input_path: Union[str, Path], **kwargs) -> Tuple[HierarchicalKnowledgeGraph, Dict[str, Any]]:
    """
    Convenience function to load FHIR hierarchical knowledge graph.
    
    Args:
        input_path: Directory path containing saved graph
        **kwargs: Additional arguments for load operation
        
    Returns:
        Tuple of (HierarchicalKnowledgeGraph, load_results)
    """
    return FHIRGraphPersistence.load_fhir_graph(input_path, **kwargs)


def get_fhir_graph_info(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to get information about a saved FHIR graph.
    
    Args:
        input_path: Directory path containing saved graph
        
    Returns:
        Dictionary with graph information
    """
    return FHIRGraphPersistence.get_saved_graph_info(input_path)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FHIR Graph Persistence using ANANT's I/O infrastructure")
    print("This module provides high-level persistence for FHIR hierarchical knowledge graphs")
    print("Built on top of ANANT's proven parquet I/O capabilities")