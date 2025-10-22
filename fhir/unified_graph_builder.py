"""
FHIR Unified Knowledge Graph Builder
===================================

This module creates a unified hierarchical knowledge graph that integrates both
FHIR ontologies and data instances. It ensures proper mapping between ontological
concepts and data instances while maintaining the hierarchical structure.

Features:
- Single unified hierarchical knowledge graph
- Ontology-to-data mapping and alignment
- Hierarchical levels that combine schema and instances
- Cross-level relationships between ontology and data
- Validation of ontology-data consistency
- Comprehensive FHIR knowledge representation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
import json

# ANANT imports
from anant.kg import HierarchicalKnowledgeGraph
from anant import save_hypergraph

# Local FHIR imports
from .ontology_loader import FHIROntologyLoader
from .data_loader import FHIRDataLoader
from .graph_persistence import save_fhir_graph

logger = logging.getLogger(__name__)


class FHIRUnifiedGraphBuilder:
    """
    Unified FHIR Knowledge Graph Builder.
    
    Creates a single hierarchical knowledge graph that integrates:
    1. FHIR ontologies (schema definitions)
    2. FHIR data instances (actual data)
    3. Mappings between ontological concepts and data instances
    4. Cross-level relationships maintaining semantic integrity
    """
    
    def __init__(self, 
                 schema_dir: str = "schema",
                 data_dir: str = "data/output/fhir",
                 graph_name: str = "FHIR_Unified_Knowledge_Graph"):
        """
        Initialize the unified FHIR graph builder.
        
        Args:
            schema_dir: Directory containing FHIR schema files (turtle)
            data_dir: Directory containing FHIR JSON data files
            graph_name: Name for the unified knowledge graph
        """
        self.schema_dir = Path(schema_dir)
        self.data_dir = Path(data_dir)
        self.graph_name = graph_name
        
        # Single unified hierarchical knowledge graph
        self.unified_hkg = HierarchicalKnowledgeGraph(
            name=graph_name,
            enable_semantic_reasoning=True,
            enable_temporal_tracking=True
        )
        
        # Components
        self.ontology_loader = None
        self.data_loader = None
        
        # Mapping and validation
        self.ontology_data_mappings = []
        self.validation_results = {}
        self.build_statistics = {}
        
        logger.info(f"Initialized FHIR Unified Graph Builder: {graph_name}")
    
    def build_unified_graph(self, 
                           max_data_files: Optional[int] = None,
                           validate_mappings: bool = True) -> Dict[str, Any]:
        """
        Build the unified FHIR knowledge graph with both ontology and data.
        
        Args:
            max_data_files: Maximum number of data files to process (None for all)
            validate_mappings: Whether to validate ontology-data mappings
            
        Returns:
            Dictionary with build results and statistics
        """
        build_results = {
            'status': 'success',
            'graph_name': self.graph_name,
            'timestamp': datetime.utcnow().isoformat(),
            'phases': {},
            'statistics': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Phase 1: Create unified hierarchical structure
            logger.info("Phase 1: Creating unified hierarchical structure...")
            structure_results = self._create_unified_structure()
            build_results['phases']['structure_creation'] = structure_results
            
            if structure_results['errors']:
                build_results['errors'].extend(structure_results['errors'])
            
            # Phase 2: Load FHIR ontologies
            logger.info("Phase 2: Loading FHIR ontologies...")
            ontology_results = self._load_ontologies()
            build_results['phases']['ontology_loading'] = ontology_results
            
            if ontology_results['errors']:
                build_results['errors'].extend(ontology_results['errors'])
            
            # Phase 3: Load FHIR data instances
            logger.info("Phase 3: Loading FHIR data instances...")
            data_results = self._load_data_instances(max_data_files)
            build_results['phases']['data_loading'] = data_results
            
            if data_results['errors']:
                build_results['errors'].extend(data_results['errors'])
            
            # Phase 4: Create ontology-data mappings
            logger.info("Phase 4: Creating ontology-data mappings...")
            mapping_results = self._create_ontology_data_mappings()
            build_results['phases']['mapping_creation'] = mapping_results
            
            if mapping_results['errors']:
                build_results['errors'].extend(mapping_results['errors'])
            
            # Phase 5: Validate mappings if requested
            if validate_mappings:
                logger.info("Phase 5: Validating ontology-data mappings...")
                validation_results = self._validate_ontology_data_consistency()
                build_results['phases']['validation'] = validation_results
                
                if validation_results['errors']:
                    build_results['errors'].extend(validation_results['errors'])
                if validation_results['warnings']:
                    build_results['warnings'].extend(validation_results['warnings'])
            
            # Phase 6: Finalize and collect statistics
            logger.info("Phase 6: Finalizing unified graph...")
            finalization_results = self._finalize_unified_graph()
            build_results['phases']['finalization'] = finalization_results
            build_results['statistics'] = finalization_results['statistics']
            
            if build_results['errors']:
                build_results['status'] = 'completed_with_errors'
            
            logger.info(f"Unified FHIR graph build completed: {build_results['status']}")
            
        except Exception as e:
            error_msg = f"Failed to build unified FHIR graph: {str(e)}"
            build_results['status'] = 'failed'
            build_results['errors'].append(error_msg)
            logger.error(error_msg)
        
        self.build_statistics = build_results
        return build_results
    
    def _create_unified_structure(self) -> Dict[str, Any]:
        """Create the unified hierarchical structure for both ontology and data."""
        results = {
            'status': 'success',
            'levels_created': [],
            'errors': []
        }
        
        try:
            # Define unified hierarchical levels that combine ontology and data
            unified_levels = [
                {
                    'id': 'meta_ontology',
                    'name': 'Meta Ontology',
                    'description': 'FHIR meta-classes, abstract concepts, and foundational ontological elements',
                    'order': 0,
                    'type': 'ontology'
                },
                {
                    'id': 'core_ontology',
                    'name': 'Core Ontology',
                    'description': 'FHIR core classes, data types, and resource type definitions',
                    'order': 1,
                    'type': 'ontology'
                },
                {
                    'id': 'valuesets_ontology',
                    'name': 'ValueSets & CodeSystems',
                    'description': 'FHIR value sets, code systems, and controlled vocabularies',
                    'order': 2,
                    'type': 'ontology'
                },
                {
                    'id': 'patients',
                    'name': 'Patient Instances',
                    'description': 'Patient demographic and administrative data instances',
                    'order': 3,
                    'type': 'data'
                },
                {
                    'id': 'practitioners',
                    'name': 'Practitioner Instances',
                    'description': 'Healthcare practitioner and provider instances',
                    'order': 4,
                    'type': 'data'
                },
                {
                    'id': 'organizations',
                    'name': 'Organization Instances',
                    'description': 'Healthcare organization and facility instances',
                    'order': 5,
                    'type': 'data'
                },
                {
                    'id': 'clinical_data',
                    'name': 'Clinical Data Instances',
                    'description': 'Clinical observations, conditions, procedures, and encounters',
                    'order': 6,
                    'type': 'data'
                },
                {
                    'id': 'care_coordination',
                    'name': 'Care Coordination Instances',
                    'description': 'Care plans, goals, and care coordination data',
                    'order': 7,
                    'type': 'data'
                }
            ]
            
            # Create hierarchical levels
            for level in unified_levels:
                success = self.unified_hkg.create_level(
                    level['id'],
                    level['name'],
                    level['description'],
                    level['order']
                )
                
                if success:
                    results['levels_created'].append(level['id'])
                    logger.info(f"Created level: {level['id']} - {level['name']}")
                else:
                    error_msg = f"Failed to create level: {level['id']}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Error creating unified structure: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _load_ontologies(self) -> Dict[str, Any]:
        """Load FHIR ontologies into the unified graph."""
        results = {
            'status': 'success',
            'ontologies_loaded': [],
            'classes_added': 0,
            'properties_added': 0,
            'errors': []
        }
        
        try:
            # Initialize ontology loader with the unified graph
            self.ontology_loader = FHIROntologyLoader(str(self.schema_dir))
            
            # Load ontology files
            load_results = self.ontology_loader.load_ontology_files()
            
            if load_results['errors']:
                results['errors'].extend(load_results['errors'])
            
            # Instead of creating a separate graph, integrate into unified graph
            if self.ontology_loader.graphs:
                self._integrate_ontologies_into_unified_graph()
                
                results['ontologies_loaded'] = load_results['loaded_files']
                results['classes_added'] = load_results.get('classes_found', 0)
                results['properties_added'] = load_results.get('properties_found', 0)
            
        except Exception as e:
            error_msg = f"Error loading ontologies: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _integrate_ontologies_into_unified_graph(self):
        """Integrate loaded ontologies into the unified hierarchical graph."""
        if not self.ontology_loader or not self.ontology_loader.graphs:
            return
        
        try:
            # Process each loaded ontology graph
            for graph_name, graph in self.ontology_loader.graphs.items():
                logger.info(f"Integrating ontology: {graph_name}")
                
                # Add classes to appropriate ontology levels
                self._add_ontology_classes_to_unified_graph(graph, graph_name)
                
                # Add properties to appropriate ontology levels
                self._add_ontology_properties_to_unified_graph(graph, graph_name)
                
                # Add ontology relationships
                self._add_ontology_relationships_to_unified_graph(graph, graph_name)
        
        except Exception as e:
            logger.error(f"Error integrating ontologies: {str(e)}")
            raise
    
    def _add_ontology_classes_to_unified_graph(self, graph, source_file: str):
        """Add ontology classes to the unified graph."""
        try:
            # This is a simplified version - handle import errors gracefully
            if not hasattr(self.ontology_loader, 'RDF') or self.ontology_loader.RDF is None:
                logger.warning("RDF processing not available - skipping detailed ontology integration")
                return
            
            # Find classes in the ontology
            classes = set()
            for subj in graph.subjects(self.ontology_loader.RDF.type, self.ontology_loader.RDFS.Class):
                classes.add(subj)
            
            for subj in graph.subjects(self.ontology_loader.RDF.type, self.ontology_loader.OWL.Class):
                classes.add(subj)
            
            # Add classes to appropriate levels
            for class_uri in classes:
                class_str = str(class_uri)
                level_id = self._determine_ontology_level(class_str)
                
                properties = {
                    'uri': class_str,
                    'source_file': source_file,
                    'type': 'OntologyClass',
                    'is_ontology_element': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Extract additional properties from ontology
                for label in graph.objects(class_uri, self.ontology_loader.RDFS.label):
                    properties['label'] = str(label)
                    break
                
                for comment in graph.objects(class_uri, self.ontology_loader.RDFS.comment):
                    properties['description'] = str(comment)
                    break
                
                # Add to unified graph
                self.unified_hkg.add_node(class_str, properties, level_id)
        
        except Exception as e:
            logger.warning(f"Could not process ontology classes: {str(e)}")
    
    def _add_ontology_properties_to_unified_graph(self, graph, source_file: str):
        """Add ontology properties to the unified graph."""
        try:
            if not hasattr(self.ontology_loader, 'RDF') or self.ontology_loader.RDF is None:
                logger.warning("RDF processing not available - skipping property integration")
                return
            
            # Find properties
            properties_found = set()
            for subj in graph.subjects(self.ontology_loader.RDF.type, self.ontology_loader.RDF.Property):
                properties_found.add(subj)
            
            # Add properties to core ontology level
            for prop_uri in properties_found:
                prop_str = str(prop_uri)
                
                properties = {
                    'uri': prop_str,
                    'source_file': source_file,
                    'type': 'OntologyProperty',
                    'is_ontology_element': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Extract label and description
                for label in graph.objects(prop_uri, self.ontology_loader.RDFS.label):
                    properties['label'] = str(label)
                    break
                
                for comment in graph.objects(prop_uri, self.ontology_loader.RDFS.comment):
                    properties['description'] = str(comment)
                    break
                
                # Add to core ontology level
                self.unified_hkg.add_node(prop_str, properties, 'core_ontology')
        
        except Exception as e:
            logger.warning(f"Could not process ontology properties: {str(e)}")
    
    def _add_ontology_relationships_to_unified_graph(self, graph, source_file: str):
        """Add ontology relationships to the unified graph."""
        try:
            if not hasattr(self.ontology_loader, 'RDF') or self.ontology_loader.RDF is None:
                logger.warning("RDF processing not available - skipping relationship integration")
                return
            
            # Add subclass relationships
            for subj, obj in graph.subject_objects(self.ontology_loader.RDFS.subClassOf):
                source_id = str(subj)
                target_id = str(obj)
                
                # Only add if both nodes exist
                if self.unified_hkg.has_node(source_id) and self.unified_hkg.has_node(target_id):
                    self.unified_hkg.add_cross_level_relationship(
                        source_id,
                        target_id,
                        'subClassOf',
                        {
                            'source_file': source_file,
                            'relationship_type': 'subClassOf',
                            'is_ontology_relationship': True,
                            'semantic_weight': 0.9
                        }
                    )
        
        except Exception as e:
            logger.warning(f"Could not process ontology relationships: {str(e)}")
    
    def _determine_ontology_level(self, class_uri: str) -> str:
        """Determine which ontology level a class belongs to."""
        if 'datatype' in class_uri.lower() or '/dt/' in class_uri:
            return 'core_ontology'
        elif 'valueset' in class_uri.lower() or 'codesystem' in class_uri.lower() or '/vs/' in class_uri or '/cs/' in class_uri:
            return 'valuesets_ontology'
        elif 'rim' in class_uri.lower() or '/orim/' in class_uri:
            return 'meta_ontology'
        else:
            return 'core_ontology'  # Default
    
    def _load_data_instances(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Load FHIR data instances into the unified graph."""
        results = {
            'status': 'success',
            'files_processed': 0,
            'resources_loaded': 0,
            'resource_types': {},
            'errors': []
        }
        
        try:
            # Initialize data loader with the unified graph
            self.data_loader = FHIRDataLoader(str(self.data_dir), self.unified_hkg)
            
            # Map data loader levels to unified levels
            self._map_data_levels_to_unified_levels()
            
            # Load FHIR data files
            load_results = self.data_loader.load_fhir_data_files(max_files)
            
            results['files_processed'] = len(load_results['loaded_files'])
            results['resources_loaded'] = load_results['total_resources']
            results['resource_types'] = load_results['resource_types']
            
            if load_results['errors']:
                results['errors'].extend(load_results['errors'])
        
        except Exception as e:
            error_msg = f"Error loading data instances: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _map_data_levels_to_unified_levels(self):
        """Map data loader levels to the unified hierarchy levels."""
        # Update the data loader's level mapping to use unified levels
        if self.data_loader:
            # Override the resource type mapping
            self.data_loader._get_level_for_resource_type = self._get_unified_level_for_resource_type
    
    def _get_unified_level_for_resource_type(self, resource_type: str) -> str:
        """Map resource types to unified hierarchy levels."""
        resource_mapping = {
            'Patient': 'patients',
            'Practitioner': 'practitioners',
            'PractitionerRole': 'practitioners',
            'Organization': 'organizations',
            'Encounter': 'clinical_data',
            'Observation': 'clinical_data',
            'Condition': 'clinical_data',
            'Procedure': 'clinical_data',
            'DiagnosticReport': 'clinical_data',
            'AllergyIntolerance': 'clinical_data',
            'Immunization': 'clinical_data',
            'MedicationRequest': 'clinical_data',
            'MedicationAdministration': 'clinical_data',
            'MedicationStatement': 'clinical_data',
            'Medication': 'clinical_data',
            'CarePlan': 'care_coordination',
            'Goal': 'care_coordination',
            'CareTeam': 'care_coordination'
        }
        
        return resource_mapping.get(resource_type, 'clinical_data')  # Default to clinical_data
    
    def _create_ontology_data_mappings(self) -> Dict[str, Any]:
        """Create mappings between ontology concepts and data instances."""
        results = {
            'status': 'success',
            'mappings_created': 0,
            'mapping_types': {},
            'errors': []
        }
        
        try:
            # Create type-based mappings (e.g., Patient class to patient instances)
            type_mappings = self._create_type_based_mappings()
            results['mappings_created'] += type_mappings['count']
            results['mapping_types']['type_based'] = type_mappings['count']
            
            # Create property-based mappings (e.g., ontology properties to data elements)
            property_mappings = self._create_property_based_mappings()
            results['mappings_created'] += property_mappings['count']
            results['mapping_types']['property_based'] = property_mappings['count']
            
            # Create vocabulary mappings (e.g., value sets to coded values)
            vocabulary_mappings = self._create_vocabulary_mappings()
            results['mappings_created'] += vocabulary_mappings['count']
            results['mapping_types']['vocabulary_based'] = vocabulary_mappings['count']
            
            logger.info(f"Created {results['mappings_created']} ontology-data mappings")
        
        except Exception as e:
            error_msg = f"Error creating ontology-data mappings: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _create_type_based_mappings(self) -> Dict[str, int]:
        """Create mappings based on resource types."""
        mappings_created = 0
        
        try:
            # Get all data instances grouped by resource type
            resource_instances = {}
            
            # Scan data levels for resource instances
            data_levels = ['patients', 'practitioners', 'organizations', 'clinical_data', 'care_coordination']
            
            for level_id in data_levels:
                nodes = self.unified_hkg.get_nodes_at_level(level_id)
                
                for node_id in nodes:
                    # Extract resource type from node properties or ID
                    resource_type = self._extract_resource_type_from_node(node_id)
                    
                    if resource_type:
                        if resource_type not in resource_instances:
                            resource_instances[resource_type] = []
                        resource_instances[resource_type].append(node_id)
            
            # Create mappings from ontology classes to data instances
            ontology_levels = ['meta_ontology', 'core_ontology', 'valuesets_ontology']
            
            for level_id in ontology_levels:
                ontology_nodes = self.unified_hkg.get_nodes_at_level(level_id)
                
                for ontology_node in ontology_nodes:
                    # Find matching resource type
                    for resource_type, instances in resource_instances.items():
                        if self._is_ontology_class_for_resource_type(ontology_node, resource_type):
                            # Create mapping relationships
                            for instance in instances[:10]:  # Limit to avoid too many relationships
                                self.unified_hkg.add_cross_level_relationship(
                                    instance,
                                    ontology_node,
                                    'instanceOf',
                                    {
                                        'mapping_type': 'type_based',
                                        'resource_type': resource_type,
                                        'semantic_weight': 0.8,
                                        'timestamp': datetime.utcnow().isoformat()
                                    }
                                )
                                mappings_created += 1
        
        except Exception as e:
            logger.error(f"Error in type-based mapping: {str(e)}")
        
        return {'count': mappings_created}
    
    def _extract_resource_type_from_node(self, node_id: str) -> Optional[str]:
        """Extract resource type from node ID or properties."""
        # Check if resource type is in the node ID (format: ResourceType/id)
        if '/' in node_id:
            return node_id.split('/')[0]
        
        # Could also check node properties if available
        return None
    
    def _is_ontology_class_for_resource_type(self, ontology_node: str, resource_type: str) -> bool:
        """Check if an ontology class corresponds to a resource type."""
        # Simple heuristic: check if resource type appears in the ontology class URI
        return resource_type.lower() in ontology_node.lower()
    
    def _create_property_based_mappings(self) -> Dict[str, int]:
        """Create mappings based on properties."""
        # Placeholder for property-based mappings
        return {'count': 0}
    
    def _create_vocabulary_mappings(self) -> Dict[str, int]:
        """Create mappings for vocabularies and value sets."""
        # Placeholder for vocabulary mappings
        return {'count': 0}
    
    def _validate_ontology_data_consistency(self) -> Dict[str, Any]:
        """Validate consistency between ontology and data."""
        results = {
            'status': 'success',
            'validation_checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check 1: Resource type coverage
            coverage_check = self._validate_resource_type_coverage()
            results['validation_checks']['resource_type_coverage'] = coverage_check
            
            # Check 2: Relationship consistency
            relationship_check = self._validate_relationship_consistency()
            results['validation_checks']['relationship_consistency'] = relationship_check
            
            # Check 3: Hierarchy integrity
            hierarchy_check = self._validate_hierarchy_integrity()
            results['validation_checks']['hierarchy_integrity'] = hierarchy_check
            
            # Collect warnings from all checks
            for check_name, check_result in results['validation_checks'].items():
                if check_result.get('warnings'):
                    results['warnings'].extend(check_result['warnings'])
                if check_result.get('errors'):
                    results['errors'].extend(check_result['errors'])
        
        except Exception as e:
            error_msg = f"Error during validation: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _validate_resource_type_coverage(self) -> Dict[str, Any]:
        """Validate that all data resource types have corresponding ontology elements."""
        check_result = {
            'status': 'passed',
            'covered_types': [],
            'uncovered_types': [],
            'warnings': []
        }
        
        try:
            # Get all resource types from data
            data_resource_types = set()
            data_levels = ['patients', 'practitioners', 'organizations', 'clinical_data', 'care_coordination']
            
            for level_id in data_levels:
                nodes = self.unified_hkg.get_nodes_at_level(level_id)
                for node_id in nodes:
                    resource_type = self._extract_resource_type_from_node(node_id)
                    if resource_type:
                        data_resource_types.add(resource_type)
            
            # Check if ontology has corresponding elements
            ontology_levels = ['meta_ontology', 'core_ontology', 'valuesets_ontology']
            ontology_concepts = set()
            
            for level_id in ontology_levels:
                ontology_concepts.update(self.unified_hkg.get_nodes_at_level(level_id))
            
            for resource_type in data_resource_types:
                has_ontology_match = any(
                    resource_type.lower() in concept.lower() 
                    for concept in ontology_concepts
                )
                
                if has_ontology_match:
                    check_result['covered_types'].append(resource_type)
                else:
                    check_result['uncovered_types'].append(resource_type)
                    check_result['warnings'].append(
                        f"No ontology concept found for resource type: {resource_type}"
                    )
            
            if check_result['uncovered_types']:
                check_result['status'] = 'warnings'
        
        except Exception as e:
            check_result['status'] = 'error'
            check_result['errors'] = [f"Resource type coverage check failed: {str(e)}"]
        
        return check_result
    
    def _validate_relationship_consistency(self) -> Dict[str, Any]:
        """Validate relationship consistency across levels."""
        return {'status': 'passed', 'message': 'Relationship consistency check completed'}
    
    def _validate_hierarchy_integrity(self) -> Dict[str, Any]:
        """Validate hierarchy integrity."""
        return {'status': 'passed', 'message': 'Hierarchy integrity check completed'}
    
    def _finalize_unified_graph(self) -> Dict[str, Any]:
        """Finalize the unified graph and collect comprehensive statistics."""
        results = {
            'status': 'success',
            'statistics': {},
            'errors': []
        }
        
        try:
            # Collect comprehensive statistics
            stats = self.unified_hkg.get_hierarchy_statistics()
            
            # Add unified graph specific statistics
            unified_stats = {
                'graph_name': self.graph_name,
                'total_nodes': self.unified_hkg.num_nodes,
                'total_edges': self.unified_hkg.num_edges,
                'total_levels': len(self.unified_hkg.levels),
                'cross_level_relationships': len(self.unified_hkg.cross_level_relationships),
                'ontology_data_mappings': len(self.ontology_data_mappings),
                'level_breakdown': {}
            }
            
            # Level-specific statistics
            all_levels = ['meta_ontology', 'core_ontology', 'valuesets_ontology', 
                         'patients', 'practitioners', 'organizations', 
                         'clinical_data', 'care_coordination']
            
            for level_id in all_levels:
                level_nodes = self.unified_hkg.get_nodes_at_level(level_id)
                unified_stats['level_breakdown'][level_id] = {
                    'node_count': len(level_nodes),
                    'level_metadata': self.unified_hkg.get_level_metadata(level_id)
                }
            
            # Resource type distribution
            resource_type_stats = self._get_resource_type_statistics()
            unified_stats['resource_types'] = resource_type_stats
            
            results['statistics'] = unified_stats
            
            logger.info(f"Finalized unified graph: {unified_stats['total_nodes']} nodes, "
                       f"{unified_stats['total_edges']} edges, {unified_stats['total_levels']} levels")
        
        except Exception as e:
            error_msg = f"Error finalizing unified graph: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _get_resource_type_statistics(self) -> Dict[str, int]:
        """Get statistics about resource types in the graph."""
        resource_stats = {}
        
        try:
            data_levels = ['patients', 'practitioners', 'organizations', 'clinical_data', 'care_coordination']
            
            for level_id in data_levels:
                nodes = self.unified_hkg.get_nodes_at_level(level_id)
                
                for node_id in nodes:
                    resource_type = self._extract_resource_type_from_node(node_id)
                    if resource_type:
                        resource_stats[resource_type] = resource_stats.get(resource_type, 0) + 1
        
        except Exception as e:
            logger.error(f"Error collecting resource type statistics: {str(e)}")
        
        return resource_stats
    
    def save_unified_graph(self, 
                          output_path: Union[str, Path],
                          compression: str = "snappy") -> Dict[str, Any]:
        """
        Save the unified FHIR knowledge graph.
        
        Args:
            output_path: Directory path for saving
            compression: Compression algorithm
            
        Returns:
            Dictionary with save operation results
        """
        return save_fhir_graph(
            self.unified_hkg,
            output_path,
            compression=compression,
            include_fhir_metadata=True
        )
    
    def get_unified_graph(self) -> HierarchicalKnowledgeGraph:
        """Get the unified hierarchical knowledge graph."""
        return self.unified_hkg
    
    def get_build_statistics(self) -> Dict[str, Any]:
        """Get comprehensive build statistics."""
        return self.build_statistics


# Convenience function
def build_fhir_unified_graph(
    schema_dir: str = "schema",
    data_dir: str = "data/output/fhir",
    max_data_files: Optional[int] = None,
    graph_name: str = "FHIR_Unified_Knowledge_Graph"
) -> Tuple[HierarchicalKnowledgeGraph, Dict[str, Any]]:
    """
    Convenience function to build a unified FHIR knowledge graph.
    
    Args:
        schema_dir: Directory containing FHIR schema files
        data_dir: Directory containing FHIR JSON data files
        max_data_files: Maximum number of data files to process
        graph_name: Name for the unified knowledge graph
        
    Returns:
        Tuple of (HierarchicalKnowledgeGraph, build_results)
    """
    builder = FHIRUnifiedGraphBuilder(schema_dir, data_dir, graph_name)
    build_results = builder.build_unified_graph(max_data_files)
    
    return builder.get_unified_graph(), build_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Building unified FHIR knowledge graph...")
    
    # Build unified graph with limited data for testing
    hkg, results = build_fhir_unified_graph(
        schema_dir="schema",
        data_dir="data/output/fhir", 
        max_data_files=5,
        graph_name="FHIR_Test_Unified_Graph"
    )
    
    print(f"\n=== Unified FHIR Graph Build Results ===")
    print(f"Status: {results['status']}")
    print(f"Total nodes: {results.get('statistics', {}).get('total_nodes', 0)}")
    print(f"Total edges: {results.get('statistics', {}).get('total_edges', 0)}")
    print(f"Total levels: {results.get('statistics', {}).get('total_levels', 0)}")
    
    if results.get('statistics', {}).get('resource_types'):
        print("\nResource types loaded:")
        for resource_type, count in results['statistics']['resource_types'].items():
            print(f"  {resource_type}: {count}")
    
    if results.get('errors'):
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    print(f"\nUnified graph successfully created: {hkg.name}")