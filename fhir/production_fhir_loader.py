"""
Production FHIR Data Loader with Ontology Integration
===================================================

This module provides a comprehensive loader for real FHIR Bundle data with:
- Hierarchical knowledge graph construction
- Full FHIR R4 ontology integration
- Memory-efficient batch processing for 100K+ files
- Advanced analytics and exploration capabilities
"""

import json
import logging
import time
import uuid
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import sys
import os
import statistics

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anant.core.knowledge_graph import KnowledgeGraph
from anant.core.hierarchical_knowledge_graph import HierarchicalKnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FHIROntologyManager:
    """Manages FHIR R4 ontology and semantic reasoning."""
    
    def __init__(self):
        self.resource_types = {}
        self.data_types = {}
        self.value_sets = {}
        self.code_systems = {}
        self.structure_definitions = {}
        self.relationships = {}
        
    def load_fhir_r4_ontology(self) -> Dict[str, Any]:
        """Load FHIR R4 ontology definitions."""
        logger.info("Loading FHIR R4 ontology...")
        
        # Core FHIR resource types with their hierarchical relationships
        self.resource_types = {
            # Base Resources
            'DomainResource': {
                'parent': 'Resource',
                'description': 'Base for all FHIR domain resources',
                'children': ['Patient', 'Practitioner', 'Organization', 'Encounter', 
                           'Observation', 'Condition', 'Procedure', 'Medication', 
                           'MedicationRequest', 'CarePlan', 'DiagnosticReport']
            },
            
            # Core Clinical Resources
            'Patient': {
                'parent': 'DomainResource',
                'description': 'Demographics and administrative information about a person',
                'relationships': ['hasEncounter', 'hasObservation', 'hasCondition', 'hasProcedure'],
                'semantic_properties': ['demographics', 'identifier', 'contact']
            },
            
            'Encounter': {
                'parent': 'DomainResource', 
                'description': 'An interaction during healthcare provision',
                'relationships': ['belongsToPatient', 'hasObservation', 'hasCondition', 'hasProcedure'],
                'semantic_properties': ['timing', 'location', 'participant']
            },
            
            'Observation': {
                'parent': 'DomainResource',
                'description': 'Measurements and assertions about a patient',
                'relationships': ['belongsToPatient', 'partOfEncounter', 'hasComponent'],
                'semantic_properties': ['value', 'category', 'code', 'interpretation']
            },
            
            'Condition': {
                'parent': 'DomainResource',
                'description': 'Detailed information about conditions, problems or diagnoses',
                'relationships': ['belongsToPatient', 'diagnosedInEncounter', 'hasEvidence'],
                'semantic_properties': ['severity', 'stage', 'category', 'code']
            },
            
            'Procedure': {
                'parent': 'DomainResource',
                'description': 'An action performed on a patient',
                'relationships': ['belongsToPatient', 'performedInEncounter', 'basedOnCarePlan'],
                'semantic_properties': ['category', 'code', 'outcome', 'performer']
            },
            
            'Medication': {
                'parent': 'DomainResource',
                'description': 'Definition of a medication',
                'relationships': ['prescribedInMedicationRequest', 'hasIngredient'],
                'semantic_properties': ['code', 'form', 'ingredient', 'manufacturer']
            },
            
            'MedicationRequest': {
                'parent': 'DomainResource',
                'description': 'Prescription or order for medication',
                'relationships': ['belongsToPatient', 'prescribesMediation', 'partOfEncounter'],
                'semantic_properties': ['dosage', 'intent', 'priority', 'requester']
            },
            
            'Practitioner': {
                'parent': 'DomainResource',
                'description': 'Healthcare professional',
                'relationships': ['participatesInEncounter', 'performsProcedure', 'prescribesMedication'],
                'semantic_properties': ['qualification', 'specialty', 'identifier']
            },
            
            'Organization': {
                'parent': 'DomainResource',
                'description': 'Healthcare organization',
                'relationships': ['employsParticipant', 'providesServices', 'hasLocation'],
                'semantic_properties': ['type', 'identifier', 'contact']
            }
        }
        
        # FHIR data types and their semantic meaning
        self.data_types = {
            'CodeableConcept': 'Concept with coding from terminology',
            'Coding': 'Reference to a code system',
            'Quantity': 'Measured amount with units',
            'Period': 'Time period with start/end',
            'Reference': 'Link to another resource',
            'Identifier': 'Business identifier',
            'HumanName': 'Person name components',
            'Address': 'Postal address',
            'ContactPoint': 'Communication details'
        }
        
        # Common FHIR code systems
        self.code_systems = {
            'http://loinc.org': 'Logical Observation Identifiers Names and Codes',
            'http://snomed.info/sct': 'SNOMED Clinical Terms',
            'http://www.nlm.nih.gov/research/umls/rxnorm': 'RxNorm',
            'http://hl7.org/fhir/administrative-gender': 'Administrative Gender',
            'http://hl7.org/fhir/observation-status': 'Observation Status',
            'http://hl7.org/fhir/condition-clinical': 'Condition Clinical Status'
        }
        
        logger.info(f"Loaded FHIR ontology: {len(self.resource_types)} resource types")
        return {
            'resource_types': len(self.resource_types),
            'data_types': len(self.data_types),
            'code_systems': len(self.code_systems)
        }
    
    def get_resource_hierarchy(self, resource_type: str) -> List[str]:
        """Get the full hierarchy path for a resource type."""
        hierarchy = [resource_type]
        current = resource_type
        
        while current in self.resource_types and 'parent' in self.resource_types[current]:
            parent = self.resource_types[current]['parent']
            hierarchy.insert(0, parent)
            current = parent
            
        return hierarchy
    
    def get_semantic_relationships(self, resource_type: str) -> List[str]:
        """Get semantic relationships for a resource type."""
        if resource_type in self.resource_types:
            return self.resource_types[resource_type].get('relationships', [])
        return []


class ProductionFHIRLoader:
    """Production-grade FHIR data loader with ontology integration."""
    
    def __init__(self, fhir_data_dir: str, graph_name: str = "ProductionFHIRKG"):
        self.fhir_data_dir = Path(fhir_data_dir)
        self.graph_name = graph_name
        self.ontology_manager = FHIROntologyManager()
        self.hierarchical_kg = None
        self.batch_size = 100  # Files per batch
        self.progress_interval = 1000  # Progress reporting interval
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_resources': 0,
            'resource_counts': defaultdict(int),
            'processing_time': 0.0,
            'errors': [],
            'batch_times': []
        }
        
        logger.info(f"Initialized Production FHIR Loader for: {self.fhir_data_dir}")
        logger.info(f"Graph name: {self.graph_name}")
    
    def initialize_hierarchical_graph(self) -> Dict[str, Any]:
        """Initialize hierarchical knowledge graph with FHIR ontology structure."""
        logger.info("Initializing hierarchical knowledge graph with FHIR ontology...")
        
        # Load FHIR ontology
        ontology_stats = self.ontology_manager.load_fhir_r4_ontology()
        
        # Create hierarchical knowledge graph
        self.hierarchical_kg = HierarchicalKnowledgeGraph(
            name=self.graph_name,
            semantic_reasoning=True
        )
        
        # Create hierarchical levels based on FHIR resource relationships
        levels = [
            ('meta_ontology', 'FHIR Meta Ontology', 0),
            ('patients', 'Patient Registry', 1),
            ('encounters', 'Clinical Encounters', 2), 
            ('observations', 'Clinical Observations', 3),
            ('conditions', 'Diagnoses & Conditions', 4),
            ('procedures', 'Medical Procedures', 5),
            ('medications', 'Medication Management', 6),
            ('practitioners', 'Healthcare Providers', 7),
            ('organizations', 'Healthcare Organizations', 8),
            ('care_coordination', 'Care Plans & Coordination', 9)
        ]
        
        for level_name, description, order in levels:
            level_kg = KnowledgeGraph(name=f"{self.graph_name}_{level_name}")
            self.hierarchical_kg.add_level(level_name, level_kg, order, description)
            logger.info(f"Created level: {level_name} - {description}")
        
        return {
            'status': 'initialized',
            'levels_created': len(levels),
            'ontology_stats': ontology_stats
        }
    
    def discover_fhir_files(self) -> Dict[str, Any]:
        """Discover and analyze FHIR files in the data directory."""
        logger.info(f"Discovering FHIR files in: {self.fhir_data_dir}")
        
        if not self.fhir_data_dir.exists():
            raise FileNotFoundError(f"FHIR data directory not found: {self.fhir_data_dir}")
        
        # Find all JSON files
        json_files = list(self.fhir_data_dir.glob("*.json"))
        total_files = len(json_files)
        
        # Sample a few files to analyze structure
        sample_size = min(10, total_files)
        sample_files = json_files[:sample_size]
        
        resource_type_counts = defaultdict(int)
        file_sizes = []
        
        for file_path in sample_files:
            try:
                file_size = file_path.stat().st_size
                file_sizes.append(file_size)
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if data.get('resourceType') == 'Bundle' and 'entry' in data:
                    for entry in data['entry']:
                        resource = entry.get('resource', {})
                        resource_type = resource.get('resourceType')
                        if resource_type:
                            resource_type_counts[resource_type] += 1
                            
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
        
        avg_file_size = statistics.mean(file_sizes) if file_sizes else 0
        total_size_gb = sum(file_sizes) * total_files / sample_size / (1024**3)
        
        discovery_stats = {
            'total_files': total_files,
            'sample_analyzed': sample_size,
            'avg_file_size_mb': avg_file_size / (1024**2),
            'estimated_total_size_gb': total_size_gb,
            'resource_types_found': dict(resource_type_counts),
            'estimated_total_resources': sum(resource_type_counts.values()) * total_files // sample_size
        }
        
        logger.info(f"Discovery complete: {total_files} files, ~{discovery_stats['estimated_total_resources']:,} resources")
        return discovery_stats
    
    def process_fhir_bundle(self, bundle_data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Process a single FHIR Bundle and add resources to hierarchical graph."""
        
        if bundle_data.get('resourceType') != 'Bundle':
            return {'status': 'skipped', 'reason': 'Not a Bundle resource'}
        
        entries = bundle_data.get('entry', [])
        if not entries:
            return {'status': 'skipped', 'reason': 'Empty bundle'}
        
        batch_stats = {
            'resources_processed': 0,
            'resource_types': defaultdict(int),
            'relationships_created': 0
        }
        
        # Group resources by type for hierarchical processing
        resources_by_type = defaultdict(list)
        
        for entry in entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type:
                resources_by_type[resource_type].append(resource)
                batch_stats['resource_types'][resource_type] += 1
                batch_stats['resources_processed'] += 1
        
        # Process resources in hierarchical order
        processing_order = [
            'Patient', 'Practitioner', 'Organization',  # Core entities
            'Encounter',  # Clinical contexts
            'Observation', 'Condition', 'Procedure', 'Medication', 'MedicationRequest',  # Clinical data
            'CarePlan', 'DiagnosticReport'  # Care coordination
        ]
        
        # Store resource references for relationship building
        resource_refs = {}
        
        # Process each resource type in order
        for resource_type in processing_order:
            if resource_type in resources_by_type:
                for resource in resources_by_type[resource_type]:
                    node_id = self._add_resource_to_graph(resource, resource_type, file_path)
                    if node_id:
                        resource_refs[resource['id']] = {
                            'node_id': node_id,
                            'resource_type': resource_type,
                            'resource': resource
                        }
        
        # Build relationships between resources
        batch_stats['relationships_created'] = self._build_resource_relationships(resource_refs)
        
        return {
            'status': 'processed',
            'file_path': file_path,
            **batch_stats
        }
    
    def _add_resource_to_graph(self, resource: Dict[str, Any], resource_type: str, file_path: str) -> Optional[str]:
        """Add a FHIR resource to the appropriate hierarchical level."""
        
        try:
            # Determine the appropriate level for this resource type
            level_mapping = {
                'Patient': 'patients',
                'Encounter': 'encounters', 
                'Observation': 'observations',
                'Condition': 'conditions',
                'Procedure': 'procedures',
                'Medication': 'medications',
                'MedicationRequest': 'medications',
                'Practitioner': 'practitioners',
                'Organization': 'organizations',
                'CarePlan': 'care_coordination',
                'DiagnosticReport': 'care_coordination'
            }
            
            level_name = level_mapping.get(resource_type, 'care_coordination')
            
            if level_name not in self.hierarchical_kg.levels:
                logger.warning(f"Level {level_name} not found, using care_coordination")
                level_name = 'care_coordination'
            
            # Create node ID
            resource_id = resource.get('id', str(uuid.uuid4()))
            node_id = f"{resource_type}_{resource_id}"
            
            # Extract semantic properties
            properties = self._extract_semantic_properties(resource, resource_type)
            properties.update({
                'fhir_resource_type': resource_type,
                'fhir_id': resource_id,
                'source_file': file_path,
                'ontology_hierarchy': self.ontology_manager.get_resource_hierarchy(resource_type)
            })
            
            # Add node to appropriate level
            level_kg = self.hierarchical_kg.get_level(level_name)
            level_kg.add_node(node_id, properties)
            
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding {resource_type} resource: {e}")
            self.stats['errors'].append(f"Error adding {resource_type}: {e}")
            return None
    
    def _extract_semantic_properties(self, resource: Dict[str, Any], resource_type: str) -> Dict[str, Any]:
        """Extract semantic properties from FHIR resource based on ontology."""
        
        properties = {}
        
        # Common properties for all resources
        if 'meta' in resource:
            properties['meta_profile'] = resource['meta'].get('profile', [])
            properties['meta_last_updated'] = resource['meta'].get('lastUpdated')
        
        # Resource-specific semantic extraction
        if resource_type == 'Patient':
            properties.update(self._extract_patient_semantics(resource))
        elif resource_type == 'Encounter':
            properties.update(self._extract_encounter_semantics(resource))
        elif resource_type == 'Observation':
            properties.update(self._extract_observation_semantics(resource))
        elif resource_type == 'Condition':
            properties.update(self._extract_condition_semantics(resource))
        elif resource_type == 'Procedure':
            properties.update(self._extract_procedure_semantics(resource))
        elif resource_type in ['Medication', 'MedicationRequest']:
            properties.update(self._extract_medication_semantics(resource))
        elif resource_type == 'Practitioner':
            properties.update(self._extract_practitioner_semantics(resource))
        elif resource_type == 'Organization':
            properties.update(self._extract_organization_semantics(resource))
        
        return properties
    
    def _extract_patient_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Patient resource."""
        properties = {}
        
        # Demographics
        if 'gender' in resource:
            properties['gender'] = resource['gender']
        
        if 'birthDate' in resource:
            properties['birth_date'] = resource['birthDate']
            # Calculate age
            try:
                birth_year = int(resource['birthDate'][:4])
                current_year = datetime.now().year
                properties['age'] = current_year - birth_year
            except:
                pass
        
        # Name
        if 'name' in resource and resource['name']:
            name = resource['name'][0]
            if 'family' in name:
                properties['family_name'] = name['family']
            if 'given' in name:
                properties['given_name'] = ' '.join(name['given'])
        
        # Identifiers
        if 'identifier' in resource:
            properties['identifiers'] = [
                f"{id.get('system', '')}|{id.get('value', '')}" 
                for id in resource['identifier']
            ]
        
        # Extensions (race, ethnicity, etc.)
        if 'extension' in resource:
            for ext in resource['extension']:
                if 'us-core-race' in ext.get('url', ''):
                    properties['race'] = self._extract_coding_display(ext)
                elif 'us-core-ethnicity' in ext.get('url', ''):
                    properties['ethnicity'] = self._extract_coding_display(ext)
        
        return properties
    
    def _extract_encounter_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Encounter resource."""
        properties = {}
        
        if 'status' in resource:
            properties['encounter_status'] = resource['status']
        
        if 'class' in resource:
            properties['encounter_class'] = resource['class'].get('display', resource['class'].get('code'))
        
        if 'period' in resource:
            period = resource['period']
            if 'start' in period:
                properties['start_time'] = period['start']
            if 'end' in period:
                properties['end_time'] = period['end']
        
        if 'subject' in resource and 'reference' in resource['subject']:
            properties['patient_reference'] = resource['subject']['reference']
        
        return properties
    
    def _extract_observation_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Observation resource."""
        properties = {}
        
        if 'status' in resource:
            properties['observation_status'] = resource['status']
        
        if 'category' in resource:
            properties['category'] = [
                cat.get('coding', [{}])[0].get('display', '') 
                for cat in resource['category']
            ]
        
        if 'code' in resource:
            code = resource['code']
            if 'coding' in code:
                coding = code['coding'][0]
                properties['observation_code'] = coding.get('code')
                properties['observation_display'] = coding.get('display')
                properties['observation_system'] = coding.get('system')
        
        # Value extraction
        for value_key in ['valueQuantity', 'valueString', 'valueBoolean', 'valueCodeableConcept']:
            if value_key in resource:
                if value_key == 'valueQuantity':
                    qty = resource[value_key]
                    properties['value_numeric'] = qty.get('value')
                    properties['value_unit'] = qty.get('unit')
                else:
                    properties['value'] = resource[value_key]
        
        if 'subject' in resource and 'reference' in resource['subject']:
            properties['patient_reference'] = resource['subject']['reference']
        
        if 'encounter' in resource and 'reference' in resource['encounter']:
            properties['encounter_reference'] = resource['encounter']['reference']
        
        return properties
    
    def _extract_condition_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Condition resource."""
        properties = {}
        
        if 'clinicalStatus' in resource:
            properties['clinical_status'] = self._extract_coding_display(resource['clinicalStatus'])
        
        if 'verificationStatus' in resource:
            properties['verification_status'] = self._extract_coding_display(resource['verificationStatus'])
        
        if 'code' in resource:
            code = resource['code']
            if 'coding' in code:
                coding = code['coding'][0]
                properties['condition_code'] = coding.get('code')
                properties['condition_display'] = coding.get('display')
                properties['condition_system'] = coding.get('system')
        
        if 'subject' in resource and 'reference' in resource['subject']:
            properties['patient_reference'] = resource['subject']['reference']
        
        if 'encounter' in resource and 'reference' in resource['encounter']:
            properties['encounter_reference'] = resource['encounter']['reference']
        
        return properties
    
    def _extract_procedure_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Procedure resource."""
        properties = {}
        
        if 'status' in resource:
            properties['procedure_status'] = resource['status']
        
        if 'code' in resource:
            code = resource['code']
            if 'coding' in code:
                coding = code['coding'][0]
                properties['procedure_code'] = coding.get('code')
                properties['procedure_display'] = coding.get('display')
                properties['procedure_system'] = coding.get('system')
        
        if 'subject' in resource and 'reference' in resource['subject']:
            properties['patient_reference'] = resource['subject']['reference']
        
        if 'encounter' in resource and 'reference' in resource['encounter']:
            properties['encounter_reference'] = resource['encounter']['reference']
        
        return properties
    
    def _extract_medication_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Medication/MedicationRequest resource."""
        properties = {}
        
        if resource.get('resourceType') == 'MedicationRequest':
            if 'status' in resource:
                properties['medication_status'] = resource['status']
            
            if 'intent' in resource:
                properties['medication_intent'] = resource['intent']
            
            if 'medicationCodeableConcept' in resource:
                code = resource['medicationCodeableConcept']
                if 'coding' in code:
                    coding = code['coding'][0]
                    properties['medication_code'] = coding.get('code')
                    properties['medication_display'] = coding.get('display')
                    properties['medication_system'] = coding.get('system')
            
            if 'subject' in resource and 'reference' in resource['subject']:
                properties['patient_reference'] = resource['subject']['reference']
        
        elif resource.get('resourceType') == 'Medication':
            if 'code' in resource:
                code = resource['code']
                if 'coding' in code:
                    coding = code['coding'][0]
                    properties['medication_code'] = coding.get('code')
                    properties['medication_display'] = coding.get('display')
                    properties['medication_system'] = coding.get('system')
        
        return properties
    
    def _extract_practitioner_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Practitioner resource."""
        properties = {}
        
        if 'name' in resource and resource['name']:
            name = resource['name'][0]
            if 'family' in name:
                properties['family_name'] = name['family']
            if 'given' in name:
                properties['given_name'] = ' '.join(name['given'])
        
        if 'qualification' in resource:
            properties['qualifications'] = [
                q.get('code', {}).get('coding', [{}])[0].get('display', '')
                for q in resource['qualification']
            ]
        
        return properties
    
    def _extract_organization_semantics(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic properties from Organization resource."""
        properties = {}
        
        if 'name' in resource:
            properties['organization_name'] = resource['name']
        
        if 'type' in resource:
            properties['organization_type'] = [
                t.get('coding', [{}])[0].get('display', '')
                for t in resource['type']
            ]
        
        return properties
    
    def _extract_coding_display(self, codeable_concept: Dict[str, Any]) -> str:
        """Extract display value from CodeableConcept."""
        if 'coding' in codeable_concept and codeable_concept['coding']:
            return codeable_concept['coding'][0].get('display', '')
        return ''
    
    def _build_resource_relationships(self, resource_refs: Dict[str, Any]) -> int:
        """Build semantic relationships between FHIR resources."""
        relationships_created = 0
        
        for resource_id, ref_info in resource_refs.items():
            resource = ref_info['resource']
            resource_type = ref_info['resource_type']
            source_node = ref_info['node_id']
            
            # Build relationships based on FHIR references
            if resource_type in ['Encounter', 'Observation', 'Condition', 'Procedure', 'MedicationRequest']:
                # Find patient reference
                if 'subject' in resource and 'reference' in resource['subject']:
                    patient_ref = resource['subject']['reference']
                    if patient_ref.startswith('Patient/'):
                        patient_id = patient_ref.split('/')[-1]
                        if patient_id in resource_refs:
                            target_node = resource_refs[patient_id]['node_id']
                            self._add_cross_level_relationship(
                                source_node, target_node, 'belongsToPatient',
                                ref_info['resource_type'], 'Patient'
                            )
                            relationships_created += 1
                
                # Find encounter reference
                if 'encounter' in resource and 'reference' in resource['encounter']:
                    encounter_ref = resource['encounter']['reference']
                    if encounter_ref.startswith('Encounter/'):
                        encounter_id = encounter_ref.split('/')[-1]
                        if encounter_id in resource_refs:
                            target_node = resource_refs[encounter_id]['node_id']
                            self._add_cross_level_relationship(
                                source_node, target_node, 'partOfEncounter',
                                ref_info['resource_type'], 'Encounter'
                            )
                            relationships_created += 1
        
        return relationships_created
    
    def _add_cross_level_relationship(self, source_node: str, target_node: str, 
                                    relationship_type: str, source_type: str, target_type: str):
        """Add relationship between nodes in different hierarchical levels."""
        try:
            # Determine levels
            level_mapping = {
                'Patient': 'patients',
                'Encounter': 'encounters',
                'Observation': 'observations',
                'Condition': 'conditions',
                'Procedure': 'procedures',
                'Medication': 'medications',
                'MedicationRequest': 'medications',
                'Practitioner': 'practitioners',
                'Organization': 'organizations'
            }
            
            source_level = level_mapping.get(source_type, 'care_coordination')
            target_level = level_mapping.get(target_type, 'care_coordination')
            
            # Add relationship with semantic properties
            relationship_props = {
                'fhir_relationship_type': relationship_type,
                'source_resource_type': source_type,
                'target_resource_type': target_type,
                'created_at': datetime.now().isoformat()
            }
            
            if source_level == target_level:
                # Same level relationship
                level_kg = self.hierarchical_kg.get_level(source_level)
                level_kg.add_edge(source_node, target_node, relationship_props)
            else:
                # Cross-level relationship - add to both levels
                if source_level in self.hierarchical_kg.levels:
                    source_kg = self.hierarchical_kg.get_level(source_level)
                    source_kg.add_edge(source_node, target_node, relationship_props)
                
                if target_level in self.hierarchical_kg.levels:
                    target_kg = self.hierarchical_kg.get_level(target_level)
                    target_kg.add_edge(target_node, source_node, {
                        **relationship_props,
                        'fhir_relationship_type': f"inverse_{relationship_type}"
                    })
                    
        except Exception as e:
            logger.error(f"Error creating relationship {source_node} -> {target_node}: {e}")
    
    def load_fhir_data_batched(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Load FHIR data in batches for memory efficiency."""
        logger.info("Starting batched FHIR data loading...")
        
        start_time = time.time()
        
        # Get list of files to process
        json_files = list(self.fhir_data_dir.glob("*.json"))
        
        if max_files:
            json_files = json_files[:max_files]
            logger.info(f"Processing limited set: {max_files} files")
        
        total_files = len(json_files)
        logger.info(f"Total files to process: {total_files:,}")
        
        # Process in batches
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = json_files[batch_start:batch_end]
            
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{(total_files-1)//self.batch_size + 1}: files {batch_start+1}-{batch_end}")
            
            # Process each file in the batch
            for file_path in batch_files:
                try:
                    with open(file_path, 'r') as f:
                        bundle_data = json.load(f)
                    
                    result = self.process_fhir_bundle(bundle_data, str(file_path))
                    
                    if result['status'] == 'processed':
                        self.stats['files_processed'] += 1
                        self.stats['total_resources'] += result['resources_processed']
                        
                        for resource_type, count in result['resource_types'].items():
                            self.stats['resource_counts'][resource_type] += count
                    
                    # Progress reporting
                    if self.stats['files_processed'] % self.progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = self.stats['files_processed'] / elapsed
                        eta = (total_files - self.stats['files_processed']) / rate
                        
                        logger.info(f"Progress: {self.stats['files_processed']:,}/{total_files:,} files "
                                  f"({self.stats['files_processed']/total_files*100:.1f}%) - "
                                  f"Rate: {rate:.1f} files/sec - ETA: {eta/60:.1f} min")
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    self.stats['errors'].append(f"File {file_path}: {e}")
            
            batch_time = time.time() - batch_start_time
            self.stats['batch_times'].append(batch_time)
            
            logger.info(f"Batch completed in {batch_time:.2f} seconds")
        
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"FHIR data loading completed!")
        logger.info(f"Files processed: {self.stats['files_processed']:,}")
        logger.info(f"Total resources: {self.stats['total_resources']:,}")
        logger.info(f"Total time: {self.stats['processing_time']:.2f} seconds")
        
        return {
            'status': 'completed',
            'stats': dict(self.stats),
            'resource_counts': dict(self.stats['resource_counts'])
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary of loaded FHIR data."""
        logger.info("Generating analytics summary...")
        
        analytics = {
            'loading_stats': {
                'files_processed': self.stats['files_processed'],
                'total_resources': self.stats['total_resources'],
                'processing_time': self.stats['processing_time'],
                'average_file_processing_time': self.stats['processing_time'] / max(self.stats['files_processed'], 1),
                'resources_per_second': self.stats['total_resources'] / max(self.stats['processing_time'], 1)
            },
            'resource_distribution': dict(self.stats['resource_counts']),
            'hierarchical_structure': {},
            'ontology_integration': {
                'resource_types_mapped': len(self.ontology_manager.resource_types),
                'semantic_relationships': 0
            }
        }
        
        # Get hierarchical structure information
        if self.hierarchical_kg:
            for level_name in self.hierarchical_kg.levels:
                level_kg = self.hierarchical_kg.get_level(level_name)
                analytics['hierarchical_structure'][level_name] = {
                    'nodes': level_kg.num_nodes(),
                    'edges': level_kg.num_edges()
                }
                analytics['ontology_integration']['semantic_relationships'] += level_kg.num_edges()
        
        return analytics


def main():
    """Main function to demonstrate production FHIR loading."""
    
    print("🚀 PRODUCTION FHIR DATA LOADER WITH ONTOLOGY INTEGRATION")
    print("=" * 80)
    
    # Initialize the production loader
    fhir_data_dir = "/home/amansingh/dev/andola/healthcare/synthea/output/fhir"
    loader = ProductionFHIRLoader(fhir_data_dir, "ProductionFHIRKG")
    
    try:
        # Initialize hierarchical graph with FHIR ontology
        print("\n🏗️ PHASE 1: INITIALIZING HIERARCHICAL KNOWLEDGE GRAPH")
        print("-" * 60)
        init_result = loader.initialize_hierarchical_graph()
        print(f"✅ Graph initialized: {init_result['levels_created']} levels created")
        print(f"📚 FHIR ontology loaded: {init_result['ontology_stats']}")
        
        # Discover FHIR files
        print("\n🔍 PHASE 2: DISCOVERING FHIR DATA")
        print("-" * 60)
        discovery = loader.discover_fhir_files()
        print(f"📁 Found {discovery['total_files']:,} FHIR files")
        print(f"💾 Estimated total size: {discovery['estimated_total_size_gb']:.2f} GB")
        print(f"📊 Estimated resources: ~{discovery['estimated_total_resources']:,}")
        print(f"🔬 Resource types found: {list(discovery['resource_types_found'].keys())}")
        
        # Load data (limited sample for demo)
        print("\n⚡ PHASE 3: LOADING FHIR DATA")
        print("-" * 60)
        print("Loading sample of 1000 files for demonstration...")
        
        loading_result = loader.load_fhir_data_batched(max_files=1000)
        
        print(f"✅ Loading completed!")
        print(f"📊 Files processed: {loading_result['stats']['files_processed']:,}")
        print(f"🔬 Total resources: {loading_result['stats']['total_resources']:,}")
        print(f"⏱️ Processing time: {loading_result['stats']['processing_time']:.2f} seconds")
        
        # Generate analytics
        print("\n📈 PHASE 4: ANALYTICS SUMMARY")
        print("-" * 60)
        analytics = loader.get_analytics_summary()
        
        print(f"🔄 Processing rate: {analytics['loading_stats']['resources_per_second']:.0f} resources/second")
        print(f"📋 Resource distribution:")
        for resource_type, count in analytics['resource_distribution'].items():
            print(f"   {resource_type}: {count:,}")
        
        print(f"\n🏗️ Hierarchical structure:")
        for level_name, stats in analytics['hierarchical_structure'].items():
            print(f"   {level_name}: {stats['nodes']:,} nodes, {stats['edges']:,} relationships")
        
        print(f"\n🧠 Ontology integration:")
        print(f"   Semantic relationships: {analytics['ontology_integration']['semantic_relationships']:,}")
        print(f"   Resource types mapped: {analytics['ontology_integration']['resource_types_mapped']}")
        
        if loading_result['stats']['errors']:
            print(f"\n⚠️ Errors encountered: {len(loading_result['stats']['errors'])}")
            for error in loading_result['stats']['errors'][:5]:  # Show first 5 errors
                print(f"   {error}")
        
        print("\n" + "=" * 80)
        print("✅ PRODUCTION FHIR LOADING DEMONSTRATION COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()