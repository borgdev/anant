"""
FHIR Data Loader for Hierarchical Knowledge Graph
===============================================

This module loads FHIR JSON data files into ANANT's hierarchical knowledge graph,
preserving all relationships and maintaining the integrity of FHIR resource structures.

Features:
- Load FHIR JSON Bundle files
- Extract all resource types (Patient, Observation, Condition, etc.)
- Preserve relationships between resources
- Create hierarchical structure based on resource types
- Handle references and identifiers
- Support for large datasets with efficient processing
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
import uuid
from collections import defaultdict

# ANANT imports
from anant.kg import HierarchicalKnowledgeGraph

logger = logging.getLogger(__name__)


class FHIRDataLoader:
    """
    Loads FHIR JSON data files into hierarchical knowledge graph.
    
    This class handles the parsing of FHIR Bundle JSON files and creates
    a structured hierarchy that preserves all resource relationships.
    """
    
    def __init__(self, data_dir: str = "data/output/fhir", hkg: Optional[HierarchicalKnowledgeGraph] = None):
        """
        Initialize the FHIR data loader.
        
        Args:
            data_dir: Directory containing FHIR JSON data files
            hkg: Existing hierarchical knowledge graph (will create new if None)
        """
        self.data_dir = Path(data_dir)
        self.hkg = hkg or HierarchicalKnowledgeGraph("FHIR_Data")
        self.loaded_files = []
        self.resource_counts = defaultdict(int)
        self.reference_map = {}  # resource_id -> full_resource_data
        self.relationships = []  # List of relationships to add
        
        logger.info(f"Initialized FHIR data loader for {data_dir}")
    
    def setup_resource_hierarchy(self):
        """Create hierarchical levels for FHIR resource organization."""
        levels = [
            {
                'id': 'patients',
                'name': 'Patients',
                'description': 'Patient demographic and administrative data',
                'order': 0
            },
            {
                'id': 'practitioners',
                'name': 'Practitioners',
                'description': 'Healthcare practitioners and providers',
                'order': 1
            },
            {
                'id': 'organizations',
                'name': 'Organizations',
                'description': 'Healthcare organizations and facilities',
                'order': 2
            },
            {
                'id': 'encounters',
                'name': 'Encounters',
                'description': 'Healthcare encounters and episodes',
                'order': 3
            },
            {
                'id': 'observations',
                'name': 'Observations',
                'description': 'Clinical observations and measurements',
                'order': 4
            },
            {
                'id': 'conditions',
                'name': 'Conditions',
                'description': 'Medical conditions and diagnoses',
                'order': 5
            },
            {
                'id': 'procedures',
                'name': 'Procedures',
                'description': 'Medical procedures and interventions',
                'order': 6
            },
            {
                'id': 'medications',
                'name': 'Medications',
                'description': 'Medication-related resources',
                'order': 7
            },
            {
                'id': 'care_plans',
                'name': 'Care Plans',
                'description': 'Care plans and goals',
                'order': 8
            },
            {
                'id': 'other_resources',
                'name': 'Other Resources',
                'description': 'Other FHIR resource types',
                'order': 9
            }
        ]
        
        for level in levels:
            if not self.hkg.get_level_metadata(level['id']):
                self.hkg.create_level(
                    level['id'],
                    level['name'],
                    level['description'],
                    level['order']
                )
    
    def load_fhir_data_files(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Load FHIR JSON data files from the data directory.
        
        Args:
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            Dictionary with loading statistics and metadata
        """
        results = {
            'loaded_files': [],
            'total_files': 0,
            'total_resources': 0,
            'resource_types': {},
            'errors': []
        }
        
        # Setup hierarchy first
        self.setup_resource_hierarchy()
        
        # Find JSON files
        json_files = list(self.data_dir.glob("*.json"))
        results['total_files'] = len(json_files)
        
        if max_files:
            json_files = json_files[:max_files]
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for i, file_path in enumerate(json_files):
            try:
                if i % 100 == 0:
                    logger.info(f"Processing file {i+1}/{len(json_files)}: {file_path.name}")
                
                file_results = self._load_single_fhir_file(file_path)
                
                results['loaded_files'].append(str(file_path))
                results['total_resources'] += file_results['resources_processed']
                
                # Update resource type counts
                for resource_type, count in file_results['resource_types'].items():
                    if resource_type in results['resource_types']:
                        results['resource_types'][resource_type] += count
                    else:
                        results['resource_types'][resource_type] = count
                
            except Exception as e:
                error_msg = f"Failed to load {file_path.name}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Process relationships after all resources are loaded
        logger.info("Processing relationships...")
        self._process_relationships()
        
        self.loaded_files = results['loaded_files']
        return results
    
    def _load_single_fhir_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single FHIR JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {
            'resources_processed': 0,
            'resource_types': defaultdict(int)
        }
        
        # Check if it's a Bundle
        if data.get('resourceType') == 'Bundle':
            entries = data.get('entry', [])
            
            for entry in entries:
                resource = entry.get('resource', {})
                if resource:
                    self._process_fhir_resource(resource, str(file_path))
                    results['resources_processed'] += 1
                    results['resource_types'][resource.get('resourceType', 'Unknown')] += 1
        
        else:
            # Single resource file
            self._process_fhir_resource(data, str(file_path))
            results['resources_processed'] = 1
            results['resource_types'][data.get('resourceType', 'Unknown')] = 1
        
        return results
    
    def _process_fhir_resource(self, resource: Dict[str, Any], source_file: str):
        """Process a single FHIR resource and add it to the knowledge graph."""
        resource_type = resource.get('resourceType', 'Unknown')
        resource_id = resource.get('id', str(uuid.uuid4()))
        
        # Store in reference map for relationship processing
        full_resource_id = f"{resource_type}/{resource_id}"
        self.reference_map[full_resource_id] = resource
        
        # Determine appropriate level for this resource type
        level_id = self._get_level_for_resource_type(resource_type)
        
        # Extract resource properties
        properties = self._extract_resource_properties(resource, source_file)
        
        # Add to knowledge graph
        self.hkg.add_node(
            full_resource_id,
            properties,
            level_id
        )
        
        # Extract relationships for later processing
        self._extract_resource_relationships(resource, full_resource_id)
        
        # Update counts
        self.resource_counts[resource_type] += 1
    
    def _get_level_for_resource_type(self, resource_type: str) -> str:
        """Determine which hierarchical level a resource type belongs to."""
        resource_type_mapping = {
            'Patient': 'patients',
            'Practitioner': 'practitioners',
            'PractitionerRole': 'practitioners',
            'Organization': 'organizations',
            'Encounter': 'encounters',
            'Observation': 'observations',
            'Condition': 'conditions',
            'Procedure': 'procedures',
            'MedicationRequest': 'medications',
            'MedicationAdministration': 'medications',
            'MedicationStatement': 'medications',
            'Medication': 'medications',
            'CarePlan': 'care_plans',
            'Goal': 'care_plans',
            'AllergyIntolerance': 'conditions',
            'DiagnosticReport': 'observations',
            'Immunization': 'procedures',
            'CareTeam': 'care_plans'
        }
        
        return resource_type_mapping.get(resource_type, 'other_resources')
    
    def _extract_resource_properties(self, resource: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Extract properties from a FHIR resource."""
        properties = {
            'resource_type': resource.get('resourceType'),
            'resource_id': resource.get('id'),
            'source_file': source_file,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Extract common FHIR elements
        if 'meta' in resource:
            meta = resource['meta']
            properties['meta_version_id'] = meta.get('versionId')
            properties['meta_last_updated'] = meta.get('lastUpdated')
            properties['meta_profile'] = meta.get('profile', [])
        
        if 'text' in resource:
            properties['narrative_status'] = resource['text'].get('status')
        
        # Resource-specific property extraction
        if resource.get('resourceType') == 'Patient':
            self._extract_patient_properties(resource, properties)
        elif resource.get('resourceType') == 'Practitioner':
            self._extract_practitioner_properties(resource, properties)
        elif resource.get('resourceType') == 'Organization':
            self._extract_organization_properties(resource, properties)
        elif resource.get('resourceType') == 'Encounter':
            self._extract_encounter_properties(resource, properties)
        elif resource.get('resourceType') == 'Observation':
            self._extract_observation_properties(resource, properties)
        elif resource.get('resourceType') == 'Condition':
            self._extract_condition_properties(resource, properties)
        elif resource.get('resourceType') == 'Procedure':
            self._extract_procedure_properties(resource, properties)
        elif resource.get('resourceType') in ['MedicationRequest', 'MedicationAdministration', 'MedicationStatement']:
            self._extract_medication_properties(resource, properties)
        
        # Store full resource data for complex queries
        properties['_raw_resource'] = json.dumps(resource)
        
        return properties
    
    def _extract_patient_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Patient-specific properties."""
        # Basic demographics
        properties['gender'] = resource.get('gender')
        properties['birth_date'] = resource.get('birthDate')
        properties['deceased'] = resource.get('deceasedBoolean', resource.get('deceasedDateTime'))
        
        # Names
        names = resource.get('name', [])
        if names:
            name = names[0]  # Use first name
            properties['family_name'] = name.get('family')
            properties['given_names'] = name.get('given', [])
            properties['name_use'] = name.get('use')
        
        # Addresses
        addresses = resource.get('address', [])
        if addresses:
            address = addresses[0]  # Use first address
            properties['city'] = address.get('city')
            properties['state'] = address.get('state')
            properties['postal_code'] = address.get('postalCode')
            properties['country'] = address.get('country')
        
        # Contact info
        telecoms = resource.get('telecom', [])
        for telecom in telecoms:
            system = telecom.get('system')
            if system == 'phone':
                properties['phone'] = telecom.get('value')
            elif system == 'email':
                properties['email'] = telecom.get('value')
        
        # Extensions
        extensions = resource.get('extension', [])
        for ext in extensions:
            url = ext.get('url', '')
            if 'race' in url:
                properties['race'] = self._extract_coding_display(ext)
            elif 'ethnicity' in url:
                properties['ethnicity'] = self._extract_coding_display(ext)
            elif 'birthsex' in url:
                properties['birth_sex'] = ext.get('valueCode')
    
    def _extract_practitioner_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Practitioner-specific properties."""
        properties['active'] = resource.get('active')
        
        # Names
        names = resource.get('name', [])
        if names:
            name = names[0]
            properties['family_name'] = name.get('family')
            properties['given_names'] = name.get('given', [])
        
        # Qualifications
        qualifications = resource.get('qualification', [])
        properties['qualifications'] = [q.get('code', {}).get('text') for q in qualifications]
    
    def _extract_organization_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Organization-specific properties."""
        properties['name'] = resource.get('name')
        properties['active'] = resource.get('active')
        
        # Type
        org_type = resource.get('type', [])
        if org_type:
            properties['organization_type'] = org_type[0].get('text')
    
    def _extract_encounter_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Encounter-specific properties."""
        properties['status'] = resource.get('status')
        properties['class'] = resource.get('class', {}).get('code')
        
        # Period
        period = resource.get('period', {})
        properties['start'] = period.get('start')
        properties['end'] = period.get('end')
        
        # Type
        encounter_type = resource.get('type', [])
        if encounter_type:
            properties['encounter_type'] = encounter_type[0].get('text')
    
    def _extract_observation_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Observation-specific properties."""
        properties['status'] = resource.get('status')
        properties['category'] = [cat.get('text') for cat in resource.get('category', [])]
        
        # Code
        code = resource.get('code', {})
        properties['observation_code'] = code.get('text')
        properties['observation_coding'] = code.get('coding', [])
        
        # Value
        if 'valueQuantity' in resource:
            value = resource['valueQuantity']
            properties['value_quantity'] = value.get('value')
            properties['value_unit'] = value.get('unit')
        elif 'valueCodeableConcept' in resource:
            properties['value_concept'] = resource['valueCodeableConcept'].get('text')
        elif 'valueString' in resource:
            properties['value_string'] = resource['valueString']
        
        # Effective date/time
        properties['effective_datetime'] = resource.get('effectiveDateTime')
        properties['effective_period'] = resource.get('effectivePeriod')
    
    def _extract_condition_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Condition-specific properties."""
        properties['clinical_status'] = resource.get('clinicalStatus', {}).get('text')
        properties['verification_status'] = resource.get('verificationStatus', {}).get('text')
        
        # Code
        code = resource.get('code', {})
        properties['condition_code'] = code.get('text')
        properties['condition_coding'] = code.get('coding', [])
        
        # Onset and abatement
        properties['onset_datetime'] = resource.get('onsetDateTime')
        properties['abatement_datetime'] = resource.get('abatementDateTime')
    
    def _extract_procedure_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Procedure-specific properties."""
        properties['status'] = resource.get('status')
        
        # Code
        code = resource.get('code', {})
        properties['procedure_code'] = code.get('text')
        properties['procedure_coding'] = code.get('coding', [])
        
        # Performed date/time
        properties['performed_datetime'] = resource.get('performedDateTime')
        properties['performed_period'] = resource.get('performedPeriod')
    
    def _extract_medication_properties(self, resource: Dict[str, Any], properties: Dict[str, Any]):
        """Extract Medication-related properties."""
        properties['status'] = resource.get('status')
        properties['intent'] = resource.get('intent')
        
        # Medication
        medication = resource.get('medicationCodeableConcept', {})
        properties['medication_code'] = medication.get('text')
        properties['medication_coding'] = medication.get('coding', [])
        
        # Dosage
        dosage_instructions = resource.get('dosageInstruction', [])
        if dosage_instructions:
            dosage = dosage_instructions[0]
            properties['dosage_text'] = dosage.get('text')
    
    def _extract_coding_display(self, extension: Dict[str, Any]) -> str:
        """Extract display value from coding extension."""
        for ext in extension.get('extension', []):
            if ext.get('url') == 'ombCategory':
                value_coding = ext.get('valueCoding', {})
                return value_coding.get('display', '')
        return ''
    
    def _extract_resource_relationships(self, resource: Dict[str, Any], resource_id: str):
        """Extract relationships from a FHIR resource."""
        resource_type = resource.get('resourceType')
        
        # Common reference fields
        reference_fields = {
            'subject': 'subject',
            'patient': 'patient',
            'practitioner': 'practitioner',
            'organization': 'organization',
            'encounter': 'encounter',
            'basedOn': 'basedOn',
            'partOf': 'partOf'
        }
        
        # Extract direct references
        for field, relation_type in reference_fields.items():
            if field in resource:
                ref_value = resource[field]
                if isinstance(ref_value, dict) and 'reference' in ref_value:
                    target_id = ref_value['reference']
                    self.relationships.append({
                        'source': resource_id,
                        'target': target_id,
                        'type': relation_type,
                        'metadata': {
                            'source_resource_type': resource_type,
                            'field': field
                        }
                    })
                elif isinstance(ref_value, list):
                    for ref in ref_value:
                        if isinstance(ref, dict) and 'reference' in ref:
                            target_id = ref['reference']
                            self.relationships.append({
                                'source': resource_id,
                                'target': target_id,
                                'type': relation_type,
                                'metadata': {
                                    'source_resource_type': resource_type,
                                    'field': field
                                }
                            })
        
        # Resource-specific relationship extraction
        if resource_type == 'Encounter':
            # Encounter participants
            participants = resource.get('participant', [])
            for participant in participants:
                if 'individual' in participant and 'reference' in participant['individual']:
                    target_id = participant['individual']['reference']
                    self.relationships.append({
                        'source': resource_id,
                        'target': target_id,
                        'type': 'participant',
                        'metadata': {
                            'source_resource_type': resource_type,
                            'participant_type': participant.get('type', [])
                        }
                    })
        
        elif resource_type == 'Observation':
            # Observation performers
            performers = resource.get('performer', [])
            for performer in performers:
                if 'reference' in performer:
                    target_id = performer['reference']
                    self.relationships.append({
                        'source': resource_id,
                        'target': target_id,
                        'type': 'performer',
                        'metadata': {
                            'source_resource_type': resource_type
                        }
                    })
    
    def _process_relationships(self):
        """Process all extracted relationships and add them to the knowledge graph."""
        relationship_id = 0
        
        for rel in self.relationships:
            relationship_id += 1
            
            source_id = rel['source']
            target_ref = rel['target']
            rel_type = rel['type']
            metadata = rel['metadata']
            
            # Resolve target reference
            target_id = self._resolve_reference(target_ref)
            
            # Only add relationship if both source and target exist
            if target_id and self.hkg.has_node(source_id) and self.hkg.has_node(target_id):
                self.hkg.add_cross_level_relationship(
                    source_id,
                    target_id,
                    rel_type,
                    {
                        **metadata,
                        'relationship_type': rel_type,
                        'semantic_weight': 0.8,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
    
    def _resolve_reference(self, reference: str) -> Optional[str]:
        """Resolve a FHIR reference to a resource ID."""
        # Handle different reference formats
        if reference.startswith('urn:uuid:'):
            # UUID reference
            uuid_part = reference.replace('urn:uuid:', '')
            # Find resource with this UUID
            for resource_id, resource_data in self.reference_map.items():
                if resource_data.get('id') == uuid_part:
                    return resource_id
        
        elif '/' in reference:
            # Resource type/ID reference
            return reference
        
        else:
            # Direct ID reference - try to find in reference map
            for resource_id in self.reference_map.keys():
                if resource_id.endswith(f"/{reference}"):
                    return resource_id
        
        return None
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded FHIR data.
        
        Returns:
            Dictionary with data loading statistics
        """
        if not self.hkg:
            return {'error': 'No hierarchical knowledge graph available'}
        
        stats = self.hkg.get_hierarchy_statistics()
        
        # Add FHIR-specific statistics
        stats.update({
            'loaded_files': len(self.loaded_files),
            'resource_counts': dict(self.resource_counts),
            'total_resources': sum(self.resource_counts.values()),
            'relationship_count': len(self.relationships),
            'levels': {
                level_id: {
                    'entities': len(self.hkg.get_nodes_at_level(level_id)),
                    'metadata': self.hkg.get_level_metadata(level_id)
                }
                for level_id in ['patients', 'practitioners', 'organizations', 
                               'encounters', 'observations', 'conditions', 
                               'procedures', 'medications', 'care_plans', 'other_resources']
                if self.hkg.get_level_metadata(level_id)
            }
        })
        
        return stats


def load_fhir_data(data_dir: str = "data/output/fhir", 
                   hkg: Optional[HierarchicalKnowledgeGraph] = None,
                   max_files: Optional[int] = None) -> Tuple[HierarchicalKnowledgeGraph, Dict[str, Any]]:
    """
    Convenience function to load FHIR data into a hierarchical knowledge graph.
    
    Args:
        data_dir: Directory containing FHIR JSON files
        hkg: Existing hierarchical knowledge graph (will create new if None)
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        Tuple of (HierarchicalKnowledgeGraph, statistics)
    """
    loader = FHIRDataLoader(data_dir, hkg)
    
    # Load FHIR data files
    load_results = loader.load_fhir_data_files(max_files)
    
    if load_results['errors']:
        logger.warning(f"Encountered {len(load_results['errors'])} errors during loading")
    
    # Get statistics
    stats = loader.get_data_statistics()
    stats['load_results'] = load_results
    
    return loader.hkg, stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load FHIR data
    hkg, stats = load_fhir_data("data/output/fhir", max_files=10)
    
    print("\n=== FHIR Data Loading Results ===")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Total resources: {stats['total_resources']}")
    print(f"Files processed: {stats['loaded_files']}")
    
    print("\nResource type breakdown:")
    for resource_type, count in stats['resource_counts'].items():
        print(f"  {resource_type}: {count}")
    
    print("\nLevel breakdown:")
    for level_id, level_info in stats['levels'].items():
        print(f"  {level_id}: {level_info['entities']} entities")