"""
Large-Scale Production FHIR Data Processor
==========================================

Handles 435GB+ FHIR datasets with:
- Memory-efficient streaming processing
- Partitioned Parquet output 
- Distributed graph construction
- Advanced memory management
- Progress tracking and monitoring

Designed for 116K+ FHIR Bundle files
"""

import json
import logging
import time
import uuid
import gc
from datetime import datetime
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import uuid

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/media/amansingh/data/fhir_test/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import hierarchical knowledge graph components
try:
    from anant.core.knowledge_graph import KnowledgeGraph
    from anant.core.hierarchical_knowledge_graph import HierarchicalKnowledgeGraph
    GRAPH_SUPPORT = True
    logger.info("Knowledge graph support enabled")
except ImportError as e:
    logger.warning(f"Knowledge graph support disabled: {e}")
    GRAPH_SUPPORT = False
    
    # Create placeholder classes for when graph support is unavailable
    class KnowledgeGraph:
        """Placeholder KnowledgeGraph class when anant.core is not available."""
        def __init__(self, name: str = "", *args, **kwargs):
            self.name = name
            self.entities = {}
            self.relationships = {}
            
        def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
            """Add entity to graph."""
            self.entities[entity_id] = {
                'type': entity_type,
                'properties': properties
            }
            
        def add_relationship(self, rel_id: str, from_entity: str, to_entity: str, rel_type: str, properties: Dict[str, Any]):
            """Add relationship to graph."""
            self.relationships[rel_id] = {
                'from': from_entity,
                'to': to_entity,
                'type': rel_type,
                'properties': properties
            }
            
        def get_entity_count(self) -> int:
            """Get number of entities in graph."""
            return len(self.entities)
            
        def get_relationship_count(self) -> int:
            """Get number of relationships in graph."""
            return len(self.relationships)
            
        # Legacy methods for compatibility
        def add_node(self, *args, **kwargs):
            pass
        def add_edge(self, *args, **kwargs):
            pass
        def num_nodes(self):
            return len(self.entities)
        def num_edges(self):
            return len(self.relationships)
    
    class HierarchicalKnowledgeGraph:
        """Placeholder HierarchicalKnowledgeGraph class when anant.core is not available."""
        def __init__(self, name: str = "", semantic_reasoning: bool = False, *args, **kwargs):
            self.name = name
            self.semantic_reasoning = semantic_reasoning
            self.levels = {}
            self.level_order = {}
            
        def add_level(self, level_name: str, kg: KnowledgeGraph, order: int, description: str = ""):
            """Add a level to the hierarchical graph."""
            self.levels[level_name] = kg
            self.level_order[level_name] = order
            
        def get_level(self, level_name: str) -> 'KnowledgeGraph':
            """Get a specific level."""
            return self.levels.get(level_name, KnowledgeGraph())
            
        def get_total_entities(self) -> int:
            """Get total entities across all levels."""
            return sum(kg.get_entity_count() for kg in self.levels.values())
            
        def get_total_relationships(self) -> int:
            """Get total relationships across all levels."""
            return sum(kg.get_relationship_count() for kg in self.levels.values())
            
        def get_level_summary(self) -> Dict[str, Dict[str, int]]:
            """Get summary of all levels."""
            summary = {}
            for level_name, kg in self.levels.items():
                summary[level_name] = {
                    'entities': kg.get_entity_count(),
                    'relationships': kg.get_relationship_count(),
                    'order': self.level_order.get(level_name, 0)
                }
            return summary


@dataclass
class ProcessingStats:
    """Statistics tracking for large-scale processing."""
    files_processed: int = 0
    total_resources: int = 0
    
    # Core clinical resources
    total_patients: int = 0
    total_encounters: int = 0
    total_observations: int = 0
    total_conditions: int = 0
    total_procedures: int = 0
    total_medications: int = 0
    
    # Additional resource counts
    total_practitioners: int = 0
    total_organizations: int = 0
    total_diagnostic_reports: int = 0
    total_immunizations: int = 0
    total_allergies: int = 0
    total_care_plans: int = 0
    total_appointments: int = 0
    total_devices: int = 0
    total_specimens: int = 0
    total_other_resources: int = 0
    
    # Processing metrics
    bytes_processed: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    parquet_files_written: int = 0
    partitions_created: int = 0
    
    # Resource type distribution
    resource_type_counts: Optional[Dict[str, int]] = None
    errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.resource_type_counts is None:
            self.resource_type_counts = defaultdict(int)


class MemoryManager:
    """Advanced memory management for large-scale processing."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.process = psutil.Process()
        self.memory_warning_threshold = 0.8  # 80% of max memory
        self.gc_threshold = 0.9  # Force GC at 90%
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in bytes and percentage."""
        memory_info = self.process.memory_info()
        current_bytes = memory_info.rss
        percentage = current_bytes / self.max_memory_bytes
        return current_bytes, percentage
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        _, percentage = self.get_memory_usage()
        
        if percentage > self.gc_threshold:
            logger.warning(f"High memory usage ({percentage:.1%}), forcing garbage collection")
            gc.collect()
            return True
        elif percentage > self.memory_warning_threshold:
            logger.warning(f"Memory usage warning: {percentage:.1%}")
            return True
        
        return False
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        current_bytes, percentage = self.get_memory_usage()
        
        return {
            'current_mb': current_bytes / (1024**2),
            'current_gb': current_bytes / (1024**3),
            'percentage': percentage * 100,
            'available_gb': (self.max_memory_bytes - current_bytes) / (1024**3)
        }


class PartitionedParquetWriter:
    """Manages partitioned Parquet output for efficient storage and querying."""
    
    def __init__(self, output_dir: str, partition_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.partition_size = partition_size
        self.current_partitions = {}
        self.partition_counters = defaultdict(int)
        self.schemas = {}
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        
        # Complete FHIR resource type directory mapping
        self.resource_dirs = self._create_fhir_resource_directories()
        
    def _create_fhir_resource_directories(self) -> Dict[str, Path]:
        """Create directories for all FHIR resource types."""
        
        # Complete FHIR R4/R5 resource types organized by category
        fhir_resource_types = {
            # Foundation Resources
            'foundation': [
                'Questionnaire', 'QuestionnaireResponse', 'List', 'Library', 'Basic',
                'Binary', 'Bundle', 'Parameters', 'OperationOutcome'
            ],
            
            # Base Resources
            'base': [
                'Patient', 'Practitioner', 'PractitionerRole', 'RelatedPerson', 'Person',
                'Organization', 'OrganizationAffiliation', 'HealthcareService', 'Endpoint',
                'Location', 'Substance', 'Group', 'Device', 'DeviceDefinition'
            ],
            
            # Clinical Resources
            'clinical': [
                'Condition', 'Observation', 'DiagnosticReport', 'ImagingStudy', 
                'Specimen', 'BodyStructure', 'AllergyIntolerance', 'Procedure',
                'FamilyMemberHistory', 'ClinicalAssessment', 'DetectedIssue',
                'RiskAssessment', 'Goal', 'CarePlan', 'CareTeam', 'Flag'
            ],
            
            # Encounters & Episodes
            'encounters': [
                'Encounter', 'EncounterHistory', 'EpisodeOfCare', 'Appointment',
                'AppointmentResponse', 'Schedule', 'Slot'
            ],
            
            # Medications
            'medications': [
                'Medication', 'MedicationRequest', 'MedicationAdministration',
                'MedicationDispense', 'MedicationStatement', 'MedicationKnowledge',
                'Immunization', 'ImmunizationEvaluation', 'ImmunizationRecommendation'
            ],
            
            # Devices & Diagnostics
            'devices_diagnostics': [
                'DeviceRequest', 'DeviceUsage', 'DeviceMetric', 'DeviceAssociation',
                'DeviceAlert', 'DeviceDispense', 'ImagingSelection'
            ],
            
            # Care Provision
            'care_provision': [
                'ServiceRequest', 'Task', 'Communication', 'CommunicationRequest',
                'RequestOrchestration', 'Transport', 'NutritionOrder', 'NutritionIntake',
                'NutritionProduct', 'VisionPrescription', 'SupplyRequest', 'SupplyDelivery'
            ],
            
            # Financial
            'financial': [
                'Coverage', 'CoverageEligibilityRequest', 'CoverageEligibilityResponse',
                'EnrollmentRequest', 'EnrollmentResponse', 'Claim', 'ClaimResponse',
                'Invoice', 'PaymentNotice', 'PaymentReconciliation', 'Account',
                'ChargeItem', 'ChargeItemDefinition', 'Contract', 'ExplanationOfBenefit',
                'InsurancePlan', 'InsuranceProduct'
            ],
            
            # Specialized
            'specialized': [
                'AdverseEvent', 'BiologicallyDerivedProduct', 'BiologicallyDerivedProductDispense',
                'ResearchStudy', 'ResearchSubject', 'Evidence', 'EvidenceVariable',
                'AuditEvent', 'Provenance', 'Consent', 'Permission', 'Citation',
                'GenomicStudy', 'MolecularDefinition', 'PersonalRelationship'
            ],
            
            # Definitional Resources
            'definitional': [
                'ActivityDefinition', 'PlanDefinition', 'Questionnaire', 'Measure',
                'EventDefinition', 'ChargeItemDefinition', 'ObservationDefinition',
                'SpecimenDefinition', 'DeviceDefinition', 'ActorDefinition',
                'Requirements', 'SubscriptionTopic'
            ],
            
            # Terminology
            'terminology': [
                'CodeSystem', 'ValueSet', 'ConceptMap', 'NamingSystem',
                'TerminologyCapabilities', 'StructureDefinition', 'StructureMap',
                'GraphDefinition', 'ExampleScenario', 'ImplementationGuide',
                'CapabilityStatement', 'OperationDefinition', 'SearchParameter',
                'CompartmentDefinition', 'MessageDefinition'
            ],
            
            # Products & Substances
            'products_substances': [
                'MedicinalProductDefinition', 'PackagedProductDefinition',
                'AdministrableProductDefinition', 'ManufacturedItemDefinition',
                'Ingredient', 'SubstanceDefinition', 'ClinicalUseDefinition',
                'RegulatedAuthorization'
            ],
            
            # Quality & Reporting
            'quality_reporting': [
                'MeasureReport', 'GuidanceResponse', 'ArtifactAssessment',
                'ConditionDefinition', 'FormularyItem', 'InventoryItem',
                'InventoryReport', 'VerificationResult', 'Linkage'
            ],
            
            # Workflow & Messaging
            'workflow_messaging': [
                'MessageHeader', 'Subscription', 'SubscriptionStatus',
                'DocumentReference', 'Composition'
            ]
        }
        
        # Create directories for all resource types
        resource_dirs = {}
        
        for category, resource_types in fhir_resource_types.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for resource_type in resource_types:
                resource_key = resource_type.lower()
                resource_dir = category_dir / resource_key
                resource_dir.mkdir(exist_ok=True)
                resource_dirs[resource_key] = resource_dir
        
        return resource_dirs
    
    def get_schema(self, resource_type: str) -> pa.Schema:
        """Get or create Arrow schema for resource type."""
        resource_key = resource_type.lower()
        
        if resource_key not in self.schemas:
            # Define schemas for core FHIR resources
            if resource_key == 'patient':
                self.schemas[resource_key] = pa.schema([
                    ('patient_id', pa.string()),
                    ('file_path', pa.string()),
                    ('gender', pa.string()),
                    ('birth_date', pa.string()),
                    ('age', pa.int32()),
                    ('family_name', pa.string()),
                    ('given_name', pa.string()),
                    ('race', pa.string()),
                    ('ethnicity', pa.string()),
                    ('marital_status', pa.string()),
                    ('city', pa.string()),
                    ('state', pa.string()),
                    ('country', pa.string()),
                    ('active', pa.bool_()),
                    ('deceased', pa.bool_()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key == 'encounter':
                self.schemas[resource_key] = pa.schema([
                    ('encounter_id', pa.string()),
                    ('patient_id', pa.string()),
                    ('file_path', pa.string()),
                    ('status', pa.string()),
                    ('class_code', pa.string()),
                    ('class_display', pa.string()),
                    ('start_time', pa.timestamp('ms')),
                    ('end_time', pa.timestamp('ms')),
                    ('duration_minutes', pa.float64()),
                    ('service_provider_id', pa.string()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key == 'observation':
                self.schemas[resource_key] = pa.schema([
                    ('observation_id', pa.string()),
                    ('patient_id', pa.string()),
                    ('encounter_id', pa.string()),
                    ('file_path', pa.string()),
                    ('status', pa.string()),
                    ('category', pa.string()),
                    ('code', pa.string()),
                    ('display', pa.string()),
                    ('code_system', pa.string()),
                    ('value_numeric', pa.float64()),
                    ('value_string', pa.string()),
                    ('unit', pa.string()),
                    ('effective_datetime', pa.timestamp('ms')),
                    ('issued', pa.timestamp('ms')),
                    ('interpretation', pa.string()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key == 'condition':
                self.schemas[resource_key] = pa.schema([
                    ('condition_id', pa.string()),
                    ('patient_id', pa.string()),
                    ('encounter_id', pa.string()),
                    ('file_path', pa.string()),
                    ('clinical_status', pa.string()),
                    ('verification_status', pa.string()),
                    ('code', pa.string()),
                    ('display', pa.string()),
                    ('code_system', pa.string()),
                    ('severity', pa.string()),
                    ('onset_datetime', pa.timestamp('ms')),
                    ('recorded_date', pa.timestamp('ms')),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key == 'procedure':
                self.schemas[resource_key] = pa.schema([
                    ('procedure_id', pa.string()),
                    ('patient_id', pa.string()),
                    ('encounter_id', pa.string()),
                    ('file_path', pa.string()),
                    ('status', pa.string()),
                    ('code', pa.string()),
                    ('display', pa.string()),
                    ('code_system', pa.string()),
                    ('performed_datetime', pa.timestamp('ms')),
                    ('performer_id', pa.string()),
                    ('outcome', pa.string()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key in ['medication', 'medicationrequest', 'medicationadministration', 'medicationdispense', 'medicationstatement']:
                self.schemas[resource_key] = pa.schema([
                    ('medication_id', pa.string()),
                    ('patient_id', pa.string()),
                    ('encounter_id', pa.string()),
                    ('file_path', pa.string()),
                    ('resource_type', pa.string()),
                    ('status', pa.string()),
                    ('intent', pa.string()),
                    ('code', pa.string()),
                    ('display', pa.string()),
                    ('code_system', pa.string()),
                    ('authored_on', pa.timestamp('ms')),
                    ('requester_id', pa.string()),
                    ('dosage_text', pa.string()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key in ['practitioner', 'practitionerrole']:
                self.schemas[resource_key] = pa.schema([
                    ('practitioner_id', pa.string()),
                    ('file_path', pa.string()),
                    ('resource_type', pa.string()),
                    ('family_name', pa.string()),
                    ('given_name', pa.string()),
                    ('qualifications', pa.string()),  # JSON array as string
                    ('specialties', pa.string()),     # JSON array as string
                    ('active', pa.bool_()),
                    ('organization_id', pa.string()),
                    ('created_at', pa.timestamp('ms'))
                ])
            
            elif resource_key in ['organization', 'organizationaffiliation']:
                self.schemas[resource_key] = pa.schema([
                    ('organization_id', pa.string()),
                    ('file_path', pa.string()),
                    ('resource_type', pa.string()),
                    ('name', pa.string()),
                    ('type', pa.string()),
                    ('identifier', pa.string()),
                    ('active', pa.bool_()),
                    ('address', pa.string()),  # JSON as string
                    ('telecom', pa.string()),  # JSON as string
                    ('created_at', pa.timestamp('ms'))
                ])
            
            # Generic schema for all other FHIR resource types
            else:
                self.schemas[resource_key] = pa.schema([
                    ('resource_id', pa.string()),
                    ('file_path', pa.string()),
                    ('resource_type', pa.string()),
                    ('status', pa.string()),
                    ('category', pa.string()),
                    ('code', pa.string()),
                    ('display', pa.string()),
                    ('code_system', pa.string()),
                    ('subject_id', pa.string()),        # Patient reference
                    ('encounter_id', pa.string()),      # Encounter reference  
                    ('practitioner_id', pa.string()),   # Practitioner reference
                    ('organization_id', pa.string()),   # Organization reference
                    ('effective_datetime', pa.timestamp('ms')),
                    ('authored_datetime', pa.timestamp('ms')),
                    ('recorded_datetime', pa.timestamp('ms')),
                    ('priority', pa.string()),
                    ('intent', pa.string()),
                    ('value_text', pa.string()),
                    ('value_numeric', pa.float64()),
                    ('value_boolean', pa.bool_()),  # Nullable boolean
                    ('identifier', pa.string()),
                    ('active', pa.bool_()),
                    ('properties', pa.string()),  # Full resource as JSON string
                    ('created_at', pa.timestamp('ms'))
                ])
        
        return self.schemas[resource_key]
    
    def add_records(self, resource_type: str, records: List[Dict[str, Any]]) -> int:
        """Add records to partitioned Parquet files."""
        if not records:
            return 0
        
        # Initialize partition if needed
        if resource_type not in self.current_partitions:
            self.current_partitions[resource_type] = []
        
        # Add records to current partition
        self.current_partitions[resource_type].extend(records)
        
        # Check if partition is full
        partitions_written = 0
        if len(self.current_partitions[resource_type]) >= self.partition_size:
            partitions_written = self._write_partition(resource_type)
        
        return partitions_written
    
    def _write_partition(self, resource_type: str) -> int:
        """Write current partition to Parquet file."""
        if resource_type not in self.current_partitions or not self.current_partitions[resource_type]:
            return 0
        
        try:
            # Create DataFrame from records
            df = pd.DataFrame(self.current_partitions[resource_type])
            
            # Convert to Arrow table with schema
            schema = self.get_schema(resource_type)
            
            # Align DataFrame columns with schema
            for field in schema:
                if field.name not in df.columns:
                    df[field.name] = None
            
            # Reorder columns to match schema
            df = df[[field.name for field in schema]]
            
            # Convert data types
            df = self._convert_data_types(df, resource_type)
            
            table = pa.Table.from_pandas(df, schema=schema)
            
            # Write partition file
            partition_num = self.partition_counters[resource_type]
            output_file = self.resource_dirs[resource_type] / f"partition_{partition_num:06d}.parquet"
            
            pq.write_table(table, output_file, compression='snappy')
            
            logger.info(f"Written {resource_type} partition {partition_num}: {len(df)} records -> {output_file}")
            
            # Clear current partition and increment counter
            self.current_partitions[resource_type] = []
            self.partition_counters[resource_type] += 1
            
            return 1
            
        except Exception as e:
            logger.error(f"Error writing {resource_type} partition: {e}")
            return 0
    
    def _convert_data_types(self, df: pd.DataFrame, resource_type: str) -> pd.DataFrame:
        """Convert DataFrame data types for Parquet compatibility."""
        
        # Convert timestamp columns with millisecond precision
        timestamp_cols = ['created_at', 'start_time', 'end_time', 'effective_datetime', 
                         'issued', 'onset_datetime', 'recorded_date', 'performed_datetime', 
                         'authored_on', 'authored_datetime', 'recorded_datetime']
        
        for col in timestamp_cols:
            if col in df.columns:
                # Convert to datetime first, handling various formats
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                # Then convert to millisecond precision to avoid casting errors
                df[col] = df[col].dt.floor('ms')
                # Handle any remaining NaT values
                df[col] = df[col].fillna(pd.Timestamp('1970-01-01', tz='UTC').floor('ms'))
        
        # Convert boolean columns with proper None handling
        bool_cols = ['active', 'deceased', 'value_boolean']
        for col in bool_cols:
            if col in df.columns:
                # Replace string 'None' with actual None first
                df[col] = df[col].replace('None', None)
                # Convert to boolean, keeping None as NaN
                df[col] = df[col].astype('boolean', errors='ignore')
        
        # Convert numeric columns with error handling
        numeric_cols = ['age', 'duration_minutes', 'value_numeric']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure string columns are strings
        string_cols = [col for col in df.columns if col not in timestamp_cols + bool_cols + numeric_cols]
        for col in string_cols:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', None)  # Replace 'nan' strings with None
        
        return df
    
    def flush_all_partitions(self) -> int:
        """Flush all remaining partitions to disk."""
        total_written = 0
        
        for resource_type in self.current_partitions:
            if self.current_partitions[resource_type]:
                total_written += self._write_partition(resource_type)
        
        return total_written
    
    def get_partition_summary(self) -> Dict[str, Any]:
        """Get summary of partitions created."""
        summary = {}
        
        for resource_type, count in self.partition_counters.items():
            resource_dir = self.resource_dirs[resource_type]
            files = list(resource_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in files)
            
            summary[resource_type] = {
                'partitions': count,
                'files': len(files),
                'total_size_mb': total_size / (1024**2),
                'avg_size_mb': (total_size / len(files) / (1024**2)) if files else 0
            }
        
        return summary


class LargeScaleFHIRProcessor:
    """Large-scale FHIR processor for 435GB+ datasets."""
    
    def __init__(self, 
                 fhir_data_dir: str,
                 output_dir: str,
                 max_memory_gb: float = 8.0,
                 batch_size: int = 50,
                 max_workers: int = 4,
                 enable_graph: bool = True):
        
        self.fhir_data_dir = Path(fhir_data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_graph = enable_graph and GRAPH_SUPPORT
        
        # Initialize components
        self.memory_manager = MemoryManager(max_memory_gb)
        self.parquet_writer = PartitionedParquetWriter(output_dir)
        self.stats = ProcessingStats()
        
        # Initialize hierarchical knowledge graph if enabled
        self.hierarchical_kg = None
        if self.enable_graph:
            self.hierarchical_kg = self._initialize_hierarchical_graph()
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        logger.info(f"Initialized Large-Scale FHIR Processor")
        logger.info(f"Source: {self.fhir_data_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Max memory: {max_memory_gb}GB")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max workers: {max_workers}")
        logger.info(f"Graph support: {'Enabled' if self.enable_graph else 'Disabled'}")
    
    def _initialize_hierarchical_graph(self) -> Optional[HierarchicalKnowledgeGraph]:
        """Initialize hierarchical knowledge graph with FHIR ontology structure."""
        if not GRAPH_SUPPORT:
            return None
            
        logger.info("Initializing hierarchical knowledge graph...")
        
        try:
            # Create hierarchical knowledge graph
            hierarchical_kg = HierarchicalKnowledgeGraph(
                name="ProductionFHIRKG",
                semantic_reasoning=True
            )
            
            # Create hierarchical levels based on FHIR resource relationships
            levels = [
                ('patients', 'Patient Registry', 0),
                ('encounters', 'Clinical Encounters', 1), 
                ('observations', 'Clinical Observations', 2),
                ('conditions', 'Diagnoses & Conditions', 3),
                ('procedures', 'Medical Procedures', 4),
                ('medications', 'Medication Management', 5),
                ('practitioners', 'Healthcare Providers', 6),
                ('organizations', 'Healthcare Organizations', 7),
                ('diagnostic_reports', 'Diagnostic Reports', 8),
                ('care_coordination', 'Care Plans & Coordination', 9),
                ('other_resources', 'Other FHIR Resources', 10)
            ]
            
            for level_name, description, order in levels:
                level_kg = KnowledgeGraph(name=f"FHIR_{level_name}")
                hierarchical_kg.add_level(level_name, level_kg, order, description)
                logger.info(f"Created graph level: {level_name} - {description}")
            
            logger.info(f"Hierarchical knowledge graph initialized with {len(levels)} levels")
            return hierarchical_kg
            
        except Exception as e:
            logger.error(f"Failed to initialize hierarchical knowledge graph: {e}")
            return None
    
    def discover_fhir_files(self) -> Dict[str, Any]:
        """Discover FHIR files and estimate processing requirements."""
        logger.info("Discovering FHIR files...")
        
        if not self.fhir_data_dir.exists():
            raise FileNotFoundError(f"FHIR data directory not found: {self.fhir_data_dir}")
        
        # Get all JSON files
        json_files = list(self.fhir_data_dir.glob("*.json"))
        total_files = len(json_files)
        
        # Calculate total size
        total_size_bytes = sum(f.stat().st_size for f in json_files[:1000])  # Sample first 1000
        avg_file_size = total_size_bytes / min(1000, total_files)
        estimated_total_size = avg_file_size * total_files
        
        discovery_info = {
            'total_files': total_files,
            'estimated_size_gb': estimated_total_size / (1024**3),
            'avg_file_size_mb': avg_file_size / (1024**2),
            'estimated_processing_time_hours': (total_files / 1000) * 0.5,  # Rough estimate
            'memory_requirement_gb': min(8.0, estimated_total_size / (1024**3) * 0.01)  # 1% of data size
        }
        
        logger.info(f"Discovery results:")
        logger.info(f"  Files: {discovery_info['total_files']:,}")
        logger.info(f"  Estimated size: {discovery_info['estimated_size_gb']:.1f} GB")
        logger.info(f"  Estimated processing time: {discovery_info['estimated_processing_time_hours']:.1f} hours")
        
        return discovery_info
    
    def _add_to_hierarchical_graph(self, resource_type: str, resource_data: Dict[str, Any]) -> None:
        """Add FHIR resource to appropriate level in hierarchical knowledge graph."""
        if not self.hierarchical_kg:
            return
        
        try:
            # Determine appropriate graph level based on resource type
            level_mapping = {
                'Patient': 'patients',
                'Encounter': 'encounters',
                'Observation': 'observations',
                'DiagnosticReport': 'diagnostic_reports',
                'Condition': 'conditions',
                'Procedure': 'procedures',
                'MedicationRequest': 'medications',
                'MedicationAdministration': 'medications',
                'MedicationDispense': 'medications',
                'Medication': 'medications',
                'Practitioner': 'practitioners',
                'PractitionerRole': 'practitioners',
                'Organization': 'organizations',
                'CarePlan': 'care_coordination',
                'CareTeam': 'care_coordination',
                'Goal': 'care_coordination'
            }
            
            # Map resource to appropriate level
            level_name = level_mapping.get(resource_type, 'other_resources')
            
            # Get resource ID for entity creation
            resource_id = resource_data.get('id', f"{resource_type}_{uuid.uuid4().hex[:8]}")
            entity_id = f"{resource_type}_{resource_id}"
            
            # Create entity with basic properties
            entity_properties = {
                'resource_type': resource_type,
                'fhir_id': resource_id,
                'status': resource_data.get('status', 'unknown'),
                'created_at': datetime.now().isoformat()
            }
            
            # Add type-specific properties
            if resource_type == 'Patient':
                entity_properties.update({
                    'gender': resource_data.get('gender'),
                    'birth_date': resource_data.get('birthDate'),
                    'active': resource_data.get('active', True)
                })
            elif resource_type == 'Observation':
                entity_properties.update({
                    'code': str(resource_data.get('code', {})),
                    'value': str(resource_data.get('valueQuantity') or resource_data.get('valueString', '')),
                    'category': str(resource_data.get('category', []))
                })
            elif resource_type == 'Condition':
                entity_properties.update({
                    'code': str(resource_data.get('code', {})),
                    'clinical_status': resource_data.get('clinicalStatus', {}).get('coding', [{}])[0].get('code'),
                    'verification_status': resource_data.get('verificationStatus', {}).get('coding', [{}])[0].get('code')
                })
            elif resource_type in ['MedicationRequest', 'MedicationAdministration', 'MedicationDispense']:
                entity_properties.update({
                    'medication': str(resource_data.get('medicationCodeableConcept') or resource_data.get('medicationReference', {})),
                    'dosage': str(resource_data.get('dosageInstruction', []))
                })
            
            # Add entity to appropriate level
            level_kg = self.hierarchical_kg.get_level(level_name)
            if level_kg:
                level_kg.add_entity(entity_id, resource_type, entity_properties)
                
                # Add relationships based on FHIR references
                self._add_fhir_relationships(level_kg, entity_id, resource_data, resource_type)
                
        except Exception as e:
            logger.warning(f"Failed to add {resource_type} to hierarchical graph: {e}")
    
    def _add_fhir_relationships(self, kg: KnowledgeGraph, entity_id: str, resource_data: Dict[str, Any], resource_type: str) -> None:
        """Add relationships between FHIR resources."""
        try:
            # Common FHIR reference patterns
            reference_fields = ['subject', 'patient', 'encounter', 'performer', 'requester', 'practitioner', 'organization']
            
            for field in reference_fields:
                if field in resource_data:
                    ref_data = resource_data[field]
                    if isinstance(ref_data, dict) and 'reference' in ref_data:
                        ref_id = ref_data['reference']
                        # Create relationship
                        relationship_id = f"{entity_id}_to_{ref_id}"
                        kg.add_relationship(
                            relationship_id,
                            entity_id,
                            ref_id,
                            f"references_{field}",
                            {'reference_type': field, 'created_at': datetime.now().isoformat()}
                        )
                        
        except Exception as e:
            logger.warning(f"Failed to add relationships for {entity_id}: {e}")
    
    def get_hierarchical_graph_summary(self) -> Dict[str, Any]:
        """Get summary of the hierarchical knowledge graph."""
        if not self.hierarchical_kg:
            return {'enabled': False, 'reason': 'Graph support disabled'}
        
        summary = {
            'enabled': True,
            'name': self.hierarchical_kg.name,
            'total_entities': self.hierarchical_kg.get_total_entities(),
            'total_relationships': self.hierarchical_kg.get_total_relationships(),
            'levels': self.hierarchical_kg.get_level_summary()
        }
        
        return summary
    
    def save_hierarchical_graph(self, output_path: str) -> None:
        """Save hierarchical knowledge graph summary to JSON file."""
        if not self.hierarchical_kg:
            logger.warning("No hierarchical knowledge graph to save")
            return
        
        summary = self.get_hierarchical_graph_summary()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Hierarchical knowledge graph summary saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save hierarchical knowledge graph: {e}")
    
    def process_file_batch(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process a batch of FHIR files."""
        batch_stats = ProcessingStats()
        batch_records = defaultdict(list)
        
        for file_path in file_paths:
            try:
                file_start_time = time.time()
                
                # Read and process file
                with open(file_path, 'r') as f:
                    bundle_data = json.load(f)
                
                file_size = file_path.stat().st_size
                
                # Extract records from bundle
                records = self._extract_records_from_bundle(bundle_data, str(file_path))
                
                # Accumulate records by type
                for resource_type, resource_records in records.items():
                    batch_records[resource_type].extend(resource_records)
                
                # Add to hierarchical knowledge graph if enabled
                if self.enable_graph and bundle_data.get('entry'):
                    for entry in bundle_data.get('entry', []):
                        if 'resource' in entry:
                            resource = entry['resource']
                            resource_type = resource.get('resourceType')
                            if resource_type:
                                self._add_to_hierarchical_graph(resource_type, resource)
                
                # Update stats
                batch_stats.files_processed += 1
                batch_stats.bytes_processed += file_size
                batch_stats.processing_time += time.time() - file_start_time
                
                # Count resources by type
                for resource_type, resource_records in records.items():
                    count = len(resource_records)
                    batch_stats.total_resources += count
                    
                    # Initialize resource_type_counts if needed
                    if batch_stats.resource_type_counts is None:
                        batch_stats.resource_type_counts = defaultdict(int)
                    batch_stats.resource_type_counts[resource_type] += count
                    
                    # Update specific counters for core resources
                    if resource_type == 'patient':
                        batch_stats.total_patients += count
                    elif resource_type == 'encounter':
                        batch_stats.total_encounters += count
                    elif resource_type == 'observation':
                        batch_stats.total_observations += count
                    elif resource_type == 'condition':
                        batch_stats.total_conditions += count
                    elif resource_type == 'procedure':
                        batch_stats.total_procedures += count
                    elif resource_type in ['medication', 'medicationrequest', 'medicationadministration', 'medicationdispense', 'medicationstatement']:
                        batch_stats.total_medications += count
                    elif resource_type in ['practitioner', 'practitionerrole']:
                        batch_stats.total_practitioners += count
                    elif resource_type in ['organization', 'organizationaffiliation']:
                        batch_stats.total_organizations += count
                    elif resource_type == 'diagnosticreport':
                        batch_stats.total_diagnostic_reports += count
                    elif resource_type in ['immunization', 'immunizationevaluation', 'immunizationrecommendation']:
                        batch_stats.total_immunizations += count
                    elif resource_type == 'allergyintolerance':
                        batch_stats.total_allergies += count
                    elif resource_type in ['careplan', 'careteam']:
                        batch_stats.total_care_plans += count
                    elif resource_type in ['appointment', 'appointmentresponse']:
                        batch_stats.total_appointments += count
                    elif resource_type in ['device', 'devicedefinition', 'devicerequest', 'deviceusage']:
                        batch_stats.total_devices += count
                    elif resource_type in ['specimen', 'specimendefinition']:
                        batch_stats.total_specimens += count
                    else:
                        batch_stats.total_other_resources += count
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                if batch_stats.errors is not None:
                    batch_stats.errors.append(error_msg)
        
        # Write batch records to Parquet
        partitions_written = 0
        for resource_type, records in batch_records.items():
            if records:
                partitions_written += self.parquet_writer.add_records(resource_type, records)
        
        batch_stats.partitions_created = partitions_written
        
        # Memory check
        memory_stats = self.memory_manager.get_memory_stats()
        batch_stats.memory_peak_mb = memory_stats['current_mb']
        
        if self.memory_manager.check_memory_pressure():
            logger.info("Memory pressure detected, running garbage collection")
            gc.collect()
        
        return {
            'stats': batch_stats,
            'memory_stats': memory_stats
        }
    
    def _extract_records_from_bundle(self, bundle_data: Dict[str, Any], file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract structured records from FHIR Bundle."""
        
        if bundle_data.get('resourceType') != 'Bundle':
            return {}
        
        records = defaultdict(list)
        current_time = datetime.now()
        # Use millisecond-precision timestamp for consistency
        current_timestamp = pd.Timestamp(current_time).floor('ms')
        
        # Group resources by type and collect references
        resources_by_type = defaultdict(list)
        resource_refs = {}
        
        for entry in bundle_data.get('entry', []):
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            resource_id = resource.get('id')
            
            if resource_type and resource_id:
                resources_by_type[resource_type].append(resource)
                resource_refs[resource_id] = resource
        
        # Process each resource type found in the bundle
        for resource_type, resource_list in resources_by_type.items():
            resource_key = resource_type.lower()
            
            for resource in resource_list:
                # Use specialized extractors for core resources
                if resource_type == 'Patient':
                    record = self._extract_patient_record(resource, file_path, current_timestamp)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type == 'Encounter':
                    record = self._extract_encounter_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type == 'Observation':
                    record = self._extract_observation_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type == 'Condition':
                    record = self._extract_condition_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type == 'Procedure':
                    record = self._extract_procedure_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type in ['Medication', 'MedicationRequest', 'MedicationAdministration', 'MedicationDispense', 'MedicationStatement']:
                    record = self._extract_medication_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type in ['Practitioner', 'PractitionerRole']:
                    record = self._extract_practitioner_record(resource, file_path, current_timestamp)
                    if record:
                        records[resource_key].append(record)
                
                elif resource_type in ['Organization', 'OrganizationAffiliation']:
                    record = self._extract_organization_record(resource, file_path, current_timestamp)
                    if record:
                        records[resource_key].append(record)
                
                # Generic extractor for all other FHIR resource types
                else:
                    record = self._extract_generic_fhir_record(resource, file_path, current_timestamp, resource_refs)
                    if record:
                        records[resource_key].append(record)
        
        return dict(records)
    
    def _extract_patient_record(self, resource: Dict[str, Any], file_path: str, created_at: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Extract patient record for Parquet storage."""
        try:
            record = {
                'patient_id': resource.get('id'),
                'file_path': file_path,
                'gender': resource.get('gender'),
                'birth_date': resource.get('birthDate'),
                'age': None,
                'family_name': None,
                'given_name': None,
                'race': None,
                'ethnicity': None,
                'marital_status': None,
                'city': None,
                'state': None,
                'country': None,
                'active': resource.get('active', True),
                'deceased': resource.get('deceasedBoolean', False),
                'created_at': created_at
            }
            
            # Calculate age
            if record['birth_date']:
                try:
                    birth_year = int(record['birth_date'][:4])
                    record['age'] = datetime.now().year - birth_year
                except:
                    pass
            
            # Extract name
            if 'name' in resource and resource['name']:
                name = resource['name'][0]
                record['family_name'] = name.get('family')
                if 'given' in name:
                    record['given_name'] = ' '.join(name['given'])
            
            # Extract marital status
            if 'maritalStatus' in resource:
                marital = resource['maritalStatus']
                if 'coding' in marital and marital['coding']:
                    record['marital_status'] = marital['coding'][0].get('display')
            
            # Extract address
            if 'address' in resource and resource['address']:
                addr = resource['address'][0]
                record['city'] = addr.get('city')
                record['state'] = addr.get('state')
                record['country'] = addr.get('country')
            
            # Extract extensions (race, ethnicity)
            if 'extension' in resource:
                for ext in resource['extension']:
                    url = ext.get('url', '')
                    if 'us-core-race' in url:
                        record['race'] = self._extract_extension_value(ext)
                    elif 'us-core-ethnicity' in url:
                        record['ethnicity'] = self._extract_extension_value(ext)
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting patient record: {e}")
            return None
    
    def _extract_encounter_record(self, resource: Dict[str, Any], file_path: str, 
                                created_at: pd.Timestamp, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract encounter record for Parquet storage."""
        try:
            record = {
                'encounter_id': resource.get('id'),
                'patient_id': self._extract_reference_id(resource.get('subject', {})),
                'file_path': file_path,
                'status': resource.get('status'),
                'class_code': None,
                'class_display': None,
                'start_time': None,
                'end_time': None,
                'duration_minutes': None,
                'service_provider_id': self._extract_reference_id(resource.get('serviceProvider', {})),
                'created_at': created_at
            }
            
            # Extract class
            if 'class' in resource:
                class_data = resource['class']
                record['class_code'] = class_data.get('code')
                record['class_display'] = class_data.get('display')
            
            # Extract period
            if 'period' in resource:
                period = resource['period']
                if 'start' in period:
                    record['start_time'] = pd.to_datetime(period['start'], errors='coerce')
                if 'end' in period:
                    record['end_time'] = pd.to_datetime(period['end'], errors='coerce')
                
                # Calculate duration
                if record['start_time'] and record['end_time']:
                    try:
                        duration = record['end_time'] - record['start_time']
                        record['duration_minutes'] = duration.total_seconds() / 60
                    except:
                        pass
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting encounter record: {e}")
            return None
    
    def _extract_observation_record(self, resource: Dict[str, Any], file_path: str,
                                  created_at: datetime, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract observation record for Parquet storage."""
        try:
            record = {
                'observation_id': resource.get('id'),
                'patient_id': self._extract_reference_id(resource.get('subject', {})),
                'encounter_id': self._extract_reference_id(resource.get('encounter', {})),
                'file_path': file_path,
                'status': resource.get('status'),
                'category': None,
                'code': None,
                'display': None,
                'code_system': None,
                'value_numeric': None,
                'value_string': None,
                'unit': None,
                'effective_datetime': None,
                'issued': None,
                'interpretation': None,
                'created_at': created_at
            }
            
            # Extract category
            if 'category' in resource and resource['category']:
                cat = resource['category'][0]
                if 'coding' in cat and cat['coding']:
                    record['category'] = cat['coding'][0].get('display')
            
            # Extract code
            if 'code' in resource:
                code = resource['code']
                if 'coding' in code and code['coding']:
                    coding = code['coding'][0]
                    record['code'] = coding.get('code')
                    record['display'] = coding.get('display')
                    record['code_system'] = coding.get('system')
            
            # Extract value
            if 'valueQuantity' in resource:
                qty = resource['valueQuantity']
                record['value_numeric'] = qty.get('value')
                record['unit'] = qty.get('unit')
            elif 'valueString' in resource:
                record['value_string'] = resource['valueString']
            elif 'valueCodeableConcept' in resource:
                concept = resource['valueCodeableConcept']
                if 'coding' in concept and concept['coding']:
                    record['value_string'] = concept['coding'][0].get('display')
            
            # Extract timing
            if 'effectiveDateTime' in resource:
                record['effective_datetime'] = pd.to_datetime(resource['effectiveDateTime'], errors='coerce')
            if 'issued' in resource:
                record['issued'] = pd.to_datetime(resource['issued'], errors='coerce')
            
            # Extract interpretation
            if 'interpretation' in resource and resource['interpretation']:
                interp = resource['interpretation'][0]
                if 'coding' in interp and interp['coding']:
                    record['interpretation'] = interp['coding'][0].get('display')
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting observation record: {e}")
            return None
    
    def _extract_condition_record(self, resource: Dict[str, Any], file_path: str,
                                created_at: datetime, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract condition record for Parquet storage."""
        try:
            record = {
                'condition_id': resource.get('id'),
                'patient_id': self._extract_reference_id(resource.get('subject', {})),
                'encounter_id': self._extract_reference_id(resource.get('encounter', {})),
                'file_path': file_path,
                'clinical_status': None,
                'verification_status': None,
                'code': None,
                'display': None,
                'code_system': None,
                'severity': None,
                'onset_datetime': None,
                'recorded_date': None,
                'created_at': created_at
            }
            
            # Extract clinical status
            if 'clinicalStatus' in resource:
                status = resource['clinicalStatus']
                if 'coding' in status and status['coding']:
                    record['clinical_status'] = status['coding'][0].get('code')
            
            # Extract verification status
            if 'verificationStatus' in resource:
                status = resource['verificationStatus']
                if 'coding' in status and status['coding']:
                    record['verification_status'] = status['coding'][0].get('code')
            
            # Extract code
            if 'code' in resource:
                code = resource['code']
                if 'coding' in code and code['coding']:
                    coding = code['coding'][0]
                    record['code'] = coding.get('code')
                    record['display'] = coding.get('display')
                    record['code_system'] = coding.get('system')
            
            # Extract severity
            if 'severity' in resource:
                severity = resource['severity']
                if 'coding' in severity and severity['coding']:
                    record['severity'] = severity['coding'][0].get('display')
            
            # Extract timing
            if 'onsetDateTime' in resource:
                record['onset_datetime'] = pd.to_datetime(resource['onsetDateTime'], errors='coerce')
            if 'recordedDate' in resource:
                record['recorded_date'] = pd.to_datetime(resource['recordedDate'], errors='coerce')
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting condition record: {e}")
            return None
    
    def _extract_procedure_record(self, resource: Dict[str, Any], file_path: str,
                                created_at: datetime, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract procedure record for Parquet storage."""
        try:
            record = {
                'procedure_id': resource.get('id'),
                'patient_id': self._extract_reference_id(resource.get('subject', {})),
                'encounter_id': self._extract_reference_id(resource.get('encounter', {})),
                'file_path': file_path,
                'status': resource.get('status'),
                'code': None,
                'display': None,
                'code_system': None,
                'performed_datetime': None,
                'performer_id': None,
                'outcome': None,
                'created_at': created_at
            }
            
            # Extract code
            if 'code' in resource:
                code = resource['code']
                if 'coding' in code and code['coding']:
                    coding = code['coding'][0]
                    record['code'] = coding.get('code')
                    record['display'] = coding.get('display')
                    record['code_system'] = coding.get('system')
            
            # Extract performed time
            if 'performedDateTime' in resource:
                record['performed_datetime'] = pd.to_datetime(resource['performedDateTime'], errors='coerce')
            elif 'performedPeriod' in resource:
                period = resource['performedPeriod']
                if 'start' in period:
                    record['performed_datetime'] = pd.to_datetime(period['start'], errors='coerce')
            
            # Extract performer
            if 'performer' in resource and resource['performer']:
                performer = resource['performer'][0]
                record['performer_id'] = self._extract_reference_id(performer.get('actor', {}))
            
            # Extract outcome
            if 'outcome' in resource:
                outcome = resource['outcome']
                if 'coding' in outcome and outcome['coding']:
                    record['outcome'] = outcome['coding'][0].get('display')
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting procedure record: {e}")
            return None
    
    def _extract_medication_record(self, resource: Dict[str, Any], file_path: str,
                                 created_at: datetime, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract medication record for Parquet storage."""
        try:
            record = {
                'medication_id': resource.get('id'),
                'patient_id': self._extract_reference_id(resource.get('subject', {})),
                'encounter_id': self._extract_reference_id(resource.get('encounter', {})),
                'file_path': file_path,
                'resource_type': resource.get('resourceType'),
                'status': resource.get('status'),
                'intent': resource.get('intent'),
                'code': None,
                'display': None,
                'code_system': None,
                'authored_on': None,
                'requester_id': None,
                'dosage_text': None,
                'created_at': created_at
            }
            
            # Extract medication code
            code_field = 'medicationCodeableConcept' if 'medicationCodeableConcept' in resource else 'code'
            if code_field in resource:
                code = resource[code_field]
                if 'coding' in code and code['coding']:
                    coding = code['coding'][0]
                    record['code'] = coding.get('code')
                    record['display'] = coding.get('display')
                    record['code_system'] = coding.get('system')
            
            # Extract authored on (for MedicationRequest)
            if 'authoredOn' in resource:
                record['authored_on'] = pd.to_datetime(resource['authoredOn'], errors='coerce')
            
            # Extract requester
            if 'requester' in resource:
                record['requester_id'] = self._extract_reference_id(resource['requester'])
            
            # Extract dosage
            if 'dosageInstruction' in resource and resource['dosageInstruction']:
                dosage = resource['dosageInstruction'][0]
                record['dosage_text'] = dosage.get('text')
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting medication record: {e}")
            return None
    
    def _extract_practitioner_record(self, resource: Dict[str, Any], file_path: str, created_at: datetime) -> Optional[Dict[str, Any]]:
        """Extract practitioner record for Parquet storage."""
        try:
            record = {
                'practitioner_id': resource.get('id'),
                'file_path': file_path,
                'resource_type': resource.get('resourceType'),
                'family_name': None,
                'given_name': None,
                'qualifications': None,
                'specialties': None,
                'active': resource.get('active', True),
                'organization_id': None,
                'created_at': created_at
            }
            
            # Extract name
            if 'name' in resource and resource['name']:
                name = resource['name'][0]
                record['family_name'] = name.get('family')
                if 'given' in name:
                    record['given_name'] = ' '.join(name['given'])
            
            # Extract qualifications
            if 'qualification' in resource:
                qualifications = []
                for qual in resource['qualification']:
                    if 'code' in qual and 'coding' in qual['code']:
                        coding = qual['code']['coding'][0]
                        qualifications.append(coding.get('display', coding.get('code', '')))
                record['qualifications'] = json.dumps(qualifications) if qualifications else None
            
            # Extract specialties (for PractitionerRole)
            if 'specialty' in resource:
                specialties = []
                for spec in resource['specialty']:
                    if 'coding' in spec:
                        coding = spec['coding'][0]
                        specialties.append(coding.get('display', coding.get('code', '')))
                record['specialties'] = json.dumps(specialties) if specialties else None
            
            # Extract organization reference (for PractitionerRole)
            if 'organization' in resource:
                record['organization_id'] = self._extract_reference_id(resource['organization'])
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting practitioner record: {e}")
            return None
    
    def _extract_organization_record(self, resource: Dict[str, Any], file_path: str, created_at: datetime) -> Optional[Dict[str, Any]]:
        """Extract organization record for Parquet storage."""
        try:
            record = {
                'organization_id': resource.get('id'),
                'file_path': file_path,
                'resource_type': resource.get('resourceType'),
                'name': resource.get('name'),
                'type': None,
                'identifier': None,
                'active': resource.get('active', True),
                'address': None,
                'telecom': None,
                'created_at': created_at
            }
            
            # Extract type
            if 'type' in resource and resource['type']:
                types = []
                for type_item in resource['type']:
                    if 'coding' in type_item:
                        coding = type_item['coding'][0]
                        types.append(coding.get('display', coding.get('code', '')))
                record['type'] = ', '.join(types) if types else None
            
            # Extract identifier
            if 'identifier' in resource and resource['identifier']:
                identifier = resource['identifier'][0]
                record['identifier'] = f"{identifier.get('system', '')}|{identifier.get('value', '')}"
            
            # Extract address
            if 'address' in resource:
                record['address'] = json.dumps(resource['address'])
            
            # Extract telecom
            if 'telecom' in resource:
                record['telecom'] = json.dumps(resource['telecom'])
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting organization record: {e}")
            return None
    
    def _extract_generic_fhir_record(self, resource: Dict[str, Any], file_path: str, 
                                   created_at: datetime, resource_refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract generic FHIR record for any resource type."""
        try:
            record = {
                'resource_id': resource.get('id'),
                'file_path': file_path,
                'resource_type': resource.get('resourceType'),
                'status': resource.get('status'),
                'category': None,
                'code': None,
                'display': None,
                'code_system': None,
                'subject_id': None,
                'encounter_id': None,
                'practitioner_id': None,
                'organization_id': None,
                'effective_datetime': None,
                'authored_datetime': None,
                'recorded_datetime': None,
                'priority': resource.get('priority'),
                'intent': resource.get('intent'),
                'value_text': None,
                'value_numeric': None,
                'value_boolean': None,
                'identifier': None,
                'active': resource.get('active'),
                'properties': json.dumps(resource),  # Store full resource
                'created_at': created_at
            }
            
            # Extract common references
            if 'subject' in resource:
                record['subject_id'] = self._extract_reference_id(resource['subject'])
            elif 'patient' in resource:
                record['subject_id'] = self._extract_reference_id(resource['patient'])
            
            if 'encounter' in resource:
                record['encounter_id'] = self._extract_reference_id(resource['encounter'])
            
            if 'requester' in resource:
                record['practitioner_id'] = self._extract_reference_id(resource['requester'])
            elif 'performer' in resource and isinstance(resource['performer'], list) and resource['performer']:
                record['practitioner_id'] = self._extract_reference_id(resource['performer'][0].get('actor', {}))
            
            if 'organization' in resource:
                record['organization_id'] = self._extract_reference_id(resource['organization'])
            
            # Extract category
            if 'category' in resource:
                if isinstance(resource['category'], list) and resource['category']:
                    cat = resource['category'][0]
                    if 'coding' in cat and cat['coding']:
                        record['category'] = cat['coding'][0].get('display', cat['coding'][0].get('code'))
                elif isinstance(resource['category'], dict):
                    if 'coding' in resource['category'] and resource['category']['coding']:
                        record['category'] = resource['category']['coding'][0].get('display', resource['category']['coding'][0].get('code'))
            
            # Extract code
            if 'code' in resource:
                code = resource['code']
                if 'coding' in code and code['coding']:
                    coding = code['coding'][0]
                    record['code'] = coding.get('code')
                    record['display'] = coding.get('display')
                    record['code_system'] = coding.get('system')
            
            # Extract timing fields with proper datetime handling
            time_fields = [
                ('effectiveDateTime', 'effective_datetime'),
                ('effectivePeriod', 'effective_datetime'),
                ('authoredOn', 'authored_datetime'),
                ('recordedDate', 'recorded_datetime'),
                ('performedDateTime', 'effective_datetime'),
                ('occurredDateTime', 'effective_datetime'),
                ('issued', 'recorded_datetime'),
                ('created', 'recorded_datetime'),
                ('date', 'recorded_datetime'),
                ('timestamp', 'recorded_datetime')
            ]
            
            for fhir_field, record_field in time_fields:
                if fhir_field in resource:
                    if fhir_field.endswith('Period') and isinstance(resource[fhir_field], dict):
                        # Extract start time from period
                        period = resource[fhir_field]
                        if 'start' in period:
                            dt = pd.to_datetime(period['start'], errors='coerce')
                            if pd.notna(dt):
                                record[record_field] = dt.floor('ms')  # Ensure millisecond precision
                    else:
                        # Handle string datetime values
                        dt_value = resource[fhir_field]
                        if isinstance(dt_value, str):
                            dt = pd.to_datetime(dt_value, errors='coerce')
                            if pd.notna(dt):
                                record[record_field] = dt.floor('ms')  # Ensure millisecond precision
                        elif dt_value is not None:
                            # Try to convert other types
                            try:
                                dt = pd.to_datetime(str(dt_value), errors='coerce')
                                if pd.notna(dt):
                                    record[record_field] = dt.floor('ms')
                            except:
                                # If conversion fails, set to None
                                record[record_field] = None
                    break  # Use first found time field
            
            # Extract value fields
            value_fields = [
                'valueString', 'valueBoolean', 'valueQuantity', 'valueCodeableConcept',
                'valueInteger', 'valueDecimal', 'valueDateTime', 'valueTime',
                'result', 'outcome', 'conclusion'
            ]
            
            for value_field in value_fields:
                if value_field in resource:
                    value = resource[value_field]
                    if value_field == 'valueQuantity' and isinstance(value, dict):
                        record['value_numeric'] = value.get('value')
                        record['value_text'] = f"{value.get('value', '')} {value.get('unit', '')}"
                    elif value_field == 'valueBoolean':
                        record['value_boolean'] = value
                    elif value_field == 'valueCodeableConcept' and isinstance(value, dict):
                        if 'coding' in value and value['coding']:
                            record['value_text'] = value['coding'][0].get('display', value['coding'][0].get('code'))
                    elif isinstance(value, (str, int, float)):
                        if isinstance(value, (int, float)):
                            record['value_numeric'] = value
                        record['value_text'] = str(value)
                    break
            
            # Extract identifier
            if 'identifier' in resource and resource['identifier']:
                identifier = resource['identifier'][0] if isinstance(resource['identifier'], list) else resource['identifier']
                if isinstance(identifier, dict):
                    record['identifier'] = f"{identifier.get('system', '')}|{identifier.get('value', '')}"
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting generic FHIR record for {resource.get('resourceType')}: {e}")
            return None
    
    def _extract_reference_id(self, reference: Dict[str, Any]) -> Optional[str]:
        """Extract ID from FHIR reference."""
        if not reference or 'reference' not in reference:
            return None
        
        ref = reference['reference']
        if '/' in ref:
            return ref.split('/')[-1]
        return ref
    
    def _extract_extension_value(self, extension: Dict[str, Any]) -> Optional[str]:
        """Extract value from FHIR extension."""
        if 'extension' in extension:
            for ext in extension['extension']:
                if 'valueCoding' in ext:
                    return ext['valueCoding'].get('display')
        return None
    
    def process_all_files(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Process all FHIR files with memory management and progress tracking."""
        logger.info("Starting large-scale FHIR processing...")
        
        start_time = time.time()
        
        # Get file list
        json_files = list(self.fhir_data_dir.glob("*.json"))
        
        if max_files:
            json_files = json_files[:max_files]
            logger.info(f"Processing limited set: {max_files:,} files")
        
        total_files = len(json_files)
        logger.info(f"Total files to process: {total_files:,}")
        
        # Process in batches
        batches = [json_files[i:i + self.batch_size] for i in range(0, len(json_files), self.batch_size)]
        total_batches = len(batches)
        
        logger.info(f"Processing {total_batches:,} batches of {self.batch_size} files each")
        
        # Process batches with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_file_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            completed_batches = 0
            
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                
                try:
                    result = future.result()
                    batch_stats = result['stats']
                    memory_stats = result['memory_stats']
                    
                    # Update global stats thread-safely
                    with self.stats_lock:
                        self.stats.files_processed += batch_stats.files_processed
                        self.stats.total_resources += batch_stats.total_resources
                        self.stats.total_patients += batch_stats.total_patients
                        self.stats.total_encounters += batch_stats.total_encounters
                        self.stats.total_observations += batch_stats.total_observations
                        self.stats.total_conditions += batch_stats.total_conditions
                        self.stats.total_procedures += batch_stats.total_procedures
                        self.stats.total_medications += batch_stats.total_medications
                        self.stats.total_practitioners += batch_stats.total_practitioners
                        self.stats.total_organizations += batch_stats.total_organizations
                        self.stats.total_diagnostic_reports += batch_stats.total_diagnostic_reports
                        self.stats.total_immunizations += batch_stats.total_immunizations
                        self.stats.total_allergies += batch_stats.total_allergies
                        self.stats.total_care_plans += batch_stats.total_care_plans
                        self.stats.total_appointments += batch_stats.total_appointments
                        self.stats.total_devices += batch_stats.total_devices
                        self.stats.total_specimens += batch_stats.total_specimens
                        self.stats.total_other_resources += batch_stats.total_other_resources
                        self.stats.bytes_processed += batch_stats.bytes_processed
                        self.stats.processing_time += batch_stats.processing_time
                        self.stats.memory_peak_mb = max(self.stats.memory_peak_mb, batch_stats.memory_peak_mb)
                        self.stats.partitions_created += batch_stats.partitions_created
                        
                        # Merge resource type counts
                        if self.stats.resource_type_counts is None:
                            self.stats.resource_type_counts = defaultdict(int)
                        if batch_stats.resource_type_counts is not None:
                            for resource_type, count in batch_stats.resource_type_counts.items():
                                self.stats.resource_type_counts[resource_type] += count
                        
                        if self.stats.errors is not None and batch_stats.errors is not None:
                            self.stats.errors.extend(batch_stats.errors)
                    
                    completed_batches += 1
                    
                    # Progress reporting
                    if completed_batches % 10 == 0 or completed_batches == total_batches:
                        elapsed = time.time() - start_time
                        rate = self.stats.files_processed / elapsed if elapsed > 0 else 0
                        eta = (total_files - self.stats.files_processed) / rate if rate > 0 else 0
                        
                        logger.info(f"Progress: {completed_batches}/{total_batches} batches "
                                  f"({self.stats.files_processed:,}/{total_files:,} files, "
                                  f"{self.stats.files_processed/total_files*100:.1f}%) - "
                                  f"Rate: {rate:.1f} files/sec - "
                                  f"Memory: {memory_stats['current_gb']:.1f}GB - "
                                  f"ETA: {eta/3600:.1f}h")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    with self.stats_lock:
                        if self.stats.errors is not None:
                            self.stats.errors.append(f"Batch {batch_num}: {e}")
        
        # Flush remaining partitions
        logger.info("Flushing remaining partitions...")
        final_partitions = self.parquet_writer.flush_all_partitions()
        self.stats.partitions_created += final_partitions
        
        self.stats.processing_time = time.time() - start_time
        
        # Get partition summary
        partition_summary = self.parquet_writer.get_partition_summary()
        
        # Get hierarchical knowledge graph summary  
        graph_summary = self.get_hierarchical_graph_summary()
        
        # Save knowledge graph summary if enabled
        if self.enable_graph:
            graph_output_path = self.output_dir / "hierarchical_knowledge_graph_summary.json"
            self.save_hierarchical_graph(str(graph_output_path))
        
        logger.info("=" * 80)
        logger.info("LARGE-SCALE FHIR PROCESSING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Files processed: {self.stats.files_processed:,}")
        logger.info(f"Total resources: {self.stats.total_resources:,}")
        logger.info(f"Processing time: {self.stats.processing_time:.2f} seconds ({self.stats.processing_time/3600:.2f} hours)")
        logger.info(f"Processing rate: {self.stats.files_processed/self.stats.processing_time:.1f} files/sec")
        logger.info(f"Data throughput: {self.stats.bytes_processed/(1024**3)/self.stats.processing_time:.2f} GB/sec")
        logger.info(f"Peak memory usage: {self.stats.memory_peak_mb:.1f} MB")
        logger.info(f"Partitions created: {self.stats.partitions_created}")
        
        # Knowledge graph summary log
        if self.enable_graph and graph_summary.get('enabled'):
            logger.info("=" * 40)
            logger.info("HIERARCHICAL KNOWLEDGE GRAPH SUMMARY")
            logger.info("=" * 40)
            logger.info(f"Total entities: {graph_summary['total_entities']:,}")
            logger.info(f"Total relationships: {graph_summary['total_relationships']:,}")
            logger.info(f"Graph levels: {len(graph_summary['levels'])}")
            for level_name, level_info in graph_summary['levels'].items():
                logger.info(f"  {level_name}: {level_info['entities']} entities, {level_info['relationships']} relationships")
        
        return {
            'status': 'completed',
            'stats': self.stats,
            'partition_summary': partition_summary,
            'hierarchical_knowledge_graph': graph_summary,
            'final_memory_stats': self.memory_manager.get_memory_stats()
        }


def main():
    """Main function for large-scale FHIR processing."""
    
    print("🚀 LARGE-SCALE FHIR DATA PROCESSOR (435GB+ Dataset)")
    print("=" * 80)
    
    # Configuration
    fhir_data_dir = "/home/amansingh/dev/andola/healthcare/synthea/output/fhir"
    output_dir = "/media/amansingh/data/fhir_test"
    max_memory_gb = 8.0
    batch_size = 50
    max_workers = 4
    
    # For initial testing, limit files
    max_files = 10000  # Process 10K files initially, remove limit for full processing
    
    try:
        # Initialize processor
        processor = LargeScaleFHIRProcessor(
            fhir_data_dir=fhir_data_dir,
            output_dir=output_dir,
            max_memory_gb=max_memory_gb,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # Discovery phase
        print("\n🔍 PHASE 1: DATASET DISCOVERY")
        print("-" * 60)
        discovery = processor.discover_fhir_files()
        print(f"📁 Total files: {discovery['total_files']:,}")
        print(f"💾 Estimated size: {discovery['estimated_size_gb']:.1f} GB")
        print(f"⏱️ Estimated processing time: {discovery['estimated_processing_time_hours']:.1f} hours")
        
        # Processing phase
        print(f"\n⚡ PHASE 2: LARGE-SCALE PROCESSING")
        print("-" * 60)
        if max_files:
            print(f"🧪 Processing sample: {max_files:,} files (remove max_files for full dataset)")
        else:
            print(f"🏭 Processing full dataset: {discovery['total_files']:,} files")
        
        result = processor.process_all_files(max_files=max_files)
        
        # Results phase
        print(f"\n📊 PHASE 3: RESULTS SUMMARY")
        print("-" * 60)
        stats = result['stats']
        partition_summary = result['partition_summary']
        
        print(f"✅ Processing completed successfully!")
        print(f"📁 Files processed: {stats.files_processed:,}")
        print(f"🔬 Total resources: {stats.total_resources:,}")
        print(f"👥 Patients: {stats.total_patients:,}")
        print(f"🏥 Encounters: {stats.total_encounters:,}")
        print(f"📊 Observations: {stats.total_observations:,}")
        print(f"🩺 Conditions: {stats.total_conditions:,}")
        print(f"⚕️ Procedures: {stats.total_procedures:,}")
        print(f"💊 Medications: {stats.total_medications:,}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"⏱️ Total time: {stats.processing_time:.2f}s ({stats.processing_time/3600:.2f}h)")
        print(f"🚀 Processing rate: {stats.files_processed/stats.processing_time:.1f} files/sec")
        print(f"📈 Throughput: {stats.bytes_processed/(1024**3)/stats.processing_time:.2f} GB/sec")
        print(f"💾 Peak memory: {stats.memory_peak_mb:.1f} MB")
        
        print(f"\n📦 Parquet Output Summary:")
        total_partitions = sum(info['partitions'] for info in partition_summary.values())
        total_size_gb = sum(info['total_size_mb'] for info in partition_summary.values()) / 1024
        print(f"📂 Total partitions: {total_partitions}")
        print(f"💾 Total output size: {total_size_gb:.2f} GB")
        print(f"📍 Output location: {output_dir}")
        
        for resource_type, info in partition_summary.items():
            print(f"  {resource_type}: {info['partitions']} partitions, {info['total_size_mb']:.1f} MB")
        
        # Display hierarchical knowledge graph summary
        graph_summary = result.get('hierarchical_knowledge_graph', {})
        if graph_summary.get('enabled'):
            print(f"\n🕸️ Hierarchical Knowledge Graph Summary:")
            print(f"🔢 Total entities: {graph_summary['total_entities']:,}")
            print(f"🔗 Total relationships: {graph_summary['total_relationships']:,}")
            print(f"📊 Graph levels: {len(graph_summary['levels'])}")
            for level_name, level_info in graph_summary['levels'].items():
                print(f"  {level_name}: {level_info['entities']} entities, {level_info['relationships']} relationships")
            graph_file = f"{output_dir}/hierarchical_knowledge_graph_summary.json"
            print(f"💾 Graph summary saved to: {graph_file}")
        else:
            reason = graph_summary.get('reason', 'Unknown')
            print(f"\n🕸️ Hierarchical Knowledge Graph: Disabled ({reason})")
        
        if stats.errors:
            print(f"\n⚠️ Errors encountered: {len(stats.errors)}")
            for error in stats.errors[:5]:  # Show first 5 errors
                print(f"  {error}")
        
        print("\n" + "=" * 80)
        print("✅ LARGE-SCALE FHIR PROCESSING DEMONSTRATION COMPLETE!")
        print("🎯 Ready for 435GB full dataset processing!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()