"""
FHIR Unified Knowledge Graph Test Suite
======================================

Comprehensive test suite for validating all FHIR unified graph functionalities:
- Ontology loading and integration
- Data loading and processing  
- Unified graph construction
- Ontology-data mappings
- Persistence operations
- End-to-end workflows
- Performance and validation checks
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# ANANT imports
from anant.kg import HierarchicalKnowledgeGraph

# Local FHIR imports
try:
    from .unified_graph_builder import FHIRUnifiedGraphBuilder, build_fhir_unified_graph
    from .ontology_loader import FHIROntologyLoader
    from .data_loader import FHIRDataLoader
    from .graph_persistence import save_fhir_graph, load_fhir_graph
except ImportError:
    # Fallback for direct execution
    from unified_graph_builder import FHIRUnifiedGraphBuilder, build_fhir_unified_graph
    from ontology_loader import FHIROntologyLoader
    from data_loader import FHIRDataLoader
    from graph_persistence import save_fhir_graph, load_fhir_graph

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFHIRUnifiedGraphBuilder(unittest.TestCase):
    """Test suite for the FHIR Unified Graph Builder."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.schema_dir = cls.test_dir / "schema"
        cls.data_dir = cls.test_dir / "data" / "output" / "fhir"
        
        # Create directories
        cls.schema_dir.mkdir(parents=True, exist_ok=True)
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test schema files
        cls._create_test_schema_files()
        
        # Create test data files
        cls._create_test_data_files()
        
        logger.info(f"Test environment created at: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        logger.info("Test environment cleaned up")
    
    @classmethod
    def _create_test_schema_files(cls):
        """Create minimal test schema files."""
        # Create minimal RIM ontology
        rim_content = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix fhir: <http://hl7.org/fhir/> .

fhir:Resource a owl:Class ;
    rdfs:label "Resource" ;
    rdfs:comment "Base Resource class" .

fhir:DomainResource a owl:Class ;
    rdfs:subClassOf fhir:Resource ;
    rdfs:label "DomainResource" ;
    rdfs:comment "Domain-specific resource" .

fhir:Patient a owl:Class ;
    rdfs:subClassOf fhir:DomainResource ;
    rdfs:label "Patient" ;
    rdfs:comment "Patient resource type" .

fhir:Practitioner a owl:Class ;
    rdfs:subClassOf fhir:DomainResource ;
    rdfs:label "Practitioner" ;
    rdfs:comment "Practitioner resource type" .
"""
        
        # Create minimal FHIR ontology
        fhir_content = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix fhir: <http://hl7.org/fhir/> .

fhir:Observation a owl:Class ;
    rdfs:subClassOf fhir:DomainResource ;
    rdfs:label "Observation" ;
    rdfs:comment "Observation resource type" .

fhir:Condition a owl:Class ;
    rdfs:subClassOf fhir:DomainResource ;
    rdfs:label "Condition" ;
    rdfs:comment "Condition resource type" .

fhir:name a rdf:Property ;
    rdfs:label "name" ;
    rdfs:comment "Name property" .

fhir:identifier a rdf:Property ;
    rdfs:label "identifier" ;
    rdfs:comment "Identifier property" .
"""
        
        # Create minimal W5 ontology
        w5_content = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix w5: <http://hl7.org/fhir/w5#> .

w5:who a owl:Class ;
    rdfs:label "Who" ;
    rdfs:comment "Who dimension" .

w5:what a owl:Class ;
    rdfs:label "What" ;
    rdfs:comment "What dimension" .

w5:when a owl:Class ;
    rdfs:label "When" ;
    rdfs:comment "When dimension" .
"""
        
        # Write test schema files
        (cls.schema_dir / "rim.ttl").write_text(rim_content)
        (cls.schema_dir / "fhir.ttl").write_text(fhir_content)
        (cls.schema_dir / "w5.ttl").write_text(w5_content)
    
    @classmethod
    def _create_test_data_files(cls):
        """Create test FHIR data files."""
        # Test Patient Bundle
        patient_bundle = {
            "resourceType": "Bundle",
            "id": "test-patients",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1",
                        "name": [
                            {
                                "family": "Doe",
                                "given": ["John"]
                            }
                        ],
                        "gender": "male",
                        "birthDate": "1980-01-01"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-2",
                        "name": [
                            {
                                "family": "Smith",
                                "given": ["Jane"]
                            }
                        ],
                        "gender": "female",
                        "birthDate": "1985-05-15"
                    }
                }
            ]
        }
        
        # Test Practitioner Bundle
        practitioner_bundle = {
            "resourceType": "Bundle",
            "id": "test-practitioners",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": "practitioner-1",
                        "name": [
                            {
                                "family": "Johnson",
                                "given": ["Dr. Robert"],
                                "prefix": ["Dr."]
                            }
                        ],
                        "qualification": [
                            {
                                "code": {
                                    "coding": [
                                        {
                                            "system": "http://snomed.info/sct",
                                            "code": "309343006",
                                            "display": "Physician"
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        # Test Observation Bundle
        observation_bundle = {
            "resourceType": "Bundle",
            "id": "test-observations",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "observation-1",
                        "status": "final",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "8480-6",
                                    "display": "Systolic blood pressure"
                                }
                            ]
                        },
                        "subject": {
                            "reference": "Patient/patient-1"
                        },
                        "valueQuantity": {
                            "value": 120,
                            "unit": "mmHg"
                        }
                    }
                }
            ]
        }
        
        # Write test data files
        with open(cls.data_dir / "patients.json", 'w') as f:
            json.dump(patient_bundle, f, indent=2)
        
        with open(cls.data_dir / "practitioners.json", 'w') as f:
            json.dump(practitioner_bundle, f, indent=2)
        
        with open(cls.data_dir / "observations.json", 'w') as f:
            json.dump(observation_bundle, f, indent=2)
    
    def setUp(self):
        """Set up for individual tests."""
        self.builder = FHIRUnifiedGraphBuilder(
            schema_dir=str(self.schema_dir),
            data_dir=str(self.data_dir),
            graph_name="Test_FHIR_Graph"
        )
    
    def test_01_initialization(self):
        """Test proper initialization of unified graph builder."""
        self.assertIsNotNone(self.builder)
        self.assertEqual(self.builder.graph_name, "Test_FHIR_Graph")
        self.assertIsInstance(self.builder.unified_hkg, HierarchicalKnowledgeGraph)
        self.assertEqual(self.builder.unified_hkg.name, "Test_FHIR_Graph")
        
        logger.info("✓ Initialization test passed")
    
    def test_02_unified_structure_creation(self):
        """Test creation of unified hierarchical structure."""
        results = self.builder._create_unified_structure()
        
        self.assertEqual(results['status'], 'success')
        self.assertEqual(len(results['levels_created']), 8)
        
        # Verify specific levels are created
        expected_levels = [
            'meta_ontology', 'core_ontology', 'valuesets_ontology',
            'patients', 'practitioners', 'organizations', 
            'clinical_data', 'care_coordination'
        ]
        
        for level_id in expected_levels:
            self.assertIn(level_id, results['levels_created'])
            # Check if level exists by checking if it's in levels dict
            self.assertIn(level_id, self.builder.unified_hkg.levels)
        
        logger.info("✓ Unified structure creation test passed")
    
    def test_03_ontology_loading(self):
        """Test FHIR ontology loading."""
        # First create structure
        self.builder._create_unified_structure()
        
        # Test ontology loading
        results = self.builder._load_ontologies()
        
        self.assertEqual(results['status'], 'success')
        self.assertGreater(len(results['ontologies_loaded']), 0)
        
        # Verify ontology files were loaded
        expected_files = ['rim.ttl', 'fhir.ttl', 'w5.ttl']
        for expected_file in expected_files:
            self.assertTrue(any(expected_file in loaded for loaded in results['ontologies_loaded']))
        
        logger.info("✓ Ontology loading test passed")
    
    def test_04_data_loading(self):
        """Test FHIR data loading."""
        # Setup structure and ontologies
        self.builder._create_unified_structure()
        self.builder._load_ontologies()
        
        # Test data loading
        results = self.builder._load_data_instances(max_files=None)
        
        self.assertEqual(results['status'], 'success')
        self.assertGreater(results['files_processed'], 0)
        self.assertGreater(results['resources_loaded'], 0)
        
        # Verify resource types
        expected_types = ['Patient', 'Practitioner', 'Observation']
        for expected_type in expected_types:
            self.assertIn(expected_type, results['resource_types'])
        
        logger.info("✓ Data loading test passed")
    
    def test_05_ontology_data_mapping(self):
        """Test creation of ontology-data mappings."""
        # Setup graph with structure, ontologies, and data
        self.builder._create_unified_structure()
        self.builder._load_ontologies()
        self.builder._load_data_instances()
        
        # Test mapping creation
        results = self.builder._create_ontology_data_mappings()
        
        self.assertEqual(results['status'], 'success')
        self.assertGreaterEqual(results['mappings_created'], 0)
        
        # Check mapping types
        if results['mappings_created'] > 0:
            self.assertIn('type_based', results['mapping_types'])
        
        logger.info("✓ Ontology-data mapping test passed")
    
    def test_06_validation(self):
        """Test validation of unified graph."""
        # Build complete graph
        self.builder._create_unified_structure()
        self.builder._load_ontologies()
        self.builder._load_data_instances()
        self.builder._create_ontology_data_mappings()
        
        # Test validation
        results = self.builder._validate_ontology_data_consistency()
        
        self.assertIn(results['status'], ['success', 'warnings'])
        self.assertIn('validation_checks', results)
        
        # Check specific validation components
        checks = results['validation_checks']
        self.assertIn('resource_type_coverage', checks)
        self.assertIn('relationship_consistency', checks)
        self.assertIn('hierarchy_integrity', checks)
        
        logger.info("✓ Validation test passed")
    
    def test_07_complete_build_workflow(self):
        """Test complete unified graph build workflow."""
        results = self.builder.build_unified_graph(max_data_files=None, validate_mappings=True)
        
        self.assertIn(results['status'], ['success', 'completed_with_errors'])
        self.assertIn('phases', results)
        self.assertIn('statistics', results)
        
        # Verify all phases completed
        expected_phases = [
            'structure_creation', 'ontology_loading', 'data_loading',
            'mapping_creation', 'validation', 'finalization'
        ]
        
        for phase in expected_phases:
            self.assertIn(phase, results['phases'])
        
        # Check statistics
        stats = results['statistics']
        self.assertGreater(stats['total_nodes'], 0)
        self.assertGreater(stats['total_levels'], 0)
        
        logger.info("✓ Complete build workflow test passed")
    
    def test_08_graph_persistence(self):
        """Test saving and loading of unified graph."""
        # Build unified graph
        build_results = self.builder.build_unified_graph(max_data_files=None)
        self.assertIn(build_results['status'], ['success', 'completed_with_errors'])
        
        # Test saving
        output_dir = self.test_dir / "saved_graphs"
        save_results = self.builder.save_unified_graph(output_dir)
        
        self.assertEqual(save_results['status'], 'success')
        self.assertTrue(output_dir.exists())
        
        # Test loading
        loaded_hkg, load_results = load_fhir_graph(output_dir)
        
        self.assertEqual(load_results['status'], 'success')
        self.assertIsInstance(loaded_hkg, HierarchicalKnowledgeGraph)
        
        # Verify loaded graph has same structure
        original_stats = self.builder.unified_hkg.get_hierarchy_statistics()
        loaded_stats = loaded_hkg.get_hierarchy_statistics()
        
        self.assertEqual(original_stats['total_levels'], loaded_stats['total_levels'])
        
        logger.info("✓ Graph persistence test passed")
    
    def test_09_convenience_function(self):
        """Test convenience function for building unified graph."""
        hkg, results = build_fhir_unified_graph(
            schema_dir=str(self.schema_dir),
            data_dir=str(self.data_dir),
            max_data_files=2,
            graph_name="Convenience_Test_Graph"
        )
        
        self.assertIsInstance(hkg, HierarchicalKnowledgeGraph)
        self.assertEqual(hkg.name, "Convenience_Test_Graph")
        self.assertIn(results['status'], ['success', 'completed_with_errors'])
        
        logger.info("✓ Convenience function test passed")
    
    def test_10_resource_type_mapping(self):
        """Test resource type mapping to unified levels."""
        # Test the resource type mapping logic
        test_cases = [
            ('Patient', 'patients'),
            ('Practitioner', 'practitioners'),
            ('Organization', 'organizations'),
            ('Observation', 'clinical_data'),
            ('Condition', 'clinical_data'),
            ('CarePlan', 'care_coordination'),
            ('UnknownType', 'clinical_data')  # Default case
        ]
        
        for resource_type, expected_level in test_cases:
            actual_level = self.builder._get_unified_level_for_resource_type(resource_type)
            self.assertEqual(actual_level, expected_level)
        
        logger.info("✓ Resource type mapping test passed")
    
    def test_11_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with non-existent directories
        bad_builder = FHIRUnifiedGraphBuilder(
            schema_dir="/non/existent/schema",
            data_dir="/non/existent/data"
        )
        
        results = bad_builder.build_unified_graph()
        self.assertIn(results['status'], ['failed', 'completed_with_errors'])
        self.assertGreater(len(results['errors']), 0)
        
        logger.info("✓ Error handling test passed")


class TestFHIROntologyLoader(unittest.TestCase):
    """Test suite for FHIR Ontology Loader."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.schema_dir = self.test_dir / "schema"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal test schema
        test_ontology = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

<http://example.org/TestClass> a owl:Class ;
    rdfs:label "Test Class" ;
    rdfs:comment "A test class" .
"""
        
        (self.schema_dir / "test.ttl").write_text(test_ontology)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_ontology_loader_initialization(self):
        """Test ontology loader initialization."""
        loader = FHIROntologyLoader(str(self.schema_dir))
        self.assertEqual(str(loader.schema_dir), str(self.schema_dir))
        self.assertEqual(len(loader.graphs), 0)
        
        logger.info("✓ Ontology loader initialization test passed")
    
    def test_ontology_file_loading(self):
        """Test loading ontology files."""
        loader = FHIROntologyLoader(str(self.schema_dir))
        results = loader.load_ontology_files()
        
        # Should handle the case gracefully even if rdflib is not available
        self.assertIn('loaded_files', results)
        
        logger.info("✓ Ontology file loading test passed")


class TestFHIRDataLoader(unittest.TestCase):
    """Test suite for FHIR Data Loader."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data file
        test_bundle = {
            "resourceType": "Bundle",
            "id": "test-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient",
                        "name": [{"family": "Test", "given": ["Patient"]}]
                    }
                }
            ]
        }
        
        with open(self.data_dir / "test.json", 'w') as f:
            json.dump(test_bundle, f)
        
        # Create test hierarchical knowledge graph
        self.test_hkg = HierarchicalKnowledgeGraph(name="Test_HKG")
        self.test_hkg.create_level("patients", "Patients", "Patient data", 0)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        loader = FHIRDataLoader(str(self.data_dir), self.test_hkg)
        self.assertEqual(str(loader.data_dir), str(self.data_dir))
        self.assertEqual(loader.hkg, self.test_hkg)
        
        logger.info("✓ Data loader initialization test passed")
    
    def test_data_file_loading(self):
        """Test loading data files."""
        loader = FHIRDataLoader(str(self.data_dir), self.test_hkg)
        results = loader.load_fhir_data_files(max_files=1)
        
        self.assertIn('loaded_files', results)
        self.assertIn('total_resources', results)
        self.assertGreaterEqual(results['total_resources'], 0)
        
        logger.info("✓ Data file loading test passed")


def run_fhir_tests():
    """Run all FHIR tests with detailed reporting."""
    print("=" * 80)
    print("FHIR Unified Knowledge Graph Test Suite")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFHIRUnifiedGraphBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestFHIROntologyLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestFHIRDataLoader))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_fhir_tests()
    exit(0 if success else 1)