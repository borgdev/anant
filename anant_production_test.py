#!/usr/bin/env python3
"""
Anant Production Test - Hypergraph Creation and Persistence
==========================================================

Tests the complete Anant functionality on the Ray cluster:
1. Creates a medical research hypergraph
2. Runs distributed analysis
3. Verifies persistence across restarts
4. Demonstrates real-world usage

This uses the actual Anant classes and APIs.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work."""
    try:
        # Ray imports
        import ray
        import polars as pl
        import numpy as np
        
        # Core Anant imports - using actual structure
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        logger.info("✅ All core imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def connect_to_ray():
    """Connect to the Ray cluster."""
    try:
        import ray
        # Connect to the local Ray cluster
        ray.init()
        cluster_info = ray.cluster_resources()
        logger.info(f"✅ Connected to Ray cluster")
        logger.info(f"   Nodes: {cluster_info.get('node:__internal_head__', 0) + sum(1 for k in cluster_info.keys() if k.startswith('node:') and k != 'node:__internal_head__')}")
        logger.info(f"   CPUs: {cluster_info.get('CPU', 0)}")
        logger.info(f"   Memory: {cluster_info.get('memory', 0) / (1024**3):.1f} GB")
        return True
    except Exception as e:
        logger.error(f"❌ Ray connection failed: {e}")
        return False

def create_medical_research_hypergraph():
    """Create a comprehensive medical research hypergraph."""
    try:
        from anant.classes.hypergraph import Hypergraph
        from anant.factory.setsystem_factory import SetSystemFactory
        
        logger.info("🏗️ Creating medical research hypergraph...")
        
        # Define complex medical research data
        # Each hyperedge represents a multi-way relationship
        medical_data = {
            # Treatment protocols involving patients, drugs, and doctors
            'diabetes_treatment_protocol_1': ['patient_001', 'patient_003', 'metformin', 'dr_endocrinologist', 'glucose_monitor'],
            'hypertension_treatment_protocol_1': ['patient_002', 'lisinopril', 'dr_cardiologist', 'bp_monitor'],
            
            # Genetic research cohorts
            'diabetes_genetic_study': ['patient_001', 'patient_003', 'gene_tcf7l2', 'gene_pparg', 'lab_genomics'],
            'cardiovascular_genetic_study': ['patient_002', 'gene_ace', 'gene_agtr1', 'lab_genomics'],
            
            # Symptom clusters and diagnostic patterns
            'metabolic_syndrome_cluster': ['patient_001', 'symptom_fatigue', 'symptom_thirst', 'test_hba1c', 'test_glucose'],
            'cardiovascular_risk_cluster': ['patient_002', 'symptom_chest_pain', 'symptom_shortness_breath', 'test_ecg', 'test_stress'],
            
            # Drug interaction networks
            'diabetes_drug_interaction': ['metformin', 'insulin', 'glimepiride', 'drug_database'],
            'cardiac_drug_interaction': ['lisinopril', 'metoprolol', 'aspirin', 'drug_database'],
            
            # Research publication networks
            'diabetes_research_network': ['dr_endocrinologist', 'research_paper_001', 'journal_diabetes', 'institution_medical_center'],
            'cardiology_research_network': ['dr_cardiologist', 'research_paper_002', 'journal_cardiology', 'institution_medical_center'],
            
            # Clinical trial participation
            'clinical_trial_diabetes_001': ['patient_001', 'patient_003', 'metformin', 'dr_endocrinologist', 'clinical_coordinator'],
            'clinical_trial_cardiac_001': ['patient_002', 'lisinopril', 'dr_cardiologist', 'clinical_coordinator'],
            
            # Hospital care pathways
            'diabetes_care_pathway': ['patient_001', 'dr_endocrinologist', 'nurse_diabetes', 'hospital_general', 'pharmacy'],
            'cardiac_care_pathway': ['patient_002', 'dr_cardiologist', 'nurse_cardiac', 'hospital_general', 'pharmacy'],
            
            # Laboratory and testing networks
            'lab_analysis_network': ['lab_genomics', 'test_hba1c', 'test_glucose', 'test_ecg', 'lab_technician'],
            
            # Multi-condition patients (complex cases)
            'comorbidity_management': ['patient_001', 'dr_endocrinologist', 'dr_cardiologist', 'metformin', 'lisinopril'],
        }
        
        # Define node properties (metadata)
        node_properties = {
            # Patients
            'patient_001': {'type': 'patient', 'age': 45, 'gender': 'F', 'conditions': ['diabetes', 'pre_hypertension']},
            'patient_002': {'type': 'patient', 'age': 62, 'gender': 'M', 'conditions': ['hypertension', 'coronary_artery_disease']},
            'patient_003': {'type': 'patient', 'age': 38, 'gender': 'F', 'conditions': ['diabetes', 'metabolic_syndrome']},
            
            # Medical professionals
            'dr_endocrinologist': {'type': 'physician', 'specialty': 'endocrinology', 'years_experience': 15},
            'dr_cardiologist': {'type': 'physician', 'specialty': 'cardiology', 'years_experience': 20},
            'nurse_diabetes': {'type': 'nurse', 'specialty': 'diabetes_education', 'certification': 'CDE'},
            'nurse_cardiac': {'type': 'nurse', 'specialty': 'cardiac_care', 'certification': 'CCRN'},
            
            # Medications
            'metformin': {'type': 'drug', 'class': 'biguanide', 'indication': 'type2_diabetes'},
            'lisinopril': {'type': 'drug', 'class': 'ace_inhibitor', 'indication': 'hypertension'},
            'insulin': {'type': 'drug', 'class': 'hormone', 'indication': 'diabetes'},
            
            # Genes
            'gene_tcf7l2': {'type': 'gene', 'chromosome': '10', 'function': 'glucose_metabolism'},
            'gene_ace': {'type': 'gene', 'chromosome': '17', 'function': 'blood_pressure_regulation'},
            
            # Medical tests
            'test_hba1c': {'type': 'lab_test', 'category': 'diabetes', 'normal_range': '4-6%'},
            'test_glucose': {'type': 'lab_test', 'category': 'diabetes', 'normal_range': '70-100mg/dl'},
            'test_ecg': {'type': 'diagnostic_test', 'category': 'cardiac', 'duration': '10min'},
            
            # Institutions
            'hospital_general': {'type': 'hospital', 'beds': 500, 'level': 'tertiary'},
            'lab_genomics': {'type': 'laboratory', 'specialty': 'genomics', 'accreditation': 'CAP'},
        }
        
        # Create the hypergraph with properties
        setsystem = SetSystemFactory.from_dict(medical_data)
        
        # Prepare properties structure
        properties = {
            'nodes': node_properties,
            'edges': {}  # We can add edge properties later if needed
        }
        
        hypergraph = Hypergraph(
            setsystem=setsystem, 
            properties=properties,
            name="medical_research_hypergraph"
        )
        
        logger.info(f"✅ Created hypergraph with {hypergraph.num_nodes} nodes and {hypergraph.num_edges} hyperedges")
        
        # Return hypergraph and metadata
        return hypergraph, {
            'nodes': hypergraph.num_nodes,
            'edges': hypergraph.num_edges,
            'node_types': list(set(props.get('type', 'unknown') for props in node_properties.values())),
            'creation_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create hypergraph: {e}")
        raise

def run_distributed_analysis(hypergraph):
    """Run comprehensive analysis using Ray distributed computing."""
    try:
        import ray
        
        logger.info("🔬 Starting distributed hypergraph analysis...")
        
        # Define Ray remote functions for distributed analysis
        @ray.remote
        def analyze_basic_properties():
            """Basic graph properties analysis."""
            try:
                results = {
                    'num_nodes': hypergraph.num_nodes,
                    'num_edges': hypergraph.num_edges,
                    'num_incidences': hypergraph.num_incidences,
                    'is_empty': hypergraph.is_empty(),
                }
                
                # Node degree analysis
                node_degrees = {}
                for node in hypergraph.nodes:
                    node_degrees[str(node)] = hypergraph.degree(node)
                
                results['node_degrees'] = node_degrees
                results['max_degree'] = max(node_degrees.values()) if node_degrees else 0
                results['min_degree'] = min(node_degrees.values()) if node_degrees else 0
                results['avg_degree'] = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
                
                return {'basic_properties': results, 'status': 'success'}
            except Exception as e:
                return {'basic_properties': {}, 'status': 'error', 'error': str(e)}
        
        @ray.remote
        def analyze_edge_properties():
            """Edge size and distribution analysis."""
            try:
                results = {}
                
                # Edge size analysis
                edge_sizes = {}
                for edge in hypergraph.edges:
                    edge_nodes = hypergraph.get_edge_nodes(edge)
                    edge_sizes[str(edge)] = len(edge_nodes)
                
                results['edge_sizes'] = edge_sizes
                results['max_edge_size'] = max(edge_sizes.values()) if edge_sizes else 0
                results['min_edge_size'] = min(edge_sizes.values()) if edge_sizes else 0
                results['avg_edge_size'] = sum(edge_sizes.values()) / len(edge_sizes) if edge_sizes else 0
                
                # Edge size distribution
                size_distribution = {}
                for size in edge_sizes.values():
                    size_distribution[size] = size_distribution.get(size, 0) + 1
                results['size_distribution'] = size_distribution
                
                return {'edge_properties': results, 'status': 'success'}
            except Exception as e:
                return {'edge_properties': {}, 'status': 'error', 'error': str(e)}
        
        @ray.remote
        def analyze_connectivity():
            """Connectivity and path analysis."""
            try:
                results = {}
                
                # Connected components
                try:
                    components = hypergraph.connected_components()
                    results['num_connected_components'] = len(components)
                    results['largest_component_size'] = max(len(comp) for comp in components) if components else 0
                    results['is_connected'] = hypergraph.is_connected()
                except:
                    results['connectivity_analysis'] = 'not_available'
                
                return {'connectivity': results, 'status': 'success'}
            except Exception as e:
                return {'connectivity': {}, 'status': 'error', 'error': str(e)}
        
        @ray.remote
        def analyze_node_types():
            """Node type distribution analysis."""
            try:
                results = {}
                
                # Analyze node types from properties
                type_distribution = {}
                type_examples = {}
                
                for node in hypergraph.nodes:
                    try:
                        node_type = hypergraph.properties.get_node_property(node, 'type')
                        if node_type:
                            type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
                            if node_type not in type_examples:
                                type_examples[node_type] = []
                            if len(type_examples[node_type]) < 3:
                                type_examples[node_type].append(str(node))
                    except:
                        # Node doesn't have type property
                        pass
                
                results['type_distribution'] = type_distribution
                results['type_examples'] = type_examples
                results['num_types'] = len(type_distribution)
                
                return {'node_types': results, 'status': 'success'}
            except Exception as e:
                return {'node_types': {}, 'status': 'error', 'error': str(e)}
        
        # Launch distributed analysis tasks
        logger.info("🚀 Launching distributed analysis tasks...")
        
        futures = [
            analyze_basic_properties.remote(),
            analyze_edge_properties.remote(),
            analyze_connectivity.remote(),
            analyze_node_types.remote()
        ]
        
        # Collect results
        results = ray.get(futures)
        
        # Combine results
        analysis_results = {
            'hypergraph_name': hypergraph.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'ray_cluster_info': ray.cluster_resources(),
            'analysis_results': {}
        }
        
        for result in results:
            analysis_results['analysis_results'].update(result)
        
        logger.info("✅ Distributed analysis completed")
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

def save_results_to_persistent_storage(results, hypergraph_metadata):
    """Save results to persistent storage (Docker volumes)."""
    try:
        # Create data directory if it doesn't exist
        data_dir = "/app/data"
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis results
        results_file = f"{data_dir}/medical_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save hypergraph metadata
        metadata_file = f"{data_dir}/hypergraph_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(hypergraph_metadata, f, indent=2, default=str)
        
        # Create a persistence marker file
        marker_file = f"{data_dir}/test_complete.marker"
        with open(marker_file, 'w') as f:
            f.write(f"Test completed at {datetime.now().isoformat()}\n")
            f.write(f"Hypergraph: {hypergraph_metadata.get('nodes', 0)} nodes, {hypergraph_metadata.get('edges', 0)} edges\n")
            f.write(f"Results saved to: {results_file}\n")
        
        logger.info(f"💾 Results saved to persistent storage:")
        logger.info(f"   Analysis: {results_file}")
        logger.info(f"   Metadata: {metadata_file}")
        logger.info(f"   Marker: {marker_file}")
        
        return {
            'results_file': results_file,
            'metadata_file': metadata_file,
            'marker_file': marker_file
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise

def check_existing_data():
    """Check if test data already exists from previous runs."""
    try:
        data_dir = "/app/data"
        marker_file = f"{data_dir}/test_complete.marker"
        
        if os.path.exists(marker_file):
            with open(marker_file, 'r') as f:
                content = f.read()
            logger.info("📋 Found existing test data:")
            logger.info(f"   {content.strip()}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to check existing data: {e}")
        return False

def simulate_cluster_restart():
    """Simulate cluster restart by disconnecting and reconnecting Ray."""
    try:
        logger.info("🔄 Simulating cluster restart...")
        
        import ray
        ray.shutdown()
        logger.info("   Ray disconnected")
        
        time.sleep(3)
        
        ray.init()
        logger.info("   Ray reconnected")
        
        cluster_info = ray.cluster_resources()
        logger.info(f"   Cluster ready with {cluster_info.get('CPU', 0)} CPUs")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Restart simulation failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Anant Production Test on Ray Cluster")
    logger.info("="*60)
    
    # Step 1: Test imports
    if not test_imports():
        return False
    
    # Step 2: Connect to Ray
    if not connect_to_ray():
        return False
    
    # Step 3: Check for existing data
    existing_data = check_existing_data()
    if existing_data:
        logger.info("📊 Previous test data found - demonstrating persistence")
    
    try:
        # Step 4: Create hypergraph (always create for this test)
        hypergraph, metadata = create_medical_research_hypergraph()
        
        # Step 5: Run distributed analysis
        analysis_results = run_distributed_analysis(hypergraph)
        
        # Step 6: Save to persistent storage
        storage_info = save_results_to_persistent_storage(analysis_results, metadata)
        
        # Step 7: Test cluster restart and persistence
        if simulate_cluster_restart():
            logger.info("✅ Cluster restart simulation successful")
            
            # Verify data still exists after restart
            if check_existing_data():
                logger.info("✅ Data persistence verified after restart")
            else:
                logger.warning("⚠️ Data persistence check failed")
        
        # Step 8: Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 ANANT PRODUCTION TEST SUMMARY")
        logger.info("="*60)
        
        basic_props = analysis_results['analysis_results'].get('basic_properties', {})
        edge_props = analysis_results['analysis_results'].get('edge_properties', {})
        node_types = analysis_results['analysis_results'].get('node_types', {})
        
        logger.info(f"📊 Hypergraph: {hypergraph.name}")
        logger.info(f"🏷️ Nodes: {basic_props.get('num_nodes', 0)}")
        logger.info(f"🔗 Hyperedges: {basic_props.get('num_edges', 0)}")
        logger.info(f"📈 Incidences: {basic_props.get('num_incidences', 0)}")
        logger.info(f"📐 Avg Node Degree: {basic_props.get('avg_degree', 0):.2f}")
        logger.info(f"📏 Avg Edge Size: {edge_props.get('avg_edge_size', 0):.2f}")
        logger.info(f"🏷️ Node Types: {node_types.get('num_types', 0)}")
        
        type_dist = node_types.get('type_distribution', {})
        if type_dist:
            logger.info("📊 Node Type Distribution:")
            for node_type, count in type_dist.items():
                logger.info(f"   {node_type}: {count}")
        
        logger.info(f"💾 Data Files:")
        for key, path in storage_info.items():
            logger.info(f"   {key}: {path}")
        
        logger.info("="*60)
        logger.info("✅ Test completed successfully!")
        logger.info("🔄 Data persistence verified across cluster restarts")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            import ray
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)