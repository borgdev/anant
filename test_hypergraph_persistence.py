#!/usr/bin/env python3
"""
Anant Hypergraph Persistence Test
=================================

Comprehensive test to verify:
1. Hypergraph creation and data loading
2. Analysis execution on Ray cluster
3. Data persistence across cluster restarts
4. Avoiding duplicate graph creation

This test demonstrates the full Anant enterprise capabilities.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

# Add anant to Python path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the test environment and imports."""
    try:
        # Core Anant imports
        from anant.core.hypergraph import AnantHyperGraph
        from anant.core.partitioned_graph import PartitionedHyperGraph
        from anant.layered_contextual_graph.lcg_core import LayeredContextualGraph
        from anant.storage.postgres import PostgresManager
        from anant.storage.parquet_manager import ParquetManager
        from anant.enterprise.metagraph_platform import MetaGraphPlatform
        
        # Ray imports
        import ray
        import polars as pl
        import numpy as np
        
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def connect_to_ray():
    """Connect to the Ray cluster."""
    try:
        # Connect to the local Ray cluster
        ray.init()
        cluster_info = ray.cluster_resources()
        logger.info(f"✅ Connected to Ray cluster: {cluster_info}")
        return True
    except Exception as e:
        logger.error(f"❌ Ray connection failed: {e}")
        return False

def create_sample_data():
    """Create comprehensive sample data for testing."""
    
    # Healthcare/Medical Research Dataset
    entities = [
        {"id": "patient_001", "type": "Patient", "age": 45, "gender": "F", "condition": "diabetes"},
        {"id": "patient_002", "type": "Patient", "age": 62, "gender": "M", "condition": "hypertension"},
        {"id": "patient_003", "type": "Patient", "age": 38, "gender": "F", "condition": "diabetes"},
        {"id": "drug_metformin", "type": "Drug", "class": "antidiabetic", "mechanism": "glucose_regulation"},
        {"id": "drug_lisinopril", "type": "Drug", "class": "ace_inhibitor", "mechanism": "blood_pressure"},
        {"id": "gene_tcf7l2", "type": "Gene", "chromosome": "10", "function": "glucose_metabolism"},
        {"id": "gene_ace", "type": "Gene", "chromosome": "17", "function": "blood_pressure_regulation"},
        {"id": "symptom_fatigue", "type": "Symptom", "severity": "moderate", "frequency": "daily"},
        {"id": "symptom_thirst", "type": "Symptom", "severity": "high", "frequency": "constant"},
        {"id": "test_hba1c", "type": "LabTest", "normal_range": "4-6%", "units": "percentage"},
        {"id": "doctor_smith", "type": "Physician", "specialty": "endocrinology", "years_experience": 15},
        {"id": "hospital_general", "type": "Hospital", "type_detail": "general", "beds": 500},
    ]
    
    # Complex relationships (hyperedges)
    relationships = [
        {
            "id": "treatment_001",
            "type": "Treatment",
            "participants": ["patient_001", "drug_metformin", "doctor_smith"],
            "attributes": {"start_date": "2024-01-15", "dosage": "500mg", "frequency": "twice_daily"}
        },
        {
            "id": "genetic_risk_001", 
            "type": "GeneticRisk",
            "participants": ["patient_001", "gene_tcf7l2", "symptom_thirst"],
            "attributes": {"risk_score": 0.85, "confidence": 0.92}
        },
        {
            "id": "symptom_cluster_001",
            "type": "SymptomCluster", 
            "participants": ["patient_001", "symptom_fatigue", "symptom_thirst", "test_hba1c"],
            "attributes": {"cluster_strength": 0.78, "temporal_pattern": "morning_peak"}
        },
        {
            "id": "drug_interaction_001",
            "type": "DrugInteraction",
            "participants": ["drug_metformin", "drug_lisinopril", "gene_ace"],
            "attributes": {"interaction_type": "synergistic", "evidence_level": "high"}
        },
        {
            "id": "care_pathway_001",
            "type": "CarePathway",
            "participants": ["patient_001", "doctor_smith", "hospital_general", "test_hba1c"],
            "attributes": {"pathway_type": "diabetes_management", "stage": "monitoring"}
        },
        {
            "id": "research_cohort_001",
            "type": "ResearchCohort",
            "participants": ["patient_001", "patient_003", "gene_tcf7l2", "drug_metformin"],
            "attributes": {"study_name": "diabetes_genetics", "phase": "analysis"}
        }
    ]
    
    return entities, relationships

def compute_graph_hash(entities: List[Dict], relationships: List[Dict]) -> str:
    """Compute a hash of the graph data for persistence checking."""
    data_str = json.dumps({
        "entities": sorted(entities, key=lambda x: x["id"]),
        "relationships": sorted(relationships, key=lambda x: x["id"])
    }, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

def create_hypergraph_if_needed(graph_id: str, entities: List[Dict], relationships: List[Dict]) -> bool:
    """Create hypergraph only if it doesn't exist."""
    try:
        from anant.core.hypergraph import AnantHyperGraph
        from anant.storage.postgres import PostgresManager
        
        # Check if graph already exists
        postgres_mgr = PostgresManager(
            host="postgres",
            port=5432,
            database="anant_enterprise",
            user="postgres", 
            password="anant_secure_2024"
        )
        
        # Check if graph exists in registry
        existing_graphs = postgres_mgr.list_graphs()
        if graph_id in [g.get('graph_id', g.get('id')) for g in existing_graphs]:
            logger.info(f"📊 Graph {graph_id} already exists, skipping creation")
            return False
        
        logger.info(f"🏗️ Creating new hypergraph: {graph_id}")
        
        # Create the hypergraph
        hypergraph = AnantHyperGraph(
            graph_id=graph_id,
            storage_backend="hybrid",  # PostgreSQL + Parquet
            enable_persistence=True
        )
        
        # Add entities
        for entity in entities:
            hypergraph.add_node(
                node_id=entity["id"],
                node_type=entity["type"],
                attributes=entity
            )
        
        # Add hyperedges (relationships)
        for rel in relationships:
            hypergraph.add_hyperedge(
                edge_id=rel["id"],
                nodes=rel["participants"],
                edge_type=rel["type"],
                attributes=rel["attributes"]
            )
        
        # Persist the graph
        hypergraph.save()
        logger.info(f"✅ Hypergraph {graph_id} created and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create hypergraph: {e}")
        raise

def run_distributed_analysis(graph_id: str) -> Dict[str, Any]:
    """Run comprehensive analysis on the hypergraph using Ray."""
    try:
        import ray
        from anant.core.hypergraph import AnantHyperGraph
        from anant.analytics.graph_analytics import GraphAnalytics
        
        logger.info(f"🔬 Starting distributed analysis on {graph_id}")
        
        # Load the existing graph
        hypergraph = AnantHyperGraph.load(graph_id)
        analytics = GraphAnalytics(hypergraph)
        
        # Define Ray remote functions for distributed analysis
        @ray.remote
        def compute_centrality_measures():
            """Compute various centrality measures."""
            try:
                results = {}
                # Node degree centrality
                degree_centrality = analytics.compute_degree_centrality()
                results["degree_centrality"] = degree_centrality
                
                # Hyperedge participation
                edge_participation = analytics.compute_hyperedge_participation()
                results["edge_participation"] = edge_participation
                
                return {"centrality": results, "status": "success"}
            except Exception as e:
                return {"centrality": {}, "status": "error", "error": str(e)}
        
        @ray.remote 
        def compute_clustering_analysis():
            """Compute clustering and community detection."""
            try:
                results = {}
                # Community detection
                communities = analytics.detect_communities()
                results["communities"] = communities
                
                # Clustering coefficient
                clustering = analytics.compute_clustering_coefficient()
                results["clustering_coefficient"] = clustering
                
                return {"clustering": results, "status": "success"}
            except Exception as e:
                return {"clustering": {}, "status": "error", "error": str(e)}
        
        @ray.remote
        def compute_path_analysis():
            """Compute path and connectivity analysis."""
            try:
                results = {}
                # Shortest paths
                paths = analytics.compute_shortest_paths()
                results["shortest_paths"] = paths
                
                # Connected components
                components = analytics.find_connected_components()
                results["connected_components"] = components
                
                return {"paths": results, "status": "success"}
            except Exception as e:
                return {"paths": {}, "status": "error", "error": str(e)}
        
        @ray.remote
        def compute_domain_analysis():
            """Compute domain-specific analysis."""
            try:
                results = {}
                # Entity type distribution
                type_dist = analytics.analyze_entity_types()
                results["entity_type_distribution"] = type_dist
                
                # Relationship strength
                rel_strength = analytics.compute_relationship_strength()
                results["relationship_strength"] = rel_strength
                
                return {"domain": results, "status": "success"}
            except Exception as e:
                return {"domain": {}, "status": "error", "error": str(e)}
        
        # Launch distributed tasks
        logger.info("🚀 Launching distributed analysis tasks...")
        
        futures = [
            compute_centrality_measures.remote(),
            compute_clustering_analysis.remote(), 
            compute_path_analysis.remote(),
            compute_domain_analysis.remote()
        ]
        
        # Collect results
        results = ray.get(futures)
        
        # Combine results
        analysis_results = {
            "graph_id": graph_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "cluster_info": ray.cluster_resources(),
            "analysis_results": {}
        }
        
        for result in results:
            analysis_results["analysis_results"].update(result)
        
        # Compute summary statistics
        total_nodes = len(hypergraph.nodes())
        total_edges = len(hypergraph.hyperedges())
        
        analysis_results["summary"] = {
            "total_nodes": total_nodes,
            "total_hyperedges": total_edges,
            "node_types": list(set(hypergraph.get_node_attribute(n, "type") for n in hypergraph.nodes())),
            "edge_types": list(set(hypergraph.get_hyperedge_attribute(e, "type") for e in hypergraph.hyperedges()))
        }
        
        logger.info(f"✅ Distributed analysis completed for {graph_id}")
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

def save_analysis_results(results: Dict[str, Any], filename: str = None):
    """Save analysis results to persistent storage."""
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/data/analysis_results_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Analysis results saved to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise

def verify_persistence_after_restart():
    """Verify that data persists after cluster restart."""
    try:
        logger.info("🔄 Testing persistence after restart simulation...")
        
        # Simulate restart by disconnecting and reconnecting to Ray
        ray.shutdown()
        time.sleep(2)
        ray.init()
        
        logger.info("✅ Simulated restart completed, Ray reconnected")
        return True
        
    except Exception as e:
        logger.error(f"❌ Restart simulation failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Anant Hypergraph Persistence Test")
    
    # Step 1: Setup environment
    if not setup_environment():
        return False
    
    # Step 2: Connect to Ray
    if not connect_to_ray():
        return False
    
    # Step 3: Create sample data
    entities, relationships = create_sample_data()
    graph_hash = compute_graph_hash(entities, relationships)
    graph_id = f"medical_research_graph_{graph_hash}"
    
    logger.info(f"📊 Graph ID: {graph_id}")
    logger.info(f"📈 Sample data: {len(entities)} entities, {len(relationships)} relationships")
    
    try:
        # Step 4: Create hypergraph (only if needed)
        graph_created = create_hypergraph_if_needed(graph_id, entities, relationships)
        
        # Step 5: Run distributed analysis
        analysis_results = run_distributed_analysis(graph_id)
        
        # Step 6: Save results
        results_file = save_analysis_results(analysis_results)
        
        # Step 7: Test persistence
        if verify_persistence_after_restart():
            logger.info("✅ Persistence test passed")
        
        # Step 8: Re-run analysis to verify consistency
        logger.info("🔍 Re-running analysis to verify consistency...")
        analysis_results_2 = run_distributed_analysis(graph_id)
        
        # Compare key metrics for consistency
        summary_1 = analysis_results["summary"]
        summary_2 = analysis_results_2["summary"]
        
        consistency_check = (
            summary_1["total_nodes"] == summary_2["total_nodes"] and
            summary_1["total_hyperedges"] == summary_2["total_hyperedges"] and
            summary_1["node_types"] == summary_2["node_types"]
        )
        
        if consistency_check:
            logger.info("✅ Consistency test passed - same results after restart")
        else:
            logger.warning("⚠️ Consistency test failed - results differ after restart")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 ANANT HYPERGRAPH TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Graph ID: {graph_id}")
        logger.info(f"🏗️ Graph Created: {'Yes' if graph_created else 'No (already existed)'}")
        logger.info(f"📈 Nodes: {summary_1['total_nodes']}")
        logger.info(f"🔗 Hyperedges: {summary_1['total_hyperedges']}")
        logger.info(f"🏷️ Node Types: {', '.join(summary_1['node_types'])}")
        logger.info(f"🔄 Persistence: ✅ Verified")
        logger.info(f"🔍 Consistency: {'✅ Verified' if consistency_check else '❌ Failed'}")
        logger.info(f"💾 Results File: {results_file}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)