#!/usr/bin/env python3
"""
ANANT Graph Methods Audit - Missing Functionality Analysis
=========================================================

This script comprehensively analyzes all four graph types to identify:
1. Currently implemented methods
2. Expected methods based on industry standards
3. Missing critical functionality
4. Priority recommendations for implementation

Graph Types Analyzed:
- Hypergraph (Base hypergraph operations)
- KnowledgeGraph (Semantic reasoning, ontologies)  
- HierarchicalKnowledgeGraph (Multi-level hierarchies)
- Metagraph (Enterprise metadata management)
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
import json

# Add anant to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class GraphMethodAuditor:
    """Comprehensive auditor for graph method completeness"""
    
    def __init__(self):
        self.audit_results = {
            'Hypergraph': {'existing': set(), 'missing': set(), 'expected': set()},
            'KnowledgeGraph': {'existing': set(), 'missing': set(), 'expected': set()},
            'HierarchicalKnowledgeGraph': {'existing': set(), 'missing': set(), 'expected': set()},
            'Metagraph': {'existing': set(), 'missing': set(), 'expected': set()}
        }
        
        # Define expected methods based on industry standards
        self._define_expected_methods()
        
    def _define_expected_methods(self):
        """Define what methods each graph type should have based on industry standards"""
        
        # Expected Hypergraph methods (NetworkX, igraph, hypergraph libraries)
        hypergraph_expected = {
            # Core structure methods
            'add_node', 'add_edge', 'remove_node', 'remove_edge',
            'num_nodes', 'num_edges', 'nodes', 'edges',
            'has_node', 'has_edge', 'clear', 'copy',
            
            # Hypergraph-specific operations
            'edge_size', 'edge_cardinality', 'node_degree', 'edge_degree',
            'incidence_matrix', 'adjacency_matrix', 'laplacian_matrix',
            'dual_graph', 'line_graph', 'intersection_graph',
            
            # Analysis methods
            'is_connected', 'connected_components', 'diameter', 
            'clustering_coefficient', 'betweenness_centrality', 'closeness_centrality',
            'eigenvector_centrality', 'pagerank', 'hits',
            
            # Algorithms
            'shortest_path', 'all_shortest_paths', 'minimum_spanning_tree',
            'k_core_decomposition', 'modularity', 'community_detection',
            'spectral_clustering', 'max_flow', 'min_cut',
            
            # Subgraph operations
            'subgraph', 'induced_subgraph', 'edge_induced_subgraph',
            'neighbors', 'successors', 'predecessors',
            
            # I/O and conversion
            'to_networkx', 'from_networkx', 'to_json', 'from_json',
            'to_graphml', 'from_graphml', 'to_gexf', 'from_gexf',
            
            # Visualization support
            'draw', 'layout', 'pos', 'get_layout_coordinates'
        }
        
        # Expected KnowledgeGraph methods (RDFLib, SPARQL, ontology libraries)
        knowledge_graph_expected = hypergraph_expected | {
            # Entity and relationship management
            'add_entity', 'add_relationship', 'remove_entity', 'remove_relationship',
            'get_entity', 'get_relationship', 'update_entity', 'update_relationship',
            
            # Semantic querying
            'query_sparql', 'query_cypher', 'semantic_search', 'pattern_matching',
            'get_entities_by_type', 'get_relationships_by_type', 
            'entity_neighborhood', 'relationship_patterns',
            
            # Reasoning and inference
            'infer_relationships', 'semantic_similarity', 'entity_linking',
            'ontology_reasoning', 'rule_based_inference', 'transitive_closure',
            'shortest_semantic_path', 'semantic_distance',
            
            # Ontology operations
            'add_ontology', 'extract_ontology', 'validate_ontology',
            'get_schema', 'schema_matching', 'ontology_alignment',
            'class_hierarchy', 'property_hierarchy',
            
            # Knowledge extraction
            'extract_triples', 'extract_concepts', 'extract_relations',
            'named_entity_recognition', 'relation_extraction',
            'concept_extraction', 'taxonomy_extraction',
            
            # Embeddings and vectors
            'generate_embeddings', 'entity_embeddings', 'relation_embeddings',
            'similarity_search', 'vector_search', 'nearest_neighbors',
            'embedding_based_reasoning',
            
            # Temporal reasoning
            'add_temporal_fact', 'temporal_query', 'temporal_reasoning',
            'time_based_inference', 'fact_validity_period',
            
            # Provenance and lineage
            'add_provenance', 'track_lineage', 'source_attribution',
            'fact_confidence', 'uncertainty_reasoning'
        }
        
        # Expected HierarchicalKnowledgeGraph methods
        hierarchical_kg_expected = knowledge_graph_expected | {
            # Hierarchy management
            'create_level', 'add_level', 'remove_level', 'get_level',
            'add_entity_to_level', 'move_entity_to_level', 'get_entity_level',
            'get_entities_at_level', 'get_level_entities',
            
            # Hierarchical navigation
            'get_parent', 'get_children', 'get_ancestors', 'get_descendants',
            'get_siblings', 'navigate_up', 'navigate_down', 'navigate_across',
            'level_distance', 'hierarchical_path',
            
            # Cross-level operations
            'add_cross_level_relationship', 'get_cross_level_relationships',
            'find_cross_level_patterns', 'hierarchical_inference',
            'level_based_reasoning', 'aggregate_across_levels',
            
            # Hierarchy analysis
            'get_hierarchy_statistics', 'max_depth', 'avg_branching_factor',
            'hierarchy_balance', 'level_distribution', 'hierarchy_metrics',
            'detect_hierarchy_anomalies', 'optimize_hierarchy',
            
            # Multi-level queries
            'query_at_level', 'aggregate_query', 'drill_down_query',
            'roll_up_query', 'slice_hierarchy', 'dice_hierarchy',
            
            # Hierarchy visualization
            'generate_hierarchy_layout', 'tree_layout', 'radial_layout',
            'hierarchical_drawing', 'level_based_positioning'
        }
        
        # Expected Metagraph methods (Enterprise metadata management)
        metagraph_expected = hypergraph_expected | {
            # Entity lifecycle management
            'create_entity', 'update_entity', 'delete_entity', 'archive_entity',
            'restore_entity', 'get_entity_history', 'entity_versions',
            
            # Metadata operations
            'add_metadata', 'update_metadata', 'get_metadata', 'remove_metadata',
            'metadata_search', 'metadata_validation', 'schema_enforcement',
            
            # Data lineage and provenance
            'add_lineage', 'get_lineage', 'upstream_lineage', 'downstream_lineage',
            'impact_analysis', 'root_cause_analysis', 'lineage_visualization',
            'data_flow_analysis', 'dependency_mapping',
            
            # Governance and compliance
            'add_policy', 'enforce_policy', 'check_compliance', 'audit_trail',
            'access_control', 'data_classification', 'retention_policy',
            'privacy_controls', 'gdpr_compliance', 'regulatory_reporting',
            
            # Quality management
            'data_quality_rules', 'quality_assessment', 'quality_metrics',
            'anomaly_detection', 'data_profiling', 'quality_monitoring',
            'quality_reporting', 'quality_improvement',
            
            # Catalog and discovery
            'catalog_search', 'semantic_discovery', 'recommendation_engine',
            'similar_datasets', 'usage_analytics', 'popularity_metrics',
            'data_freshness', 'usage_patterns',
            
            # Cost and resource management
            'cost_tracking', 'resource_utilization', 'optimization_suggestions',
            'storage_analytics', 'compute_costs', 'cost_allocation',
            
            # Temporal and versioning
            'version_control', 'temporal_queries', 'time_travel',
            'snapshot_management', 'change_tracking', 'rollback',
            
            # Integration and federation
            'federated_query', 'cross_system_lineage', 'metadata_synchronization',
            'system_integration', 'api_management', 'connector_management',
            
            # Analytics and reporting
            'get_statistics', 'generate_reports', 'dashboard_data',
            'trend_analysis', 'usage_reports', 'governance_reports'
        }
        
        self.audit_results['Hypergraph']['expected'] = hypergraph_expected
        self.audit_results['KnowledgeGraph']['expected'] = knowledge_graph_expected
        self.audit_results['HierarchicalKnowledgeGraph']['expected'] = hierarchical_kg_expected
        self.audit_results['Metagraph']['expected'] = metagraph_expected
    
    def analyze_existing_methods(self):
        """Analyze what methods currently exist in each graph class"""
        
        try:
            # Analyze Hypergraph
            from anant.classes.hypergraph import Hypergraph
            hypergraph_methods = set(method for method in dir(Hypergraph) 
                                   if not method.startswith('_') and callable(getattr(Hypergraph, method)))
            self.audit_results['Hypergraph']['existing'] = hypergraph_methods
            print(f"✅ Found {len(hypergraph_methods)} Hypergraph methods")
            
        except Exception as e:
            print(f"❌ Error analyzing Hypergraph: {e}")
        
        try:
            # Analyze KnowledgeGraph
            from anant.kg import KnowledgeGraph
            kg_methods = set(method for method in dir(KnowledgeGraph) 
                           if not method.startswith('_') and callable(getattr(KnowledgeGraph, method)))
            self.audit_results['KnowledgeGraph']['existing'] = kg_methods
            print(f"✅ Found {len(kg_methods)} KnowledgeGraph methods")
            
        except Exception as e:
            print(f"❌ Error analyzing KnowledgeGraph: {e}")
        
        try:
            # Analyze HierarchicalKnowledgeGraph
            from anant.kg import HierarchicalKnowledgeGraph
            hkg_methods = set(method for method in dir(HierarchicalKnowledgeGraph) 
                            if not method.startswith('_') and callable(getattr(HierarchicalKnowledgeGraph, method)))
            self.audit_results['HierarchicalKnowledgeGraph']['existing'] = hkg_methods
            print(f"✅ Found {len(hkg_methods)} HierarchicalKnowledgeGraph methods")
            
        except Exception as e:
            print(f"❌ Error analyzing HierarchicalKnowledgeGraph: {e}")
        
        try:
            # Analyze Metagraph
            from anant.metagraph import Metagraph
            mg_methods = set(method for method in dir(Metagraph) 
                           if not method.startswith('_') and callable(getattr(Metagraph, method)))
            self.audit_results['Metagraph']['existing'] = mg_methods
            print(f"✅ Found {len(mg_methods)} Metagraph methods")
            
        except Exception as e:
            print(f"❌ Error analyzing Metagraph: {e}")
    
    def compute_missing_methods(self):
        """Compute missing methods for each graph type"""
        
        for graph_type in self.audit_results:
            existing = self.audit_results[graph_type]['existing']
            expected = self.audit_results[graph_type]['expected']
            missing = expected - existing
            self.audit_results[graph_type]['missing'] = missing
    
    def categorize_missing_methods(self):
        """Categorize missing methods by functionality area"""
        
        categories = {
            'core_structure': ['add_', 'remove_', 'num_', 'has_', 'get_', 'nodes', 'edges'],
            'algorithms': ['shortest_', 'community_', 'clustering_', 'centrality', 'pagerank', 'modularity'],
            'analysis': ['connected_', 'diameter', 'betweenness_', 'closeness_', 'spectral_'],
            'semantic': ['semantic_', 'ontology_', 'reasoning_', 'inference_', 'sparql_', 'entity_', 'relation'],
            'temporal': ['temporal_', 'time_', 'history_', 'version_'],
            'hierarchy': ['level_', 'hierarchy_', 'parent_', 'children_', 'ancestor_', 'descendant_'],
            'governance': ['policy_', 'compliance_', 'audit_', 'quality_', 'lineage_'],
            'io_conversion': ['to_', 'from_', 'import_', 'export_', 'load_', 'save_'],
            'visualization': ['draw_', 'layout_', 'plot_', 'render_', 'visualize_']
        }
        
        categorized_missing = {}
        
        for graph_type in self.audit_results:
            missing = self.audit_results[graph_type]['missing']
            categorized_missing[graph_type] = defaultdict(list)
            
            for method in missing:
                categorized = False
                for category, keywords in categories.items():
                    if any(keyword in method for keyword in keywords):
                        categorized_missing[graph_type][category].append(method)
                        categorized = True
                        break
                
                if not categorized:
                    categorized_missing[graph_type]['other'].append(method)
        
        return categorized_missing
    
    def prioritize_missing_methods(self):
        """Prioritize missing methods by importance"""
        
        # Critical methods that should be implemented first
        critical_methods = {
            'Hypergraph': {
                'shortest_path', 'connected_components', 'diameter', 'clustering_coefficient',
                'k_core_decomposition', 'modularity', 'dual_graph', 'line_graph'
            },
            'KnowledgeGraph': {
                'add_entity', 'add_relationship', 'semantic_search', 'infer_relationships',
                'semantic_similarity', 'shortest_semantic_path', 'extract_ontology',
                'get_entities_by_type', 'get_relationships_by_type'
            },
            'HierarchicalKnowledgeGraph': {
                'get_parent', 'get_children', 'get_ancestors', 'get_descendants',
                'max_depth', 'avg_branching_factor', 'get_hierarchy_statistics',
                'cross_level_relationships'
            },
            'Metagraph': {
                'get_lineage', 'impact_analysis', 'check_compliance', 'get_statistics',
                'data_quality_rules', 'audit_trail', 'cost_tracking'
            }
        }
        
        prioritized = {}
        for graph_type in self.audit_results:
            missing = self.audit_results[graph_type]['missing']
            critical = critical_methods.get(graph_type, set())
            
            prioritized[graph_type] = {
                'critical': list(missing & critical),
                'important': list(missing - critical)
            }
        
        return prioritized
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive audit report"""
        
        print("\n" + "="*80)
        print("🔍 ANANT GRAPH METHODS AUDIT REPORT")
        print("="*80)
        
        # Summary statistics
        total_expected = sum(len(self.audit_results[gt]['expected']) for gt in self.audit_results)
        total_existing = sum(len(self.audit_results[gt]['existing']) for gt in self.audit_results)
        total_missing = sum(len(self.audit_results[gt]['missing']) for gt in self.audit_results)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Expected Methods: {total_expected}")
        print(f"   Total Existing Methods: {total_existing}")
        print(f"   Total Missing Methods:  {total_missing}")
        print(f"   Implementation Rate:    {(total_existing/total_expected*100):.1f}%")
        
        # Per-graph analysis
        categorized = self.categorize_missing_methods()
        prioritized = self.prioritize_missing_methods()
        
        for graph_type in self.audit_results:
            existing = self.audit_results[graph_type]['existing']
            expected = self.audit_results[graph_type]['expected']
            missing = self.audit_results[graph_type]['missing']
            
            print(f"\n" + "-"*60)
            print(f"📋 {graph_type.upper()} ANALYSIS")
            print(f"-"*60)
            print(f"Expected Methods: {len(expected)}")
            print(f"Existing Methods: {len(existing)}")
            print(f"Missing Methods:  {len(missing)}")
            print(f"Completion Rate:  {(len(existing)/len(expected)*100):.1f}%")
            
            # Show missing methods by category
            missing_by_category = categorized[graph_type]
            print(f"\n🔍 Missing Methods by Category:")
            for category, methods in missing_by_category.items():
                if methods:
                    print(f"   {category.title()}: {len(methods)} methods")
                    for method in sorted(methods)[:5]:  # Show first 5
                        print(f"     - {method}")
                    if len(methods) > 5:
                        print(f"     ... and {len(methods) - 5} more")
            
            # Show critical missing methods
            critical_missing = prioritized[graph_type]['critical']
            if critical_missing:
                print(f"\n🚨 CRITICAL Missing Methods:")
                for method in sorted(critical_missing):
                    print(f"   - {method}")
        
        return {
            'summary': {
                'total_expected': total_expected,
                'total_existing': total_existing,
                'total_missing': total_missing,
                'implementation_rate': total_existing/total_expected*100
            },
            'by_graph_type': self.audit_results,
            'categorized_missing': categorized,
            'prioritized_missing': prioritized
        }
    
    def save_detailed_report(self, filename='graph_methods_audit.json'):
        """Save detailed audit results to JSON file"""
        
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for graph_type, data in self.audit_results.items():
            serializable_results[graph_type] = {
                'existing': sorted(list(data['existing'])),
                'missing': sorted(list(data['missing'])),
                'expected': sorted(list(data['expected']))
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {filename}")

def main():
    """Run the comprehensive graph methods audit"""
    
    print("🔍 Starting ANANT Graph Methods Audit...")
    print("="*60)
    
    auditor = GraphMethodAuditor()
    
    print("\n📊 Analyzing existing methods...")
    auditor.analyze_existing_methods()
    
    print("\n🔍 Computing missing methods...")
    auditor.compute_missing_methods()
    
    print("\n📋 Generating comprehensive report...")
    report_data = auditor.generate_comprehensive_report()
    
    print("\n💾 Saving detailed audit results...")
    auditor.save_detailed_report()
    
    print("\n✅ Audit complete!")
    return report_data

if __name__ == "__main__":
    main()