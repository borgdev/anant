#!/usr/bin/env python3
"""
Comprehensive Graph Analysis Test Suite
======================================

Advanced testing suite to identify missing functionalities and test edge cases
across all four graph types: Hypergraph, KnowledgeGraph, HierarchicalKnowledgeGraph, and Metagraph.

This test suite focuses on:
1. Advanced operations and complex scenarios
2. Performance and scalability testing
3. Edge case handling and error resilience
4. Feature completeness and functionality gaps
5. Interoperability between graph types
"""

import pytest
import time
import traceback
import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from anant.classes.hypergraph import Hypergraph
from anant.classes.incidence_store import IncidenceStore
from anant.kg.core import KnowledgeGraph
try:
    from anant.kg.core import HierarchicalKnowledgeGraph
except ImportError:
    HierarchicalKnowledgeGraph = None
from anant.metagraph.core.metagraph import Metagraph

# Advanced functionality imports
try:
    import anant.algorithms.centrality as centrality
    import anant.algorithms.clustering as clustering
    ALGORITHMS_AVAILABLE = True
except ImportError:
    ALGORITHMS_AVAILABLE = False

try:
    from anant.streaming.core.stream_processor import GraphStreamProcessor
    from anant.streaming.core.event_store import EventStore
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False


class GraphTestMetrics:
    """Collect comprehensive test metrics for analysis"""
    
    def __init__(self):
        self.performance_data = {}
        self.functionality_gaps = {}
        self.error_patterns = {}
        self.feature_coverage = {}
    
    def record_performance(self, graph_type: str, operation: str, duration: float, data_size: int):
        """Record performance metrics"""
        if graph_type not in self.performance_data:
            self.performance_data[graph_type] = {}
        
        self.performance_data[graph_type][operation] = {
            'duration': duration,
            'data_size': data_size,
            'ops_per_second': data_size / duration if duration > 0 else 0
        }
    
    def record_gap(self, graph_type: str, missing_feature: str, description: str):
        """Record missing functionality"""
        if graph_type not in self.functionality_gaps:
            self.functionality_gaps[graph_type] = {}
        
        self.functionality_gaps[graph_type][missing_feature] = description
    
    def record_error(self, graph_type: str, operation: str, error_type: str, error_msg: str):
        """Record error patterns"""
        if graph_type not in self.error_patterns:
            self.error_patterns[graph_type] = {}
        
        self.error_patterns[graph_type][operation] = {
            'error_type': error_type,
            'error_msg': error_msg
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = "\n" + "="*80 + "\n"
        report += "🔍 COMPREHENSIVE GRAPH ANALYSIS REPORT\n"
        report += "="*80 + "\n"
        
        # Performance Analysis
        report += "\n📊 PERFORMANCE ANALYSIS:\n" + "-"*40 + "\n"
        for graph_type, operations in self.performance_data.items():
            report += f"\n{graph_type}:\n"
            for op, metrics in operations.items():
                report += f"  • {op}: {metrics['duration']:.3f}s ({metrics['ops_per_second']:.1f} ops/sec)\n"
        
        # Functionality Gaps
        if self.functionality_gaps:
            report += "\n🚨 FUNCTIONALITY GAPS:\n" + "-"*40 + "\n"
            for graph_type, gaps in self.functionality_gaps.items():
                if gaps:
                    report += f"\n{graph_type}:\n"
                    for feature, desc in gaps.items():
                        report += f"  ❌ {feature}: {desc}\n"
        
        # Error Analysis
        if self.error_patterns:
            report += "\n⚠️  ERROR PATTERNS:\n" + "-"*40 + "\n"
            for graph_type, errors in self.error_patterns.items():
                if errors:
                    report += f"\n{graph_type}:\n"
                    for op, error_info in errors.items():
                        report += f"  🔥 {op}: {error_info['error_type']} - {error_info['error_msg'][:100]}...\n"
        
        return report


# Global metrics collector
test_metrics = GraphTestMetrics()


def test_advanced_hypergraph_operations():
    """Test advanced Hypergraph operations and identify missing features"""
    print("\n🔬 Testing Advanced Hypergraph Operations...")
    
    # Create test data with varying complexity
    test_cases = [
        # Simple case
        {"edge_id": ["e1", "e1", "e2", "e2"], "node_id": ["n1", "n2", "n2", "n3"], "weight": [1.0, 1.0, 1.5, 1.5]},
        
        # Complex case with large hyperedges
        {"edge_id": ["e1"] * 10 + ["e2"] * 5 + ["e3"] * 8,
         "node_id": [f"n{i}" for i in range(10)] + [f"n{i}" for i in range(5, 10)] + [f"n{i}" for i in range(15, 23)],
         "weight": [1.0] * 23},
        
        # Sparse case
        {"edge_id": ["e1", "e2", "e3"], "node_id": ["n1", "n100", "n200"], "weight": [1.0, 2.0, 3.0]}
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\n  Test Case {i+1}: {len(set(test_data['edge_id']))} edges, {len(set(test_data['node_id']))} nodes")
        
        # Create hypergraph
        start_time = time.time()
        df = pl.DataFrame(test_data)
        hg = Hypergraph(IncidenceStore(df))
        creation_time = time.time() - start_time
        
        test_metrics.record_performance("Hypergraph", f"creation_case_{i+1}", creation_time, len(test_data['edge_id']))
        
        # Test 1: Basic Operations
        basic_ops_passed = []
        
        try:
            # Node and edge counts
            num_nodes = hg.num_nodes
            num_edges = hg.num_edges
            basic_ops_passed.append("node/edge_counts")
            print(f"    ✅ Basic counts: {num_nodes} nodes, {num_edges} edges")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "basic_counts", type(e).__name__, str(e))
            print(f"    ❌ Basic counts failed: {e}")
        
        try:
            # Node and edge iteration
            nodes = list(hg.nodes)
            edges = list(hg.edges)
            basic_ops_passed.append("iteration")
            print(f"    ✅ Iteration: {len(nodes)} nodes, {len(edges)} edges accessible")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "iteration", type(e).__name__, str(e))
            print(f"    ❌ Iteration failed: {e}")
        
        # Test 2: Advanced Query Operations
        advanced_ops = []
        
        try:
            # Test degree computation for all nodes
            start_time = time.time()
            degrees = {}
            for node in hg.nodes:
                try:
                    degree = hg.degree(node)
                    degrees[node] = degree
                except AttributeError:
                    # Try alternative method
                    node_edges = hg.incidences.get_node_edges(node)
                    degrees[node] = len(node_edges)
            
            degree_time = time.time() - start_time
            test_metrics.record_performance("Hypergraph", f"degree_computation_case_{i+1}", degree_time, len(degrees))
            advanced_ops.append("degree_computation")
            print(f"    ✅ Degree computation: avg={np.mean(list(degrees.values())):.2f}, time={degree_time:.3f}s")
        
        except Exception as e:
            test_metrics.record_error("Hypergraph", "degree_computation", type(e).__name__, str(e))
            print(f"    ❌ Degree computation failed: {e}")
        
        try:
            # Test edge size computation
            edge_sizes = {}
            for edge in hg.edges:
                try:
                    size = hg.size_of_edge(edge)
                    edge_sizes[edge] = size
                except AttributeError:
                    # Try alternative method
                    edge_nodes = hg.incidences.get_edge_nodes(edge)
                    edge_sizes[edge] = len(edge_nodes)
            
            advanced_ops.append("edge_sizes")
            print(f"    ✅ Edge sizes: avg={np.mean(list(edge_sizes.values())):.2f}")
        
        except Exception as e:
            test_metrics.record_error("Hypergraph", "edge_sizes", type(e).__name__, str(e))
            print(f"    ❌ Edge size computation failed: {e}")
        
        # Test 3: Missing Advanced Features
        missing_features = []
        
        # Test for k-core decomposition
        try:
            # This likely doesn't exist yet
            k_cores = hg.k_core_decomposition()
            print(f"    ✅ K-core decomposition available")
        except AttributeError:
            missing_features.append("k_core_decomposition")
            test_metrics.record_gap("Hypergraph", "k_core_decomposition", "K-core decomposition for hypergraph clustering")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "k_core_decomposition", type(e).__name__, str(e))
        
        # Test for hypergraph diameter
        try:
            diameter = hg.diameter()
            print(f"    ✅ Hypergraph diameter: {diameter}")
        except AttributeError:
            missing_features.append("diameter")
            test_metrics.record_gap("Hypergraph", "diameter", "Hypergraph diameter computation")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "diameter", type(e).__name__, str(e))
        
        # Test for modularity computation
        try:
            modularity = hg.modularity()
            print(f"    ✅ Modularity: {modularity}")
        except AttributeError:
            missing_features.append("modularity")
            test_metrics.record_gap("Hypergraph", "modularity", "Hypergraph modularity computation")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "modularity", type(e).__name__, str(e))
        
        # Test for dual hypergraph construction
        try:
            dual_hg = hg.dual()
            print(f"    ✅ Dual hypergraph: {dual_hg.num_nodes} nodes, {dual_hg.num_edges} edges")
        except AttributeError:
            missing_features.append("dual")
            test_metrics.record_gap("Hypergraph", "dual", "Dual hypergraph construction (nodes<->edges)")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "dual", type(e).__name__, str(e))
        
        # Test for hypergraph line graph
        try:
            line_graph = hg.line_graph()
            print(f"    ✅ Line graph construction available")
        except AttributeError:
            missing_features.append("line_graph")
            test_metrics.record_gap("Hypergraph", "line_graph", "Line graph construction for hypergraph analysis")
        except Exception as e:
            test_metrics.record_error("Hypergraph", "line_graph", type(e).__name__, str(e))
        
        if missing_features:
            print(f"    🚨 Missing features: {', '.join(missing_features)}")


def test_advanced_knowledge_graph_operations():
    """Test advanced KnowledgeGraph operations and semantic reasoning"""
    print("\n🧠 Testing Advanced KnowledgeGraph Operations...")
    
    # Create test knowledge graph with complex entities and relationships
    kg = KnowledgeGraph()
    
    # Add entities with rich properties
    test_entities = [
        ("person:john", "person", {"name": "John Doe", "age": 30, "occupation": "Engineer", "skills": ["Python", "AI"]}),
        ("person:jane", "person", {"name": "Jane Smith", "age": 28, "occupation": "Scientist", "skills": ["ML", "Data"]}),
        ("org:acme", "organization", {"name": "Acme Corp", "industry": "Technology", "employees": 1000}),
        ("proj:alpha", "project", {"name": "Project Alpha", "budget": 100000, "status": "active"}),
        ("skill:python", "skill", {"name": "Python Programming", "category": "Programming", "difficulty": 0.7})
    ]
    
    start_time = time.time()
    for entity_id, entity_type, properties in test_entities:
        try:
            kg.add_node(entity_id, properties=properties, entity_type=entity_type)
        except Exception as e:
            print(f"    ❌ Failed to add entity {entity_id}: {e}")
    
    creation_time = time.time() - start_time
    test_metrics.record_performance("KnowledgeGraph", "entity_creation", creation_time, len(test_entities))
    
    # Add relationships with semantic meaning
    test_relationships = [
        ("person:john", "person:jane", "knows", {"strength": 0.8, "context": "work"}),
        ("person:john", "org:acme", "works_for", {"role": "Senior Engineer", "since": "2020"}),
        ("person:jane", "org:acme", "works_for", {"role": "Data Scientist", "since": "2021"}),
        ("proj:alpha", "org:acme", "owned_by", {"ownership": 1.0}),
        ("person:john", "proj:alpha", "participates_in", {"contribution": 0.6}),
        ("person:john", "skill:python", "has_skill", {"proficiency": 0.9})
    ]
    
    start_time = time.time()
    for source, target, rel_type, properties in test_relationships:
        try:
            edge_id = f"{source}_{rel_type}_{target}"
            kg.add_edge(edge_id, [source, target], properties=properties, edge_type=rel_type)
        except Exception as e:
            print(f"    ❌ Failed to add relationship {source}->{target}: {e}")
    
    relationship_time = time.time() - start_time
    test_metrics.record_performance("KnowledgeGraph", "relationship_creation", relationship_time, len(test_relationships))
    
    print(f"    ✅ Created KG: {kg.num_nodes} entities, {kg.num_edges} relationships")
    
    # Test advanced semantic operations
    missing_features = []
    
    # Test 1: Semantic Query Operations
    try:
        # Entity type queries
        persons = kg.get_entities_by_type("person")
        print(f"    ✅ Type-based query: found {len(persons)} persons")
    except AttributeError:
        missing_features.append("get_entities_by_type")
        test_metrics.record_gap("KnowledgeGraph", "get_entities_by_type", "Query entities by semantic type")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "get_entities_by_type", type(e).__name__, str(e))
    
    # Test 2: Path Analysis
    try:
        # Shortest semantic path
        path = kg.shortest_semantic_path("person:john", "skill:python")
        print(f"    ✅ Semantic pathfinding available")
    except AttributeError:
        missing_features.append("shortest_semantic_path")
        test_metrics.record_gap("KnowledgeGraph", "shortest_semantic_path", "Semantic pathfinding with relationship type awareness")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "shortest_semantic_path", type(e).__name__, str(e))
    
    # Test 3: Semantic Similarity
    try:
        # Entity similarity based on relationships and properties
        similarity = kg.semantic_similarity("person:john", "person:jane")
        print(f"    ✅ Semantic similarity: {similarity}")
    except AttributeError:
        missing_features.append("semantic_similarity")
        test_metrics.record_gap("KnowledgeGraph", "semantic_similarity", "Entity similarity based on semantic features")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "semantic_similarity", type(e).__name__, str(e))
    
    # Test 4: Knowledge Inference
    try:
        # Infer new relationships based on existing patterns
        inferred = kg.infer_relationships(confidence_threshold=0.7)
        print(f"    ✅ Relationship inference: {len(inferred)} new relationships")
    except AttributeError:
        missing_features.append("infer_relationships")
        test_metrics.record_gap("KnowledgeGraph", "infer_relationships", "Automatic relationship inference and knowledge discovery")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "infer_relationships", type(e).__name__, str(e))
    
    # Test 5: Ontology Operations
    try:
        # Schema extraction and ontology mapping
        ontology = kg.extract_ontology()
        print(f"    ✅ Ontology extraction available")
    except AttributeError:
        missing_features.append("extract_ontology")
        test_metrics.record_gap("KnowledgeGraph", "extract_ontology", "Automatic ontology extraction from graph structure")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "extract_ontology", type(e).__name__, str(e))
    
    # Test 6: SPARQL-like Queries
    try:
        # Complex pattern matching queries
        results = kg.query_pattern([
            ("?person", "works_for", "org:acme"),
            ("?person", "has_skill", "?skill")
        ])
        print(f"    ✅ Pattern matching queries available")
    except AttributeError:
        missing_features.append("query_pattern")
        test_metrics.record_gap("KnowledgeGraph", "query_pattern", "SPARQL-like pattern matching queries")
    except Exception as e:
        test_metrics.record_error("KnowledgeGraph", "query_pattern", type(e).__name__, str(e))
    
    if missing_features:
        print(f"    🚨 Missing semantic features: {', '.join(missing_features)}")


def test_advanced_hierarchical_kg_operations():
    """Test advanced HierarchicalKnowledgeGraph operations"""
    print("\n🏗️ Testing Advanced HierarchicalKnowledgeGraph Operations...")
    
    # Create multi-level hierarchy using factory function
    try:
        from anant.kg import create_enterprise_hierarchy
        hkg = create_enterprise_hierarchy()
    except Exception as e:
        print(f"    ❌ Failed to create HierarchicalKG: {e}")
        return
    
    # Build complex hierarchy
    hierarchy_data = [
        # Domain level
        ("domain:tech", "domain", {"name": "Technology Domain", "scope": "global"}),
        
        # Organization level
        ("org:corp1", "organization", {"name": "Tech Corp 1", "parent": "domain:tech"}),
        ("org:corp2", "organization", {"name": "Tech Corp 2", "parent": "domain:tech"}),
        
        # Department level
        ("dept:eng", "department", {"name": "Engineering", "parent": "org:corp1"}),
        ("dept:sales", "department", {"name": "Sales", "parent": "org:corp1"}),
        ("dept:rd", "department", {"name": "R&D", "parent": "org:corp2"}),
        
        # Team level
        ("team:backend", "team", {"name": "Backend Team", "parent": "dept:eng"}),
        ("team:frontend", "team", {"name": "Frontend Team", "parent": "dept:eng"}),
        
        # Individual level
        ("person:alice", "individual", {"name": "Alice", "parent": "team:backend"}),
        ("person:bob", "individual", {"name": "Bob", "parent": "team:frontend"})
    ]
    
    start_time = time.time()
    for entity_id, level_type, properties in hierarchy_data:
        try:
            hkg.add_entity(entity_id, properties={"entity_type": level_type, **properties})
        except Exception as e:
            print(f"    ❌ Failed to add hierarchical entity {entity_id}: {e}")
    
    creation_time = time.time() - start_time
    test_metrics.record_performance("HierarchicalKnowledgeGraph", "hierarchy_creation", creation_time, len(hierarchy_data))
    
    print(f"    ✅ Created HKG: {hkg.num_nodes} entities across {len(hkg.levels)} levels")
    
    # Test hierarchical-specific operations
    missing_features = []
    
    # Test 1: Level-based Operations
    try:
        # Entities at specific level
        orgs = hkg.get_entities_at_level("organization")
        print(f"    ✅ Level queries: {len(orgs)} organizations")
    except AttributeError:
        missing_features.append("get_entities_at_level")
        test_metrics.record_gap("HierarchicalKnowledgeGraph", "get_entities_at_level", "Query entities at specific hierarchy level")
    except Exception as e:
        test_metrics.record_error("HierarchicalKnowledgeGraph", "get_entities_at_level", type(e).__name__, str(e))
    
    # Test 2: Cross-Level Analysis
    try:
        # Find all descendants of an entity
        descendants = hkg.get_descendants("org:corp1")
        print(f"    ✅ Descendant analysis: {len(descendants)} descendants")
    except AttributeError:
        missing_features.append("get_descendants")
        test_metrics.record_gap("HierarchicalKnowledgeGraph", "get_descendants", "Find all descendants in hierarchy")
    except Exception as e:
        test_metrics.record_error("HierarchicalKnowledgeGraph", "get_descendants", type(e).__name__, str(e))
    
    # Test 3: Hierarchy Metrics
    try:
        # Hierarchy depth and breadth analysis
        depth = hkg.max_depth()
        breadth = hkg.avg_branching_factor()
        print(f"    ✅ Hierarchy metrics: depth={depth}, avg_branching={breadth}")
    except AttributeError:
        missing_features.append("hierarchy_metrics")
        test_metrics.record_gap("HierarchicalKnowledgeGraph", "hierarchy_metrics", "Depth, breadth, and structural hierarchy metrics")
    except Exception as e:
        test_metrics.record_error("HierarchicalKnowledgeGraph", "hierarchy_metrics", type(e).__name__, str(e))
    
    # Test 4: Cross-Level Relationships
    try:
        # Relationships that skip hierarchy levels
        cross_level_rels = hkg.find_cross_level_relationships()
        print(f"    ✅ Cross-level relationship detection available")
    except AttributeError:
        missing_features.append("find_cross_level_relationships")
        test_metrics.record_gap("HierarchicalKnowledgeGraph", "find_cross_level_relationships", "Detect relationships that bypass hierarchy levels")
    except Exception as e:
        test_metrics.record_error("HierarchicalKnowledgeGraph", "find_cross_level_relationships", type(e).__name__, str(e))
    
    # Test 5: Hierarchy Visualization
    try:
        # Generate hierarchy layout data
        layout_data = hkg.generate_hierarchy_layout()
        print(f"    ✅ Hierarchy layout generation available")
    except AttributeError:
        missing_features.append("generate_hierarchy_layout")
        test_metrics.record_gap("HierarchicalKnowledgeGraph", "generate_hierarchy_layout", "Automatic hierarchy visualization layout")
    except Exception as e:
        test_metrics.record_error("HierarchicalKnowledgeGraph", "generate_hierarchy_layout", type(e).__name__, str(e))
    
    if missing_features:
        print(f"    🚨 Missing hierarchical features: {', '.join(missing_features)}")


def test_advanced_metagraph_operations():
    """Test advanced Metagraph enterprise operations"""
    print("\n🏢 Testing Advanced Metagraph Operations...")
    
    mg = Metagraph()
    
    # Create complex enterprise data ecosystem
    enterprise_entities = [
        # Data Assets
        ("dataset:sales_2023", "Dataset", {"owner": "sales_team", "size_gb": 150, "format": "parquet", "pii": True}),
        ("dataset:customer_profiles", "Dataset", {"owner": "marketing", "size_gb": 80, "format": "json", "pii": True}),
        ("dataset:product_catalog", "Dataset", {"owner": "product", "size_gb": 5, "format": "csv", "pii": False}),
        
        # Processing Components
        ("pipeline:etl_main", "Pipeline", {"owner": "data_eng", "status": "active", "sla": "99.9%"}),
        ("model:churn_prediction", "MLModel", {"owner": "ds_team", "accuracy": 0.87, "framework": "sklearn"}),
        ("api:customer_service", "API", {"owner": "backend_team", "version": "2.1", "rate_limit": 1000}),
        
        # Governance
        ("policy:gdpr_compliance", "Policy", {"type": "privacy", "mandatory": True, "scope": "eu_data"}),
        ("schema:customer_v2", "Schema", {"version": "2.0", "fields": 25, "validation": "strict"})
    ]
    
    start_time = time.time()
    for entity_id, entity_type, properties in enterprise_entities:
        try:
            mg.create_entity(entity_id, entity_type, properties)
        except Exception as e:
            print(f"    ❌ Failed to create entity {entity_id}: {e}")
    
    creation_time = time.time() - start_time
    test_metrics.record_performance("Metagraph", "enterprise_entity_creation", creation_time, len(enterprise_entities))
    
    print(f"    ✅ Created Metagraph: {mg.get_stats()['total_entities']} entities")
    
    # Test advanced enterprise operations
    missing_features = []
    
    # Test 1: Data Lineage Tracking
    try:
        # Track data flow through the system
        lineage = mg.get_data_lineage("dataset:sales_2023")
        print(f"    ✅ Data lineage tracking available")
    except AttributeError:
        missing_features.append("get_data_lineage")
        test_metrics.record_gap("Metagraph", "get_data_lineage", "End-to-end data lineage tracking")
    except Exception as e:
        test_metrics.record_error("Metagraph", "get_data_lineage", type(e).__name__, str(e))
    
    # Test 2: Impact Analysis
    try:
        # Analyze impact of changing/removing an entity
        impact = mg.analyze_impact("dataset:customer_profiles")
        print(f"    ✅ Impact analysis: {len(impact)} affected entities")
    except AttributeError:
        missing_features.append("analyze_impact")
        test_metrics.record_gap("Metagraph", "analyze_impact", "Change impact analysis for enterprise planning")
    except Exception as e:
        test_metrics.record_error("Metagraph", "analyze_impact", type(e).__name__, str(e))
    
    # Test 3: Compliance Monitoring
    try:
        # Check policy compliance across entities
        compliance_report = mg.check_compliance("policy:gdpr_compliance")
        print(f"    ✅ Compliance monitoring available")
    except AttributeError:
        missing_features.append("check_compliance")
        test_metrics.record_gap("Metagraph", "check_compliance", "Automated governance and compliance checking")
    except Exception as e:
        test_metrics.record_error("Metagraph", "check_compliance", type(e).__name__, str(e))
    
    # Test 4: Resource Optimization
    try:
        # Find optimization opportunities
        optimizations = mg.find_optimization_opportunities()
        print(f"    ✅ Resource optimization analysis available")
    except AttributeError:
        missing_features.append("find_optimization_opportunities")
        test_metrics.record_gap("Metagraph", "find_optimization_opportunities", "Automated resource and cost optimization")
    except Exception as e:
        test_metrics.record_error("Metagraph", "find_optimization_opportunities", type(e).__name__, str(e))
    
    # Test 5: Temporal Analysis
    try:
        # Analyze changes over time
        temporal_metrics = mg.get_temporal_metrics(days=30)
        print(f"    ✅ Temporal analysis available")
    except AttributeError:
        missing_features.append("get_temporal_metrics")
        test_metrics.record_gap("Metagraph", "get_temporal_metrics", "Time-based analysis and trend detection")
    except Exception as e:
        test_metrics.record_error("Metagraph", "get_temporal_metrics", type(e).__name__, str(e))
    
    # Test 6: Cost Analysis
    try:
        # Calculate storage and processing costs
        cost_breakdown = mg.calculate_cost_breakdown()
        print(f"    ✅ Cost analysis available")
    except AttributeError:
        missing_features.append("calculate_cost_breakdown")
        test_metrics.record_gap("Metagraph", "calculate_cost_breakdown", "Enterprise cost tracking and analysis")
    except Exception as e:
        test_metrics.record_error("Metagraph", "calculate_cost_breakdown", type(e).__name__, str(e))
    
    if missing_features:
        print(f"    🚨 Missing enterprise features: {', '.join(missing_features)}")


def test_cross_graph_interoperability():
    """Test interoperability and data migration between graph types"""
    print("\n🔄 Testing Cross-Graph Interoperability...")
    
    missing_features = []
    
    # Test 1: Hypergraph to KnowledgeGraph conversion
    try:
        # Create a hypergraph
        hg_data = {"edge_id": ["e1", "e1", "e2"], "node_id": ["n1", "n2", "n2"], "weight": [1.0, 1.0, 1.5]}
        hg = Hypergraph(IncidenceStore(pl.DataFrame(hg_data)))
        
        # Convert to knowledge graph
        kg = hg.to_knowledge_graph()
        print(f"    ✅ HG->KG conversion: {kg.num_nodes} entities, {kg.num_edges} relationships")
    
    except AttributeError:
        missing_features.append("hypergraph_to_kg_conversion")
        test_metrics.record_gap("CrossGraph", "hypergraph_to_kg_conversion", "Convert Hypergraph to KnowledgeGraph")
    except Exception as e:
        test_metrics.record_error("CrossGraph", "hypergraph_to_kg_conversion", type(e).__name__, str(e))
    
    # Test 2: KnowledgeGraph to Metagraph migration
    try:
        kg = KnowledgeGraph()
        kg.add_entity("entity1", properties={"type": "person"})
        
        # Migrate to metagraph
        mg = kg.to_metagraph()
        print(f"    ✅ KG->MG migration available")
    
    except AttributeError:
        missing_features.append("kg_to_metagraph_migration")
        test_metrics.record_gap("CrossGraph", "kg_to_metagraph_migration", "Migrate KnowledgeGraph to Metagraph")
    except Exception as e:
        test_metrics.record_error("CrossGraph", "kg_to_metagraph_migration", type(e).__name__, str(e))
    
    # Test 3: Unified query interface
    try:
        # Query across multiple graph types
        unified_query = {
            "select": ["entity_id", "properties"],
            "from": ["hypergraph", "knowledge_graph", "metagraph"],
            "where": {"property": "type", "value": "person"}
        }
        
        # This would be a unified query processor
        results = query_unified_graphs(unified_query)
        print(f"    ✅ Unified query interface available")
    
    except NameError:  # Function doesn't exist
        missing_features.append("unified_query_interface")
        test_metrics.record_gap("CrossGraph", "unified_query_interface", "Unified query interface across all graph types")
    except Exception as e:
        test_metrics.record_error("CrossGraph", "unified_query_interface", type(e).__name__, str(e))
    
    # Test 4: Graph fusion operations
    try:
        # Combine multiple graphs into one
        hg = Hypergraph(IncidenceStore(pl.DataFrame({"edge_id": ["e1"], "node_id": ["n1"], "weight": [1.0]})))
        kg = KnowledgeGraph()
        
        fused_graph = fuse_graphs([hg, kg], strategy="union")
        print(f"    ✅ Graph fusion available")
    
    except NameError:  # Function doesn't exist
        missing_features.append("graph_fusion")
        test_metrics.record_gap("CrossGraph", "graph_fusion", "Combine multiple graph types into unified structure")
    except Exception as e:
        test_metrics.record_error("CrossGraph", "graph_fusion", type(e).__name__, str(e))
    
    if missing_features:
        print(f"    🚨 Missing interoperability features: {', '.join(missing_features)}")


def test_scalability_and_performance():
    """Test scalability limits and performance characteristics"""
    print("\n⚡ Testing Scalability and Performance...")
    
    # Test different scales
    scales = [100, 500, 1000]  # Number of entities/edges
    
    for scale in scales:
        print(f"\n  Scale Test: {scale} entities/edges")
        
        # Test Hypergraph scalability
        try:
            start_time = time.time()
            
            # Generate large hypergraph
            edges_data = []
            for i in range(scale):
                edge_id = f"e{i}"
                # Each edge connects to 3-7 nodes
                num_nodes = 3 + (i % 5)
                for j in range(num_nodes):
                    node_id = f"n{(i * 3 + j) % (scale * 2)}"  # Ensure overlap
                    edges_data.append({"edge_id": edge_id, "node_id": node_id, "weight": 1.0})
            
            df = pl.DataFrame(edges_data)
            hg = Hypergraph(IncidenceStore(df))
            
            creation_time = time.time() - start_time
            test_metrics.record_performance("Hypergraph", f"large_scale_{scale}", creation_time, scale)
            
            # Test basic operations on large graph
            start_time = time.time()
            num_nodes = hg.num_nodes
            num_edges = hg.num_edges
            query_time = time.time() - start_time
            
            print(f"    ✅ Hypergraph {scale}: {num_nodes} nodes, {num_edges} edges, created in {creation_time:.3f}s")
            
        except Exception as e:
            test_metrics.record_error("Hypergraph", f"scalability_{scale}", type(e).__name__, str(e))
            print(f"    ❌ Hypergraph {scale} failed: {e}")
        
        # Test KnowledgeGraph scalability
        try:
            start_time = time.time()
            
            kg = KnowledgeGraph()
            for i in range(scale):
                entity_id = f"entity_{i}"
                kg.add_entity(entity_id, properties={"type": f"type_{i % 10}", "value": i})
            
            creation_time = time.time() - start_time
            test_metrics.record_performance("KnowledgeGraph", f"large_scale_{scale}", creation_time, scale)
            
            print(f"    ✅ KnowledgeGraph {scale}: created in {creation_time:.3f}s")
            
        except Exception as e:
            test_metrics.record_error("KnowledgeGraph", f"scalability_{scale}", type(e).__name__, str(e))
            print(f"    ❌ KnowledgeGraph {scale} failed: {e}")
        
        # Test Metagraph scalability
        try:
            start_time = time.time()
            
            mg = Metagraph()
            for i in range(scale):
                entity_id = f"data_{i}"
                entity_type = f"DataType_{i % 5}"
                properties = {"size": i * 100, "owner": f"team_{i % 3}"}
                mg.create_entity(entity_id, entity_type, properties)
            
            creation_time = time.time() - start_time
            test_metrics.record_performance("Metagraph", f"large_scale_{scale}", creation_time, scale)
            
            print(f"    ✅ Metagraph {scale}: created in {creation_time:.3f}s")
            
        except Exception as e:
            test_metrics.record_error("Metagraph", f"scalability_{scale}", type(e).__name__, str(e))
            print(f"    ❌ Metagraph {scale} failed: {e}")


def test_edge_cases_and_robustness():
    """Test edge cases and error handling robustness"""
    print("\n🛡️ Testing Edge Cases and Robustness...")
    
    edge_cases = [
        "empty_graphs",
        "single_entity_graphs", 
        "massive_hyperedges",
        "circular_references",
        "invalid_data_types",
        "memory_pressure",
        "concurrent_access"
    ]
    
    for case in edge_cases:
        print(f"\n  Edge Case: {case}")
        
        if case == "empty_graphs":
            # Test all graph types with no data
            try:
                hg = Hypergraph()
                assert hg.num_nodes == 0 and hg.num_edges == 0
                print(f"    ✅ Empty Hypergraph handles gracefully")
            except Exception as e:
                test_metrics.record_error("Hypergraph", "empty_graph", type(e).__name__, str(e))
            
            try:
                kg = KnowledgeGraph()
                assert kg.num_nodes == 0
                print(f"    ✅ Empty KnowledgeGraph handles gracefully")
            except Exception as e:
                test_metrics.record_error("KnowledgeGraph", "empty_graph", type(e).__name__, str(e))
        
        elif case == "massive_hyperedges":
            # Test hyperedges with many nodes
            try:
                massive_edge_data = []
                edge_id = "massive_edge"
                for i in range(1000):  # 1000 nodes in one hyperedge
                    massive_edge_data.append({"edge_id": edge_id, "node_id": f"n{i}", "weight": 1.0})
                
                df = pl.DataFrame(massive_edge_data)
                hg = Hypergraph(IncidenceStore(df))
                
                print(f"    ✅ Massive hyperedge: {hg.num_nodes} nodes in 1 edge")
            except Exception as e:
                test_metrics.record_error("Hypergraph", "massive_hyperedge", type(e).__name__, str(e))
                print(f"    ❌ Massive hyperedge failed: {e}")
        
        elif case == "invalid_data_types":
            # Test with invalid data types
            try:
                invalid_data = {"edge_id": [None, "e1"], "node_id": ["n1", None], "weight": ["invalid", 1.0]}
                df = pl.DataFrame(invalid_data)
                hg = Hypergraph(IncidenceStore(df))
                print(f"    ⚠️  Invalid data handled: {hg.num_nodes} nodes")
            except Exception as e:
                print(f"    ✅ Invalid data properly rejected: {type(e).__name__}")


@pytest.mark.asyncio
async def test_streaming_integration():
    """Test integration with streaming framework"""
    if not STREAMING_AVAILABLE:
        pytest.skip("Streaming framework not available")
    
    print("\n🌊 Testing Streaming Integration...")
    
    # Test streaming updates to each graph type
    missing_features = []
    
    try:
        # Test Hypergraph streaming updates
        hg = Hypergraph(IncidenceStore(pl.DataFrame({"edge_id": ["e1"], "node_id": ["n1"], "weight": [1.0]})))
        
        # This would integrate with the streaming framework
        stream_processor = GraphStreamProcessor(graph=hg)
        
        # Test streaming edge additions
        await stream_processor.add_edge_async("e2", ["n1", "n2"])
        print(f"    ✅ Hypergraph streaming updates available")
    
    except AttributeError:
        missing_features.append("hypergraph_streaming")
        test_metrics.record_gap("Streaming", "hypergraph_streaming", "Real-time streaming updates for Hypergraph")
    except Exception as e:
        test_metrics.record_error("Streaming", "hypergraph_streaming", type(e).__name__, str(e))
    
    if missing_features:
        print(f"    🚨 Missing streaming features: {', '.join(missing_features)}")


def test_algorithm_integration():
    """Test integration with algorithm library"""
    if not ALGORITHMS_AVAILABLE:
        print("\n🧮 Algorithm library not available, skipping algorithm tests...")
        return
    
    print("\n🧮 Testing Algorithm Integration...")
    
    # Create test graphs for algorithm testing
    hg_data = {"edge_id": ["e1", "e1", "e2", "e2", "e3"], 
               "node_id": ["n1", "n2", "n2", "n3", "n1"], 
               "weight": [1.0, 1.0, 1.5, 1.5, 2.0]}
    hg = Hypergraph(IncidenceStore(pl.DataFrame(hg_data)))
    
    # Test algorithm availability
    algorithm_tests = [
        ("centrality.degree_centrality", lambda: centrality.degree_centrality(hg)),
        ("centrality.closeness_centrality", lambda: centrality.closeness_centrality(hg)),
        ("clustering.modularity_clustering", lambda: clustering.modularity_clustering(hg))
    ]
    
    for alg_name, alg_func in algorithm_tests:
        try:
            start_time = time.time()
            result = alg_func()
            exec_time = time.time() - start_time
            
            test_metrics.record_performance("Algorithms", alg_name, exec_time, hg.num_nodes)
            print(f"    ✅ {alg_name}: {len(result)} results in {exec_time:.3f}s")
            
        except Exception as e:
            test_metrics.record_error("Algorithms", alg_name, type(e).__name__, str(e))
            print(f"    ❌ {alg_name} failed: {e}")


def run_comprehensive_analysis():
    """Run all comprehensive tests and generate analysis report"""
    print("🔬 COMPREHENSIVE GRAPH ANALYSIS SUITE")
    print("="*80)
    
    # Run all test suites
    test_advanced_hypergraph_operations()
    test_advanced_knowledge_graph_operations()
    test_advanced_hierarchical_kg_operations() 
    test_advanced_metagraph_operations()
    test_cross_graph_interoperability()
    test_scalability_and_performance()
    test_edge_cases_and_robustness()
    test_algorithm_integration()
    
    # Generate comprehensive report
    report = test_metrics.generate_report()
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "comprehensive_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📋 Analysis report saved to: {report_file}")
    return test_metrics


if __name__ == "__main__":
    # Run comprehensive analysis
    metrics = run_comprehensive_analysis()
    
    # Additional analysis
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   • Performance tests: {len(metrics.performance_data)} graph types")
    print(f"   • Functionality gaps: {sum(len(gaps) for gaps in metrics.functionality_gaps.values())} total")
    print(f"   • Error patterns: {sum(len(errors) for errors in metrics.error_patterns.values())} total")