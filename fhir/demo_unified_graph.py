"""
FHIR Unified Knowledge Graph Demonstration
==========================================

Comprehensive demonstration scripts showing the capabilities of the unified 
FHIR hierarchical knowledge graph system.

Features demonstrated:
- Building unified FHIR knowledge graphs
- Loading ontologies and data into single hierarchy
- Querying across ontology and data levels
- Semantic reasoning with FHIR data
- Cross-level relationship exploration
- Graph persistence and reconstruction
- Performance analysis and statistics
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANANT imports
from anant.kg import HierarchicalKnowledgeGraph

# FHIR imports
try:
    from .unified_graph_builder import FHIRUnifiedGraphBuilder, build_fhir_unified_graph
    from .graph_persistence import save_fhir_graph, load_fhir_graph
except ImportError:
    # Fallback for direct execution
    from unified_graph_builder import FHIRUnifiedGraphBuilder, build_fhir_unified_graph
    from graph_persistence import save_fhir_graph, load_fhir_graph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FHIRGraphDemonstration:
    """Demonstrate FHIR unified knowledge graph capabilities."""
    
    def __init__(self, 
                 schema_dir: str = "schema",
                 data_dir: str = "data/output/fhir"):
        """
        Initialize the demonstration.
        
        Args:
            schema_dir: Directory containing FHIR schema files
            data_dir: Directory containing FHIR data files
        """
        self.schema_dir = Path(schema_dir)
        self.data_dir = Path(data_dir)
        self.demos_run = []
        self.performance_stats = {}
        
        logger.info(f"Initialized FHIR Graph Demonstration")
        logger.info(f"Schema directory: {self.schema_dir}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def run_all_demonstrations(self) -> Dict[str, Any]:
        """Run all demonstration scenarios."""
        print("=" * 80)
        print("FHIR UNIFIED KNOWLEDGE GRAPH DEMONSTRATION")
        print("=" * 80)
        
        results = {
            'timestamp': time.time(),
            'demonstrations': {},
            'performance_summary': {},
            'errors': []
        }
        
        try:
            # Demo 1: Basic unified graph construction
            demo1_results = self.demo_01_basic_construction()
            results['demonstrations']['basic_construction'] = demo1_results
            
            # Demo 2: Ontology integration  
            demo2_results = self.demo_02_ontology_integration()
            results['demonstrations']['ontology_integration'] = demo2_results
            
            # Demo 3: Data loading and mapping
            demo3_results = self.demo_03_data_integration()
            results['demonstrations']['data_integration'] = demo3_results
            
            # Demo 4: Cross-level querying
            demo4_results = self.demo_04_cross_level_querying()
            results['demonstrations']['cross_level_querying'] = demo4_results
            
            # Demo 5: Semantic reasoning
            demo5_results = self.demo_05_semantic_reasoning()
            results['demonstrations']['semantic_reasoning'] = demo5_results
            
            # Demo 6: Persistence and reconstruction
            demo6_results = self.demo_06_persistence_reconstruction()
            results['demonstrations']['persistence_reconstruction'] = demo6_results
            
            # Demo 7: Performance analysis
            demo7_results = self.demo_07_performance_analysis()
            results['demonstrations']['performance_analysis'] = demo7_results
            
            # Performance summary
            results['performance_summary'] = self.generate_performance_summary()
            
        except Exception as e:
            error_msg = f"Demonstration failed: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_01_basic_construction(self) -> Dict[str, Any]:
        """Demonstrate basic unified graph construction."""
        print("\\n" + "=" * 60)
        print("DEMO 1: Basic Unified Graph Construction")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            print("Creating FHIR Unified Graph Builder...")
            builder = FHIRUnifiedGraphBuilder(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                graph_name="Demo_FHIR_Graph"
            )
            
            print(f"✓ Builder initialized: {builder.graph_name}")
            print(f"✓ Underlying HKG: {builder.unified_hkg.name}")
            
            # Create unified structure
            print("\\nCreating unified hierarchical structure...")
            structure_results = builder._create_unified_structure()
            
            print(f"✓ Structure creation status: {structure_results['status']}")
            print(f"✓ Levels created: {len(structure_results['levels_created'])}")
            
            # Show the hierarchy
            print("\\nUnified Hierarchy Levels:")
            for i, level_id in enumerate(structure_results['levels_created']):
                level_metadata = builder.unified_hkg.get_level_metadata(level_id)
                level_name = level_metadata.get('name', level_id) if level_metadata else level_id
                print(f"  {i+1}. {level_id}: {level_name}")
            
            # Collect metrics
            results['metrics'] = {
                'levels_created': len(structure_results['levels_created']),
                'total_time': time.time() - start_time,
                'graph_name': builder.graph_name
            }
            
            print(f"\\n✓ Demo 1 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 1 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_02_ontology_integration(self) -> Dict[str, Any]:
        """Demonstrate FHIR ontology integration."""
        print("\\n" + "=" * 60)
        print("DEMO 2: FHIR Ontology Integration")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            # Check if schema files exist
            schema_files = ['rim.ttl', 'fhir.ttl', 'w5.ttl']
            available_files = []
            
            for file_name in schema_files:
                file_path = self.schema_dir / file_name
                if file_path.exists():
                    available_files.append(file_name)
                    print(f"✓ Found schema file: {file_name}")
                else:
                    print(f"⚠ Schema file not found: {file_name}")
            
            if not available_files:
                print("\\n⚠ No FHIR schema files found. Creating minimal demo ontology...")
                # Create a minimal ontology for demonstration
                self._create_demo_ontology()
                available_files = ['demo_fhir.ttl']
            
            # Test ontology loading with available files
            print(f"\\nTesting ontology loading with {len(available_files)} files...")
            
            builder = FHIRUnifiedGraphBuilder(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                graph_name="Demo_Ontology_Graph"
            )
            
            builder._create_unified_structure()
            ontology_results = builder._load_ontologies()
            
            print(f"✓ Ontology loading status: {ontology_results['status']}")
            print(f"✓ Files loaded: {len(ontology_results['ontologies_loaded'])}")
            print(f"✓ Classes found: {ontology_results.get('classes_added', 0)}")
            print(f"✓ Properties found: {ontology_results.get('properties_added', 0)}")
            
            # Show ontology levels
            print("\\nOntology Levels Content:")
            ontology_levels = ['meta_ontology', 'core_ontology', 'valuesets_ontology']
            
            for level_id in ontology_levels:
                nodes = builder.unified_hkg.get_nodes_at_level(level_id)
                print(f"  {level_id}: {len(nodes)} nodes")
                
                # Show a few example nodes
                for i, node in enumerate(nodes[:3]):
                    print(f"    - {node}")
                if len(nodes) > 3:
                    print(f"    ... and {len(nodes) - 3} more")
            
            results['metrics'] = {
                'files_processed': len(ontology_results['ontologies_loaded']),
                'classes_added': ontology_results.get('classes_added', 0),
                'properties_added': ontology_results.get('properties_added', 0),
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 2 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 2 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_03_data_integration(self) -> Dict[str, Any]:
        """Demonstrate FHIR data integration."""
        print("\\n" + "=" * 60)
        print("DEMO 3: FHIR Data Integration")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            # Check for data files
            data_files = list(self.data_dir.glob("*.json")) if self.data_dir.exists() else []
            
            if not data_files:
                print("⚠ No FHIR data files found. Creating demo data...")
                self._create_demo_data()
                data_files = list(self.data_dir.glob("*.json"))
            
            print(f"Found {len(data_files)} FHIR data files:")
            for file_path in data_files[:5]:  # Show first 5
                print(f"  - {file_path.name}")
            if len(data_files) > 5:
                print(f"  ... and {len(data_files) - 5} more")
            
            # Test complete build with limited data
            print("\\nBuilding complete unified graph with data...")
            
            hkg, build_results = build_fhir_unified_graph(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                max_data_files=3,  # Limit for demo
                graph_name="Demo_Data_Integration_Graph"
            )
            
            print(f"✓ Build status: {build_results['status']}")
            
            # Show statistics
            stats = {}
            if 'statistics' in build_results:
                stats = build_results['statistics']
                print("\\nGraph Statistics:")
                print(f"  Total nodes: {stats.get('total_nodes', 0)}")
                print(f"  Total edges: {stats.get('total_edges', 0)}")
                print(f"  Total levels: {stats.get('total_levels', 0)}")
                print(f"  Cross-level relationships: {stats.get('cross_level_relationships', 0)}")
                
                if 'resource_types' in stats:
                    print("\\nResource Types Loaded:")
                    for resource_type, count in stats['resource_types'].items():
                        print(f"    {resource_type}: {count}")
            
            # Show data levels
            print("\\nData Levels Content:")
            data_levels = ['patients', 'practitioners', 'organizations', 'clinical_data', 'care_coordination']
            
            for level_id in data_levels:
                nodes = hkg.get_nodes_at_level(level_id)
                print(f"  {level_id}: {len(nodes)} nodes")
                
                # Show example nodes
                for i, node in enumerate(nodes[:2]):
                    print(f"    - {node}")
                if len(nodes) > 2:
                    print(f"    ... and {len(nodes) - 2} more")
            
            results['metrics'] = {
                'files_processed': len(data_files),
                'total_nodes': stats.get('total_nodes', 0),
                'total_edges': stats.get('total_edges', 0),
                'resource_types': len(stats.get('resource_types', {})),
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 3 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 3 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_04_cross_level_querying(self) -> Dict[str, Any]:
        """Demonstrate cross-level querying capabilities."""
        print("\\n" + "=" * 60)
        print("DEMO 4: Cross-Level Querying")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            # Build a graph for querying
            print("Building graph for querying demonstration...")
            
            hkg, build_results = build_fhir_unified_graph(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                max_data_files=2,
                graph_name="Demo_Query_Graph"
            )
            
            print(f"✓ Graph built with {hkg.num_nodes} nodes")
            
            # Demonstrate different query patterns
            print("\\nQuery Pattern 1: Level-specific node retrieval")
            for level_name in ['patients', 'clinical_data', 'core_ontology']:
                nodes = hkg.get_nodes_at_level(level_name)
                print(f"  {level_name}: {len(nodes)} nodes")
                if nodes:
                    print(f"    Example: {nodes[0]}")
            
            print("\\nQuery Pattern 2: Cross-level relationship exploration")
            if hasattr(hkg, 'cross_level_relationships') and hkg.cross_level_relationships:
                print(f"  Found {len(hkg.cross_level_relationships)} cross-level relationships")
                
                # Show a few examples
                for i, rel in enumerate(hkg.cross_level_relationships[:3]):
                    rel_type = rel.get('relationship_type', 'unknown')
                    print(f"    Relationship {i+1}: {rel_type}")
            else:
                print("  No cross-level relationships found")
            
            print("\\nQuery Pattern 3: Hierarchical navigation")
            # Show level structure
            if hasattr(hkg, 'levels'):
                print("  Available levels:")
                for level_id, level_data in hkg.levels.items():
                    level_name = level_data.get('name', level_id)
                    level_order = level_data.get('order', 0)
                    print(f"    {level_order}: {level_id} ({level_name})")
            
            print("\\nQuery Pattern 4: Semantic search (if enabled)")
            if hkg.enable_semantic_reasoning:
                # Try semantic search
                search_terms = ['Patient', 'Observation', 'Resource']
                for term in search_terms:
                    try:
                        search_results = hkg.semantic_search(term, level_ids=None)
                        print(f"  Search '{term}': {len(search_results)} results")
                    except Exception as e:
                        print(f"  Search '{term}': Error - {str(e)}")
            else:
                print("  Semantic reasoning not enabled")
            
            results['metrics'] = {
                'total_nodes_queried': hkg.num_nodes,
                'total_levels': len(hkg.levels) if hasattr(hkg, 'levels') else 0,
                'cross_level_relationships': len(hkg.cross_level_relationships) if hasattr(hkg, 'cross_level_relationships') else 0,
                'semantic_search_enabled': hkg.enable_semantic_reasoning,
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 4 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 4 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_05_semantic_reasoning(self) -> Dict[str, Any]:
        """Demonstrate semantic reasoning capabilities."""
        print("\\n" + "=" * 60)
        print("DEMO 5: Semantic Reasoning")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            print("Building graph with semantic reasoning enabled...")
            
            # Create builder with semantic reasoning
            builder = FHIRUnifiedGraphBuilder(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                graph_name="Demo_Semantic_Graph"
            )
            
            # Ensure semantic reasoning is enabled
            builder.unified_hkg.enable_semantic_reasoning = True
            
            # Build with limited data
            build_results = builder.build_unified_graph(max_data_files=2)
            hkg = builder.unified_hkg
            
            print(f"✓ Graph built with semantic reasoning enabled")
            print(f"✓ Nodes: {hkg.num_nodes}, Edges: {hkg.num_edges}")
            
            # Demonstrate semantic reasoning features
            print("\\nSemantic Reasoning Feature 1: Ontology-Data Alignment")
            
            # Check for ontology-data mappings
            mappings_created = build_results.get('phases', {}).get('mapping_creation', {}).get('mappings_created', 0)
            print(f"  Ontology-data mappings created: {mappings_created}")
            
            print("\\nSemantic Reasoning Feature 2: Cross-Level Semantic Search")
            
            # Try semantic searches
            search_terms = ['Patient', 'clinical', 'data', 'resource']
            for term in search_terms[:2]:  # Limit for demo
                try:
                    search_results = hkg.semantic_search(term)
                    print(f"  Search '{term}': {len(search_results)} semantic matches")
                    
                    # Show example results
                    for i, result in enumerate(search_results[:2]):
                        score = result.get('score', 0)
                        node_id = result.get('node_id', 'unknown')
                        print(f"    {i+1}. {node_id} (score: {score:.3f})")
                        
                except Exception as e:
                    print(f"  Search '{term}': {str(e)}")
            
            print("\\nSemantic Reasoning Feature 3: Hierarchical Inference")
            
            # Show hierarchical relationships
            if hasattr(hkg, 'levels'):
                ontology_levels = ['meta_ontology', 'core_ontology']
                data_levels = ['patients', 'clinical_data']
                
                print("  Ontology levels:")
                for level in ontology_levels:
                    if level in hkg.levels:
                        nodes = hkg.get_nodes_at_level(level)
                        print(f"    {level}: {len(nodes)} concepts")
                
                print("  Data levels:")
                for level in data_levels:
                    if level in hkg.levels:
                        nodes = hkg.get_nodes_at_level(level)
                        print(f"    {level}: {len(nodes)} instances")
            
            print("\\nSemantic Reasoning Feature 4: Relationship Inference")
            
            # Show cross-level relationships
            if hasattr(hkg, 'cross_level_relationships'):
                semantic_rels = [
                    rel for rel in hkg.cross_level_relationships 
                    if rel.get('mapping_type') == 'type_based'
                ]
                print(f"  Type-based semantic relationships: {len(semantic_rels)}")
                
                # Show examples
                for i, rel in enumerate(semantic_rels[:2]):
                    rel_type = rel.get('relationship_type', 'unknown')
                    resource_type = rel.get('resource_type', 'unknown')
                    print(f"    {i+1}. {rel_type} for {resource_type}")
            
            results['metrics'] = {
                'semantic_reasoning_enabled': True,
                'ontology_data_mappings': mappings_created,
                'semantic_relationships': len([r for r in hkg.cross_level_relationships if r.get('mapping_type')]) if hasattr(hkg, 'cross_level_relationships') else 0,
                'search_terms_tested': len(search_terms),
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 5 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 5 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_06_persistence_reconstruction(self) -> Dict[str, Any]:
        """Demonstrate graph persistence and reconstruction."""
        print("\\n" + "=" * 60)
        print("DEMO 6: Persistence and Reconstruction")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            # Build a graph to persist
            print("Building graph for persistence demonstration...")
            
            hkg, build_results = build_fhir_unified_graph(
                schema_dir=str(self.schema_dir),
                data_dir=str(self.data_dir),
                max_data_files=2,
                graph_name="Demo_Persistence_Graph"
            )
            
            original_stats = {
                'nodes': hkg.num_nodes,
                'edges': hkg.num_edges,
                'levels': len(hkg.levels) if hasattr(hkg, 'levels') else 0
            }
            
            print(f"✓ Original graph: {original_stats['nodes']} nodes, {original_stats['edges']} edges, {original_stats['levels']} levels")
            
            # Save the graph
            print("\\nSaving graph to parquet format...")
            
            save_dir = Path("demo_saved_graph")
            save_dir.mkdir(exist_ok=True)
            
            save_results = save_fhir_graph(hkg, save_dir)
            
            print(f"✓ Save status: {save_results['status']}")
            print(f"✓ Files created: {len(save_results['files_created'])}")
            for file_path in save_results['files_created']:
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                print(f"    - {Path(file_path).name}: {file_size:,} bytes")
            
            # Load the graph
            print("\\nLoading graph from parquet format...")
            
            loaded_hkg, load_results = load_fhir_graph(save_dir)
            
            print(f"✓ Load status: {load_results['status']}")
            print(f"✓ Files loaded: {len(load_results['files_loaded'])}")
            
            # Compare original and loaded
            loaded_stats = {
                'nodes': loaded_hkg.num_nodes,
                'edges': loaded_hkg.num_edges,
                'levels': len(loaded_hkg.levels) if hasattr(loaded_hkg, 'levels') else 0
            }
            
            print("\\nComparison of original vs loaded:")
            print(f"  Nodes: {original_stats['nodes']} → {loaded_stats['nodes']}")
            print(f"  Edges: {original_stats['edges']} → {loaded_stats['edges']}")
            print(f"  Levels: {original_stats['levels']} → {loaded_stats['levels']}")
            
            # Verify data integrity
            nodes_match = original_stats['nodes'] == loaded_stats['nodes']
            edges_match = original_stats['edges'] == loaded_stats['edges']
            levels_match = original_stats['levels'] == loaded_stats['levels']
            
            print(f"\\nData integrity check:")
            print(f"  Nodes match: {'✓' if nodes_match else '✗'}")
            print(f"  Edges match: {'✓' if edges_match else '✗'}")
            print(f"  Levels match: {'✓' if levels_match else '✓'}")
            
            # Show persistence statistics
            if 'statistics' in save_results:
                save_stats = save_results['statistics']
                print(f"\\nPersistence Statistics:")
                print(f"  Compression: {save_results.get('compression', 'unknown')}")
                print(f"  Total files: {save_stats.get('total_files', 0)}")
                print(f"  Total size: {save_stats.get('total_size_bytes', 0):,} bytes")
            
            results['metrics'] = {
                'original_nodes': original_stats['nodes'],
                'loaded_nodes': loaded_stats['nodes'],
                'data_integrity': nodes_match and edges_match and levels_match,
                'files_created': len(save_results['files_created']),
                'files_loaded': len(load_results['files_loaded']),
                'compression_used': save_results.get('compression', 'unknown'),
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 6 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 6 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def demo_07_performance_analysis(self) -> Dict[str, Any]:
        """Demonstrate performance analysis capabilities."""
        print("\\n" + "=" * 60)
        print("DEMO 7: Performance Analysis")
        print("=" * 60)
        
        start_time = time.time()
        results = {'status': 'success', 'metrics': {}, 'errors': []}
        
        try:
            print("Analyzing FHIR graph construction performance...")
            
            # Test with different data sizes
            data_sizes = [1, 2, 3] if self.data_dir.exists() and list(self.data_dir.glob("*.json")) else [0]
            performance_data = []
            
            for size in data_sizes:
                print(f"\\nTesting with {size} data files...")
                
                size_start = time.time()
                
                hkg, build_results = build_fhir_unified_graph(
                    schema_dir=str(self.schema_dir),
                    data_dir=str(self.data_dir),
                    max_data_files=size if size > 0 else None,
                    graph_name=f"Demo_Performance_Graph_{size}"
                )
                
                build_time = time.time() - size_start
                
                stats = build_results.get('statistics', {})
                performance_data.append({
                    'data_files': size,
                    'build_time': build_time,
                    'nodes': stats.get('total_nodes', 0),
                    'edges': stats.get('total_edges', 0),
                    'levels': stats.get('total_levels', 0),
                    'resource_types': len(stats.get('resource_types', {}))
                })
                
                print(f"  Build time: {build_time:.2f}s")
                print(f"  Nodes: {stats.get('total_nodes', 0)}")
                print(f"  Edges: {stats.get('total_edges', 0)}")
            
            # Analyze performance trends
            print("\\nPerformance Analysis:")
            
            time_ratio = 1.0
            data_ratio = 1.0
            
            if len(performance_data) > 1:
                # Calculate scaling factors
                first = performance_data[0]
                last = performance_data[-1]
                
                if first['data_files'] > 0 and last['data_files'] > 0:
                    time_ratio = last['build_time'] / first['build_time']
                    data_ratio = last['data_files'] / first['data_files']
                    node_ratio = last['nodes'] / first['nodes'] if first['nodes'] > 0 else 0
                    
                    print(f"  Time scaling: {time_ratio:.2f}x for {data_ratio:.1f}x data")
                    print(f"  Node scaling: {node_ratio:.2f}x")
                    print(f"  Time per node: {last['build_time'] / last['nodes']:.4f}s" if last['nodes'] > 0 else "  No nodes created")
            
            # Memory and efficiency analysis
            print("\\nEfficiency Metrics:")
            
            for i, data in enumerate(performance_data):
                if data['nodes'] > 0:
                    nodes_per_second = data['nodes'] / data['build_time']
                    edges_per_second = data['edges'] / data['build_time']
                    
                    print(f"  Test {i+1}: {nodes_per_second:.1f} nodes/sec, {edges_per_second:.1f} edges/sec")
            
            # Phase-by-phase analysis (if we have build results from last iteration)
            try:
                if 'build_results' in locals() and build_results.get('phases'):
                    print("\\nPhase-by-Phase Performance:")
                    
                    phases = build_results['phases']
                    
                    for phase_name, phase_data in phases.items():
                        # Extract timing if available
                        if isinstance(phase_data, dict):
                            print(f"  {phase_name}: {'✓' if phase_data.get('status') == 'success' else '✗'}")
            except:
                pass  # Skip phase analysis if not available
            
            results['metrics'] = {
                'performance_tests': len(performance_data),
                'max_nodes': max([d['nodes'] for d in performance_data]) if performance_data else 0,
                'max_build_time': max([d['build_time'] for d in performance_data]) if performance_data else 0,
                'avg_nodes_per_second': sum([d['nodes']/d['build_time'] for d in performance_data if d['build_time'] > 0]) / len([d for d in performance_data if d['build_time'] > 0]) if performance_data else 0,
                'scaling_efficiency': time_ratio / data_ratio if data_ratio > 0 else 1.0,
                'total_time': time.time() - start_time
            }
            
            print(f"\\n✓ Demo 7 completed in {results['metrics']['total_time']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Demo 7 error: {str(e)}"
            results['status'] = 'error'
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        return {
            'demonstrations_completed': len(self.demos_run),
            'total_demo_time': sum([demo.get('total_time', 0) for demo in self.performance_stats.values()]),
            'success_rate': len([demo for demo in self.performance_stats.values() if demo.get('status') == 'success']) / len(self.performance_stats) if self.performance_stats else 0
        }
    
    def _create_demo_ontology(self):
        """Create minimal demo ontology files."""
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        demo_ontology = '''@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix fhir: <http://hl7.org/fhir/> .

fhir:Resource a owl:Class ;
    rdfs:label "Resource" ;
    rdfs:comment "Base FHIR Resource" .

fhir:Patient a owl:Class ;
    rdfs:subClassOf fhir:Resource ;
    rdfs:label "Patient" ;
    rdfs:comment "Patient resource" .

fhir:Observation a owl:Class ;
    rdfs:subClassOf fhir:Resource ;
    rdfs:label "Observation" ;
    rdfs:comment "Observation resource" .
'''
        
        with open(self.schema_dir / "demo_fhir.ttl", 'w') as f:
            f.write(demo_ontology)
    
    def _create_demo_data(self):
        """Create minimal demo data files."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        demo_bundle = {
            "resourceType": "Bundle",
            "id": "demo-bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "demo-patient-1",
                        "name": [{"family": "Demo", "given": ["Patient"]}],
                        "gender": "unknown",
                        "birthDate": "2000-01-01"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "demo-observation-1",
                        "status": "final",
                        "code": {"text": "Demo observation"},
                        "subject": {"reference": "Patient/demo-patient-1"},
                        "valueString": "Demo value"
                    }
                }
            ]
        }
        
        with open(self.data_dir / "demo_data.json", 'w') as f:
            json.dump(demo_bundle, f, indent=2)


def main():
    """Run the complete FHIR demonstration."""
    print("FHIR Unified Knowledge Graph Demonstration")
    print("==========================================")
    
    # Initialize demonstration
    demo = FHIRGraphDemonstration()
    
    # Run all demonstrations
    results = demo.run_all_demonstrations()
    
    # Final summary
    print("\\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    print(f"Total demonstrations run: {len(results['demonstrations'])}")
    
    successful_demos = [name for name, result in results['demonstrations'].items() 
                       if result.get('status') == 'success']
    failed_demos = [name for name, result in results['demonstrations'].items() 
                   if result.get('status') != 'success']
    
    print(f"Successful: {len(successful_demos)}")
    print(f"Failed: {len(failed_demos)}")
    
    if successful_demos:
        print("\\nSuccessful demonstrations:")
        for demo_name in successful_demos:
            print(f"  ✓ {demo_name}")
    
    if failed_demos:
        print("\\nFailed demonstrations:")
        for demo_name in failed_demos:
            print(f"  ✗ {demo_name}")
    
    # Performance summary
    if 'performance_summary' in results:
        perf = results['performance_summary']
        print(f"\\nOverall Performance:")
        print(f"  Total time: {perf.get('total_demo_time', 0):.2f} seconds")
        print(f"  Success rate: {perf.get('success_rate', 0)*100:.1f}%")
    
    if results.get('errors'):
        print(f"\\nErrors encountered: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3
            print(f"  - {error}")
    
    print("\\n" + "=" * 80)
    print("FHIR UNIFIED KNOWLEDGE GRAPH DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()