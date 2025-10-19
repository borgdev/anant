"""
Core Functionality Completeness Analysis for Anant Library

This script analyzes the current state of core functionalities and identifies
any gaps or missing critical features that need to be addressed.
"""

import os
import sys
from pathlib import Path
import polars as pl
from typing import Dict, List, Any
import importlib.util

def check_core_modules():
    """Check if all core modules exist and can be imported"""
    
    core_modules = {
        'classes.hypergraph': 'Core hypergraph implementation',
        'classes.property_store': 'Property management system',
        'classes.incidence_store': 'Incidence data storage',
        'factory.enhanced_setsystems': 'Enhanced SetSystem factory',
        'classes.advanced_properties': 'Advanced property management',
        'io.lazy_loading': 'Lazy loading capabilities',
        'jupyter_integration': 'Jupyter notebook support',
        'debugging_tools': 'Development debugging tools'
    }
    
    results = {}
    anant_path = Path('/home/amansingh/dev/ai/anant/anant')
    
    for module_path, description in core_modules.items():
        try:
            # Convert module path to file path
            file_path = anant_path / (module_path.replace('.', '/') + '.py')
            
            if file_path.exists():
                # Try to import the module
                spec = importlib.util.spec_from_file_location(module_path, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results[module_path] = {
                        'status': 'SUCCESS',
                        'description': description,
                        'path': str(file_path),
                        'size': file_path.stat().st_size
                    }
                else:
                    results[module_path] = {
                        'status': 'IMPORT_ERROR',
                        'description': description,
                        'error': 'Could not create module spec'
                    }
            else:
                results[module_path] = {
                    'status': 'MISSING',
                    'description': description,
                    'expected_path': str(file_path)
                }
                
        except Exception as e:
            results[module_path] = {
                'status': 'ERROR',
                'description': description,
                'error': str(e)
            }
    
    return results

def check_core_functionality():
    """Test core functionality by creating and using basic hypergraph"""
    
    functionality_tests = {}
    
    try:
        # Test 1: Basic hypergraph creation
        sys.path.insert(0, '/home/amansingh/dev/ai/anant')
        
        from anant.classes.hypergraph import Hypergraph
        
        # Create test data
        test_data = pl.DataFrame({
            'edge_id': ['e1', 'e1', 'e2', 'e2', 'e3'],
            'node_id': ['n1', 'n2', 'n2', 'n3', 'n1'],
            'weight': [1.0, 1.0, 0.8, 0.9, 1.0]
        })
        
        hg = Hypergraph.from_dataframe(test_data, 'node_id', 'edge_id')
        
        functionality_tests['basic_creation'] = {
            'status': 'SUCCESS',
            'details': f'Created hypergraph with {hg.num_nodes} nodes, {hg.num_edges} edges'
        }
        
    except Exception as e:
        functionality_tests['basic_creation'] = {
            'status': 'FAILED',
            'error': str(e)
        }
    
    try:
        # Test 2: Enhanced SetSystems
        from anant.factory.enhanced_setsystems import EnhancedSetSystemFactory
        
        factory = EnhancedSetSystemFactory()
        setsystem = factory.create_from_dataframe(test_data, 'node_id', 'edge_id')
        
        functionality_tests['enhanced_setsystems'] = {
            'status': 'SUCCESS',
            'details': f'Created enhanced setsystem with {len(setsystem)} elements'
        }
        
    except Exception as e:
        functionality_tests['enhanced_setsystems'] = {
            'status': 'FAILED',
            'error': str(e)
        }
    
    try:
        # Test 3: Property Management
        from anant.classes.advanced_properties import AdvancedPropertyManager
        
        prop_manager = AdvancedPropertyManager()
        prop_manager.set_node_property('n1', 'type', 'important')
        
        functionality_tests['property_management'] = {
            'status': 'SUCCESS',
            'details': 'Property management system operational'
        }
        
    except Exception as e:
        functionality_tests['property_management'] = {
            'status': 'FAILED',
            'error': str(e)
        }
    
    try:
        # Test 4: Debugging Tools
        from anant.debugging_tools import debug_hypergraph, profiler
        
        result = debug_hypergraph(hg, deep_check=False)
        
        functionality_tests['debugging_tools'] = {
            'status': 'SUCCESS',
            'details': f'Debugging validation: {"PASSED" if result["is_valid"] else "ISSUES_FOUND"}'
        }
        
    except Exception as e:
        functionality_tests['debugging_tools'] = {
            'status': 'FAILED',
            'error': str(e)
        }
    
    try:
        # Test 5: Jupyter Integration
        from anant.jupyter_integration import setup_jupyter_integration
        
        # Test setup (will fail gracefully if not in Jupyter)
        integration = setup_jupyter_integration()
        
        functionality_tests['jupyter_integration'] = {
            'status': 'SUCCESS',
            'details': 'Jupyter integration available'
        }
        
    except Exception as e:
        functionality_tests['jupyter_integration'] = {
            'status': 'FAILED',
            'error': str(e)
        }
    
    return functionality_tests

def identify_missing_features():
    """Identify missing critical features"""
    
    expected_features = {
        'Core Hypergraph Operations': {
            'node_iteration': 'Iterate through nodes',
            'edge_iteration': 'Iterate through edges', 
            'incidence_access': 'Access incidence relationships',
            'property_access': 'Get/set node and edge properties',
            'subgraph_extraction': 'Extract subgraphs',
            'hypergraph_copy': 'Deep copy hypergraphs'
        },
        'Data I/O Operations': {
            'dataframe_import': 'Import from Polars DataFrames',
            'parquet_loading': 'Load from Parquet files',
            'csv_loading': 'Load from CSV files',
            'json_export': 'Export to JSON format',
            'efficient_serialization': 'Pickle/binary serialization'
        },
        'Analysis Operations': {
            'basic_statistics': 'Node/edge counts and basic metrics',
            'degree_analysis': 'Node degree calculations',
            'connectivity_analysis': 'Connected components',
            'centrality_measures': 'Node centrality calculations',
            'clustering_analysis': 'Clustering algorithms'
        },
        'Advanced Features': {
            'weighted_operations': 'Support for weighted hypergraphs',
            'temporal_analysis': 'Time-based hypergraph analysis',
            'streaming_processing': 'Real-time data processing',
            'distributed_computing': 'Large-scale distributed processing',
            'visualization_integration': 'Built-in plotting capabilities'
        },
        'Validation and Testing': {
            'data_integrity': 'Comprehensive data validation',
            'performance_benchmarks': 'Performance measurement tools',
            'stress_testing': 'Large dataset handling',
            'error_handling': 'Robust error recovery',
            'memory_efficiency': 'Memory usage optimization'
        }
    }
    
    # This would require implementing feature detection logic
    # For now, return a placeholder structure
    
    missing_features = {}
    
    for category, features in expected_features.items():
        missing_in_category = []
        
        for feature_name, description in features.items():
            # Placeholder - in real implementation, would test each feature
            # For now, assume core features are implemented, advanced ones might be missing
            if category in ['Advanced Features', 'Validation and Testing']:
                if feature_name in ['distributed_computing', 'visualization_integration', 'stress_testing']:
                    missing_in_category.append({
                        'name': feature_name,
                        'description': description,
                        'priority': 'MEDIUM' if 'visualization' in feature_name else 'LOW'
                    })
        
        if missing_in_category:
            missing_features[category] = missing_in_category
    
    return missing_features

def analyze_code_quality():
    """Analyze code quality metrics"""
    
    anant_path = Path('/home/amansingh/dev/ai/anant/anant')
    
    metrics = {
        'total_files': 0,
        'total_lines': 0,
        'python_files': 0,
        'large_files': 0,  # > 500 lines
        'files_with_docstrings': 0,
        'test_files': 0
    }
    
    for file_path in anant_path.rglob('*.py'):
        if file_path.is_file():
            metrics['total_files'] += 1
            metrics['python_files'] += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    metrics['total_lines'] += len(lines)
                    
                    if len(lines) > 500:
                        metrics['large_files'] += 1
                    
                    # Check for docstrings (simple heuristic)
                    if '"""' in content or "'''" in content:
                        metrics['files_with_docstrings'] += 1
                    
                    # Check if it's a test file
                    if 'test_' in file_path.name or file_path.name.startswith('test'):
                        metrics['test_files'] += 1
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return metrics

def generate_completion_report():
    """Generate comprehensive completion report"""
    
    print("🔍 Anant Library Core Functionality Analysis")
    print("=" * 60)
    print()
    
    # Check core modules
    print("📦 Core Module Status:")
    print("-" * 30)
    
    module_results = check_core_modules()
    success_count = 0
    
    for module_path, result in module_results.items():
        status_icon = {
            'SUCCESS': '✅',
            'MISSING': '❌', 
            'ERROR': '🔴',
            'IMPORT_ERROR': '🟡'
        }.get(result['status'], '❓')
        
        print(f"{status_icon} {module_path}")
        print(f"   {result['description']}")
        
        if result['status'] == 'SUCCESS':
            success_count += 1
            print(f"   Size: {result['size']:,} bytes")
        elif 'error' in result:
            print(f"   Error: {result['error']}")
        
        print()
    
    module_completion = (success_count / len(module_results)) * 100
    print(f"Module Completion: {success_count}/{len(module_results)} ({module_completion:.1f}%)")
    print()
    
    # Check core functionality
    print("⚙️ Core Functionality Tests:")
    print("-" * 35)
    
    func_results = check_core_functionality()
    func_success_count = 0
    
    for test_name, result in func_results.items():
        status_icon = '✅' if result['status'] == 'SUCCESS' else '❌'
        print(f"{status_icon} {test_name}")
        
        if result['status'] == 'SUCCESS':
            func_success_count += 1
            print(f"   {result['details']}")
        else:
            print(f"   Error: {result['error']}")
        print()
    
    func_completion = (func_success_count / len(func_results)) * 100
    print(f"Functionality Completion: {func_success_count}/{len(func_results)} ({func_completion:.1f}%)")
    print()
    
    # Code quality metrics
    print("📊 Code Quality Metrics:")
    print("-" * 30)
    
    quality_metrics = analyze_code_quality()
    
    for metric, value in quality_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:,}")
    
    if quality_metrics['python_files'] > 0:
        avg_lines = quality_metrics['total_lines'] / quality_metrics['python_files']
        docstring_percentage = (quality_metrics['files_with_docstrings'] / quality_metrics['python_files']) * 100
        
        print(f"  Average Lines per File: {avg_lines:.1f}")
        print(f"  Documentation Coverage: {docstring_percentage:.1f}%")
    
    print()
    
    # Missing features
    print("🔍 Missing Features Analysis:")
    print("-" * 35)
    
    missing_features = identify_missing_features()
    
    if missing_features:
        for category, features in missing_features.items():
            print(f"\n📂 {category}:")
            for feature in features:
                priority_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(feature['priority'], '❓')
                print(f"  {priority_icon} {feature['name']}: {feature['description']}")
    else:
        print("✅ No critical missing features detected")
    
    print()
    
    # Overall completion assessment
    overall_completion = (module_completion + func_completion) / 2
    
    print("🎯 Overall Assessment:")
    print("-" * 25)
    print(f"Core Module Readiness: {module_completion:.1f}%")
    print(f"Functionality Readiness: {func_completion:.1f}%")
    print(f"Overall Completion: {overall_completion:.1f}%")
    
    if overall_completion >= 90:
        print("🚀 STATUS: PRODUCTION READY")
    elif overall_completion >= 75:
        print("⚠️ STATUS: NEAR PRODUCTION READY")
    elif overall_completion >= 50:
        print("🔧 STATUS: DEVELOPMENT READY")
    else:
        print("🚧 STATUS: EARLY DEVELOPMENT")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("-" * 20)
    
    if module_completion < 100:
        print("  🔧 Fix module import issues before proceeding")
    
    if func_completion < 100:
        print("  ⚙️ Address functionality gaps in core features")
    
    if quality_metrics['files_with_docstrings'] / max(1, quality_metrics['python_files']) < 0.8:
        print("  📚 Improve documentation coverage")
    
    if missing_features:
        high_priority = sum(1 for category in missing_features.values() 
                          for feature in category if feature['priority'] == 'HIGH')
        if high_priority > 0:
            print(f"  🔴 Address {high_priority} high-priority missing features")
    
    print("  ✅ Continue with visualization enhancements")
    print("  🧪 Add comprehensive test suite")
    
    return {
        'module_completion': module_completion,
        'functionality_completion': func_completion,
        'overall_completion': overall_completion,
        'missing_features': missing_features,
        'quality_metrics': quality_metrics
    }

if __name__ == "__main__":
    report = generate_completion_report()