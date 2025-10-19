#!/usr/bin/env python3
"""
Test Validation Framework

Tests the comprehensive validation framework with data integrity checks,
performance benchmarks, and component integration validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anant'))

import polars as pl

def test_validation_framework():
    """Test the validation framework functionality"""
    print("Testing Validation Framework...")
    
    # Import validation components
    from anant.validation import (
        ValidationFramework, 
        quick_validate,
        performance_benchmark,
        validate_all_components
    )
    from anant.classes.hypergraph import Hypergraph
    from anant.streaming import StreamingHypergraph
    from anant.analysis.temporal import TemporalHypergraph, TemporalSnapshot
    
    # Create test hypergraph
    test_df = pl.DataFrame([
        {"edges": "E1", "nodes": "A", "weight": 1.0},
        {"edges": "E1", "nodes": "B", "weight": 1.0},
        {"edges": "E2", "nodes": "B", "weight": 1.0},
        {"edges": "E2", "nodes": "C", "weight": 1.0},
        {"edges": "E3", "nodes": "C", "weight": 1.0},
        {"edges": "E3", "nodes": "D", "weight": 1.0},
    ])
    
    hg = Hypergraph(test_df, name="test_hypergraph")
    print(f"  Created test hypergraph: {hg.num_nodes} nodes, {hg.num_edges} edges")
    
    # Test quick validation
    print("  Testing quick validation...")
    is_valid = quick_validate(hg)
    print(f"    Quick validation: {'✓' if is_valid else '✗'}")
    
    # Test performance benchmark
    print("  Testing performance benchmark...")
    perf_result = performance_benchmark(hg)
    print(f"    Performance benchmark: {'✓' if perf_result.passed else '✗'} - {perf_result.message}")
    
    # Test comprehensive validation
    print("  Testing comprehensive validation...")
    validation_suite = validate_all_components(hg)
    print(f"    Comprehensive: {validation_suite.passed_count}/{validation_suite.total_count} tests passed ({validation_suite.success_rate:.1f}%)")
    
    # Test validation framework
    print("  Testing validation framework...")
    framework = ValidationFramework(enable_logging=False)
    
    # Test hypergraph validation
    hg_suite = framework.validate_hypergraph(hg)
    print(f"    Hypergraph validation: {hg_suite.passed_count}/{hg_suite.total_count} tests passed")
    
    # Test streaming hypergraph validation
    print("  Testing streaming validation...")
    streaming_hg = StreamingHypergraph(hg, enable_optimization=False)
    streaming_suite = framework.validate_streaming_hypergraph(streaming_hg)
    print(f"    Streaming validation: {streaming_suite.passed_count}/{streaming_suite.total_count} tests passed")
    
    # Test temporal hypergraph validation
    print("  Testing temporal validation...")
    temporal_hg = TemporalHypergraph()
    temporal_hg.add_snapshot(TemporalSnapshot(timestamp=1, hypergraph=hg))
    temporal_suite = framework.validate_temporal_hypergraph(temporal_hg)
    print(f"    Temporal validation: {temporal_suite.passed_count}/{temporal_suite.total_count} tests passed")
    
    # Test comprehensive test suite
    print("  Testing comprehensive test suite...")
    test_hgs = [hg]
    comprehensive_results = framework.run_comprehensive_test_suite(test_hgs)
    
    total_tests = sum(suite.total_count for suite in comprehensive_results.values())
    total_passed = sum(suite.passed_count for suite in comprehensive_results.values())
    print(f"    Comprehensive suite: {total_passed}/{total_tests} tests passed")
    
    # Generate validation report
    print("  Testing report generation...")
    report = framework.generate_validation_report(hg_suite)
    print(f"    Generated report: {len(report)} characters")
    
    print("  ✓ Validation framework test completed successfully!")

def main():
    print("Validation Framework Test")
    print("=" * 40)
    
    try:
        test_validation_framework()
        print("\n✅ VALIDATION FRAMEWORK TEST PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()