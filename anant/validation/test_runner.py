"""
Comprehensive Validation Test Runner

Master test runner that orchestrates all validation components:
- Automated algorithm testing
- Performance benchmarking  
- Stress testing
- Integration validation
- Report generation
"""

import polars as pl
import numpy as np
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from ..classes.hypergraph import Hypergraph
from . import ValidationSuite, ValidationResult, ValidationFramework
from .automated_testing import AlgorithmTester, run_all_tests
from .performance_benchmarks import PerformanceBenchmark, ScalabilityTester
from .stress_testing import StressTester


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    total_execution_time: float
    
    # Category results
    algorithm_tests: Optional[ValidationResult] = None
    performance_tests: Optional[ValidationResult] = None  
    stress_tests: Optional[ValidationResult] = None
    integration_tests: Optional[ValidationResult] = None
    
    # Summary statistics
    performance_summary: Dict[str, Any] = None
    memory_summary: Dict[str, Any] = None
    scalability_summary: Dict[str, Any] = None
    
    # Issues and recommendations
    critical_issues: List[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None


class ComprehensiveValidationRunner:
    """Master validation runner for all Anant components"""
    
    def __init__(self, 
                 enable_stress_testing: bool = True,
                 enable_performance_testing: bool = True,
                 enable_scalability_testing: bool = True,
                 max_test_nodes: int = 500):
        """
        Initialize comprehensive validation runner
        
        Args:
            enable_stress_testing: Whether to run stress tests
            enable_performance_testing: Whether to run performance benchmarks
            enable_scalability_testing: Whether to run scalability analysis
            max_test_nodes: Maximum nodes for testing (controls test intensity)
        """
        self.enable_stress_testing = enable_stress_testing
        self.enable_performance_testing = enable_performance_testing
        self.enable_scalability_testing = enable_scalability_testing
        self.max_test_nodes = max_test_nodes
        
        # Initialize validators
        self.framework = ValidationFramework()
        self.algorithm_tester = AlgorithmTester()
        
        if enable_performance_testing:
            self.performance_benchmark = PerformanceBenchmark()
            
        if enable_scalability_testing:
            self.scalability_tester = ScalabilityTester()
            
        if enable_stress_testing:
            self.stress_tester = StressTester()
    
    def run_complete_validation(self) -> ValidationReport:
        """Run complete validation suite and generate comprehensive report"""
        
        print("🔍 STARTING COMPREHENSIVE ANANT VALIDATION")
        print("=" * 60)
        
        start_time = time.perf_counter()
        validation_results = {}
        
        # 1. Algorithm Testing
        print("\n📊 Running Algorithm Tests...")
        algorithm_result = self.algorithm_tester.validate()
        validation_results['algorithms'] = algorithm_result
        
        self._print_result_summary("Algorithm Tests", algorithm_result)
        
        # 2. Performance Testing
        if self.enable_performance_testing:
            print("\n⚡ Running Performance Tests...")
            test_hg = self._create_performance_test_hypergraph()
            performance_result = self.performance_benchmark.validate(test_hg)
            validation_results['performance'] = performance_result
            
            self._print_result_summary("Performance Tests", performance_result)
        
        # 3. Scalability Testing
        if self.enable_scalability_testing:
            print("\n📈 Running Scalability Tests...")
            scalability_result = self.scalability_tester.validate()
            validation_results['scalability'] = scalability_result
            
            self._print_result_summary("Scalability Tests", scalability_result)
        
        # 4. Stress Testing
        if self.enable_stress_testing:
            print("\n🔥 Running Stress Tests...")
            stress_result = self.stress_tester.validate()
            validation_results['stress'] = stress_result
            
            self._print_result_summary("Stress Tests", stress_result)
        
        # 5. Integration Testing
        print("\n🔗 Running Integration Tests...")
        integration_result = self._run_integration_tests()
        validation_results['integration'] = integration_result
        
        self._print_result_summary("Integration Tests", integration_result)
        
        # Generate comprehensive report
        total_time = time.perf_counter() - start_time
        report = self._generate_report(validation_results, total_time)
        
        # Print final summary
        self._print_final_summary(report)
        
        return report
    
    def run_quick_validation(self) -> ValidationReport:
        """Run quick validation (essential tests only)"""
        
        print("🚀 RUNNING QUICK ANANT VALIDATION")
        print("=" * 40)
        
        start_time = time.perf_counter()
        validation_results = {}
        
        # Run only essential tests
        algorithm_result = self.algorithm_tester.validate()
        validation_results['algorithms'] = algorithm_result
        
        # Basic integration test
        test_hg = self._create_test_hypergraph()
        integration_suite = self.framework.validate_hypergraph(test_hg, ['data_integrity', 'integration'])
        
        # Convert suite to single result for consistency
        integration_result = ValidationResult(
            test_name="Quick Integration Test",
            passed=integration_suite.success_rate == 100.0,
            message=f"Integration: {integration_suite.passed_count}/{integration_suite.total_count} passed",
            execution_time=integration_suite.total_execution_time,
            details={'suite': integration_suite}
        )
        validation_results['integration'] = integration_result
        
        total_time = time.perf_counter() - start_time
        report = self._generate_report(validation_results, total_time)
        
        self._print_final_summary(report)
        
        return report
    
    def validate_specific_algorithms(self, 
                                   algorithms: List[str],
                                   test_hypergraph: Optional[Hypergraph] = None) -> ValidationReport:
        """Validate specific algorithms only"""
        
        print(f"🎯 VALIDATING SPECIFIC ALGORITHMS: {', '.join(algorithms)}")
        print("=" * 50)
        
        start_time = time.perf_counter()
        validation_results = {}
        
        # Use provided hypergraph or create test one
        test_hg = test_hypergraph or self._create_test_hypergraph()
        
        # Performance benchmark on specific algorithms
        if self.enable_performance_testing:
            performance_benchmark = PerformanceBenchmark(target_algorithms=algorithms)
            performance_result = performance_benchmark.validate(test_hg)
            validation_results['performance'] = performance_result
        
        # Scalability testing on specific algorithms
        if self.enable_scalability_testing:
            scalability_tester = ScalabilityTester(target_algorithms=algorithms)
            scalability_result = scalability_tester.validate()
            validation_results['scalability'] = scalability_result
        
        total_time = time.perf_counter() - start_time
        report = self._generate_report(validation_results, total_time)
        
        self._print_final_summary(report)
        
        return report
    
    def _create_test_hypergraph(self) -> Hypergraph:
        """Create standard test hypergraph"""
        hg = Hypergraph()
        hg.add_edge("e1", ["n1", "n2", "n3"])
        hg.add_edge("e2", ["n2", "n4", "n5"])
        hg.add_edge("e3", ["n1", "n4"])
        hg.add_edge("e4", ["n3", "n5", "n6"])
        hg.name = "standard_test_graph"
        return hg
    
    def _create_performance_test_hypergraph(self) -> Hypergraph:
        """Create hypergraph for performance testing"""
        hg = Hypergraph()
        
        # Create medium-sized hypergraph
        num_nodes = min(100, self.max_test_nodes // 5)
        num_edges = num_nodes // 2
        
        nodes = [f"n{i}" for i in range(num_nodes)]
        
        for i in range(num_edges):
            # Create edges of varying sizes
            edge_size = 3 + (i % 4)  # Edge sizes 3-6
            start_idx = (i * 2) % num_nodes
            edge_nodes = []
            
            for j in range(edge_size):
                node_idx = (start_idx + j) % num_nodes
                edge_nodes.append(nodes[node_idx])
            
            hg.add_edge(f"e{i}", edge_nodes)
        
        hg.name = "performance_test_graph"
        return hg
    
    def _run_integration_tests(self) -> ValidationResult:
        """Run integration tests across all components"""
        
        start_time = time.perf_counter()
        
        try:
            test_hg = self._create_test_hypergraph()
            
            # Use existing integration validator
            suite = self.framework.validate_hypergraph(test_hg)
            
            # Convert suite to single result
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Integration Testing",
                passed=suite.success_rate == 100.0,
                message=f"Integration tests: {suite.passed_count}/{suite.total_count} passed ({suite.success_rate:.1f}%)",
                execution_time=execution_time,
                details={
                    'suite_results': suite.results,
                    'success_rate': suite.success_rate,
                    'total_tests': suite.total_count
                }
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Integration Testing",
                passed=False,
                message=f"Integration testing failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _generate_report(self, 
                        validation_results: Dict[str, ValidationResult], 
                        total_time: float) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        # Calculate overall statistics
        all_results = list(validation_results.values())
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Extract summaries
        performance_summary = {}
        memory_summary = {}
        scalability_summary = {}
        
        if 'performance' in validation_results:
            perf_result = validation_results['performance']
            performance_summary = perf_result.performance_metrics
        
        if 'stress' in validation_results:
            stress_result = validation_results['stress']
            if 'memory_results' in stress_result.details:
                memory_summary = stress_result.details['memory_results']
        
        if 'scalability' in validation_results:
            scal_result = validation_results['scalability']
            scalability_summary = scal_result.details
        
        # Collect issues and recommendations
        critical_issues = []
        warnings = []
        recommendations = []
        
        for category, result in validation_results.items():
            if not result.passed:
                critical_issues.append(f"{category}: {result.message}")
            
            # Extract issues from details
            if result.details and 'issues' in result.details:
                for issue in result.details['issues']:
                    if 'error' in issue.lower() or 'failed' in issue.lower():
                        critical_issues.append(f"{category}: {issue}")
                    else:
                        warnings.append(f"{category}: {issue}")
        
        # Generate recommendations
        if performance_summary:
            avg_time = performance_summary.get('average_execution_time', 0)
            if avg_time > 5.0:
                recommendations.append("Consider optimizing algorithm performance - average execution time is high")
        
        if memory_summary:
            peak_memory = memory_summary.get('peak_memory_mb', 0)
            if peak_memory > 1000:
                recommendations.append("Monitor memory usage - peak memory consumption is high")
        
        return ValidationReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            total_execution_time=total_time,
            algorithm_tests=validation_results.get('algorithms'),
            performance_tests=validation_results.get('performance'),
            stress_tests=validation_results.get('stress'),
            integration_tests=validation_results.get('integration'),
            performance_summary=performance_summary,
            memory_summary=memory_summary,
            scalability_summary=scalability_summary,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _print_result_summary(self, category: str, result: ValidationResult):
        """Print summary of a validation result"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"   {status} {category}: {result.message}")
        print(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
        
        if result.performance_metrics:
            print("   📊 Performance metrics:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    print(f"      {metric}: {value:.3f}")
                else:
                    print(f"      {metric}: {value}")
    
    def _print_final_summary(self, report: ValidationReport):
        """Print final validation summary"""
        
        print(f"\n🎉 VALIDATION COMPLETE")
        print("=" * 40)
        print(f"📊 Results: {report.passed_tests}/{report.total_tests} tests passed ({report.success_rate:.1f}%)")
        print(f"⏱️  Total time: {report.total_execution_time:.2f}s")
        
        if report.critical_issues:
            print(f"\n❌ Critical Issues ({len(report.critical_issues)}):")
            for issue in report.critical_issues[:5]:  # Show first 5
                print(f"   • {issue}")
        
        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings[:3]:  # Show first 3
                print(f"   • {warning}")
        
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"   • {rec}")
        
        # Overall status
        if report.success_rate >= 95:
            print("\n🟢 VALIDATION STATUS: EXCELLENT")
        elif report.success_rate >= 80:
            print("\n🟡 VALIDATION STATUS: GOOD")  
        else:
            print("\n🔴 VALIDATION STATUS: NEEDS ATTENTION")
    
    def save_report(self, report: ValidationReport, filepath: str):
        """Save validation report to file"""
        
        # Convert report to serializable format
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'summary': {
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'success_rate': report.success_rate,
                'total_execution_time': report.total_execution_time
            },
            'results': {
                'algorithms': self._serialize_result(report.algorithm_tests),
                'performance': self._serialize_result(report.performance_tests),
                'stress': self._serialize_result(report.stress_tests),
                'integration': self._serialize_result(report.integration_tests)
            },
            'summaries': {
                'performance': report.performance_summary,
                'memory': report.memory_summary,
                'scalability': report.scalability_summary
            },
            'issues': {
                'critical': report.critical_issues,
                'warnings': report.warnings,
                'recommendations': report.recommendations
            }
        }
        
        # Save to file
        Path(filepath).write_text(json.dumps(report_data, indent=2))
        print(f"📝 Report saved to: {filepath}")
    
    def _serialize_result(self, result: Optional[ValidationResult]) -> Optional[Dict]:
        """Convert ValidationResult to serializable dict"""
        if result is None:
            return None
        
        return {
            'test_name': result.test_name,
            'passed': result.passed,
            'message': result.message,
            'execution_time': result.execution_time,
            'performance_metrics': result.performance_metrics,
            'details_summary': str(result.details) if result.details else None
        }


# Convenience functions
def run_comprehensive_validation(max_nodes: int = 500) -> ValidationReport:
    """Run comprehensive validation with all components"""
    runner = ComprehensiveValidationRunner(max_test_nodes=max_nodes)
    return runner.run_complete_validation()


def run_quick_validation() -> ValidationReport:
    """Run quick validation (essential tests only)"""
    runner = ComprehensiveValidationRunner()
    return runner.run_quick_validation()


def validate_algorithms(algorithms: List[str], 
                       test_hypergraph: Optional[Hypergraph] = None) -> ValidationReport:
    """Validate specific algorithms"""
    runner = ComprehensiveValidationRunner()
    return runner.validate_specific_algorithms(algorithms, test_hypergraph)


__all__ = [
    'ValidationReport',
    'ComprehensiveValidationRunner',
    'run_comprehensive_validation',
    'run_quick_validation', 
    'validate_algorithms'
]