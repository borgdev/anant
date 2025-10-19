"""
Validation Runner Module

Orchestrates validation across different modules including:
- Coordinated test execution
- Result aggregation
- Report generation
- Summary statistics
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from ..classes.hypergraph import Hypergraph
from .data_integrity import DataIntegrityValidator, ValidationResult
from .performance_validation import PerformanceBenchmarkValidator
from .integration_testing import ComponentIntegrationValidator, AlgorithmIntegrationTester
from .functional_validator import FunctionalValidator


@dataclass
class ValidationSuite:
    """Collection of validation results"""
    name: str
    results: List[ValidationResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def passed_count(self) -> int:
        """Count of passed tests"""
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        """Count of failed tests"""
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total_count(self) -> int:
        """Total test count"""
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_count == 0:
            return 0.0
        return (self.passed_count / self.total_count) * 100
    
    @property
    def total_execution_time(self) -> float:
        """Total execution time for all tests"""
        return sum(r.execution_time for r in self.results)


class ValidationRunner:
    """Orchestrates comprehensive validation across multiple modules"""
    
    def __init__(self, performance_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize validation runner
        
        Args:
            performance_thresholds: Custom performance thresholds
        """
        self.performance_thresholds = performance_thresholds
        
        # Initialize validators
        self.validators = {
            'data_integrity': DataIntegrityValidator(),
            'performance': PerformanceBenchmarkValidator(performance_thresholds),
            'component_integration': ComponentIntegrationValidator(),
            'algorithm_integration': AlgorithmIntegrationTester(),
            'functional': FunctionalValidator()
        }
        
        # Results storage
        self.validation_history: List[ValidationSuite] = []
    
    def run_validation_suite(self, 
                           hg: Hypergraph, 
                           validators: Optional[List[str]] = None,
                           suite_name: Optional[str] = None) -> ValidationSuite:
        """
        Run comprehensive validation on a hypergraph
        
        Args:
            hg: Hypergraph to validate
            validators: List of validator names to run (None = all)
            suite_name: Custom name for the validation suite
            
        Returns:
            ValidationSuite with all results
        """
        suite_name = suite_name or f"Hypergraph Validation - {getattr(hg, 'name', 'Unnamed')}"
        suite = ValidationSuite(name=suite_name)
        suite.start_time = datetime.now()
        
        # Determine which validators to run
        validator_names = validators or list(self.validators.keys())
        
        print(f"🧪 Starting validation suite: {suite_name}")
        print(f"   Running {len(validator_names)} validators...")
        
        # Run each validator
        for validator_name in validator_names:
            if validator_name not in self.validators:
                print(f"   ⚠️  Unknown validator: {validator_name}")
                continue
            
            validator = self.validators[validator_name]
            print(f"   🔍 Running {validator.name}...")
            
            try:
                if validator_name == 'functional':
                    # Functional validator returns a dict of results
                    func_results = validator.run_functional_validation()
                    # Convert to individual ValidationResult objects
                    for category, tests in func_results.items():
                        if category == 'summary':
                            continue
                        for test_name, test_result in tests.items():
                            if isinstance(test_result, dict) and 'status' in test_result:
                                result = ValidationResult(
                                    test_name=f"{category}_{test_name}",
                                    passed=test_result['status'] == 'PASS',
                                    message=test_result.get('result', 'No message'),
                                    execution_time=0.0,
                                    details=test_result
                                )
                                suite.results.append(result)
                elif validator_name == 'algorithm_integration':
                    # Algorithm integration tester has special method
                    result = validator.test_algorithm_chain(hg)
                    suite.results.append(result)
                else:
                    # Standard validators
                    result = validator.validate(hg)
                    suite.results.append(result)
                
                print(f"      ✅ {validator.name} completed")
                    
            except Exception as e:
                error_result = ValidationResult(
                    test_name=f"{validator.name} (Error)",
                    passed=False,
                    message=f"Validator crashed: {str(e)}",
                    execution_time=0.0,
                    details={"error": str(e)}
                )
                suite.results.append(error_result)
                print(f"      ❌ {validator.name} crashed: {e}")
        
        suite.end_time = datetime.now()
        self.validation_history.append(suite)
        
        # Print summary
        self._print_validation_summary(suite)
        
        return suite
    
    def run_quick_validation(self, hg: Hypergraph) -> ValidationSuite:
        """Run essential validation checks quickly"""
        return self.run_validation_suite(
            hg, 
            validators=['data_integrity', 'component_integration'],
            suite_name="Quick Validation"
        )
    
    def run_comprehensive_validation(self, hg: Hypergraph) -> ValidationSuite:
        """Run all available validation checks"""
        return self.run_validation_suite(hg, suite_name="Comprehensive Validation")
    
    def validate_multiple_hypergraphs(self, 
                                    hypergraphs: List[Hypergraph],
                                    validator_subset: Optional[List[str]] = None) -> Dict[str, ValidationSuite]:
        """
        Run validation across multiple hypergraphs
        
        Args:
            hypergraphs: List of hypergraphs to validate
            validator_subset: Subset of validators to run
            
        Returns:
            Dictionary mapping hypergraph names to validation suites
        """
        print(f"🔍 Validating {len(hypergraphs)} hypergraphs...")
        
        results = {}
        
        for i, hg in enumerate(hypergraphs):
            hg_name = getattr(hg, 'name', f"hypergraph_{i}")
            print(f"\n📊 Validating {hg_name}...")
            
            suite = self.run_validation_suite(
                hg, 
                validators=validator_subset,
                suite_name=f"Validation - {hg_name}"
            )
            results[hg_name] = suite
        
        # Overall summary
        total_tests = sum(suite.total_count for suite in results.values())
        total_passed = sum(suite.passed_count for suite in results.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎉 Multi-hypergraph validation completed:")
        print(f"   📈 Total tests: {total_tests}")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   📊 Success rate: {overall_success_rate:.1f}%")
        
        return results
    
    def generate_validation_report(self, suite: ValidationSuite) -> str:
        """Generate detailed validation report"""
        report_lines = []
        
        # Header
        report_lines.append(f"Validation Report: {suite.name}")
        report_lines.append("=" * 60)
        
        if suite.start_time and suite.end_time:
            duration = (suite.end_time - suite.start_time).total_seconds()
            report_lines.append(f"Execution Time: {duration:.2f} seconds")
        
        report_lines.append(f"Total Tests: {suite.total_count}")
        report_lines.append(f"Passed: {suite.passed_count}")
        report_lines.append(f"Failed: {suite.failed_count}")
        report_lines.append(f"Success Rate: {suite.success_rate:.1f}%")
        report_lines.append("")
        
        # Individual test results
        for result in suite.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report_lines.append(f"{status} {result.test_name}")
            report_lines.append(f"    Message: {result.message}")
            report_lines.append(f"    Time: {result.execution_time:.3f}s")
            
            if not result.passed and result.details.get('issues'):
                report_lines.append("    Issues:")
                for issue in result.details['issues'][:3]:  # Show first 3 issues
                    report_lines.append(f"      - {issue}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _print_validation_summary(self, suite: ValidationSuite):
        """Print validation summary"""
        print(f"\n📋 Validation Summary for {suite.name}:")
        print(f"   🎯 Tests: {suite.passed_count}/{suite.total_count} passed ({suite.success_rate:.1f}%)")
        print(f"   ⏱️  Time: {suite.total_execution_time:.2f}s")
        
        if suite.failed_count > 0:
            print(f"   ⚠️  {suite.failed_count} tests failed:")
            for result in suite.results:
                if not result.passed:
                    print(f"      - {result.test_name}: {result.message}")
        else:
            print("   🏆 All tests passed!")


# Convenience functions
def quick_validate(hg: Hypergraph) -> bool:
    """Quick validation check - returns True if basic validation passes"""
    runner = ValidationRunner()
    suite = runner.run_quick_validation(hg)
    return suite.success_rate == 100.0


def comprehensive_validate(hg: Hypergraph) -> ValidationSuite:
    """Run comprehensive validation on hypergraph"""
    runner = ValidationRunner()
    return runner.run_comprehensive_validation(hg)


def validate_multiple(hypergraphs: List[Hypergraph]) -> Dict[str, ValidationSuite]:
    """Validate multiple hypergraphs"""
    runner = ValidationRunner()
    return runner.validate_multiple_hypergraphs(hypergraphs)