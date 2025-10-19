"""
Functional Validation Framework for Anant Library

This module provides focused validation capabilities emphasizing:
- Functional testing of all algorithms
- Correctness validation
- Edge case handling
- Integration testing
- Core functionality verification

The validation framework is modular and organized into focused components.
"""

# Core validation components
from .functional_validator import FunctionalValidator, run_functional_validation
from .automated_testing import AlgorithmTester, ValidationSuite, run_all_tests

# Modular validation components  
from .data_integrity import DataIntegrityValidator, validate_data_integrity
from .performance_validation import PerformanceBenchmarkValidator, benchmark_performance
from .integration_testing import (
    ComponentIntegrationValidator, 
    AlgorithmIntegrationTester,
    validate_component_integration,
    test_algorithm_integration
)
from .validation_runner import (
    ValidationRunner,
    quick_validate,
    comprehensive_validate,
    validate_multiple
)

__all__ = [
    # Core validators
    'FunctionalValidator',
    'AlgorithmTester',
    'ValidationSuite',
    'run_functional_validation',
    'run_all_tests',
    
    # Modular validators
    'DataIntegrityValidator',
    'PerformanceBenchmarkValidator', 
    'ComponentIntegrationValidator',
    'AlgorithmIntegrationTester',
    
    # Validation runner
    'ValidationRunner',
    
    # Convenience functions
    'validate_data_integrity',
    'benchmark_performance',
    'validate_component_integration',
    'test_algorithm_integration',
    'quick_validate',
    'comprehensive_validate',
    'validate_multiple'
]
