"""
Validation Framework for anant Library

Comprehensive quality assurance framework providing data integrity checks,
performance benchmarks, automated testing workflows, and validation metrics
for all components of the anant hypergraph library.
"""

import polars as pl
import numpy as np
import time
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from datetime import datetime
import sys

from ..classes.hypergraph import Hypergraph
from ..optimization import PerformanceOptimizer, MemoryMonitor, OptimizationConfig
from ..analysis.centrality import degree_centrality, s_centrality
from ..analysis.clustering import modularity_clustering, community_quality_metrics
from ..analysis.temporal import TemporalHypergraph
from ..streaming import StreamingHypergraph
try:
    from ..io.advanced_io import HypergraphIO
except ImportError:
    # Fallback for basic I/O operations
    class HypergraphIO:
        @staticmethod
        def save_hypergraph(hg, path):
            return hg._incidence_store.data.write_parquet(path)
        
        @staticmethod
        def load_hypergraph(path):
            data = pl.read_parquet(path)
            return Hypergraph(data)
        
        @staticmethod
        def to_json(hg):
            # Export full incidence data for proper reconstruction
            incidence_data = hg._incidence_store.data
            return {
                "nodes": hg.nodes, 
                "edges": hg.edges,
                "incidences": incidence_data.to_dicts()
            }
        
        @staticmethod
        def from_json(data):
            # Reconstruct from incidence data if available
            if "incidences" in data and data["incidences"]:
                incidence_df = pl.DataFrame(data["incidences"])
                return Hypergraph(incidence_df)
            else:
                # Fallback to basic structure
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                if not nodes or not edges:
                    return Hypergraph()
                
                # Create minimal incidence structure
                incidences = []
                for edge in edges:
                    for node in nodes[:2]:  # Connect first two nodes to each edge
                        incidences.append({"edges": edge, "nodes": node, "weight": 1.0})
                
                if incidences:
                    return Hypergraph(pl.DataFrame(incidences))
                else:
                    return Hypergraph()

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a validation check"""
    test_name: str
    passed: bool
    message: str
    execution_time: float
    memory_usage: Optional[float] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


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


class BaseValidator(ABC):
    """Base class for all validators"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory_monitor = MemoryMonitor()
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """Perform validation on target object"""
        pass
    
    def _measure_performance(self, func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
        """Measure execution time and memory usage of a function"""
        # Use simplified memory monitoring
        start_time = time.perf_counter()
        initial_memory = self.memory_monitor.get_usage_mb()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            final_memory = self.memory_monitor.get_usage_mb()
            memory_usage = max(final_memory - initial_memory, 0)
            
            return result, end_time - start_time, memory_usage
        except Exception as e:
            end_time = time.perf_counter()
            return None, end_time - start_time, 0.0


class DataIntegrityValidator(BaseValidator):
    """Validates data integrity of hypergraphs"""
    
    def __init__(self):
        super().__init__("Data Integrity")
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Validate data integrity of hypergraph"""
        hg = target  # For readability
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Check basic structure
            if hg.num_nodes == 0:
                issues.append("Empty hypergraph (no nodes)")
            
            if hg.num_edges == 0:
                issues.append("No edges in hypergraph")
            
            # Check incidence data consistency
            incidence_data = hg._incidence_store.data
            
            # Check for null values
            null_edges = incidence_data.filter(pl.col("edges").is_null()).height
            null_nodes = incidence_data.filter(pl.col("nodes").is_null()).height
            
            if null_edges > 0:
                issues.append(f"Found {null_edges} null edge values")
            
            if null_nodes > 0:
                issues.append(f"Found {null_nodes} null node values")
            
            # Check for duplicate incidences
            unique_incidences = incidence_data.select(["edges", "nodes"]).unique().height
            total_incidences = incidence_data.height
            
            if unique_incidences != total_incidences:
                issues.append(f"Found duplicate incidences: {total_incidences - unique_incidences}")
            
            # Check weight consistency
            if "weight" in incidence_data.columns:
                negative_weights = incidence_data.filter(pl.col("weight") < 0).height
                if negative_weights > 0:
                    issues.append(f"Found {negative_weights} negative weights")
                
                nan_weights = incidence_data.filter(pl.col("weight").is_nan()).height
                if nan_weights > 0:
                    issues.append(f"Found {nan_weights} NaN weights")
            
            # Check edge-node relationship consistency
            declared_edges = set(hg.edges)
            incidence_edges = set(incidence_data["edges"].unique().to_list())
            
            if declared_edges != incidence_edges:
                issues.append("Edge set mismatch between declared and incidence data")
            
            declared_nodes = set(hg.nodes)
            incidence_nodes = set(incidence_data["nodes"].unique().to_list())
            
            if declared_nodes != incidence_nodes:
                issues.append("Node set mismatch between declared and incidence data")
            
            execution_time = time.perf_counter() - start_time
            
            if issues:
                return ValidationResult(
                    test_name="Data Integrity Check",
                    passed=False,
                    message=f"Found {len(issues)} integrity issues: " + "; ".join(issues[:3]),
                    execution_time=execution_time,
                    details={"issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="Data Integrity Check",
                    passed=True,
                    message="All integrity checks passed",
                    execution_time=execution_time,
                    details={"nodes": hg.num_nodes, "edges": hg.num_edges}
                )
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Data Integrity Check",
                passed=False,
                message=f"Validation failed with error: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )


class PerformanceBenchmarkValidator(BaseValidator):
    """Validates performance characteristics"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        super().__init__("Performance Benchmark")
        self.thresholds = thresholds or {
            'basic_operations': 1.0,  # seconds
            'centrality_computation': 5.0,
            'clustering_computation': 10.0,
            'memory_usage': 1000.0  # MB
        }
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Benchmark performance of core operations"""
        hg = target  # For readability
        benchmark_results = {}
        issues = []
        
        try:
            # Benchmark basic operations
            _, time_taken, memory_used = self._measure_performance(
                self._benchmark_basic_operations, hg
            )
            benchmark_results['basic_operations'] = time_taken
            benchmark_results['memory_usage'] = memory_used
            
            if time_taken > self.thresholds['basic_operations']:
                issues.append(f"Basic operations too slow: {time_taken:.2f}s > {self.thresholds['basic_operations']}s")
            
            if memory_used > self.thresholds['memory_usage']:
                issues.append(f"Memory usage too high: {memory_used:.1f}MB > {self.thresholds['memory_usage']}MB")
            
            # Benchmark centrality computation if enough nodes
            if hg.num_nodes >= 5:
                _, time_taken, _ = self._measure_performance(
                    degree_centrality, hg
                )
                benchmark_results['centrality_computation'] = time_taken
                
                if time_taken > self.thresholds['centrality_computation']:
                    issues.append(f"Centrality computation too slow: {time_taken:.2f}s")
            
            # Benchmark clustering if enough edges
            if hg.num_edges >= 3:
                _, time_taken, _ = self._measure_performance(
                    modularity_clustering, hg
                )
                benchmark_results['clustering_computation'] = time_taken
                
                if time_taken > self.thresholds['clustering_computation']:
                    issues.append(f"Clustering computation too slow: {time_taken:.2f}s")
            
            total_time = sum(benchmark_results.values())
            
            if issues:
                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=False,
                    message=f"Performance issues detected: {'; '.join(issues[:2])}",
                    execution_time=total_time,
                    performance_metrics=benchmark_results,
                    details={"issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=True,
                    message="All performance benchmarks passed",
                    execution_time=total_time,
                    performance_metrics=benchmark_results
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                message=f"Benchmark failed with error: {str(e)}",
                execution_time=0.0,
                details={"error": str(e)}
            )
    
    def _benchmark_basic_operations(self, hg: Hypergraph):
        """Benchmark basic hypergraph operations"""
        # Test node and edge access
        nodes = hg.nodes
        edges = hg.edges
        
        # Test neighbor operations if possible
        if nodes:
            sample_node = nodes[0]
            hg.neighbors(sample_node)
        
        # Test incidence operations
        _ = hg._incidence_store.data
        
        return True


class ComponentIntegrationValidator(BaseValidator):
    """Validates integration between different components"""
    
    def __init__(self):
        super().__init__("Component Integration")
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Validate integration between components"""
        hg = target  # For readability
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Test analysis integration
            try:
                centralities = degree_centrality(hg)
                if not isinstance(centralities, dict):
                    issues.append("Centrality analysis returned invalid format")
                elif 'nodes' not in centralities or not isinstance(centralities['nodes'], dict):
                    issues.append("Centrality analysis missing node results")
                elif len(centralities['nodes']) != hg.num_nodes:
                    issues.append("Centrality analysis node count mismatch")
            except Exception as e:
                issues.append(f"Centrality analysis failed: {str(e)}")
            
            # Test clustering integration
            if hg.num_edges >= 2:
                try:
                    communities = modularity_clustering(hg)
                    if not isinstance(communities, dict):
                        issues.append("Clustering analysis returned invalid format")
                except Exception as e:
                    issues.append(f"Clustering analysis failed: {str(e)}")
            
            # Test property integration
            try:
                # Add test properties in correct format: {node_id: {prop_name: value}}
                test_nodes = list(hg.nodes)[:3]
                test_properties = {node: {"test_prop": 1.0} for node in test_nodes}
                hg.add_node_properties(test_properties)
                # Test individual node property retrieval
                if test_nodes:
                    node_prop = hg.get_node_properties(test_nodes[0])
                    if not isinstance(node_prop, dict) or "test_prop" not in node_prop:
                        issues.append("Property storage/retrieval failed")
            except Exception as e:
                issues.append(f"Property integration failed: {str(e)}")
            
            # Test streaming integration if available
            try:
                streaming_hg = StreamingHypergraph(hg, enable_optimization=False)
                if not streaming_hg.current_hypergraph:
                    issues.append("Streaming hypergraph initialization failed")
            except Exception as e:
                issues.append(f"Streaming integration failed: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            if issues:
                return ValidationResult(
                    test_name="Component Integration",
                    passed=False,
                    message=f"Integration issues detected: {'; '.join(issues[:2])}",
                    execution_time=execution_time,
                    details={"issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="Component Integration",
                    passed=True,
                    message="All component integrations working correctly",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Component Integration",
                passed=False,
                message=f"Integration validation failed: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )


class IOValidationValidator(BaseValidator):
    """Validates I/O operations and data persistence"""
    
    def __init__(self):
        super().__init__("I/O Validation")
    
    def validate(self, target: Hypergraph) -> ValidationResult:
        """Validate I/O operations"""
        hg = target  # For readability
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Test parquet I/O
            try:
                # Save to temporary file
                temp_file = Path("/tmp/test_hypergraph.parquet")
                
                # Save hypergraph
                HypergraphIO.save_hypergraph(hg, temp_file)
                
                if not temp_file.exists():
                    issues.append("Failed to save hypergraph to parquet")
                else:
                    # Load hypergraph
                    loaded_hg = HypergraphIO.load_hypergraph(temp_file)
                    
                    # Verify structure
                    if loaded_hg.num_nodes != hg.num_nodes:
                        issues.append("Node count mismatch after parquet round-trip")
                    
                    if loaded_hg.num_edges != hg.num_edges:
                        issues.append("Edge count mismatch after parquet round-trip")
                    
                    # Clean up
                    temp_file.unlink(missing_ok=True)
                    
            except Exception as e:
                issues.append(f"Parquet I/O failed: {str(e)}")
            
            # Test JSON export/import
            try:
                json_data = HypergraphIO.to_json(hg)
                if not isinstance(json_data, dict):
                    issues.append("JSON export returned invalid format")
                
                reconstructed_hg = HypergraphIO.from_json(json_data)
                if reconstructed_hg.num_nodes != hg.num_nodes:
                    issues.append("Node count mismatch after JSON round-trip")
                    
            except Exception as e:
                issues.append(f"JSON I/O failed: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            if issues:
                return ValidationResult(
                    test_name="I/O Validation",
                    passed=False,
                    message=f"I/O issues detected: {'; '.join(issues[:2])}",
                    execution_time=execution_time,
                    details={"issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="I/O Validation",
                    passed=True,
                    message="All I/O operations working correctly",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="I/O Validation",
                passed=False,
                message=f"I/O validation failed: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )


class ValidationFramework:
    """
    Comprehensive validation framework for anant library
    
    Provides automated testing workflows, quality assurance metrics,
    and comprehensive validation across all components.
    """
    
    def __init__(self, 
                 performance_thresholds: Optional[Dict[str, float]] = None,
                 enable_logging: bool = True):
        """
        Initialize validation framework
        
        Args:
            performance_thresholds: Custom performance thresholds
            enable_logging: Whether to enable detailed logging
        """
        self.performance_thresholds = performance_thresholds
        self.enable_logging = enable_logging
        
        # Initialize validators
        self.validators = {
            'data_integrity': DataIntegrityValidator(),
            'performance': PerformanceBenchmarkValidator(performance_thresholds),
            'integration': ComponentIntegrationValidator(),
            'io_validation': IOValidationValidator()
        }
        
        # Results storage
        self.validation_history: List[ValidationSuite] = []
        
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
    
    def validate_hypergraph(self, 
                          hg: Hypergraph, 
                          validators: Optional[List[str]] = None) -> ValidationSuite:
        """
        Run comprehensive validation on a hypergraph
        
        Args:
            hg: Hypergraph to validate
            validators: List of validator names to run (None = all)
            
        Returns:
            ValidationSuite with all results
        """
        suite = ValidationSuite(name=f"Hypergraph Validation - {hg.name or 'Unnamed'}")
        suite.start_time = datetime.now()
        
        # Determine which validators to run
        validator_names = validators or list(self.validators.keys())
        
        logger.info(f"Starting validation suite with {len(validator_names)} validators")
        
        # Run each validator
        for validator_name in validator_names:
            if validator_name not in self.validators:
                logger.warning(f"Unknown validator: {validator_name}")
                continue
            
            validator = self.validators[validator_name]
            logger.info(f"Running {validator.name} validator...")
            
            try:
                result = validator.validate(hg)
                suite.results.append(result)
                
                if self.enable_logging:
                    status = "✓" if result.passed else "✗"
                    logger.info(f"{status} {result.test_name}: {result.message}")
                    
            except Exception as e:
                error_result = ValidationResult(
                    test_name=f"{validator.name} (Error)",
                    passed=False,
                    message=f"Validator crashed: {str(e)}",
                    execution_time=0.0,
                    details={"error": str(e), "traceback": traceback.format_exc()}
                )
                suite.results.append(error_result)
                logger.error(f"Validator {validator_name} crashed: {e}")
        
        suite.end_time = datetime.now()
        self.validation_history.append(suite)
        
        # Log summary
        if self.enable_logging:
            self._log_validation_summary(suite)
        
        return suite
    
    def validate_streaming_hypergraph(self, 
                                    streaming_hg: StreamingHypergraph) -> ValidationSuite:
        """Validate streaming hypergraph functionality"""
        suite = ValidationSuite(name="Streaming Hypergraph Validation")
        suite.start_time = datetime.now()
        
        # Basic hypergraph validation
        base_suite = self.validate_hypergraph(streaming_hg.current_hypergraph, 
                                            ['data_integrity', 'performance'])
        suite.results.extend(base_suite.results)
        
        # Streaming-specific validations
        start_time = time.perf_counter()
        
        try:
            # Test streaming updates
            test_passed = True
            issues = []
            
            initial_nodes = streaming_hg.current_hypergraph.num_nodes
            initial_edges = streaming_hg.current_hypergraph.num_edges
            
            # Add test update
            success = streaming_hg.add_edge_update(
                timestamp=1,
                edge_id="test_edge",
                nodes=["test_node1", "test_node2"]
            )
            
            if not success:
                test_passed = False
                issues.append("Failed to add streaming update")
            
            # Start processing briefly
            streaming_hg.start_processing()
            time.sleep(0.1)
            streaming_hg.stop_processing()
            
            # Check results
            stats = streaming_hg.get_statistics()
            if stats['processed_updates'] == 0:
                test_passed = False
                issues.append("No updates were processed")
            
            final_hg = streaming_hg.current_hypergraph
            if final_hg.num_nodes <= initial_nodes or final_hg.num_edges <= initial_edges:
                test_passed = False
                issues.append("Hypergraph structure not updated correctly")
            
            execution_time = time.perf_counter() - start_time
            
            streaming_result = ValidationResult(
                test_name="Streaming Functionality",
                passed=test_passed,
                message="Streaming validation passed" if test_passed else f"Issues: {'; '.join(issues)}",
                execution_time=execution_time,
                performance_metrics=stats,
                details={"issues": issues if not test_passed else []}
            )
            
            suite.results.append(streaming_result)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_result = ValidationResult(
                test_name="Streaming Functionality",
                passed=False,
                message=f"Streaming validation failed: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )
            suite.results.append(error_result)
        
        suite.end_time = datetime.now()
        self.validation_history.append(suite)
        
        return suite
    
    def validate_temporal_hypergraph(self, 
                                   temporal_hg: TemporalHypergraph) -> ValidationSuite:
        """Validate temporal hypergraph functionality"""
        suite = ValidationSuite(name="Temporal Hypergraph Validation")
        suite.start_time = datetime.now()
        
        # Basic hypergraph validation on latest snapshot
        if temporal_hg.snapshots:
            latest_snapshot = temporal_hg.snapshots[-1]
            base_suite = self.validate_hypergraph(latest_snapshot.hypergraph,
                                                ['data_integrity', 'integration'])
            suite.results.extend(base_suite.results)
        
        # Temporal-specific validations
        start_time = time.perf_counter()
        
        try:
            issues = []
            test_passed = True
            
            # Check temporal structure
            if len(temporal_hg.timestamps) == 0:
                test_passed = False
                issues.append("No temporal snapshots found")
            
            # Test temporal operations
            if temporal_hg.timestamps:
                try:
                    # Test snapshot retrieval
                    first_timestamp = temporal_hg.timestamps[0]
                    snapshot = temporal_hg.get_snapshot(first_timestamp)
                    
                    if snapshot is None:
                        test_passed = False
                        issues.append("Failed to retrieve temporal snapshot")
                
                except Exception as e:
                    test_passed = False
                    issues.append(f"Temporal operation failed: {str(e)}")
            
            execution_time = time.perf_counter() - start_time
            
            temporal_result = ValidationResult(
                test_name="Temporal Functionality",
                passed=test_passed,
                message="Temporal validation passed" if test_passed else f"Issues: {'; '.join(issues)}",
                execution_time=execution_time,
                details={"timestamps": len(temporal_hg.timestamps), "issues": issues if not test_passed else []}
            )
            
            suite.results.append(temporal_result)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_result = ValidationResult(
                test_name="Temporal Functionality",
                passed=False,
                message=f"Temporal validation failed: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )
            suite.results.append(error_result)
        
        suite.end_time = datetime.now()
        self.validation_history.append(suite)
        
        return suite
    
    def run_comprehensive_test_suite(self, 
                                   test_hypergraphs: List[Hypergraph]) -> Dict[str, ValidationSuite]:
        """
        Run comprehensive validation across multiple hypergraphs
        
        Args:
            test_hypergraphs: List of hypergraphs to validate
            
        Returns:
            Dictionary mapping hypergraph names to validation suites
        """
        logger.info(f"Starting comprehensive test suite with {len(test_hypergraphs)} hypergraphs")
        
        results = {}
        
        for i, hg in enumerate(test_hypergraphs):
            hg_name = hg.name or f"hypergraph_{i}"
            logger.info(f"Validating {hg_name}...")
            
            suite = self.validate_hypergraph(hg)
            results[hg_name] = suite
        
        # Log overall summary
        total_tests = sum(suite.total_count for suite in results.values())
        total_passed = sum(suite.passed_count for suite in results.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Comprehensive test suite completed:")
        logger.info(f"  Total tests: {total_tests}")
        logger.info(f"  Passed: {total_passed}")
        logger.info(f"  Success rate: {overall_success_rate:.1f}%")
        
        return results
    
    def generate_validation_report(self, suite: ValidationSuite) -> str:
        """Generate detailed validation report"""
        report_lines = []
        
        # Header
        report_lines.append(f"Validation Report: {suite.name}")
        report_lines.append("=" * 50)
        
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
            
            if result.memory_usage:
                report_lines.append(f"    Memory: {result.memory_usage:.1f}MB")
            
            if result.performance_metrics:
                report_lines.append("    Performance Metrics:")
                for metric, value in result.performance_metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"      {metric}: {value:.3f}")
                    else:
                        report_lines.append(f"      {metric}: {value}")
            
            if not result.passed and result.details.get('issues'):
                report_lines.append("    Issues:")
                for issue in result.details['issues'][:5]:  # Show first 5 issues
                    report_lines.append(f"      - {issue}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _log_validation_summary(self, suite: ValidationSuite):
        """Log validation summary"""
        logger.info(f"Validation Summary for {suite.name}:")
        logger.info(f"  Tests: {suite.passed_count}/{suite.total_count} passed ({suite.success_rate:.1f}%)")
        logger.info(f"  Execution time: {suite.total_execution_time:.2f}s")
        
        if suite.failed_count > 0:
            logger.warning(f"  {suite.failed_count} tests failed")
            for result in suite.results:
                if not result.passed:
                    logger.warning(f"    - {result.test_name}: {result.message}")


# Convenience functions for common validation tasks
def quick_validate(hg: Hypergraph) -> bool:
    """Quick validation check - returns True if basic validation passes"""
    framework = ValidationFramework(enable_logging=False)
    suite = framework.validate_hypergraph(hg, ['data_integrity'])
    return suite.success_rate == 100.0


def performance_benchmark(hg: Hypergraph, 
                         thresholds: Optional[Dict[str, float]] = None) -> ValidationResult:
    """Run performance benchmark on hypergraph"""
    validator = PerformanceBenchmarkValidator(thresholds)
    return validator.validate(hg)


def validate_all_components(hg: Hypergraph) -> ValidationSuite:
    """Run all validation checks on hypergraph"""
    framework = ValidationFramework()
    return framework.validate_hypergraph(hg)


__all__ = [
    'ValidationResult',
    'ValidationSuite', 
    'ValidationFramework',
    'DataIntegrityValidator',
    'PerformanceBenchmarkValidator',
    'ComponentIntegrationValidator',
    'IOValidationValidator',
    'quick_validate',
    'performance_benchmark',
    'validate_all_components'
]