"""
Integration Testing Module

Provides integration testing for hypergraph components including:
- Algorithm integration validation
- Component interaction testing  
- Cross-module functionality verification
- End-to-end workflow testing
"""

import time
from typing import Dict, Any
from ..classes.hypergraph import Hypergraph
from ..analysis.centrality import degree_centrality
from ..analysis.clustering import modularity_clustering
from ..streaming import StreamingHypergraph
from .data_integrity import ValidationResult


class ComponentIntegrationValidator:
    """Validates integration between different components"""
    
    def __init__(self):
        self.name = "Component Integration"
    
    def validate(self, hg: Hypergraph) -> ValidationResult:
        """Validate integration between components"""
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


class AlgorithmIntegrationTester:
    """Tests integration between different algorithms"""
    
    def __init__(self):
        self.name = "Algorithm Integration"
    
    def test_algorithm_chain(self, hg: Hypergraph) -> ValidationResult:
        """Test a chain of algorithms working together"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Step 1: Basic centrality analysis
            centralities = degree_centrality(hg)
            if not centralities or 'nodes' not in centralities:
                issues.append("Failed to compute centralities")
                
            # Step 2: Use centrality results for further analysis
            if 'nodes' in centralities:
                node_centralities = centralities['nodes']
                if len(node_centralities) != hg.num_nodes:
                    issues.append("Centrality results inconsistent with hypergraph structure")
            
            # Step 3: Community detection
            if hg.num_edges >= 2:
                communities = modularity_clustering(hg)
                if not communities:
                    issues.append("Failed to detect communities")
            
            execution_time = time.perf_counter() - start_time
            
            if issues:
                return ValidationResult(
                    test_name="Algorithm Integration Chain",
                    passed=False,
                    message=f"Algorithm chain issues: {'; '.join(issues[:2])}",
                    execution_time=execution_time,
                    details={"issues": issues}
                )
            else:
                return ValidationResult(
                    test_name="Algorithm Integration Chain",
                    passed=True,
                    message="Algorithm chain executed successfully",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                test_name="Algorithm Integration Chain",
                passed=False,
                message=f"Algorithm chain failed: {str(e)}",
                execution_time=execution_time,
                details={"error": str(e)}
            )


def validate_component_integration(hg: Hypergraph) -> ValidationResult:
    """Convenience function for component integration validation"""
    validator = ComponentIntegrationValidator()
    return validator.validate(hg)


def test_algorithm_integration(hg: Hypergraph) -> ValidationResult:
    """Convenience function for algorithm integration testing"""
    tester = AlgorithmIntegrationTester()
    return tester.test_algorithm_chain(hg)