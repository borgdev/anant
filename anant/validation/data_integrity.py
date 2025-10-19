"""
Data Integrity Validation Module

Provides data integrity validation for hypergraphs including:
- Structure consistency checks
- Data completeness validation  
- Relationship integrity verification
- Weight consistency checks
"""

import time
import polars as pl
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from ..classes.hypergraph import Hypergraph


@dataclass
class ValidationResult:
    """Results from a validation check"""
    test_name: str
    passed: bool
    message: str
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)


class DataIntegrityValidator:
    """Validates data integrity of hypergraphs"""
    
    def __init__(self):
        self.name = "Data Integrity"
    
    def validate(self, hg: Hypergraph) -> ValidationResult:
        """Validate data integrity of hypergraph"""
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Check basic structure
            if hg.num_nodes == 0:
                issues.append("Empty hypergraph (no nodes)")
            
            if hg.num_edges == 0:
                issues.append("No edges in hypergraph")
            
            # Check incidence data consistency
            incidence_data = hg.incidences.data
            
            # Check for null values
            null_edges = incidence_data.filter(pl.col("edge_id").is_null()).height
            null_nodes = incidence_data.filter(pl.col("node_id").is_null()).height
            
            if null_edges > 0:
                issues.append(f"Found {null_edges} null edge values")
            
            if null_nodes > 0:
                issues.append(f"Found {null_nodes} null node values")
            
            # Check for duplicate incidences
            unique_incidences = incidence_data.select(["edge_id", "node_id"]).unique().height
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
            incidence_edges = set(incidence_data["edge_id"].unique().to_list())
            
            if declared_edges != incidence_edges:
                issues.append("Edge set mismatch between declared and incidence data")
            
            declared_nodes = set(hg.nodes)
            incidence_nodes = set(incidence_data["node_id"].unique().to_list())
            
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
                details={"error": str(e)}
            )


def validate_data_integrity(hg: Hypergraph) -> ValidationResult:
    """Convenience function for data integrity validation"""
    validator = DataIntegrityValidator()
    return validator.validate(hg)