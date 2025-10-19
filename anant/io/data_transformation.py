"""
Data Transformation Utilities for Anant Library

Provides comprehensive data preparation and transformation tools:
- Data cleaning and preprocessing pipelines
- Format conversion between graph/hypergraph representations
- Dimensionality reduction and feature engineering
- Data validation and quality assessment
- ETL (Extract, Transform, Load) pipelines
- Statistical preprocessing and normalization
- Graph-to-hypergraph conversion strategies
- Data sampling and filtering utilities

This module bridges the gap between raw data and hypergraph analysis,
providing robust tools for data preparation and transformation.
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
from enum import Enum

from ..classes.hypergraph import Hypergraph


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_RECORDS = "duplicate_records"
    INVALID_FORMAT = "invalid_format"
    OUTLIERS = "outliers"
    INCONSISTENT_TYPES = "inconsistent_types"
    EMPTY_COLUMNS = "empty_columns"
    HIGH_CARDINALITY = "high_cardinality"


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int
    total_columns: int
    issues: Dict[DataQualityIssue, List[Dict[str, Any]]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    
    def add_issue(self, issue_type: DataQualityIssue, details: Dict[str, Any]) -> None:
        """Add a data quality issue"""
        if issue_type not in self.issues:
            self.issues[issue_type] = []
        self.issues[issue_type].append(details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_records": self.total_records,
            "total_columns": self.total_columns,
            "issues": {k.value: v for k, v in self.issues.items()},
            "recommendations": self.recommendations,
            "overall_score": self.overall_score
        }


@dataclass
class TransformationConfig:
    """Configuration for data transformations"""
    handle_missing_values: str = "drop"  # "drop", "fill", "interpolate"
    missing_fill_value: Any = None
    remove_duplicates: bool = True
    normalize_columns: List[str] = field(default_factory=list)
    categorical_encoding: str = "label"  # "label", "onehot", "target"
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation"
    text_preprocessing: bool = True
    date_parsing: bool = True
    type_inference: bool = True


class DataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    def __init__(self, config: TransformationConfig):
        self.config = config
    
    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the data"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get transformation metadata"""
        pass


class DataQualityAssessor:
    """Assess data quality and generate reports"""
    
    def __init__(self):
        self.assessors = {
            DataQualityIssue.MISSING_VALUES: self._assess_missing_values,
            DataQualityIssue.DUPLICATE_RECORDS: self._assess_duplicates,
            DataQualityIssue.INVALID_FORMAT: self._assess_format_issues,
            DataQualityIssue.OUTLIERS: self._assess_outliers,
            DataQualityIssue.INCONSISTENT_TYPES: self._assess_type_consistency,
            DataQualityIssue.EMPTY_COLUMNS: self._assess_empty_columns,
            DataQualityIssue.HIGH_CARDINALITY: self._assess_cardinality
        }
    
    def assess(self, data: pl.DataFrame) -> DataQualityReport:
        """Comprehensive data quality assessment"""
        report = DataQualityReport(
            total_records=len(data),
            total_columns=len(data.columns)
        )
        
        # Run all assessments
        for issue_type, assessor in self.assessors.items():
            try:
                issues = assessor(data)
                for issue in issues:
                    report.add_issue(issue_type, issue)
            except Exception as e:
                print(f"Error in {issue_type.value} assessment: {e}")
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Calculate overall score
        report.overall_score = self._calculate_score(report)
        
        return report
    
    def _assess_missing_values(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess missing values"""
        issues = []
        
        for col in data.columns:
            null_count = data[col].null_count()
            if null_count > 0:
                missing_rate = null_count / len(data)
                issues.append({
                    "column": col,
                    "missing_count": null_count,
                    "missing_rate": missing_rate,
                    "severity": "high" if missing_rate > 0.5 else "medium" if missing_rate > 0.1 else "low"
                })
        
        return issues
    
    def _assess_duplicates(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess duplicate records"""
        issues = []
        
        total_records = len(data)
        unique_records = len(data.unique())
        duplicate_count = total_records - unique_records
        
        if duplicate_count > 0:
            duplicate_rate = duplicate_count / total_records
            issues.append({
                "duplicate_count": duplicate_count,
                "duplicate_rate": duplicate_rate,
                "severity": "high" if duplicate_rate > 0.2 else "medium" if duplicate_rate > 0.05 else "low"
            })
        
        return issues
    
    def _assess_format_issues(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess format consistency issues"""
        issues = []
        
        for col in data.columns:
            if data[col].dtype == pl.Utf8:
                # Check for inconsistent string formats
                unique_patterns = set()
                for value in data[col].drop_nulls().unique():
                    if isinstance(value, str):
                        # Simple pattern detection
                        pattern = re.sub(r'\d', 'D', str(value))
                        pattern = re.sub(r'[a-zA-Z]', 'A', pattern)
                        unique_patterns.add(pattern)
                
                if len(unique_patterns) > len(data[col].unique()) * 0.5:
                    issues.append({
                        "column": col,
                        "issue": "inconsistent_string_formats",
                        "pattern_count": len(unique_patterns),
                        "severity": "medium"
                    })
        
        return issues
    
    def _assess_outliers(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess outliers in numeric columns"""
        issues = []
        
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        
        for col in numeric_cols:
            col_data = data[col].drop_nulls()
            if len(col_data) > 0:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = col_data.filter((col_data < lower_bound) | (col_data > upper_bound))
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_rate = outlier_count / len(col_data)
                    issues.append({
                        "column": col,
                        "outlier_count": outlier_count,
                        "outlier_rate": outlier_rate,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "severity": "high" if outlier_rate > 0.1 else "medium" if outlier_rate > 0.05 else "low"
                    })
        
        return issues
    
    def _assess_type_consistency(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess type consistency"""
        issues = []
        
        for col in data.columns:
            if data[col].dtype == pl.Utf8:
                # Check if column should be numeric
                non_null_values = data[col].drop_nulls()
                if len(non_null_values) > 0:
                    numeric_count = 0
                    for value in non_null_values:
                        try:
                            float(value)
                            numeric_count += 1
                        except ValueError:
                            pass
                    
                    numeric_rate = numeric_count / len(non_null_values)
                    if numeric_rate > 0.8:
                        issues.append({
                            "column": col,
                            "issue": "should_be_numeric",
                            "numeric_rate": numeric_rate,
                            "severity": "medium"
                        })
        
        return issues
    
    def _assess_empty_columns(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess empty columns"""
        issues = []
        
        for col in data.columns:
            if data[col].null_count() == len(data):
                issues.append({
                    "column": col,
                    "issue": "completely_empty",
                    "severity": "high"
                })
        
        return issues
    
    def _assess_cardinality(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Assess high cardinality issues"""
        issues = []
        
        for col in data.columns:
            unique_count = len(data[col].unique())
            cardinality_rate = unique_count / len(data)
            
            if cardinality_rate > 0.95 and unique_count > 100:
                issues.append({
                    "column": col,
                    "unique_count": unique_count,
                    "cardinality_rate": cardinality_rate,
                    "issue": "very_high_cardinality",
                    "severity": "medium"
                })
        
        return issues
    
    def _generate_recommendations(self, report: DataQualityReport) -> List[str]:
        """Generate recommendations based on issues"""
        recommendations = []
        
        if DataQualityIssue.MISSING_VALUES in report.issues:
            recommendations.append("Consider handling missing values through imputation or removal")
        
        if DataQualityIssue.DUPLICATE_RECORDS in report.issues:
            recommendations.append("Remove duplicate records to improve data quality")
        
        if DataQualityIssue.OUTLIERS in report.issues:
            recommendations.append("Investigate outliers - they may indicate data errors or important patterns")
        
        if DataQualityIssue.INCONSISTENT_TYPES in report.issues:
            recommendations.append("Convert string columns to appropriate numeric types where possible")
        
        if DataQualityIssue.EMPTY_COLUMNS in report.issues:
            recommendations.append("Remove completely empty columns")
        
        if DataQualityIssue.HIGH_CARDINALITY in report.issues:
            recommendations.append("Consider grouping high-cardinality columns or using dimensionality reduction")
        
        return recommendations
    
    def _calculate_score(self, report: DataQualityReport) -> float:
        """Calculate overall data quality score (0-100)"""
        if report.total_records == 0:
            return 0.0
        
        penalty = 0.0
        
        # Penalize based on issues
        for issue_type, issues in report.issues.items():
            for issue in issues:
                severity = issue.get("severity", "low")
                if severity == "high":
                    penalty += 20
                elif severity == "medium":
                    penalty += 10
                else:
                    penalty += 5
        
        # Cap penalty at 100
        penalty = min(penalty, 100)
        
        return max(0.0, 100.0 - penalty)


class DataCleaner(DataTransformer):
    """Data cleaning and preprocessing"""
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        super().__init__(config or TransformationConfig())
        self.cleaning_steps = []
    
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply data cleaning transformations"""
        cleaned_data = data.clone()
        
        # Handle missing values
        if self.config.handle_missing_values == "drop":
            cleaned_data = self._drop_missing(cleaned_data)
        elif self.config.handle_missing_values == "fill":
            cleaned_data = self._fill_missing(cleaned_data)
        elif self.config.handle_missing_values == "interpolate":
            cleaned_data = self._interpolate_missing(cleaned_data)
        
        # Remove duplicates
        if self.config.remove_duplicates:
            cleaned_data = self._remove_duplicates(cleaned_data)
        
        # Type inference
        if self.config.type_inference:
            cleaned_data = self._infer_types(cleaned_data)
        
        # Text preprocessing
        if self.config.text_preprocessing:
            cleaned_data = self._preprocess_text(cleaned_data)
        
        # Date parsing
        if self.config.date_parsing:
            cleaned_data = self._parse_dates(cleaned_data)
        
        return cleaned_data
    
    def _drop_missing(self, data: pl.DataFrame) -> pl.DataFrame:
        """Drop rows with missing values"""
        self.cleaning_steps.append("dropped_missing_values")
        return data.drop_nulls()
    
    def _fill_missing(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fill missing values"""
        filled_data = data.clone()
        
        for col in data.columns:
            if data[col].null_count() > 0:
                if self.config.missing_fill_value is not None:
                    filled_data = filled_data.with_columns(
                        pl.col(col).fill_null(self.config.missing_fill_value)
                    )
                elif data[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                    # Fill numeric with median
                    median_val = data[col].median()
                    filled_data = filled_data.with_columns(
                        pl.col(col).fill_null(median_val)
                    )
                else:
                    # Fill categorical with mode
                    mode_val = data[col].mode().first()
                    filled_data = filled_data.with_columns(
                        pl.col(col).fill_null(mode_val)
                    )
        
        self.cleaning_steps.append("filled_missing_values")
        return filled_data
    
    def _interpolate_missing(self, data: pl.DataFrame) -> pl.DataFrame:
        """Interpolate missing values for numeric columns"""
        interpolated_data = data.clone()
        
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        
        for col in numeric_cols:
            if data[col].null_count() > 0:
                # Simple linear interpolation
                interpolated_data = interpolated_data.with_columns(
                    pl.col(col).interpolate()
                )
        
        self.cleaning_steps.append("interpolated_missing_values")
        return interpolated_data
    
    def _remove_duplicates(self, data: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate rows"""
        self.cleaning_steps.append("removed_duplicates")
        return data.unique()
    
    def _infer_types(self, data: pl.DataFrame) -> pl.DataFrame:
        """Infer and convert appropriate data types"""
        converted_data = data.clone()
        
        for col in data.columns:
            if data[col].dtype == pl.Utf8:
                # Try to convert to numeric
                try:
                    # Check if all non-null values can be converted to float
                    non_null_data = data[col].drop_nulls()
                    if len(non_null_data) > 0:
                        # Test conversion
                        test_values = non_null_data.head(min(100, len(non_null_data)))
                        numeric_count = 0
                        for val in test_values:
                            try:
                                float(val)
                                numeric_count += 1
                            except ValueError:
                                pass
                        
                        if numeric_count / len(test_values) > 0.9:
                            # Convert to numeric
                            converted_data = converted_data.with_columns(
                                pl.col(col).cast(pl.Float64, strict=False)
                            )
                except Exception:
                    pass
        
        self.cleaning_steps.append("inferred_types")
        return converted_data
    
    def _preprocess_text(self, data: pl.DataFrame) -> pl.DataFrame:
        """Basic text preprocessing"""
        processed_data = data.clone()
        
        text_cols = [col for col in data.columns if data[col].dtype == pl.Utf8]
        
        for col in text_cols:
            # Strip whitespace and convert to lowercase
            processed_data = processed_data.with_columns(
                pl.col(col).str.strip_chars().str.to_lowercase()
            )
        
        self.cleaning_steps.append("preprocessed_text")
        return processed_data
    
    def _parse_dates(self, data: pl.DataFrame) -> pl.DataFrame:
        """Parse date columns"""
        parsed_data = data.clone()
        
        for col in data.columns:
            if data[col].dtype == pl.Utf8:
                # Try to parse as date
                try:
                    sample_values = data[col].drop_nulls().head(10)
                    date_count = 0
                    
                    for val in sample_values:
                        try:
                            pd.to_datetime(val)
                            date_count += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if date_count / len(sample_values) > 0.8:
                        # Convert to datetime
                        parsed_data = parsed_data.with_columns(
                            pl.col(col).str.strptime(pl.Datetime, strict=False)
                        )
                except Exception:
                    pass
        
        self.cleaning_steps.append("parsed_dates")
        return parsed_data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get cleaning metadata"""
        return {
            "transformer_type": "DataCleaner",
            "cleaning_steps": self.cleaning_steps,
            "config": {
                "handle_missing_values": self.config.handle_missing_values,
                "remove_duplicates": self.config.remove_duplicates,
                "type_inference": self.config.type_inference,
                "text_preprocessing": self.config.text_preprocessing,
                "date_parsing": self.config.date_parsing
            }
        }


class HypergraphConverter:
    """Convert various data formats to hypergraphs"""
    
    def __init__(self):
        self.conversion_strategies = {
            "transaction": self._transaction_to_hypergraph,
            "bipartite": self._bipartite_to_hypergraph,
            "multilayer": self._multilayer_to_hypergraph,
            "cooccurrence": self._cooccurrence_to_hypergraph,
            "hierarchical": self._hierarchical_to_hypergraph
        }
    
    def convert(self, data: pl.DataFrame, strategy: str, **kwargs) -> Hypergraph:
        """Convert data to hypergraph using specified strategy"""
        if strategy not in self.conversion_strategies:
            raise ValueError(f"Unknown conversion strategy: {strategy}")
        
        return self.conversion_strategies[strategy](data, **kwargs)
    
    def _transaction_to_hypergraph(self, data: pl.DataFrame, 
                                  transaction_col: str = "transaction_id",
                                  item_col: str = "item_id",
                                  weight_col: Optional[str] = None) -> Hypergraph:
        """Convert transaction data to hypergraph (transactions as hyperedges)"""
        hg = Hypergraph()
        
        # Group by transaction
        transaction_groups = data.group_by(transaction_col)
        
        for transaction_id, group in transaction_groups:
            items = group[item_col].to_list()
            weight = 1.0
            
            if weight_col and weight_col in group.columns:
                weights = group[weight_col].to_list()
                weight = sum(weights) / len(weights)  # Average weight
            
            hg.add_edge(transaction_id[0], items, weight=weight)
        
        return hg
    
    def _bipartite_to_hypergraph(self, data: pl.DataFrame,
                                source_col: str = "source",
                                target_col: str = "target",
                                weight_col: Optional[str] = None) -> Hypergraph:
        """Convert bipartite graph to hypergraph (sources as nodes, targets as hyperedges)"""
        hg = Hypergraph()
        
        # Group by target (which becomes hyperedge)
        target_groups = data.group_by(target_col)
        
        for target_id, group in target_groups:
            sources = group[source_col].to_list()
            weight = 1.0
            
            if weight_col and weight_col in group.columns:
                weights = group[weight_col].to_list()
                weight = sum(weights) / len(weights)
            
            hg.add_edge(target_id[0], sources, weight=weight)
        
        return hg
    
    def _multilayer_to_hypergraph(self, data: pl.DataFrame,
                                 node_col: str = "node",
                                 layer_col: str = "layer",
                                 group_col: Optional[str] = None) -> Hypergraph:
        """Convert multilayer network to hypergraph"""
        hg = Hypergraph()
        
        if group_col:
            # Group by specified column
            groups = data.group_by(group_col)
            for group_id, group_data in groups:
                nodes = group_data[node_col].to_list()
                hg.add_edge(f"group_{group_id[0]}", nodes)
        else:
            # Group by layer
            layer_groups = data.group_by(layer_col)
            for layer_id, layer_data in layer_groups:
                nodes = layer_data[node_col].to_list()
                hg.add_edge(f"layer_{layer_id[0]}", nodes)
        
        return hg
    
    def _cooccurrence_to_hypergraph(self, data: pl.DataFrame,
                                   entity_cols: List[str],
                                   threshold: int = 2) -> Hypergraph:
        """Convert co-occurrence data to hypergraph"""
        hg = Hypergraph()
        
        # Find co-occurring entities
        edge_counter = 0
        for row in data.iter_rows(named=True):
            entities = []
            for col in entity_cols:
                if row[col] is not None and str(row[col]).strip():
                    entities.append(str(row[col]))
            
            if len(entities) >= threshold:
                hg.add_edge(f"cooccurrence_{edge_counter}", entities)
                edge_counter += 1
        
        return hg
    
    def _hierarchical_to_hypergraph(self, data: pl.DataFrame,
                                   parent_col: str = "parent",
                                   child_col: str = "child") -> Hypergraph:
        """Convert hierarchical data to hypergraph"""
        hg = Hypergraph()
        
        # Group children by parent
        parent_groups = data.group_by(parent_col)
        
        for parent_id, group in parent_groups:
            children = group[child_col].to_list()
            children.append(parent_id[0])  # Include parent in hyperedge
            hg.add_edge(f"hierarchy_{parent_id[0]}", children)
        
        return hg


class FeatureEngineer:
    """Feature engineering for hypergraph data"""
    
    def __init__(self):
        self.features = {}
    
    def create_node_features(self, hg: Hypergraph) -> pl.DataFrame:
        """Create node-level features"""
        node_features = []
        
        for node in hg.nodes:
            features = {
                "node_id": node,
                "degree": hg.get_node_degree(node),
                "edge_count": len([e for e in hg.edges if node in hg.get_edge_nodes(e)]),
            }
            
            # Edge size statistics
            edge_sizes = [hg.get_edge_size(e) for e in hg.edges if node in hg.get_edge_nodes(e)]
            if edge_sizes:
                features.update({
                    "avg_edge_size": np.mean(edge_sizes),
                    "max_edge_size": np.max(edge_sizes),
                    "min_edge_size": np.min(edge_sizes),
                    "std_edge_size": np.std(edge_sizes)
                })
            else:
                features.update({
                    "avg_edge_size": 0,
                    "max_edge_size": 0,
                    "min_edge_size": 0,
                    "std_edge_size": 0
                })
            
            node_features.append(features)
        
        return pl.DataFrame(node_features)
    
    def create_edge_features(self, hg: Hypergraph) -> pl.DataFrame:
        """Create edge-level features"""
        edge_features = []
        
        for edge in hg.edges:
            edge_nodes = hg.get_edge_nodes(edge)
            features = {
                "edge_id": edge,
                "size": len(edge_nodes),
                "weight": hg.get_edge_weight(edge) if hasattr(hg, 'get_edge_weight') else 1.0,
            }
            
            # Node degree statistics within edge
            node_degrees = [hg.get_node_degree(node) for node in edge_nodes]
            features.update({
                "avg_node_degree": np.mean(node_degrees),
                "max_node_degree": np.max(node_degrees),
                "min_node_degree": np.min(node_degrees),
                "std_node_degree": np.std(node_degrees)
            })
            
            edge_features.append(features)
        
        return pl.DataFrame(edge_features)
    
    def create_global_features(self, hg: Hypergraph) -> Dict[str, Any]:
        """Create global hypergraph features"""
        return {
            "num_nodes": hg.num_nodes,
            "num_edges": hg.num_edges,
            "density": hg.num_edges / (hg.num_nodes * (hg.num_nodes - 1) / 2) if hg.num_nodes > 1 else 0,
            "avg_degree": np.mean([hg.get_node_degree(node) for node in hg.nodes]) if hg.nodes else 0,
            "avg_edge_size": np.mean([hg.get_edge_size(edge) for edge in hg.edges]) if hg.edges else 0,
        }


class ETLPipeline:
    """Extract, Transform, Load pipeline for hypergraphs"""
    
    def __init__(self):
        self.extractors = {}
        self.transformers = []
        self.loaders = {}
        self.pipeline_metadata = {}
    
    def add_extractor(self, name: str, extractor: Callable) -> None:
        """Add data extractor"""
        self.extractors[name] = extractor
    
    def add_transformer(self, transformer: DataTransformer) -> None:
        """Add data transformer"""
        self.transformers.append(transformer)
    
    def add_loader(self, name: str, loader: Callable) -> None:
        """Add data loader"""
        self.loaders[name] = loader
    
    def run(self, extractor_name: str, loader_name: str, 
            extract_params: Optional[Dict[str, Any]] = None,
            load_params: Optional[Dict[str, Any]] = None) -> Any:
        """Run the ETL pipeline"""
        extract_params = extract_params or {}
        load_params = load_params or {}
        
        # Extract
        if extractor_name not in self.extractors:
            raise ValueError(f"Unknown extractor: {extractor_name}")
        
        data = self.extractors[extractor_name](**extract_params)
        
        # Transform
        for transformer in self.transformers:
            data = transformer.transform(data)
            self.pipeline_metadata[transformer.__class__.__name__] = transformer.get_metadata()
        
        # Load
        if loader_name not in self.loaders:
            raise ValueError(f"Unknown loader: {loader_name}")
        
        result = self.loaders[loader_name](data, **load_params)
        
        return result
    
    def get_pipeline_metadata(self) -> Dict[str, Any]:
        """Get pipeline execution metadata"""
        return self.pipeline_metadata.copy()


# Convenience functions
def assess_data_quality(data: Union[pl.DataFrame, str, Path]) -> DataQualityReport:
    """Assess data quality from DataFrame or file"""
    if isinstance(data, (str, Path)):
        data = pl.read_csv(data)
    
    assessor = DataQualityAssessor()
    return assessor.assess(data)


def clean_data(data: Union[pl.DataFrame, str, Path], 
               config: Optional[TransformationConfig] = None) -> pl.DataFrame:
    """Clean data using default configuration"""
    if isinstance(data, (str, Path)):
        data = pl.read_csv(data)
    
    cleaner = DataCleaner(config)
    return cleaner.transform(data)


def convert_to_hypergraph(data: Union[pl.DataFrame, str, Path],
                         strategy: str = "transaction",
                         **kwargs) -> Hypergraph:
    """Convert data to hypergraph using specified strategy"""
    if isinstance(data, (str, Path)):
        data = pl.read_csv(data)
    
    converter = HypergraphConverter()
    return converter.convert(data, strategy, **kwargs)


def create_etl_pipeline() -> ETLPipeline:
    """Create a new ETL pipeline"""
    return ETLPipeline()