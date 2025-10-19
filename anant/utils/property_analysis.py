"""
Property Analysis Framework for Anant

This module provides comprehensive property analysis capabilities including
correlation analysis, anomaly detection, distribution analysis, and property
relationship discovery for hypergraph properties.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Set
from enum import Enum
import polars as pl
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json

from .property_types import PropertyType, PropertyTypeManager


class CorrelationType(Enum):
    """Types of correlation analysis"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    MUTUAL_INFO = "mutual_information"


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    ISOLATION_FOREST = "isolation_forest"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    property1: str
    property2: str
    correlation_type: CorrelationType
    correlation_value: float
    p_value: Optional[float] = None
    significance_level: float = 0.05
    is_significant: bool = False
    sample_size: int = 0


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    property_name: str
    anomaly_type: AnomalyType
    anomalous_indices: List[int]
    anomaly_scores: List[float]
    threshold: float
    detection_params: Dict[str, Any]


@dataclass
class PropertyDistribution:
    """Statistical distribution information for a property"""
    property_name: str
    property_type: PropertyType
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    quartiles: Optional[Tuple[float, float, float]] = None
    mode: Optional[Any] = None
    unique_count: int = 0
    null_count: int = 0
    total_count: int = 0
    distribution_params: Dict[str, Any] = None


class PropertyAnalysisFramework:
    """
    Comprehensive framework for analyzing hypergraph properties
    
    Features:
    - Correlation analysis between properties
    - Anomaly detection in property values
    - Distribution analysis and characterization
    - Property relationship discovery
    - Statistical significance testing
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.type_manager = PropertyTypeManager()
        self.analysis_cache = {}
        self.correlation_cache = {}
        
    def analyze_property_correlations(
        self,
        df: pl.DataFrame,
        properties: Optional[List[str]] = None,
        correlation_types: List[CorrelationType] = None,
        min_correlation: float = 0.1
    ) -> List[CorrelationResult]:
        """
        Analyze correlations between properties
        
        Args:
            df: DataFrame containing properties
            properties: List of property names to analyze (None for all)
            correlation_types: Types of correlation to compute
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of correlation results
        """
        if properties is None:
            properties = df.columns
            
        if correlation_types is None:
            correlation_types = [CorrelationType.PEARSON, CorrelationType.SPEARMAN]
            
        results = []
        
        # Filter to numerical properties for correlation analysis
        numerical_props = []
        for prop in properties:
            if prop in df.columns:
                prop_type = self.type_manager.detect_property_type(df[prop])
                if prop_type == PropertyType.NUMERICAL:
                    numerical_props.append(prop)
        
        # Compute pairwise correlations
        for i, prop1 in enumerate(numerical_props):
            for prop2 in numerical_props[i+1:]:
                for corr_type in correlation_types:
                    result = self._compute_correlation(df, prop1, prop2, corr_type)
                    if result and abs(result.correlation_value) >= min_correlation:
                        results.append(result)
        
        return sorted(results, key=lambda x: abs(x.correlation_value), reverse=True)
    
    def _compute_correlation(
        self,
        df: pl.DataFrame,
        prop1: str,
        prop2: str,
        corr_type: CorrelationType
    ) -> Optional[CorrelationResult]:
        """Compute correlation between two properties"""
        try:
            # Get non-null pairs
            clean_df = df.select([prop1, prop2]).drop_nulls()
            if len(clean_df) < 3:  # Need at least 3 points
                return None
                
            vals1 = clean_df[prop1].to_numpy()
            vals2 = clean_df[prop2].to_numpy()
            
            correlation_value = 0.0
            p_value = None
            
            if corr_type == CorrelationType.PEARSON:
                correlation_value = np.corrcoef(vals1, vals2)[0, 1]
                
            elif corr_type == CorrelationType.SPEARMAN:
                from scipy.stats import spearmanr
                correlation_value, p_value = spearmanr(vals1, vals2)
                
            elif corr_type == CorrelationType.KENDALL:
                from scipy.stats import kendalltau
                correlation_value, p_value = kendalltau(vals1, vals2)
                
            elif corr_type == CorrelationType.MUTUAL_INFO:
                from sklearn.feature_selection import mutual_info_regression
                correlation_value = mutual_info_regression(
                    vals1.reshape(-1, 1), vals2
                )[0]
            
            is_significant = (
                p_value is not None and p_value < self.significance_level
            ) if p_value is not None else False
            
            return CorrelationResult(
                property1=prop1,
                property2=prop2,
                correlation_type=corr_type,
                correlation_value=correlation_value,
                p_value=p_value,
                significance_level=self.significance_level,
                is_significant=is_significant,
                sample_size=len(clean_df)
            )
            
        except Exception as e:
            # Return None for failed correlations
            return None
    
    def detect_property_anomalies(
        self,
        df: pl.DataFrame,
        property_name: str,
        anomaly_types: List[AnomalyType] = None,
        contamination: float = 0.1
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in property values
        
        Args:
            df: DataFrame containing the property
            property_name: Name of property to analyze
            anomaly_types: Types of anomaly detection to perform
            contamination: Expected proportion of anomalies
            
        Returns:
            List of anomaly detection results
        """
        if property_name not in df.columns:
            return []
            
        if anomaly_types is None:
            anomaly_types = [AnomalyType.STATISTICAL_OUTLIER, AnomalyType.ISOLATION_FOREST]
            
        results = []
        prop_data = df[property_name].drop_nulls()
        
        if len(prop_data) < 10:  # Need sufficient data
            return results
            
        prop_type = self.type_manager.detect_property_type(prop_data)
        
        for anomaly_type in anomaly_types:
            result = self._detect_anomalies(
                df, property_name, prop_type, anomaly_type, contamination
            )
            if result:
                results.append(result)
                
        return results
    
    def _detect_anomalies(
        self,
        df: pl.DataFrame,
        property_name: str,
        prop_type: PropertyType,
        anomaly_type: AnomalyType,
        contamination: float
    ) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using specified method"""
        try:
            prop_data = df[property_name]
            
            if anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
                return self._detect_statistical_outliers(df, property_name, prop_type)
                
            elif anomaly_type == AnomalyType.ISOLATION_FOREST:
                return self._detect_isolation_forest_anomalies(
                    df, property_name, prop_type, contamination
                )
                
            elif anomaly_type == AnomalyType.DISTRIBUTION_ANOMALY:
                return self._detect_distribution_anomalies(df, property_name, prop_type)
                
            elif anomaly_type == AnomalyType.PATTERN_ANOMALY:
                return self._detect_pattern_anomalies(df, property_name, prop_type)
                
        except Exception:
            return None
            
        return None
    
    def _detect_statistical_outliers(
        self,
        df: pl.DataFrame,
        property_name: str,
        prop_type: PropertyType
    ) -> Optional[AnomalyDetectionResult]:
        """Detect statistical outliers using IQR method"""
        if prop_type != PropertyType.NUMERICAL:
            return None
            
        prop_data = df[property_name].drop_nulls()
        values = prop_data.to_numpy()
        
        # IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        
        # Calculate anomaly scores as distance from nearest bound
        anomaly_scores = []
        for val in values:
            if val < lower_bound:
                score = (lower_bound - val) / iqr
            elif val > upper_bound:
                score = (val - upper_bound) / iqr
            else:
                score = 0.0
            anomaly_scores.append(score)
        
        return AnomalyDetectionResult(
            property_name=property_name,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            anomalous_indices=outlier_indices,
            anomaly_scores=anomaly_scores,
            threshold=1.5,  # IQR multiplier
            detection_params={
                'method': 'IQR',
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        )
    
    def _detect_isolation_forest_anomalies(
        self,
        df: pl.DataFrame,
        property_name: str,
        prop_type: PropertyType,
        contamination: float
    ) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using Isolation Forest"""
        if prop_type != PropertyType.NUMERICAL:
            return None
            
        try:
            from sklearn.ensemble import IsolationForest
            
            prop_data = df[property_name].drop_nulls()
            values = prop_data.to_numpy().reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            predictions = iso_forest.fit_predict(values)
            anomaly_scores = iso_forest.score_samples(values)
            
            # Find anomalies (predictions == -1)
            anomalous_indices = np.where(predictions == -1)[0].tolist()
            
            return AnomalyDetectionResult(
                property_name=property_name,
                anomaly_type=AnomalyType.ISOLATION_FOREST,
                anomalous_indices=anomalous_indices,
                anomaly_scores=(-anomaly_scores).tolist(),  # Convert to positive scores
                threshold=contamination,
                detection_params={
                    'contamination': contamination,
                    'n_estimators': 100
                }
            )
            
        except ImportError:
            return None
    
    def _detect_distribution_anomalies(
        self,
        df: pl.DataFrame,
        property_name: str,
        prop_type: PropertyType
    ) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies based on distribution fit"""
        # Placeholder for distribution-based anomaly detection
        # Would implement distribution fitting and outlier detection
        return None
    
    def _detect_pattern_anomalies(
        self,
        df: pl.DataFrame,
        property_name: str,
        prop_type: PropertyType
    ) -> Optional[AnomalyDetectionResult]:
        """Detect pattern-based anomalies"""
        # Placeholder for pattern-based anomaly detection
        # Would implement sequence/pattern analysis
        return None
    
    def analyze_property_distribution(
        self,
        df: pl.DataFrame,
        property_name: str
    ) -> PropertyDistribution:
        """
        Analyze statistical distribution of a property
        
        Args:
            df: DataFrame containing the property
            property_name: Name of property to analyze
            
        Returns:
            PropertyDistribution object with statistics
        """
        if property_name not in df.columns:
            return PropertyDistribution(
                property_name=property_name,
                property_type=PropertyType.UNKNOWN
            )
            
        prop_data = df[property_name]
        prop_type = self.type_manager.detect_property_type(prop_data)
        
        total_count = len(prop_data)
        null_count = prop_data.null_count()
        clean_data = prop_data.drop_nulls()
        unique_count = len(clean_data.unique())
        
        distribution = PropertyDistribution(
            property_name=property_name,
            property_type=prop_type,
            total_count=total_count,
            null_count=null_count,
            unique_count=unique_count
        )
        
        if prop_type == PropertyType.NUMERICAL and len(clean_data) > 0:
            values = clean_data.to_numpy()
            distribution.mean = float(np.mean(values))
            distribution.median = float(np.median(values))
            distribution.std = float(np.std(values))
            distribution.min_val = float(np.min(values))
            distribution.max_val = float(np.max(values))
            distribution.quartiles = (
                float(np.percentile(values, 25)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 75))
            )
            
        elif prop_type == PropertyType.CATEGORICAL and len(clean_data) > 0:
            # Find mode for categorical data
            value_counts = clean_data.value_counts()
            if len(value_counts) > 0:
                distribution.mode = value_counts[0, 0]  # Most frequent value
                
        return distribution
    
    def discover_property_relationships(
        self,
        df: pl.DataFrame,
        target_property: str,
        min_correlation: float = 0.3,
        max_relationships: int = 10
    ) -> Dict[str, Any]:
        """
        Discover relationships between target property and other properties
        
        Args:
            df: DataFrame containing properties
            target_property: Property to find relationships for
            min_correlation: Minimum correlation threshold
            max_relationships: Maximum number of relationships to return
            
        Returns:
            Dictionary with relationship analysis results
        """
        if target_property not in df.columns:
            return {"error": f"Property {target_property} not found"}
            
        # Get correlations with target property
        other_properties = [col for col in df.columns if col != target_property]
        correlations = []
        
        for prop in other_properties:
            for corr_type in [CorrelationType.PEARSON, CorrelationType.SPEARMAN]:
                result = self._compute_correlation(df, target_property, prop, corr_type)
                if result and abs(result.correlation_value) >= min_correlation:
                    correlations.append(result)
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x.correlation_value), reverse=True)
        correlations = correlations[:max_relationships]
        
        # Analyze target property distribution
        target_distribution = self.analyze_property_distribution(df, target_property)
        
        # Find potential confounding variables
        confounders = self._find_confounding_variables(df, target_property, correlations)
        
        return {
            "target_property": target_property,
            "distribution": target_distribution,
            "correlations": correlations,
            "confounding_variables": confounders,
            "analysis_summary": {
                "total_relationships": len(correlations),
                "strong_correlations": len([c for c in correlations if abs(c.correlation_value) > 0.7]),
                "moderate_correlations": len([c for c in correlations if 0.3 <= abs(c.correlation_value) <= 0.7]),
                "significant_relationships": len([c for c in correlations if c.is_significant])
            }
        }
    
    def _find_confounding_variables(
        self,
        df: pl.DataFrame,
        target_property: str,
        correlations: List[CorrelationResult]
    ) -> List[Dict[str, Any]]:
        """Find potential confounding variables"""
        confounders = []
        
        # Look for variables that correlate with both target and other correlated variables
        correlated_props = [c.property2 if c.property1 == target_property else c.property1 
                          for c in correlations]
        
        for prop1 in correlated_props:
            for prop2 in correlated_props:
                if prop1 != prop2:
                    # Check if prop1 and prop2 are also correlated
                    cross_corr = self._compute_correlation(df, prop1, prop2, CorrelationType.PEARSON)
                    if cross_corr and abs(cross_corr.correlation_value) > 0.5:
                        confounders.append({
                            "variable1": prop1,
                            "variable2": prop2,
                            "correlation": cross_corr.correlation_value,
                            "potential_confounder": True
                        })
        
        return confounders
    
    def generate_analysis_report(
        self,
        df: pl.DataFrame,
        properties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for properties
        
        Args:
            df: DataFrame to analyze
            properties: Properties to include (None for all)
            
        Returns:
            Comprehensive analysis report
        """
        if properties is None:
            properties = df.columns
            
        report = {
            "dataset_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "analyzed_properties": len(properties)
            },
            "property_distributions": {},
            "correlations": [],
            "anomalies": {},
            "property_relationships": {}
        }
        
        # Analyze each property
        for prop in properties:
            if prop in df.columns:
                # Distribution analysis
                report["property_distributions"][prop] = self.analyze_property_distribution(df, prop)
                
                # Anomaly detection
                anomalies = self.detect_property_anomalies(df, prop)
                if anomalies:
                    report["anomalies"][prop] = anomalies
        
        # Correlation analysis
        report["correlations"] = self.analyze_property_correlations(df, properties)
        
        # Relationship discovery for key properties
        numerical_props = [
            prop for prop in properties 
            if prop in df.columns and 
            self.type_manager.detect_property_type(df[prop]) == PropertyType.NUMERICAL
        ]
        
        for prop in numerical_props[:5]:  # Limit to top 5 for performance
            report["property_relationships"][prop] = self.discover_property_relationships(df, prop)
        
        return report