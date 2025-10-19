"""
Weight Analysis Module for Anant

This module provides comprehensive analysis capabilities for hypergraph weights,
including weight distribution analysis, optimization, normalization, and
weight-based clustering and pattern detection.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
import polars as pl
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import math

from .property_types import PropertyType, PropertyTypeManager


class WeightNormalizationType(Enum):
    """Types of weight normalization"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    UNIT_VECTOR = "unit_vector"
    SOFTMAX = "softmax"
    RANK = "rank"
    QUANTILE = "quantile"


class WeightDistributionType(Enum):
    """Types of weight distributions"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    LOG_NORMAL = "log_normal"
    UNKNOWN = "unknown"


@dataclass
class WeightStatistics:
    """Statistical properties of weights"""
    entity_type: str  # 'node' or 'edge'
    property_name: str
    total_count: int
    non_zero_count: int
    zero_count: int
    mean: float
    median: float
    std: float
    min_weight: float
    max_weight: float
    quartiles: Tuple[float, float, float]
    skewness: float
    kurtosis: float
    entropy: float
    sparsity: float  # Fraction of zero weights


@dataclass
class WeightDistribution:
    """Weight distribution analysis result"""
    distribution_type: WeightDistributionType
    parameters: Dict[str, float]
    goodness_of_fit: float
    confidence_level: float
    distribution_summary: str


@dataclass
class WeightCluster:
    """Weight-based cluster result"""
    cluster_id: int
    entity_indices: List[int]
    centroid_weights: Dict[str, float]
    cluster_size: int
    intra_cluster_similarity: float
    representative_entities: List[int]


class WeightAnalyzer:
    """
    Comprehensive analyzer for hypergraph weights
    
    Features:
    - Weight distribution analysis and characterization
    - Weight normalization and standardization
    - Weight-based clustering and pattern detection
    - Weight correlation and dependency analysis
    - Weight optimization and transformation
    - Sparse weight handling and compression
    """
    
    def __init__(self, sparse_threshold: float = 0.1):
        self.sparse_threshold = sparse_threshold
        self.type_manager = PropertyTypeManager()
        self.normalization_cache = {}
        self.distribution_cache = {}
        
    def analyze_weight_statistics(
        self,
        df: pl.DataFrame,
        weight_columns: List[str],
        entity_type: str = "unknown"
    ) -> Dict[str, WeightStatistics]:
        """
        Analyze statistical properties of weight columns
        
        Args:
            df: DataFrame containing weights
            weight_columns: List of weight column names
            entity_type: Type of entity ('node', 'edge', etc.)
            
        Returns:
            Dictionary mapping column names to WeightStatistics
        """
        results = {}
        
        for col in weight_columns:
            if col not in df.columns:
                continue
                
            # Ensure column is numerical
            if self.type_manager.detect_property_type(df[col]) != PropertyType.NUMERICAL:
                continue
                
            weights = df[col].drop_nulls()
            if len(weights) == 0:
                continue
                
            values = weights.to_numpy()
            
            # Basic statistics
            total_count = len(values)
            zero_count = np.sum(values == 0)
            non_zero_count = total_count - zero_count
            
            mean_val = float(np.mean(values))
            median_val = float(np.median(values))
            std_val = float(np.std(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            
            # Quartiles
            q1 = float(np.percentile(values, 25))
            q2 = float(np.percentile(values, 50))
            q3 = float(np.percentile(values, 75))
            
            # Advanced statistics
            from scipy import stats
            skewness = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))
            
            # Entropy calculation (for non-zero weights)
            entropy = 0.0
            if non_zero_count > 0:
                non_zero_values = values[values != 0]
                # Normalize to probabilities
                probs = np.abs(non_zero_values) / np.sum(np.abs(non_zero_values))
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            sparsity = zero_count / total_count
            
            stats_obj = WeightStatistics(
                entity_type=entity_type,
                property_name=col,
                total_count=total_count,
                non_zero_count=non_zero_count,
                zero_count=zero_count,
                mean=mean_val,
                median=median_val,
                std=std_val,
                min_weight=min_val,
                max_weight=max_val,
                quartiles=(q1, q2, q3),
                skewness=skewness,
                kurtosis=kurtosis,
                entropy=entropy,
                sparsity=sparsity
            )
            
            results[col] = stats_obj
            
        return results
    
    def detect_weight_distribution(
        self,
        df: pl.DataFrame,
        weight_column: str,
        confidence_threshold: float = 0.05
    ) -> WeightDistribution:
        """
        Detect the type of distribution that best fits the weights
        
        Args:
            df: DataFrame containing weights
            weight_column: Name of weight column
            confidence_threshold: Statistical significance threshold
            
        Returns:
            WeightDistribution object with best fit results
        """
        if weight_column not in df.columns:
            return WeightDistribution(
                distribution_type=WeightDistributionType.UNKNOWN,
                parameters={},
                goodness_of_fit=0.0,
                confidence_level=0.0,
                distribution_summary="Column not found"
            )
            
        weights = df[weight_column].drop_nulls()
        if len(weights) < 10:
            return WeightDistribution(
                distribution_type=WeightDistributionType.UNKNOWN,
                parameters={},
                goodness_of_fit=0.0,
                confidence_level=0.0,
                distribution_summary="Insufficient data"
            )
            
        values = weights.to_numpy()
        
        # Test different distributions
        distributions_to_test = [
            (WeightDistributionType.NORMAL, self._test_normal_distribution),
            (WeightDistributionType.EXPONENTIAL, self._test_exponential_distribution),
            (WeightDistributionType.LOG_NORMAL, self._test_lognormal_distribution),
            (WeightDistributionType.POWER_LAW, self._test_power_law_distribution),
            (WeightDistributionType.UNIFORM, self._test_uniform_distribution)
        ]
        
        best_fit = None
        best_score = 0.0
        
        for dist_type, test_func in distributions_to_test:
            try:
                result = test_func(values)
                if result and result.goodness_of_fit > best_score:
                    best_score = result.goodness_of_fit
                    best_fit = result
                    best_fit.distribution_type = dist_type
            except Exception:
                continue
                
        if best_fit is None:
            return WeightDistribution(
                distribution_type=WeightDistributionType.UNKNOWN,
                parameters={},
                goodness_of_fit=0.0,
                confidence_level=0.0,
                distribution_summary="No suitable distribution found"
            )
            
        return best_fit
    
    def _test_normal_distribution(self, values: np.ndarray) -> Optional[WeightDistribution]:
        """Test if weights follow normal distribution"""
        try:
            from scipy import stats
            
            # Fit normal distribution
            mu, sigma = stats.norm.fit(values)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(values, lambda x: stats.norm.cdf(x, mu, sigma))
            
            return WeightDistribution(
                distribution_type=WeightDistributionType.NORMAL,
                parameters={"mu": mu, "sigma": sigma},
                goodness_of_fit=1.0 - ks_stat,  # Convert to goodness measure
                confidence_level=p_value,
                distribution_summary=f"Normal(μ={mu:.3f}, σ={sigma:.3f})"
            )
        except Exception:
            return None
    
    def _test_exponential_distribution(self, values: np.ndarray) -> Optional[WeightDistribution]:
        """Test if weights follow exponential distribution"""
        if np.any(values < 0):
            return None  # Exponential only for non-negative values
            
        try:
            from scipy import stats
            
            # Fit exponential distribution
            loc, scale = stats.expon.fit(values)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(values, lambda x: stats.expon.cdf(x, loc, scale))
            
            return WeightDistribution(
                distribution_type=WeightDistributionType.EXPONENTIAL,
                parameters={"loc": loc, "scale": scale, "lambda": 1.0/scale},
                goodness_of_fit=1.0 - ks_stat,
                confidence_level=p_value,
                distribution_summary=f"Exponential(λ={1.0/scale:.3f})"
            )
        except Exception:
            return None
    
    def _test_lognormal_distribution(self, values: np.ndarray) -> Optional[WeightDistribution]:
        """Test if weights follow log-normal distribution"""
        if np.any(values <= 0):
            return None  # Log-normal only for positive values
            
        try:
            from scipy import stats
            
            # Fit log-normal distribution
            shape, loc, scale = stats.lognorm.fit(values)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(values, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
            
            return WeightDistribution(
                distribution_type=WeightDistributionType.LOG_NORMAL,
                parameters={"shape": shape, "loc": loc, "scale": scale},
                goodness_of_fit=1.0 - ks_stat,
                confidence_level=p_value,
                distribution_summary=f"LogNormal(shape={shape:.3f})"
            )
        except Exception:
            return None
    
    def _test_power_law_distribution(self, values: np.ndarray) -> Optional[WeightDistribution]:
        """Test if weights follow power law distribution"""
        if np.any(values <= 0):
            return None
            
        try:
            # Simple power law test using log-log regression
            log_values = np.log(values)
            log_ranks = np.log(np.arange(1, len(values) + 1))
            
            # Sort values in descending order
            sorted_vals = np.sort(values)[::-1]
            log_sorted = np.log(sorted_vals)
            log_ranks_sorted = np.log(np.arange(1, len(sorted_vals) + 1))
            
            # Linear regression in log-log space
            coeffs = np.polyfit(log_ranks_sorted, log_sorted, 1)
            alpha = -coeffs[0]  # Power law exponent
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_ranks_sorted)
            ss_res = np.sum((log_sorted - y_pred) ** 2)
            ss_tot = np.sum((log_sorted - np.mean(log_sorted)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return WeightDistribution(
                distribution_type=WeightDistributionType.POWER_LAW,
                parameters={"alpha": alpha, "r_squared": r_squared},
                goodness_of_fit=r_squared,
                confidence_level=0.0,  # Simplified test
                distribution_summary=f"PowerLaw(α={alpha:.3f})"
            )
        except Exception:
            return None
    
    def _test_uniform_distribution(self, values: np.ndarray) -> Optional[WeightDistribution]:
        """Test if weights follow uniform distribution"""
        try:
            from scipy import stats
            
            # Fit uniform distribution
            loc, scale = stats.uniform.fit(values)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(values, lambda x: stats.uniform.cdf(x, loc, scale))
            
            return WeightDistribution(
                distribution_type=WeightDistributionType.UNIFORM,
                parameters={"loc": loc, "scale": scale, "min": loc, "max": loc + scale},
                goodness_of_fit=1.0 - ks_stat,
                confidence_level=p_value,
                distribution_summary=f"Uniform({loc:.3f}, {loc+scale:.3f})"
            )
        except Exception:
            return None
    
    def normalize_weights(
        self,
        df: pl.DataFrame,
        weight_columns: List[str],
        normalization_type: WeightNormalizationType,
        **kwargs
    ) -> pl.DataFrame:
        """
        Normalize weight columns using specified method
        
        Args:
            df: DataFrame containing weights
            weight_columns: List of weight column names to normalize
            normalization_type: Type of normalization to apply
            **kwargs: Additional parameters for normalization
            
        Returns:
            DataFrame with normalized weights
        """
        result_df = df.clone()
        
        for col in weight_columns:
            if col not in df.columns:
                continue
                
            if normalization_type == WeightNormalizationType.MIN_MAX:
                result_df = self._normalize_min_max(result_df, col, **kwargs)
            elif normalization_type == WeightNormalizationType.Z_SCORE:
                result_df = self._normalize_z_score(result_df, col, **kwargs)
            elif normalization_type == WeightNormalizationType.UNIT_VECTOR:
                result_df = self._normalize_unit_vector(result_df, col, **kwargs)
            elif normalization_type == WeightNormalizationType.SOFTMAX:
                result_df = self._normalize_softmax(result_df, col, **kwargs)
            elif normalization_type == WeightNormalizationType.RANK:
                result_df = self._normalize_rank(result_df, col, **kwargs)
            elif normalization_type == WeightNormalizationType.QUANTILE:
                result_df = self._normalize_quantile(result_df, col, **kwargs)
                
        return result_df
    
    def _normalize_min_max(
        self,
        df: pl.DataFrame,
        column: str,
        feature_range: Tuple[float, float] = (0.0, 1.0)
    ) -> pl.DataFrame:
        """Min-max normalization"""
        min_val, max_val = feature_range
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max == col_min:
            # Handle constant values
            return df.with_columns([
                pl.lit(min_val).alias(column)
            ])
        
        return df.with_columns([
            ((pl.col(column) - col_min) / (col_max - col_min) * (max_val - min_val) + min_val).alias(column)
        ])
    
    def _normalize_z_score(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Z-score normalization"""
        col_mean = df[column].mean()
        col_std = df[column].std()
        
        if col_std == 0:
            # Handle constant values
            return df.with_columns([
                pl.lit(0.0).alias(column)
            ])
        
        return df.with_columns([
            ((pl.col(column) - col_mean) / col_std).alias(column)
        ])
    
    def _normalize_unit_vector(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Unit vector normalization (L2 norm)"""
        col_norm = (df[column] ** 2).sum() ** 0.5
        
        if col_norm == 0:
            return df
        
        return df.with_columns([
            (pl.col(column) / col_norm).alias(column)
        ])
    
    def _normalize_softmax(
        self,
        df: pl.DataFrame,
        column: str,
        temperature: float = 1.0
    ) -> pl.DataFrame:
        """Softmax normalization"""
        # Softmax with temperature scaling
        col_max = df[column].max()  # For numerical stability
        
        return df.with_columns([
            (pl.col(column) / temperature - col_max / temperature).exp().alias(f"{column}_exp")
        ]).with_columns([
            (pl.col(f"{column}_exp") / pl.col(f"{column}_exp").sum()).alias(column)
        ]).drop(f"{column}_exp")
    
    def _normalize_rank(
        self,
        df: pl.DataFrame,
        column: str,
        method: str = "average"
    ) -> pl.DataFrame:
        """Rank-based normalization"""
        # Convert to ranks and normalize by count
        n_rows = len(df)
        
        return df.with_columns([
            (pl.col(column).rank(method=method) / n_rows).alias(column)
        ])
    
    def _normalize_quantile(
        self,
        df: pl.DataFrame,
        column: str,
        n_quantiles: int = 100
    ) -> pl.DataFrame:
        """Quantile normalization"""
        # Map values to quantile ranks
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        values = df[column].to_numpy()
        
        # Calculate quantile values
        quantile_values = np.quantile(values, quantiles)
        
        # Map each value to its quantile
        quantile_ranks = np.searchsorted(quantile_values, values, side='right') - 1
        quantile_ranks = np.clip(quantile_ranks, 0, n_quantiles - 1)
        normalized = quantile_ranks / (n_quantiles - 1)
        
        return df.with_columns([
            pl.Series(column, normalized)
        ])
    
    def cluster_by_weights(
        self,
        df: pl.DataFrame,
        weight_columns: List[str],
        n_clusters: int = 5,
        clustering_method: str = "kmeans"
    ) -> List[WeightCluster]:
        """
        Cluster entities based on their weight patterns
        
        Args:
            df: DataFrame containing weights
            weight_columns: List of weight column names
            n_clusters: Number of clusters to create
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            List of WeightCluster objects
        """
        # Filter valid weight columns
        valid_columns = [
            col for col in weight_columns 
            if col in df.columns and 
            self.type_manager.detect_property_type(df[col]) == PropertyType.NUMERICAL
        ]
        
        if not valid_columns:
            return []
            
        # Extract weight matrix
        weight_data = df.select(valid_columns).fill_null(0.0)
        weight_matrix = weight_data.to_numpy()
        
        if clustering_method == "kmeans":
            return self._cluster_kmeans(weight_matrix, valid_columns, n_clusters)
        elif clustering_method == "dbscan":
            return self._cluster_dbscan(weight_matrix, valid_columns)
        elif clustering_method == "hierarchical":
            return self._cluster_hierarchical(weight_matrix, valid_columns, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    def _cluster_kmeans(
        self,
        weight_matrix: np.ndarray,
        column_names: List[str],
        n_clusters: int
    ) -> List[WeightCluster]:
        """K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(weight_matrix)
            centroids = kmeans.cluster_centers_
            
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0].tolist()
                if not cluster_indices:
                    continue
                    
                # Calculate centroid weights
                centroid_weights = {
                    col: float(centroids[i, j]) 
                    for j, col in enumerate(column_names)
                }
                
                # Calculate intra-cluster similarity
                cluster_points = weight_matrix[cluster_indices]
                centroid = centroids[i]
                
                # Average cosine similarity to centroid
                similarities = []
                for point in cluster_points:
                    sim = np.dot(point, centroid) / (np.linalg.norm(point) * np.linalg.norm(centroid) + 1e-10)
                    similarities.append(sim)
                
                avg_similarity = float(np.mean(similarities))
                
                # Find representative entities (closest to centroid)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                
                cluster = WeightCluster(
                    cluster_id=i,
                    entity_indices=cluster_indices,
                    centroid_weights=centroid_weights,
                    cluster_size=len(cluster_indices),
                    intra_cluster_similarity=avg_similarity,
                    representative_entities=[representative_idx]
                )
                clusters.append(cluster)
                
            return clusters
            
        except ImportError:
            return []
    
    def _cluster_dbscan(
        self,
        weight_matrix: np.ndarray,
        column_names: List[str],
        eps: float = 0.5,
        min_samples: int = 5
    ) -> List[WeightCluster]:
        """DBSCAN clustering"""
        try:
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(weight_matrix)
            
            clusters = []
            unique_labels = np.unique(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                cluster_indices = np.where(cluster_labels == label)[0].tolist()
                cluster_points = weight_matrix[cluster_indices]
                
                # Calculate centroid
                centroid = np.mean(cluster_points, axis=0)
                centroid_weights = {
                    col: float(centroid[j]) 
                    for j, col in enumerate(column_names)
                }
                
                # Calculate intra-cluster similarity
                similarities = []
                for point in cluster_points:
                    sim = np.dot(point, centroid) / (np.linalg.norm(point) * np.linalg.norm(centroid) + 1e-10)
                    similarities.append(sim)
                
                avg_similarity = float(np.mean(similarities))
                
                # Find representative entities
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                
                cluster = WeightCluster(
                    cluster_id=int(label),
                    entity_indices=cluster_indices,
                    centroid_weights=centroid_weights,
                    cluster_size=len(cluster_indices),
                    intra_cluster_similarity=avg_similarity,
                    representative_entities=[representative_idx]
                )
                clusters.append(cluster)
                
            return clusters
            
        except ImportError:
            return []
    
    def _cluster_hierarchical(
        self,
        weight_matrix: np.ndarray,
        column_names: List[str],
        n_clusters: int
    ) -> List[WeightCluster]:
        """Hierarchical clustering"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(weight_matrix)
            
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0].tolist()
                if not cluster_indices:
                    continue
                    
                cluster_points = weight_matrix[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                
                centroid_weights = {
                    col: float(centroid[j]) 
                    for j, col in enumerate(column_names)
                }
                
                # Calculate intra-cluster similarity
                similarities = []
                for point in cluster_points:
                    sim = np.dot(point, centroid) / (np.linalg.norm(point) * np.linalg.norm(centroid) + 1e-10)
                    similarities.append(sim)
                
                avg_similarity = float(np.mean(similarities))
                
                # Find representative entities
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                
                cluster = WeightCluster(
                    cluster_id=i,
                    entity_indices=cluster_indices,
                    centroid_weights=centroid_weights,
                    cluster_size=len(cluster_indices),
                    intra_cluster_similarity=avg_similarity,
                    representative_entities=[representative_idx]
                )
                clusters.append(cluster)
                
            return clusters
            
        except ImportError:
            return []
    
    def analyze_weight_correlations(
        self,
        df: pl.DataFrame,
        weight_columns: List[str],
        min_correlation: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze correlations between different weight properties
        
        Args:
            df: DataFrame containing weights
            weight_columns: List of weight column names
            min_correlation: Minimum correlation threshold
            
        Returns:
            Dictionary with correlation analysis results
        """
        valid_columns = [
            col for col in weight_columns 
            if col in df.columns and 
            self.type_manager.detect_property_type(df[col]) == PropertyType.NUMERICAL
        ]
        
        if len(valid_columns) < 2:
            return {"error": "Need at least 2 valid weight columns"}
            
        correlations = []
        
        # Compute pairwise correlations
        for i, col1 in enumerate(valid_columns):
            for col2 in valid_columns[i+1:]:
                # Clean data
                clean_df = df.select([col1, col2]).drop_nulls()
                if len(clean_df) < 3:
                    continue
                    
                vals1 = clean_df[col1].to_numpy()
                vals2 = clean_df[col2].to_numpy()
                
                # Pearson correlation
                corr_coeff = np.corrcoef(vals1, vals2)[0, 1]
                
                if abs(corr_coeff) >= min_correlation:
                    correlations.append({
                        "weight1": col1,
                        "weight2": col2,
                        "correlation": float(corr_coeff),
                        "sample_size": len(clean_df)
                    })
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "correlations": correlations,
            "summary": {
                "total_pairs": len(valid_columns) * (len(valid_columns) - 1) // 2,
                "significant_correlations": len(correlations),
                "strong_correlations": len([c for c in correlations if abs(c["correlation"]) > 0.7]),
                "moderate_correlations": len([c for c in correlations if 0.3 <= abs(c["correlation"]) <= 0.7])
            }
        }
    
    def optimize_weight_storage(
        self,
        df: pl.DataFrame,
        weight_columns: List[str],
        compression_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Optimize storage of weight columns through compression and type optimization
        
        Args:
            df: DataFrame containing weights
            weight_columns: List of weight column names
            compression_threshold: Sparsity threshold for compression
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        optimization_results = {}
        total_memory_before = 0
        total_memory_after = 0
        
        for col in weight_columns:
            if col not in df.columns:
                continue
                
            original_memory = df[col].estimated_size("bytes")
            total_memory_before += original_memory
            
            # Analyze sparsity
            stats = self.analyze_weight_statistics(df, [col])[col]
            
            recommendations = []
            optimized_memory = original_memory
            
            # Check if sparse representation would be beneficial
            if stats.sparsity > compression_threshold:
                recommendations.append(f"Consider sparse representation (sparsity: {stats.sparsity:.2%})")
                # Estimate sparse memory usage
                optimized_memory *= (1 - stats.sparsity) * 1.2  # 20% overhead for indices
            
            # Check for type optimization
            if stats.min_weight >= 0:
                if stats.max_weight <= 255:
                    recommendations.append("Consider UInt8 type")
                    optimized_memory *= 0.125  # 1 byte vs 8 bytes
                elif stats.max_weight <= 65535:
                    recommendations.append("Consider UInt16 type")
                    optimized_memory *= 0.25
                elif stats.max_weight <= 4294967295:
                    recommendations.append("Consider UInt32 type")
                    optimized_memory *= 0.5
            else:
                if stats.min_weight >= -128 and stats.max_weight <= 127:
                    recommendations.append("Consider Int8 type")
                    optimized_memory *= 0.125
                elif stats.min_weight >= -32768 and stats.max_weight <= 32767:
                    recommendations.append("Consider Int16 type")
                    optimized_memory *= 0.25
                elif stats.min_weight >= -2147483648 and stats.max_weight <= 2147483647:
                    recommendations.append("Consider Int32 type")
                    optimized_memory *= 0.5
            
            # Check for quantization opportunities
            unique_values = len(df[col].unique())
            if unique_values <= 256:
                recommendations.append(f"Consider quantization ({unique_values} unique values)")
            
            total_memory_after += optimized_memory
            
            optimization_results[col] = {
                "original_memory_bytes": original_memory,
                "optimized_memory_bytes": optimized_memory,
                "memory_reduction": (original_memory - optimized_memory) / original_memory,
                "sparsity": stats.sparsity,
                "unique_values": unique_values,
                "recommendations": recommendations
            }
        
        return {
            "column_results": optimization_results,
            "summary": {
                "total_memory_before_bytes": total_memory_before,
                "total_memory_after_bytes": total_memory_after,
                "total_memory_reduction": (total_memory_before - total_memory_after) / total_memory_before if total_memory_before > 0 else 0,
                "columns_analyzed": len(optimization_results)
            }
        }