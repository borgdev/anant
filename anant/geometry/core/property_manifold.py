"""
Property Manifold
=================

Geometric framework for property space analytics.

Key Idea:
- Treat entity properties as coordinates on a manifold
- Metric tensor derived from property covariance / importance
- Curvature reveals correlations, redundancies, anomalies
- Geodesics provide natural interpolation in property space

Use-cases:
- Property-level similarity and clustering
- Correlation discovery via curvature analysis
- Property evolution trajectories (geodesics)
- Dimensional reduction with geometric meaning
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    from numpy.typing import ArrayLike
    PROPERTY_GEOMETRY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy must exist for geometry
    PROPERTY_GEOMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PropertyMetric:
    """Metric tensor represented as covariance matrix"""

    properties: List[str]
    matrix: np.ndarray
    inverse: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def condition_number(self) -> float:
        return float(np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-12))


@dataclass
class PropertyCurvature:
    """Curvature metadata for property subspace"""

    scalar_curvature: float
    sectional_curvature: Dict[Tuple[str, str], float]
    gaussian_curvature: Optional[float] = None


class PropertyManifold:
    """Manifold built from property vectors."""

    def __init__(
        self,
        property_vectors: Dict[str, Dict[str, float]],
        property_weights: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-9,
    ) -> None:
        if not PROPERTY_GEOMETRY_AVAILABLE:
            raise RuntimeError("NumPy is required for PropertyManifold")

        if not property_vectors:
            raise ValueError("`property_vectors` must not be empty")

        self.property_vectors = property_vectors
        self.property_weights = property_weights or {}
        self.epsilon = epsilon

        self.property_list = sorted({prop for props in property_vectors.values() for prop in props})
        self.entity_ids = list(property_vectors.keys())
        self._matrix_cache: Optional[np.ndarray] = None
        self._metric: Optional[PropertyMetric] = None

        logger.info(
            "PropertyManifold initialized with %d entities and %d properties",
            len(self.entity_ids),
            len(self.property_list),
        )

    # ------------------------------------------------------------------
    # Data preparation utilities
    # ------------------------------------------------------------------
    def _as_matrix(self) -> np.ndarray:
        if self._matrix_cache is not None:
            return self._matrix_cache

        matrix = np.zeros((len(self.entity_ids), len(self.property_list)), dtype=float)
        for i, entity_id in enumerate(self.entity_ids):
            props = self.property_vectors.get(entity_id, {})
            for j, prop in enumerate(self.property_list):
                matrix[i, j] = props.get(prop, 0.0)

        self._matrix_cache = matrix
        return matrix

    def get_property_matrix(self) -> np.ndarray:
        """Raw property matrix"""

        return self._as_matrix().copy()

    # ------------------------------------------------------------------
    # Metric construction
    # ------------------------------------------------------------------
    def compute_metric(self) -> PropertyMetric:
        if self._metric is not None:
            return self._metric

        matrix = self._as_matrix()
        # Weighted covariance matrix acts as metric tensor
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False)

        # Apply property weights directly in metric
        for i, prop in enumerate(self.property_list):
            weight = self.property_weights.get(prop, 1.0)
            covariance[i, :] *= weight
            covariance[:, i] *= weight

        # Regularize to ensure positive-definite
        covariance += self.epsilon * np.eye(len(self.property_list))

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse = np.linalg.inv(covariance)

        self._metric = PropertyMetric(
            properties=self.property_list,
            matrix=covariance,
            inverse=inverse,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        return self._metric

    # ------------------------------------------------------------------
    # Distances & similarity
    # ------------------------------------------------------------------
    def property_distance(self, entity_a: str, entity_b: str) -> float:
        metric = self.compute_metric()
        vec_a = self._vector_for(entity_a)
        vec_b = self._vector_for(entity_b)
        diff = vec_a - vec_b
        return float(np.sqrt(diff.T @ metric.inverse @ diff))

    def geodesic_interpolation(self, entity_a: str, entity_b: str, t: float) -> Dict[str, float]:
        vec_a = self._vector_for(entity_a)
        vec_b = self._vector_for(entity_b)
        interpolated = (1 - t) * vec_a + t * vec_b
        return {
            prop: float(interpolated[idx])
            for idx, prop in enumerate(self.property_list)
        }

    # ------------------------------------------------------------------
    # Curvature-based analytics
    # ------------------------------------------------------------------
    def compute_curvature(self) -> PropertyCurvature:
        metric = self.compute_metric()
        covariance = metric.matrix
        inverse = metric.inverse

        # Sectional curvature approximated via determinant ratios
        sectional_curvature: Dict[Tuple[str, str], float] = {}
        dim = covariance.shape[0]
        for i in range(dim):
            for j in range(i + 1, dim):
                minor = covariance[np.ix_([i, j], [i, j])]
                det_minor = np.linalg.det(minor)
                sectional_curvature[(self.property_list[i], self.property_list[j])] = float(det_minor)

        # Scalar curvature: trace(g^-1)
        scalar_curvature = float(np.trace(inverse))

        gaussian_curvature = None
        if dim == 2:
            gaussian_curvature = sectional_curvature[(self.property_list[0], self.property_list[1])]

        return PropertyCurvature(
            scalar_curvature=scalar_curvature,
            sectional_curvature=sectional_curvature,
            gaussian_curvature=gaussian_curvature,
        )

    def find_correlated_properties(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        metric = self.compute_metric()
        covariance = metric.matrix
        std = np.sqrt(np.diag(covariance))
        correlations: List[Tuple[str, str, float]] = []

        for i in range(len(self.property_list)):
            for j in range(i + 1, len(self.property_list)):
                denom = std[i] * std[j] + self.epsilon
                corr = covariance[i, j] / denom
                if abs(corr) >= threshold:
                    correlations.append((self.property_list[i], self.property_list[j], float(corr)))
        return correlations

    def property_variance(self) -> Dict[str, float]:
        metric = self.compute_metric()
        variances = np.diag(metric.matrix)
        return {prop: float(var) for prop, var in zip(self.property_list, variances)}

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def detect_property_outliers(self, z_threshold: float = 3.0) -> List[str]:
        metric = self.compute_metric()
        inverse = metric.inverse
        matrix = self._as_matrix()
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)

        # Mahalanobis distance in property space
        mahalanobis = np.sqrt(np.sum(centered @ inverse * centered, axis=1))
        mean = float(np.mean(mahalanobis))
        std = float(np.std(mahalanobis) + self.epsilon)

        outliers = []
        for idx, value in enumerate(mahalanobis):
            if abs(value - mean) / std > z_threshold:
                outliers.append(self.entity_ids[idx])
        return outliers

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _vector_for(self, entity_id: str) -> np.ndarray:
        props = self.property_vectors.get(entity_id)
        if props is None:
            raise KeyError(f"Entity '{entity_id}' not found in property manifold")

        vector = np.zeros(len(self.property_list), dtype=float)
        for idx, prop in enumerate(self.property_list):
            vector[idx] = props.get(prop, 0.0)
        return vector

    def summary(self) -> Dict[str, float]:
        metric = self.compute_metric()
        curvature = self.compute_curvature()
        return {
            "entities": len(self.entity_ids),
            "properties": len(self.property_list),
            "scalar_curvature": curvature.scalar_curvature,
            "condition_number": metric.condition_number(),
        }


def build_property_manifold(
    property_vectors: Dict[str, Dict[str, float]],
    property_weights: Optional[Dict[str, float]] = None,
) -> PropertyManifold:
    """Convenience function"""

    return PropertyManifold(property_vectors=property_vectors, property_weights=property_weights)
