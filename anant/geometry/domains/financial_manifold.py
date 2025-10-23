"""
Financial Manifold
==================

Risk geometry for portfolios and markets.

Concepts:
- Assets as points in manifold with covariance metric
- Geodesics represent diversification trajectories
- Curvature measures volatility / systemic stress
- Negative curvature highlights diversification benefits

Supports rapid risk diagnostics without bespoke quant code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    FINANCE_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy assumed present
    FINANCE_AVAILABLE = False

from ..core.property_manifold import PropertyManifold

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRisk:
    asset_a: str
    asset_b: str
    geodesic_distance: float
    correlation: float


class FinancialManifold:
    """Risk-aware manifold for financial assets."""

    def __init__(
        self,
        returns_matrix: Dict[str, List[float]],
        risk_weights: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-9,
    ) -> None:
        if not FINANCE_AVAILABLE:
            raise RuntimeError("NumPy is required for FinancialManifold")
        if not returns_matrix:
            raise ValueError("returns_matrix must not be empty")

        self.returns_matrix = returns_matrix
        self.asset_ids = list(returns_matrix.keys())
        self.epsilon = epsilon

        self.property_manifold = PropertyManifold(
            property_vectors=self._build_property_vectors(),
            property_weights=risk_weights,
        )

        logger.info(
            "FinancialManifold initialized with %d assets and %d time steps",
            len(self.asset_ids),
            len(next(iter(returns_matrix.values()))),
        )

    def _build_property_vectors(self) -> Dict[str, Dict[str, float]]:
        property_vectors: Dict[str, Dict[str, float]] = {}
        for asset, returns in self.returns_matrix.items():
            stats = {
                "mean_return": float(np.mean(returns)),
                "volatility": float(np.std(returns)),
                "skewness": float(self._skewness(returns)),
                "kurtosis": float(self._kurtosis(returns)),
            }
            property_vectors[asset] = stats
        return property_vectors

    # ------------------------------------------------------------------
    # Basic moments
    # ------------------------------------------------------------------
    def _skewness(self, data: Iterable[float]) -> float:
        arr = np.array(list(data))
        mean = np.mean(arr)
        std = np.std(arr) + self.epsilon
        return float(np.mean(((arr - mean) / std) ** 3))

    def _kurtosis(self, data: Iterable[float]) -> float:
        arr = np.array(list(data))
        mean = np.mean(arr)
        std = np.std(arr) + self.epsilon
        return float(np.mean(((arr - mean) / std) ** 4) - 3)

    # ------------------------------------------------------------------
    # Risk diagnostics
    # ------------------------------------------------------------------
    def geodesic_risk(self, asset_a: str, asset_b: str) -> PortfolioRisk:
        distance = self.property_manifold.property_distance(asset_a, asset_b)
        correlation = self.asset_correlation(asset_a, asset_b)
        return PortfolioRisk(
            asset_a=asset_a,
            asset_b=asset_b,
            geodesic_distance=distance,
            correlation=correlation,
        )

    def asset_correlation(self, asset_a: str, asset_b: str) -> float:
        returns_a = np.array(self.returns_matrix[asset_a])
        returns_b = np.array(self.returns_matrix[asset_b])
        return float(np.corrcoef(returns_a, returns_b)[0, 1])

    def curvature_signal(self) -> float:
        curvature = self.property_manifold.compute_curvature()
        return curvature.scalar_curvature

    def diversification_benefit(self, asset_a: str, asset_b: str) -> float:
        risk = self.geodesic_risk(asset_a, asset_b)
        return max(0.0, risk.geodesic_distance - abs(risk.correlation))

    def systemic_risk_index(self) -> float:
        metric = self.property_manifold.compute_metric()
        eigenvalues = metric.eigenvalues
        return float(np.sum(np.log(eigenvalues + self.epsilon)))

    def summary(self) -> Dict[str, float]:
        curvature = self.property_manifold.compute_curvature()
        return {
            "assets": len(self.asset_ids),
            "scalar_curvature": curvature.scalar_curvature,
            "max_sectional": max(curvature.sectional_curvature.values(), default=0.0),
            "systemic_risk": self.systemic_risk_index(),
        }


def compute_risk_geometric(
    returns_matrix: Dict[str, List[float]],
    asset_a: str,
    asset_b: str,
    **kwargs: float,
) -> PortfolioRisk:
    """Convenience wrapper returning geodesic risk between two assets."""

    manifold = FinancialManifold(returns_matrix=returns_matrix, **kwargs)
    return manifold.geodesic_risk(asset_a=asset_a, asset_b=asset_b)
