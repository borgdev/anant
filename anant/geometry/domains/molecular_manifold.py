"""
Molecular Manifold
==================

Maps molecular conformations into configuration manifolds.

Highlights:
- Treats molecular descriptors as coordinates
- Uses PropertyManifold for metric / curvature
- Supports reaction pathway interpolation (geodesic)
- Detects strained rings (high curvature)
- Identifies similar conformers via geodesic distance
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    MOLECULAR_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy assumed present
    MOLECULAR_AVAILABLE = False

from ..core.property_manifold import PropertyManifold

logger = logging.getLogger(__name__)


@dataclass
class ConformerSimilarity:
    conformer_a: str
    conformer_b: str
    geodesic_distance: float
    rmsd: float


class MolecularManifold:
    """Configuration space manifold for molecules."""

    def __init__(
        self,
        conformer_properties: Dict[str, Dict[str, float]],
        energies: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not MOLECULAR_AVAILABLE:
            raise RuntimeError("NumPy is required for MolecularManifold")
        if not conformer_properties:
            raise ValueError("conformer_properties must not be empty")

        self.conformer_properties = conformer_properties
        self.energies = energies or {}
        self.manifold = PropertyManifold(
            property_vectors=conformer_properties,
            property_weights=weights,
        )
        logger.info(
            "MolecularManifold initialized with %d conformers and %d properties",
            len(conformer_properties),
            len(next(iter(conformer_properties.values()))),
        )

    def conformer_distance(self, conf_a: str, conf_b: str) -> float:
        return self.manifold.property_distance(conf_a, conf_b)

    def reaction_path(self, start_conf: str, end_conf: str, steps: int = 10) -> List[Dict[str, float]]:
        path: List[Dict[str, float]] = []
        for i in range(steps + 1):
            t = i / steps
            interpolated = self.manifold.geodesic_interpolation(start_conf, end_conf, t)
            path.append(interpolated)
        return path

    def curvature_stress(self) -> float:
        curvature = self.manifold.compute_curvature()
        return curvature.scalar_curvature

    def detect_strained_conformers(self, threshold: float = 2.5) -> List[str]:
        return self.manifold.detect_property_outliers(z_threshold=threshold)

    def compare_conformers(
        self,
        conf_a: str,
        conf_b: str,
        coordinates_a: Optional[Iterable[Iterable[float]]] = None,
        coordinates_b: Optional[Iterable[Iterable[float]]] = None,
    ) -> ConformerSimilarity:
        geodesic = self.conformer_distance(conf_a, conf_b)
        rmsd = 0.0
        if coordinates_a is not None and coordinates_b is not None:
            coords_a = np.array(list(coordinates_a))
            coords_b = np.array(list(coordinates_b))
            if coords_a.shape == coords_b.shape:
                diff = coords_a - coords_b
                rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        return ConformerSimilarity(
            conformer_a=conf_a,
            conformer_b=conf_b,
            geodesic_distance=geodesic,
            rmsd=rmsd,
        )

    def energy_landscape_curvature(self) -> float:
        if not self.energies:
            return 0.0
        energy_values = np.array(list(self.energies.values()))
        curvature = np.var(energy_values)
        return float(curvature)

    def summary(self) -> Dict[str, float]:
        metric = self.manifold.compute_metric()
        curvature = self.manifold.compute_curvature()
        return {
            "conformers": len(self.conformer_properties),
            "properties": len(metric.properties),
            "scalar_curvature": curvature.scalar_curvature,
            "condition_number": metric.condition_number(),
        }


def find_strained_conformers_geometric(
    conformer_properties: Dict[str, Dict[str, float]],
    threshold: float = 2.5,
    **kwargs: float,
) -> List[str]:
    """Return conformers with high curvature stress (potentially strained)."""

    manifold = MolecularManifold(conformer_properties=conformer_properties, **kwargs)
    return manifold.detect_strained_conformers(threshold=threshold)
