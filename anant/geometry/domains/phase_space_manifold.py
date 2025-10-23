"""
Phase Space Manifold
====================

Geometric view of dynamical systems.

Concepts:
- State trajectories as geodesic flows on manifold
- Hamiltonian / energy defines metric weights
- Curvature indicates stability vs chaos
- Invariants detected as Killing vectors (approximate)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    PHASE_AVAILABLE = True
except ImportError:  # pragma: no cover
    PHASE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OrbitStability:
    trajectory_id: str
    lyapunov_exponent: float
    stable: bool


class PhaseSpaceManifold:
    """Phase space manifold for dynamical systems."""

    def __init__(
        self,
        trajectories: Dict[str, np.ndarray],
        energy_function: Optional[Callable[[np.ndarray], float]] = None,
        time_step: float = 1.0,
    ) -> None:
        if not PHASE_AVAILABLE:
            raise RuntimeError("NumPy required for PhaseSpaceManifold")
        if not trajectories:
            raise ValueError("trajectories must not be empty")

        self.trajectories = trajectories
        self.energy_function = energy_function or (lambda x: float(np.linalg.norm(x) ** 2))
        self.time_step = time_step

        self.dim = next(iter(trajectories.values())).shape[1]
        logger.info(
            "PhaseSpaceManifold initialized with %d trajectories (dim=%d)",
            len(trajectories),
            self.dim,
        )

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------
    def lyapunov_exponents(self) -> List[OrbitStability]:
        results: List[OrbitStability] = []
        for traj_id, traj in self.trajectories.items():
            exponent = self._estimate_lyapunov(traj)
            results.append(
                OrbitStability(
                    trajectory_id=traj_id,
                    lyapunov_exponent=exponent,
                    stable=exponent < 0,
                )
            )
        results.sort(key=lambda r: r.lyapunov_exponent, reverse=True)
        return results

    def _estimate_lyapunov(self, trajectory: np.ndarray) -> float:
        diffs = np.diff(trajectory, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        norms = norms[norms > 0]
        if norms.size == 0:
            return 0.0
        exponent = np.mean(np.log(norms)) / self.time_step
        return float(exponent)

    # ------------------------------------------------------------------
    # Curvature / energy diagnostics
    # ------------------------------------------------------------------
    def energy_profile(self, trajectory_id: str) -> np.ndarray:
        traj = self.trajectories[trajectory_id]
        energies = np.array([self.energy_function(point) for point in traj])
        return energies

    def curvature_signal(self, trajectory_id: str) -> float:
        energies = self.energy_profile(trajectory_id)
        curvature = float(np.var(np.gradient(energies)))
        return curvature

    def detect_chaotic_orbits(self, threshold: float = 0.1) -> List[str]:
        lyapunov = self.lyapunov_exponents()
        return [item.trajectory_id for item in lyapunov if item.lyapunov_exponent > threshold]

    # ------------------------------------------------------------------
    # Invariant detection (approximate)
    # ------------------------------------------------------------------
    def approximate_invariants(self, tolerance: float = 1e-3) -> Dict[str, List[int]]:
        invariants: Dict[str, List[int]] = {}
        for traj_id, traj in self.trajectories.items():
            energies = self.energy_profile(traj_id)
            constant_indices = list(np.where(np.abs(np.gradient(energies)) < tolerance)[0])
            invariants[traj_id] = constant_indices
        return invariants

    def summary(self) -> Dict[str, float]:
        lyapunov = self.lyapunov_exponents()
        return {
            "trajectories": len(self.trajectories),
            "dimension": self.dim,
            "max_lyapunov": lyapunov[0].lyapunov_exponent if lyapunov else 0.0,
        }
