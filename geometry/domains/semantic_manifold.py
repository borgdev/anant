"""
Semantic Manifold
=================

Language embeddings as a geometric manifold.

Key Insights:
- Word / sentence embeddings become coordinates
- Parallel transport captures analogies
- Curvature reveals polysemy or semantic drift
- Geodesic distance measures semantic similarity
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
    SEMANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy expected
    SEMANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalogyResult:
    base: str
    relation: str
    result: str
    confidence: float


class SemanticManifold:
    """Semantic space manifold for embeddings."""

    def __init__(
        self,
        embeddings: Dict[str, Iterable[float]],
        normalize: bool = True,
    ) -> None:
        if not SEMANTIC_AVAILABLE:
            raise RuntimeError("NumPy is required for SemanticManifold")
        if not embeddings:
            raise ValueError("embeddings must not be empty")

        self.normalize = normalize
        self.embeddings = {
            token: self._normalize(np.array(vec, dtype=float))
            for token, vec in embeddings.items()
        }
        self.tokens = list(self.embeddings.keys())
        self.dim = len(next(iter(self.embeddings.values())))

        self.token_index = {token: idx for idx, token in enumerate(self.tokens)}
        self.embedding_matrix = np.stack([self.embeddings[token] for token in self.tokens])

        logger.info(
            "SemanticManifold initialized with %d tokens (dim=%d)",
            len(self.tokens),
            self.dim,
        )

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vec
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    # ------------------------------------------------------------------
    # Geodesic distance & similarity
    # ------------------------------------------------------------------
    def geodesic_distance(self, token_a: str, token_b: str) -> float:
        vec_a = self.embeddings[token_a]
        vec_b = self.embeddings[token_b]
        diff = vec_a - vec_b
        return float(np.linalg.norm(diff))

    def cosine_similarity(self, token_a: str, token_b: str) -> float:
        vec_a = self.embeddings[token_a]
        vec_b = self.embeddings[token_b]
        return float(np.dot(vec_a, vec_b))

    # ------------------------------------------------------------------
    # Analogies via parallel transport
    # ------------------------------------------------------------------
    def analogy(self, token_a: str, token_b: str, token_c: str, top_k: int = 5) -> List[AnalogyResult]:
        vec_a = self.embeddings[token_a]
        vec_b = self.embeddings[token_b]
        vec_c = self.embeddings[token_c]

        relation = vec_b - vec_a
        target = vec_c + relation

        similarities = self.embedding_matrix @ target
        best_indices = np.argsort(-similarities)[: top_k + 3]

        results: List[AnalogyResult] = []
        for idx in best_indices:
            token = self.tokens[idx]
            if token in {token_a, token_b, token_c}:
                continue
            results.append(
                AnalogyResult(
                    base=token_c,
                    relation=f"{token_a}->{token_b}",
                    result=token,
                    confidence=float(similarities[idx]),
                )
            )
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Polysemy / semantic drift
    # ------------------------------------------------------------------
    def polysemy_score(self, token: str) -> float:
        vec = self.embeddings[token]
        neighbors = self.nearest_tokens(token, top_k=20)
        neighbor_vectors = np.stack([self.embeddings[n] for n in neighbors])
        covariance = np.cov(neighbor_vectors, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        variance_ratio = float(eigenvalues[-1] / (np.sum(eigenvalues) + 1e-12))
        return variance_ratio

    def nearest_tokens(self, token: str, top_k: int = 10) -> List[str]:
        vec = self.embeddings[token]
        similarities = self.embedding_matrix @ vec
        best_indices = np.argsort(-similarities)
        neighbors: List[str] = []
        for idx in best_indices:
            candidate = self.tokens[idx]
            if candidate == token:
                continue
            neighbors.append(candidate)
            if len(neighbors) >= top_k:
                break
        return neighbors

    def semantic_shift(self, token: str, other_embeddings: Dict[str, Iterable[float]]) -> float:
        if token not in other_embeddings:
            raise KeyError(f"Token '{token}' missing in other embedding set")
        vec_old = self.embeddings[token]
        vec_new = self._normalize(np.array(list(other_embeddings[token]), dtype=float))
        return float(np.linalg.norm(vec_new - vec_old))

    def summary(self) -> Dict[str, float]:
        return {
            "tokens": len(self.tokens),
            "dimension": self.dim,
        }
