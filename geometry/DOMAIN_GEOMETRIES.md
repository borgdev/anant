# Domain Geometries 2.0 🌀

## Overview

Each data modality receives its own Riemannian interpretation. Curvature, geodesics, and vector flows expose anomalies, clusters, and pathways without bespoke algorithms.

---

## Property Space (`PropertyManifold`)

- Metric tensor = weighted covariance across properties.
- Scalar curvature measures global correlation strength.
- Mahalanobis geodesics give true property-aware distance.
- Outliers pop out as high-curvature entities.

```python
from geometry.core import PropertyManifold

pm = PropertyManifold(property_vectors)
curvature = pm.compute_curvature()
outliers = pm.detect_property_outliers()
```

---

## Time Series (`TimeSeriesManifold`)

- Delay embeddings convert sequences into smooth trajectories.
- Closed geodesics reveal periodicity automatically.
- Curvature spikes correspond to temporal anomalies.
- Geodesic extension yields structure-aware forecasts.

```python
from geometry.domains import TimeSeriesManifold

ts = TimeSeriesManifold(series)
cycles = ts.find_closed_geodesics()
anomalies = ts.detect_curvature_anomalies()
forecast = ts.geodesic_forecast(steps=24)
```

---

## Network Flow (`NetworkFlowManifold`)

- Edge flows map to tangent vectors on the manifold.
- Divergence isolates bottlenecks and influence sources.
- Curl magnitude highlights feedback loops / echo chambers.
- Wrapper: `find_bottlenecks_geometric()`.

```python
from geometry.domains import find_bottlenecks_geometric

hotspots = find_bottlenecks_geometric(graph, flow_attribute="traffic", top_k=5)
```

---

## Finance (`FinancialManifold`)

- Asset statistics define coordinates (mean, volatility, skew, kurtosis).
- Geodesic distance quantifies diversification power.
- Curvature spikes warn about systemic stress.
- Wrapper: `compute_risk_geometric()`.

```python
from geometry.domains import compute_risk_geometric

risk = compute_risk_geometric(returns_matrix, "AAPL", "TSLA")
print(risk.geodesic_distance, risk.correlation)
```

---

## Molecular Configuration (`MolecularManifold`)

- Conformer descriptors embed structures on the manifold.
- Geodesic interpolation traces reaction pathways.
- High curvature detects strained rings or unstable conformers.
- Wrapper: `find_strained_conformers_geometric()`.

```python
from geometry.domains import find_strained_conformers_geometric

strained = find_strained_conformers_geometric(conformer_properties)
```

---

## Phase Space (`PhaseSpaceManifold`)

- Dynamical trajectories analyzed via Lyapunov exponents.
- Energy curvature separates stable vs chaotic regimes.
- Approximate invariants emerge from near-zero gradients.

```python
from geometry.domains import PhaseSpaceManifold

ps = PhaseSpaceManifold(trajectories)
chaotic = ps.detect_chaotic_orbits(threshold=0.05)
```

---

## Semantics (`SemanticManifold`)

- Embeddings normalized as manifold coordinates.
- Parallel transport solves analogies geometrically.
- Curvature of neighborhood covariance scores polysemy.

```python
from geometry.domains import SemanticManifold

sm = SemanticManifold(embeddings)
analogy = sm.analogy("king", "queen", "man")[0]
polysemy = sm.polysemy_score("bank")
```

---

## Summary Table

| Domain | Manifold | Key Signal | Helper |
|--------|----------|------------|--------|
| Property | `PropertyManifold` | Correlation curvature | — |
| Time Series | `TimeSeriesManifold` | Curvature spikes | `detect_cycles_geometric` |
| Network Flow | `NetworkFlowManifold` | Divergence / curl | `find_bottlenecks_geometric` |
| Finance | `FinancialManifold` | Geodesic risk | `compute_risk_geometric` |
| Molecular | `MolecularManifold` | Strain curvature | `find_strained_conformers_geometric` |
| Phase Space | `PhaseSpaceManifold` | Lyapunov exponent | — |
| Semantic | `SemanticManifold` | Polysemy curvature | — |

---

Curvature replaces heuristics. Geometry itself surfaces the stories hidden inside each modality.
