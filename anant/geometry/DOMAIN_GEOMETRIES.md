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
from anant.geometry.core import PropertyManifold

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
from anant.geometry.domains import TimeSeriesManifold

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
from anant.geometry.domains import find_bottlenecks_geometric

hotspots = find_bottlenecks_geometric(graph, flow_attribute="traffic", top_k=5)
```

---

## Finance (`FinancialManifold`)

- Asset statistics define coordinates (mean, volatility, skew, kurtosis).
- Geodesic distance quantifies diversification power.
- Curvature spikes warn about systemic stress.
- Wrapper: `compute_risk_geometric()`.

```python
from anant.geometry.domains import compute_risk_geometric

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
from anant.geometry.domains import find_strained_conformers_geometric

strained = find_strained_conformers_geometric(conformer_properties)
```

---

## Phase Space (`PhaseSpaceManifold`)

- Dynamical trajectories analyzed via Lyapunov exponents.
- Energy curvature separates stable vs chaotic regimes.
- Approximate invariants emerge from near-zero gradients.

```python
from anant.geometry.domains import PhaseSpaceManifold

ps = PhaseSpaceManifold(trajectories)
chaotic = ps.detect_chaotic_orbits(threshold=0.05)
```

---

## Semantics (`SemanticManifold`)

- Embeddings normalized as manifold coordinates.
- Parallel transport solves analogies geometrically.
- Curvature of neighborhood covariance scores polysemy.

```python
from anant.geometry.domains import SemanticManifold

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


---

## Spread Dynamics (`SpreadDynamicsManifold`)

Universal contagion/propagation framework.

- **Applies to**: epidemic spread, viral content, network failures, financial contagion, trait propagation
- Curvature acceleration zones locate outbreak sources
- Geodesics trace propagation pathways
- R0 estimation from network geometry
- Helper: `detect_spread_hotspots()`

```python
from anant.geometry.domains import SpreadDynamicsManifold

epidemic = SpreadDynamicsManifold(contact_network, 'infection_rate')
hotspots = epidemic.detect_acceleration_zones()
sources = epidemic.find_propagation_sources()
r0 = epidemic.estimate_r0()
```

---

## Allocation (`AllocationManifold`)

Multi-resource optimization framework.

- **Applies to**: healthcare resources (beds, nurses), cloud (CPU, memory), logistics, energy grids
- Scarcity-weighted metric tensor
- Curvature reveals allocation stress/imbalance
- Geodesics provide optimal reallocation paths
- Helper: `find_allocation_stress()`

```python
from anant.geometry.domains import AllocationManifold

resources = AllocationManifold(allocations, capacities)
stress = resources.detect_stress_points()
transfers = resources.suggest_optimal_transfers(stressed_entity='ICU-A')
```

---

## Process (`ProcessManifold`)

Workflow/pipeline analysis framework.

- **Applies to**: manufacturing pipelines, healthcare pathways, software CI/CD, business processes
- Curvature identifies process friction/bottlenecks
- Geodesics reveal streamlined workflow paths
- Failure-prone step detection
- Helper: `find_workflow_bottlenecks()`

```python
from anant.geometry.domains import ProcessManifold

workflow = ProcessManifold(production_pipeline)
bottlenecks = workflow.find_bottlenecks()
optimal = workflow.discover_streamlined_path(start, end)
improvements = workflow.suggest_process_improvements()
```

---

## Matching (`MatchingManifold`)

Similarity-based pairing framework.

- **Applies to**: patient-trial matching, HR recruitment, education, dating, rideshare
- Geodesic distance measures match quality
- Natural clustering discovers groups
- Bidirectional or directional matching
- Helper: `find_best_matches()`

```python
from anant.geometry.domains import MatchingManifold

matching = MatchingManifold(set_a=patients, set_b=trials)
matches = matching.find_optimal_matches(entity_id='patient-123', top_k=5)
groups = matching.discover_natural_groups()
```

---

## Hierarchical (`HierarchicalManifold`)

Multi-level system analysis via fiber bundle geometry.

- **Applies to**: healthcare systems (patient → department → hospital), organizations, geography, infrastructure
- Vertical structure = hierarchy levels
- Cross-level impact propagation
- Metric aggregation across levels
- Escalation path computation

```python
from anant.geometry.domains import HierarchicalManifold

hierarchy = HierarchicalManifold(org_structure, level_names=['employee', 'team', 'division'])
impact = hierarchy.compute_cross_level_impact('dept-A', change={'budget': -100K})
aggregated = hierarchy.aggregate_metrics(level=2, aggregation='sum')
path = hierarchy.compute_escalation_path('employee-42', target_level=3)
```

---

## Updated Summary Table

| Domain | Manifold | Key Signal | Helper | Scope |
|--------|----------|------------|--------|-------|
| Property | `PropertyManifold` | Correlation curvature | — | General |
| Time Series | `TimeSeriesManifold` | Curvature spikes | `detect_cycles_geometric` | Domain |
| Network Flow | `NetworkFlowManifold` | Divergence / curl | `find_bottlenecks_geometric` | Domain |
| Finance | `FinancialManifold` | Geodesic risk | `compute_risk_geometric` | Domain |
| Molecular | `MolecularManifold` | Strain curvature | `find_strained_conformers_geometric` | Domain |
| Phase Space | `PhaseSpaceManifold` | Lyapunov exponent | — | Domain |
| Semantic | `SemanticManifold` | Polysemy curvature | — | Domain |
| **Spread** | **`SpreadDynamicsManifold`** | **Acceleration zones** | **`detect_spread_hotspots`** | **General** |
| **Allocation** | **`AllocationManifold`** | **Stress curvature** | **`find_allocation_stress`** | **General** |
| **Process** | **`ProcessManifold`** | **Friction curvature** | **`find_workflow_bottlenecks`** | **General** |
| **Matching** | **`MatchingManifold`** | **Match distance** | **`find_best_matches`** | **General** |
| **Hierarchy** | **`HierarchicalManifold`** | **Cross-level impact** | **—** | **General** |

---

**General-Purpose manifolds** apply to multiple domains. **Domain-Specific manifolds** target specialized use cases. Together they cover ~95% of real-world analytics needs through unified geometric framework.
