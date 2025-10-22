# Multi-Modal Analysis - Quick Start

## 🚀 5-Minute Quick Start

### Installation

The multi-modal analysis module is part of the Anant library:

```bash
# Install Anant with dependencies
pip install polars numpy scikit-learn

# Navigate to the multi_modal_analysis directory
cd /Users/binoyayyagari/anant/anant/multi_modal_analysis
```

### Run Your First Demo

```bash
# Run the e-commerce demo
python demos/demo_ecommerce.py

# Run the simple example
python examples/simple_example.py
```

### Basic Usage (3 Steps)

#### Step 1: Create Modality Hypergraphs

```python
from anant import Hypergraph
import polars as pl

# Create data for different relationship types
purchases = pl.DataFrame([
    {"edges": "p1", "nodes": "customer_1", "weight": 100},
    {"edges": "p1", "nodes": "product_A", "weight": 1},
])

reviews = pl.DataFrame([
    {"edges": "r1", "nodes": "customer_1", "weight": 5},
    {"edges": "r1", "nodes": "product_A", "weight": 5},
])

# Create hypergraphs
purchase_hg = Hypergraph(purchases)
review_hg = Hypergraph(reviews)
```

#### Step 2: Build Multi-Modal Hypergraph

```python
from multi_modal_analysis import MultiModalHypergraph

# Create multi-modal hypergraph
mmhg = MultiModalHypergraph(name="customer_behavior")

# Add modalities
mmhg.add_modality("purchases", purchase_hg, weight=2.0)
mmhg.add_modality("reviews", review_hg, weight=1.0)
```

#### Step 3: Analyze Cross-Modal Patterns

```python
# Find entities in multiple modalities
bridges = mmhg.find_modal_bridges(min_modalities=2)
print(f"Entities in both: {len(bridges)}")

# Compute cross-modal centrality
centrality = mmhg.compute_cross_modal_centrality(
    "customer_1",
    metric="degree",
    aggregation="weighted_average"
)
print(f"Centrality: {centrality['aggregated']}")

# Discover inter-modal relationships
relationships = mmhg.discover_inter_modal_relationships(
    "purchases", "reviews"
)
print(f"Found {len(relationships)} connections")
```

## 📊 What You Get

### Cross-Domain Insights

Discover patterns that span multiple relationship types:
- **Modal Bridges**: Entities active across multiple domains
- **Cross-Modal Patterns**: Behavioral patterns across modalities
- **Inter-Modal Relationships**: Connections between different relationship types

### Business Applications

#### E-Commerce
- Customer engagement segmentation
- Wishlist → Purchase conversion analysis
- Review incentive targeting
- Return pattern detection

#### Healthcare
- Patient journey optimization
- Provider coordination analysis
- Treatment effectiveness tracking
- Care pathway discovery

#### Research Networks
- Collaboration pattern analysis
- Citation vs collaboration gaps
- Funding impact assessment
- Mentor-mentee relationship tracking

#### Social Media
- Influence propagation analysis
- Content engagement patterns
- Community formation dynamics
- Cross-platform behavior

## 🎯 Key Features

### 1. Multi-Modal Construction
```python
mmhg = MultiModalHypergraph()
mmhg.add_modality("type1", hg1, weight=2.0)
mmhg.add_modality("type2", hg2, weight=1.0)
```

### 2. Modal Bridge Detection
```python
# Find highly engaged entities
bridges = mmhg.find_modal_bridges(
    min_modalities=3,
    min_connections=5
)
```

### 3. Cross-Modal Centrality
```python
# Aggregate centrality across modalities
centrality = mmhg.compute_cross_modal_centrality(
    "entity_123",
    metric="degree",
    aggregation="weighted_average"
)
```

### 4. Pattern Detection
```python
# Detect cross-modal patterns
patterns = mmhg.detect_cross_modal_patterns(min_support=10)
```

### 5. Modal Correlation
```python
# Measure modality overlap
corr = mmhg.compute_modal_correlation("mod1", "mod2")
```

### 6. Inter-Modal Relationships
```python
# Find implicit connections
relationships = mmhg.discover_inter_modal_relationships(
    source_modality="purchases",
    target_modality="reviews"
)
```

## 📖 Next Steps

### Learn More
- **[README.md](./README.md)**: Complete overview
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)**: Detailed guide
- **[demos/](./demos/)**: Full demonstrations

### Run Demos
```bash
# E-commerce analysis
python demos/demo_ecommerce.py

# Healthcare analysis (if available)
python demos/demo_healthcare.py

# Simple example
python examples/simple_example.py
```

### Explore Use Cases
- Customer behavior segmentation
- Conversion funnel analysis
- Influence analysis
- Anomaly detection
- Temporal pattern tracking

## 🔧 Troubleshooting

### Import Error
```python
# If you get ImportError for Anant
# The demos include fallback mock implementations
# They will work without full Anant installation
```

### Missing Dependencies
```bash
# Install required packages
pip install polars numpy scikit-learn
```

### Data Format
```python
# Ensure your data has required columns
df = pl.DataFrame({
    "edges": ["e1", "e1"],      # Edge identifiers
    "nodes": ["n1", "n2"],      # Node identifiers  
    "weight": [1.0, 1.0]        # Optional weights
})
```

## 💡 Tips

1. **Start Simple**: Use `examples/simple_example.py` to understand basics
2. **Weight Modalities**: Assign weights based on business importance
3. **Filter Early**: Use `min_modalities` to focus on engaged entities
4. **Batch Processing**: Process large datasets in chunks
5. **Cache Results**: Entity index is cached for performance

## 📊 Expected Output

Running the e-commerce demo produces:

```
🔍 Demo 1: Basic Multi-Modal Hypergraph Construction
   ✅ Purchases hypergraph created
   ✅ Reviews hypergraph created
   ...
   Total Unique Entities: 500
   Avg Modalities per Entity: 2.3

🔍 Demo 2: Finding Modal Bridges
   Entities in 2+ Modalities: 150
   Highly Engaged (3+ Modalities): 45
   ...

🔍 Demo 7: Business Insights
   Power users: 12
   Conversion Rate: 0.34
   → Action: Send promotions for wishlisted items
```

## 🎊 Success!

You now have a working multi-modal analysis setup. Explore the demos and adapt them to your use cases!

---

**Questions?** Check [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) or [README.md](./README.md)

**Version**: 1.0.0  
**Status**: Production Ready
