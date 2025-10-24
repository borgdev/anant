# Anant Graph Registry Architecture

> **PostgreSQL as Graph Registry/Catalog + Anant's Native Parquet Storage**

## 🎯 Architecture Overview

This implementation aligns with **Anant's native Parquet+Polars architecture** by using PostgreSQL as a lightweight **graph registry/catalog** rather than full graph storage.

### 🏗️ Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Anant Graph Registry                     │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL (Registry/Catalog)    │  Parquet Files (Data)   │
│  ├─ Graph metadata & catalog      │  ├─ Hypergraphs         │
│  ├─ User management               │  ├─ Metagraphs          │
│  ├─ Access control                │  ├─ Knowledge Graphs    │
│  ├─ Query history                 │  └─ All graph data      │
│  └─ Audit trails                  │                         │
├─────────────────────────────────────────────────────────────┤
│           Ray Cluster (Distributed Computing)              │
│           Redis (Caching & Sessions)                       │
│           FastAPI (Registry API)                           │
└─────────────────────────────────────────────────────────────┘
```

### ✅ Why This Approach is Correct

1. **Aligns with Anant's Native Storage**: Anant uses Parquet+Polars throughout its codebase
2. **PostgreSQL as Registry**: Optimized for metadata, not full graph storage  
3. **Performance**: Polars provides blazing-fast analytics on Parquet files
4. **Scalability**: Parquet files can be distributed across storage systems
5. **Compatibility**: Works seamlessly with existing Anant architecture

## 📋 PostgreSQL Registry Schema

The registry schema is **lightweight and focused**:

### Core Tables

- **`graph_registry`** - Graph catalog with metadata and Parquet file paths
- **`users`** - User management and authentication
- **`graph_permissions`** - Access control for graphs  
- **`query_history`** - Analytics and usage tracking
- **`storage_locations`** - Storage backend configuration
- **`audit_logs`** - Security and compliance

### Sample Registry Entry

```sql
-- Graph registered in PostgreSQL catalog
INSERT INTO graph_registry (
    name, graph_type, storage_path, 
    node_count, edge_count, properties
) VALUES (
    'my_hypergraph', 'hypergraph', 
    './parquet_data/my_hypergraph.parquet',
    1000, 5000, 
    '{"format": "parquet", "engine": "polars"}'
);
```

## 🚀 Quick Start

### 1. Setup Registry Database

```bash
# Create the registry schema
python create_registry_schema.py
```

### 2. Deploy with Docker

```bash
# Deploy the registry-optimized stack
docker-compose -f docker-compose.registry.yml up -d

# Verify services
docker-compose ps
```

### 3. Run the Demo

```bash
# Demonstrate the registry architecture
python registry_architecture_demo.py
```

### 4. Start Registry API

```bash
# Run the FastAPI registry server
python registry_server.py
```

## 📊 Registry API Endpoints

### Graph Registry Operations

```bash
# List graphs in registry
curl http://localhost:8080/graphs

# Create graph registration
curl -X POST http://localhost:8080/graphs \
  -H "Content-Type: application/json" \
  -d '{"name": "my_graph", "graph_type": "hypergraph"}'

# Get graph metadata
curl http://localhost:8080/graphs/{graph_id}

# Get computed statistics
curl http://localhost:8080/graphs/{graph_id}/stats

# Execute distributed queries
curl -X POST http://localhost:8080/graphs/{graph_id}/query \
  -H "Content-Type: application/json" \
  -d '{"operation": "analyze", "parameters": {}}'
```

### Registry Statistics

```bash
# Overall registry stats
curl http://localhost:8080/registry/stats
```

## 💾 Storage Architecture

### Parquet File Organization

```
parquet_data/
├── hypergraphs/
│   ├── research_collab.parquet      # Hypergraph data
│   └── social_network.parquet
├── metagraphs/ 
│   ├── ontology_mapping.parquet     # Metagraph entities/relations
│   └── concept_hierarchy.parquet
├── knowledge_graphs/
│   ├── biomedical_kg.parquet        # Knowledge graph triples
│   └── enterprise_kg.parquet
└── metadata/
    ├── schemas/                     # Schema definitions
    └── indices/                     # Precomputed indices
```

### High-Performance Operations

```python
import polars as pl
from anant.io.parquet_io import ParquetIO

# Anant's native Parquet operations
parquet_io = ParquetIO()

# Read hypergraph with Polars (blazing fast)
df = pl.scan_parquet("./parquet_data/my_hypergraph.parquet")

# Lazy evaluation for massive datasets  
result = (df
    .filter(pl.col("type") == "node")
    .group_by("category") 
    .len()
    .collect())

# Use Anant's hypergraph operations
hypergraph = parquet_io.load_hypergraph("./parquet_data/my_hypergraph.parquet")
```

## 🔄 Data Flow

### Graph Creation Flow

1. **API Request** → FastAPI registry server
2. **Generate Parquet Path** → `./parquet_data/{graph_name}.parquet` 
3. **Register in PostgreSQL** → Metadata only
4. **Create Parquet File** → Using Anant's native storage
5. **Return Registry ID** → For future operations

### Graph Query Flow

1. **Query Request** → Registry API with graph ID
2. **Lookup Storage Path** → From PostgreSQL registry
3. **Distributed Execution** → Ray cluster + Parquet operations
4. **Return Results** → High-performance analytics
5. **Log Usage** → Query history for analytics

## 🎯 Benefits of This Architecture

### ✅ Performance Benefits

- **Polars Speed**: Vectorized operations on Parquet files
- **Lazy Evaluation**: Process massive datasets efficiently  
- **Columnar Storage**: Optimized for analytical workloads
- **Compression**: Efficient storage with Snappy/LZ4

### ✅ Scalability Benefits

- **Distributed Storage**: Parquet files across multiple systems
- **Ray Computing**: Distributed graph operations
- **Registry Queries**: Fast metadata lookups
- **Caching**: Redis for frequently accessed data

### ✅ Architecture Benefits

- **Native Compatibility**: Aligns with Anant's existing codebase
- **Storage Flexibility**: Local, S3, GCS, Azure storage backends
- **Registry Focus**: PostgreSQL optimized for catalog operations
- **API Consistency**: REST API for all registry operations

## 🛠️ Configuration

### Environment Variables

```bash
# PostgreSQL Registry
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres  
POSTGRES_DB=anant_registry

# Redis Caching
REDIS_URL=redis://localhost:6379/0

# Ray Distributed Computing
RAY_ADDRESS=ray://localhost:10001

# Parquet Storage
PARQUET_BASE_PATH=./parquet_data

# API Configuration  
JWT_SECRET=your_secret_key
DEBUG=false
```

### Storage Configuration

```python
# Configure storage backends in PostgreSQL
INSERT INTO storage_locations (
    name, storage_type, base_path, configuration
) VALUES (
    'local_parquet', 'local', './parquet_data',
    '{"format": "parquet", "compression": "snappy", "engine": "polars"}'
),
(
    's3_production', 's3', 's3://anant-graphs/',
    '{"region": "us-west-2", "compression": "lz4", "partitioning": "date"}'
);
```

## 📈 Monitoring & Analytics

### Registry Analytics

```sql
-- Graph usage analytics
SELECT 
    gr.name,
    gr.graph_type,
    COUNT(qh.id) as query_count,
    AVG(qh.execution_time_ms) as avg_execution_time,
    SUM(gr.file_size_bytes) as total_size
FROM graph_registry gr
LEFT JOIN query_history qh ON gr.id = qh.graph_id
GROUP BY gr.id, gr.name, gr.graph_type
ORDER BY query_count DESC;
```

### Performance Monitoring

- **Ray Dashboard**: http://localhost:8265
- **Registry API**: http://localhost:8080/health  
- **Jupyter Analysis**: http://localhost:8888
- **Optional Grafana**: http://localhost:3000

## 🔐 Security & Access Control

### User Management

```sql
-- Create users
INSERT INTO users (username, email, password_hash, is_admin)
VALUES ('researcher', 'researcher@org.com', 'hashed_pw', false);

-- Grant graph access
INSERT INTO graph_permissions (graph_id, user_id, permission_type)
VALUES ('graph-uuid', 'user-uuid', 'read');
```

### Audit Trails

All operations are logged in `audit_logs` table:
- User actions  
- Graph access
- Query execution
- Registry modifications

## 🧪 Testing & Validation

### Run Tests

```bash
# Test registry functionality
python -m pytest tests/test_registry.py

# Test Parquet operations  
python -m pytest tests/test_parquet_ops.py

# Integration tests
python -m pytest tests/test_integration.py
```

### Validate Architecture

```bash
# Run the architecture demo
python registry_architecture_demo.py

# Verify registry-Parquet alignment
python validate_architecture.py
```

## 🚀 Production Deployment

### Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.registry.yml up -d

# Scale Ray workers
docker-compose up --scale ray-worker=4 -d
```

### Kubernetes (Advanced)

```bash
# Deploy with Helm
helm install anant-registry ./charts/anant-registry

# Scale components
kubectl scale deployment ray-workers --replicas=6
```

## 📚 Integration with Anant

This registry architecture **seamlessly integrates** with Anant's existing codebase:

```python
from anant.hypergraph.core import Hypergraph
from anant.metagraph.core.metagraph import Metagraph  
from anant.io.parquet_io import ParquetIO

# Registry manages metadata, Anant handles graph operations
registry_client = RegistryClient("http://localhost:8080")
parquet_io = ParquetIO()

# Load graph via registry
graph_info = registry_client.get_graph("my-graph-id")
hypergraph = parquet_io.load_hypergraph(graph_info["storage_path"])

# Use Anant's native operations
result = hypergraph.compute_centrality()
```

---

## 🎯 Key Takeaway

This architecture **perfectly aligns** with Anant's native Parquet+Polars approach:

- ✅ **PostgreSQL**: Lightweight registry/catalog (metadata only)
- ✅ **Parquet**: Native graph data storage (Anant's approach)  
- ✅ **Polars**: High-performance analytics (Anant's engine)
- ✅ **Registry**: Discovery, access control, and management
- ✅ **Compatibility**: Seamless integration with existing Anant codebase

You were absolutely correct - Anant uses Parquet natively, so PostgreSQL should serve as a **graph registry**, not full graph storage! 🎯