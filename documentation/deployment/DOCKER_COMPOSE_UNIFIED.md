# Anant Enterprise Platform - Unified Docker Compose

> **Single Docker Compose file with multiple deployment profiles**

## 🎯 Overview

This unified Docker Compose configuration replaces multiple separate files and uses **profiles** to support different deployment scenarios with a single, maintainable configuration.

## 📋 Available Profiles

### 🚀 **Production Profile**
Full enterprise Ray cluster with all production services
```bash
docker-compose --profile production up -d
```

**Services:**
- Ray Head Node + 3 Worker Nodes (geometric, contextual, multi-purpose)
- PostgreSQL (enterprise features)  
- Redis (cluster coordination)
- Nginx (reverse proxy)
- Full monitoring stack

### 🛠️ **Development Profile**  
Lightweight setup for local development
```bash
docker-compose --profile development up -d
```

**Services:**
- Ray Head Node (dev mode)
- Single Ray Worker
- PostgreSQL (dev database)
- Redis (caching)
- Source code mounting for live reload

### 📊 **Registry Profile**
Graph registry with PostgreSQL + Parquet storage
```bash
docker-compose --profile registry up -d
```

**Services:**
- PostgreSQL (graph registry/catalog)
- Registry API Server (FastAPI)
- Redis (caching)
- Parquet storage volumes

### 📈 **Monitoring Profile**
Prometheus + Grafana monitoring stack
```bash
docker-compose --profile monitoring up -d
```

**Services:**
- Prometheus (metrics collection)
- Grafana (visualization dashboard)

### 🔬 **Jupyter Profile**
Interactive development environment
```bash
docker-compose --profile jupyter up -d
```

**Services:**
- Jupyter Lab with Anant libraries
- Access to notebooks and tutorials
- Parquet data access

## 🚀 Usage Examples

### Quick Start - Development
```bash
# Start development environment
docker-compose --profile development up -d

# View logs
docker-compose logs -f anant-ray-head-dev

# Scale workers (in separate terminal)
docker-compose --profile development up --scale anant-ray-worker-dev=2 -d
```

### Production Deployment
```bash
# Full production stack
docker-compose --profile production --profile monitoring up -d

# Check service health
docker-compose ps
docker-compose exec anant-ray-head curl localhost:8000/health
```

### Registry + Development
```bash
# Registry with development tools
docker-compose --profile registry --profile jupyter up -d

# Access Jupyter: http://localhost:8888 (token: anant_dev_token)
# Access Registry API: http://localhost:8080
```

### Monitoring Only
```bash
# Add monitoring to existing deployment
docker-compose --profile monitoring up -d

# Access Grafana: http://localhost:3000 (admin/anant_admin_2024)
# Access Prometheus: http://localhost:9090
```

## 🔧 Service Management

### Start Services
```bash
# Start specific profile
docker-compose --profile <profile-name> up -d

# Start multiple profiles
docker-compose --profile production --profile monitoring up -d

# Start with build
docker-compose --profile development up --build -d
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific profile services
docker-compose stop anant-ray-head-dev
```

### Scale Services
```bash
# Scale Ray workers (production)
docker-compose --profile production up --scale anant-ray-worker-geo-1=2 -d

# Scale development workers  
docker-compose --profile development up --scale anant-ray-worker-dev=3 -d
```

## 🌐 Service Access Points

| Service | Profile | URL | Description |
|---------|---------|-----|-------------|
| **Ray Dashboard** | production | http://localhost:8265 | Production Ray cluster dashboard |
| **Ray Dashboard** | development | http://localhost:8266 | Development Ray cluster dashboard |
| **Anant API** | production | http://localhost:8000 | Enterprise API server |
| **Anant API** | development | http://localhost:8001 | Development API server |
| **Registry API** | registry | http://localhost:8080 | Graph registry REST API |
| **Jupyter Lab** | jupyter | http://localhost:8888 | Interactive development (token: anant_dev_token) |
| **Grafana** | monitoring | http://localhost:3000 | Monitoring dashboard (admin/anant_admin_2024) |
| **Prometheus** | monitoring | http://localhost:9090 | Metrics collection |
| **Nginx** | production | http://localhost:80 | Load balancer & reverse proxy |

## 🗃️ Database Connections

| Database | Profile | Connection | Purpose |
|----------|---------|------------|---------|
| **Registry DB** | registry | postgresql://postgres:postgres@localhost:5432/anant_registry | Graph registry/catalog |
| **Enterprise DB** | production | postgresql://anant:anant_secure_2024@localhost:5433/anant_enterprise | Production features |
| **Development DB** | development | postgresql://anant_dev:dev_password@localhost:5434/anant_dev | Development/testing |
| **Redis Cache** | all | redis://localhost:6379 | Caching & sessions |

## 📁 Volume Management

### Data Persistence
```bash
# List all volumes
docker volume ls | grep anant

# Backup production data
docker run --rm -v anant_anant-data:/data -v $(pwd):/backup alpine tar czf /backup/anant-data-backup.tar.gz -C /data .

# Restore data
docker run --rm -v anant_anant-data:/data -v $(pwd):/backup alpine tar xzf /backup/anant-data-backup.tar.gz -C /data
```

### Volume Types
- **anant-data**: Production graph data
- **anant-parquet-storage**: Registry Parquet files  
- **anant-dev-data**: Development data
- **anant-logs**: Application logs
- **anant-*-postgres-data**: Database storage
- **anant-redis-data**: Redis persistence

## 🔍 Debugging & Troubleshooting

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f anant-ray-head

# Recent logs only
docker-compose logs --tail=100 -f anant-registry-api
```

### Service Health
```bash
# Check service status
docker-compose ps

# Health check specific service
docker-compose exec anant-ray-head curl localhost:8000/health

# Ray cluster status  
docker-compose exec anant-ray-head ray status
```

### Shell Access
```bash
# Access Ray head node
docker-compose exec anant-ray-head bash

# Access registry database
docker-compose exec anant-registry-db psql -U postgres -d anant_registry

# Access development environment
docker-compose exec anant-ray-head-dev bash
```

## ⚙️ Configuration

### Environment Variables
Create `.env` file for custom configuration:
```bash
# Ray Configuration
RAY_DASHBOARD_HOST=0.0.0.0
RAY_REDIS_PASSWORD=anant_cluster_2024

# Database Configuration  
POSTGRES_PASSWORD=your_secure_password
REGISTRY_POSTGRES_PASSWORD=registry_password

# API Configuration
ANANT_DEBUG=false
ANANT_LOG_LEVEL=INFO

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_admin_password
```

### Custom Configurations
Mount custom configuration files:
```yaml
volumes:
  - ./my-config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
  - ./my-config/grafana:/etc/grafana/provisioning:ro
  - ./my-config/nginx.conf:/etc/nginx/nginx.conf:ro
```

## 🔄 Migration from Old Files

If you have existing separate Docker Compose files:

```bash
# Backup old configurations
mv docker-compose.dev.yml docker-compose.dev.yml.bak
mv docker-compose.registry.yml docker-compose.registry.yml.bak

# Use new unified file with profiles
docker-compose --profile development up -d  # Instead of docker-compose -f docker-compose.dev.yml up -d
docker-compose --profile registry up -d     # Instead of docker-compose -f docker-compose.registry.yml up -d
```

## 📋 Profile Combinations

You can combine multiple profiles for complex deployments:

```bash
# Full stack with monitoring
docker-compose --profile production --profile monitoring --profile jupyter up -d

# Development with registry  
docker-compose --profile development --profile registry up -d

# Registry with monitoring
docker-compose --profile registry --profile monitoring up -d
```

## 🎯 Benefits of Unified Approach

✅ **Single Source of Truth**: One file to maintain instead of multiple  
✅ **Profile-Based**: Clean separation of concerns with Docker Compose profiles  
✅ **Flexible Deployment**: Mix and match services as needed  
✅ **Consistent Networking**: Unified network for all services  
✅ **Shared Volumes**: Efficient resource usage  
✅ **Easy Maintenance**: Updates apply to all deployment scenarios  
✅ **Clear Documentation**: Single reference for all deployment options  

---

## 📞 Support

For issues with specific profiles:
- **Production**: Check Ray dashboard and service health endpoints
- **Development**: Check mounted source code and hot-reload functionality  
- **Registry**: Verify PostgreSQL connection and Parquet storage
- **Monitoring**: Check Prometheus targets and Grafana datasources
- **Jupyter**: Verify token and notebook access

Use `docker-compose logs <service-name>` for detailed debugging information.