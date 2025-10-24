# Docker Compose Port Reference

This document provides a comprehensive reference for all port mappings in the Anant Enterprise Platform Docker Compose configuration.

## Port Allocation Strategy

All external ports follow industry best practices to avoid conflicts with common system services:

- **PostgreSQL**: Use 545x range instead of default 5432
- **API Services**: Use 909x range instead of 8000-8080 
- **Ray Dashboard**: Use 828x instead of 8265
- **Redis**: Use 6380 instead of 6379
- **Web Services**: Use high ports to avoid root requirements
- **Monitoring**: Use 909x range for Prometheus

## Production Profile Ports

| Service | External Port | Internal Port | Description |
|---------|---------------|---------------|-------------|
| Anant Enterprise API | 9094 | 8000 | Main API endpoint |
| Ray Dashboard | 8285 | 8265 | Ray cluster monitoring |
| Ray Cluster Communication | 10001 | 10001 | Internal Ray communication |
| PostgreSQL (Unified) | 5454 | 5432 | Enterprise & registry databases |
| Registry API | 9096 | 8080 | Registry operations API |
| Redis Cache | 6380 | 6379 | Caching and sessions |
| Nginx HTTP | 8080 | 80 | Web server |
| Nginx HTTPS | 8443 | 443 | Secure web server |

## Development Profile Ports

| Service | External Port | Internal Port | Description |
|---------|---------------|---------------|-------------|
| Anant Enterprise API (Dev) | 9095 | 8000 | Development API |
| Ray Dashboard (Dev) | 8286 | 8265 | Development Ray dashboard |
| Ray Cluster Communication (Dev) | 10002 | 10001 | Development Ray communication |
| PostgreSQL (Dev) | 5456 | 5432 | Development database |
| Jupyter Lab | 8890 | 8888 | Interactive development |

## Monitoring Profile Ports

| Service | External Port | Internal Port | Description |
|---------|---------------|---------------|-------------|
| Prometheus | 9091 | 9090 | Metrics collection |
| Grafana | 3001 | 3000 | Metrics visualization |

## Registry Profile Ports

| Service | External Port | Internal Port | Description |
|---------|---------------|---------------|-------------|
| PostgreSQL (Unified) | 5454 | 5432 | Enterprise & registry databases |
| Registry API | 9096 | 8080 | Registry operations |
| Redis Cache | 6380 | 6379 | Registry caching |

## Access URLs

### Production Environment
- **Anant API**: http://localhost:9094
- **Ray Dashboard**: http://localhost:8285
- **Registry API**: http://localhost:9096
- **Web Interface**: http://localhost:8080
- **PostgreSQL (Unified)**: localhost:5454
- **Redis**: localhost:6380

### Development Environment
- **Anant API (Dev)**: http://localhost:9095
- **Ray Dashboard (Dev)**: http://localhost:8286
- **Jupyter Lab**: http://localhost:8890
- **PostgreSQL (Dev)**: localhost:5456

### Monitoring Stack
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3001

## Connection Strings

### PostgreSQL Connections
```bash
# Unified PostgreSQL (both enterprise and registry databases)
psql -h localhost -p 5454 -U postgres -d anant_enterprise
psql -h localhost -p 5454 -U postgres -d anant_registry

# Development Database
psql -h localhost -p 5456 -U anant_dev -d anant_dev
```

### Redis Connection
```bash
# Redis CLI
redis-cli -h localhost -p 6380 -a anant_cluster_2024
```

### Ray Cluster Connection
```python
import ray
# Production
ray.init(address="ray://localhost:10001")

# Development
ray.init(address="ray://localhost:10002")
```

## Environment Variables for External Connections

```env
# Production
ANANT_API_URL=http://localhost:9094
RAY_DASHBOARD_URL=http://localhost:8285
REGISTRY_API_URL=http://localhost:9096
POSTGRES_UNIFIED_URL=postgresql://postgres:anant_secure_2024@localhost:5454/anant_enterprise
POSTGRES_REGISTRY_URL=postgresql://postgres:anant_secure_2024@localhost:5454/anant_registry
REDIS_URL=redis://:anant_cluster_2024@localhost:6380/0

# Development
ANANT_API_URL_DEV=http://localhost:9095
RAY_DASHBOARD_URL_DEV=http://localhost:8286
JUPYTER_URL=http://localhost:8890
POSTGRES_DEV_URL=postgresql://anant_dev:dev_password@localhost:5456/anant_dev

# Monitoring
PROMETHEUS_URL=http://localhost:9091
GRAFANA_URL=http://localhost:3001
```

## Port Conflict Prevention

These ports are specifically chosen to avoid conflicts with:
- System PostgreSQL (5432)
- System Redis (6379) 
- Common development servers (3000, 8000, 8080)
- Common Ray deployments (8265)
- Standard Jupyter (8888)
- System HTTP/HTTPS (80, 443)

## Usage Examples

```bash
# Start production environment
docker-compose --profile production up -d

# Access services
curl http://localhost:9094/health
curl http://localhost:9096/registry/graphs

# Start development environment  
docker-compose --profile development up -d

# Access development services
curl http://localhost:9095/health
open http://localhost:8890  # Jupyter Lab

# Start monitoring stack
docker-compose --profile monitoring up -d

# Access monitoring
open http://localhost:9091  # Prometheus
open http://localhost:3001  # Grafana
```

## Security Considerations

1. **Non-root Ports**: Using ports >1024 avoids requiring root privileges
2. **Firewall Configuration**: Configure firewall rules for production deployment
3. **Network Isolation**: All services communicate via Docker network internally
4. **Authentication**: Proper authentication configured for all databases and services

## Troubleshooting

### Check Port Availability
```bash
# Check if ports are available
ss -tlnp | grep -E ":(9094|8285|5454|6380)"

# Check specific service
docker-compose ps
docker-compose logs anant-ray-head
```

### Port Conflicts
If you encounter port conflicts, you can modify the external ports in `docker-compose.yml` while keeping the internal container ports unchanged.