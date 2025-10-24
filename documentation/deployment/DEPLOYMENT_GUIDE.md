# Anant Enterprise Ray Cluster - Deployment Guide

## Overview

This guide covers deployment of the Anant Enterprise Ray Cluster using Docker, Docker Compose, and Kubernetes/Helm. The system provides distributed geometric and contextual processing capabilities with enterprise-grade features.

## Prerequisites

### System Requirements
- **Docker**: Version 20.10+ with BuildKit support
- **Docker Compose**: Version 2.0+
- **Kubernetes**: Version 1.24+ (for Helm deployment)
- **Helm**: Version 3.8+
- **Resources**: Minimum 8GB RAM, 4 CPU cores

### Development Tools
- **Python**: 3.11+
- **Ray**: 2.31.0+
- **PostgreSQL**: 15+
- **Redis**: 7+

## Docker Deployment

### Quick Start with Docker Compose

#### Development Environment
```bash
# Clone and navigate to project
cd /path/to/anant

# Start development environment
./scripts/deploy-docker.sh deploy-dev

# Check status
./scripts/deploy-docker.sh status

# View logs
./scripts/deploy-docker.sh logs ray-head

# Access services
# - API: http://localhost:8000
# - Ray Dashboard: http://localhost:8265
# - Jupyter: http://localhost:8888
```

#### Production Environment
```bash
# Start production environment
./scripts/deploy-docker.sh deploy-prod

# Check health
./scripts/deploy-docker.sh health

# Scale workers
./scripts/deploy-docker.sh scale geometric 5
./scripts/deploy-docker.sh scale contextual 3

# Monitor with Grafana: http://localhost:3000
# Default credentials: admin/admin
```

### Manual Docker Commands

#### Building Images
```bash
# Production image
docker build -t anant-enterprise:latest .

# Development image
docker build --target development -t anant-enterprise:dev .

# Testing image
docker build --target testing -t anant-enterprise:test .
```

#### Running Containers
```bash
# Ray head node
docker run -d --name ray-head \
  -p 8000:8000 -p 8265:8265 \
  -e RAY_HEAD=true \
  anant-enterprise:latest

# Ray worker
docker run -d --name ray-worker-1 \
  --link ray-head \
  -e RAY_HEAD_ADDRESS=ray-head:10001 \
  anant-enterprise:latest
```

## Kubernetes Deployment

### Quick Start with Helm

#### Development Deployment
```bash
# Deploy to development
./scripts/deploy-k8s.sh deploy-dev

# Check status
./scripts/deploy-k8s.sh status anant-enterprise-dev

# Port forward services
./scripts/deploy-k8s.sh port-forward api 8000
./scripts/deploy-k8s.sh port-forward dashboard 8265
```

#### Production Deployment
```bash
# Deploy to production
./scripts/deploy-k8s.sh deploy-prod

# Perform health check
./scripts/deploy-k8s.sh health

# Scale workers
./scripts/deploy-k8s.sh scale geometric 10
./scripts/deploy-k8s.sh scale contextual 8
```

### Manual Helm Commands

#### Install with Custom Values
```bash
# Create custom values file
cat > custom-values.yaml << EOF
rayWorkers:
  geometric:
    replicaCount: 5
  contextual:
    replicaCount: 3

resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
EOF

# Deploy with custom values
helm install anant-enterprise ./helm/anant-enterprise \
  --namespace anant-enterprise \
  --create-namespace \
  --values custom-values.yaml
```

#### Upgrade Deployment
```bash
# Upgrade with new values
helm upgrade anant-enterprise ./helm/anant-enterprise \
  --namespace anant-enterprise \
  --values production-values.yaml

# Rollback if needed
helm rollback anant-enterprise 1 --namespace anant-enterprise
```

## Configuration

### Environment Variables

#### Core Configuration
```bash
# Ray Configuration
RAY_HEAD=true                    # Set to true for head node
RAY_HEAD_ADDRESS=ray-head:10001  # Address for workers to connect
RAY_WORKER_TYPE=geometric        # Worker specialization

# Database Configuration
DATABASE_URL=postgresql://user:pass@db:5432/anant
REDIS_URL=redis://redis:6379/0

# Application Configuration
ANANT_MODE=production            # production, development, testing
ANANT_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
ANANT_ENABLE_JUPYTER=false      # Enable Jupyter for development
```

#### Security Configuration
```bash
# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24

# CORS Settings
CORS_ORIGINS=["http://localhost:3000", "https://yourapp.com"]
CORS_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
```

### Helm Values Configuration

#### Production Values Example
```yaml
# values-prod.yaml
anant:
  mode: "production"
  security:
    enabled: true
    jwtSecret: "your-jwt-secret"
  enterprise:
    monitoring: true
    auditing: true

rayHead:
  replicaCount: 1
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi

rayWorkers:
  geometric:
    replicaCount: 5
    resources:
      limits:
        cpu: 2000m
        memory: 4Gi
  contextual:
    replicaCount: 3
    resources:
      limits:
        cpu: 2000m
        memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: anant.yourcompany.com
      paths:
        - path: /
          pathType: Prefix

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "secure-password"
```

## Service Access

### Local Development (Docker Compose)

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Main Anant API |
| Ray Dashboard | http://localhost:8265 | Ray cluster monitoring |
| Jupyter | http://localhost:8888 | Development notebooks |
| Grafana | http://localhost:3000 | Metrics dashboard |
| Prometheus | http://localhost:9090 | Metrics collection |

### Kubernetes (with Ingress)

| Service | URL | Description |
|---------|-----|-------------|
| API | https://anant.yourcompany.com | Main Anant API |
| Dashboard | https://anant.yourcompany.com/dashboard | Ray cluster monitoring |
| Grafana | https://anant.yourcompany.com/grafana | Metrics dashboard |

### Port Forwarding (Kubernetes)
```bash
# API access
kubectl port-forward svc/anant-enterprise-ray-head 8000:8000

# Ray dashboard
kubectl port-forward svc/anant-enterprise-ray-head 8265:8265

# Grafana
kubectl port-forward svc/anant-enterprise-grafana 3000:80
```

## Monitoring and Observability

### Health Checks

#### API Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "ray_cluster": {
    "nodes": 4,
    "cpus": 16,
    "memory_gb": 32
  },
  "database": "connected",
  "redis": "connected"
}
```

#### Ray Cluster Status
```bash
# Via API
curl http://localhost:8000/ray/status

# Via Ray CLI (in container)
ray status --address=ray://ray-head:10001
```

### Metrics and Logging

#### Prometheus Metrics Endpoints
- **Ray Head**: http://ray-head:8000/metrics
- **Ray Workers**: http://ray-worker:8000/metrics
- **Application**: http://api:8000/metrics

#### Grafana Dashboards
- **Ray Cluster Overview**: System resources, job queues
- **Application Metrics**: Request rates, response times
- **Business Metrics**: Processing throughput, success rates

#### Log Aggregation
```bash
# Docker Compose
docker-compose logs -f ray-head
docker-compose logs -f ray-worker-geometric

# Kubernetes
kubectl logs -f deployment/anant-enterprise-ray-head
kubectl logs -f deployment/anant-enterprise-ray-worker-geometric
```

## Scaling and Performance

### Horizontal Scaling

#### Docker Compose Scaling
```bash
# Scale workers
./scripts/deploy-docker.sh scale geometric 5
./scripts/deploy-docker.sh scale contextual 3

# Manual scaling
docker-compose up -d --scale ray-worker-geometric=5
```

#### Kubernetes Scaling
```bash
# Manual scaling
kubectl scale deployment anant-enterprise-ray-worker-geometric --replicas=10

# Autoscaling
kubectl autoscale deployment anant-enterprise-ray-worker-geometric \
  --min=2 --max=20 --cpu-percent=70
```

### Performance Tuning

#### Ray Configuration
```yaml
# Increase object store size
ray_head_memory_gb: 8
ray_worker_memory_gb: 4

# Optimize for CPU-intensive tasks
ray_cpu_allocation: 0.8
ray_memory_allocation: 0.7
```

#### Resource Limits
```yaml
resources:
  limits:
    cpu: "4"
    memory: "8Gi"
    ephemeral-storage: "10Gi"
  requests:
    cpu: "2"
    memory: "4Gi"
    ephemeral-storage: "5Gi"
```

## Troubleshooting

### Common Issues

#### Ray Cluster Connection Issues
```bash
# Check Ray head status
ray status --address=ray://ray-head:10001

# Verify network connectivity
docker exec ray-worker-1 ping ray-head
kubectl exec deployment/ray-worker -- ping ray-head
```

#### Database Connection Issues
```bash
# Test database connection
docker exec ray-head psql $DATABASE_URL -c "SELECT 1;"

# Check database logs
docker-compose logs postgres
kubectl logs deployment/postgresql
```

#### Memory Issues
```bash
# Monitor Ray memory usage
ray memory --address=ray://ray-head:10001

# Check container memory
docker stats
kubectl top pods
```

### Debug Commands

#### Container Debugging
```bash
# Enter container shell
docker exec -it ray-head bash
kubectl exec -it deployment/ray-head -- bash

# Check Ray processes
ps aux | grep ray

# Verify Python environment
python -c "import ray; print(ray.cluster_resources())"
```

#### Network Debugging
```bash
# Test service connectivity
curl -f http://ray-head:8000/health
kubectl exec deployment/ray-worker -- curl -f http://ray-head:8000/health

# Check port availability
netstat -tulpn | grep :8000
kubectl get services
```

## Security Considerations

### Production Security

#### Container Security
- Use non-root user (uid 1000)
- Read-only root filesystem where possible
- Security context restrictions
- Resource limits and quotas

#### Network Security
- TLS termination at ingress
- Internal service mesh encryption
- Network policies for pod isolation
- Rate limiting and DDoS protection

#### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Audit logging

### Secrets Management

#### Docker Compose
```yaml
secrets:
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  db_password:
    file: ./secrets/db_password.txt
```

#### Kubernetes Secrets
```bash
# Create secrets
kubectl create secret generic anant-secrets \
  --from-literal=jwt-secret="your-secret" \
  --from-literal=db-password="db-password"

# Use in deployment
env:
  - name: JWT_SECRET_KEY
    valueFrom:
      secretKeyRef:
        name: anant-secrets
        key: jwt-secret
```

## Backup and Recovery

### Database Backups

#### Automated Backups (PostgreSQL)
```bash
# Create backup job
kubectl create job --from=cronjob/postgres-backup postgres-backup-manual

# Restore from backup
kubectl exec deployment/postgresql -- psql -U postgres -d anant < backup.sql
```

#### Manual Backup
```bash
# Docker Compose
docker exec postgres pg_dump -U postgres anant > backup_$(date +%Y%m%d).sql

# Kubernetes
kubectl exec deployment/postgresql -- pg_dump -U postgres anant > backup.sql
```

### Data Persistence

#### Volume Management
```bash
# List persistent volumes
kubectl get pv,pvc

# Backup volume data
kubectl exec deployment/anant-enterprise -- tar -czf - /data | \
  kubectl exec -i backup-pod -- tar -xzf - -C /backup
```

## Maintenance

### Updates and Upgrades

#### Rolling Updates
```bash
# Docker Compose
docker-compose pull
docker-compose up -d --remove-orphans

# Kubernetes
helm upgrade anant-enterprise ./helm/anant-enterprise \
  --namespace anant-enterprise \
  --values production-values.yaml
```

#### Zero-Downtime Deployments
```yaml
# Deployment strategy
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1
    maxSurge: 1

# Health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Resource Cleanup

#### Docker Cleanup
```bash
# Remove stopped containers and unused images
docker system prune -a

# Clean up volumes
docker volume prune
```

#### Kubernetes Cleanup
```bash
# Delete deployment
./scripts/deploy-k8s.sh cleanup anant-enterprise

# Clean up resources
kubectl delete namespace anant-enterprise
```

## Support and Community

### Getting Help
- **Issues**: File issues on GitHub repository
- **Documentation**: Check docs/ directory
- **Community**: Join discussions in community forums

### Contributing
- Follow contribution guidelines in CONTRIBUTING.md
- Submit pull requests for improvements
- Report bugs and feature requests

## Appendix

### Complete File Structure
```
/home/amansingh/dev/ai/anant/
├── Dockerfile                          # Multi-stage container build
├── docker-compose.yml                  # Production environment
├── docker-compose.dev.yml             # Development environment
├── requirements.txt                    # Python dependencies
├── scripts/
│   ├── deploy-docker.sh               # Docker deployment script
│   └── deploy-k8s.sh                  # Kubernetes deployment script
├── helm/anant-enterprise/             # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-dev.yaml               # Development overrides
│   ├── values-prod.yaml              # Production overrides
│   └── templates/                     # Kubernetes templates
├── config/
│   ├── nginx.conf                     # Reverse proxy config
│   ├── prometheus.yml                 # Monitoring config
│   └── postgres-init.sql             # Database initialization
└── anant/                            # Main application code
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RAY_HEAD` | `false` | Enable Ray head node |
| `RAY_HEAD_ADDRESS` | `ray-head:10001` | Ray cluster address |
| `RAY_WORKER_TYPE` | `multipurpose` | Worker specialization |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |
| `ANANT_MODE` | `development` | Application mode |
| `ANANT_LOG_LEVEL` | `INFO` | Logging level |
| `JWT_SECRET_KEY` | - | JWT signing secret |
| `CORS_ORIGINS` | `[]` | Allowed CORS origins |

### Resource Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **Network**: 1Gbps

#### Recommended Production
- **CPU**: 16+ cores
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps
- **Nodes**: 3+ for HA