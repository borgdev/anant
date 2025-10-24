# FastAPI + Ray Distributed Application

A comprehensive FastAPI application that leverages Ray for distributed computing, machine learning, and data processing.

## 🚀 Features

- **Distributed Analytics**: Real-time analytics and metrics computation using Ray workers
- **Machine Learning Service**: Distributed ML training and inference with multiple algorithms
- **Data Processing**: ETL pipelines and data transformation with Ray actors
- **System Monitoring**: Cluster health monitoring and performance metrics
- **Scalable Architecture**: Horizontal scaling with Ray's distributed computing capabilities

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- At least 4GB RAM for the cluster

## 🛠️ Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Deploy the entire cluster**:
   ```bash
   # From the root ray directory
   ./scripts/deploy_fastapi.sh deploy
   ```

2. **Check deployment status**:
   ```bash
   ./scripts/deploy_fastapi.sh status
   ```

3. **Access the services**:
   - FastAPI Application: http://localhost:8080
   - API Documentation: http://localhost:8080/docs
   - Ray Dashboard: http://localhost:8265

### Option 2: Local Development

1. **Start Ray cluster in Docker**:
   ```bash
   docker-compose up -d ray-head ray-worker-1 ray-worker-2
   ```

2. **Install dependencies**:
   ```bash
   cd fastapi
   pip install -r requirements.txt
   ```

3. **Start FastAPI application**:
   ```bash
   export RAY_ADDRESS="ray://localhost:10001"
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

## 🏗️ Architecture

```
FastAPI Application
├── apps/
│   ├── analytics.py      # Distributed analytics service
│   ├── ml_service.py     # Machine learning training & inference
│   ├── data_processing.py # ETL and data pipelines
│   └── monitoring.py     # System monitoring and health checks
├── workers/
│   ├── compute_workers.py # General compute workers
│   ├── ml_workers.py     # ML-specific workers
│   └── data_workers.py   # Data processing workers
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
└── requirements.txt     # Python dependencies
```

## 🔧 API Endpoints

### Analytics Service (`/analytics`)
- `GET /analytics/` - Analytics service overview
- `POST /analytics/metrics` - Generate distributed metrics
- `GET /analytics/real-time` - Real-time analytics dashboard

### Machine Learning Service (`/ml`)
- `GET /ml/` - ML service overview
- `POST /ml/train` - Train models with distributed workers
- `POST /ml/predict` - Distributed inference
- `GET /ml/models` - List available models

### Data Processing Service (`/data`)
- `GET /data/` - Data processing overview
- `POST /data/process-csv` - Process CSV data
- `POST /data/process-json` - Process JSON data
- `POST /data/pipeline` - Execute data pipeline

### Monitoring Service (`/monitoring`)
- `GET /monitoring/` - Monitoring overview
- `POST /monitoring/metrics` - Collect system metrics
- `GET /monitoring/health` - Health check
- `GET /monitoring/dashboard` - Dashboard data

## 🎯 Usage Examples

### 1. Distributed Analytics

```bash
# Generate analytics with Ray workers
curl -X POST "http://localhost:8080/analytics/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "data_size": 10000,
    "metrics": ["mean", "std", "correlation"],
    "workers": 3
  }'
```

### 2. Machine Learning Training

```bash
# Train a model using distributed workers
curl -X POST "http://localhost:8080/ml/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "linear_regression",
    "data_size": 1000,
    "features": 5,
    "distributed": true
  }'
```

### 3. Data Processing Pipeline

```bash
# Execute data processing pipeline
curl -X POST "http://localhost:8080/data/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "stages": [
      {"type": "validate", "config": {"required_fields": ["id", "name"]}},
      {"type": "clean", "config": {}},
      {"type": "transform", "config": {"operation": "normalize"}}
    ],
    "data": [{"id": 1, "name": "test", "value": 100}]
  }'
```

### 4. System Monitoring

```bash
# Get comprehensive system metrics
curl -X POST "http://localhost:8080/monitoring/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_types": ["system", "ray", "application"]
  }'
```

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file:

```env
# Application settings
APP_NAME=FastAPI Ray Cluster
DEBUG=false

# Server settings
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8080

# Ray settings
RAY_ADDRESS=ray://ray-head:10001
RAY_NAMESPACE=fastapi

# Worker settings
COMPUTE_WORKERS=3
ML_TRAINERS=2
DATA_PROCESSORS=3

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

## 🐳 Docker Services

The deployment includes the following services:

- **ray-head**: Ray cluster head node with dashboard
- **ray-worker-1/2**: Ray worker nodes
- **fastapi-app**: FastAPI application server
- **ray-monitor**: Optional monitoring service

## 📊 Monitoring

### Ray Dashboard
Access the Ray dashboard at http://localhost:8265 to monitor:
- Cluster resources (CPU, memory, GPU)
- Running tasks and actors
- Job scheduling and execution

### Application Metrics
The monitoring service provides:
- System resource utilization
- Application performance metrics
- Ray cluster health status
- Custom business metrics

## 🔍 Development

### Running Tests

```bash
cd fastapi
python -m pytest tests/ -v
```

### Code Formatting

```bash
black .
isort .
flake8 .
```

### Adding New Services

1. Create a new sub-app in `apps/`
2. Add corresponding workers in `workers/`
3. Mount the sub-app in `main.py`
4. Update the Docker configuration if needed

## 🚀 Scaling

### Horizontal Scaling

Scale Ray workers:
```bash
./scripts/deploy_fastapi.sh scale 5
```

Scale FastAPI instances (requires load balancer):
```bash
docker-compose up -d --scale fastapi-app=3
```

### Vertical Scaling

Adjust resource limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

## 🛡️ Production Considerations

1. **Security**: Add authentication and authorization
2. **Monitoring**: Integrate with Prometheus/Grafana
3. **Logging**: Use structured logging (JSON)
4. **Load Balancing**: Add nginx or other load balancer
5. **SSL/TLS**: Enable HTTPS for production
6. **Resource Limits**: Set appropriate CPU/memory limits
7. **Health Checks**: Configure proper health check endpoints

## 🐛 Troubleshooting

### Common Issues

1. **Ray connection failed**:
   ```bash
   # Check Ray cluster status
   ./scripts/deploy_fastapi.sh exec ray-head ray status
   ```

2. **Service not responding**:
   ```bash
   # Check service logs
   ./scripts/deploy_fastapi.sh logs fastapi-app
   ```

3. **Out of memory**:
   ```bash
   # Scale down workers or increase memory limits
   ./scripts/deploy_fastapi.sh scale 2
   ```

### Log Analysis

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi-app
docker-compose logs -f ray-head
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/)