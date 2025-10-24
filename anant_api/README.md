# Anant Graph API

A FastAPI-based REST API for Anant graph operations, running within the Ray cluster for distributed processing.

## Overview

The Anant Graph API provides a REST interface to the Anant graph system with the following capabilities:

- **Graph Operations**: CRUD operations for nodes and edges
- **Graph Analytics**: Distributed graph analytics and pattern recognition
- **Knowledge Graph**: Entity extraction and semantic querying
- **Monitoring**: System and performance monitoring
- **Authentication**: JWT token and API key authentication

## Architecture

- **Ray Integration**: Runs within the Ray cluster for distributed processing
- **FastAPI Framework**: Modern async web framework with automatic API documentation
- **PostgreSQL**: Database for user management and metadata
- **Redis**: Caching and session management
- **Authentication**: JWT tokens and API keys for secure access

## Services

### Graph Operations (`/graph`)
- Node and edge CRUD operations
- Graph querying and traversal
- Statistics and metrics

### Analytics (`/analytics`)
- Distributed graph analytics
- Pattern recognition
- Centrality analysis

### Knowledge Graph (`/knowledge`)
- Entity and relation management
- Text-based knowledge extraction
- Semantic querying

### Monitoring (`/monitoring`)
- System metrics collection
- Ray cluster monitoring
- Performance dashboards

## Deployment

The API is deployed within the Ray cluster:

### Production
```bash
docker-compose --profile production up -d
```
- Access API at: http://localhost:8888
- Ray Dashboard at: http://localhost:8285

### Development
```bash
docker-compose --profile development up -d
```
- Access API at: http://localhost:8889 (with hot reload)
- Ray Dashboard at: http://localhost:8286

## API Documentation

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Authentication

The API supports dual authentication:

1. **JWT Tokens**: For user-based access
   ```bash
   curl -H "Authorization: Bearer YOUR_JWT_TOKEN" http://localhost:8888/graph/nodes
   ```

2. **API Keys**: For service-to-service access
   ```bash
   curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8888/graph/nodes
   ```

## Configuration

Key environment variables:
- `ENVIRONMENT`: production/development/testing
- `RAY_ADDRESS`: Ray cluster address
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key

## Health Checks

- Main health: `/health`
- Graph service: `/graph/health`
- Analytics service: `/analytics/health`
- Knowledge service: `/knowledge/health`
- Monitoring service: `/monitoring/health`

## Example Usage

### Create a Node
```bash
curl -X POST http://localhost:8888/graph/nodes \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "node_id": "person_1",
    "node_type": "person",
    "properties": {"name": "John Doe", "age": 30}
  }'
```

### Query Graph
```bash
curl -X POST http://localhost:8888/graph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "query_type": "neighbors",
    "parameters": {"node_id": "person_1", "depth": 2}
  }'
```

### Get System Status
```bash
curl http://localhost:8888/cluster/status
```