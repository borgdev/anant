#!/bin/bash
# Test Anant Graph API Deployment

echo "🚀 Testing Anant Graph API Deployment"
echo "======================================"

# Function to check if service is ready
check_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts: $name not ready yet..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ $name failed to start within expected time"
    return 1
}

# Function to test API endpoint
test_endpoint() {
    local url=$1
    local name=$2
    
    echo "🔍 Testing $name endpoint..."
    response=$(curl -s "$url")
    
    if [ $? -eq 0 ]; then
        echo "✅ $name endpoint working"
        echo "   Response: $(echo $response | jq -r '.message // .service // "OK"' 2>/dev/null || echo "OK")"
    else
        echo "❌ $name endpoint failed"
        return 1
    fi
}

echo ""
echo "1. Starting Docker Compose (production profile)..."
docker-compose --profile production up -d

echo ""
echo "2. Checking service health..."

# Check main API
check_service "http://localhost:8888/health" "Anant Graph API"

# Check Ray Dashboard
check_service "http://localhost:8285" "Ray Dashboard"

# Check database
check_service "http://localhost:5454" "PostgreSQL" || echo "⚠️  PostgreSQL check skipped (no HTTP endpoint)"

echo ""
echo "3. Testing API endpoints..."

# Test main endpoints
test_endpoint "http://localhost:8888/" "Root API"
test_endpoint "http://localhost:8888/health" "Health Check"
test_endpoint "http://localhost:8888/cluster/status" "Ray Cluster Status"

# Test sub-applications
test_endpoint "http://localhost:8888/graph/" "Graph Service"
test_endpoint "http://localhost:8888/analytics/" "Analytics Service"
test_endpoint "http://localhost:8888/knowledge/" "Knowledge Service"
test_endpoint "http://localhost:8888/monitoring/" "Monitoring Service"

echo ""
echo "4. Testing Ray cluster integration..."

# Test Ray cluster status
echo "🔍 Checking Ray cluster resources..."
ray_status=$(curl -s "http://localhost:8888/cluster/status")
if echo "$ray_status" | grep -q "alive_nodes"; then
    echo "✅ Ray cluster integration working"
    nodes=$(echo "$ray_status" | jq -r '.cluster_overview.alive_nodes // 0' 2>/dev/null || echo "unknown")
    echo "   Active nodes: $nodes"
else
    echo "❌ Ray cluster integration failed"
fi

echo ""
echo "5. Checking logs..."
echo "📋 Recent API logs:"
docker logs anant-ray-head --tail 10

echo ""
echo "🎉 Deployment test complete!"
echo ""
echo "📖 Access points:"
echo "   • Anant Graph API: http://localhost:8888"
echo "   • API Documentation: http://localhost:8888/docs"
echo "   • Ray Dashboard: http://localhost:8285"
echo ""
echo "🛠️  Useful commands:"
echo "   • View logs: docker logs anant-ray-head -f"
echo "   • Stop services: docker-compose --profile production down"
echo "   • Check status: curl http://localhost:8888/health"