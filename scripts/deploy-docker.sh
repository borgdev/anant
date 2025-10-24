#!/bin/bash

# Anant Enterprise Ray Cluster - Docker Deployment Script
# Builds and deploys the complete Ray cluster using Docker Compose

set -euo pipefail

# Configuration
PROJECT_NAME="anant-enterprise"
DOCKER_REGISTRY="docker.io"
IMAGE_NAME="anant/enterprise-ray"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose (v2 format)
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build production image
    docker build \
        --target production \
        --tag "${IMAGE_NAME}:${VERSION}" \
        --tag "${IMAGE_NAME}:latest" \
        .
    
    # Build development image
    docker build \
        --target development \
        --tag "${IMAGE_NAME}:dev" \
        .
    
    log_success "Docker images built successfully"
}

# Deploy development environment
deploy_development() {
    log_info "Deploying development environment..."
    
    # Stop existing containers
    docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" down
    
    # Start development environment
    docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" up -d
    
    log_success "Development environment deployed"
    log_info "Ray Dashboard: http://localhost:8265"
    log_info "Anant API: http://localhost:8000"
    log_info "Jupyter Lab: http://localhost:8888 (token: anant_dev)"
}

# Deploy production environment
deploy_production() {
    log_info "Deploying production environment..."
    
    # Create necessary directories
    mkdir -p data logs config
    
    # Stop existing containers
    docker compose -f docker-compose.yml -p "${PROJECT_NAME}" down
    
    # Start production environment
    docker compose -f docker-compose.yml -p "${PROJECT_NAME}" up -d
    
    log_success "Production environment deployed"
    log_info "Ray Dashboard: http://localhost:8265"
    log_info "Anant API: http://localhost:8000"
    log_info "Grafana: http://localhost:3000 (admin/anant_admin_2024)"
    log_info "Prometheus: http://localhost:9090"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Show status
show_status() {
    log_info "Container status:"
    
    if [ "$1" = "dev" ]; then
        docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" ps
    else
        docker compose -f docker-compose.yml -p "${PROJECT_NAME}" ps
    fi
}

# Show logs
show_logs() {
    local service=${2:-""}
    
    if [ "$1" = "dev" ]; then
        if [ -n "$service" ]; then
            docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" logs -f "$service"
        else
            docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" logs -f
        fi
    else
        if [ -n "$service" ]; then
            docker compose -f docker-compose.yml -p "${PROJECT_NAME}" logs -f "$service"
        else
            docker compose -f docker-compose.yml -p "${PROJECT_NAME}" logs -f
        fi
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker compose -f docker-compose.yml -p "${PROJECT_NAME}" down --volumes --remove-orphans
    docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" down --volumes --remove-orphans
    
    # Remove images (optional)
    if [ "${1:-}" = "--remove-images" ]; then
        docker rmi "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest" "${IMAGE_NAME}:dev" 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

# Scale services
scale_services() {
    local environment=$1
    local service=$2
    local replicas=$3
    
    log_info "Scaling $service to $replicas replicas in $environment environment..."
    
    if [ "$environment" = "dev" ]; then
        docker compose -f docker-compose.dev.yml -p "${PROJECT_NAME}-dev" up -d --scale "$service=$replicas"
    else
        docker compose -f docker-compose.yml -p "${PROJECT_NAME}" up -d --scale "$service=$replicas"
    fi
    
    log_success "Service $service scaled to $replicas replicas"
}

# Usage information
usage() {
    cat << EOF
Anant Enterprise Ray Cluster - Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build               Build Docker images
    deploy-dev          Deploy development environment
    deploy-prod         Deploy production environment
    status [dev|prod]   Show container status
    logs [dev|prod] [service]  Show logs
    health              Perform health check
    scale ENV SERVICE N Scale service to N replicas
    cleanup [--remove-images]  Stop and cleanup containers
    help                Show this help message

Examples:
    $0 build
    $0 deploy-dev
    $0 deploy-prod
    $0 status prod
    $0 logs dev anant-ray-head-dev
    $0 scale prod anant-ray-worker-geo-1 3
    $0 cleanup --remove-images

EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_prerequisites
            build_images
            ;;
        "deploy-dev")
            check_prerequisites
            build_images
            deploy_development
            health_check
            show_status "dev"
            ;;
        "deploy-prod")
            check_prerequisites
            build_images
            deploy_production
            health_check
            show_status "prod"
            ;;
        "status")
            show_status "${2:-prod}"
            ;;
        "logs")
            show_logs "${2:-prod}" "${3:-}"
            ;;
        "health")
            health_check
            ;;
        "scale")
            if [ $# -lt 4 ]; then
                log_error "Scale command requires: environment service replicas"
                exit 1
            fi
            scale_services "$2" "$3" "$4"
            ;;
        "cleanup")
            cleanup "${2:-}"
            ;;
        "help"|*)
            usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"