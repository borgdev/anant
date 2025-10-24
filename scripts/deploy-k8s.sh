#!/bin/bash

# Anant Enterprise Ray Cluster - Kubernetes/Helm Deployment Script
# Deploys Ray cluster to Kubernetes using Helm

set -euo pipefail

# Configuration
NAMESPACE="anant-enterprise"
RELEASE_NAME="anant-enterprise"
CHART_PATH="./helm/anant-enterprise"
VALUES_FILE="values.yaml"
KUBECTL_TIMEOUT="300s"

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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if chart exists
    if [ ! -d "$CHART_PATH" ]; then
        log_error "Helm chart not found at $CHART_PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace $NAMESPACE..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespace $NAMESPACE ready"
}

# Install/upgrade Helm dependencies
install_dependencies() {
    log_info "Installing Helm chart dependencies..."
    
    cd "$CHART_PATH"
    helm dependency update
    cd - > /dev/null
    
    log_success "Helm dependencies updated"
}

# Validate Helm chart
validate_chart() {
    log_info "Validating Helm chart..."
    
    helm lint "$CHART_PATH"
    
    log_success "Helm chart validation passed"
}

# Deploy to development
deploy_development() {
    log_info "Deploying to development environment..."
    
    local dev_values="$CHART_PATH/values-dev.yaml"
    
    # Create development values if not exists
    if [ ! -f "$dev_values" ]; then
        cat > "$dev_values" << EOF
# Development overrides
development:
  enabled: true
  debugMode: true
  jupyterEnabled: true

rayHead:
  replicaCount: 1
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 500m
      memory: 1Gi

rayWorkers:
  geometric:
    replicaCount: 1
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
      requests:
        cpu: 250m
        memory: 512Mi
  contextual:
    replicaCount: 1
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
      requests:
        cpu: 250m
        memory: 512Mi
  multipurpose:
    enabled: false

postgresql:
  primary:
    persistence:
      size: 5Gi
    resources:
      limits:
        memory: 1Gi
      requests:
        memory: 256Mi

redis:
  master:
    persistence:
      size: 2Gi
    resources:
      limits:
        memory: 512Mi
      requests:
        memory: 128Mi

monitoring:
  prometheus:
    enabled: false
  grafana:
    enabled: false
EOF
    fi
    
    helm upgrade --install "$RELEASE_NAME-dev" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$dev_values" \
        --timeout "$KUBECTL_TIMEOUT" \
        --wait
    
    log_success "Development deployment completed"
}

# Deploy to production
deploy_production() {
    log_info "Deploying to production environment..."
    
    local prod_values="$CHART_PATH/values-prod.yaml"
    
    # Create production values if not exists
    if [ ! -f "$prod_values" ]; then
        cat > "$prod_values" << EOF
# Production overrides
anant:
  mode: "production"
  security:
    enabled: true
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
    replicaCount: 3
  contextual:
    replicaCount: 3
  multipurpose:
    replicaCount: 2

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10

persistence:
  data:
    size: 100Gi
  logs:
    size: 50Gi

postgresql:
  primary:
    persistence:
      size: 50Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true

ingress:
  enabled: true
  hosts:
    - host: anant-enterprise.example.com
      paths:
        - path: /
          pathType: Prefix
EOF
    fi
    
    helm upgrade --install "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$prod_values" \
        --timeout "$KUBECTL_TIMEOUT" \
        --wait
    
    log_success "Production deployment completed"
}

# Deploy with custom values
deploy_custom() {
    local custom_values="$1"
    local release_suffix="${2:-}"
    
    if [ ! -f "$custom_values" ]; then
        log_error "Custom values file not found: $custom_values"
        exit 1
    fi
    
    local release="${RELEASE_NAME}${release_suffix:+-$release_suffix}"
    
    log_info "Deploying with custom values: $custom_values"
    
    helm upgrade --install "$release" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$custom_values" \
        --timeout "$KUBECTL_TIMEOUT" \
        --wait
    
    log_success "Custom deployment completed: $release"
}

# Show status
show_status() {
    local release="${1:-$RELEASE_NAME}"
    
    log_info "Showing status for release: $release"
    
    # Helm status
    echo -e "\n${BLUE}Helm Status:${NC}"
    helm status "$release" -n "$NAMESPACE"
    
    # Pod status
    echo -e "\n${BLUE}Pod Status:${NC}"
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release"
    
    # Service status
    echo -e "\n${BLUE}Service Status:${NC}"
    kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release"
    
    # Ingress status (if exists)
    if kubectl get ingress -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release" &> /dev/null; then
        echo -e "\n${BLUE}Ingress Status:${NC}"
        kubectl get ingress -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release"
    fi
}

# Show logs
show_logs() {
    local release="${1:-$RELEASE_NAME}"
    local component="${2:-ray-head}"
    local lines="${3:-100}"
    
    log_info "Showing logs for $component in release: $release"
    
    local pod_selector="app.kubernetes.io/instance=$release,app.kubernetes.io/component=$component"
    
    kubectl logs -n "$NAMESPACE" -l "$pod_selector" --tail="$lines" -f
}

# Port forward services
port_forward() {
    local service="${1:-api}"
    local local_port="${2:-8000}"
    
    case "$service" in
        "api")
            local remote_port="8000"
            local k8s_service="$RELEASE_NAME-ray-head"
            ;;
        "dashboard")
            local remote_port="8265"
            local k8s_service="$RELEASE_NAME-ray-head"
            ;;
        "grafana")
            local remote_port="80"
            local k8s_service="$RELEASE_NAME-grafana"
            ;;
        "prometheus")
            local remote_port="80"
            local k8s_service="$RELEASE_NAME-prometheus-server"
            ;;
        *)
            log_error "Unknown service: $service"
            log_info "Available services: api, dashboard, grafana, prometheus"
            exit 1
            ;;
    esac
    
    log_info "Port forwarding $service: http://localhost:$local_port"
    log_warning "Press Ctrl+C to stop port forwarding"
    
    kubectl port-forward -n "$NAMESPACE" "service/$k8s_service" "$local_port:$remote_port"
}

# Scale deployment
scale_deployment() {
    local component="$1"
    local replicas="$2"
    local release="${3:-$RELEASE_NAME}"
    
    log_info "Scaling $component to $replicas replicas in release: $release"
    
    local deployment_name="$release-ray-worker-$component"
    
    kubectl scale deployment "$deployment_name" -n "$NAMESPACE" --replicas="$replicas"
    
    log_success "Deployment $deployment_name scaled to $replicas replicas"
}

# Cleanup deployment
cleanup() {
    local release="${1:-$RELEASE_NAME}"
    local delete_namespace="${2:-false}"
    
    log_info "Cleaning up release: $release"
    
    # Uninstall Helm release
    helm uninstall "$release" -n "$NAMESPACE" || true
    
    # Delete PVCs if requested
    if [ "${3:-}" = "--delete-data" ]; then
        log_warning "Deleting persistent data..."
        kubectl delete pvc -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release" || true
    fi
    
    # Delete namespace if requested and empty
    if [ "$delete_namespace" = "true" ]; then
        log_warning "Deleting namespace: $NAMESPACE"
        kubectl delete namespace "$NAMESPACE" || true
    fi
    
    log_success "Cleanup completed for release: $release"
}

# Health check
health_check() {
    local release="${1:-$RELEASE_NAME}"
    local timeout=300
    local interval=10
    local elapsed=0
    
    log_info "Performing health check for release: $release"
    
    while [ $elapsed -lt $timeout ]; do
        # Check if all pods are ready
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release" \
            -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -o "True" | wc -l)
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release" \
            --no-headers | wc -l)
        
        if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
            # Try to reach the API
            local api_pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$release,app.kubernetes.io/component=ray-head" \
                -o jsonpath='{.items[0].metadata.name}')
            
            if kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -f -s http://localhost:8000/health &> /dev/null; then
                log_success "Health check passed"
                return 0
            fi
        fi
        
        log_info "Health check: $ready_pods/$total_pods pods ready. Retrying in ${interval}s... (${elapsed}s elapsed)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log_error "Health check failed after ${timeout}s"
    return 1
}

# Usage information
usage() {
    cat << EOF
Anant Enterprise Ray Cluster - Kubernetes Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy-dev                    Deploy development environment
    deploy-prod                   Deploy production environment
    deploy-custom VALUES [SUFFIX] Deploy with custom values file
    status [RELEASE]              Show deployment status
    logs [RELEASE] [COMPONENT] [LINES] Show logs
    port-forward SERVICE [PORT]   Port forward service (api, dashboard, grafana, prometheus)
    scale COMPONENT REPLICAS [RELEASE] Scale worker deployment
    health [RELEASE]              Perform health check
    cleanup [RELEASE] [--delete-namespace] [--delete-data] Cleanup deployment
    help                          Show this help message

Examples:
    $0 deploy-dev
    $0 deploy-prod
    $0 deploy-custom ./custom-values.yaml staging
    $0 status
    $0 logs anant-enterprise ray-head 200
    $0 port-forward api 8000
    $0 scale geometric 5
    $0 cleanup anant-enterprise-dev --delete-data

EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        "deploy-dev")
            check_prerequisites
            create_namespace
            install_dependencies
            validate_chart
            deploy_development
            health_check "$RELEASE_NAME-dev"
            show_status "$RELEASE_NAME-dev"
            ;;
        "deploy-prod")
            check_prerequisites
            create_namespace
            install_dependencies
            validate_chart
            deploy_production
            health_check
            show_status
            ;;
        "deploy-custom")
            if [ $# -lt 2 ]; then
                log_error "Custom deployment requires values file"
                exit 1
            fi
            check_prerequisites
            create_namespace
            install_dependencies
            validate_chart
            deploy_custom "$2" "${3:-}"
            ;;
        "status")
            show_status "${2:-$RELEASE_NAME}"
            ;;
        "logs")
            show_logs "${2:-$RELEASE_NAME}" "${3:-ray-head}" "${4:-100}"
            ;;
        "port-forward")
            if [ $# -lt 2 ]; then
                log_error "Port forward requires service name"
                exit 1
            fi
            port_forward "$2" "${3:-8000}"
            ;;
        "scale")
            if [ $# -lt 3 ]; then
                log_error "Scale command requires: component replicas"
                exit 1
            fi
            scale_deployment "$2" "$3" "${4:-$RELEASE_NAME}"
            ;;
        "health")
            health_check "${2:-$RELEASE_NAME}"
            ;;
        "cleanup")
            cleanup "${2:-$RELEASE_NAME}" "${3:-false}" "${4:-}"
            ;;
        "help"|*)
            usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"