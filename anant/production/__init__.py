"""
ANANT Production Deployment Framework
====================================

Enterprise-grade production deployment system for ANANT's Polars+Parquet architecture.
Provides comprehensive monitoring, logging, performance optimization, and scalability.

Core Components:
- Production Manager: Central orchestration and lifecycle management
- Performance Monitor: Real-time performance tracking and optimization
- Scalability Engine: Automatic scaling and load distribution
- Health Checker: System health monitoring and alerting
- Deployment Orchestrator: Blue-green deployments and rollbacks
- Resource Optimizer: Memory, CPU, and storage optimization

Features:
- Enterprise monitoring with comprehensive metrics collection
- Distributed deployment across multiple nodes
- Automatic performance tuning for Polars operations
- Real-time health checks and alerting
- Comprehensive logging and audit trails
- Zero-downtime deployments
"""

from .core.production_manager import ProductionManager
from .core.config_manager import ProductionConfig, EnvironmentType
from .monitoring.performance_monitor import PerformanceMonitor, MetricsCollector
from .monitoring.health_checker import HealthChecker, HealthStatus
from .deployment.orchestrator import DeploymentOrchestrator, DeploymentStrategy
from .optimization.resource_optimizer import ResourceOptimizer, OptimizationConfig
from .scaling.auto_scaler import AutoScaler, ScalingPolicy

__version__ = "1.0.0-production"
__author__ = "ANANT Production Team"

# Main exports
__all__ = [
    # Core management
    "ProductionManager",
    "ProductionConfig",
    "EnvironmentType",
    
    # Monitoring
    "PerformanceMonitor", 
    "MetricsCollector",
    "HealthChecker",
    "HealthStatus",
    
    # Deployment
    "DeploymentOrchestrator",
    "DeploymentStrategy",
    
    # Optimization
    "ResourceOptimizer",
    "OptimizationConfig", 
    
    # Scaling
    "AutoScaler",
    "ScalingPolicy",
    
    # Convenience functions
    "create_production_environment",
    "deploy_anant_service",
    "monitor_anant_cluster"
]

def create_production_environment(environment: str = "production",
                                 storage_path: str = "/data/anant",
                                 monitoring_enabled: bool = True,
                                 auto_scaling: bool = True) -> ProductionManager:
    """
    Create a fully configured production environment for ANANT.
    
    Args:
        environment: Environment type (production, staging, development)
        storage_path: Base path for Parquet storage
        monitoring_enabled: Enable comprehensive monitoring
        auto_scaling: Enable automatic scaling
        
    Returns:
        Configured ProductionManager instance
        
    Example:
        >>> prod_mgr = create_production_environment("production")
        >>> prod_mgr.start_services()
        >>> status = prod_mgr.get_cluster_status()
    """
    config = ProductionConfig(
        environment=EnvironmentType(environment),
        storage_path=storage_path,
        monitoring_enabled=monitoring_enabled,
        auto_scaling_enabled=auto_scaling
    )
    
    return ProductionManager(config)

def deploy_anant_service(service_name: str,
                        replicas: int = 3,
                        strategy: str = "blue_green") -> bool:
    """
    Deploy ANANT service with specified configuration.
    
    Args:
        service_name: Name of the service to deploy
        replicas: Number of service replicas
        strategy: Deployment strategy (blue_green, rolling, canary)
        
    Returns:
        True if deployment successful
    """
    orchestrator = DeploymentOrchestrator()
    return orchestrator.deploy_service(
        service_name=service_name,
        replicas=replicas,
        strategy=strategy
    )

def monitor_anant_cluster(cluster_id: str = "default") -> dict:
    """
    Get comprehensive monitoring data for ANANT cluster.
    
    Args:
        cluster_id: Cluster identifier
        
    Returns:
        Dictionary with cluster metrics and health status
    """
    monitor = PerformanceMonitor()
    health_checker = HealthChecker()
    
    return {
        "performance": monitor.get_cluster_metrics(cluster_id),
        "health": health_checker.check_cluster_health(cluster_id),
        "timestamp": monitor.get_current_timestamp()
    }

# Production readiness check
def validate_production_readiness() -> dict:
    """Validate that all production components are ready."""
    
    checks = {
        "polars_config": False,
        "parquet_storage": False,
        "monitoring_stack": False,
        "security_config": False,
        "network_config": False,
        "backup_strategy": False
    }
    
    try:
        # Check Polars configuration
        import polars as pl
        # Basic Polars functionality test
        test_df = pl.DataFrame({"test": [1, 2, 3]})
        if test_df.height == 3:
            checks["polars_config"] = True
            
        # Check other components
        # (Implementation would check actual production requirements)
        
    except Exception as e:
        print(f"Production readiness check failed: {e}")
        
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "score": sum(checks.values()) / len(checks) * 100
    }