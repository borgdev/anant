"""
Production Configuration Manager
===============================

Centralized configuration management for ANANT production deployments.
Handles environment-specific settings, resource allocation, and deployment parameters.
"""

import os
import json
import yaml
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl


class EnvironmentType(Enum):
    """Production environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentMode(Enum):
    """Deployment modes for different use cases."""
    SINGLE_NODE = "single_node"
    MULTI_NODE = "multi_node"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    SERVERLESS = "serverless"


@dataclass
class ResourceLimits:
    """Resource allocation limits."""
    max_memory_gb: int = 32
    max_cpu_cores: int = 16
    max_storage_gb: int = 1000
    max_concurrent_operations: int = 100
    parquet_compression: str = "zstd"
    polars_thread_count: Optional[int] = None


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    log_level: str = "INFO"
    metrics_retention_days: int = 30
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    alert_manager_enabled: bool = True
    custom_metrics: List[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security and compliance configuration."""
    auth_enabled: bool = True
    tls_enabled: bool = True
    audit_logging: bool = True
    data_encryption: bool = True
    access_control: bool = True
    compliance_mode: str = "enterprise"  # enterprise, standard, basic
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class StorageConfig:
    """Storage configuration for Parquet and metadata."""
    base_path: str = "/data/anant"
    backup_enabled: bool = True
    backup_retention_days: int = 90
    compression: str = "zstd"
    partition_strategy: str = "date"  # date, hash, range
    replication_factor: int = 3
    storage_tier: str = "ssd"  # ssd, hdd, cloud


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    deployment_mode: DeploymentMode = DeploymentMode.MULTI_NODE
    
    # Core settings
    storage_path: str = "/data/anant"
    service_name: str = "anant-service"
    namespace: str = "anant-production"
    
    # Component configurations
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    
    # Feature flags
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = True
    backup_enabled: bool = True
    high_availability: bool = True
    
    def __post_init__(self):
        """Post-initialization configuration setup."""
        # Set environment-specific defaults
        if self.environment == EnvironmentType.DEVELOPMENT:
            self.resources.max_memory_gb = 8
            self.resources.max_cpu_cores = 4
            self.scaling.min_replicas = 1
            self.scaling.max_replicas = 3
            self.monitoring.prometheus_enabled = False
            
        elif self.environment == EnvironmentType.STAGING:
            self.resources.max_memory_gb = 16
            self.resources.max_cpu_cores = 8
            self.scaling.min_replicas = 2
            self.scaling.max_replicas = 5
            
        # Configure Polars for production
        self._configure_polars()
        
    def _configure_polars(self):
        """Configure Polars for optimal production performance."""
        try:
            # Set thread count using environment variable (Polars respects POLARS_MAX_THREADS)
            if self.resources.polars_thread_count:
                os.environ['POLARS_MAX_THREADS'] = str(self.resources.polars_thread_count)
            else:
                # Auto-detect optimal thread count
                thread_count = min(self.resources.max_cpu_cores, os.cpu_count() or 4)
                os.environ['POLARS_MAX_THREADS'] = str(thread_count)
            
            # Configure Polars settings through context
            self._polars_config = {
                'streaming_chunk_size': 10000,
                'table_width': 120,
                'string_cache': True if self.environment == EnvironmentType.PRODUCTION else False
            }
            
            print(f"Configured Polars with {os.environ.get('POLARS_MAX_THREADS', 'auto')} threads")
                
        except Exception as e:
            print(f"Warning: Failed to configure Polars: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.value,
            'deployment_mode': self.deployment_mode.value,
            'storage_path': self.storage_path,
            'service_name': self.service_name,
            'namespace': self.namespace,
            'resources': {
                'max_memory_gb': self.resources.max_memory_gb,
                'max_cpu_cores': self.resources.max_cpu_cores,
                'max_storage_gb': self.resources.max_storage_gb,
                'max_concurrent_operations': self.resources.max_concurrent_operations,
                'parquet_compression': self.resources.parquet_compression,
                'polars_thread_count': self.resources.polars_thread_count
            },
            'monitoring': {
                'enabled': self.monitoring.enabled,
                'metrics_port': self.monitoring.metrics_port,
                'health_check_port': self.monitoring.health_check_port,
                'log_level': self.monitoring.log_level,
                'prometheus_enabled': self.monitoring.prometheus_enabled,
                'grafana_enabled': self.monitoring.grafana_enabled
            },
            'scaling': {
                'enabled': self.scaling.enabled,
                'min_replicas': self.scaling.min_replicas,
                'max_replicas': self.scaling.max_replicas,
                'target_cpu_percent': self.scaling.target_cpu_percent,
                'target_memory_percent': self.scaling.target_memory_percent
            }
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        path = Path(file_path)
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ProductionConfig':
        """Load configuration from file."""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        
        # Convert back to config object
        config = cls()
        
        if 'environment' in config_dict:
            config.environment = EnvironmentType(config_dict['environment'])
        if 'deployment_mode' in config_dict:
            config.deployment_mode = DeploymentMode(config_dict['deployment_mode'])
        if 'storage_path' in config_dict:
            config.storage_path = config_dict['storage_path']
        if 'service_name' in config_dict:
            config.service_name = config_dict['service_name']
        if 'namespace' in config_dict:
            config.namespace = config_dict['namespace']
            
        # Update nested configurations
        if 'resources' in config_dict:
            resources = config_dict['resources']
            for key, value in resources.items():
                if hasattr(config.resources, key):
                    setattr(config.resources, key, value)
                    
        if 'monitoring' in config_dict:
            monitoring = config_dict['monitoring']
            for key, value in monitoring.items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
                    
        if 'scaling' in config_dict:
            scaling = config_dict['scaling']
            for key, value in scaling.items():
                if hasattr(config.scaling, key):
                    setattr(config.scaling, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Resource validation
        if self.resources.max_memory_gb < 4:
            issues.append("Minimum 4GB memory required for production")
        if self.resources.max_cpu_cores < 2:
            issues.append("Minimum 2 CPU cores required for production")
        if self.resources.max_storage_gb < 100:
            issues.append("Minimum 100GB storage recommended for production")
            
        # Environment-specific validation
        if self.environment == EnvironmentType.PRODUCTION:
            if not self.security.tls_enabled:
                issues.append("TLS must be enabled in production")
            if not self.security.auth_enabled:
                issues.append("Authentication must be enabled in production")
            if self.scaling.min_replicas < 2:
                issues.append("Minimum 2 replicas required for production HA")
                
        # Storage validation
        if not os.path.exists(os.path.dirname(self.storage_path)):
            issues.append(f"Storage path directory does not exist: {self.storage_path}")
            
        return issues


class ConfigManager:
    """Configuration manager for production environments."""
    
    def __init__(self, config_dir: str = "/etc/anant"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def create_default_configs(self):
        """Create default configuration files for all environments."""
        environments = [
            EnvironmentType.DEVELOPMENT,
            EnvironmentType.STAGING, 
            EnvironmentType.PRODUCTION
        ]
        
        for env in environments:
            config = ProductionConfig(environment=env)
            config_file = self.config_dir / f"{env.value}.yaml"
            config.save_to_file(str(config_file))
            print(f"Created default config: {config_file}")
    
    def get_config(self, environment: str) -> ProductionConfig:
        """Get configuration for specified environment."""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if config_file.exists():
            return ProductionConfig.load_from_file(str(config_file))
        else:
            # Return default config for environment
            return ProductionConfig(environment=EnvironmentType(environment))
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all configuration files."""
        results = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            env_name = config_file.stem
            try:
                config = ProductionConfig.load_from_file(str(config_file))
                issues = config.validate()
                results[env_name] = issues
            except Exception as e:
                results[env_name] = [f"Failed to load config: {e}"]
                
        return results