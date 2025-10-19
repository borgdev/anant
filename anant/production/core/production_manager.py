"""
Production Manager
==================

Central orchestration and lifecycle management for ANANT production deployments.
Manages the entire production environment including services, monitoring, and scaling.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import polars as pl

from .config_manager import ProductionConfig, EnvironmentType


@dataclass
class ServiceStatus:
    """Status information for a service."""
    name: str
    status: str  # running, stopped, starting, stopping, error
    health: str  # healthy, unhealthy, unknown
    replicas: int
    ready_replicas: int
    cpu_usage: float
    memory_usage: float
    last_updated: datetime
    errors: List[str]


@dataclass
class ClusterStatus:
    """Overall cluster status."""
    cluster_id: str
    total_nodes: int
    ready_nodes: int
    services: List[ServiceStatus]
    overall_health: str
    cpu_total: float
    cpu_used: float
    memory_total: float
    memory_used: float
    storage_total: float
    storage_used: float
    last_updated: datetime


class ProductionManager:
    """
    Central production management system for ANANT.
    
    Handles:
    - Service lifecycle management
    - Monitoring and health checks
    - Auto-scaling and load balancing
    - Configuration management
    - Backup and disaster recovery
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.cluster_id = f"anant-{config.environment.value}"
        self.services: Dict[str, ServiceStatus] = {}
        self.is_running = False
        self.background_tasks: List[threading.Thread] = []
        
        # Initialize components based on configuration
        self._init_storage()
        self._init_monitoring()
        self._init_logging()
        
        print(f"🚀 ProductionManager initialized for {config.environment.value} environment")
    
    def _init_storage(self):
        """Initialize storage subsystem."""
        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_path / "data").mkdir(exist_ok=True)
        (storage_path / "metadata").mkdir(exist_ok=True)
        (storage_path / "backups").mkdir(exist_ok=True)
        (storage_path / "logs").mkdir(exist_ok=True)
        
        print(f"📁 Storage initialized at {storage_path}")
    
    def _init_monitoring(self):
        """Initialize monitoring subsystem."""
        if self.config.monitoring_enabled:
            self.metrics_collector = None  # Will be initialized when monitoring classes are created
            print("📊 Monitoring subsystem initialized")
    
    def _init_logging(self):
        """Initialize logging subsystem."""
        import logging
        
        log_level = getattr(logging, self.config.monitoring.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.storage_path}/logs/anant-production.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("anant.production")
        self.logger.info(f"Logging initialized at {log_level} level")
    
    def start_services(self) -> bool:
        """Start all production services."""
        try:
            self.logger.info("Starting ANANT production services...")
            self.is_running = True
            
            # Start core services
            self._start_core_services()
            
            # Start monitoring if enabled
            if self.config.monitoring_enabled:
                self._start_monitoring_services()
            
            # Start auto-scaling if enabled
            if self.config.auto_scaling_enabled:
                self._start_scaling_services()
            
            # Start background management tasks
            self._start_background_tasks()
            
            self.logger.info("✅ All services started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start services: {e}")
            return False
    
    def stop_services(self) -> bool:
        """Stop all production services gracefully."""
        try:
            self.logger.info("Stopping ANANT production services...")
            self.is_running = False
            
            # Stop background tasks
            for task in self.background_tasks:
                if task.is_alive():
                    task.join(timeout=30)
            
            # Stop services in reverse order
            self._stop_scaling_services()
            self._stop_monitoring_services()
            self._stop_core_services()
            
            self.logger.info("✅ All services stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop services: {e}")
            return False
    
    def _start_core_services(self):
        """Start core ANANT services."""
        # Simulate core service startup
        services = [
            "anant-hypergraph-api",
            "anant-metagraph-api", 
            "anant-query-processor",
            "anant-storage-manager"
        ]
        
        for service_name in services:
            status = ServiceStatus(
                name=service_name,
                status="running",
                health="healthy",
                replicas=self.config.scaling.min_replicas,
                ready_replicas=self.config.scaling.min_replicas,
                cpu_usage=0.1,
                memory_usage=0.2,
                last_updated=datetime.now(),
                errors=[]
            )
            self.services[service_name] = status
            self.logger.info(f"✅ Started {service_name}")
    
    def _start_monitoring_services(self):
        """Start monitoring services."""
        monitoring_services = [
            "anant-metrics-collector",
            "anant-health-checker",
            "anant-log-aggregator"
        ]
        
        if self.config.monitoring.prometheus_enabled:
            monitoring_services.append("prometheus")
        if self.config.monitoring.grafana_enabled:
            monitoring_services.append("grafana")
        
        for service_name in monitoring_services:
            status = ServiceStatus(
                name=service_name,
                status="running",
                health="healthy",
                replicas=1,
                ready_replicas=1,
                cpu_usage=0.05,
                memory_usage=0.1,
                last_updated=datetime.now(),
                errors=[]
            )
            self.services[service_name] = status
            self.logger.info(f"📊 Started {service_name}")
    
    def _start_scaling_services(self):
        """Start auto-scaling services."""
        scaling_services = [
            "anant-auto-scaler",
            "anant-load-balancer"
        ]
        
        for service_name in scaling_services:
            status = ServiceStatus(
                name=service_name,
                status="running",
                health="healthy",
                replicas=1,
                ready_replicas=1,
                cpu_usage=0.02,
                memory_usage=0.05,
                last_updated=datetime.now(),
                errors=[]
            )
            self.services[service_name] = status
            self.logger.info(f"⚖️ Started {service_name}")
    
    def _start_background_tasks(self):
        """Start background management tasks."""
        # Health monitoring task
        health_task = threading.Thread(
            target=self._health_monitoring_loop,
            name="health-monitor",
            daemon=True
        )
        health_task.start()
        self.background_tasks.append(health_task)
        
        # Resource monitoring task
        resource_task = threading.Thread(
            target=self._resource_monitoring_loop,
            name="resource-monitor", 
            daemon=True
        )
        resource_task.start()
        self.background_tasks.append(resource_task)
        
        # Backup task (if enabled)
        if self.config.storage.backup_enabled:
            backup_task = threading.Thread(
                target=self._backup_loop,
                name="backup-manager",
                daemon=True
            )
            backup_task.start()
            self.background_tasks.append(backup_task)
    
    def _health_monitoring_loop(self):
        """Continuous health monitoring."""
        while self.is_running:
            try:
                self._check_service_health()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _resource_monitoring_loop(self):
        """Continuous resource monitoring."""
        while self.is_running:
            try:
                self._collect_resource_metrics()
                time.sleep(60)  # Collect every minute
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(120)
    
    def _backup_loop(self):
        """Periodic backup management."""
        while self.is_running:
            try:
                self._perform_backup()
                # Sleep for 24 hours
                time.sleep(24 * 60 * 60)
            except Exception as e:
                self.logger.error(f"Backup error: {e}")
                time.sleep(60 * 60)  # Retry in 1 hour
    
    def _check_service_health(self):
        """Check health of all services."""
        for service_name, service in self.services.items():
            try:
                # Simulate health check
                # In real implementation, this would make HTTP health check calls
                if service.status == "running":
                    service.health = "healthy"
                    service.last_updated = datetime.now()
                else:
                    service.health = "unhealthy"
                    
            except Exception as e:
                service.health = "unhealthy"
                service.errors.append(f"Health check failed: {e}")
                self.logger.warning(f"Health check failed for {service_name}: {e}")
    
    def _collect_resource_metrics(self):
        """Collect resource usage metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(self.config.storage_path)
            
            # Update service metrics (simulated)
            for service_name, service in self.services.items():
                # Simulate some variance in resource usage
                import random
                service.cpu_usage = max(0.01, service.cpu_usage + random.uniform(-0.02, 0.02))
                service.memory_usage = max(0.05, service.memory_usage + random.uniform(-0.05, 0.05))
                service.last_updated = datetime.now()
                
            self.logger.debug(f"Resource metrics: CPU={cpu_percent}%, Memory={memory.percent}%, Disk={disk.percent}%")
            
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
    
    def _perform_backup(self):
        """Perform data backup."""
        try:
            backup_path = Path(self.config.storage_path) / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup metadata and configuration
            config_backup = backup_path / "config.json"
            with open(config_backup, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            self.logger.info(f"✅ Backup completed: {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backup files."""
        try:
            backup_dir = Path(self.config.storage_path) / "backups"
            retention_date = datetime.now() - timedelta(days=self.config.storage.backup_retention_days)
            
            for backup_path in backup_dir.iterdir():
                if backup_path.is_dir():
                    # Extract date from backup directory name
                    try:
                        backup_date_str = backup_path.name.split('_')[1] + '_' + backup_path.name.split('_')[2]
                        backup_date = datetime.strptime(backup_date_str, '%Y%m%d_%H%M%S')
                        
                        if backup_date < retention_date:
                            import shutil
                            shutil.rmtree(backup_path)
                            self.logger.info(f"🗑️ Cleaned up old backup: {backup_path}")
                            
                    except (IndexError, ValueError):
                        # Skip directories that don't match backup naming pattern
                        continue
                        
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def _stop_core_services(self):
        """Stop core services."""
        for service_name in list(self.services.keys()):
            if "anant-" in service_name and "monitor" not in service_name and "scaler" not in service_name:
                self.services[service_name].status = "stopped"
                self.logger.info(f"⏹️ Stopped {service_name}")
    
    def _stop_monitoring_services(self):
        """Stop monitoring services."""
        for service_name in list(self.services.keys()):
            if any(x in service_name for x in ["monitor", "metrics", "prometheus", "grafana"]):
                self.services[service_name].status = "stopped"
                self.logger.info(f"📊 Stopped {service_name}")
    
    def _stop_scaling_services(self):
        """Stop scaling services."""
        for service_name in list(self.services.keys()):
            if any(x in service_name for x in ["scaler", "balancer"]):
                self.services[service_name].status = "stopped"
                self.logger.info(f"⚖️ Stopped {service_name}")
    
    def get_cluster_status(self) -> ClusterStatus:
        """Get current cluster status."""
        running_services = [s for s in self.services.values() if s.status == "running"]
        healthy_services = [s for s in running_services if s.health == "healthy"]
        
        # Calculate overall health
        if len(healthy_services) == len(running_services) and len(running_services) > 0:
            overall_health = "healthy"
        elif len(healthy_services) > len(running_services) * 0.5:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"
        
        # Aggregate resource usage
        total_cpu = sum(s.cpu_usage for s in running_services)
        total_memory = sum(s.memory_usage for s in running_services)
        
        return ClusterStatus(
            cluster_id=self.cluster_id,
            total_nodes=1,  # Single node for now
            ready_nodes=1 if overall_health != "unhealthy" else 0,
            services=list(self.services.values()),
            overall_health=overall_health,
            cpu_total=self.config.resources.max_cpu_cores,
            cpu_used=total_cpu,
            memory_total=self.config.resources.max_memory_gb,
            memory_used=total_memory,
            storage_total=self.config.resources.max_storage_gb,
            storage_used=0.1,  # Simulated
            last_updated=datetime.now()
        )
    
    def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale a specific service."""
        if service_name not in self.services:
            self.logger.error(f"Service {service_name} not found")
            return False
        
        if replicas < 1 or replicas > self.config.scaling.max_replicas:
            self.logger.error(f"Invalid replica count: {replicas}")
            return False
        
        service = self.services[service_name]
        old_replicas = service.replicas
        service.replicas = replicas
        service.ready_replicas = replicas  # Simulate immediate readiness
        service.last_updated = datetime.now()
        
        self.logger.info(f"⚖️ Scaled {service_name} from {old_replicas} to {replicas} replicas")
        return True
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        if service_name not in self.services:
            self.logger.error(f"Service {service_name} not found")
            return False
        
        service = self.services[service_name]
        service.status = "restarting"
        service.health = "unknown"
        service.last_updated = datetime.now()
        
        # Simulate restart delay
        import time
        time.sleep(2)
        
        service.status = "running"
        service.health = "healthy"
        service.errors = []
        service.last_updated = datetime.now()
        
        self.logger.info(f"🔄 Restarted {service_name}")
        return True
    
    def get_service_logs(self, service_name: str, lines: int = 100) -> List[str]:
        """Get recent logs for a service."""
        # Simulate log retrieval
        logs = [
            f"[{datetime.now()}] INFO: {service_name} service running normally",
            f"[{datetime.now()}] DEBUG: Processing requests efficiently",
            f"[{datetime.now()}] INFO: Resource usage within normal limits"
        ]
        
        return logs[-lines:]
    
    def execute_maintenance(self, task: str) -> bool:
        """Execute maintenance task."""
        self.logger.info(f"🔧 Executing maintenance task: {task}")
        
        maintenance_tasks = {
            "backup": self._perform_backup,
            "cleanup": self._cleanup_old_backups,
            "health_check": self._check_service_health,
            "resource_check": self._collect_resource_metrics
        }
        
        if task in maintenance_tasks:
            try:
                maintenance_tasks[task]()
                self.logger.info(f"✅ Maintenance task completed: {task}")
                return True
            except Exception as e:
                self.logger.error(f"❌ Maintenance task failed: {task} - {e}")
                return False
        else:
            self.logger.error(f"Unknown maintenance task: {task}")
            return False