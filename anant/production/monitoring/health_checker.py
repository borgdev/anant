"""
Health Checker System
====================

Comprehensive health monitoring for ANANT production deployments.
Monitors service health, data integrity, and system availability.
"""

import time
import asyncio
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import polars as pl


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    HTTP = "http"
    DATABASE = "database"
    STORAGE = "storage"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_type: CheckType
    interval_seconds: int
    timeout_seconds: int
    critical_threshold: float
    warning_threshold: float
    endpoint: Optional[str] = None
    custom_checker: Optional[Callable] = None


@dataclass
class HealthResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    value: Optional[float]
    message: str
    timestamp: datetime
    response_time: float
    details: Dict[str, Any]


@dataclass
class ServiceHealth:
    """Overall health status for a service."""
    service_name: str
    overall_status: HealthStatus
    checks: List[HealthResult]
    last_updated: datetime
    uptime_percentage: float


class HealthChecker:
    """
    Comprehensive health checking system for ANANT services.
    
    Monitors:
    - Service availability and response times
    - Database connectivity and performance
    - Storage system health and capacity
    - Memory and CPU utilization
    - Data integrity and consistency
    - Custom application-specific metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, List[HealthResult]] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Default thresholds
        self.default_thresholds = {
            'response_time_warning': 1.0,  # seconds
            'response_time_critical': 5.0,  # seconds
            'cpu_warning': 70,  # percentage
            'cpu_critical': 85,  # percentage
            'memory_warning': 75,  # percentage
            'memory_critical': 90,  # percentage
            'storage_warning': 80,  # percentage
            'storage_critical': 95,  # percentage
        }
        self.default_thresholds.update(self.config.get('thresholds', {}))
        
        # Setup default checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks for ANANT services."""
        
        # System resource checks
        self.add_check(HealthCheck(
            name="cpu_usage",
            check_type=CheckType.CPU,
            interval_seconds=30,
            timeout_seconds=5,
            warning_threshold=self.default_thresholds['cpu_warning'],
            critical_threshold=self.default_thresholds['cpu_critical']
        ))
        
        self.add_check(HealthCheck(
            name="memory_usage",
            check_type=CheckType.MEMORY,
            interval_seconds=30,
            timeout_seconds=5,
            warning_threshold=self.default_thresholds['memory_warning'],
            critical_threshold=self.default_thresholds['memory_critical']
        ))
        
        self.add_check(HealthCheck(
            name="storage_usage",
            check_type=CheckType.STORAGE,
            interval_seconds=60,
            timeout_seconds=10,
            warning_threshold=self.default_thresholds['storage_warning'],
            critical_threshold=self.default_thresholds['storage_critical']
        ))
        
        # Polars/Parquet specific checks
        self.add_check(HealthCheck(
            name="polars_operations",
            check_type=CheckType.CUSTOM,
            interval_seconds=60,
            timeout_seconds=30,
            warning_threshold=2.0,  # seconds for basic operation
            critical_threshold=10.0,  # seconds for basic operation
            custom_checker=self._check_polars_health
        ))
        
        self.add_check(HealthCheck(
            name="parquet_io",
            check_type=CheckType.CUSTOM,
            interval_seconds=120,
            timeout_seconds=60,
            warning_threshold=5.0,  # seconds for I/O operation
            critical_threshold=30.0,  # seconds for I/O operation
            custom_checker=self._check_parquet_health
        ))
    
    def add_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.checks[health_check.name] = health_check
        self.results[health_check.name] = []
        print(f"✅ Added health check: {health_check.name}")
    
    def remove_check(self, check_name: str):
        """Remove a health check."""
        if check_name in self.checks:
            del self.checks[check_name]
            del self.results[check_name]
            print(f"🗑️ Removed health check: {check_name}")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        
        # Start monitoring tasks for each check
        for check_name, check in self.checks.items():
            task = asyncio.create_task(self._monitoring_loop(check))
            self.monitoring_tasks.append(task)
        
        print(f"🔍 Health monitoring started for {len(self.checks)} checks")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        self.monitoring_tasks.clear()
        print("🔍 Health monitoring stopped")
    
    async def _monitoring_loop(self, check: HealthCheck):
        """Monitoring loop for a specific health check."""
        while self.is_monitoring:
            try:
                result = await self._perform_check(check)
                self._record_result(result)
                await asyncio.sleep(check.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Record error result
                error_result = HealthResult(
                    check_name=check.name,
                    status=HealthStatus.CRITICAL,
                    value=None,
                    message=f"Health check failed: {e}",
                    timestamp=datetime.now(),
                    response_time=0.0,
                    details={"error": str(e)}
                )
                self._record_result(error_result)
                await asyncio.sleep(check.interval_seconds)
    
    async def _perform_check(self, check: HealthCheck) -> HealthResult:
        """Perform a single health check."""
        start_time = time.time()
        
        try:
            if check.check_type == CheckType.HTTP:
                result = await self._check_http(check)
            elif check.check_type == CheckType.CPU:
                result = await self._check_cpu(check)
            elif check.check_type == CheckType.MEMORY:
                result = await self._check_memory(check)
            elif check.check_type == CheckType.STORAGE:
                result = await self._check_storage(check)
            elif check.check_type == CheckType.CUSTOM and check.custom_checker:
                result = await self._check_custom(check)
            else:
                result = HealthResult(
                    check_name=check.name,
                    status=HealthStatus.UNKNOWN,
                    value=None,
                    message="Unknown check type",
                    timestamp=datetime.now(),
                    response_time=0.0,
                    details={}
                )
            
            result.response_time = time.time() - start_time
            return result
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                value=None,
                message=f"Check execution failed: {e}",
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )
    
    async def _check_http(self, check: HealthCheck) -> HealthResult:
        """Perform HTTP health check."""
        if not check.endpoint:
            raise ValueError("HTTP check requires endpoint")
        
        try:
            response = requests.get(check.endpoint, timeout=check.timeout_seconds)
            response_time = response.elapsed.total_seconds()
            
            # Determine status based on response time and HTTP status
            if response.status_code == 200:
                if response_time >= check.critical_threshold:
                    status = HealthStatus.CRITICAL
                elif response_time >= check.warning_threshold:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.CRITICAL
            
            return HealthResult(
                check_name=check.name,
                status=status,
                value=response_time,
                message=f"HTTP {response.status_code}, response time: {response_time:.2f}s",
                timestamp=datetime.now(),
                response_time=response_time,
                details={
                    "status_code": response.status_code,
                    "endpoint": check.endpoint
                }
            )
            
        except requests.RequestException as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                value=None,
                message=f"HTTP request failed: {e}",
                timestamp=datetime.now(),
                response_time=0.0,
                details={"error": str(e), "endpoint": check.endpoint}
            )
    
    async def _check_cpu(self, check: HealthCheck) -> HealthResult:
        """Check CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent >= check.critical_threshold:
                status = HealthStatus.CRITICAL
            elif cpu_percent >= check.warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthResult(
                check_name=check.name,
                status=status,
                value=cpu_percent,
                message=f"CPU usage: {cpu_percent:.1f}%",
                timestamp=datetime.now(),
                response_time=0.0,
                details={"cpu_count": psutil.cpu_count()}
            )
            
        except ImportError:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.UNKNOWN,
                value=None,
                message="psutil not available for CPU monitoring",
                timestamp=datetime.now(),
                response_time=0.0,
                details={}
            )
    
    async def _check_memory(self, check: HealthCheck) -> HealthResult:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent >= check.critical_threshold:
                status = HealthStatus.CRITICAL
            elif memory_percent >= check.warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthResult(
                check_name=check.name,
                status=status,
                value=memory_percent,
                message=f"Memory usage: {memory_percent:.1f}% ({memory.used // 1024 // 1024}MB used)",
                timestamp=datetime.now(),
                response_time=0.0,
                details={
                    "total_mb": memory.total // 1024 // 1024,
                    "used_mb": memory.used // 1024 // 1024,
                    "available_mb": memory.available // 1024 // 1024
                }
            )
            
        except ImportError:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.UNKNOWN,
                value=None,
                message="psutil not available for memory monitoring",
                timestamp=datetime.now(),
                response_time=0.0,
                details={}
            )
    
    async def _check_storage(self, check: HealthCheck) -> HealthResult:
        """Check storage usage."""
        try:
            import psutil
            import os
            
            # Check storage for ANANT data directory
            storage_path = self.config.get('storage_path', '/data/anant')
            if not os.path.exists(storage_path):
                storage_path = '/'  # Fallback to root
            
            disk = psutil.disk_usage(storage_path)
            storage_percent = disk.used / disk.total * 100
            
            if storage_percent >= check.critical_threshold:
                status = HealthStatus.CRITICAL
            elif storage_percent >= check.warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthResult(
                check_name=check.name,
                status=status,
                value=storage_percent,
                message=f"Storage usage: {storage_percent:.1f}% ({disk.used // 1024 // 1024 // 1024}GB used)",
                timestamp=datetime.now(),
                response_time=0.0,
                details={
                    "path": storage_path,
                    "total_gb": disk.total // 1024 // 1024 // 1024,
                    "used_gb": disk.used // 1024 // 1024 // 1024,
                    "free_gb": disk.free // 1024 // 1024 // 1024
                }
            )
            
        except ImportError:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.UNKNOWN,
                value=None,
                message="psutil not available for storage monitoring",
                timestamp=datetime.now(),
                response_time=0.0,
                details={}
            )
    
    async def _check_custom(self, check: HealthCheck) -> HealthResult:
        """Perform custom health check."""
        if not check.custom_checker:
            raise ValueError("Custom check requires custom_checker function")
        
        try:
            # Execute custom checker
            if asyncio.iscoroutinefunction(check.custom_checker):
                result = await check.custom_checker()
            else:
                result = check.custom_checker()
            
            # Parse result
            if isinstance(result, dict):
                value = result.get('value', 0.0)
                message = result.get('message', 'Custom check completed')
                details = result.get('details', {})
            else:
                value = float(result) if result is not None else 0.0
                message = f"Custom check value: {value}"
                details = {}
            
            # Determine status
            if value >= check.critical_threshold:
                status = HealthStatus.CRITICAL
            elif value >= check.warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthResult(
                check_name=check.name,
                status=status,
                value=value,
                message=message,
                timestamp=datetime.now(),
                response_time=0.0,
                details=details
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                value=None,
                message=f"Custom check failed: {e}",
                timestamp=datetime.now(),
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def _check_polars_health(self) -> Dict[str, Any]:
        """Check Polars operations health."""
        try:
            import polars as pl
            import time
            
            start_time = time.time()
            
            # Perform basic Polars operations to test health
            df = pl.DataFrame({
                "test_column": range(1000),
                "values": [i * 2 for i in range(1000)]
            })
            
            # Basic operations
            result = df.filter(pl.col("test_column") > 500).select(pl.sum("values"))
            
            execution_time = time.time() - start_time
            
            return {
                "value": execution_time,
                "message": f"Polars operations completed in {execution_time:.3f}s",
                "details": {
                    "rows_processed": 1000,
                    "operations": "filter, select, sum",
                    "result_value": result.item() if result.height > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                "value": 999.0,  # High value to trigger critical status
                "message": f"Polars health check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _check_parquet_health(self) -> Dict[str, Any]:
        """Check Parquet I/O health."""
        try:
            import polars as pl
            import tempfile
            import os
            import time
            
            start_time = time.time()
            
            # Create test data
            df = pl.DataFrame({
                "id": range(1000),
                "data": [f"test_data_{i}" for i in range(1000)],
                "timestamp": [datetime.now()] * 1000
            })
            
            # Test Parquet write/read cycle
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Write to Parquet
                df.write_parquet(temp_path, compression="zstd")
                
                # Read from Parquet
                read_df = pl.read_parquet(temp_path)
                
                # Verify data integrity
                if read_df.height != df.height:
                    raise ValueError("Data integrity check failed: row count mismatch")
                
                execution_time = time.time() - start_time
                
                return {
                    "value": execution_time,
                    "message": f"Parquet I/O completed in {execution_time:.3f}s",
                    "details": {
                        "rows_written": df.height,
                        "rows_read": read_df.height,
                        "file_size_kb": os.path.getsize(temp_path) // 1024,
                        "compression": "zstd"
                    }
                }
                
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            return {
                "value": 999.0,  # High value to trigger critical status
                "message": f"Parquet health check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _record_result(self, result: HealthResult):
        """Record a health check result."""
        check_name = result.check_name
        
        # Add result to history
        if check_name not in self.results:
            self.results[check_name] = []
        
        self.results[check_name].append(result)
        
        # Keep only recent results (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.results[check_name] = [
            r for r in self.results[check_name] 
            if r.timestamp > cutoff_time
        ]
        
        # Update service health
        self._update_service_health()
    
    def _update_service_health(self):
        """Update overall service health based on recent check results."""
        for service_name in ["anant-core", "anant-monitoring", "anant-storage"]:
            
            # Get recent results for this service
            recent_results = []
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for check_name, results in self.results.items():
                service_results = [r for r in results if r.timestamp > cutoff_time]
                recent_results.extend(service_results)
            
            if not recent_results:
                continue
            
            # Calculate overall status
            critical_count = sum(1 for r in recent_results if r.status == HealthStatus.CRITICAL)
            warning_count = sum(1 for r in recent_results if r.status == HealthStatus.WARNING)
            total_count = len(recent_results)
            
            if critical_count > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_count > total_count * 0.3:  # More than 30% warnings
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Calculate uptime percentage
            healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
            uptime_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
            
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                overall_status=overall_status,
                checks=recent_results[-10:],  # Last 10 results
                last_updated=datetime.now(),
                uptime_percentage=uptime_percentage
            )
    
    def get_health_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current health status."""
        if service_name and service_name in self.service_health:
            service = self.service_health[service_name]
            return {
                "service": service_name,
                "status": service.overall_status.value,
                "uptime_percentage": service.uptime_percentage,
                "last_updated": service.last_updated.isoformat(),
                "recent_checks": [
                    {
                        "name": check.check_name,
                        "status": check.status.value,
                        "value": check.value,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat()
                    }
                    for check in service.checks[-5:]  # Last 5 checks
                ]
            }
        else:
            # Return overall health status
            all_services = list(self.service_health.values())
            if not all_services:
                return {"overall_status": "unknown", "services": []}
            
            # Calculate overall status
            critical_services = [s for s in all_services if s.overall_status == HealthStatus.CRITICAL]
            warning_services = [s for s in all_services if s.overall_status == HealthStatus.WARNING]
            
            if critical_services:
                overall_status = "critical"
            elif warning_services:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            return {
                "overall_status": overall_status,
                "total_services": len(all_services),
                "healthy_services": len([s for s in all_services if s.overall_status == HealthStatus.HEALTHY]),
                "warning_services": len(warning_services),
                "critical_services": len(critical_services),
                "services": [
                    {
                        "name": service.service_name,
                        "status": service.overall_status.value,
                        "uptime_percentage": service.uptime_percentage,
                        "last_updated": service.last_updated.isoformat()
                    }
                    for service in all_services
                ]
            }
    
    def check_cluster_health(self, cluster_id: str) -> Dict[str, Any]:
        """Check health of an entire cluster."""
        cluster_health = self.get_health_status()
        
        # Add cluster-specific information
        cluster_health.update({
            "cluster_id": cluster_id,
            "timestamp": datetime.now().isoformat(),
            "monitoring_enabled": self.is_monitoring,
            "total_checks": len(self.checks),
            "check_types": list(set(check.check_type.value for check in self.checks.values()))
        })
        
        return cluster_health