"""
Monitoring Sub-Application
=========================

System monitoring, performance metrics, and cluster health monitoring
using Ray for distributed monitoring tasks.
"""

import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import psutil
import time
from datetime import datetime, timedelta
import json


monitoring_app = FastAPI(
    title="Monitoring Service",
    description="System and cluster monitoring with Ray"
)


class MetricsRequest(BaseModel):
    metric_types: List[str]
    duration_minutes: Optional[int] = 5
    interval_seconds: Optional[int] = 10


class AlertRequest(BaseModel):
    metric: str
    threshold: float
    condition: str  # "greater_than", "less_than"
    duration_minutes: int = 5


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    collection_time: str
    cluster_info: Dict[str, Any]


# In-memory storage for monitoring data
metrics_history = []
alerts = []
monitoring_jobs = {}


@ray.remote
class SystemMonitor:
    """Distributed system monitor using Ray."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics from the current node."""
        import psutil
        import platform
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        return {
            "timestamp": datetime.now().isoformat(),
            "node_id": ray.get_runtime_context().get_node_id() if ray.is_initialized() else "local",
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node()
            },
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            },
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "percent": swap.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round((disk.used / disk.total) * 100, 2)
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "processes": {
                "count": process_count
            }
        }
    
    def collect_ray_metrics(self) -> Dict[str, Any]:
        """Collect Ray-specific metrics."""
        if not ray.is_initialized():
            return {"error": "Ray not initialized"}
        
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            nodes = ray.nodes()
            
            # Calculate resource utilization
            cpu_total = cluster_resources.get("CPU", 0)
            cpu_available = available_resources.get("CPU", 0)
            cpu_used = cpu_total - cpu_available
            cpu_utilization = (cpu_used / cpu_total * 100) if cpu_total > 0 else 0
            
            memory_total = cluster_resources.get("memory", 0)
            memory_available = available_resources.get("memory", 0)
            memory_used = memory_total - memory_available
            memory_utilization = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            # Node health
            alive_nodes = sum(1 for node in nodes if node["Alive"])
            dead_nodes = sum(1 for node in nodes if not node["Alive"])
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "resource_utilization": {
                    "cpu_percent": round(cpu_utilization, 2),
                    "memory_percent": round(memory_utilization, 2),
                    "cpu_total": cpu_total,
                    "cpu_used": cpu_used,
                    "memory_total_gb": round(memory_total / (1024**3), 2),
                    "memory_used_gb": round(memory_used / (1024**3), 2)
                },
                "nodes": {
                    "total": len(nodes),
                    "alive": alive_nodes,
                    "dead": dead_nodes,
                    "details": nodes
                }
            }
        except Exception as e:
            return {"error": f"Failed to collect Ray metrics: {str(e)}"}
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        uptime = time.time() - self.start_time
        
        # Mock application metrics (in real app, collect from your services)
        import random
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "application": {
                "requests_per_second": random.randint(10, 100),
                "error_rate_percent": round(random.uniform(0, 5), 2),
                "response_time_ms": round(random.uniform(50, 500), 2),
                "active_connections": random.randint(5, 50),
                "cache_hit_rate_percent": round(random.uniform(80, 95), 2)
            },
            "custom_metrics": {
                "business_metric_1": random.uniform(1000, 5000),
                "business_metric_2": random.randint(0, 1000),
                "feature_usage_count": random.randint(100, 1000)
            }
        }


@ray.remote
class MetricsCollector:
    """Centralized metrics collector."""
    
    def __init__(self):
        self.monitors = []
        self.collected_metrics = []
    
    def initialize_monitors(self, node_count: int):
        """Initialize monitors on available nodes."""
        self.monitors = [SystemMonitor.remote() for _ in range(node_count)]
        return len(self.monitors)
    
    def collect_all_metrics(self, metric_types: List[str]) -> Dict[str, Any]:
        """Collect metrics from all monitors."""
        if not self.monitors:
            # Initialize with single monitor if none exist
            self.monitors = [SystemMonitor.remote()]
        
        collection_start = time.time()
        all_metrics = {"collection_metadata": {"start_time": datetime.now().isoformat()}}
        
        # Collect from all monitors in parallel
        if "system" in metric_types:
            system_tasks = [monitor.collect_system_metrics.remote() for monitor in self.monitors]
            system_metrics = ray.get(system_tasks)
            all_metrics["system_metrics"] = system_metrics
        
        if "ray" in metric_types:
            # Ray metrics are cluster-wide, so collect from one monitor
            ray_task = self.monitors[0].collect_ray_metrics.remote()
            ray_metrics = ray.get(ray_task)
            all_metrics["ray_metrics"] = ray_metrics
        
        if "application" in metric_types:
            app_tasks = [monitor.collect_application_metrics.remote() for monitor in self.monitors]
            app_metrics = ray.get(app_tasks)
            all_metrics["application_metrics"] = app_metrics
        
        collection_time = time.time() - collection_start
        all_metrics["collection_metadata"]["duration_seconds"] = round(collection_time, 3)
        all_metrics["collection_metadata"]["end_time"] = datetime.now().isoformat()
        
        return all_metrics


# Global metrics collector
metrics_collector = None


@monitoring_app.get("/")
async def monitoring_root():
    """Monitoring service root."""
    return {
        "service": "Monitoring",
        "description": "System and cluster monitoring with Ray",
        "endpoints": [
            "/metrics - Collect system metrics",
            "/health - Health check",
            "/alerts - Alert management",
            "/dashboard - Monitoring dashboard data"
        ]
    }


@monitoring_app.post("/metrics", response_model=MetricsResponse)
async def collect_metrics(request: MetricsRequest):
    """Collect system and cluster metrics."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    global metrics_collector
    
    try:
        # Initialize collector if needed
        if metrics_collector is None:
            metrics_collector = MetricsCollector.remote()
            node_count = len(ray.nodes()) if ray.is_initialized() else 1
            await asyncio.create_task(
                asyncio.to_thread(
                    ray.get, metrics_collector.initialize_monitors.remote(node_count)
                )
            )
        
        # Collect metrics
        metrics_ref = metrics_collector.collect_all_metrics.remote(request.metric_types)
        metrics_data = await asyncio.create_task(
            asyncio.to_thread(ray.get, metrics_ref)
        )
        
        # Store in history
        metrics_history.append(metrics_data)
        
        # Keep only recent metrics (last 100 collections)
        if len(metrics_history) > 100:
            metrics_history.pop(0)
        
        # Prepare cluster info
        cluster_info = {
            "nodes": len(ray.nodes()) if ray.is_initialized() else 1,
            "resources": ray.cluster_resources() if ray.is_initialized() else {},
            "ray_version": ray.__version__ if ray.is_initialized() else "N/A"
        }
        
        return MetricsResponse(
            metrics=metrics_data,
            collection_time=datetime.now().isoformat(),
            cluster_info=cluster_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@monitoring_app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {"status": "healthy", "checks": {}}
    
    # Ray cluster health
    if ray.is_initialized():
        try:
            nodes = ray.nodes()
            alive_nodes = sum(1 for node in nodes if node["Alive"])
            total_nodes = len(nodes)
            
            health_status["checks"]["ray_cluster"] = {
                "status": "healthy" if alive_nodes == total_nodes else "degraded",
                "alive_nodes": alive_nodes,
                "total_nodes": total_nodes
            }
        except Exception as e:
            health_status["checks"]["ray_cluster"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    else:
        health_status["checks"]["ray_cluster"] = {
            "status": "unavailable",
            "message": "Ray not initialized"
        }
    
    # System resource health
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_status = "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
        disk_status = "healthy" if disk.percent < 80 else "warning" if disk.percent < 90 else "critical"
        
        health_status["checks"]["system_resources"] = {
            "status": memory_status if memory_status == disk_status else "warning",
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        }
    except Exception as e:
        health_status["checks"]["system_resources"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    # Overall status
    check_statuses = [check["status"] for check in health_status["checks"].values()]
    if any(status == "critical" for status in check_statuses):
        health_status["status"] = "critical"
    elif any(status in ["unhealthy", "warning"] for status in check_statuses):
        health_status["status"] = "warning"
    
    return health_status


@monitoring_app.get("/metrics/history")
async def get_metrics_history(limit: int = 50):
    """Get metrics history."""
    return {
        "metrics_history": metrics_history[-limit:],
        "total_collections": len(metrics_history)
    }


@monitoring_app.post("/alerts")
async def create_alert(request: AlertRequest):
    """Create a monitoring alert."""
    alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    alert = {
        "id": alert_id,
        "metric": request.metric,
        "threshold": request.threshold,
        "condition": request.condition,
        "duration_minutes": request.duration_minutes,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "triggered_count": 0
    }
    
    alerts.append(alert)
    
    return {"alert_id": alert_id, "alert": alert}


@monitoring_app.get("/alerts")
async def list_alerts():
    """List all alerts."""
    return {"alerts": alerts}


@monitoring_app.get("/dashboard")
async def get_dashboard_data():
    """Get dashboard data for monitoring UI."""
    if not ray.is_initialized():
        return {"error": "Ray cluster not available"}
    
    try:
        # Current cluster status
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        nodes = ray.nodes()
        
        # Recent metrics summary
        recent_metrics = metrics_history[-10:] if metrics_history else []
        
        # Calculate trends
        cpu_trend = []
        memory_trend = []
        
        for metric in recent_metrics:
            ray_metrics = metric.get("ray_metrics", {})
            util = ray_metrics.get("resource_utilization", {})
            if util:
                cpu_trend.append(util.get("cpu_percent", 0))
                memory_trend.append(util.get("memory_percent", 0))
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "cluster_overview": {
                "total_nodes": len(nodes),
                "alive_nodes": sum(1 for node in nodes if node["Alive"]),
                "total_cpu": cluster_resources.get("CPU", 0),
                "available_cpu": available_resources.get("CPU", 0),
                "total_memory_gb": round(cluster_resources.get("memory", 0) / (1024**3), 2),
                "available_memory_gb": round(available_resources.get("memory", 0) / (1024**3), 2)
            },
            "resource_trends": {
                "cpu_utilization": cpu_trend,
                "memory_utilization": memory_trend
            },
            "recent_metrics_count": len(recent_metrics),
            "active_alerts": len([alert for alert in alerts if alert["status"] == "active"]),
            "nodes_detail": nodes
        }
        
        return dashboard_data
        
    except Exception as e:
        return {"error": f"Dashboard data collection failed: {str(e)}"}


@monitoring_app.get("/performance")
async def get_performance_metrics():
    """Get performance-focused metrics."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    # Collect quick performance snapshot
    monitor = SystemMonitor.remote()
    
    system_metrics_ref = monitor.collect_system_metrics.remote()
    ray_metrics_ref = monitor.collect_ray_metrics.remote()
    
    system_metrics, ray_metrics = await asyncio.create_task(
        asyncio.to_thread(ray.get, [system_metrics_ref, ray_metrics_ref])
    )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "performance_summary": {
            "cpu_utilization": system_metrics.get("cpu", {}).get("percent", 0),
            "memory_utilization": system_metrics.get("memory", {}).get("percent", 0),
            "ray_cpu_utilization": ray_metrics.get("resource_utilization", {}).get("cpu_percent", 0),
            "ray_memory_utilization": ray_metrics.get("resource_utilization", {}).get("memory_percent", 0),
            "active_nodes": ray_metrics.get("nodes", {}).get("alive", 0),
            "total_nodes": ray_metrics.get("nodes", {}).get("total", 0)
        },
        "detailed_metrics": {
            "system": system_metrics,
            "ray": ray_metrics
        }
    }