"""
Analytics Sub-Application
========================

Handles analytics, reporting, and data visualization using Ray for
distributed computation.
"""

import ray
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from datetime import datetime, timedelta


analytics_app = FastAPI(
    title="Analytics Service",
    description="Analytics and reporting with Ray scaling"
)


class AnalyticsRequest(BaseModel):
    data_source: str
    metrics: List[str]
    date_range: Dict[str, str]
    filters: Optional[Dict[str, Any]] = {}
    aggregation: str = "sum"


class AnalyticsResponse(BaseModel):
    request_id: str
    results: Dict[str, Any]
    computation_time: float
    nodes_used: int
    timestamp: str


@ray.remote
def compute_metric(data: List[float], metric_type: str) -> Dict[str, float]:
    """Compute a specific metric on distributed data."""
    import time
    start_time = time.time()
    
    data_array = np.array(data)
    
    if metric_type == "sum":
        result = float(np.sum(data_array))
    elif metric_type == "mean":
        result = float(np.mean(data_array))
    elif metric_type == "std":
        result = float(np.std(data_array))
    elif metric_type == "max":
        result = float(np.max(data_array))
    elif metric_type == "min":
        result = float(np.min(data_array))
    elif metric_type == "percentile_95":
        result = float(np.percentile(data_array, 95))
    else:
        result = 0.0
    
    computation_time = time.time() - start_time
    
    return {
        "metric": metric_type,
        "value": result,
        "data_points": len(data),
        "computation_time": computation_time
    }


@ray.remote
def generate_sample_data(size: int, data_source: str) -> List[float]:
    """Generate sample data for analytics."""
    np.random.seed(hash(data_source) % 1000)
    
    if data_source == "sales":
        # Sales data (positive, with trend)
        base = np.random.exponential(100, size)
        trend = np.linspace(0, 50, size)
        return (base + trend).tolist()
    elif data_source == "website_traffic":
        # Website traffic (periodic patterns)
        t = np.linspace(0, 4*np.pi, size)
        base = 1000 + 500 * np.sin(t) + np.random.normal(0, 100, size)
        return np.maximum(0, base).tolist()
    elif data_source == "user_engagement":
        # User engagement (beta distribution)
        return (np.random.beta(2, 5, size) * 100).tolist()
    else:
        # Default random data
        return np.random.normal(50, 15, size).tolist()


@analytics_app.get("/")
async def analytics_root():
    """Analytics service root."""
    return {
        "service": "Analytics",
        "description": "Distributed analytics and reporting",
        "endpoints": [
            "/compute - Run analytics computation",
            "/reports - Generate reports", 
            "/realtime - Real-time analytics"
        ]
    }


@analytics_app.post("/compute", response_model=AnalyticsResponse)
async def compute_analytics(request: AnalyticsRequest):
    """Compute analytics metrics using Ray distributed processing."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    request_id = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.now()
    
    try:
        # Generate sample data for the data source
        data_size = 10000  # Large dataset for meaningful distribution
        data_ref = generate_sample_data.remote(data_size, request.data_source)
        
        # Get the data
        data = await asyncio.create_task(
            asyncio.to_thread(ray.get, data_ref)
        )
        
        # Distribute metric computations across Ray workers
        metric_tasks = []
        for metric in request.metrics:
            task = compute_metric.remote(data, metric)
            metric_tasks.append(task)
        
        # Wait for all computations to complete
        metric_results = await asyncio.create_task(
            asyncio.to_thread(ray.get, metric_tasks)
        )
        
        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()
        
        # Aggregate results
        results = {
            "data_source": request.data_source,
            "total_data_points": data_size,
            "metrics": {result["metric"]: result for result in metric_results},
            "filters_applied": request.filters,
            "aggregation_method": request.aggregation
        }
        
        return AnalyticsResponse(
            request_id=request_id,
            results=results,
            computation_time=computation_time,
            nodes_used=len(ray.nodes()) if ray.is_initialized() else 1,
            timestamp=end_time.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics computation failed: {str(e)}")


@analytics_app.get("/reports/{report_type}")
async def generate_report(report_type: str, days: int = 7):
    """Generate predefined reports."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    # Define report templates
    report_configs = {
        "sales_summary": {
            "data_source": "sales",
            "metrics": ["sum", "mean", "max", "percentile_95"],
            "description": "Sales performance summary"
        },
        "traffic_analysis": {
            "data_source": "website_traffic", 
            "metrics": ["mean", "max", "min", "std"],
            "description": "Website traffic analysis"
        },
        "engagement_metrics": {
            "data_source": "user_engagement",
            "metrics": ["mean", "percentile_95", "std"],
            "description": "User engagement metrics"
        }
    }
    
    if report_type not in report_configs:
        raise HTTPException(status_code=404, detail="Report type not found")
    
    config = report_configs[report_type]
    
    # Create analytics request
    request = AnalyticsRequest(
        data_source=config["data_source"],
        metrics=config["metrics"],
        date_range={
            "start": (datetime.now() - timedelta(days=days)).isoformat(),
            "end": datetime.now().isoformat()
        }
    )
    
    # Compute analytics
    result = await compute_analytics(request)
    
    return {
        "report_type": report_type,
        "description": config["description"],
        "period_days": days,
        "analytics_result": result
    }


@analytics_app.get("/realtime")
async def realtime_analytics():
    """Simulate real-time analytics."""
    if not ray.is_initialized():
        raise HTTPException(status_code=503, detail="Ray cluster not available")
    
    # Simulate real-time data streams
    streams = ["orders", "page_views", "api_calls", "errors"]
    
    @ray.remote
    def process_stream(stream_name: str) -> Dict[str, Any]:
        """Process a real-time data stream."""
        # Simulate real-time metrics
        current_value = np.random.poisson(100 if stream_name != "errors" else 5)
        trend = np.random.choice(["up", "down", "stable"], p=[0.4, 0.3, 0.3])
        
        return {
            "stream": stream_name,
            "current_value": int(current_value),
            "trend": trend,
            "timestamp": datetime.now().isoformat(),
            "status": "healthy" if stream_name != "errors" or current_value < 10 else "warning"
        }
    
    # Process all streams in parallel
    stream_tasks = [process_stream.remote(stream) for stream in streams]
    stream_results = await asyncio.create_task(
        asyncio.to_thread(ray.get, stream_tasks)
    )
    
    return {
        "realtime_metrics": stream_results,
        "cluster_status": {
            "nodes": len(ray.nodes()) if ray.is_initialized() else 1,
            "available_resources": ray.available_resources() if ray.is_initialized() else {}
        },
        "timestamp": datetime.now().isoformat()
    }