#!/usr/bin/env python3
"""
FastAPI + Ray Application Test Script
====================================

Test script to verify the FastAPI application is working correctly
with the Ray cluster.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


def test_endpoint(url: str, method: str = "GET", data: Dict[Any, Any] = None, timeout: int = 30) -> Dict[str, Any]:
    """Test an endpoint and return results."""
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "response": response.json() if response.content else {},
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e),
            "response_time": None
        }


def main():
    """Run FastAPI application tests."""
    base_url = "http://localhost:8080"
    
    print("🚀 Testing FastAPI + Ray Application")
    print("=" * 50)
    
    # Test basic endpoints
    tests = [
        {
            "name": "Root Endpoint",
            "url": f"{base_url}/",
            "method": "GET"
        },
        {
            "name": "Health Check",
            "url": f"{base_url}/health",
            "method": "GET"
        },
        {
            "name": "Configuration",
            "url": f"{base_url}/config",
            "method": "GET"
        },
        {
            "name": "Cluster Status",
            "url": f"{base_url}/cluster/status",
            "method": "GET"
        },
        {
            "name": "Analytics Overview",
            "url": f"{base_url}/analytics/",
            "method": "GET"
        },
        {
            "name": "ML Service Overview",
            "url": f"{base_url}/ml/",
            "method": "GET"
        },
        {
            "name": "Data Processing Overview",
            "url": f"{base_url}/data/",
            "method": "GET"
        },
        {
            "name": "Monitoring Overview",
            "url": f"{base_url}/monitoring/",
            "method": "GET"
        }
    ]
    
    # Advanced tests with data
    advanced_tests = [
        {
            "name": "Analytics Metrics",
            "url": f"{base_url}/analytics/metrics",
            "method": "POST",
            "data": {
                "data_size": 1000,
                "metrics": ["mean", "std"],
                "workers": 2
            }
        },
        {
            "name": "ML Training",
            "url": f"{base_url}/ml/train",
            "method": "POST",
            "data": {
                "model_type": "linear_regression",
                "data_size": 500,
                "features": 3
            }
        },
        {
            "name": "Data Processing",
            "url": f"{base_url}/data/process-json",
            "method": "POST",
            "data": {
                "data": [
                    {"id": 1, "name": "test1", "value": 100},
                    {"id": 2, "name": "test2", "value": 200}
                ],
                "operations": ["validate", "clean"]
            }
        },
        {
            "name": "System Monitoring",
            "url": f"{base_url}/monitoring/metrics",
            "method": "POST",
            "data": {
                "metric_types": ["system", "ray"]
            }
        }
    ]
    
    # Run basic tests
    print("\n📋 Basic Endpoint Tests:")
    basic_results = []
    for test in tests:
        print(f"  Testing {test['name']}...", end=" ")
        result = test_endpoint(test["url"], test["method"])
        basic_results.append({**test, **result})
        
        if result["status"] == "success" and result["status_code"] == 200:
            print(f"✅ OK ({result['response_time']:.2f}s)")
        else:
            print(f"❌ FAILED ({result.get('status_code', 'N/A')})")
    
    # Run advanced tests
    print("\n🧪 Advanced Feature Tests:")
    advanced_results = []
    for test in advanced_tests:
        print(f"  Testing {test['name']}...", end=" ")
        result = test_endpoint(test["url"], test["method"], test.get("data"))
        advanced_results.append({**test, **result})
        
        if result["status"] == "success" and result["status_code"] == 200:
            print(f"✅ OK ({result['response_time']:.2f}s)")
        else:
            print(f"❌ FAILED ({result.get('status_code', 'N/A')})")
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    basic_passed = sum(1 for r in basic_results if r["status"] == "success" and r["status_code"] == 200)
    advanced_passed = sum(1 for r in advanced_results if r["status"] == "success" and r["status_code"] == 200)
    
    print(f"Basic Tests:    {basic_passed}/{len(basic_results)} passed")
    print(f"Advanced Tests: {advanced_passed}/{len(advanced_tests)} passed")
    print(f"Total:          {basic_passed + advanced_passed}/{len(basic_results) + len(advanced_tests)} passed")
    
    # Show failures
    all_results = basic_results + advanced_results
    failures = [r for r in all_results if r["status"] != "success" or r["status_code"] != 200]
    
    if failures:
        print(f"\n❌ Failed Tests ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure['name']}: {failure.get('error', f'HTTP {failure.get(\"status_code\", \"N/A\")}'}")
    else:
        print("\n🎉 All tests passed!")
    
    # Show sample responses for successful tests
    print("\n📋 Sample Responses:")
    print("-" * 30)
    
    # Show root endpoint response
    root_test = next((r for r in basic_results if r["name"] == "Root Endpoint" and r["status"] == "success"), None)
    if root_test:
        print("Root Endpoint Response:")
        print(json.dumps(root_test["response"], indent=2)[:500] + "..." if len(str(root_test["response"])) > 500 else json.dumps(root_test["response"], indent=2))
    
    # Show cluster status if available
    cluster_test = next((r for r in basic_results if r["name"] == "Cluster Status" and r["status"] == "success"), None)
    if cluster_test:
        print("\nCluster Status:")
        cluster_overview = cluster_test["response"].get("cluster_overview", {})
        print(f"  Nodes: {cluster_overview.get('total_nodes', 'N/A')}")
        print(f"  CPU Utilization: {cluster_overview.get('cpu_utilization_percent', 'N/A')}%")
        print(f"  Memory Utilization: {cluster_overview.get('memory_utilization_percent', 'N/A')}%")
    
    return len(failures) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)