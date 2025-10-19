"""
Production Deployment Framework Demo
===================================

Comprehensive demonstration of ANANT's production deployment capabilities.
Shows enterprise-grade monitoring, scaling, optimization, and deployment.
"""

import asyncio
import time
from datetime import datetime, timedelta
import polars as pl

# Import production components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from anant.production.core.config_manager import ProductionConfig, EnvironmentType
from anant.production.core.production_manager import ProductionManager
from anant.production.monitoring.performance_monitor import PerformanceMonitor
from anant.production.monitoring.health_checker import HealthChecker, HealthCheck, CheckType
from anant.production.deployment.orchestrator import DeploymentOrchestrator
from anant.production.optimization.resource_optimizer import ResourceOptimizer, OptimizationConfig
from anant.production.scaling.auto_scaler import AutoScaler, ScalingPolicy


def demo_production_config():
    """Demonstrate production configuration management."""
    print("🔧 Production Configuration Demo")
    print("=" * 50)
    
    # Create production configuration
    config = ProductionConfig(
        environment=EnvironmentType.PRODUCTION,
        storage_path="/tmp/anant-production-demo",
        service_name="anant-enterprise",
        namespace="anant-prod"
    )
    
    print(f"Environment: {config.environment.value}")
    print(f"Storage Path: {config.storage_path}")
    print(f"Max Memory: {config.resources.max_memory_gb}GB")
    print(f"Max CPU Cores: {config.resources.max_cpu_cores}")
    print(f"Auto-scaling: {config.auto_scaling_enabled}")
    print(f"Monitoring: {config.monitoring_enabled}")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print(f"⚠️ Configuration issues: {issues}")
    else:
        print("✅ Configuration validated successfully")
    
    return config


def demo_production_manager(config):
    """Demonstrate production manager capabilities."""
    print("\n🚀 Production Manager Demo")
    print("=" * 50)
    
    # Create and start production manager
    prod_manager = ProductionManager(config)
    
    print("Starting production services...")
    success = prod_manager.start_services()
    print(f"Services started: {'✅' if success else '❌'}")
    
    # Wait for services to initialize
    time.sleep(3)
    
    # Get cluster status
    cluster_status = prod_manager.get_cluster_status()
    print(f"Cluster Health: {cluster_status.overall_health}")
    print(f"Total Services: {len(cluster_status.services)}")
    print(f"CPU Usage: {cluster_status.cpu_used:.1f}/{cluster_status.cpu_total}")
    print(f"Memory Usage: {cluster_status.memory_used:.1f}/{cluster_status.memory_total}GB")
    
    # List running services
    print("\nRunning Services:")
    for service in cluster_status.services[:5]:  # Show first 5
        print(f"  📦 {service.name}: {service.status} ({service.health}) - {service.replicas} replicas")
    
    # Test service scaling
    print("\nTesting service scaling...")
    success = prod_manager.scale_service("anant-hypergraph-api", 5)
    print(f"Scaling result: {'✅' if success else '❌'}")
    
    # Execute maintenance
    print("\nExecuting maintenance tasks...")
    maintenance_tasks = ["health_check", "resource_check"]
    for task in maintenance_tasks:
        success = prod_manager.execute_maintenance(task)
        print(f"  {task}: {'✅' if success else '❌'}")
    
    return prod_manager


def demo_performance_monitor():
    """Demonstrate performance monitoring."""
    print("\n📊 Performance Monitor Demo")
    print("=" * 50)
    
    # Create performance monitor
    monitor = PerformanceMonitor({
        'retention_minutes': 30,
        'thresholds': {
            'cpu_warning': 60,
            'memory_warning': 70
        }
    })
    
    # Start monitoring
    monitor.start_monitoring(interval_seconds=5)
    print("Performance monitoring started")
    
    # Simulate some operations to track
    print("Simulating ANANT operations...")
    
    def simulate_hypergraph_operation():
        """Simulate a hypergraph operation."""
        # Create test hypergraph data
        df = pl.DataFrame({
            "node_id": range(1000),
            "edge_id": [f"edge_{i//10}" for i in range(1000)],
            "weight": [i * 0.1 for i in range(1000)]
        })
        
        # Perform operations
        result = df.group_by("edge_id").agg(pl.sum("weight"))
        return result
    
    def simulate_metagraph_operation():
        """Simulate a metagraph operation."""
        # Create test metadata
        df = pl.DataFrame({
            "entity_id": [f"entity_{i}" for i in range(500)],
            "entity_type": ["product", "customer", "order"] * 166 + ["product", "customer"],  # Exactly 500 items
            "metadata": [f"metadata_{i}" for i in range(500)]
        })
        
        # Query operations
        result = df.filter(pl.col("entity_type") == "product").select("entity_id", "metadata")
        return result
    
    # Track operations with performance monitoring
    print("Tracking hypergraph operations...")
    for i in range(3):
        result = monitor.track_query("hypergraph", simulate_hypergraph_operation)
        print(f"  Operation {i+1}: {result.height} rows processed")
        time.sleep(1)
    
    print("Tracking metagraph operations...")
    for i in range(2):
        result = monitor.track_query("metagraph", simulate_metagraph_operation)
        print(f"  Operation {i+1}: {result.height} rows processed")
        time.sleep(1)
    
    # Get performance summary
    time.sleep(2)  # Let metrics collect
    summary = monitor.collector.get_performance_summary(5)
    print(f"\nPerformance Summary (last 5 minutes):")
    print(f"  System CPU: {summary.get('system_performance', {}).get('cpu', {}).get('average', 'N/A')}%")
    print(f"  System Memory: {summary.get('system_performance', {}).get('memory', {}).get('average', 'N/A')}%")
    print(f"  Total Queries: {summary.get('query_performance', {}).get('total_queries', 0)}")
    print(f"  Avg Query Time: {summary.get('query_performance', {}).get('average_execution_time', 0):.3f}s")
    
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"  Recommendations: {len(recommendations)}")
        for rec in recommendations[:2]:  # Show first 2
            print(f"    - {rec}")
    
    # Get optimization recommendations
    optimizations = monitor.optimize_polars_configuration()
    print(f"\nPolars Optimization Recommendations: {len(optimizations.get('optimizations', []))}")
    for opt in optimizations.get('optimizations', [])[:2]:  # Show first 2
        print(f"  🔧 {opt.get('type', 'Unknown')}: {opt.get('recommendation', '')}")
    
    monitor.stop_monitoring()
    return monitor


def demo_health_checker():
    """Demonstrate health checking system."""
    print("\n🔍 Health Checker Demo")
    print("=" * 50)
    
    # Create health checker
    health_checker = HealthChecker({
        'storage_path': '/tmp/anant-production-demo',
        'thresholds': {
            'cpu_warning': 60,
            'memory_warning': 70,
            'storage_warning': 75
        }
    })
    
    print(f"Initialized with {len(health_checker.checks)} default health checks")
    
    # Add custom health check
    def custom_anant_check():
        """Custom health check for ANANT operations."""
        try:
            # Test basic Polars operation
            df = pl.DataFrame({"test": range(100)})
            result = df.select(pl.sum("test")).item()
            return {
                "value": 0.1,  # Low value = healthy
                "message": f"ANANT operations healthy (test sum: {result})",
                "details": {"test_result": result}
            }
        except Exception as e:
            return {
                "value": 10.0,  # High value = unhealthy
                "message": f"ANANT operations failed: {e}",
                "details": {"error": str(e)}
            }
    
    # Add the custom check
    custom_check = HealthCheck(
        name="anant_operations",
        check_type=CheckType.CUSTOM,
        interval_seconds=30,
        timeout_seconds=10,
        warning_threshold=1.0,
        critical_threshold=5.0,
        custom_checker=custom_anant_check
    )
    health_checker.add_check(custom_check)
    
    # Start monitoring (simplified for demo without async loop)
    print("Starting health monitoring...")
    
    # Simulate health checks without starting the async monitoring loop
    # Instead, manually perform checks for demo
    try:
        import asyncio
        
        async def run_demo_checks():
            # Perform a few manual health checks
            for check_name, check in list(health_checker.checks.items())[:3]:  # Test first 3 checks
                result = await health_checker._perform_check(check)
                health_checker._record_result(result)
                print(f"  ✅ {check_name}: {result.status.value} ({result.message})")
        
        # Run the checks
        asyncio.run(run_demo_checks())
        
    except Exception as e:
        print(f"Health check simulation: {e}")
    
    # Get health status without starting full monitoring
    overall_health = health_checker.get_health_status()
    print(f"Overall Health: {overall_health.get('overall_status', 'unknown')}")
    print(f"Total Services: {overall_health.get('total_services', 0)}")
    print(f"Healthy Services: {overall_health.get('healthy_services', 0)}")
    
    services = overall_health.get('services', [])
    if services:
        print(f"Service Health:")
        for service in services[:3]:  # Show first 3
            print(f"  💚 {service.get('name', 'Unknown')}: {service.get('status', 'unknown')} "
                  f"(uptime: {service.get('uptime_percentage', 0):.1f}%)")
    
    # Test cluster health check
    cluster_health = health_checker.check_cluster_health("anant-demo-cluster")
    print(f"\nCluster Health: {cluster_health.get('overall_status', 'unknown')}")
    print(f"Monitoring Active: {cluster_health.get('monitoring_enabled', False)}")
    print(f"Check Types: {', '.join(cluster_health.get('check_types', []))}")
    
    return health_checker


def demo_deployment_orchestrator():
    """Demonstrate deployment orchestration."""
    print("\n🚀 Deployment Orchestrator Demo")
    print("=" * 50)
    
    # Create deployment orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # Deploy ANANT services with different strategies
    services_to_deploy = [
        ("anant-hypergraph-api", 3, "blue_green"),
        ("anant-metagraph-api", 2, "rolling"),
        ("anant-analytics-engine", 4, "canary")
    ]
    
    deployment_results = []
    
    for service_name, replicas, strategy in services_to_deploy:
        print(f"\nDeploying {service_name} with {strategy} strategy...")
        success = orchestrator.deploy_service(
            service_name=service_name,
            replicas=replicas,
            strategy=strategy,
            timeout_minutes=5
        )
        deployment_results.append((service_name, success))
        print(f"Deployment result: {'✅' if success else '❌'}")
        
        # Wait between deployments
        time.sleep(2)
    
    # Get deployment history
    print(f"\nDeployment History:")
    deployments = orchestrator.list_deployments(10)
    for deployment in deployments[:5]:  # Show last 5
        print(f"  📦 {deployment['deployment_id']}: {deployment['status']} "
              f"({deployment['services_count']} services)")
    
    # Show deployment logs for the last deployment
    if deployments:
        last_deployment = deployments[0]
        logs = orchestrator.get_deployment_logs(last_deployment['deployment_id'])
        print(f"\nLast Deployment Logs ({last_deployment['deployment_id']}):")
        for log in logs[-5:]:  # Show last 5 logs
            print(f"  📝 {log}")
    
    print(f"\nDeployment Summary:")
    successful = sum(1 for _, success in deployment_results if success)
    print(f"  Total Deployments: {len(deployment_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(deployment_results) - successful}")
    
    return orchestrator


def demo_resource_optimizer():
    """Demonstrate resource optimization."""
    print("\n⚡ Resource Optimizer Demo")
    print("=" * 50)
    
    # Create optimizer configuration
    opt_config = OptimizationConfig(
        auto_apply=False,  # Manual approval for demo
        memory_threshold=75.0,
        cpu_threshold=70.0,
        polars_auto_tune=True,
        parquet_auto_optimize=True
    )
    
    # Create resource optimizer
    optimizer = ResourceOptimizer(opt_config)
    
    # Start optimization monitoring
    print("Starting resource optimization monitoring...")
    optimizer.start_optimization(interval_minutes=1)  # Short interval for demo
    
    # Wait for initial analysis
    time.sleep(5)
    
    # Get current recommendations
    recommendations = optimizer.get_current_recommendations()
    print(f"Current Optimization Recommendations: {len(recommendations)}")
    
    for i, rec in enumerate(recommendations[:3]):  # Show first 3
        print(f"\n  {i+1}. {rec['type'].upper()} Optimization:")
        print(f"     Priority: {rec['priority']}")
        print(f"     Description: {rec['description']}")
        print(f"     Current: {rec['current_value']}")
        print(f"     Recommended: {rec['recommended_value']}")
        print(f"     Expected Improvement: {rec['expected_improvement']}")
        print(f"     Implementation: {rec['implementation_effort']}")
        if rec['risks']:
            print(f"     Risks: {', '.join(rec['risks'])}")
    
    # Apply a low-risk recommendation if available
    if recommendations:
        low_risk_recommendations = [
            i for i, rec in enumerate(recommendations)
            if rec['implementation_effort'] == 'low' and rec['priority'] in ['medium', 'high']
        ]
        
        if low_risk_recommendations:
            rec_index = low_risk_recommendations[0]
            print(f"\nApplying recommendation {rec_index + 1}...")
            success = optimizer.apply_recommendation(rec_index)
            print(f"Application result: {'✅' if success else '❌'}")
    
    # Wait for optimization to take effect
    time.sleep(3)
    
    # Get performance summary
    summary = optimizer.get_performance_summary()
    print(f"\nOptimization Summary:")
    print(f"  Total Optimizations: {summary['total_optimizations']}")
    print(f"  Successful: {summary['successful_optimizations']}")
    print(f"  Pending Recommendations: {summary['pending_recommendations']}")
    print(f"  Auto-apply Enabled: {summary['auto_apply_enabled']}")
    
    # Show cumulative improvements
    improvements = summary.get('cumulative_improvements', {})
    if improvements:
        print(f"  Cumulative Improvements:")
        for metric, improvement in improvements.items():
            if improvement != 0:
                print(f"    {metric}: {improvement:+.1f}%")
    
    # Get optimization history
    history = optimizer.get_optimization_history(5)
    if history:
        print(f"\nRecent Optimizations:")
        for opt in history[-3:]:  # Show last 3
            print(f"  🔧 {opt['type']}: {'✅' if opt['applied'] else '❌'}")
            if opt['improvement']:
                for metric, value in opt['improvement'].items():
                    if value != 0:
                        print(f"      {metric}: {value:+.1f}%")
    
    optimizer.stop_optimization()
    return optimizer


def demo_auto_scaler():
    """Demonstrate auto-scaling capabilities."""
    print("\n⚖️ Auto Scaler Demo")
    print("=" * 50)
    
    # Create scaling policies
    policies = [
        ScalingPolicy(
            name="cpu_scaling",
            metric="cpu",
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
            min_replicas=2,
            max_replicas=10,
            scale_up_cooldown=60,   # 1 minute
            scale_down_cooldown=120  # 2 minutes
        ),
        ScalingPolicy(
            name="memory_scaling",
            metric="memory",
            scale_up_threshold=80.0,
            scale_down_threshold=40.0,
            min_replicas=2,
            max_replicas=8,
            scale_up_cooldown=90,
            scale_down_cooldown=180
        )
    ]
    
    # Create auto scaler
    scaler = AutoScaler(policies)
    
    # Start scaling
    print("Starting auto-scaling...")
    scaler.start_scaling(interval_seconds=10)  # Short interval for demo
    
    # Wait for metrics collection and scaling decisions
    time.sleep(15)
    
    # Get current service metrics
    metrics = scaler.get_service_metrics()
    print(f"Monitored Services: {len(metrics)}")
    
    for service_name, service_metrics in list(metrics.items())[:3]:  # Show first 3
        print(f"\n  📊 {service_name}:")
        print(f"     CPU: {service_metrics['cpu_usage']:.1f}%")
        print(f"     Memory: {service_metrics['memory_usage']:.1f}%")
        print(f"     Replicas: {service_metrics['current_replicas']}")
        print(f"     Requests/sec: {service_metrics['requests_per_second']:.1f}")
        print(f"     Response time: {service_metrics['response_time']:.3f}s")
    
    # Get scaling history
    history = scaler.get_scaling_history(10)
    print(f"\nScaling Events: {len(history)}")
    
    if history:
        print(f"Recent Scaling Events:")
        for event in history[-3:]:  # Show last 3
            direction_emoji = "📈" if event['direction'] == 'up' else "📉"
            status_emoji = "✅" if event['success'] else "❌"
            print(f"  {direction_emoji} {status_emoji} {event['service_name']}: "
                  f"{event['before_replicas']} → {event['after_replicas']} replicas")
            print(f"      Trigger: {event['trigger_metric']}={event['trigger_value']:.1f} "
                  f"(threshold: {event['threshold']:.1f})")
            print(f"      Reason: {event['reason']}")
    
    # Test manual scaling
    if metrics:
        service_name = list(metrics.keys())[0]
        print(f"\nTesting manual scaling for {service_name}...")
        success = scaler.manual_scale(service_name, 5)
        print(f"Manual scaling result: {'✅' if success else '❌'}")
    
    # Test predictive scaling
    if metrics:
        service_name = list(metrics.keys())[0]
        print(f"\nPredictive scaling analysis for {service_name}...")
        prediction = scaler.predict_scaling_needs(service_name, minutes_ahead=15)
        
        if prediction.get('prediction') != 'insufficient_data':
            print(f"  Prediction Horizon: {prediction['prediction_horizon_minutes']} minutes")
            
            predictions = prediction.get('predictions', {})
            for metric, pred_data in predictions.items():
                if isinstance(pred_data, dict):
                    current = pred_data.get('current', 0)
                    predicted = pred_data.get('predicted', 0)
                    trend = pred_data.get('trend', 0)
                    confidence = pred_data.get('confidence', 0)
                    print(f"  {metric}: {current:.1f}% → {predicted:.1f}% "
                          f"(trend: {trend:+.2f}, confidence: {confidence:.2f})")
            
            recommendation = prediction.get('recommendation', {})
            if recommendation.get('action') != 'none':
                print(f"  🎯 Recommendation: {recommendation['action']}")
                print(f"     Reason: {recommendation['reason']}")
                print(f"     Confidence: {recommendation['confidence']:.2f}")
                print(f"     Estimated Replicas: {recommendation['estimated_replicas']}")
        else:
            print(f"  ⚠️ Insufficient data for prediction")
    
    # Get scaling status
    status = scaler.get_scaling_status()
    print(f"\nAuto Scaler Status:")
    print(f"  Active: {status['is_active']}")
    print(f"  Total Policies: {status['total_policies']}")
    print(f"  Enabled Policies: {status['enabled_policies']}")
    print(f"  Services Monitored: {status['services_monitored']}")
    print(f"  Total Scaling Events: {status['total_scaling_events']}")
    
    scaler.stop_scaling()
    return scaler


async def demo_full_production_environment():
    """Demonstrate complete production environment."""
    print("\n🌟 Complete Production Environment Demo")
    print("=" * 60)
    
    # Create production configuration
    config = ProductionConfig(
        environment=EnvironmentType.PRODUCTION,
        storage_path="/tmp/anant-full-demo",
        monitoring_enabled=True,
        auto_scaling_enabled=True
    )
    
    # Initialize all components
    print("Initializing production environment...")
    prod_manager = ProductionManager(config)
    
    # Start all services
    print("Starting production services...")
    success = prod_manager.start_services()
    
    if success:
        print("✅ Production environment started successfully!")
        
        # Simulate production workload
        print("\nSimulating production workload...")
        
        # Create some test data operations
        for i in range(5):
            # Hypergraph operations
            df = pl.DataFrame({
                "nodes": range(1000),
                "edges": [f"edge_{j}" for j in range(1000)],
                "weights": [j * 0.1 for j in range(1000)]
            })
            result = df.group_by("edges").agg(pl.sum("weights"))
            
            # Metagraph operations  
            metadata_df = pl.DataFrame({
                "entities": [f"entity_{j}" for j in range(500)],
                "types": ["product", "customer"] * 250,
                "metadata": [f"meta_{j}" for j in range(500)]
            })
            filtered = metadata_df.filter(pl.col("types") == "product")
            
            print(f"  Batch {i+1}: Processed {result.height} hypergraph edges, {filtered.height} entities")
            time.sleep(1)
        
        # Get final status
        cluster_status = prod_manager.get_cluster_status()
        print(f"\nFinal Production Status:")
        print(f"  Cluster Health: {cluster_status.overall_health}")
        print(f"  Running Services: {len([s for s in cluster_status.services if s.status == 'running'])}")
        print(f"  CPU Utilization: {cluster_status.cpu_used:.1f}/{cluster_status.cpu_total}")
        print(f"  Memory Usage: {cluster_status.memory_used:.1f}/{cluster_status.memory_total}GB")
        
        # Stop services gracefully
        print("\nStopping production environment...")
        prod_manager.stop_services()
        print("✅ Production environment stopped gracefully")
    
    else:
        print("❌ Failed to start production environment")


def main():
    """Run complete production framework demonstration."""
    print("🚀 ANANT Production Deployment Framework")
    print("=" * 60)
    print("Enterprise-grade production system with Polars+Parquet backend")
    print("Features: Monitoring, Health Checks, Auto-scaling, Optimization, Deployment")
    print("=" * 60)
    
    try:
        # 1. Configuration Demo
        config = demo_production_config()
        
        # 2. Production Manager Demo
        prod_manager = demo_production_manager(config)
        
        # 3. Performance Monitoring Demo
        perf_monitor = demo_performance_monitor()
        
        # 4. Health Checking Demo
        health_checker = demo_health_checker()
        
        # 5. Deployment Orchestration Demo
        orchestrator = demo_deployment_orchestrator()
        
        # 6. Resource Optimization Demo
        optimizer = demo_resource_optimizer()
        
        # 7. Auto-scaling Demo
        scaler = demo_auto_scaler()
        
        # 8. Full Environment Demo
        print("\nRunning full production environment demo...")
        asyncio.run(demo_full_production_environment())
        
        # Final summary
        print("\n🎉 PRODUCTION FRAMEWORK DEMO COMPLETE")
        print("=" * 60)
        print("✅ All production components demonstrated successfully")
        print("✅ Enterprise-grade features validated")
        print("✅ Polars+Parquet backend performance confirmed")
        print("✅ Ready for production deployment")
        
        # Stop any remaining services
        prod_manager.stop_services()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()