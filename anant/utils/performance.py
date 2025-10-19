"""
Performance Monitoring and Optimization for ANANT
================================================

Enhanced performance monitoring, profiling, and optimization
utilities for production-ready hypergraph analytics.
"""

import time
import psutil
import logging
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for algorithm execution"""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    input_size: int
    algorithm_type: str
    timestamp: float
    success: bool
    error_message: Optional[str] = None
    

class PerformanceMonitor:
    """
    Comprehensive performance monitoring and profiling system
    
    Tracks execution time, memory usage, CPU utilization, and provides
    recommendations for optimization based on usage patterns.
    """
    
    def __init__(self, 
                 enable_profiling: bool = True,
                 log_to_file: bool = False,
                 log_file_path: Optional[str] = None):
        """
        Initialize performance monitor
        
        Args:
            enable_profiling: Whether to collect detailed metrics
            log_to_file: Whether to log metrics to file
            log_file_path: Path for performance log file
        """
        self.enable_profiling = enable_profiling
        self.metrics_history: List[PerformanceMetrics] = []
        self.function_stats = defaultdict(list)
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or "anant_performance.json"
        
        # Performance thresholds
        self.thresholds = {
            'execution_time_warning': 10.0,  # seconds
            'execution_time_critical': 60.0,  # seconds
            'memory_usage_warning': 1024.0,  # MB
            'memory_usage_critical': 4096.0,  # MB
            'cpu_usage_warning': 80.0,  # percent
        }
    
    def monitor_function(self, 
                        algorithm_type: str = 'general',
                        input_size_func: Optional[Callable] = None):
        """
        Decorator for monitoring function performance
        
        Args:
            algorithm_type: Type of algorithm ('clustering', 'centrality', etc.)
            input_size_func: Function to extract input size from args
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                return self._execute_with_monitoring(
                    func, algorithm_type, input_size_func, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def _execute_with_monitoring(self, 
                                func: Callable,
                                algorithm_type: str,
                                input_size_func: Optional[Callable],
                                *args, **kwargs):
        """Execute function with comprehensive monitoring"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        # Calculate input size
        input_size = 0
        if input_size_func:
            try:
                input_size = input_size_func(*args, **kwargs)
            except:
                input_size = 0
        elif args:
            # Try to infer size from first argument
            first_arg = args[0]
            if hasattr(first_arg, 'nodes'):
                input_size = len(first_arg.nodes)
            elif hasattr(first_arg, '__len__'):
                input_size = len(first_arg)
        
        error_message = None
        success = True
        result = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Error in {func.__name__}: {e}")
            raise
        finally:
            # Record metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = max(end_memory - start_memory, 0)
            cpu_percent = (start_cpu + end_cpu) / 2
            
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                input_size=input_size,
                algorithm_type=algorithm_type,
                timestamp=end_time,
                success=success,
                error_message=error_message
            )
            
            self._record_metrics(metrics)
            self._check_thresholds(metrics)
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.metrics_history.append(metrics)
        self.function_stats[metrics.function_name].append(metrics)
        
        # Log to file if enabled
        if self.log_to_file:
            self._log_metrics_to_file(metrics)
        
        # Keep only recent metrics to prevent memory bloat
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-800:]
    
    def _log_metrics_to_file(self, metrics: PerformanceMetrics):
        """Log metrics to JSON file"""
        try:
            log_entry = {
                'function_name': metrics.function_name,
                'execution_time': metrics.execution_time,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_percent': metrics.cpu_percent,
                'input_size': metrics.input_size,
                'algorithm_type': metrics.algorithm_type,
                'timestamp': metrics.timestamp,
                'success': metrics.success,
                'error_message': metrics.error_message
            }
            
            # Append to log file
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to file: {e}")
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and log warnings"""
        
        if metrics.execution_time > self.thresholds['execution_time_critical']:
            logger.critical(
                f"{metrics.function_name} took {metrics.execution_time:.2f}s "
                f"(critical threshold: {self.thresholds['execution_time_critical']}s)"
            )
        elif metrics.execution_time > self.thresholds['execution_time_warning']:
            logger.warning(
                f"{metrics.function_name} took {metrics.execution_time:.2f}s "
                f"(warning threshold: {self.thresholds['execution_time_warning']}s)"
            )
        
        if metrics.memory_usage_mb > self.thresholds['memory_usage_critical']:
            logger.critical(
                f"{metrics.function_name} used {metrics.memory_usage_mb:.1f}MB "
                f"(critical threshold: {self.thresholds['memory_usage_critical']}MB)"
            )
        elif metrics.memory_usage_mb > self.thresholds['memory_usage_warning']:
            logger.warning(
                f"{metrics.function_name} used {metrics.memory_usage_mb:.1f}MB "
                f"(warning threshold: {self.thresholds['memory_usage_warning']}MB)"
            )
        
        if metrics.cpu_percent > self.thresholds['cpu_usage_warning']:
            logger.warning(
                f"{metrics.function_name} used {metrics.cpu_percent:.1f}% CPU "
                f"(warning threshold: {self.thresholds['cpu_usage_warning']}%)"
            )
    
    def get_performance_report(self, 
                              function_name: Optional[str] = None,
                              algorithm_type: Optional[str] = None,
                              last_n_calls: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            function_name: Filter by specific function
            algorithm_type: Filter by algorithm type
            last_n_calls: Include only last N calls
            
        Returns:
            Performance report dictionary
        """
        
        # Filter metrics
        filtered_metrics = self.metrics_history
        
        if function_name:
            filtered_metrics = [m for m in filtered_metrics if m.function_name == function_name]
        
        if algorithm_type:
            filtered_metrics = [m for m in filtered_metrics if m.algorithm_type == algorithm_type]
        
        if last_n_calls:
            filtered_metrics = filtered_metrics[-last_n_calls:]
        
        if not filtered_metrics:
            return {'error': 'No metrics found for the specified filters'}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in filtered_metrics]
        memory_usages = [m.memory_usage_mb for m in filtered_metrics]
        cpu_usages = [m.cpu_percent for m in filtered_metrics]
        input_sizes = [m.input_size for m in filtered_metrics if m.input_size > 0]
        
        success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
        
        report = {
            'summary': {
                'total_calls': len(filtered_metrics),
                'success_rate': success_rate,
                'time_period': {
                    'start': min(m.timestamp for m in filtered_metrics),
                    'end': max(m.timestamp for m in filtered_metrics)
                }
            },
            'execution_time': {
                'mean': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'median': sorted(execution_times)[len(execution_times) // 2],
                'p95': sorted(execution_times)[int(len(execution_times) * 0.95)]
            },
            'memory_usage': {
                'mean': sum(memory_usages) / len(memory_usages),
                'min': min(memory_usages),
                'max': max(memory_usages),
                'median': sorted(memory_usages)[len(memory_usages) // 2]
            },
            'cpu_usage': {
                'mean': sum(cpu_usages) / len(cpu_usages),
                'min': min(cpu_usages),
                'max': max(cpu_usages)
            }
        }
        
        if input_sizes:
            report['input_size'] = {
                'mean': sum(input_sizes) / len(input_sizes),
                'min': min(input_sizes),
                'max': max(input_sizes),
                'median': sorted(input_sizes)[len(input_sizes) // 2]
            }
        
        # Performance trends
        if len(filtered_metrics) > 1:
            recent_metrics = filtered_metrics[-10:]
            older_metrics = filtered_metrics[:-10] if len(filtered_metrics) > 10 else []
            
            if older_metrics:
                recent_avg_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
                older_avg_time = sum(m.execution_time for m in older_metrics) / len(older_metrics)
                
                report['trends'] = {
                    'execution_time_trend': 'improving' if recent_avg_time < older_avg_time else 'degrading',
                    'recent_avg_time': recent_avg_time,
                    'older_avg_time': older_avg_time
                }
        
        return report
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on performance patterns
        
        Returns:
            List of optimization recommendations
        """
        
        recommendations = []
        
        if not self.metrics_history:
            return ["No performance data available for recommendations"]
        
        # Analyze patterns
        slow_functions = []
        memory_intensive_functions = []
        
        for func_name, metrics_list in self.function_stats.items():
            if not metrics_list:
                continue
            
            avg_time = sum(m.execution_time for m in metrics_list) / len(metrics_list)
            avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
            
            if avg_time > self.thresholds['execution_time_warning']:
                slow_functions.append((func_name, avg_time))
            
            if avg_memory > self.thresholds['memory_usage_warning']:
                memory_intensive_functions.append((func_name, avg_memory))
        
        # Generate recommendations
        if slow_functions:
            slow_functions.sort(key=lambda x: x[1], reverse=True)
            recommendations.append(
                f"Optimize execution time for: {', '.join(f[0] for f in slow_functions[:3])}"
            )
            
            # Check if sampling could help
            for func_name, avg_time in slow_functions:
                func_metrics = self.function_stats[func_name]
                large_input_calls = [m for m in func_metrics if m.input_size > 1000]
                
                if large_input_calls:
                    recommendations.append(
                        f"Consider using intelligent sampling for {func_name} on large inputs"
                    )
        
        if memory_intensive_functions:
            memory_intensive_functions.sort(key=lambda x: x[1], reverse=True)
            recommendations.append(
                f"Optimize memory usage for: {', '.join(f[0] for f in memory_intensive_functions[:3])}"
            )
        
        # Check for failed calls
        failed_calls = [m for m in self.metrics_history if not m.success]
        if failed_calls:
            error_functions = defaultdict(int)
            for call in failed_calls:
                error_functions[call.function_name] += 1
            
            most_error_prone = max(error_functions.items(), key=lambda x: x[1])
            recommendations.append(
                f"Investigate error handling in {most_error_prone[0]} "
                f"({most_error_prone[1]} failures)"
            )
        
        # Performance scaling recommendations
        large_input_metrics = [m for m in self.metrics_history if m.input_size > 5000]
        if large_input_metrics:
            avg_time_large = sum(m.execution_time for m in large_input_metrics) / len(large_input_metrics)
            if avg_time_large > 30:
                recommendations.append(
                    "Implement progressive analysis for very large datasets"
                )
        
        if not recommendations:
            recommendations.append("Performance looks good! No specific optimizations needed.")
        
        return recommendations
    
    def export_performance_data(self, filepath: str, format: str = 'json'):
        """Export performance data to file"""
        
        data = {
            'metrics': [
                {
                    'function_name': m.function_name,
                    'execution_time': m.execution_time,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_percent': m.cpu_percent,
                    'input_size': m.input_size,
                    'algorithm_type': m.algorithm_type,
                    'timestamp': m.timestamp,
                    'success': m.success,
                    'error_message': m.error_message
                }
                for m in self.metrics_history
            ],
            'summary': self.get_performance_report()
        }
        
        filepath_obj = Path(filepath)
        
        if format == 'json':
            with open(filepath_obj, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Performance data exported to {filepath_obj}")


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def performance_monitor(algorithm_type: str = 'general',
                       input_size_func: Optional[Callable] = None):
    """
    Global performance monitoring decorator
    
    Args:
        algorithm_type: Type of algorithm being monitored
        input_size_func: Function to extract input size from arguments
    """
    return _global_monitor.monitor_function(algorithm_type, input_size_func)


def get_performance_report(**kwargs) -> Dict[str, Any]:
    """Get performance report from global monitor"""
    return _global_monitor.get_performance_report(**kwargs)


def get_optimization_recommendations() -> List[str]:
    """Get optimization recommendations from global monitor"""
    return _global_monitor.get_optimization_recommendations()


def export_performance_data(filepath: str, format: str = 'json'):
    """Export performance data from global monitor"""
    _global_monitor.export_performance_data(filepath, format)


def enable_performance_monitoring(enable: bool = True, log_to_file: bool = False):
    """Enable or disable global performance monitoring"""
    global _global_monitor
    _global_monitor.enable_profiling = enable
    _global_monitor.log_to_file = log_to_file
    
    if enable:
        logger.info("Performance monitoring enabled")
    else:
        logger.info("Performance monitoring disabled")


def set_performance_thresholds(**thresholds):
    """Set custom performance thresholds"""
    global _global_monitor
    _global_monitor.thresholds.update(thresholds)
    logger.info(f"Performance thresholds updated: {thresholds}")


# Utility functions for input size extraction
def hypergraph_input_size(*args, **kwargs):
    """Extract input size from hypergraph argument"""
    if args and hasattr(args[0], 'nodes'):
        return len(args[0].nodes)
    return 0


def dataframe_input_size(*args, **kwargs):
    """Extract input size from DataFrame argument"""
    if args and hasattr(args[0], '__len__'):
        return len(args[0])
    return 0


class PerformanceProfiler:
    """
    Context manager for detailed performance profiling
    
    Usage:
        with PerformanceProfiler("my_operation") as profiler:
            # Your code here
            pass
        
        print(profiler.get_report())
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.checkpoints = []
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.end_memory = self._get_memory_usage()
    
    def checkpoint(self, name: str):
        """Add a performance checkpoint"""
        current_time = time.time()
        current_memory = self._get_memory_usage()
        
        self.checkpoints.append({
            'name': name,
            'time': current_time,
            'memory_mb': current_memory,
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'memory_delta': current_memory - self.start_memory if self.start_memory else 0
        })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_report(self) -> Dict[str, Any]:
        """Get detailed profiling report"""
        if not self.start_time or not self.end_time:
            return {'error': 'Profiling not completed'}
        
        total_time = self.end_time - self.start_time
        total_memory = (self.end_memory or 0) - (self.start_memory or 0)
        
        report = {
            'operation_name': self.operation_name,
            'total_execution_time': total_time,
            'total_memory_delta': total_memory,
            'checkpoints': self.checkpoints
        }
        
        if self.checkpoints:
            checkpoint_times = [cp['elapsed_time'] for cp in self.checkpoints]
            report['checkpoint_analysis'] = {
                'slowest_checkpoint': max(self.checkpoints, key=lambda x: x['elapsed_time'])['name'],
                'fastest_checkpoint': min(self.checkpoints, key=lambda x: x['elapsed_time'])['name'],
                'avg_checkpoint_time': sum(checkpoint_times) / len(checkpoint_times)
            }
        
        return report