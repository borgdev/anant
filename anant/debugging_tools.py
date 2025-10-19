"""
Advanced Debugging Tools for Anant Library

Provides comprehensive debugging utilities including:
- Performance profiling and bottleneck detection
- Memory usage analysis
- Data integrity validation
- Operation tracing and logging
- Interactive debugging helpers

These tools help developers identify and resolve issues quickly during development.
"""

import time
import traceback
import functools
import sys
import gc
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import polars as pl
import psutil
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
import json
from datetime import datetime

# Try importing memory_profiler if available
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profile = None

# Try importing line_profiler if available
try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    line_profiler = None


class PerformanceProfiler:
    """
    Comprehensive performance profiling for Anant operations
    """
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        self.call_stack = []
        
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling function execution"""
        
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(name, func, *args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def _profile_execution(self, name: str, func: Callable, *args, **kwargs):
        """Execute function with profiling"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Track call stack
        self.call_stack.append(name)
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Record profile data
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'memory_peak': max(start_memory, end_memory),
                'timestamp': datetime.now().isoformat(),
                'call_depth': len(self.call_stack),
                'success': True,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
            
            self._store_profile(name, profile_data)
            
            return result
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Record failed execution
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'memory_peak': max(start_memory, end_memory),
                'timestamp': datetime.now().isoformat(),
                'call_depth': len(self.call_stack),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
            
            self._store_profile(name, profile_data)
            
            raise
        
        finally:
            # Pop from call stack
            if self.call_stack:
                self.call_stack.pop()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _store_profile(self, name: str, data: Dict):
        """Store profile data"""
        if name not in self.profiles:
            self.profiles[name] = []
        
        self.profiles[name].append(data)
        
        # Keep only last 100 entries per function
        if len(self.profiles[name]) > 100:
            self.profiles[name] = self.profiles[name][-100:]
    
    @contextmanager
    def profile_context(self, name: str):
        """Context manager for profiling code blocks"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'memory_peak': max(start_memory, end_memory),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'context': True
            }
            
            self._store_profile(name, profile_data)
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'memory_peak': max(start_memory, end_memory),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'context': True
            }
            
            self._store_profile(name, profile_data)
            
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        
        summary = {
            'total_functions': len(self.profiles),
            'total_calls': sum(len(calls) for calls in self.profiles.values()),
            'functions': {}
        }
        
        for func_name, calls in self.profiles.items():
            successful_calls = [c for c in calls if c['success']]
            failed_calls = [c for c in calls if not c['success']]
            
            if successful_calls:
                durations = [c['duration'] for c in successful_calls]
                memories = [c['memory_delta'] for c in successful_calls]
                
                func_summary = {
                    'total_calls': len(calls),
                    'successful_calls': len(successful_calls),
                    'failed_calls': len(failed_calls),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations),
                    'avg_memory_delta': sum(memories) / len(memories),
                    'max_memory_delta': max(memories) if memories else 0,
                    'last_call': calls[-1]['timestamp']
                }
                
                summary['functions'][func_name] = func_summary
        
        return summary
    
    def print_summary(self):
        """Print formatted profiling summary"""
        
        summary = self.get_summary()
        
        print("🔍 Performance Profiling Summary")
        print("=" * 50)
        print(f"Total Functions Profiled: {summary['total_functions']}")
        print(f"Total Function Calls: {summary['total_calls']}")
        print()
        
        # Sort by total duration
        functions = summary['functions']
        sorted_functions = sorted(
            functions.items(),
            key=lambda x: x[1]['total_duration'],
            reverse=True
        )
        
        print("Top Functions by Total Duration:")
        print("-" * 50)
        
        for func_name, stats in sorted_functions[:10]:
            print(f"\n📊 {func_name}")
            print(f"  Calls: {stats['total_calls']} (✅ {stats['successful_calls']}, ❌ {stats['failed_calls']})")
            print(f"  Duration: {stats['total_duration']:.3f}s total, {stats['avg_duration']:.4f}s avg")
            print(f"  Memory: {stats['avg_memory_delta']:+.2f} MB avg, {stats['max_memory_delta']:+.2f} MB max")
    
    def save_report(self, filepath: str):
        """Save detailed profiling report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'detailed_profiles': self.profiles
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Profiling report saved to {filepath}")


class DataIntegrityValidator:
    """
    Comprehensive data integrity validation for hypergraphs and datasets
    """
    
    def __init__(self):
        self.validation_history = []
        self.issues_found = defaultdict(list)
    
    def validate_hypergraph(self, hypergraph, deep_check: bool = False) -> Dict[str, Any]:
        """Validate hypergraph data integrity"""
        
        validation_start = time.time()
        issues = []
        warnings = []
        
        try:
            # Basic structure validation
            issues.extend(self._validate_structure(hypergraph))
            
            # Data consistency validation
            issues.extend(self._validate_consistency(hypergraph))
            
            # Performance validation
            warnings.extend(self._validate_performance(hypergraph))
            
            if deep_check:
                # Deep validation (more expensive)
                issues.extend(self._validate_deep(hypergraph))
            
            validation_duration = time.time() - validation_start
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'validation_duration': validation_duration,
                'hypergraph_id': id(hypergraph),
                'num_nodes': getattr(hypergraph, 'num_nodes', 0),
                'num_edges': getattr(hypergraph, 'num_edges', 0),
                'issues': issues,
                'warnings': warnings,
                'is_valid': len(issues) == 0,
                'deep_check': deep_check
            }
            
            self.validation_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'validation_duration': time.time() - validation_start,
                'error': f"Validation failed: {e}",
                'is_valid': False
            }
    
    def _validate_structure(self, hg) -> List[Dict]:
        """Validate basic hypergraph structure"""
        issues = []
        
        try:
            # Check if basic attributes exist
            if not hasattr(hg, 'nodes'):
                issues.append({
                    'type': 'missing_attribute',
                    'severity': 'error',
                    'message': 'Hypergraph missing nodes attribute'
                })
            
            if not hasattr(hg, 'edges'):
                issues.append({
                    'type': 'missing_attribute',
                    'severity': 'error',
                    'message': 'Hypergraph missing edges attribute'
                })
            
            if not hasattr(hg, 'incidences'):
                issues.append({
                    'type': 'missing_attribute',
                    'severity': 'error',
                    'message': 'Hypergraph missing incidences attribute'
                })
            
            # Check node/edge counts
            if hasattr(hg, 'num_nodes') and hg.num_nodes < 0:
                issues.append({
                    'type': 'invalid_count',
                    'severity': 'error',
                    'message': f'Negative node count: {hg.num_nodes}'
                })
            
            if hasattr(hg, 'num_edges') and hg.num_edges < 0:
                issues.append({
                    'type': 'invalid_count',
                    'severity': 'error',
                    'message': f'Negative edge count: {hg.num_edges}'
                })
            
        except Exception as e:
            issues.append({
                'type': 'structure_validation_error',
                'severity': 'error',
                'message': f'Error during structure validation: {e}'
            })
        
        return issues
    
    def _validate_consistency(self, hg) -> List[Dict]:
        """Validate data consistency"""
        issues = []
        
        try:
            # Check node-edge consistency
            if hasattr(hg, 'nodes') and hasattr(hg, 'edges'):
                
                # Sample check for large graphs
                sample_size = min(100, len(hg.edges) if hasattr(hg.edges, '__len__') else 10)
                edge_sample = list(hg.edges)[:sample_size] if hasattr(hg.edges, '__iter__') else []
                
                for edge in edge_sample:
                    try:
                        edge_nodes = hg.edges[edge]
                        for node in edge_nodes:
                            if node not in hg.nodes:
                                issues.append({
                                    'type': 'consistency_error',
                                    'severity': 'error',
                                    'message': f'Edge {edge} contains node {node} not in node set'
                                })
                                break  # Don't flood with similar errors
                    except Exception as e:
                        issues.append({
                            'type': 'edge_access_error',
                            'severity': 'warning',
                            'message': f'Could not access edge {edge}: {e}'
                        })
            
            # Check incidence data consistency
            if hasattr(hg, 'incidences') and hasattr(hg.incidences, 'data'):
                try:
                    data = hg.incidences.data
                    if len(data) > 0:
                        # Check for null values in critical columns
                        if hasattr(data, 'null_count'):
                            try:
                                null_counts = data.null_count()
                                # Convert to dict if it's a Polars DataFrame
                                if hasattr(null_counts, 'to_dict'):
                                    null_counts_dict = null_counts.to_dict(as_series=False)
                                    # Get first row values if it's a nested dict
                                    if isinstance(next(iter(null_counts_dict.values())), list):
                                        null_counts_dict = {k: v[0] for k, v in null_counts_dict.items()}
                                elif hasattr(null_counts, 'to_pandas'):
                                    null_counts_dict = null_counts.to_pandas().iloc[0].to_dict()
                                else:
                                    null_counts_dict = dict(null_counts)
                                
                                for col, null_count in null_counts_dict.items():
                                    if null_count and null_count > len(data) * 0.1:  # More than 10% null
                                        issues.append({
                                            'type': 'data_quality',
                                            'severity': 'warning',
                                            'message': f'Column {col} has {null_count} null values ({null_count/len(data)*100:.1f}%)'
                                        })
                            except Exception as e:
                                # If null count analysis fails, just skip it
                                pass
                
                except Exception as e:
                    issues.append({
                        'type': 'data_consistency_error',
                        'severity': 'warning',
                        'message': f'Could not validate incidence data: {e}'
                    })
            
        except Exception as e:
            issues.append({
                'type': 'consistency_validation_error',
                'severity': 'error',
                'message': f'Error during consistency validation: {e}'
            })
        
        return issues
    
    def _validate_performance(self, hg) -> List[Dict]:
        """Validate performance characteristics"""
        warnings = []
        
        try:
            # Check size warnings
            if hasattr(hg, 'num_nodes') and hg.num_nodes > 1000000:
                warnings.append({
                    'type': 'performance_warning',
                    'severity': 'warning',
                    'message': f'Large node count ({hg.num_nodes:,}) may cause performance issues'
                })
            
            if hasattr(hg, 'num_edges') and hg.num_edges > 1000000:
                warnings.append({
                    'type': 'performance_warning',
                    'severity': 'warning',
                    'message': f'Large edge count ({hg.num_edges:,}) may cause performance issues'
                })
            
            # Check data size
            if hasattr(hg, 'incidences') and hasattr(hg.incidences, 'data'):
                try:
                    data_rows = len(hg.incidences.data)
                    if data_rows > 10000000:  # 10M rows
                        warnings.append({
                            'type': 'performance_warning',
                            'severity': 'warning',
                            'message': f'Large dataset ({data_rows:,} rows) may cause memory issues'
                        })
                except:
                    pass
            
        except Exception as e:
            warnings.append({
                'type': 'performance_validation_error',
                'severity': 'warning',
                'message': f'Error during performance validation: {e}'
            })
        
        return warnings
    
    def _validate_deep(self, hg) -> List[Dict]:
        """Deep validation (expensive operations)"""
        issues = []
        
        try:
            # Check for duplicate edges
            if hasattr(hg, 'edges'):
                edge_list = list(hg.edges)
                if len(edge_list) != len(set(edge_list)):
                    issues.append({
                        'type': 'duplicate_edges',
                        'severity': 'warning',
                        'message': 'Hypergraph contains duplicate edge identifiers'
                    })
            
            # Check for isolated nodes (nodes not in any edge)
            if hasattr(hg, 'nodes') and hasattr(hg, 'edges'):
                try:
                    all_edge_nodes = set()
                    for edge in list(hg.edges)[:1000]:  # Sample for performance
                        try:
                            edge_nodes = hg.edges[edge]
                            all_edge_nodes.update(edge_nodes)
                        except:
                            pass
                    
                    isolated_nodes = set(hg.nodes) - all_edge_nodes
                    if isolated_nodes:
                        issues.append({
                            'type': 'isolated_nodes',
                            'severity': 'warning',
                            'message': f'Found {len(isolated_nodes)} isolated nodes (not in any edge)'
                        })
                        
                except Exception as e:
                    issues.append({
                        'type': 'isolation_check_error',
                        'severity': 'warning',
                        'message': f'Could not check for isolated nodes: {e}'
                    })
            
        except Exception as e:
            issues.append({
                'type': 'deep_validation_error',
                'severity': 'error',
                'message': f'Error during deep validation: {e}'
            })
        
        return issues
    
    def print_validation_report(self, result: Dict):
        """Print formatted validation report"""
        
        print("🔍 Data Integrity Validation Report")
        print("=" * 50)
        print(f"Timestamp: {result['timestamp']}")
        print(f"Validation Duration: {result['validation_duration']:.3f}s")
        print(f"Hypergraph: {result.get('num_nodes', 'N/A')} nodes, {result.get('num_edges', 'N/A')} edges")
        print(f"Status: {'✅ VALID' if result['is_valid'] else '❌ INVALID'}")
        print()
        
        if result.get('issues'):
            print("🚨 Issues Found:")
            print("-" * 30)
            for i, issue in enumerate(result['issues'], 1):
                severity_icon = "🔴" if issue['severity'] == 'error' else "🟡"
                print(f"{i}. {severity_icon} [{issue['type']}] {issue['message']}")
            print()
        
        if result.get('warnings'):
            print("⚠️ Warnings:")
            print("-" * 20)
            for i, warning in enumerate(result['warnings'], 1):
                print(f"{i}. 🟡 [{warning['type']}] {warning['message']}")
            print()
        
        if result['is_valid'] and not result.get('warnings'):
            print("✅ No issues found - hypergraph is valid!")


class OperationTracer:
    """
    Operation tracing and logging for debugging complex workflows
    """
    
    def __init__(self, max_trace_length: int = 1000):
        self.traces = deque(maxlen=max_trace_length)
        self.active_operations = {}
        self.trace_id_counter = 0
    
    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations"""
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace_id = self._start_trace(operation_name, func, args, kwargs)
                
                try:
                    result = func(*args, **kwargs)
                    self._complete_trace(trace_id, success=True, result=result)
                    return result
                
                except Exception as e:
                    self._complete_trace(trace_id, success=False, error=e)
                    raise
            
            return wrapper
        
        return decorator
    
    def _start_trace(self, operation_name: str, func: Callable, args: tuple, kwargs: dict) -> int:
        """Start tracing an operation"""
        
        self.trace_id_counter += 1
        trace_id = self.trace_id_counter
        
        trace_entry = {
            'trace_id': trace_id,
            'operation_name': operation_name,
            'function_name': f"{func.__module__}.{func.__name__}",
            'start_time': time.time(),
            'timestamp': datetime.now().isoformat(),
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys()),
            'status': 'running',
            'thread_id': threading.current_thread().ident
        }
        
        self.active_operations[trace_id] = trace_entry
        
        return trace_id
    
    def _complete_trace(self, trace_id: int, success: bool, result=None, error=None):
        """Complete a trace operation"""
        
        if trace_id not in self.active_operations:
            return
        
        trace_entry = self.active_operations.pop(trace_id)
        
        end_time = time.time()
        trace_entry.update({
            'end_time': end_time,
            'duration': end_time - trace_entry['start_time'],
            'status': 'success' if success else 'error',
            'success': success
        })
        
        if success:
            trace_entry['result_type'] = type(result).__name__ if result is not None else 'None'
        else:
            trace_entry.update({
                'error_message': str(error),
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc()
            })
        
        self.traces.append(trace_entry)
    
    @contextmanager
    def trace_context(self, operation_name: str, **context_data):
        """Context manager for tracing code blocks"""
        
        self.trace_id_counter += 1
        trace_id = self.trace_id_counter
        
        trace_entry = {
            'trace_id': trace_id,
            'operation_name': operation_name,
            'start_time': time.time(),
            'timestamp': datetime.now().isoformat(),
            'context_data': context_data,
            'status': 'running',
            'thread_id': threading.current_thread().ident,
            'context': True
        }
        
        self.active_operations[trace_id] = trace_entry
        
        try:
            yield trace_id
            
            end_time = time.time()
            trace_entry.update({
                'end_time': end_time,
                'duration': end_time - trace_entry['start_time'],
                'status': 'success',
                'success': True
            })
            
        except Exception as e:
            end_time = time.time()
            trace_entry.update({
                'end_time': end_time,
                'duration': end_time - trace_entry['start_time'],
                'status': 'error',
                'success': False,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            raise
        
        finally:
            if trace_id in self.active_operations:
                self.active_operations.pop(trace_id)
            self.traces.append(trace_entry)
    
    def get_recent_traces(self, count: int = 10) -> List[Dict]:
        """Get recent trace entries"""
        return list(self.traces)[-count:]
    
    def get_failed_traces(self, count: int = 10) -> List[Dict]:
        """Get recent failed operations"""
        failed = [trace for trace in self.traces if not trace.get('success', True)]
        return failed[-count:]
    
    def print_trace_summary(self):
        """Print trace summary"""
        
        total_traces = len(self.traces)
        successful_traces = len([t for t in self.traces if t.get('success', True)])
        failed_traces = total_traces - successful_traces
        
        print("📋 Operation Trace Summary")
        print("=" * 40)
        print(f"Total Operations: {total_traces}")
        print(f"Successful: {successful_traces} (✅ {successful_traces/max(1,total_traces)*100:.1f}%)")
        print(f"Failed: {failed_traces} (❌ {failed_traces/max(1,total_traces)*100:.1f}%)")
        print(f"Currently Running: {len(self.active_operations)}")
        print()
        
        if failed_traces > 0:
            print("Recent Failures:")
            print("-" * 20)
            for trace in self.get_failed_traces(5):
                print(f"❌ {trace['operation_name']} - {trace.get('error_type', 'Unknown Error')}")
                print(f"   {trace.get('error_message', '')[:100]}...")
        
        if self.active_operations:
            print(f"\nCurrently Running Operations:")
            print("-" * 30)
            for trace_id, trace in self.active_operations.items():
                duration = time.time() - trace['start_time']
                print(f"🔄 {trace['operation_name']} (running {duration:.1f}s)")


# Global instances for easy access
profiler = PerformanceProfiler()
validator = DataIntegrityValidator()
tracer = OperationTracer()


# Convenience functions

def debug_hypergraph(hypergraph, deep_check: bool = False):
    """Quick debug analysis of a hypergraph"""
    
    print("🔍 Quick Hypergraph Debug Analysis")
    print("=" * 45)
    
    # Basic info
    try:
        print(f"Nodes: {hypergraph.num_nodes:,}")
        print(f"Edges: {hypergraph.num_edges:,}")
        
        if hasattr(hypergraph.incidences, 'data'):
            data = hypergraph.incidences.data
            print(f"Data: {len(data):,} rows × {len(data.columns)} cols")
    except Exception as e:
        print(f"❌ Error getting basic info: {e}")
    
    # Run validation
    result = validator.validate_hypergraph(hypergraph, deep_check=deep_check)
    validator.print_validation_report(result)
    
    return result


def start_profiling():
    """Start global profiling"""
    print("🔍 Performance profiling started")
    return profiler


def stop_profiling():
    """Stop profiling and show summary"""
    profiler.print_summary()
    return profiler


def trace_calls():
    """Get operation tracer for debugging function calls"""
    return tracer


@contextmanager
def debug_context(name: str):
    """Context manager for debugging code blocks"""
    
    print(f"🔍 Starting debug context: {name}")
    
    with profiler.profile_context(f"debug_{name}"):
        with tracer.trace_context(f"debug_{name}"):
            yield
    
    print(f"✅ Debug context completed: {name}")


def memory_usage():
    """Get current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print(f"🧠 Memory Usage:")
        print(f"  RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"  VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        }
    except Exception as e:
        print(f"❌ Could not get memory usage: {e}")
        return None


def cleanup_debugging():
    """Clean up debugging resources"""
    global profiler, validator, tracer
    
    print("🧹 Cleaning up debugging resources...")
    
    # Clear histories
    profiler.profiles.clear()
    validator.validation_history.clear()
    tracer.traces.clear()
    
    # Force garbage collection
    gc.collect()
    
    print("✅ Debugging cleanup completed")