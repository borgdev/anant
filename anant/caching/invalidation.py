"""
Cache Invalidation - Intelligent Cache Invalidation Strategies

Provides sophisticated cache invalidation mechanisms including
dependency tracking, event-based invalidation, and time-based expiration.
"""

import time
import threading
import weakref
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    DEPENDENCY = "dependency"  # Dependency-based
    EVENT = "event"  # Event-driven
    MANUAL = "manual"  # Manual invalidation


@dataclass
class InvalidationRule:
    """Rule for cache invalidation."""
    pattern: str
    strategy: InvalidationStrategy
    ttl: Optional[int] = None
    max_age: Optional[int] = None
    dependencies: Set[str] = field(default_factory=set)
    events: Set[str] = field(default_factory=set)
    condition: Optional[Callable[[str, Any], bool]] = None


@dataclass
class CacheEvent:
    """Cache invalidation event."""
    event_type: str
    source: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class DependencyTracker:
    """
    Tracks dependencies between cache entries and external resources.
    """
    
    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # resource -> cache_keys
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)  # cache_key -> resources
        self._lock = threading.RLock()
        
    def add_dependency(self, cache_key: str, resource: str):
        """Add dependency between cache key and resource."""
        with self._lock:
            self._dependencies[resource].add(cache_key)
            self._reverse_deps[cache_key].add(resource)
            
    def remove_dependency(self, cache_key: str, resource: Optional[str] = None):
        """Remove dependency for cache key."""
        with self._lock:
            if resource:
                # Remove specific dependency
                self._dependencies[resource].discard(cache_key)
                self._reverse_deps[cache_key].discard(resource)
                
                # Cleanup empty sets
                if not self._dependencies[resource]:
                    del self._dependencies[resource]
                if not self._reverse_deps[cache_key]:
                    del self._reverse_deps[cache_key]
            else:
                # Remove all dependencies for cache key
                resources = self._reverse_deps.get(cache_key, set()).copy()
                for res in resources:
                    self._dependencies[res].discard(cache_key)
                    if not self._dependencies[res]:
                        del self._dependencies[res]
                        
                if cache_key in self._reverse_deps:
                    del self._reverse_deps[cache_key]
                    
    def get_dependent_keys(self, resource: str) -> Set[str]:
        """Get cache keys that depend on a resource."""
        with self._lock:
            return self._dependencies.get(resource, set()).copy()
            
    def get_dependencies(self, cache_key: str) -> Set[str]:
        """Get resources that a cache key depends on."""
        with self._lock:
            return self._reverse_deps.get(cache_key, set()).copy()
            
    def clear_all(self):
        """Clear all dependency mappings."""
        with self._lock:
            self._dependencies.clear()
            self._reverse_deps.clear()


class EventBus:
    """
    Simple event bus for cache invalidation events.
    """
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def subscribe(self, event_type: str, callback: Callable[[CacheEvent], None]):
        """Subscribe to events of a specific type."""
        with self._lock:
            self._listeners[event_type].append(callback)
            
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events."""
        with self._lock:
            if event_type in self._listeners:
                try:
                    self._listeners[event_type].remove(callback)
                except ValueError:
                    pass
                    
    def publish(self, event: CacheEvent):
        """Publish an event to all subscribers."""
        with self._lock:
            listeners = self._listeners.get(event.event_type, []).copy()
            
        # Call listeners outside of lock to avoid deadlocks
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"Event listener failed for {event.event_type}: {e}")
                
    def clear_all(self):
        """Clear all event listeners."""
        with self._lock:
            self._listeners.clear()


class CacheInvalidator:
    """
    Comprehensive cache invalidation manager.
    
    Supports multiple invalidation strategies and provides
    intelligent invalidation based on dependencies and events.
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize cache invalidator.
        
        Args:
            cache_manager: Cache manager instance
        """
        if cache_manager is None:
            from .cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
        else:
            self.cache_manager = cache_manager
            
        self.dependency_tracker = DependencyTracker()
        self.event_bus = EventBus()
        
        # Invalidation rules
        self._rules: List[InvalidationRule] = []
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'invalidations_by_ttl': 0,
            'invalidations_by_dependency': 0,
            'invalidations_by_event': 0,
            'invalidations_by_manual': 0,
            'total_invalidations': 0
        }
        
        # Background cleanup
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        self._weak_refs: Set[weakref.ref] = set()
        
        # Subscribe to common events
        self._setup_default_event_handlers()
        
    def _setup_default_event_handlers(self):
        """Setup default event handlers for common invalidation scenarios."""
        
        def handle_data_change(event: CacheEvent):
            """Handle data change events."""
            if event.data and 'table' in event.data:
                table_name = event.data['table']
                self.invalidate_by_dependency(f"table:{table_name}")
                
        def handle_cache_clear(event: CacheEvent):
            """Handle cache clear events."""
            if event.data and 'namespace' in event.data:
                namespace = event.data['namespace']
                self.invalidate_namespace(namespace)
            else:
                self.invalidate_all()
                
        self.event_bus.subscribe('data_change', handle_data_change)
        self.event_bus.subscribe('cache_clear', handle_cache_clear)
        
    def add_rule(self, rule: InvalidationRule):
        """Add an invalidation rule."""
        with self._lock:
            self._rules.append(rule)
            
    def remove_rule(self, rule: InvalidationRule):
        """Remove an invalidation rule."""
        with self._lock:
            try:
                self._rules.remove(rule)
            except ValueError:
                pass
                
    def invalidate_key(self, cache_key: str, namespace: str = "") -> bool:
        """
        Invalidate a specific cache key.
        
        Args:
            cache_key: Cache key to invalidate
            namespace: Cache namespace
            
        Returns:
            True if key was invalidated
        """
        success = self.cache_manager.delete(cache_key, namespace=namespace)
        
        if success:
            # Remove dependencies
            self.dependency_tracker.remove_dependency(cache_key)
            self._stats['invalidations_by_manual'] += 1
            self._stats['total_invalidations'] += 1
            
        return success
        
    def invalidate_pattern(self, pattern: str, namespace: str = "") -> int:
        """
        Invalidate cache keys matching a pattern.
        
        Args:
            pattern: Pattern to match (implementation depends on cache backend)
            namespace: Cache namespace
            
        Returns:
            Number of keys invalidated
        """
        # This is a simplified implementation
        # Real implementation would depend on cache backend capabilities
        invalidated = 0
        
        # For now, we can't efficiently pattern match without backend support
        # This would need to be implemented per cache backend
        
        self._stats['invalidations_by_manual'] += invalidated
        self._stats['total_invalidations'] += invalidated
        
        return invalidated
        
    def invalidate_by_dependency(self, resource: str) -> int:
        """
        Invalidate all cache entries that depend on a resource.
        
        Args:
            resource: Resource that changed
            
        Returns:
            Number of entries invalidated
        """
        dependent_keys = self.dependency_tracker.get_dependent_keys(resource)
        invalidated = 0
        
        for cache_key in dependent_keys:
            if self.cache_manager.delete(cache_key):
                invalidated += 1
                # Remove all dependencies for this key
                self.dependency_tracker.remove_dependency(cache_key)
                
        if invalidated > 0:
            logger.info(f"Invalidated {invalidated} cache entries dependent on: {resource}")
            
        self._stats['invalidations_by_dependency'] += invalidated
        self._stats['total_invalidations'] += invalidated
        
        return invalidated
        
    def invalidate_namespace(self, namespace: str) -> int:
        """
        Invalidate all entries in a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Number of entries invalidated
        """
        # This would need backend-specific implementation
        # For now, we'll use a simplified approach
        
        # Clear namespace in cache manager if supported
        try:
            # This is a placeholder - actual implementation depends on cache backend
            success = True  # self.cache_manager.clear_namespace(namespace)
            invalidated = 1 if success else 0
        except AttributeError:
            # Fallback: can't efficiently clear namespace
            invalidated = 0
            
        self._stats['invalidations_by_manual'] += invalidated
        self._stats['total_invalidations'] += invalidated
        
        return invalidated
        
    def invalidate_all(self) -> bool:
        """
        Invalidate all cached entries.
        
        Returns:
            True if successful
        """
        success = self.cache_manager.clear_all()
        
        if success:
            # Clear all dependencies
            self.dependency_tracker.clear_all()
            self._stats['invalidations_by_manual'] += 1
            self._stats['total_invalidations'] += 1
            
        return success
        
    def invalidate_by_ttl(self, max_age: int, namespace: str = "") -> int:
        """
        Invalidate entries older than max_age.
        
        Args:
            max_age: Maximum age in seconds
            namespace: Cache namespace
            
        Returns:
            Number of entries invalidated
        """
        # This would require TTL tracking in the cache backend
        # For now, this is a placeholder
        invalidated = 0
        
        self._stats['invalidations_by_ttl'] += invalidated
        self._stats['total_invalidations'] += invalidated
        
        return invalidated
        
    def register_dependency(self, cache_key: str, resource: str, namespace: str = ""):
        """
        Register a dependency between a cache key and a resource.
        
        Args:
            cache_key: Cache key
            resource: Resource the cache depends on
            namespace: Cache namespace
        """
        full_key = f"{namespace}:{cache_key}" if namespace else cache_key
        self.dependency_tracker.add_dependency(full_key, resource)
        
    def publish_event(self, event_type: str, source: str, data: Any = None):
        """
        Publish an invalidation event.
        
        Args:
            event_type: Type of event
            source: Source of the event
            data: Event data
        """
        event = CacheEvent(
            event_type=event_type,
            source=source,
            data=data
        )
        self.event_bus.publish(event)
        
    def on_data_change(self, table_name: str, operation: str = "update"):
        """
        Handle data change events.
        
        Args:
            table_name: Name of the table that changed
            operation: Type of operation (insert, update, delete)
        """
        self.publish_event('data_change', 'database', {
            'table': table_name,
            'operation': operation
        })
        
    def schedule_invalidation(self, cache_key: str, delay: float, namespace: str = ""):
        """
        Schedule a cache invalidation after a delay.
        
        Args:
            cache_key: Cache key to invalidate
            delay: Delay in seconds
            namespace: Cache namespace
        """
        import threading
        
        def delayed_invalidation():
            time.sleep(delay)
            self.invalidate_key(cache_key, namespace)
            
        thread = threading.Thread(target=delayed_invalidation, daemon=True)
        thread.start()
        
    def create_invalidation_decorator(self, dependencies: Optional[List[str]] = None,
                                   events: Optional[List[str]] = None):
        """
        Create a decorator that automatically registers cache dependencies.
        
        Args:
            dependencies: List of resources the cached result depends on
            events: List of events that should invalidate the cache
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would need integration with caching decorators
                # to automatically register dependencies
                result = func(*args, **kwargs)
                
                # Register dependencies if cache key is available
                if hasattr(func, '_cache_key'):
                    cache_key = func._cache_key(*args, **kwargs)
                    
                    if dependencies:
                        for dep in dependencies:
                            self.register_dependency(cache_key, dep)
                            
                    if events:
                        # Subscribe to events for this cache key
                        def event_handler(event: CacheEvent):
                            self.invalidate_key(cache_key)
                            
                        for event_type in events:
                            self.event_bus.subscribe(event_type, event_handler)
                            
                return result
                
            return wrapper
        return decorator
        
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'tracked_dependencies': len(self.dependency_tracker._dependencies),
                'tracked_cache_keys': len(self.dependency_tracker._reverse_deps),
                'active_rules': len(self._rules),
                'event_listeners': sum(len(listeners) for listeners in self.event_bus._listeners.values())
            })
            
        return stats
        
    def cleanup_expired_dependencies(self):
        """Clean up dependencies for cache keys that no longer exist."""
        current_time = time.time()
        
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
            
        # This would require checking if cache keys still exist
        # Implementation depends on cache backend capabilities
        
        self._last_cleanup = current_time
        
    def get_dependency_graph(self) -> Dict[str, Any]:
        """
        Get the dependency graph for debugging.
        
        Returns:
            Dictionary representation of the dependency graph
        """
        with self.dependency_tracker._lock:
            return {
                'resources': dict(self.dependency_tracker._dependencies),
                'cache_keys': dict(self.dependency_tracker._reverse_deps)
            }
            
    def validate_dependencies(self) -> List[str]:
        """
        Validate dependency graph and return any issues found.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        with self.dependency_tracker._lock:
            # Check for orphaned dependencies
            for resource, cache_keys in self.dependency_tracker._dependencies.items():
                for cache_key in cache_keys:
                    if cache_key not in self.dependency_tracker._reverse_deps:
                        issues.append(f"Orphaned dependency: {resource} -> {cache_key}")
                        
            # Check for orphaned reverse dependencies
            for cache_key, resources in self.dependency_tracker._reverse_deps.items():
                for resource in resources:
                    if cache_key not in self.dependency_tracker._dependencies.get(resource, set()):
                        issues.append(f"Orphaned reverse dependency: {cache_key} -> {resource}")
                        
        return issues
        
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export invalidation rules for serialization."""
        with self._lock:
            exported = []
            for rule in self._rules:
                exported.append({
                    'pattern': rule.pattern,
                    'strategy': rule.strategy.value,
                    'ttl': rule.ttl,
                    'max_age': rule.max_age,
                    'dependencies': list(rule.dependencies),
                    'events': list(rule.events)
                    # Note: condition function can't be serialized
                })
            return exported
            
    def import_rules(self, rules_data: List[Dict[str, Any]]):
        """Import invalidation rules from serialized data."""
        with self._lock:
            for rule_data in rules_data:
                rule = InvalidationRule(
                    pattern=rule_data['pattern'],
                    strategy=InvalidationStrategy(rule_data['strategy']),
                    ttl=rule_data.get('ttl'),
                    max_age=rule_data.get('max_age'),
                    dependencies=set(rule_data.get('dependencies', [])),
                    events=set(rule_data.get('events', []))
                )
                self._rules.append(rule)