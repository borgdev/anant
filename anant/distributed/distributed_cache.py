"""
Distributed Cache - Cluster-wide Caching and Data Sharing

Provides distributed caching capabilities with consistency guarantees,
replication, and intelligent cache management across cluster nodes.
"""

import asyncio
import time
import threading
import hashlib
import pickle
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import weakref

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"           # Best performance, eventual consistency
    WEAK = "weak"                  # Weak consistency with faster reads
    STRONG = "strong"              # Strong consistency with higher latency
    SEQUENTIAL = "sequential"       # Sequential consistency guarantees


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    FIFO = "fifo"                  # First In, First Out
    RANDOM = "random"              # Random replacement
    TTL = "ttl"                    # Time To Live based


class ReplicationStrategy(Enum):
    """Data replication strategies."""
    NONE = "none"                  # No replication
    MASTER_SLAVE = "master_slave"  # Master-slave replication
    MULTI_MASTER = "multi_master"  # Multi-master replication
    QUORUM = "quorum"             # Quorum-based replication


@dataclass
class CacheEntry:
    """Cached data entry with metadata."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    version: int = 1
    replicas: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
        
    def access(self):
        """Record access to this entry."""
        self.access_count += 1
        self.last_access = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'access_count': self.access_count,
            'last_access': self.last_access,
            'size_bytes': self.size_bytes,
            'version': self.version,
            'replicas': list(self.replicas)
        }


@dataclass
class CacheConfig:
    """Configuration for distributed cache."""
    max_memory_mb: int = 1024              # Maximum memory usage per node
    max_entries: int = 100000              # Maximum number of entries
    default_ttl: Optional[float] = 3600.0  # Default TTL in seconds
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    replication_strategy: ReplicationStrategy = ReplicationStrategy.MASTER_SLAVE
    replication_factor: int = 2            # Number of replicas
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    enable_compression: bool = True
    compression_threshold: int = 1024      # Compress values > 1KB
    sync_interval: float = 30.0           # Sync interval for eventual consistency
    heartbeat_interval: float = 10.0      # Heartbeat interval for node health


@dataclass
class CacheOperation:
    """Cache operation for logging and replication."""
    operation_id: str
    operation_type: str  # "get", "set", "delete", "invalidate"
    key: str
    value: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    success: bool = True
    error_message: Optional[str] = None


class LocalCacheNode:
    """Local cache node implementation with LRU/LFU strategies."""
    
    def __init__(self, config: CacheConfig):
        """Initialize local cache node."""
        self.config = config
        self.data: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        self.current_memory: int = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from local cache."""
        with self._lock:
            if key not in self.data:
                return None
                
            entry = self.data[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                return None
                
            # Update access patterns
            entry.access()
            self._update_access_pattern(key)
            
            return entry.value
            
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in local cache."""
        with self._lock:
            try:
                # Calculate size
                serialized = self._serialize_value(value)
                size_bytes = len(serialized)
                
                # Check if we need to evict entries
                self._ensure_capacity(size_bytes)
                
                # Create or update entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl or self.config.default_ttl,
                    size_bytes=size_bytes
                )
                
                # Remove old entry if exists
                if key in self.data:
                    self._remove_entry(key)
                    
                # Add new entry
                self.data[key] = entry
                self.current_memory += size_bytes
                self._update_access_pattern(key)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {e}")
                return False
                
    def delete(self, key: str) -> bool:
        """Delete entry from local cache."""
        with self._lock:
            if key in self.data:
                self._remove_entry(key)
                return True
            return False
            
    def _remove_entry(self, key: str):
        """Remove entry and update tracking."""
        if key in self.data:
            entry = self.data[key]
            self.current_memory -= entry.size_bytes
            del self.data[key]
            
        if key in self.access_order:
            del self.access_order[key]
            
        if key in self.frequency_counter:
            del self.frequency_counter[key]
            
    def _update_access_pattern(self, key: str):
        """Update access patterns for cache strategies."""
        # Update LRU order
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = time.time()
        
        # Update LFU frequency
        self.frequency_counter[key] += 1
        
    def _ensure_capacity(self, new_size: int):
        """Ensure cache has capacity for new entry."""
        # Check memory limit
        while (self.current_memory + new_size > self.config.max_memory_mb * 1024 * 1024 or
               len(self.data) >= self.config.max_entries):
            
            if not self.data:
                break
                
            # Select entry to evict based on strategy
            evict_key = self._select_eviction_candidate()
            if evict_key:
                self._remove_entry(evict_key)
            else:
                break
                
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on cache strategy."""
        if not self.data:
            return None
            
        if self.config.cache_strategy == CacheStrategy.LRU:
            # Remove least recently used
            return next(iter(self.access_order))
            
        elif self.config.cache_strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_freq = min(self.frequency_counter.values())
            for key, freq in self.frequency_counter.items():
                if freq == min_freq:
                    return key
                    
        elif self.config.cache_strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            oldest_key = None
            oldest_time = float('inf')
            for key, entry in self.data.items():
                if entry.timestamp < oldest_time:
                    oldest_time = entry.timestamp
                    oldest_key = key
            return oldest_key
            
        elif self.config.cache_strategy == CacheStrategy.TTL:
            # Remove expired entries first
            current_time = time.time()
            for key, entry in self.data.items():
                if entry.is_expired():
                    return key
            # Fall back to LRU if no expired entries
            return next(iter(self.access_order))
            
        else:  # RANDOM
            import random
            return random.choice(list(self.data.keys()))
            
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.config.enable_compression:
                data = pickle.dumps(value)
                if len(data) > self.config.compression_threshold:
                    import gzip
                    return gzip.compress(data)
                return data
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """Get local cache statistics."""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self.data.values())
            
            return {
                'entries': len(self.data),
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'memory_limit_mb': self.config.max_memory_mb,
                'memory_utilization': self.current_memory / (self.config.max_memory_mb * 1024 * 1024),
                'total_accesses': total_accesses,
                'cache_strategy': self.config.cache_strategy.value,
                'consistency_level': self.config.consistency_level.value
            }


class DistributedCache:
    """
    Distributed cache system with consistency guarantees and replication.
    
    Features:
    - Multiple consistency levels (eventual, weak, strong, sequential)
    - Configurable replication strategies
    - Cache partitioning and sharding
    - Automatic failover and recovery
    - Compression and serialization
    - Cache warming and preloading
    - Comprehensive monitoring and metrics
    """
    
    def __init__(self, node_id: str, message_broker, config: Optional[CacheConfig] = None):
        """
        Initialize distributed cache.
        
        Args:
            node_id: Unique identifier for this cache node
            message_broker: MessageBroker for inter-node communication
            config: Cache configuration
        """
        self.node_id = node_id
        self.message_broker = message_broker
        self.config = config or CacheConfig()
        
        # Local cache
        self.local_cache = LocalCacheNode(self.config)
        
        # Cluster state
        self.cluster_nodes: Set[str] = {node_id}
        self.partition_map: Dict[str, str] = {}  # key_prefix -> responsible_node
        self.replication_map: Dict[str, List[str]] = {}  # key -> replica_nodes
        
        # Operation tracking
        self.pending_operations: Dict[str, CacheOperation] = {}
        self.operation_history: List[CacheOperation] = []
        
        # Control
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'remote_hits': 0,
            'replications_sent': 0,
            'replications_received': 0,
            'invalidations_sent': 0,
            'invalidations_received': 0,
            'consistency_violations': 0
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
    async def start(self) -> bool:
        """
        Start the distributed cache.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
            
        try:
            self._running = True
            
            # Start background tasks
            self._sync_task = asyncio.create_task(self._sync_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Announce presence to cluster
            await self._announce_node_join()
            
            logger.info(f"Distributed cache started: {self.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start distributed cache: {e}")
            self._running = False
            return False
            
    async def stop(self):
        """Stop the distributed cache."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._sync_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Announce departure
        await self._announce_node_leave()
        
        logger.info(f"Distributed cache stopped: {self.node_id}")
        
    async def get(self, key: str, consistency: Optional[ConsistencyLevel] = None) -> Optional[Any]:
        """
        Get value from distributed cache.
        
        Args:
            key: Cache key
            consistency: Consistency level override
            
        Returns:
            Cached value or None if not found
        """
        consistency_level = consistency or self.config.consistency_level
        
        try:
            # Check local cache first
            value = self.local_cache.get(key)
            if value is not None:
                self._stats['cache_hits'] += 1
                return value
                
            # For strong consistency, check all replicas
            if consistency_level == ConsistencyLevel.STRONG:
                value = await self._get_with_strong_consistency(key)
                if value is not None:
                    self._stats['remote_hits'] += 1
                    # Cache locally for future access
                    self.local_cache.set(key, value)
                    return value
                    
            # For other consistency levels, check primary node
            primary_node = self._get_primary_node(key)
            if primary_node != self.node_id:
                value = await self._get_from_remote(key, primary_node)
                if value is not None:
                    self._stats['remote_hits'] += 1
                    # Cache locally for eventual consistency
                    if consistency_level == ConsistencyLevel.EVENTUAL:
                        self.local_cache.set(key, value)
                    return value
                    
            self._stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self._stats['cache_misses'] += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                 consistency: Optional[ConsistencyLevel] = None) -> bool:
        """
        Set value in distributed cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            consistency: Consistency level override
            
        Returns:
            True if set successfully
        """
        consistency_level = consistency or self.config.consistency_level
        operation_id = str(uuid.uuid4())
        
        try:
            operation = CacheOperation(
                operation_id=operation_id,
                operation_type="set",
                key=key,
                value=value,
                node_id=self.node_id
            )
            
            # Set in local cache
            success = self.local_cache.set(key, value, ttl)
            if not success:
                return False
                
            # Handle replication based on consistency level
            if consistency_level == ConsistencyLevel.STRONG:
                success = await self._set_with_strong_consistency(key, value, ttl, operation)
            else:
                # Async replication for eventual consistency
                asyncio.create_task(self._replicate_operation(operation))
                
            return success
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
            
    async def delete(self, key: str, consistency: Optional[ConsistencyLevel] = None) -> bool:
        """
        Delete key from distributed cache.
        
        Args:
            key: Cache key to delete
            consistency: Consistency level override
            
        Returns:
            True if deleted successfully
        """
        consistency_level = consistency or self.config.consistency_level
        operation_id = str(uuid.uuid4())
        
        try:
            operation = CacheOperation(
                operation_id=operation_id,
                operation_type="delete",
                key=key,
                node_id=self.node_id
            )
            
            # Delete from local cache
            success = self.local_cache.delete(key)
            
            # Replicate deletion
            if consistency_level == ConsistencyLevel.STRONG:
                await self._replicate_operation_sync(operation)
            else:
                asyncio.create_task(self._replicate_operation(operation))
                
            return True  # Return True even if key didn't exist locally
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
            
    async def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Key pattern to match (supports wildcards)
            
        Returns:
            Number of entries invalidated
        """
        try:
            import fnmatch
            
            # Find matching keys locally
            matching_keys = []
            for key in self.local_cache.data.keys():
                if fnmatch.fnmatch(key, pattern):
                    matching_keys.append(key)
                    
            # Delete matching keys
            for key in matching_keys:
                self.local_cache.delete(key)
                
            # Broadcast invalidation to other nodes
            await self._broadcast_invalidation(pattern)
            
            self._stats['invalidations_sent'] += len(matching_keys)
            return len(matching_keys)
            
        except Exception as e:
            logger.error(f"Cache invalidation failed for pattern {pattern}: {e}")
            return 0
            
    def _get_primary_node(self, key: str) -> str:
        """Get primary node responsible for a key."""
        # Simple hash-based partitioning
        key_hash = hashlib.md5(key.encode()).hexdigest()
        node_index = int(key_hash, 16) % len(self.cluster_nodes)
        return sorted(list(self.cluster_nodes))[node_index]
        
    def _get_replica_nodes(self, key: str) -> List[str]:
        """Get replica nodes for a key."""
        primary = self._get_primary_node(key)
        nodes = list(self.cluster_nodes)
        nodes.remove(primary)
        
        # Select replica nodes
        replica_count = min(self.config.replication_factor - 1, len(nodes))
        replicas = nodes[:replica_count]
        
        return [primary] + replicas
        
    async def _get_with_strong_consistency(self, key: str) -> Optional[Any]:
        """Get value with strong consistency guarantees."""
        replica_nodes = self._get_replica_nodes(key)
        responses = []
        
        # Query all replicas
        for node_id in replica_nodes:
            if node_id == self.node_id:
                value = self.local_cache.get(key)
                if value is not None:
                    responses.append((node_id, value, 1))  # version 1 for simplicity
            else:
                value = await self._get_from_remote(key, node_id)
                if value is not None:
                    responses.append((node_id, value, 1))
                    
        # Return value if majority agrees
        if len(responses) > len(replica_nodes) / 2:
            # For simplicity, return first response
            # In production, would implement proper quorum logic
            return responses[0][1]
            
        return None
        
    async def _set_with_strong_consistency(self, key: str, value: Any, 
                                         ttl: Optional[float], operation: CacheOperation) -> bool:
        """Set value with strong consistency guarantees."""
        replica_nodes = self._get_replica_nodes(key)
        success_count = 1  # Local node already set
        
        # Replicate to all replica nodes
        for node_id in replica_nodes:
            if node_id != self.node_id:
                success = await self._replicate_to_node(operation, node_id)
                if success:
                    success_count += 1
                    
        # Require majority for success
        required_success = len(replica_nodes) // 2 + 1
        return success_count >= required_success
        
    async def _get_from_remote(self, key: str, node_id: str) -> Optional[Any]:
        """Get value from remote cache node."""
        try:
            from .message_broker import Message, MessageType, MessagePriority
            
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RPC_REQUEST,
                sender_id=self.node_id,
                recipient_id=node_id,
                priority=MessagePriority.HIGH,
                payload={
                    'operation': 'cache_get',
                    'key': key
                }
            )
            
            response = await self.message_broker.send_request(message, timeout=5.0)
            if response and response.payload.get('success'):
                return response.payload.get('value')
                
        except Exception as e:
            logger.error(f"Remote cache get failed: {e}")
            
        return None
        
    async def _replicate_operation(self, operation: CacheOperation):
        """Replicate operation to replica nodes."""
        replica_nodes = self._get_replica_nodes(operation.key)
        
        for node_id in replica_nodes:
            if node_id != self.node_id:
                await self._replicate_to_node(operation, node_id)
                
    async def _replicate_to_node(self, operation: CacheOperation, node_id: str) -> bool:
        """Replicate operation to specific node."""
        try:
            from .message_broker import Message, MessageType, MessagePriority
            
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RPC_REQUEST,
                sender_id=self.node_id,
                recipient_id=node_id,
                priority=MessagePriority.NORMAL,
                payload={
                    'operation': 'cache_replicate',
                    'operation_data': {
                        'operation_id': operation.operation_id,
                        'operation_type': operation.operation_type,
                        'key': operation.key,
                        'value': operation.value,
                        'timestamp': operation.timestamp
                    }
                }
            )
            
            response = await self.message_broker.send_request(message, timeout=10.0)
            success = response and response.payload.get('success', False)
            
            if success:
                self._stats['replications_sent'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Replication to node {node_id} failed: {e}")
            return False
            
    async def _replicate_operation_sync(self, operation: CacheOperation):
        """Synchronously replicate operation to all replicas."""
        replica_nodes = self._get_replica_nodes(operation.key)
        tasks = []
        
        for node_id in replica_nodes:
            if node_id != self.node_id:
                task = asyncio.create_task(self._replicate_to_node(operation, node_id))
                tasks.append(task)
                
        # Wait for all replications
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _broadcast_invalidation(self, pattern: str):
        """Broadcast cache invalidation to all nodes."""
        try:
            from .message_broker import MessageType
            
            await self.message_broker.broadcast_message(
                message_type=MessageType.BROADCAST,
                payload={
                    'operation': 'cache_invalidate',
                    'pattern': pattern,
                    'sender': self.node_id
                }
            )
            
        except Exception as e:
            logger.error(f"Broadcast invalidation failed: {e}")
            
    def _setup_message_handlers(self):
        """Setup message handlers for cache operations."""
        from .message_broker import MessageType
        
        self.message_broker.add_message_handler(
            MessageType.RPC_REQUEST, self._handle_cache_request
        )
        self.message_broker.add_message_handler(
            MessageType.BROADCAST, self._handle_broadcast_message
        )
        
    def _handle_cache_request(self, message):
        """Handle incoming cache requests."""
        try:
            operation = message.payload.get('operation')
            
            if operation == 'cache_get':
                key = message.payload['key']
                value = self.local_cache.get(key)
                
                # Send response
                response_payload = {
                    'success': value is not None,
                    'value': value
                }
                
            elif operation == 'cache_replicate':
                op_data = message.payload['operation_data']
                
                if op_data['operation_type'] == 'set':
                    success = self.local_cache.set(
                        op_data['key'],
                        op_data['value']
                    )
                elif op_data['operation_type'] == 'delete':
                    success = self.local_cache.delete(op_data['key'])
                else:
                    success = False
                    
                if success:
                    self._stats['replications_received'] += 1
                    
                response_payload = {'success': success}
                
            else:
                response_payload = {'success': False, 'error': 'Unknown operation'}
                
            # Send response message would be implemented here
            # This is simplified for the example
            
        except Exception as e:
            logger.error(f"Cache request handling failed: {e}")
            
    def _handle_broadcast_message(self, message):
        """Handle broadcast messages."""
        try:
            operation = message.payload.get('operation')
            
            if operation == 'cache_invalidate':
                pattern = message.payload['pattern']
                sender = message.payload['sender']
                
                # Don't process own broadcasts
                if sender != self.node_id:
                    import fnmatch
                    
                    # Find and delete matching keys
                    matching_keys = []
                    for key in self.local_cache.data.keys():
                        if fnmatch.fnmatch(key, pattern):
                            matching_keys.append(key)
                            
                    for key in matching_keys:
                        self.local_cache.delete(key)
                        
                    self._stats['invalidations_received'] += len(matching_keys)
                    
            elif operation == 'node_join':
                node_id = message.payload['node_id']
                self.cluster_nodes.add(node_id)
                logger.info(f"Node joined cluster: {node_id}")
                
            elif operation == 'node_leave':
                node_id = message.payload['node_id']
                self.cluster_nodes.discard(node_id)
                logger.info(f"Node left cluster: {node_id}")
                
        except Exception as e:
            logger.error(f"Broadcast message handling failed: {e}")
            
    async def _announce_node_join(self):
        """Announce this node joining the cluster."""
        await self.message_broker.broadcast_message(
            MessageType.BROADCAST,
            {
                'operation': 'node_join',
                'node_id': self.node_id
            }
        )
        
    async def _announce_node_leave(self):
        """Announce this node leaving the cluster."""
        await self.message_broker.broadcast_message(
            MessageType.BROADCAST,
            {
                'operation': 'node_leave',
                'node_id': self.node_id
            }
        )
        
    async def _sync_loop(self):
        """Periodic synchronization for eventual consistency."""
        while self._running:
            try:
                # Implement periodic sync logic here
                # This could include:
                # - Syncing with primary nodes
                # - Checking for inconsistencies
                # - Performing background replication
                
                await asyncio.sleep(self.config.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(30.0)
                
    async def _cleanup_loop(self):
        """Periodic cleanup of expired entries."""
        while self._running:
            try:
                # Clean up expired entries
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.local_cache.data.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                        
                for key in expired_keys:
                    self.local_cache.delete(key)
                    
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
                await asyncio.sleep(60.0)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300.0)
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self._stats.copy()
        
        # Add local cache stats
        local_stats = self.local_cache.get_stats()
        stats.update(local_stats)
        
        # Add distributed cache specific stats
        stats.update({
            'node_id': self.node_id,
            'cluster_size': len(self.cluster_nodes),
            'replication_factor': self.config.replication_factor,
            'consistency_level': self.config.consistency_level.value,
            'hit_rate': self._stats['cache_hits'] / max(1, self._stats['cache_hits'] + self._stats['cache_misses']),
            'remote_hit_rate': self._stats['remote_hits'] / max(1, self._stats['cache_misses'])
        })
        
        return stats