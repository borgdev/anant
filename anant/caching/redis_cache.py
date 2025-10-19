"""
Redis Cache - Distributed Caching with Redis

Provides distributed caching using Redis with connection pooling,
serialization, and graceful fallbacks when Redis is unavailable.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
import json
import pickle

# Optional imports with graceful fallbacks
try:
    import redis
    from redis.connection import ConnectionPool
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionError = Exception
    TimeoutError = Exception
    RedisError = Exception

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration parameters."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    health_check_interval: int = 30
    max_connections: int = 50
    retry_on_timeout: bool = True
    decode_responses: bool = False
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


class RedisCache:
    """
    Redis-based distributed cache with advanced features.
    
    Features:
    - Connection pooling and health checking
    - Multiple serialization formats (msgpack, pickle, json)
    - Automatic retry and circuit breaker pattern
    - Batch operations for performance
    - Key namespacing and pattern matching
    - TTL and memory usage monitoring
    """
    
    def __init__(self, 
                 config: Optional[RedisConfig] = None,
                 url: Optional[str] = None,
                 key_prefix: str = "anant:",
                 serialization: str = "msgpack"):
        """
        Initialize Redis cache.
        
        Args:
            config: Redis configuration
            url: Redis URL (overrides config)
            key_prefix: Prefix for all keys
            serialization: Serialization method (msgpack, pickle, json)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
            
        self.config = config or RedisConfig()
        self.key_prefix = key_prefix
        self.serialization = serialization
        self._client = None
        self._pool = None
        self._is_available = False
        self._last_health_check = 0
        self._lock = threading.RLock()
        
        # Circuit breaker state
        self._failure_count = 0
        self._max_failures = 5
        self._failure_timeout = 60  # seconds
        self._last_failure = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._operations = 0
        
        # Initialize connection
        if url:
            self._init_from_url(url)
        else:
            self._init_from_config()
            
    def _init_from_url(self, url: str):
        """Initialize from Redis URL."""
        try:
            self._pool = ConnectionPool.from_url(
                url,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                socket_keepalive=self.config.socket_keepalive,
                health_check_interval=self.config.health_check_interval,
                decode_responses=self.config.decode_responses
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._check_connection()
            logger.info(f"Redis cache initialized from URL: {url}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis from URL {url}: {e}")
            self._client = None
            
    def _init_from_config(self):
        """Initialize from configuration."""
        try:
            pool_kwargs = {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db,
                'max_connections': self.config.max_connections,
                'retry_on_timeout': self.config.retry_on_timeout,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.socket_connect_timeout,
                'socket_keepalive': self.config.socket_keepalive,
                'health_check_interval': self.config.health_check_interval,
                'decode_responses': self.config.decode_responses
            }
            
            if self.config.password:
                pool_kwargs['password'] = self.config.password
            if self.config.username:
                pool_kwargs['username'] = self.config.username
                
            if self.config.ssl:
                pool_kwargs['ssl'] = True
                if self.config.ssl_cert_reqs:
                    pool_kwargs['ssl_cert_reqs'] = self.config.ssl_cert_reqs
                if self.config.ssl_ca_certs:
                    pool_kwargs['ssl_ca_certs'] = self.config.ssl_ca_certs
                if self.config.ssl_certfile:
                    pool_kwargs['ssl_certfile'] = self.config.ssl_certfile
                if self.config.ssl_keyfile:
                    pool_kwargs['ssl_keyfile'] = self.config.ssl_keyfile
                    
            self._pool = ConnectionPool(**pool_kwargs)
            self._client = redis.Redis(connection_pool=self._pool)
            self._check_connection()
            logger.info(f"Redis cache initialized: {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._client = None
            
    def _check_connection(self) -> bool:
        """Check if Redis connection is available."""
        if not self._client:
            return False
            
        try:
            self._client.ping()
            self._is_available = True
            self._failure_count = 0
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._is_available = False
            self._failure_count += 1
            self._last_failure = time.time()
            return False
            
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._failure_count < self._max_failures:
            return False
            
        time_since_failure = time.time() - self._last_failure
        return time_since_failure < self._failure_timeout
        
    def _maybe_health_check(self):
        """Perform health check if needed."""
        current_time = time.time()
        if (current_time - self._last_health_check > self.config.health_check_interval or
            not self._is_available):
            self._check_connection()
            
    def _full_key(self, key: str) -> str:
        """Generate full key with prefix."""
        return f"{self.key_prefix}{key}"
        
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.serialization == "msgpack" and MSGPACK_AVAILABLE:
                return msgpack.packb(value, use_bin_type=True)
            elif self.serialization == "json":
                return json.dumps(value, default=str).encode('utf-8')
            else:  # pickle as fallback
                return pickle.dumps(value)
        except Exception as e:
            logger.warning(f"Serialization failed, falling back to pickle: {e}")
            return pickle.dumps(value)
            
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.serialization == "msgpack" and MSGPACK_AVAILABLE:
                return msgpack.unpackb(data, raw=False)
            elif self.serialization == "json":
                return json.loads(data.decode('utf-8'))
            else:  # pickle as fallback
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Deserialization failed, trying pickle: {e}")
            return pickle.loads(data)
            
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute Redis operation with retry logic."""
        if not self._client or self._is_circuit_open():
            raise ConnectionError("Redis unavailable")
            
        self._maybe_health_check()
        
        if not self._is_available:
            raise ConnectionError("Redis health check failed")
            
        try:
            result = func(*args, **kwargs)
            self._operations += 1
            return result
        except (ConnectionError, TimeoutError) as e:
            self._errors += 1
            self._failure_count += 1
            self._last_failure = time.time()
            self._is_available = False
            raise e
        except Exception as e:
            self._errors += 1
            raise e
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from Redis cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                data = self._execute_with_retry(self._client.get, full_key)
                
                if data is None:
                    self._misses += 1
                    return default
                    
                value = self._deserialize(data)
                self._hits += 1
                return value
                
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            self._misses += 1
            return default
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                data = self._serialize(value)
                
                if ttl:
                    success = self._execute_with_retry(
                        self._client.setex, full_key, ttl, data
                    )
                else:
                    success = self._execute_with_retry(
                        self._client.set, full_key, data
                    )
                    
                return bool(success)
                
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Delete value from Redis cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                count = self._execute_with_retry(self._client.delete, full_key)
                return count > 0
                
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                count = self._execute_with_retry(self._client.exists, full_key)
                return count > 0
                
        except Exception as e:
            logger.warning(f"Redis exists failed for key {key}: {e}")
            return False
            
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from Redis.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}
            
        try:
            with self._lock:
                full_keys = [self._full_key(key) for key in keys]
                values = self._execute_with_retry(self._client.mget, full_keys)
                
                result = {}
                for i, (key, data) in enumerate(zip(keys, values)):
                    if data is not None:
                        try:
                            result[key] = self._deserialize(data)
                            self._hits += 1
                        except Exception as e:
                            logger.warning(f"Deserialization failed for key {key}: {e}")
                            self._misses += 1
                    else:
                        self._misses += 1
                        
                return result
                
        except Exception as e:
            logger.warning(f"Redis get_many failed: {e}")
            self._misses += len(keys)
            return {}
            
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """
        Set multiple values in Redis.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds for all keys
            
        Returns:
            Number of keys successfully set
        """
        if not mapping:
            return 0
            
        try:
            with self._lock:
                # Serialize all values
                serialized_mapping = {}
                for key, value in mapping.items():
                    full_key = self._full_key(key)
                    try:
                        serialized_mapping[full_key] = self._serialize(value)
                    except Exception as e:
                        logger.warning(f"Serialization failed for key {key}: {e}")
                        
                if not serialized_mapping:
                    return 0
                    
                # Use pipeline for batch operations
                pipe = self._client.pipeline()
                
                if ttl:
                    for full_key, data in serialized_mapping.items():
                        pipe.setex(full_key, ttl, data)
                else:
                    pipe.mset(serialized_mapping)
                    
                results = self._execute_with_retry(pipe.execute)
                
                # Count successful operations
                if ttl:
                    return sum(1 for result in results if result)
                else:
                    return len(serialized_mapping) if results[0] else 0
                    
        except Exception as e:
            logger.warning(f"Redis set_many failed: {e}")
            return 0
            
    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys from Redis.
        
        Args:
            keys: List of cache keys to delete
            
        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0
            
        try:
            with self._lock:
                full_keys = [self._full_key(key) for key in keys]
                count = self._execute_with_retry(self._client.delete, *full_keys)
                return count
                
        except Exception as e:
            logger.warning(f"Redis delete_many failed: {e}")
            return 0
            
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports * and ?)
            
        Returns:
            Number of keys deleted
        """
        try:
            with self._lock:
                full_pattern = self._full_key(pattern)
                keys = self._execute_with_retry(self._client.keys, full_pattern)
                
                if keys:
                    count = self._execute_with_retry(self._client.delete, *keys)
                    return count
                return 0
                
        except Exception as e:
            logger.warning(f"Redis clear_pattern failed for pattern {pattern}: {e}")
            return 0
            
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, None if no TTL, -1 if key doesn't exist
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                ttl = self._execute_with_retry(self._client.ttl, full_key)
                return ttl if ttl >= 0 else None
                
        except Exception as e:
            logger.warning(f"Redis get_ttl failed for key {key}: {e}")
            return None
            
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if TTL was set
        """
        try:
            with self._lock:
                full_key = self._full_key(key)
                success = self._execute_with_retry(self._client.expire, full_key, ttl)
                return bool(success)
                
        except Exception as e:
            logger.warning(f"Redis expire failed for key {key}: {e}")
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            with self._lock:
                info = self._execute_with_retry(self._client.info)
                return info
                
        except Exception as e:
            logger.warning(f"Redis get_info failed: {e}")
            return {}
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        stats = {
            'available': self._is_available,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'errors': self._errors,
            'operations': self._operations,
            'failure_count': self._failure_count,
            'circuit_open': self._is_circuit_open(),
            'last_health_check': self._last_health_check
        }
        
        # Add Redis info if available
        if self._is_available:
            try:
                info = self.get_info()
                stats.update({
                    'memory_usage': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_connections_received': info.get('total_connections_received', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception:
                pass
                
        return stats
        
    def is_available(self) -> bool:
        """Check if Redis is available."""
        self._maybe_health_check()
        return self._is_available
        
    def flush_all(self) -> bool:
        """Flush all data from current database."""
        try:
            with self._lock:
                success = self._execute_with_retry(self._client.flushdb)
                return bool(success)
                
        except Exception as e:
            logger.warning(f"Redis flush_all failed: {e}")
            return False
            
    def close(self):
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
                
        if self._pool:
            try:
                self._pool.disconnect()
            except Exception:
                pass
                
        self._client = None
        self._pool = None
        self._is_available = False