"""
Disk Cache - Persistent Local Caching

Provides persistent disk-based caching with compression, encryption,
and efficient file management using diskcache or fallback implementation.
"""

import os
import time
import threading
import hashlib
import shutil
import tempfile
import sqlite3
from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
import logging
import json
import pickle

# Optional imports with graceful fallbacks
try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    diskcache = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    try:
        import lz4
        LZ4_AVAILABLE = True
    except ImportError:
        LZ4_AVAILABLE = False
        lz4 = None

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)


class SimpleDiskCache:
    """
    Simple disk cache implementation using SQLite and file storage.
    
    Used as fallback when diskcache is not available.
    """
    
    def __init__(self, directory: str, size_limit: int = 1024 * 1024 * 1024):  # 1GB
        self.directory = Path(directory)
        self.size_limit = size_limit
        self._lock = threading.RLock()
        
        # Create directories
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.directory / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.directory / "cache.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl REAL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")
            conn.commit()
            
    def _hash_key(self, key: str) -> str:
        """Generate filename from key."""
        return hashlib.sha256(key.encode()).hexdigest()
        
    def _get_file_path(self, filename: str) -> Path:
        """Get full file path."""
        return self.data_dir / filename
        
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Find expired entries
            cursor = conn.execute(
                "SELECT key, filename FROM cache_entries WHERE ttl IS NOT NULL AND (timestamp + ttl) < ?",
                (current_time,)
            )
            expired = cursor.fetchall()
            
            # Remove files and database entries
            for key, filename in expired:
                file_path = self._get_file_path(filename)
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
                    
            if expired:
                conn.execute(
                    "DELETE FROM cache_entries WHERE ttl IS NOT NULL AND (timestamp + ttl) < ?",
                    (current_time,)
                )
                conn.commit()
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            self._cleanup_expired()
            
            filename = self._hash_key(key)
            file_path = self._get_file_path(filename)
            
            if not file_path.exists():
                return default
                
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT ttl, timestamp FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return default
                        
                    ttl, timestamp = row
                    current_time = time.time()
                    
                    # Check if expired
                    if ttl is not None and (current_time - timestamp) > ttl:
                        self._delete_entry(conn, key, filename)
                        return default
                        
                    # Update access count
                    conn.execute(
                        "UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?",
                        (key,)
                    )
                    conn.commit()
                    
                # Read file
                with open(file_path, 'rb') as f:
                    data = f.read()
                    
                return pickle.loads(data)
                
            except Exception as e:
                logger.warning(f"Disk cache get failed for {key}: {e}")
                return default
                
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            try:
                filename = self._hash_key(key)
                file_path = self._get_file_path(filename)
                
                # Serialize data
                data = pickle.dumps(value)
                size = len(data)
                
                # Check size limit
                if size > self.size_limit:
                    return False
                    
                # Write to temporary file first
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    f.write(data)
                    
                # Atomic move
                temp_path.replace(file_path)
                
                # Update database
                current_time = time.time()
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache_entries (key, filename, size, timestamp, ttl) VALUES (?, ?, ?, ?, ?)",
                        (key, filename, size, current_time, ttl)
                    )
                    conn.commit()
                    
                return True
                
            except Exception as e:
                logger.warning(f"Disk cache set failed for {key}: {e}")
                return False
                
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            try:
                filename = self._hash_key(key)
                
                with sqlite3.connect(str(self.db_path)) as conn:
                    return self._delete_entry(conn, key, filename)
                    
            except Exception as e:
                logger.warning(f"Disk cache delete failed for {key}: {e}")
                return False
                
    def _delete_entry(self, conn, key: str, filename: str) -> bool:
        """Delete entry from database and file system."""
        cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        
        if deleted:
            file_path = self._get_file_path(filename)
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
            conn.commit()
            
        return deleted
        
    def clear(self):
        """Clear all entries."""
        with self._lock:
            try:
                # Remove all files
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                    self.data_dir.mkdir()
                    
                # Clear database
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                    
            except Exception as e:
                logger.warning(f"Disk cache clear failed: {e}")
                
    def get_size(self) -> int:
        """Get total cache size in bytes."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT SUM(size) FROM cache_entries")
                result = cursor.fetchone()
                return result[0] or 0
        except Exception:
            return 0
            
    def get_count(self) -> int:
        """Get number of entries."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                result = cursor.fetchone()
                return result[0] or 0
        except Exception:
            return 0


class DiskCache:
    """
    Advanced disk cache with compression, encryption, and efficient storage.
    
    Uses diskcache if available, otherwise falls back to simple implementation.
    """
    
    def __init__(self,
                 directory: str,
                 size_limit: int = 1024 * 1024 * 1024,  # 1GB
                 enable_compression: bool = True,
                 enable_encryption: bool = False,
                 encryption_key: Optional[bytes] = None):
        """
        Initialize disk cache.
        
        Args:
            directory: Cache directory path
            size_limit: Maximum cache size in bytes
            enable_compression: Enable LZ4 compression
            enable_encryption: Enable encryption (requires cryptography)
            encryption_key: Encryption key (generated if None)
        """
        self.directory = directory
        self.size_limit = size_limit
        self.enable_compression = enable_compression and LZ4_AVAILABLE
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        
        # Encryption setup
        self._cipher = None
        if self.enable_encryption:
            if encryption_key:
                self._cipher = Fernet(encryption_key)
            else:
                key = Fernet.generate_key()
                self._cipher = Fernet(key)
                # Save key to file for persistence
                key_file = Path(directory) / ".encryption_key"
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                    
        # Initialize cache backend
        if DISKCACHE_AVAILABLE:
            self._init_diskcache()
        else:
            self._init_simple_cache()
            logger.info("Using simple disk cache (diskcache not available)")
            
    def _init_diskcache(self):
        """Initialize using diskcache library."""
        try:
            self._cache = diskcache.Cache(
                directory=self.directory,
                size_limit=self.size_limit,
                disk_min_file_size=1024  # Use disk for files > 1KB
            )
            self._use_diskcache = True
            logger.info(f"Disk cache initialized with diskcache: {self.directory}")
        except Exception as e:
            logger.warning(f"Failed to initialize diskcache: {e}, using simple cache")
            self._init_simple_cache()
            
    def _init_simple_cache(self):
        """Initialize using simple cache implementation."""
        self._cache = SimpleDiskCache(self.directory, self.size_limit)
        self._use_diskcache = False
        
    def _process_value(self, value: Any, encode: bool = True) -> Any:
        """Process value for storage (compression/encryption) or retrieval."""
        if encode:
            # Serialize
            if MSGPACK_AVAILABLE:
                try:
                    data = msgpack.packb(value, use_bin_type=True)
                except Exception:
                    data = pickle.dumps(value)
            else:
                data = pickle.dumps(value)
                
            # Compress
            if self.enable_compression:
                try:
                    data = lz4.compress(data)
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")
                    
            # Encrypt
            if self.enable_encryption and self._cipher:
                try:
                    data = self._cipher.encrypt(data)
                except Exception as e:
                    logger.warning(f"Encryption failed: {e}")
                    
            return data
        else:
            # Decode process (reverse order)
            data = value
            
            # Decrypt
            if self.enable_encryption and self._cipher:
                try:
                    data = self._cipher.decrypt(data)
                except Exception as e:
                    logger.warning(f"Decryption failed: {e}")
                    raise
                    
            # Decompress
            if self.enable_compression:
                try:
                    data = lz4.decompress(data)
                except Exception as e:
                    logger.warning(f"Decompression failed: {e}")
                    raise
                    
            # Deserialize
            if MSGPACK_AVAILABLE:
                try:
                    return msgpack.unpackb(data, raw=False)
                except Exception:
                    pass
                    
            return pickle.loads(data)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        try:
            with self._lock:
                if self._use_diskcache:
                    raw_value = self._cache.get(key)
                else:
                    raw_value = self._cache.get(key)
                    
                if raw_value is None:
                    self._misses += 1
                    return default
                    
                if self.enable_compression or self.enable_encryption:
                    value = self._process_value(raw_value, encode=False)
                else:
                    value = raw_value
                    
                self._hits += 1
                return value
                
        except Exception as e:
            logger.warning(f"Disk cache get failed for {key}: {e}")
            self._misses += 1
            return default
            
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        try:
            with self._lock:
                if self.enable_compression or self.enable_encryption:
                    processed_value = self._process_value(value, encode=True)
                else:
                    processed_value = value
                    
                if self._use_diskcache:
                    if ttl:
                        success = self._cache.set(key, processed_value, expire=ttl)
                    else:
                        success = self._cache.set(key, processed_value)
                else:
                    success = self._cache.set(key, processed_value, ttl)
                    
                if success:
                    self._sets += 1
                    
                return success
                
        except Exception as e:
            logger.warning(f"Disk cache set failed for {key}: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            with self._lock:
                if self._use_diskcache:
                    success = self._cache.delete(key)
                else:
                    success = self._cache.delete(key)
                    
                if success:
                    self._deletes += 1
                    
                return success
                
        except Exception as e:
            logger.warning(f"Disk cache delete failed for {key}: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            with self._lock:
                if self._use_diskcache:
                    return key in self._cache
                else:
                    return self._cache.get(key) is not None
                    
        except Exception as e:
            logger.warning(f"Disk cache exists failed for {key}: {e}")
            return False
            
    def clear(self):
        """Clear all entries from cache."""
        try:
            with self._lock:
                self._cache.clear()
                
        except Exception as e:
            logger.warning(f"Disk cache clear failed: {e}")
            
    def get_size(self) -> int:
        """Get current cache size in bytes."""
        try:
            with self._lock:
                if self._use_diskcache:
                    return self._cache.volume()
                else:
                    return self._cache.get_size()
                    
        except Exception:
            return 0
            
    def get_count(self) -> int:
        """Get number of entries in cache."""
        try:
            with self._lock:
                if self._use_diskcache:
                    # diskcache doesn't have direct count, iterate
                    return len(list(self._cache.iterkeys()))
                else:
                    return self._cache.get_count()
                    
        except Exception:
            return 0
            
    def keys(self) -> Iterator[str]:
        """Get iterator over all keys."""
        try:
            with self._lock:
                if self._use_diskcache:
                    return iter(self._cache.iterkeys())
                else:
                    # Simple cache doesn't have iterkeys, implement it
                    return iter([])
                    
        except Exception:
            return iter([])
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        stats = {
            'backend': 'diskcache' if self._use_diskcache else 'simple',
            'directory': self.directory,
            'size': self.get_size(),
            'count': self.get_count(),
            'size_limit': self.size_limit,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'sets': self._sets,
            'deletes': self._deletes,
            'compression_enabled': self.enable_compression,
            'encryption_enabled': self.enable_encryption
        }
        
        # Add diskcache-specific stats
        if self._use_diskcache:
            try:
                cache_stats = self._cache.stats()
                stats.update({
                    'cache_hits': cache_stats[0],
                    'cache_misses': cache_stats[1]
                })
            except Exception:
                pass
                
        return stats
        
    def expire(self, key: str, ttl: float) -> bool:
        """Set TTL for existing key."""
        try:
            with self._lock:
                if self._use_diskcache:
                    return self._cache.touch(key, expire=ttl)
                else:
                    # Simple cache doesn't support touch, need to get and set
                    value = self._cache.get(key)
                    if value is not None:
                        return self._cache.set(key, value, ttl)
                    return False
                    
        except Exception as e:
            logger.warning(f"Disk cache expire failed for {key}: {e}")
            return False
            
    def vacuum(self) -> bool:
        """Optimize cache storage."""
        try:
            with self._lock:
                if self._use_diskcache:
                    # diskcache automatically manages storage
                    self._cache.cull()
                    return True
                else:
                    # For simple cache, just cleanup expired entries
                    self._cache._cleanup_expired()
                    return True
                    
        except Exception as e:
            logger.warning(f"Disk cache vacuum failed: {e}")
            return False
            
    def close(self):
        """Close cache and cleanup resources."""
        try:
            if hasattr(self._cache, 'close'):
                self._cache.close()
        except Exception:
            pass