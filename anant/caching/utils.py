"""
Cache Utilities - Serialization, Key Generation, and Helper Functions

Provides utility functions for cache key generation, serialization,
compression, and other common caching operations.
"""

import hashlib
import json
import pickle
import time
import gzip
import base64
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging

# Optional imports with graceful fallbacks
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    orjson = None

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

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""
    pass


class KeyGenerator:
    """
    Advanced cache key generator with multiple strategies.
    """
    
    def __init__(self, hash_algorithm: str = "xxhash"):
        """
        Initialize key generator.
        
        Args:
            hash_algorithm: Hash algorithm (xxhash, sha256, md5)
        """
        self.hash_algorithm = hash_algorithm
        
    def _hash_string(self, data: str) -> str:
        """Hash a string using the configured algorithm."""
        if self.hash_algorithm == "xxhash" and XXHASH_AVAILABLE:
            return xxhash.xxh64(data.encode()).hexdigest()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        elif self.hash_algorithm == "md5":
            return hashlib.md5(data.encode()).hexdigest()
        else:
            # Fallback to SHA256
            return hashlib.sha256(data.encode()).hexdigest()[:16]
            
    def generate_key(self, *args, prefix: str = "", 
                    include_types: bool = False, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            prefix: Key prefix
            include_types: Include type information in key
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        key_parts = []
        
        if prefix:
            key_parts.append(prefix)
            
        # Process positional arguments
        for arg in args:
            if include_types:
                type_name = type(arg).__name__
                key_parts.append(f"{type_name}:")
                
            key_parts.append(self._serialize_for_key(arg))
            
        # Process keyword arguments
        for k, v in sorted(kwargs.items()):
            if include_types:
                type_name = type(v).__name__
                key_parts.append(f"{k}:{type_name}:")
            else:
                key_parts.append(f"{k}:")
                
            key_parts.append(self._serialize_for_key(v))
            
        key_string = "|".join(key_parts)
        return self._hash_string(key_string)
        
    def _serialize_for_key(self, obj: Any) -> str:
        """Serialize object for use in cache key."""
        if obj is None:
            return "None"
        elif isinstance(obj, (str, int, float, bool)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return f"[{','.join(self._serialize_for_key(item) for item in obj)}]"
        elif isinstance(obj, dict):
            items = [f"{k}:{self._serialize_for_key(v)}" 
                    for k, v in sorted(obj.items())]
            return f"{{{','.join(items)}}}"
        elif hasattr(obj, '__dict__'):
            # For objects, use class name and key attributes
            class_name = obj.__class__.__name__
            if hasattr(obj, '__cache_key__'):
                return f"{class_name}:{obj.__cache_key__()}"
            else:
                return f"{class_name}:{id(obj)}"
        else:
            # Fallback: convert to string
            return str(obj)
            
    def function_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate key for function call."""
        func_name = f"{func.__module__}.{func.__qualname__}"
        return self.generate_key(*args, prefix=func_name, **kwargs)
        
    def method_key(self, instance: Any, method_name: str, 
                  *args, **kwargs) -> str:
        """Generate key for method call."""
        class_name = instance.__class__.__name__
        instance_key = getattr(instance, '__cache_key__', lambda: id(instance))()
        prefix = f"{class_name}.{method_name}:{instance_key}"
        return self.generate_key(*args, prefix=prefix, **kwargs)


class Serializer:
    """
    Advanced serialization with multiple formats and compression.
    """
    
    def __init__(self, 
                 format: str = "msgpack",
                 compression: str = "none",
                 compression_level: int = 6):
        """
        Initialize serializer.
        
        Args:
            format: Serialization format (msgpack, pickle, json, orjson)
            compression: Compression method (none, gzip, lz4)
            compression_level: Compression level (1-9 for gzip)
        """
        self.format = format
        self.compression = compression
        self.compression_level = compression_level
        
        # Validate format availability
        if format == "msgpack" and not MSGPACK_AVAILABLE:
            logger.warning("msgpack not available, falling back to pickle")
            self.format = "pickle"
        elif format == "orjson" and not ORJSON_AVAILABLE:
            logger.warning("orjson not available, falling back to json")
            self.format = "json"
            
        # Validate compression availability
        if compression == "lz4" and not LZ4_AVAILABLE:
            logger.warning("lz4 not available, falling back to gzip")
            self.compression = "gzip"
            
    def serialize(self, obj: Any) -> bytes:
        """
        Serialize object to bytes.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized bytes
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            # Serialize to bytes
            if self.format == "msgpack":
                data = msgpack.packb(obj, use_bin_type=True)
            elif self.format == "pickle":
                data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.format == "json":
                json_str = json.dumps(obj, default=self._json_default, 
                                    separators=(',', ':'))
                data = json_str.encode('utf-8')
            elif self.format == "orjson":
                data = orjson.dumps(obj, default=self._json_default)
            else:
                raise SerializationError(f"Unknown format: {self.format}")
                
            # Apply compression
            if self.compression == "gzip":
                data = gzip.compress(data, compresslevel=self.compression_level)
            elif self.compression == "lz4":
                data = lz4.compress(data, compression_level=self.compression_level)
            elif self.compression != "none":
                raise SerializationError(f"Unknown compression: {self.compression}")
                
            return data
            
        except Exception as e:
            raise SerializationError(f"Serialization failed: {e}") from e
            
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to object.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized object
            
        Raises:
            SerializationError: If deserialization fails
        """
        try:
            # Decompress
            if self.compression == "gzip":
                data = gzip.decompress(data)
            elif self.compression == "lz4":
                data = lz4.decompress(data)
            elif self.compression != "none":
                raise SerializationError(f"Unknown compression: {self.compression}")
                
            # Deserialize
            if self.format == "msgpack":
                return msgpack.unpackb(data, raw=False)
            elif self.format == "pickle":
                return pickle.loads(data)
            elif self.format == "json":
                json_str = data.decode('utf-8')
                return json.loads(json_str)
            elif self.format == "orjson":
                return orjson.loads(data)
            else:
                raise SerializationError(f"Unknown format: {self.format}")
                
        except Exception as e:
            raise SerializationError(f"Deserialization failed: {e}") from e
            
    def _json_default(self, obj: Any) -> Any:
        """Default function for JSON serialization of custom objects."""
        if hasattr(obj, '__json__'):
            return obj.__json__()
        elif hasattr(obj, '__dict__'):
            return {
                '__class__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                '__data__': obj.__dict__
            }
        else:
            return str(obj)
            
    def estimate_size(self, obj: Any) -> int:
        """Estimate serialized size without full serialization."""
        try:
            # Quick estimation based on type
            if isinstance(obj, str):
                base_size = len(obj.encode('utf-8'))
            elif isinstance(obj, bytes):
                base_size = len(obj)
            elif isinstance(obj, (int, float)):
                base_size = 8
            elif isinstance(obj, bool):
                base_size = 1
            elif isinstance(obj, (list, tuple)):
                base_size = sum(self.estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                base_size = sum(self.estimate_size(k) + self.estimate_size(v) 
                              for k, v in obj.items())
            else:
                # For complex objects, do actual serialization
                return len(self.serialize(obj))
                
            # Apply compression estimation
            if self.compression == "gzip":
                # Gzip typically achieves 2-10x compression
                return max(base_size // 4, 64)
            elif self.compression == "lz4":
                # LZ4 typically achieves 2-4x compression
                return max(base_size // 3, 64)
            else:
                return base_size
                
        except Exception:
            # Fallback: assume reasonable size
            return 1024


class CacheKeyBuilder:
    """
    Builder pattern for complex cache key generation.
    """
    
    def __init__(self):
        self._parts = []
        self._separator = ":"
        self._hash_final = True
        
    def add_string(self, value: str) -> 'CacheKeyBuilder':
        """Add string component."""
        self._parts.append(value)
        return self
        
    def add_object(self, obj: Any, include_type: bool = False) -> 'CacheKeyBuilder':
        """Add object component."""
        if include_type:
            self._parts.append(f"{type(obj).__name__}")
            
        if hasattr(obj, '__cache_key__'):
            self._parts.append(obj.__cache_key__())
        elif isinstance(obj, (str, int, float, bool)):
            self._parts.append(str(obj))
        elif isinstance(obj, (list, tuple)):
            self._parts.append(f"[{','.join(str(item) for item in obj)}]")
        elif isinstance(obj, dict):
            items = [f"{k}={v}" for k, v in sorted(obj.items())]
            self._parts.append(f"{{{','.join(items)}}}")
        else:
            self._parts.append(str(id(obj)))
            
        return self
        
    def add_timestamp(self, precision: str = "minute") -> 'CacheKeyBuilder':
        """Add timestamp component for time-based cache invalidation."""
        current_time = time.time()
        
        if precision == "second":
            timestamp = int(current_time)
        elif precision == "minute":
            timestamp = int(current_time // 60)
        elif precision == "hour":
            timestamp = int(current_time // 3600)
        elif precision == "day":
            timestamp = int(current_time // 86400)
        else:
            timestamp = int(current_time)
            
        self._parts.append(f"t{timestamp}")
        return self
        
    def add_version(self, version: str) -> 'CacheKeyBuilder':
        """Add version component."""
        self._parts.append(f"v{version}")
        return self
        
    def set_separator(self, separator: str) -> 'CacheKeyBuilder':
        """Set component separator."""
        self._separator = separator
        return self
        
    def disable_hashing(self) -> 'CacheKeyBuilder':
        """Disable final hashing (return raw key)."""
        self._hash_final = False
        return self
        
    def build(self) -> str:
        """Build the final cache key."""
        key = self._separator.join(self._parts)
        
        if self._hash_final:
            if XXHASH_AVAILABLE:
                return xxhash.xxh64(key.encode()).hexdigest()
            else:
                return hashlib.sha256(key.encode()).hexdigest()[:16]
        else:
            return key


def safe_serialize(obj: Any, fallback_format: str = "pickle") -> bytes:
    """
    Safely serialize object with fallback options.
    
    Args:
        obj: Object to serialize
        fallback_format: Format to use if preferred methods fail
        
    Returns:
        Serialized bytes
    """
    # Try msgpack first (fastest)
    if MSGPACK_AVAILABLE:
        try:
            return msgpack.packb(obj, use_bin_type=True)
        except Exception:
            pass
            
    # Try orjson for JSON-serializable objects
    if ORJSON_AVAILABLE:
        try:
            return orjson.dumps(obj)
        except Exception:
            pass
            
    # Try standard JSON
    try:
        return json.dumps(obj, default=str).encode('utf-8')
    except Exception:
        pass
        
    # Fallback to pickle
    if fallback_format == "pickle":
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise SerializationError("All serialization methods failed")


def safe_deserialize(data: bytes, format_hint: Optional[str] = None) -> Any:
    """
    Safely deserialize data with format detection.
    
    Args:
        data: Serialized bytes
        format_hint: Hint about the format
        
    Returns:
        Deserialized object
    """
    if format_hint == "msgpack" and MSGPACK_AVAILABLE:
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception:
            pass
            
    if format_hint == "orjson" and ORJSON_AVAILABLE:
        try:
            return orjson.loads(data)
        except Exception:
            pass
            
    if format_hint == "json":
        try:
            return json.loads(data.decode('utf-8'))
        except Exception:
            pass
            
    # Try to detect format
    try:
        # Try JSON first (starts with { or [)
        if data.startswith((b'{', b'[')):
            return json.loads(data.decode('utf-8'))
    except Exception:
        pass
        
    try:
        # Try msgpack
        if MSGPACK_AVAILABLE:
            return msgpack.unpackb(data, raw=False)
    except Exception:
        pass
        
    # Fallback to pickle
    try:
        return pickle.loads(data)
    except Exception:
        raise SerializationError("Could not deserialize data with any format")


def compress_data(data: bytes, method: str = "lz4", level: int = 6) -> bytes:
    """
    Compress data using specified method.
    
    Args:
        data: Data to compress
        method: Compression method (lz4, gzip)
        level: Compression level
        
    Returns:
        Compressed data
    """
    if method == "lz4" and LZ4_AVAILABLE:
        return lz4.compress(data, compression_level=level)
    elif method == "gzip":
        return gzip.compress(data, compresslevel=level)
    else:
        # No compression
        return data


def decompress_data(data: bytes, method: str = "lz4") -> bytes:
    """
    Decompress data using specified method.
    
    Args:
        data: Compressed data
        method: Compression method
        
    Returns:
        Decompressed data
    """
    if method == "lz4" and LZ4_AVAILABLE:
        return lz4.decompress(data)
    elif method == "gzip":
        return gzip.decompress(data)
    else:
        # No compression
        return data


def encode_cache_value(value: Any, encoding: str = "base64") -> str:
    """
    Encode cache value as string.
    
    Args:
        value: Value to encode
        encoding: Encoding method (base64, hex)
        
    Returns:
        Encoded string
    """
    serialized = safe_serialize(value)
    
    if encoding == "base64":
        return base64.b64encode(serialized).decode('ascii')
    elif encoding == "hex":
        return serialized.hex()
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def decode_cache_value(encoded: str, encoding: str = "base64") -> Any:
    """
    Decode cache value from string.
    
    Args:
        encoded: Encoded string
        encoding: Encoding method
        
    Returns:
        Decoded value
    """
    if encoding == "base64":
        serialized = base64.b64decode(encoded.encode('ascii'))
    elif encoding == "hex":
        serialized = bytes.fromhex(encoded)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
        
    return safe_deserialize(serialized)


def get_cache_key_info(key: str) -> Dict[str, Any]:
    """
    Extract information from a cache key.
    
    Args:
        key: Cache key
        
    Returns:
        Key information
    """
    info = {
        'key': key,
        'length': len(key),
        'hash_type': 'unknown',
        'estimated_entropy': 0.0
    }
    
    # Detect hash type by length
    if len(key) == 16:
        info['hash_type'] = 'xxhash64_short'
    elif len(key) == 32:
        info['hash_type'] = 'md5'
    elif len(key) == 64:
        info['hash_type'] = 'sha256'
    elif len(key) == 16:
        info['hash_type'] = 'sha256_short'
        
    # Estimate entropy (simple check for randomness)
    unique_chars = len(set(key))
    info['estimated_entropy'] = unique_chars / len(key) if key else 0.0
    
    return info


# Pre-configured instances for common use cases
default_key_generator = KeyGenerator()
default_serializer = Serializer()
fast_serializer = Serializer(format="msgpack", compression="lz4")
compact_serializer = Serializer(format="msgpack", compression="gzip")