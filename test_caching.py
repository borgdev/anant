#!/usr/bin/env python3
"""
Test script for the Advanced Caching System

This script tests the basic functionality of the caching system
and validates that all components work correctly with graceful fallbacks.
"""

import time
import tempfile
import shutil
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cache_manager():
    """Test the core cache manager functionality."""
    print("Testing CacheManager...")
    
    from anant.caching import CacheManager
    
    # Create cache manager with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = CacheManager(
            redis_url=None,  # Don't require Redis for testing
            disk_cache_dir=temp_dir
        )
        
        # Test basic get/set
        key = "test_key"
        value = {"message": "Hello, Cache!", "number": 42}
        
        # Set value
        success = cache_manager.set(key, value, ttl=60)
        assert success, "Failed to set cache value"
        print("✓ Cache set successful")
        
        # Get value
        retrieved = cache_manager.get(key)
        assert retrieved == value, f"Retrieved value {retrieved} != {value}"
        print("✓ Cache get successful")
        
        # Test non-existent key
        missing = cache_manager.get("non_existent", default="not_found")
        assert missing == "not_found", "Default value not returned"
        print("✓ Default value handling successful")
        
        # Test delete
        deleted = cache_manager.delete(key)
        assert deleted, "Failed to delete cache key"
        
        # Verify deletion
        after_delete = cache_manager.get(key)
        assert after_delete is None, "Key still exists after deletion"
        print("✓ Cache deletion successful")
        
        # Get statistics
        stats = cache_manager.get_stats()
        assert stats.hits >= 1, "No cache hits recorded"
        assert stats.sets >= 1, "No cache sets recorded"
        print(f"✓ Cache stats: {stats.hits} hits, {stats.misses} misses, {stats.sets} sets")
        
    print("CacheManager tests passed!\n")


def test_memory_cache():
    """Test the memory cache implementation."""
    print("Testing MemoryCache...")
    
    from anant.caching import MemoryCache
    
    # Create memory cache with small limits for testing
    cache = MemoryCache(max_size=3, max_memory=1024)
    
    # Test basic operations
    cache.set("key1", "value1")
    cache.set("key2", "value2") 
    cache.set("key3", "value3")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    print("✓ Basic memory cache operations successful")
    
    # Test LRU eviction
    cache.set("key4", "value4")  # Should evict key1 (least recently used)
    
    assert cache.get("key1") is None, "LRU eviction failed"
    assert cache.get("key4") == "value4", "New key not found"
    print("✓ LRU eviction successful")
    
    # Test TTL
    cache.set("ttl_key", "ttl_value", ttl=0.1)  # 100ms TTL
    time.sleep(0.2)  # Wait for expiration
    
    assert cache.get("ttl_key") is None, "TTL expiration failed"
    print("✓ TTL expiration successful")
    
    # Test statistics
    stats = cache.get_stats()
    assert stats['size'] > 0, "Cache size not tracked"
    print(f"✓ Memory cache stats: {stats['size']} items, {stats['hits']} hits")
    
    print("MemoryCache tests passed!\n")


def test_disk_cache():
    """Test the disk cache implementation."""
    print("Testing DiskCache...")
    
    from anant.caching import DiskCache
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskCache(temp_dir, size_limit=10*1024*1024)  # 10MB limit
        
        # Test basic operations
        test_data = {"large_data": "x" * 1000, "nested": {"key": "value"}}
        
        success = cache.set("disk_key", test_data)
        assert success, "Failed to set disk cache value"
        
        retrieved = cache.get("disk_key")
        assert retrieved == test_data, "Disk cache data corrupted"
        print("✓ Disk cache basic operations successful")
        
        # Test persistence
        cache2 = DiskCache(temp_dir)  # New cache instance, same directory
        persisted = cache2.get("disk_key")
        assert persisted == test_data, "Data not persisted across instances"
        print("✓ Disk cache persistence successful")
        
        # Test deletion
        deleted = cache.delete("disk_key")
        assert deleted, "Failed to delete from disk cache"
        
        assert cache.get("disk_key") is None, "Key still exists after deletion"
        print("✓ Disk cache deletion successful")
        
        # Test statistics
        stats = cache.get_stats()
        print(f"✓ Disk cache stats: {stats['count']} items, {stats['size']} bytes")
        
    print("DiskCache tests passed!\n")


def test_decorators():
    """Test caching decorators."""
    print("Testing caching decorators...")
    
    from anant.caching import memoize, cached_property
    import tempfile
    
    # Test memoize decorator
    call_count = 0
    
    @memoize(ttl=60)
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate expensive operation
        return x + y
        
    # First call should compute
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # Second call should use cache
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # Should not increment
    print("✓ Memoize decorator successful")
    
    # Test cached property
    class TestClass:
        def __init__(self):
            self.compute_count = 0
            
        @cached_property
        def expensive_property(self):
            self.compute_count += 1
            return "computed_value"
            
    obj = TestClass()
    
    # First access should compute
    value1 = obj.expensive_property
    assert value1 == "computed_value"
    assert obj.compute_count == 1
    
    # Second access should use cache
    value2 = obj.expensive_property
    assert value2 == "computed_value"
    assert obj.compute_count == 1  # Should not increment
    print("✓ Cached property successful")
    
    print("Caching decorators tests passed!\n")


def test_query_cache():
    """Test query-specific caching."""
    print("Testing QueryCache...")
    
    from anant.caching import QueryCache, QueryType
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache manager for query cache
        from anant.caching import CacheManager
        cache_manager = CacheManager(disk_cache_dir=temp_dir)
        
        query_cache = QueryCache(cache_manager)
        
        # Test SQL query caching
        sql_query = "SELECT * FROM users WHERE age > 25"
        query_result = [{"id": 1, "name": "Alice", "age": 30}]
        
        # Cache the query result
        success = query_cache.set(
            sql_query, 
            query_result, 
            cost=1.5,
            dependencies={"users"}
        )
        assert success, "Failed to cache query result"
        
        # Retrieve cached result
        cached_result = query_cache.get(sql_query)
        assert cached_result == query_result, "Query result not cached correctly"
        print("✓ Query caching successful")
        
        # Test table invalidation
        invalidated = query_cache.invalidate_table("users")
        assert invalidated >= 1, "Table invalidation failed"
        
        # Verify invalidation
        after_invalidation = query_cache.get(sql_query)
        assert after_invalidation is None, "Query not invalidated"
        print("✓ Query invalidation successful")
        
        # Test statistics
        stats = query_cache.get_query_stats()
        print(f"✓ Query cache stats: {stats['cached_queries']} queries cached")
        
    print("QueryCache tests passed!\n")


def test_utils():
    """Test cache utilities."""
    print("Testing cache utilities...")
    
    from anant.caching.utils import (
        KeyGenerator, Serializer, CacheKeyBuilder,
        safe_serialize, safe_deserialize
    )
    
    # Test key generation
    key_gen = KeyGenerator()
    
    key1 = key_gen.generate_key("arg1", "arg2", param1="value1")
    key2 = key_gen.generate_key("arg1", "arg2", param1="value1")
    key3 = key_gen.generate_key("arg1", "arg2", param1="value2")
    
    assert key1 == key2, "Same inputs should generate same key"
    assert key1 != key3, "Different inputs should generate different keys"
    print("✓ Key generation successful")
    
    # Test serialization
    test_data = {"complex": [1, 2, {"nested": "data"}]}
    
    serialized = safe_serialize(test_data)
    deserialized = safe_deserialize(serialized)
    
    assert deserialized == test_data, "Serialization round-trip failed"
    print("✓ Serialization successful")
    
    # Test cache key builder
    builder = CacheKeyBuilder()
    key = (builder
           .add_string("function_name")
           .add_object([1, 2, 3])
           .add_timestamp("minute")
           .build())
    
    assert isinstance(key, str), "Key builder failed to generate string key"
    assert len(key) > 0, "Empty key generated"
    print("✓ Cache key builder successful")
    
    print("Cache utilities tests passed!\n")


def test_error_handling():
    """Test error handling and graceful fallbacks."""
    print("Testing error handling...")
    
    from anant.caching import CacheManager
    
    # Test with invalid Redis URL (should fallback gracefully)
    cache_manager = CacheManager(
        redis_url="redis://invalid:6379/0",  # Invalid Redis URL
        max_memory_size=1024
    )
    
    # Should still work with memory and disk cache
    success = cache_manager.set("test_key", "test_value")
    assert success, "Cache should work even without Redis"
    
    value = cache_manager.get("test_key")
    assert value == "test_value", "Value not retrieved from fallback cache"
    print("✓ Graceful Redis fallback successful")
    
    # Test with very large value (should handle gracefully)
    large_value = "x" * (10 * 1024 * 1024)  # 10MB string
    large_success = cache_manager.set("large_key", large_value)
    # Success may vary depending on limits, but shouldn't crash
    print(f"✓ Large value handling: {'successful' if large_success else 'handled gracefully'}")
    
    print("Error handling tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Advanced Caching System Tests")
    print("=" * 60)
    print()
    
    try:
        test_cache_manager()
        test_memory_cache()
        test_disk_cache()
        test_decorators()
        test_query_cache()
        test_utils()
        test_error_handling()
        
        print("=" * 60)
        print("✅ All tests passed successfully!")
        print("The Advanced Caching System is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()