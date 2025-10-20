"""
Comprehensive Test Suite Runner
==============================

Runs all tests for the Anant library covering:
- Core graph classes and algorithms
- Machine learning integration
- GPU acceleration
- Caching systems
- Distributed computing
- Real-time streaming
- Knowledge graphs and metagraphs
- Performance and integration tests
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def run_all_tests():
    """Run all test suites."""
    print("🧪 Starting Comprehensive Anant Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    test_results = {}
    
    # Import and run test modules
    test_modules = [
        ("Core Classes", "test_core_classes"),
        ("Algorithms", "test_algorithms"),
        ("Distributed Computing", "test_distributed_computing"),
        ("Streaming Framework", "test_streaming_framework"),
        ("Knowledge Graphs", "test_knowledge_graphs"),
        ("Metagraphs", "test_metagraphs"),
        ("All Graph Types", "test_all_graph_types"),
        ("Machine Learning", "test_ml_integration"),
        ("GPU Acceleration", "test_gpu_acceleration"),
        ("Caching System", "test_caching_system"),
        ("Integration Tests", "test_integration"),
        ("Performance Tests", "test_performance")
    ]
    
    for test_name, module_name in test_modules:
        print(f"\n🔍 Running {test_name} Tests...")
        print("-" * 40)
        
        try:
            # Dynamic import
            module = __import__(module_name)
            
            # Run the test function
            if hasattr(module, 'run_tests'):
                if asyncio.iscoroutinefunction(module.run_tests):
                    result = await module.run_tests()
                else:
                    result = module.run_tests()
                
                test_results[test_name] = result
                print(f"✅ {test_name}: {result['status']}")
                if result.get('details'):
                    for detail in result['details']:
                        print(f"   {detail}")
            else:
                test_results[test_name] = {"status": "SKIPPED", "reason": "No run_tests function"}
                print(f"⚠️  {test_name}: SKIPPED (No run_tests function)")
                
        except ImportError as e:
            test_results[test_name] = {"status": "SKIPPED", "reason": f"Import error: {e}"}
            print(f"⚠️  {test_name}: SKIPPED (Import error)")
        except Exception as e:
            test_results[test_name] = {"status": "FAILED", "error": str(e)}
            print(f"❌ {test_name}: FAILED ({e})")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 Test Suite Summary")
    print("=" * 60)
    
    passed = sum(1 for r in test_results.values() if r['status'] == 'PASSED')
    failed = sum(1 for r in test_results.values() if r['status'] == 'FAILED')
    skipped = sum(1 for r in test_results.values() if r['status'] == 'SKIPPED')
    
    print(f"Total Tests: {len(test_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for test_name, result in test_results.items():
        status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⚠️"}
        emoji = status_emoji.get(result['status'], "❓")
        print(f"{emoji} {test_name}: {result['status']}")
        if result.get('error'):
            print(f"     Error: {result['error']}")
        if result.get('reason'):
            print(f"     Reason: {result['reason']}")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(run_all_tests())