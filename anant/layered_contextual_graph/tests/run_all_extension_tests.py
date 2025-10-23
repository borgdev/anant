"""
Master Test Runner for LCG Extensions
=====================================

Runs all three extension test suites and generates a summary.
"""

import sys
import subprocess
from pathlib import Path

def run_test_suite(test_file):
    """Run a single test suite"""
    test_path = Path(__file__).parent / test_file
    venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
    
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [str(venv_python), str(test_path)],
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all extension test suites"""
    
    print("\n" + "="*70)
    print("LCG EXTENSIONS - MASTER TEST RUNNER")
    print("="*70)
    print("\nRunning comprehensive test suites for all three extensions\n")
    
    test_suites = [
        ("test_streaming_extension.py", "Streaming & Event-Driven"),
        ("test_ml_extension.py", "Machine Learning"),
        ("test_reasoning_extension.py", "Advanced Reasoning")
    ]
    
    results = {}
    
    for test_file, name in test_suites:
        success = run_test_suite(test_file)
        results[name] = success
    
    # Summary
    print("\n" + "="*70)
    print("TEST EXECUTION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL EXTENSION TESTS PASSED")
        print("="*70)
        print("\n🎉 All three extensions are fully tested and functional!\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        failed = [name for name, passed in results.items() if not passed]
        print(f"\nFailed suites: {', '.join(failed)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
