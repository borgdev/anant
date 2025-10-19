#!/usr/bin/env python3
"""
ANANT Test Runner
=================

Convenient test runner for the ANANT library test suite.
Provides organized test execution with reporting.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🔍 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, cwd="/home/amansingh/dev/ai/anant", 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def main():
    """Main test runner."""
    print("🧪 ANANT Test Suite Runner")
    print("=" * 60)
    print("Organized test execution for ANANT library")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "all"
    
    results = []
    
    if test_type in ["all", "unit"]:
        # Run unit tests
        print("\n📦 Unit Tests")
        for test_file in Path("/home/amansingh/dev/ai/anant/anant_test/unit").glob("test_*.py"):
            success = run_command(
                f"cd /home/amansingh/dev/ai/anant && python {test_file}",
                f"Unit Test: {test_file.name}"
            )
            results.append(("Unit", test_file.name, success))
    
    if test_type in ["all", "integration"]:
        # Run integration tests
        print("\n🔗 Integration Tests")
        for test_file in Path("/home/amansingh/dev/ai/anant/anant_test/integration").glob("test_*.py"):
            success = run_command(
                f"cd /home/amansingh/dev/ai/anant && python {test_file}",
                f"Integration Test: {test_file.name}"
            )
            results.append(("Integration", test_file.name, success))
    
    if test_type in ["all", "debug"]:
        # Run debug scripts
        print("\n🐛 Debug Scripts")
        for debug_file in Path("/home/amansingh/dev/ai/anant/anant_test/debug").glob("debug_*.py"):
            success = run_command(
                f"cd /home/amansingh/dev/ai/anant && python {debug_file}",
                f"Debug Script: {debug_file.name}"
            )
            results.append(("Debug", debug_file.name, success))
    
    if test_type in ["all", "analysis"]:
        # Run analysis scripts
        print("\n📊 Analysis Scripts")
        for analysis_file in Path("/home/amansingh/dev/ai/anant/anant_test/analysis").glob("*.py"):
            success = run_command(
                f"cd /home/amansingh/dev/ai/anant && python {analysis_file}",
                f"Analysis Script: {analysis_file.name}"
            )
            results.append(("Analysis", analysis_file.name, success))
    
    # Test results summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, _, success in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    
    # Detailed results by category
    categories = {}
    for category, filename, success in results:
        if category not in categories:
            categories[category] = {"passed": 0, "failed": 0, "files": []}
        
        if success:
            categories[category]["passed"] += 1
        else:
            categories[category]["failed"] += 1
        
        categories[category]["files"].append((filename, success))
    
    print("\n📋 Results by Category:")
    for category, data in categories.items():
        total = data["passed"] + data["failed"]
        success_rate = (data["passed"] / total * 100) if total > 0 else 0
        print(f"  {category}: {data['passed']}/{total} passed ({success_rate:.1f}%)")
        
        # Show failed tests
        failed_files = [f for f, success in data["files"] if not success]
        if failed_files:
            print(f"    Failed: {', '.join(failed_files)}")
    
    # Exit with appropriate code
    if failed_tests == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {failed_tests} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()