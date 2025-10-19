#!/usr/bin/env python3
"""
Virtual Environment and Anant Library Verification Script

This script verifies that the virtual environment is properly set up
and that the anant library is working correctly.
"""

import sys
import os
from pathlib import Path

def check_virtual_env():
    """Check if we're in a virtual environment"""
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"✅ Virtual environment active: {venv}")
        return True
    else:
        print("❌ No virtual environment detected")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 9

def check_dependencies():
    """Check that required dependencies are installed"""
    required_packages = [
        ('polars', 'polars'),
        ('numpy', 'numpy'), 
        ('networkx', 'networkx'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn')  # package name vs import name
    ]
    missing = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} missing")
            missing.append(package_name)
    
    return len(missing) == 0

def check_anant_library():
    """Check that anant library works"""
    try:
        import anant
        print(f"✅ anant library imported (version {anant.__version__})")
        
        # Test PropertyStore
        ps = anant.PropertyStore(level=1)
        ps.set_property('test_node', 'test_prop', 'test_value')
        print("✅ PropertyStore working")
        
        # Test IncidenceStore
        import polars as pl
        data = pl.DataFrame({
            'edges': ['e1', 'e1', 'e2'],
            'nodes': ['n1', 'n2', 'n1']
        })
        inc = anant.IncidenceStore(data)
        neighbors = inc.get_neighbors(0, 'e1')  # Get nodes in edge e1
        print(f"✅ IncidenceStore working (e1 has {len(neighbors)} nodes)")
        
        return True
        
    except Exception as e:
        print(f"❌ anant library error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 Anant Library & Virtual Environment Verification")
    print("=" * 50)
    
    checks = [
        ("Virtual Environment", check_virtual_env),
        ("Python Version", check_python_version), 
        ("Dependencies", check_dependencies),
        ("Anant Library", check_anant_library)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n🧪 Checking {name}...")
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {name} check failed")
        except Exception as e:
            print(f"❌ {name} check error: {e}")
    
    print("\n" + "=" * 50)
    if passed == total:
        print(f"🎉 All {total} checks passed! Anant is ready for development.")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Please fix the issues above.")
    
    print(f"\nEnvironment Summary:")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📦 Python path: {sys.path[0]}")

if __name__ == "__main__":
    main()