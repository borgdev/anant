# ANANT Test Suite
====================

This directory contains the complete test suite for the ANANT library, organized by test type and purpose.

## 📁 Directory Structure

```
anant_test/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── debug/          # Debug scripts for troubleshooting
├── analysis/       # Analysis scripts for functionality review
└── README.md       # This file
```

## 🧪 Test Categories

### **Unit Tests** (`unit/`)
Individual component testing and basic functionality validation:

- `test_correct_property_format.py` - Property format validation tests
- `test_direct_io.py` - Direct I/O operation tests
- `test_edge_cases.py` - Edge case handling tests
- `test_minor_issues_check.py` - Minor issue validation tests
- `test_streaming.py` - Streaming functionality tests
- `test_streaming_simple.py` - Basic streaming tests
- `test_validation_framework.py` - Validation framework tests
- `test_working_io.py` - Working I/O functionality tests

### **Integration Tests** (`integration/`)
Component interaction and advanced feature testing:

- `test_advanced_io.py` - Advanced I/O integration tests
- `test_advanced_io_integration.py` - Comprehensive I/O integration
- `test_advanced_properties.py` - Advanced property management tests
- `test_enhanced_centrality.py` - Enhanced centrality algorithm tests
- `test_enhanced_community.py` - Enhanced community detection tests
- `test_enhanced_integration.py` - Full integration testing
- `test_enhanced_setsystems.py` - Enhanced SetSystem tests

### **Debug Scripts** (`debug/`)
Troubleshooting and diagnostic scripts:

- `debug_centrality.py` - Centrality algorithm debugging
- `debug_exact_property_issue.py` - Property issue diagnostics
- `debug_json_io.py` - JSON I/O debugging
- `debug_properties.py` - Property management debugging
- `debug_properties_detailed.py` - Detailed property debugging
- `debug_streaming.py` - Streaming functionality debugging

### **Analysis Scripts** (`analysis/`)
Functionality analysis and gap identification:

- `core_functionality_analysis.py` - Core functionality review
- `gap_analysis.py` - Feature gap analysis
- `remaining_work_analysis.py` - Work remaining analysis
- `test_temporal_analysis.py` - Temporal functionality analysis

## 🚀 Running Tests

### Run All Tests
```bash
# From the project root
cd /home/amansingh/dev/ai/anant
python -m pytest anant_test/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest anant_test/unit/ -v

# Integration tests only
python -m pytest anant_test/integration/ -v

# Specific test file
python anant_test/unit/test_streaming.py
```

### Run Debug Scripts
```bash
# Run individual debug scripts
python anant_test/debug/debug_properties.py
python anant_test/debug/debug_streaming.py
```

### Run Analysis Scripts
```bash
# Run analysis scripts
python anant_test/analysis/core_functionality_analysis.py
python anant_test/analysis/gap_analysis.py
```

## 🎯 Test Coverage

The test suite covers:

- ✅ **Core Hypergraph Operations** - Creation, modification, querying
- ✅ **Metagraph Functionality** - Enterprise features, metadata management
- ✅ **I/O Operations** - Parquet, JSON, CSV, GraphML, HDF5
- ✅ **Advanced Properties** - Property management and correlation
- ✅ **Streaming Operations** - Real-time data processing
- ✅ **Enhanced SetSystems** - Parquet, MultiModal, Streaming
- ✅ **Algorithm Integration** - Centrality, community detection
- ✅ **Validation Framework** - Data integrity and quality checks
- ✅ **Performance Testing** - Polars backend optimization

## 🔧 Development Workflow

### Adding New Tests
1. **Unit Tests**: Add to `unit/` for single component testing
2. **Integration Tests**: Add to `integration/` for multi-component testing
3. **Debug Scripts**: Add to `debug/` for troubleshooting specific issues
4. **Analysis Scripts**: Add to `analysis/` for functionality review

### Test Naming Convention
- `test_*.py` - Formal test files (unit and integration)
- `debug_*.py` - Debug and troubleshooting scripts
- `*_analysis.py` - Analysis and review scripts

### Running Before Commits
```bash
# Quick smoke test
python -m pytest anant_test/unit/test_streaming_simple.py -v

# Full test suite (for major changes)
python -m pytest anant_test/ -v --tb=short
```

## 📊 Test Results Summary

Recent test execution status:
- **Unit Tests**: ✅ All core functionality validated
- **Integration Tests**: ✅ Advanced features working
- **Debug Scripts**: ✅ Issues identified and resolved
- **Analysis Scripts**: ✅ Feature gaps documented

## 🏗️ ANANT Library Architecture

The tests validate the complete ANANT dual-capability architecture:

```
ANANT Library
├── Traditional Hypergraph (Polars-based)
├── Enterprise Metagraph (Polars+Parquet)
├── Advanced I/O Systems
├── Enhanced SetSystems
├── Algorithm Integration
├── Production Framework
└── Development Tools
```

## 📝 Notes

- All tests use **Polars backend** for performance
- Tests validate **dual capability** (Hypergraph + Metagraph)
- Integration tests cover **enterprise features**
- Debug scripts help with **issue resolution**
- Analysis scripts guide **future development**

## 🔗 Related Documentation

- Main library: `/anant/`
- Production framework: `/anant/production/`
- Examples: `/anant/examples/`
- Documentation: Project root markdown files