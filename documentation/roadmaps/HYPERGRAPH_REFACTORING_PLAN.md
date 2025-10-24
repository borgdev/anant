# Hypergraph Refactoring Plan

## Overview
Refactoring hypergraph.py (2,931 lines) into modular architecture following the successful metagraph.py pattern.

## Current Structure Analysis
- **PropertyWrapper class**: Utility class for property management
- **Hypergraph class**: Massive class with 50+ methods covering:
  - Core graph operations (nodes, edges, add/remove)
  - Performance optimization (caching, indexing, batch operations)
  - Graph algorithms (centrality, PageRank, HITS)
  - Visualization and layout
  - I/O operations (save/load, format conversion)
  - Set theory operations (union, intersection, difference)
  - Advanced graph theory (dual, line graphs, random walks)

## Proposed Modular Architecture

### 1. Core Module (anant/classes/hypergraph/core/)
- **hypergraph.py**: Main orchestrator class (similar to metagraph pattern)
- **property_wrapper.py**: PropertyWrapper utility class

### 2. Operations Modules

#### CoreOperations (anant/classes/hypergraph/operations/core_operations.py)
- Basic graph structure operations
- Node/edge addition/removal
- Basic queries (has_node, has_edge, neighbors)
- Graph properties (num_nodes, num_edges, is_empty)

#### PerformanceOperations (anant/classes/hypergraph/operations/performance_operations.py)
- Performance indexing and caching
- Batch operations (get_multiple_edge_nodes, get_multiple_node_edges)
- Performance statistics and reporting
- Memory optimization

#### AlgorithmOperations (anant/classes/hypergraph/operations/algorithm_operations.py)
- Centrality measures (degree, betweenness, closeness, eigenvector)
- PageRank and HITS algorithms
- Graph algorithms (MST, max flow, min cut)
- Random walks and path finding

#### VisualizationOperations (anant/classes/hypergraph/operations/visualization_operations.py)
- Layout generation (spring, circular, random, bipartite)
- Coordinate calculation
- Visualization support methods

#### IOOperations (anant/classes/hypergraph/operations/io_operations.py)
- Save/load operations (pickle, JSON, CSV)
- Format conversion (NetworkX, GraphML, GEXF)
- Import/export utilities

#### SetOperations (anant/classes/hypergraph/operations/set_operations.py)
- Set theory operations (union, intersection, difference)
- Subgraph operations (node-induced, edge-induced)
- Graph composition and decomposition

#### AdvancedOperations (anant/classes/hypergraph/operations/advanced_operations.py)
- Dual graph construction
- Line graph operations (s_line_graph)
- Advanced graph transformations
- Specialized algorithms

### 3. View Classes (anant/classes/hypergraph/views/)
- **edge_view.py**: EdgeView class for edge access patterns

## Implementation Strategy

### Phase 1: Extract Utility Classes
1. Move PropertyWrapper to separate module
2. Move EdgeView to views module
3. Test basic functionality

### Phase 2: Create Operations Modules
1. Create CoreOperations with basic graph operations
2. Create PerformanceOperations with caching/indexing
3. Create IOOperations with save/load functionality
4. Test core functionality

### Phase 3: Extract Algorithm Modules
1. Create AlgorithmOperations with centrality and graph algorithms
2. Create VisualizationOperations with layout generation
3. Create SetOperations with set theory operations
4. Create AdvancedOperations with specialized algorithms
5. Test all algorithmic functionality

### Phase 4: Refactor Main Class
1. Create new lean Hypergraph class with delegation pattern
2. Integrate all operation modules
3. Maintain full API compatibility
4. Comprehensive testing

### Phase 5: Validation and Optimization
1. Run full test suite
2. Performance benchmarking
3. Exception handling validation
4. Documentation updates

## Expected Outcomes
- **Hypergraph core class**: ~200 lines (93% reduction from 2,931 lines)
- **7 specialized operation modules**: ~2,700 lines total
- **Improved maintainability**: Clear separation of concerns
- **Enhanced testability**: Isolated functionality
- **Better performance**: Optimized caching and indexing
- **Full API compatibility**: No breaking changes

## File Structure After Refactoring
```
anant/classes/hypergraph/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── hypergraph.py           # Main class (~200 lines)
│   └── property_wrapper.py     # PropertyWrapper utility
├── operations/
│   ├── __init__.py
│   ├── core_operations.py      # Basic graph operations
│   ├── performance_operations.py # Caching and performance
│   ├── algorithm_operations.py # Graph algorithms
│   ├── visualization_operations.py # Layout and visualization
│   ├── io_operations.py        # Save/load operations
│   ├── set_operations.py       # Set theory operations
│   └── advanced_operations.py  # Specialized algorithms
└── views/
    ├── __init__.py
    └── edge_view.py            # EdgeView class
```

## Risk Mitigation
- Maintain exact API compatibility
- Comprehensive test coverage for each module
- Performance validation at each step
- Rollback plan if issues arise
- Progressive migration approach