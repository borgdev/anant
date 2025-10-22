# Entity Terminology Standardization Plan

## Goal
Standardize all graph types to use consistent `entity` terminology instead of mixed `node`/`entity` usage.

## Current State Analysis

### Hypergraph (Uses `node`)
- `add_node()` → `add_entity()`
- `remove_node()` → `remove_entity()`
- `has_node()` → `has_entity()`
- `nodes` property → `entities` property
- `num_nodes()` → `num_entities()`
- `get_node_degree()` → `get_entity_degree()`
- `get_node_edges()` → `get_entity_edges()`
- `get_edge_nodes()` → `get_edge_entities()`

### KnowledgeGraph (Mixed `node`/`entity`)
- `add_node()` → `add_entity()`
- `remove_node()` → `remove_entity()`
- `nodes` property → `entities` property
- `get_node_type()` → `get_entity_type()`
- `set_node_type()` → `set_entity_type()`
- `get_node_uri()` → `get_entity_uri()`
- Keep: `extract_entity_type()` (already consistent)

### HierarchicalKnowledgeGraph (Mixed)
- `nodes()` → `entities()`
- `num_nodes()` → `num_entities()`
- `has_node()` → `has_entity()`
- Keep: `add_entity_to_level()`, `get_entity_level()`, `add_entity()`, `remove_entity()` (already consistent)

### MetaGraph (Already consistent with `entity`)
- No changes needed - already uses `entity` terminology

## Implementation Steps

1. **Phase 1: Core Hypergraph**
   - Update method names and signatures
   - Update internal references
   - Update property names
   - Add backward compatibility aliases

2. **Phase 2: KnowledgeGraph**
   - Update method names from `node` to `entity`
   - Ensure consistent parameter naming
   - Update internal references

3. **Phase 3: HierarchicalKnowledgeGraph**
   - Update remaining `node` methods to `entity`
   - Ensure consistency with core KnowledgeGraph

4. **Phase 4: Update all test files and examples**
   - Update test_schemaorg_hierarchical.py
   - Update schema_org_analysis.py
   - Update any other test files

5. **Phase 5: Documentation update**
   - Update docstrings
   - Update method documentation

## Backward Compatibility
- Keep old method names as deprecated aliases
- Add deprecation warnings
- Plan removal for future major version