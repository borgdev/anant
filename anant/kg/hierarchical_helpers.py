"""
Hierarchical Knowledge Graph Helper Functions

Utility functions for creating pre-configured hierarchical knowledge graphs
for common domains and use cases.
"""

from typing import List, Dict, Any
from .hierarchical import HierarchicalKnowledgeGraph


def create_domain_hierarchy(domain_name: str, 
                          levels_config: List[Dict[str, Any]]) -> HierarchicalKnowledgeGraph:
    """
    Create a hierarchical knowledge graph for a specific domain.
    
    Args:
        domain_name: Name of the domain
        levels_config: List of level configurations with 'id', 'name', 'description', 'order'
        
    Returns:
        Configured HierarchicalKnowledgeGraph
        
    Example:
        >>> levels = [
        ...     {"id": "enterprise", "name": "Enterprise", "description": "Organization level", "order": 0},
        ...     {"id": "division", "name": "Division", "description": "Business divisions", "order": 1},
        ...     {"id": "department", "name": "Department", "description": "Departments", "order": 2},
        ...     {"id": "team", "name": "Team", "description": "Work teams", "order": 3}
        ... ]
        >>> hkg = create_domain_hierarchy("Corporate Structure", levels)
    """
    
    hkg = HierarchicalKnowledgeGraph(name=f"{domain_name}_Hierarchy")
    
    for level_config in levels_config:
        hkg.create_level(
            level_id=level_config["id"],
            level_name=level_config["name"],
            level_description=level_config.get("description", ""),
            level_order=level_config["order"]
        )
    
    return hkg


def create_research_hierarchy() -> HierarchicalKnowledgeGraph:
    """Create a pre-configured hierarchy for research knowledge organization."""
    
    levels = [
        {"id": "field", "name": "Research Field", "description": "Major research fields", "order": 0},
        {"id": "area", "name": "Research Area", "description": "Specific research areas", "order": 1},
        {"id": "topic", "name": "Research Topic", "description": "Research topics", "order": 2},
        {"id": "paper", "name": "Research Paper", "description": "Individual papers", "order": 3},
        {"id": "concept", "name": "Concept", "description": "Research concepts and methods", "order": 4}
    ]
    
    return create_domain_hierarchy("Research Knowledge", levels)


def create_enterprise_hierarchy() -> HierarchicalKnowledgeGraph:
    """Create a pre-configured hierarchy for enterprise knowledge management."""
    
    levels = [
        {"id": "enterprise", "name": "Enterprise", "description": "Organization level", "order": 0},
        {"id": "business_unit", "name": "Business Unit", "description": "Business units/divisions", "order": 1},
        {"id": "data_domain", "name": "Data Domain", "description": "Logical data domains", "order": 2},
        {"id": "dataset", "name": "Dataset", "description": "Individual datasets", "order": 3},
        {"id": "schema", "name": "Schema", "description": "Data schemas and tables", "order": 4},
        {"id": "field", "name": "Field", "description": "Individual data fields", "order": 5}
    ]
    
    return create_domain_hierarchy("Enterprise Data", levels)


__all__ = [
    'create_domain_hierarchy',
    'create_research_hierarchy', 
    'create_enterprise_hierarchy'
]