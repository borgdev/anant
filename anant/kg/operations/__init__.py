"""
Knowledge Graph Operations Modules
=================================

Modular operations for knowledge graph functionality including:
- Semantic operations (entity typing, URI handling)
- Ontology operations (schema integration, validation) 
- Query operations (SPARQL, semantic queries)
- Indexing operations (semantic indexing, caching)
- Reasoning operations (inference, rule processing)
- NLP operations (entity extraction, linking)
"""

from .semantic_operations import SemanticOperations
from .ontology_operations import OntologyOperations
from .query_operations import QueryOperations
from .indexing_operations import IndexingOperations
from .reasoning_operations import ReasoningOperations
from .nlp_operations import NLPOperations

__all__ = [
    'SemanticOperations',
    'OntologyOperations', 
    'QueryOperations',
    'IndexingOperations',
    'ReasoningOperations',
    'NLPOperations'
]