"""
LLM Integration Package for Metagraph

This package provides Large Language Model integration capabilities for the
enterprise metagraph system, enabling natural language querying, semantic
understanding, and AI-powered insights.

Phase 2 Components:
- Natural Language Query Processing
- Enhanced Semantic Engine with LLM capabilities
- RAG Context Generation
- Intelligent Recommendations
- Business Glossary Enhancement

Author: anant development team
Date: October 2025
"""

from .query_processor import (
    QueryIntent,
    QueryComplexity,
    ParsedQuery,
    QueryExecutionPlan,
    NLPQueryProcessor,
    LLMQueryExecutor
)

from .enhanced_semantic_engine import (
    SemanticEmbedding,
    BusinessTerm,
    SemanticRelationship,
    EnhancedSemanticEngine
)

from .rag_context_generator import (
    ContextType,
    ContextPriority,
    ContextFragment,
    RAGContext,
    RAGContextGenerator
)

from .intelligent_recommendations import (
    RecommendationType,
    RecommendationPriority,
    RecommendationStatus,
    Recommendation,
    RecommendationBatch,
    IntelligentRecommendationEngine
)

# Version information
__version__ = "0.2.0"
__phase__ = "Phase 2 - LLM Integration"

# Package metadata
__all__ = [
    # Query processing
    "QueryIntent",
    "QueryComplexity", 
    "ParsedQuery",
    "QueryExecutionPlan",
    "NLPQueryProcessor",
    "LLMQueryExecutor",
    
    # Enhanced semantic engine
    "SemanticEmbedding",
    "BusinessTerm",
    "SemanticRelationship",
    "EnhancedSemanticEngine",
    
    # RAG context generation
    "ContextType",
    "ContextPriority",
    "ContextFragment",
    "RAGContext",
    "RAGContextGenerator",
    
    # Intelligent recommendations
    "RecommendationType",
    "RecommendationPriority",
    "RecommendationStatus",
    "Recommendation",
    "RecommendationBatch",
    "IntelligentRecommendationEngine",
    
    # Convenience functions
    "create_query_processor",
    "create_semantic_engine",
    "create_rag_generator",
    "create_recommendation_engine",
    "process_natural_language_query"
]


def create_query_processor(backend="auto", **kwargs):
    """
    Create a natural language query processor
    
    Parameters
    ----------
    backend : str
        NLP backend to use ("auto", "openai", "transformers", "rule_based")
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    NLPQueryProcessor
        Configured query processor instance
    """
    return NLPQueryProcessor(backend=backend, **kwargs)


def create_semantic_engine(embedding_model="sentence-transformers/all-MiniLM-L6-v2", **kwargs):
    """
    Create an enhanced semantic engine with LLM capabilities
    
    Parameters
    ----------
    embedding_model : str
        Sentence transformer model for embeddings
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    EnhancedSemanticEngine
        Configured semantic engine instance
    """
    return EnhancedSemanticEngine(embedding_model=embedding_model, **kwargs)


def create_rag_generator(metagraph_instance, **kwargs):
    """
    Create a RAG context generator
    
    Parameters
    ----------
    metagraph_instance : Metagraph
        The metagraph instance to generate context from
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    RAGContextGenerator
        Configured RAG context generator instance
    """
    return RAGContextGenerator(metagraph_instance, **kwargs)


def create_recommendation_engine(metagraph_instance, **kwargs):
    """
    Create an intelligent recommendation engine
    
    Parameters
    ----------
    metagraph_instance : Metagraph
        The metagraph instance to analyze
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    IntelligentRecommendationEngine
        Configured recommendation engine instance
    """
    return IntelligentRecommendationEngine(metagraph_instance, **kwargs)


def process_natural_language_query(query, metagraph_instance, context=None):
    """
    Convenience function to process a natural language query
    
    Parameters
    ----------
    query : str
        Natural language query
    metagraph_instance : Metagraph
        The metagraph instance to query against
    context : dict, optional
        Additional context for query processing
        
    Returns
    -------
    dict
        Query results with metadata
    """
    executor = LLMQueryExecutor(metagraph_instance)
    return executor.execute_natural_language_query(query, context)