"""
ANANT Knowledge Graph Module
============================

A comprehensive, domain-agnostic knowledge graph platform built on semantic hypergraphs.
Provides advanced querying, reasoning, entity resolution, and ontology analysis capabilities.
"""

from .core import SemanticHypergraph, KnowledgeGraph
from .query import SemanticQueryEngine, SPARQLEngine
from .ontology import OntologyAnalyzer, SchemaExtractor
from .entity import EntityResolver, EntityLinker
from .reasoning import PathReasoner, InferenceEngine
from .embeddings import KGEmbedder, EmbeddingConfig, EmbeddingResult
from .vectors import VectorEngine, VectorSearchConfig, SearchResult
from .neural_reasoning import NeuralReasoner, ReasoningConfig, ReasoningResult, AttentionReasoner
from .query_optimization import QueryOptimizer, OptimizationResult, ExecutionPlan, QueryStatistics
# Federated Query Engine
from .federated_query import (
    FederatedQueryEngine,
    FederatedQueryResult,
    DataSource,
    FederationProtocol
)

# Natural Language Interface
from .natural_language import (
    NaturalLanguageInterface,
    QueryResult,
    QueryInterpretation,
    ConversationContext,
    EntityMention,
    RelationMention,
    Intent,
    QueryType,
    ConfidenceLevel
)

# Performance monitoring
from ..utils.performance import performance_monitor, PerformanceProfiler

# Version info
__version__ = "0.4.0"
__author__ = "ANANT AI"

# Main exports
__all__ = [
    # Core classes
    'SemanticHypergraph',
    'KnowledgeGraph',
    
    # Query engines
    'SemanticQueryEngine',
    'SPARQLEngine',
    
    # Analysis tools
    'OntologyAnalyzer',
    'SchemaExtractor',
    
    # Entity resolution
    'EntityResolver',
    'EntityLinker',
    
    # Reasoning
    'PathReasoner',
    'InferenceEngine',
    
    # Advanced AI/ML features
    'KGEmbedder',
    'EmbeddingConfig', 
    'EmbeddingResult',
    'VectorEngine',
    'VectorSearchConfig',
    'SearchResult',
    'NeuralReasoner',
    'ReasoningConfig',
    'ReasoningResult', 
    'AttentionReasoner',
    'QueryOptimizer',
    'OptimizationResult',
    'ExecutionPlan',
    'QueryStatistics',
    'FederatedQueryEngine',
    'FederatedQueryResult',
    'DataSource',
    'FederationProtocol',
    
    # Natural Language Interface
    'NaturalLanguageInterface',
    'QueryResult',
    'QueryInterpretation', 
    'ConversationContext',
    'EntityMention',
    'RelationMention',
    'Intent',
    'QueryType',
    'ConfidenceLevel',
    
    # Utilities
    'performance_monitor',
    'PerformanceProfiler'
]


def create_knowledge_graph(data_source=None, **kwargs) -> KnowledgeGraph:
    """
    Convenience function to create a new KnowledgeGraph instance
    
    Args:
        data_source: Optional data source to load
        **kwargs: Additional arguments for KnowledgeGraph
        
    Returns:
        KnowledgeGraph instance
    """
    kg = KnowledgeGraph(**kwargs)
    
    if data_source:
        if isinstance(data_source, dict):
            kg.add_triples_from_dict(data_source)
        elif isinstance(data_source, str):
            # Try to load from file
            try:
                kg.load(data_source)
            except Exception:
                # Assume it's a semantic triple string
                kg.add_semantic_triple(data_source)
        elif hasattr(data_source, '__iter__'):
            # Assume it's an iterable of triples
            for triple in data_source:
                kg.add_semantic_triple(triple)
    
    return kg


def quick_query(kg: KnowledgeGraph, query: str, **kwargs):
    """
    Convenience function for quick semantic queries
    
    Args:
        kg: KnowledgeGraph instance
        query: Query string
        **kwargs: Additional query parameters
        
    Returns:
        Query results
    """
    engine = SemanticQueryEngine(kg)
    return engine.query(query, **kwargs)


def analyze_ontology(kg: KnowledgeGraph, **kwargs):
    """
    Convenience function for ontology analysis
    
    Args:
        kg: KnowledgeGraph instance
        **kwargs: Additional analysis parameters
        
    Returns:
        Ontology analysis results
    """
    analyzer = OntologyAnalyzer()
    return analyzer.analyze_knowledge_graph(kg, **kwargs)


def resolve_entities(kg: KnowledgeGraph, entities, **kwargs):
    """
    Convenience function for entity resolution
    
    Args:
        kg: KnowledgeGraph instance
        entities: Entities to resolve
        **kwargs: Additional resolution parameters
        
    Returns:
        Entity resolution results
    """
    resolver = EntityResolver()
    return resolver.resolve_entities_in_kg(kg, entities, **kwargs)


def infer_relationships(kg: KnowledgeGraph, **kwargs):
    """
    Convenience function for relationship inference
    
    Args:
        kg: KnowledgeGraph instance
        **kwargs: Additional inference parameters
        
    Returns:
        Inferred relationships
    """
    inferrer = InferenceEngine(kg)
    return inferrer.infer_missing_relationships(**kwargs)


def generate_embeddings(kg: KnowledgeGraph, algorithm: str = 'TransE', **kwargs) -> EmbeddingResult:
    """
    Convenience function for generating knowledge graph embeddings
    
    Args:
        kg: KnowledgeGraph instance
        algorithm: Embedding algorithm ('TransE', 'ComplEx', 'RotatE', etc.)
        **kwargs: Additional embedding parameters
        
    Returns:
        EmbeddingResult with generated embeddings
    """
    config = EmbeddingConfig(algorithm=algorithm, **kwargs)
    embedder = KGEmbedder(kg, config)
    return embedder.generate_embeddings()


def create_vector_search(embeddings: dict, **kwargs) -> VectorEngine:
    """
    Convenience function for creating vector search engine
    
    Args:
        embeddings: Dictionary of entity embeddings
        **kwargs: Additional vector search parameters
        
    Returns:
        VectorEngine instance
    """
    config = VectorSearchConfig(**kwargs)
    engine = VectorEngine(config)
    engine.build_index(embeddings)
    return engine


def create_neural_reasoner(kg: KnowledgeGraph, **kwargs) -> NeuralReasoner:
    """
    Convenience function for creating neural reasoner
    
    Args:
        kg: KnowledgeGraph instance
        **kwargs: Additional reasoning parameters
        
    Returns:
        NeuralReasoner instance
    """
    config = ReasoningConfig(**kwargs)
    return NeuralReasoner(kg, config)


def optimize_query(kg: KnowledgeGraph, query: str, **kwargs) -> OptimizationResult:
    """
    Convenience function for query optimization
    
    Args:
        kg: KnowledgeGraph instance
        query: Query to optimize
        **kwargs: Additional optimization parameters
        
    Returns:
        OptimizationResult with optimized query and execution plan
    """
    optimizer = QueryOptimizer(kg)
    return optimizer.optimize_query(query, **kwargs)

__version__ = "1.0.0"
__author__ = "ANANT Team"

# Core imports - only import what's been implemented
try:
    from .core import KnowledgeGraph, SemanticHypergraph
    _core_available = True
except ImportError as e:
    _core_available = False
    import logging
    logging.warning(f"Core KG classes not available: {e}")

try:
    from .query import SemanticQueryEngine, SPARQLEngine
    _query_available = True
except ImportError as e:
    _query_available = False
    import logging
    logging.warning(f"Query engines not available: {e}")

try:
    from .ontology import OntologyAnalyzer, SchemaExtractor
    _ontology_available = True
except ImportError as e:
    _ontology_available = False
    import logging
    logging.warning(f"Ontology analysis not available: {e}")

try:
    from .entity import EntityResolver, EntityLinker
    _entity_available = True
except ImportError as e:
    _entity_available = False
    import logging
    logging.warning(f"Entity resolution not available: {e}")

try:
    from .reasoning import PathReasoner, InferenceEngine
    _reasoning_available = True
except ImportError as e:
    _reasoning_available = False
    import logging
    logging.warning(f"Reasoning capabilities not available: {e}")

# Build __all__ dynamically based on what's available
__all__ = []

if _core_available:
    __all__.extend(['KnowledgeGraph', 'SemanticHypergraph'])

if _query_available:
    __all__.extend(['SemanticQueryEngine', 'SPARQLEngine'])

if _ontology_available:
    __all__.extend(['OntologyAnalyzer', 'SchemaExtractor'])

if _entity_available:
    __all__.extend(['EntityResolver', 'EntityLinker'])

if _reasoning_available:
    __all__.extend(['PathReasoner', 'InferenceEngine'])


# Convenience functions for quick access
def create_knowledge_graph(data=None, **kwargs):
    """
    Create a new KnowledgeGraph instance
    
    Args:
        data: Initial graph data
        **kwargs: Additional arguments for KnowledgeGraph
        
    Returns:
        KnowledgeGraph instance
    """
    if not _core_available:
        raise ImportError("Core KG classes not available")
    
    return KnowledgeGraph(data=data, **kwargs)


def analyze_ontology(knowledge_graph):
    """
    Create an OntologyAnalyzer for a knowledge graph
    
    Args:
        knowledge_graph: KnowledgeGraph to analyze
        
    Returns:
        OntologyAnalyzer instance
    """
    if not _ontology_available:
        raise ImportError("Ontology analysis not available")
    
    return OntologyAnalyzer(knowledge_graph)


def resolve_entities(knowledge_graph):
    """
    Create an EntityResolver for a knowledge graph
    
    Args:
        knowledge_graph: KnowledgeGraph to process
        
    Returns:
        EntityResolver instance
    """
    if not _entity_available:
        raise ImportError("Entity resolution not available")
    
    return EntityResolver(knowledge_graph)


# Module information
def get_module_info():
    """Get information about available KG components"""
    
    info = {
        'version': __version__,
        'available_components': {
            'core': _core_available,
            'query': _query_available,
            'ontology': _ontology_available,
            'entity': _entity_available,
            'reasoning': _reasoning_available
        },
        'implemented_features': [],
        'planned_features': [
            'Validation framework',
            'ML integration and embeddings',
            'Temporal analysis',
            'Advanced I/O operations',
            'Visualization capabilities'
        ]
    }
    
    if _core_available:
        info['implemented_features'].append('Core KG classes with semantic hypergraphs')
    if _query_available:
        info['implemented_features'].append('Semantic query engine with pattern matching')
    if _ontology_available:
        info['implemented_features'].append('Domain-agnostic ontology analysis')
    if _entity_available:
        info['implemented_features'].append('Entity resolution and duplicate detection')
    if _reasoning_available:
        info['implemented_features'].append('Path reasoning and inference engine')
    
    return info


# Quick start documentation
QUICK_START = """
ANANT Knowledge Graph Quick Start
================================

# Create a knowledge graph
from anant.kg import create_knowledge_graph

kg = create_knowledge_graph(your_data)

# Analyze ontology structure (domain-agnostic)
from anant.kg import analyze_ontology

analyzer = analyze_ontology(kg)
classes = analyzer.analyze_class_hierarchy()
stats = analyzer.calculate_ontology_statistics()

# Semantic querying
results = kg.query.sparql_like_query('''
    SELECT ?entity ?type WHERE {
        ?entity rdf:type ?type
    }
''')

# Entity resolution
from anant.kg import resolve_entities

resolver = resolve_entities(kg)
duplicates = resolver.find_duplicates()
clusters = resolver.cluster_duplicates(duplicates)

# Path reasoning
paths = kg.reasoning.find_paths("entity1", "entity2", max_length=3)

# Get comprehensive statistics
domain_analysis = analyzer.get_domain_analysis()
"""


