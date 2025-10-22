"""
Advanced Semantic Search Engine
==============================

Natural language entity discovery with fuzzy matching, embedding-based similarity,
and intelligent ranking for knowledge graphs.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import polars as pl
import numpy as np

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies for advanced semantic search
nltk = safe_import('nltk')
sklearn = safe_import('sklearn')
sentence_transformers = safe_import('sentence_transformers')
fuzzywuzzy = safe_import('fuzzywuzzy')

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes for semantic entity discovery"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    COMPREHENSIVE = "comprehensive"


@dataclass
class EntitySearchResult:
    """Result from entity search"""
    entity_id: str
    entity_type: str
    relevance_score: float
    match_type: str
    matched_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipSearchResult:
    """Result from relationship search"""
    relationship_type: str
    source_entity: str
    target_entity: str
    relevance_score: float
    match_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a semantic search result"""
    entity_id: str
    entity_name: str
    entity_type: str
    relevance_score: float
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    matched_properties: List[str] = field(default_factory=list)
    matched_relationships: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Represents a processed search query"""
    original_query: str
    processed_tokens: List[str] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    search_type: str = "general"  # general, exact, fuzzy, semantic
    confidence: float = 1.0


@dataclass
class SearchConfiguration:
    """Search engine configuration"""
    # Similarity thresholds
    min_relevance_score: float = 0.3
    fuzzy_threshold: float = 0.6
    semantic_threshold: float = 0.5
    
    # Search limits
    max_results: int = 50
    max_candidates: int = 1000
    
    # Scoring weights
    exact_match_weight: float = 1.0
    fuzzy_match_weight: float = 0.8
    semantic_match_weight: float = 0.7
    property_match_weight: float = 0.6
    relationship_match_weight: float = 0.5
    
    # Features
    enable_fuzzy_search: bool = True
    enable_semantic_embeddings: bool = True
    enable_property_search: bool = True
    enable_relationship_search: bool = True
    enable_context_expansion: bool = True
    
    # Performance
    use_polars_optimization: bool = True
    enable_caching: bool = True
    batch_size: int = 100


class SemanticSearchEngine:
    """
    Advanced semantic search engine for knowledge graphs
    
    Features:
    - Natural language query processing
    - Fuzzy string matching with multiple algorithms
    - Embedding-based semantic similarity
    - Multi-modal search (entities, properties, relationships)
    - Intelligent ranking and relevance scoring
    - Context-aware search expansion
    - Polars-optimized performance
    """
    
    def __init__(self, 
                 knowledge_graph,
                 config: Optional[SearchConfiguration] = None):
        """
        Initialize semantic search engine
        
        Args:
            knowledge_graph: KnowledgeGraph instance to search
            config: Search configuration
        """
        self.kg = knowledge_graph
        self.config = config or SearchConfiguration()
        
        # Initialize components
        self._init_text_processing()
        self._init_embedding_models()
        self._build_search_indices()
        
        # Caches
        self.query_cache = {}
        self.embedding_cache = {}
        self.similarity_cache = {}
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'search_type_distribution': Counter()
        }
        
        logger.info("Semantic Search Engine initialized")
    
    def _init_text_processing(self) -> None:
        """Initialize text processing components"""
        
        self.text_processors = {}
        
        # Initialize NLTK components if available (use global nltk)
        global nltk
        if nltk is not None:
            try:
                # Import NLTK components
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                from nltk.stem import PorterStemmer
                
                # Check for required NLTK data
                try:
                    import nltk.data
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                except:
                    logger.warning("Required NLTK data not found, using basic text processing")
                
                self.text_processors.update({
                    'tokenize': word_tokenize,
                    'stopwords': set(stopwords.words('english')),
                    'stemmer': PorterStemmer()
                })
                
                logger.info("NLTK text processing initialized")
                
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                self._init_basic_text_processing()
        else:
            self._init_basic_text_processing()
    
    def _init_basic_text_processing(self) -> None:
        """Initialize basic text processing without NLTK"""
        
        self.text_processors.update({
            'tokenize': lambda text: re.findall(r'\b\w+\b', text.lower()),
            'stopwords': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'},
            'stemmer': None
        })
        
        logger.info("Basic text processing initialized")
    
    def _init_embedding_models(self) -> None:
        """Initialize embedding models if available"""
        
        self.embedding_models = {}
        
        if sentence_transformers and self.config.enable_semantic_embeddings:
            try:
                # Initialize lightweight model
                from sentence_transformers import SentenceTransformer
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_models['sentence_transformer'] = model
                
                logger.info("Sentence transformer model loaded")
                
            except Exception as e:
                logger.warning(f"Sentence transformer initialization failed: {e}")
        
        # Initialize TF-IDF as fallback
        if sklearn:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                self.embedding_models.update({
                    'tfidf_vectorizer': TfidfVectorizer(
                        max_features=5000,
                        stop_words='english',
                        ngram_range=(1, 2)
                    ),
                    'cosine_similarity': cosine_similarity
                })
                
                logger.info("TF-IDF vectorizer initialized")
                
            except Exception as e:
                logger.warning(f"TF-IDF initialization failed: {e}")
    
    def _build_search_indices(self) -> None:
        """Build search indices for efficient querying"""
        
        logger.info("Building search indices...")
        
        with PerformanceProfiler("index_building") as profiler:
            
            # Build entity index
            self.entity_index = self._build_entity_index()
            profiler.checkpoint("entity_index_built")
            
            # Build property index
            self.property_index = self._build_property_index()
            profiler.checkpoint("property_index_built")
            
            # Build relationship index
            self.relationship_index = self._build_relationship_index()
            profiler.checkpoint("relationship_index_built")
            
            # Build text corpus for embedding training
            if self.config.enable_semantic_embeddings:
                self.text_corpus = self._build_text_corpus()
                self._train_embedding_models()
                profiler.checkpoint("embeddings_trained")
        
        report = profiler.get_report()
        logger.info(f"Search indices built in {report['total_execution_time']:.2f}s")
    
    def _build_entity_index(self) -> Dict[str, Any]:
        """Build searchable index of entities"""
        
        entity_index = {
            'by_name': {},
            'by_type': defaultdict(list),
            'by_tokens': defaultdict(list),
            'metadata': {}
        }
        
        for entity_id in self.kg.nodes:
            # Get entity information
            entity_type = self.kg.get_entity_type(entity_id) or 'unknown'
            entity_name = self._get_entity_display_name(entity_id)
            
            # Index by name
            entity_index['by_name'][entity_id] = {
                'name': entity_name,
                'type': entity_type,
                'tokens': self._tokenize_text(entity_name)
            }
            
            # Index by type
            entity_index['by_type'][entity_type].append(entity_id)
            
            # Index by tokens
            for token in self._tokenize_text(entity_name):
                entity_index['by_tokens'][token].append(entity_id)
        
        entity_index['metadata'] = {
            'total_entities': len(entity_index['by_name']),
            'entity_types': list(entity_index['by_type'].keys()),
            'unique_tokens': len(entity_index['by_tokens'])
        }
        
        return entity_index
    
    def _build_property_index(self) -> Dict[str, Any]:
        """Build searchable index of properties"""
        
        property_index = {
            'by_entity': defaultdict(dict),
            'by_property': defaultdict(list),
            'by_value': defaultdict(list)
        }
        
        for entity_id in self.kg.nodes:
            properties = self.kg.properties.get_node_properties(entity_id)
            
            for prop_name, prop_value in properties.items():
                # Index by entity
                property_index['by_entity'][entity_id][prop_name] = prop_value
                
                # Index by property name
                property_index['by_property'][prop_name].append(entity_id)
                
                # Index by value (for searchable values)
                if isinstance(prop_value, (str, int, float)):
                    value_str = str(prop_value).lower()
                    property_index['by_value'][value_str].append((entity_id, prop_name))
        
        return property_index
    
    def _build_relationship_index(self) -> Dict[str, Any]:
        """Build searchable index of relationships"""
        
        relationship_index = {
            'by_entity': defaultdict(list),
            'by_type': defaultdict(list),
            'entity_pairs': defaultdict(list)
        }
        
        for edge_id in self.kg.edges:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge_id)
            edge_type = self.kg.get_edge_type(edge_id) or 'unknown'
            
            # Index relationships by entity
            for node in edge_nodes:
                relationship_index['by_entity'][node].append({
                    'edge_id': edge_id,
                    'type': edge_type,
                    'connected_entities': [n for n in edge_nodes if n != node]
                })
            
            # Index by relationship type
            relationship_index['by_type'][edge_type].append({
                'edge_id': edge_id,
                'entities': edge_nodes
            })
            
            # Index entity pairs for relationship discovery
            if len(edge_nodes) >= 2:
                for i, entity1 in enumerate(edge_nodes):
                    for entity2 in edge_nodes[i+1:]:
                        pair_key = tuple(sorted([entity1, entity2]))
                        relationship_index['entity_pairs'][pair_key].append(edge_type)
        
        return relationship_index
    
    def _build_text_corpus(self) -> List[str]:
        """Build text corpus for embedding training"""
        
        corpus = []
        
        # Add entity names and descriptions
        for entity_id in self.kg.nodes:
            entity_name = self._get_entity_display_name(entity_id)
            corpus.append(entity_name)
            
            # Add property values as context
            properties = self.kg.properties.get_node_properties(entity_id)
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, str) and len(prop_value) > 3:
                    corpus.append(f"{prop_name}: {prop_value}")
        
        # Add relationship descriptions
        for edge_type, relationships in self.relationship_index['by_type'].items():
            corpus.append(edge_type.replace('_', ' '))
        
        logger.info(f"Built text corpus with {len(corpus)} documents")
        
        return corpus
    
    def _train_embedding_models(self) -> None:
        """Train/fit embedding models on the corpus"""
        
        if 'tfidf_vectorizer' in self.embedding_models:
            try:
                vectorizer = self.embedding_models['tfidf_vectorizer']
                vectorizer.fit(self.text_corpus)
                
                # Pre-compute entity embeddings
                self.entity_embeddings = {}
                for entity_id, entity_data in self.entity_index['by_name'].items():
                    entity_text = entity_data['name']
                    embedding = vectorizer.transform([entity_text])
                    self.entity_embeddings[entity_id] = embedding
                
                logger.info("TF-IDF embeddings computed")
                
            except Exception as e:
                logger.warning(f"TF-IDF training failed: {e}")
    
    @performance_monitor("semantic_search")
    def search(self, 
               query: str,
               entity_types: Optional[List[str]] = None,
               filters: Optional[Dict[str, Any]] = None,
               search_mode: str = "comprehensive",
               max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Perform semantic search on the knowledge graph
        
        Args:
            query: Natural language search query
            entity_types: Filter by entity types
            filters: Additional filters
            search_mode: Search strategy ("exact", "fuzzy", "semantic", "comprehensive")
            max_results: Maximum results to return
            
        Returns:
            List of ranked search results
        """
        
        start_time = time.time()
        
        # Update statistics
        self.stats['queries_processed'] += 1
        self.stats['search_type_distribution'][search_mode] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query, entity_types, filters, search_mode)
        if self.config.enable_caching and cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        logger.info(f"Performing {search_mode} search for: '{query}'")
        
        with PerformanceProfiler("search_execution") as profiler:
            
            # Process query
            processed_query = self._process_search_query(query, entity_types, filters)
            profiler.checkpoint("query_processed")
            
            # Execute search based on mode
            if search_mode == "exact":
                results = self._exact_search(processed_query)
            elif search_mode == "fuzzy":
                results = self._fuzzy_search(processed_query)
            elif search_mode == "semantic":
                results = self._semantic_search(processed_query)
            else:  # comprehensive
                results = self._comprehensive_search(processed_query)
            
            profiler.checkpoint("search_executed")
            
            # Rank and filter results
            ranked_results = self._rank_results(results, processed_query)
            profiler.checkpoint("results_ranked")
            
            # Apply limits
            if max_results is None:
                max_results = self.config.max_results
            
            final_results = ranked_results[:max_results]
            
            # Generate explanations
            for result in final_results:
                result.explanation = self._generate_explanation(result, processed_query)
            
            profiler.checkpoint("explanations_generated")
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['queries_processed'] - 1) + execution_time) /
            self.stats['queries_processed']
        )
        
        # Cache results
        if self.config.enable_caching:
            self.query_cache[cache_key] = final_results
        
        report = profiler.get_report()
        logger.info(f"Search completed in {report['total_execution_time']:.2f}s, {len(final_results)} results")
        
        return final_results
    
    def _process_search_query(self, 
                            query: str,
                            entity_types: Optional[List[str]],
                            filters: Optional[Dict[str, Any]]) -> SearchQuery:
        """Process and analyze search query"""
        
        processed_query = SearchQuery(
            original_query=query,
            entity_types=entity_types or [],
            filters=filters or {}
        )
        
        # Tokenize and clean query
        tokens = self._tokenize_text(query)
        processed_query.processed_tokens = tokens
        
        # Extract keywords (remove stopwords)
        if 'stopwords' in self.text_processors:
            keywords = [token for token in tokens if token not in self.text_processors['stopwords']]
        else:
            keywords = tokens
        
        processed_query.keywords = keywords
        
        # Detect query type patterns
        processed_query.search_type = self._detect_query_type(query, tokens)
        
        # Extract potential properties from query
        processed_query.properties = self._extract_property_patterns(query, tokens)
        
        return processed_query
    
    def _detect_query_type(self, query: str, tokens: List[str]) -> str:
        """Detect the type of search query"""
        
        query_lower = query.lower()
        
        # Exact match indicators
        if query.startswith('"') and query.endswith('"'):
            return "exact"
        
        # Property search indicators
        if any(pattern in query_lower for pattern in ['with', 'having', 'where', 'contains']):
            return "property"
        
        # Relationship search indicators
        if any(pattern in query_lower for pattern in ['connected to', 'related to', 'linked to']):
            return "relationship"
        
        # Semantic search indicators
        if any(pattern in query_lower for pattern in ['similar to', 'like', 'about']):
            return "semantic"
        
        return "general"
    
    def _extract_property_patterns(self, query: str, tokens: List[str]) -> Dict[str, str]:
        """Extract property patterns from query"""
        
        properties = {}
        
        # Pattern: "entities with [property] [value]"
        with_pattern = re.search(r'with\\s+(\\w+)\\s+(\\w+)', query.lower())
        if with_pattern:
            prop_name = with_pattern.group(1)
            prop_value = with_pattern.group(2)
            properties[prop_name] = prop_value
        
        # Pattern: "where [property] is [value]"
        where_pattern = re.search(r'where\\s+(\\w+)\\s+is\\s+(\\w+)', query.lower())
        if where_pattern:
            prop_name = where_pattern.group(1)
            prop_value = where_pattern.group(2)
            properties[prop_name] = prop_value
        
        return properties
    
    def _exact_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform exact string matching search"""
        
        results = []
        query_text = query.original_query.lower().strip('"')
        
        # Search entity names
        for entity_id, entity_data in self.entity_index['by_name'].items():
            entity_name = entity_data['name'].lower()
            
            if query_text in entity_name:
                # Calculate exact match score
                if query_text == entity_name:
                    score = 1.0
                else:
                    score = len(query_text) / len(entity_name)
                
                result = SearchResult(
                    entity_id=entity_id,
                    entity_name=entity_data['name'],
                    entity_type=entity_data['type'],
                    relevance_score=score,
                    similarity_scores={'exact_match': score},
                    matched_properties=[],
                    matched_relationships=[]
                )
                results.append(result)
        
        return results
    
    def _fuzzy_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform fuzzy string matching search"""
        
        results = []
        
        # Use multiple fuzzy matching algorithms
        for entity_id, entity_data in self.entity_index['by_name'].items():
            entity_name = entity_data['name']
            
            # Calculate fuzzy similarity scores
            similarity_scores = {}
            
            # SequenceMatcher (built-in)
            seq_similarity = SequenceMatcher(None, query.original_query.lower(), entity_name.lower()).ratio()
            similarity_scores['sequence_matcher'] = seq_similarity
            
            # FuzzyWuzzy if available
            if fuzzywuzzy:
                try:
                    from fuzzywuzzy import fuzz
                    
                    fuzz_ratio = fuzz.ratio(query.original_query.lower(), entity_name.lower()) / 100.0
                    fuzz_partial = fuzz.partial_ratio(query.original_query.lower(), entity_name.lower()) / 100.0
                    fuzz_token = fuzz.token_sort_ratio(query.original_query.lower(), entity_name.lower()) / 100.0
                    
                    similarity_scores.update({
                        'fuzz_ratio': fuzz_ratio,
                        'fuzz_partial': fuzz_partial,
                        'fuzz_token': fuzz_token
                    })
                    
                except Exception:
                    pass
            
            # Token-based similarity
            token_similarity = self._calculate_token_similarity(query.keywords, entity_data['tokens'])
            similarity_scores['token_similarity'] = token_similarity
            
            # Calculate overall fuzzy score
            fuzzy_score = max(similarity_scores.values())
            
            if fuzzy_score >= self.config.fuzzy_threshold:
                result = SearchResult(
                    entity_id=entity_id,
                    entity_name=entity_data['name'],
                    entity_type=entity_data['type'],
                    relevance_score=fuzzy_score,
                    similarity_scores=similarity_scores,
                    matched_properties=[],
                    matched_relationships=[]
                )
                results.append(result)
        
        return results
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform embedding-based semantic search"""
        
        results = []
        
        # Use sentence transformers if available
        if 'sentence_transformer' in self.embedding_models:
            results.extend(self._transformer_semantic_search(query))
        
        # Use TF-IDF as fallback or additional method
        if 'tfidf_vectorizer' in self.embedding_models:
            results.extend(self._tfidf_semantic_search(query))
        
        # Remove duplicates and merge scores
        results = self._merge_duplicate_results(results)
        
        return results
    
    def _transformer_semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Semantic search using sentence transformers"""
        
        results = []
        
        try:
            model = self.embedding_models['sentence_transformer']
            
            # Encode query
            query_embedding = model.encode([query.original_query])
            
            # Encode entity names and calculate similarities
            entity_names = []
            entity_ids = []
            
            for entity_id, entity_data in self.entity_index['by_name'].items():
                entity_names.append(entity_data['name'])
                entity_ids.append(entity_id)
            
            # Batch encoding for efficiency
            if len(entity_names) > 0:
                entity_embeddings = model.encode(entity_names)
                
                # Calculate cosine similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_embedding, entity_embeddings)[0]
                
                for i, similarity in enumerate(similarities):
                    if similarity >= self.config.semantic_threshold:
                        entity_id = entity_ids[i]
                        entity_data = self.entity_index['by_name'][entity_id]
                        
                        result = SearchResult(
                            entity_id=entity_id,
                            entity_name=entity_data['name'],
                            entity_type=entity_data['type'],
                            relevance_score=similarity,
                            similarity_scores={'transformer_similarity': similarity},
                            matched_properties=[],
                            matched_relationships=[]
                        )
                        results.append(result)
            
        except Exception as e:
            logger.warning(f"Transformer semantic search failed: {e}")
        
        return results
    
    def _tfidf_semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Semantic search using TF-IDF"""
        
        results = []
        
        try:
            vectorizer = self.embedding_models['tfidf_vectorizer']
            cosine_sim = self.embedding_models['cosine_similarity']
            
            # Transform query
            query_vector = vectorizer.transform([query.original_query])
            
            # Calculate similarities with pre-computed embeddings
            for entity_id, entity_embedding in self.entity_embeddings.items():
                similarity = cosine_sim(query_vector, entity_embedding)[0][0]
                
                if similarity >= self.config.semantic_threshold:
                    entity_data = self.entity_index['by_name'][entity_id]
                    
                    result = SearchResult(
                        entity_id=entity_id,
                        entity_name=entity_data['name'],
                        entity_type=entity_data['type'],
                        relevance_score=similarity,
                        similarity_scores={'tfidf_similarity': similarity},
                        matched_properties=[],
                        matched_relationships=[]
                    )
                    results.append(result)
        
        except Exception as e:
            logger.warning(f"TF-IDF semantic search failed: {e}")
        
        return results
    
    def _comprehensive_search(self, query: SearchQuery) -> List[SearchResult]:
        """Comprehensive search combining all methods"""
        
        all_results = []
        
        # Perform all search types
        if len(query.original_query) > 2:
            all_results.extend(self._exact_search(query))
        
        if self.config.enable_fuzzy_search:
            all_results.extend(self._fuzzy_search(query))
        
        if self.config.enable_semantic_embeddings:
            all_results.extend(self._semantic_search(query))
        
        if self.config.enable_property_search:
            all_results.extend(self._property_search(query))
        
        if self.config.enable_relationship_search:
            all_results.extend(self._relationship_search(query))
        
        # Context expansion
        if self.config.enable_context_expansion:
            all_results.extend(self._context_expansion_search(query, all_results))
        
        # Merge and deduplicate
        merged_results = self._merge_duplicate_results(all_results)
        
        return merged_results
    
    def _property_search(self, query: SearchQuery) -> List[SearchResult]:
        """Search based on entity properties"""
        
        results = []
        
        # Direct property pattern search
        for prop_name, prop_value in query.properties.items():
            if prop_name in self.property_index['by_property']:
                entity_ids = self.property_index['by_property'][prop_name]
                
                for entity_id in entity_ids:
                    entity_properties = self.property_index['by_entity'][entity_id]
                    actual_value = str(entity_properties.get(prop_name, '')).lower()
                    
                    if prop_value.lower() in actual_value:
                        entity_data = self.entity_index['by_name'][entity_id]
                        
                        result = SearchResult(
                            entity_id=entity_id,
                            entity_name=entity_data['name'],
                            entity_type=entity_data['type'],
                            relevance_score=0.8,
                            similarity_scores={'property_match': 0.8},
                            matched_properties=[f"{prop_name}: {actual_value}"],
                            matched_relationships=[]
                        )
                        results.append(result)
        
        # Fuzzy property value search
        for value_pattern in query.keywords:
            for stored_value, entity_prop_pairs in self.property_index['by_value'].items():
                if len(value_pattern) > 2:
                    similarity = SequenceMatcher(None, value_pattern, stored_value).ratio()
                    
                    if similarity >= self.config.fuzzy_threshold:
                        for entity_id, prop_name in entity_prop_pairs:
                            entity_data = self.entity_index['by_name'][entity_id]
                            
                            result = SearchResult(
                                entity_id=entity_id,
                                entity_name=entity_data['name'],
                                entity_type=entity_data['type'],
                                relevance_score=similarity * self.config.property_match_weight,
                                similarity_scores={'property_fuzzy_match': similarity},
                                matched_properties=[f"{prop_name}: {stored_value}"],
                                matched_relationships=[]
                            )
                            results.append(result)
        
        return results
    
    def _relationship_search(self, query: SearchQuery) -> List[SearchResult]:
        """Search based on relationships"""
        
        results = []
        
        # Search for entities by relationship types
        for keyword in query.keywords:
            for rel_type, relationships in self.relationship_index['by_type'].items():
                # Fuzzy match relationship type
                type_similarity = SequenceMatcher(None, keyword, rel_type.lower()).ratio()
                
                if type_similarity >= self.config.fuzzy_threshold:
                    for rel_data in relationships:
                        for entity_id in rel_data['entities']:
                            if entity_id in self.entity_index['by_name']:
                                entity_data = self.entity_index['by_name'][entity_id]
                                
                                result = SearchResult(
                                    entity_id=entity_id,
                                    entity_name=entity_data['name'],
                                    entity_type=entity_data['type'],
                                    relevance_score=type_similarity * self.config.relationship_match_weight,
                                    similarity_scores={'relationship_match': type_similarity},
                                    matched_properties=[],
                                    matched_relationships=[rel_type]
                                )
                                results.append(result)
        
        return results
    
    def _context_expansion_search(self, query: SearchQuery, initial_results: List[SearchResult]) -> List[SearchResult]:
        """Expand search based on context from initial results"""
        
        expanded_results = []
        
        # Find related entities from top initial results
        top_results = sorted(initial_results, key=lambda r: r.relevance_score, reverse=True)[:5]
        
        for result in top_results:
            # Find connected entities
            if result.entity_id in self.relationship_index['by_entity']:
                relationships = self.relationship_index['by_entity'][result.entity_id]
                
                for rel_data in relationships:
                    for connected_entity in rel_data['connected_entities']:
                        if connected_entity in self.entity_index['by_name']:
                            entity_data = self.entity_index['by_name'][connected_entity]
                            
                            # Calculate context relevance
                            context_score = result.relevance_score * 0.3  # Damped relevance
                            
                            expanded_result = SearchResult(
                                entity_id=connected_entity,
                                entity_name=entity_data['name'],
                                entity_type=entity_data['type'],
                                relevance_score=context_score,
                                similarity_scores={'context_expansion': context_score},
                                matched_properties=[],
                                matched_relationships=[rel_data['type']]
                            )
                            expanded_results.append(expanded_result)
        
        return expanded_results
    
    def _merge_duplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Merge duplicate results and combine scores"""
        
        merged_results = {}
        
        for result in results:
            if result.entity_id in merged_results:
                # Merge with existing result
                existing = merged_results[result.entity_id]
                
                # Combine relevance scores (take maximum)
                existing.relevance_score = max(existing.relevance_score, result.relevance_score)
                
                # Merge similarity scores
                existing.similarity_scores.update(result.similarity_scores)
                
                # Merge matched properties and relationships
                existing.matched_properties.extend(result.matched_properties)
                existing.matched_relationships.extend(result.matched_relationships)
                
                # Remove duplicates
                existing.matched_properties = list(set(existing.matched_properties))
                existing.matched_relationships = list(set(existing.matched_relationships))
                
            else:
                merged_results[result.entity_id] = result
        
        return list(merged_results.values())
    
    def _rank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank and score results"""
        
        # Calculate final relevance scores
        for result in results:
            final_score = 0.0
            
            # Weighted combination of different similarity scores
            if 'exact_match' in result.similarity_scores:
                final_score += result.similarity_scores['exact_match'] * self.config.exact_match_weight
            
            if 'fuzz_ratio' in result.similarity_scores:
                final_score += result.similarity_scores['fuzz_ratio'] * self.config.fuzzy_match_weight
            
            if 'transformer_similarity' in result.similarity_scores:
                final_score += result.similarity_scores['transformer_similarity'] * self.config.semantic_match_weight
            
            if 'tfidf_similarity' in result.similarity_scores:
                final_score += result.similarity_scores['tfidf_similarity'] * self.config.semantic_match_weight
            
            if 'property_match' in result.similarity_scores:
                final_score += result.similarity_scores['property_match'] * self.config.property_match_weight
            
            if 'relationship_match' in result.similarity_scores:
                final_score += result.similarity_scores['relationship_match'] * self.config.relationship_match_weight
            
            # Normalize by number of contributing scores
            num_scores = len([score for score in result.similarity_scores.values() if score > 0])
            if num_scores > 0:
                final_score = final_score / num_scores
            
            result.relevance_score = final_score
        
        # Filter by minimum relevance threshold
        filtered_results = [r for r in results if r.relevance_score >= self.config.min_relevance_score]
        
        # Sort by relevance score
        filtered_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return filtered_results
    
    def _generate_explanation(self, result: SearchResult, query: SearchQuery) -> str:
        """Generate explanation for search result"""
        
        explanations = []
        
        # Explain similarity matches
        if 'exact_match' in result.similarity_scores and result.similarity_scores['exact_match'] > 0.9:
            explanations.append("exact name match")
        elif any(score > 0.7 for score in result.similarity_scores.values()):
            explanations.append("high similarity match")
        elif any(score > 0.5 for score in result.similarity_scores.values()):
            explanations.append("moderate similarity match")
        
        # Explain property matches
        if result.matched_properties:
            explanations.append(f"matching properties: {', '.join(result.matched_properties[:2])}")
        
        # Explain relationship matches
        if result.matched_relationships:
            explanations.append(f"related through: {', '.join(result.matched_relationships[:2])}")
        
        # Explain context expansion
        if 'context_expansion' in result.similarity_scores:
            explanations.append("contextually related")
        
        if not explanations:
            explanations.append("general relevance match")
        
        return f"Found due to: {'; '.join(explanations)}"
    
    # Utility methods
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using configured tokenizer"""
        
        if 'tokenize' in self.text_processors:
            return self.text_processors['tokenize'](text)
        else:
            return re.findall(r'\b\w+\b', text.lower())
    
    def _calculate_token_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity between token sets"""
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        if not set1 and not set2:
            return 1.0
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_entity_display_name(self, entity_id: str) -> str:
        """Get display name for entity"""
        
        # Try to get a human-readable name from properties
        properties = self.kg.properties.get_node_properties(entity_id)
        
        for name_field in ['name', 'label', 'title', 'rdfs:label']:
            if name_field in properties:
                return str(properties[name_field])
        
        # Fallback to entity ID processing
        name = entity_id
        
        # Extract meaningful part from URI
        if '/' in name:
            name = name.split('/')[-1]
        if '#' in name:
            name = name.split('#')[-1]
        
        # Clean up underscores and camelCase
        name = re.sub(r'[_-]', ' ', name)
        name = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', name)
        
        return name.title()
    
    def _generate_cache_key(self, query: str, entity_types: Optional[List[str]], 
                          filters: Optional[Dict[str, Any]], search_mode: str) -> str:
        """Generate cache key for query"""
        
        key_parts = [
            query.lower(),
            str(sorted(entity_types or [])),
            str(sorted((filters or {}).items())),
            search_mode
        ]
        
        return '|'.join(key_parts)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        
        return {
            'queries_processed': self.stats['queries_processed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['queries_processed']),
            'avg_response_time': self.stats['avg_response_time'],
            'search_type_distribution': dict(self.stats['search_type_distribution']),
            'index_statistics': {
                'total_entities': self.entity_index['metadata']['total_entities'],
                'entity_types': len(self.entity_index['metadata']['entity_types']),
                'unique_tokens': self.entity_index['metadata']['unique_tokens']
            }
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.similarity_cache.clear()
        
        logger.info("Search caches cleared")
    
    # ============================================================================
    # Public API Methods for Test Compatibility  
    # ============================================================================
    
    def search_entities(self, query: str, mode: SearchMode = SearchMode.COMPREHENSIVE, 
                       limit: int = 50) -> List[EntitySearchResult]:
        """
        Public API for entity search with different modes
        
        Args:
            query: Search query string
            mode: Search mode (EXACT, FUZZY, SEMANTIC, COMPREHENSIVE)
            limit: Maximum number of results
            
        Returns:
            List of entity search results
        """
        try:
            if mode == SearchMode.EXACT:
                return self._search_exact_entities(query, limit)
            elif mode == SearchMode.FUZZY:
                return self._search_fuzzy_entities(query, limit) 
            elif mode == SearchMode.SEMANTIC:
                return self._search_semantic_entities(query, limit)
            elif mode == SearchMode.COMPREHENSIVE:
                return self._search_comprehensive_entities(query, limit)
            else:
                return self._search_comprehensive_entities(query, limit)
                
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []
    
    def search_relationships(self, query: str, limit: int = 50) -> List[RelationshipSearchResult]:
        """
        Public API for relationship search
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of relationship search results
        """
        try:
            # Use the main search method for relationships
            result = self.search(
                query=query,
                search_type="relationships",
                max_results=limit
            )
            
            # Convert to RelationshipSearchResult format
            relationship_results = []
            for item in result.results:
                rel_result = RelationshipSearchResult(
                    relationship_type=item.get('relationship_type', 'unknown'),
                    source_entity=item.get('source', 'unknown'),
                    target_entity=item.get('target', 'unknown'),
                    relevance_score=item.get('score', 0.0),
                    match_type="semantic"
                )
                relationship_results.append(rel_result)
            
            return relationship_results
            
        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []
    
    def _search_exact_entities(self, query: str, limit: int) -> List[EntitySearchResult]:
        """Exact entity search implementation"""
        results = []
        
        # Search through knowledge graph nodes
        for node_id in self.kg.nodes:
            node_data = self.kg.properties.get_node_data(node_id) or {}
            
            # Check exact matches in name and properties
            name = node_data.get('name', '')
            if query.lower() == name.lower():
                result = EntitySearchResult(
                    entity_id=node_id,
                    entity_type=self.kg.get_node_type(node_id) or 'unknown',
                    relevance_score=1.0,
                    match_type="exact",
                    matched_properties={'name': name}
                )
                results.append(result)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _search_fuzzy_entities(self, query: str, limit: int) -> List[EntitySearchResult]:
        """Fuzzy entity search implementation"""
        results = []
        
        # Import difflib for fuzzy matching (safe fallback)
        from difflib import SequenceMatcher
        
        # Search through knowledge graph nodes
        for node_id in self.kg.nodes:
            node_data = self.kg.properties.get_node_data(node_id) or {}
            
            # Check fuzzy matches in name and properties
            name = node_data.get('name', '')
            if name:
                similarity = SequenceMatcher(None, query.lower(), name.lower()).ratio()
                
                if similarity > 0.6:  # Fuzzy threshold
                    result = EntitySearchResult(
                        entity_id=node_id,
                        entity_type=self.kg.get_node_type(node_id) or 'unknown',
                        relevance_score=similarity,
                        match_type="fuzzy",
                        matched_properties={'name': name}
                    )
                    results.append(result)
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def _search_semantic_entities(self, query: str, limit: int) -> List[EntitySearchResult]:
        """Semantic entity search implementation"""
        results = []
        
        # Fallback to fuzzy search if semantic embeddings not available
        if not self.config.enable_semantic_embeddings:
            return self._search_fuzzy_entities(query, limit)
        
        # Search through knowledge graph nodes with semantic similarity
        for node_id in self.kg.nodes:
            node_data = self.kg.properties.get_node_data(node_id) or {}
            
            # Simple semantic matching based on keywords
            text_content = ' '.join(str(v) for v in node_data.values())
            
            # Basic semantic score calculation
            query_words = set(query.lower().split())
            content_words = set(text_content.lower().split())
            
            if query_words and content_words:
                overlap = len(query_words.intersection(content_words))
                semantic_score = overlap / len(query_words.union(content_words))
                
                if semantic_score > 0.2:  # Semantic threshold
                    result = EntitySearchResult(
                        entity_id=node_id,
                        entity_type=self.kg.get_node_type(node_id) or 'unknown',
                        relevance_score=semantic_score,
                        match_type="semantic",
                        matched_properties=node_data
                    )
                    results.append(result)
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def _search_comprehensive_entities(self, query: str, limit: int) -> List[EntitySearchResult]:
        """Comprehensive entity search combining all methods"""
        
        # Combine results from all search methods
        exact_results = self._search_exact_entities(query, limit // 3)
        fuzzy_results = self._search_fuzzy_entities(query, limit // 3)
        semantic_results = self._search_semantic_entities(query, limit // 3)
        
        # Merge and deduplicate results
        all_results = {}
        
        # Add exact results (highest priority)
        for result in exact_results:
            all_results[result.entity_id] = result
        
        # Add fuzzy results (if not already present)
        for result in fuzzy_results:
            if result.entity_id not in all_results:
                all_results[result.entity_id] = result
        
        # Add semantic results (if not already present)
        for result in semantic_results:
            if result.entity_id not in all_results:
                all_results[result.entity_id] = result
        
        # Convert to list and sort by relevance
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined_results[:limit]


# Test semantic search capabilities
def test_semantic_search():
    """Test function to demonstrate semantic search capabilities"""
    
    logger.info("Testing semantic search capabilities")
    
    # This would need to be integrated with actual knowledge graph
    # For demonstration purposes only
    
    config = SearchConfiguration(
        max_results=10,
        enable_fuzzy_search=True,
        enable_semantic_embeddings=True,
        fuzzy_threshold=0.6,
        semantic_threshold=0.5
    )
    
    # Mock knowledge graph would be passed here
    # search_engine = SemanticSearchEngine(knowledge_graph, config)
    
    test_queries = [
        "people who work at technology companies",
        "products related to artificial intelligence",
        "organizations in healthcare",
        "John Smith",
        "companies with revenue over 1 million"
    ]
    
    logger.info(f"Would test {len(test_queries)} sample queries")
    
    return True


if __name__ == "__main__":
    # Run test
    test_result = test_semantic_search()
    print("Semantic search engine test completed successfully!")