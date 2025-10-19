"""
Natural Language Interface for Knowledge Graphs
==============================================

Advanced natural language to formal query translation system with context management,
conversation memory, intent recognition, and semantic understanding capabilities.
"""

import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timedelta
import threading

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import
from .federated_query import FederatedQueryEngine, FederatedQueryResult
from .query_optimization import QueryOptimizer

# Optional dependencies for advanced NLP
transformers = safe_import('transformers')
torch = safe_import('torch')
spacy = safe_import('spacy')
nltk = safe_import('nltk')

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the NL interface can handle"""
    ENTITY_SEARCH = "entity_search"
    RELATIONSHIP_QUERY = "relationship_query"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class Intent(Enum):
    """User intents for natural language queries"""
    FIND = "find"
    COUNT = "count"
    LIST = "list"
    DESCRIBE = "describe"
    COMPARE = "compare"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    FILTER = "filter"
    RELATE = "relate"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for query interpretation"""
    HIGH = "high"        # > 0.8
    MEDIUM = "medium"    # 0.5 - 0.8
    LOW = "low"         # 0.2 - 0.5
    VERY_LOW = "very_low"  # < 0.2


@dataclass
class EntityMention:
    """Detected entity mention in natural language"""
    text: str
    start_pos: int
    end_pos: int
    entity_type: Optional[str] = None
    confidence: float = 0.0
    uri: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class RelationMention:
    """Detected relation mention in natural language"""
    text: str
    start_pos: int
    end_pos: int
    relation_type: Optional[str] = None
    confidence: float = 0.0
    direction: str = "forward"  # forward, backward, bidirectional


@dataclass
class QueryInterpretation:
    """Interpretation of a natural language query"""
    intent: Intent
    query_type: QueryType
    confidence: float
    entities: List[EntityMention]
    relations: List[RelationMention]
    constraints: Dict[str, Any]
    temporal_info: Dict[str, Any]
    formal_query: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class ConversationContext:
    """Context information for conversation continuity"""
    conversation_id: str
    user_id: Optional[str]
    previous_queries: deque = field(default_factory=lambda: deque(maxlen=10))
    entity_context: Dict[str, Any] = field(default_factory=dict)
    topic_context: Set[str] = field(default_factory=set)
    last_interaction: datetime = field(default_factory=datetime.now)
    session_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result from natural language query processing"""
    query_id: str
    original_query: str
    interpretation: QueryInterpretation
    formal_query: str
    execution_result: Optional[FederatedQueryResult] = None
    response_text: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NaturalLanguageInterface:
    """
    Advanced Natural Language Interface for Knowledge Graphs
    
    Features:
    - Natural language to SPARQL/Cypher translation
    - Context-aware conversation management
    - Entity and relation recognition
    - Intent classification and understanding
    - Interactive query refinement
    - Multi-turn conversation support
    - Semantic understanding with fallbacks
    - Integration with federated query engine
    """
    
    def __init__(self, 
                 knowledge_graph=None,
                 federated_engine: Optional[FederatedQueryEngine] = None,
                 use_transformers: bool = True):
        """
        Initialize Natural Language Interface
        
        Args:
            knowledge_graph: Primary knowledge graph instance
            federated_engine: Federated query engine for distributed queries
            use_transformers: Whether to use transformer models for NLP
        """
        self.kg = knowledge_graph
        self.federated_engine = federated_engine
        self.use_transformers = use_transformers and transformers is not None
        
        # Query processing components
        self.query_optimizer = QueryOptimizer(knowledge_graph) if knowledge_graph else None
        
        # NLP models and tools
        self._nlp_models = {}
        self._entity_recognizer = None
        self._intent_classifier = None
        self._semantic_parser = None
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self.conversation_lock = threading.RLock()
        
        # Query templates and patterns
        self.query_templates = self._load_query_templates()
        self.entity_patterns = self._load_entity_patterns()
        self.relation_patterns = self._load_relation_patterns()
        
        # Caching and performance
        self.interpretation_cache: Dict[str, QueryInterpretation] = {}
        self.query_cache: Dict[str, str] = {}
        
        # Configuration
        self.config = {
            'max_conversation_history': 10,
            'context_window_minutes': 30,
            'confidence_threshold': 0.5,
            'max_suggestions': 3,
            'enable_caching': True,
            'cache_ttl_hours': 24,
            'fallback_to_simple': True,
            'use_context_enhancement': True
        }
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'successful_interpretations': 0,
            'cache_hits': 0,
            'average_confidence': 0.0,
            'intent_distribution': defaultdict(int),
            'query_type_distribution': defaultdict(int)
        }
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        logger.info("Natural Language Interface initialized")
    
    def _initialize_nlp_components(self):
        """Initialize NLP models and components"""
        
        logger.info("Initializing NLP components...")
        
        try:
            # Initialize spaCy if available
            if spacy:
                try:
                    self._nlp_models['spacy'] = spacy.load('en_core_web_sm')
                    logger.info("SpaCy model loaded successfully")
                except OSError:
                    logger.warning("SpaCy model 'en_core_web_sm' not found. Using fallback.")
                    self._nlp_models['spacy'] = None
            
            # Initialize transformers if requested and available
            if self.use_transformers and transformers:
                try:
                    # Load BERT for semantic understanding
                    self._nlp_models['tokenizer'] = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
                    self._nlp_models['model'] = transformers.AutoModel.from_pretrained('bert-base-uncased')
                    logger.info("Transformer models loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load transformer models: {e}")
                    self.use_transformers = False
            
            # Initialize intent classifier
            self._intent_classifier = self._create_intent_classifier()
            
            # Initialize entity recognizer
            self._entity_recognizer = self._create_entity_recognizer()
            
            # Initialize semantic parser
            self._semantic_parser = self._create_semantic_parser()
            
        except Exception as e:
            logger.warning(f"NLP initialization had issues: {e}")
            logger.info("Falling back to rule-based processing")
    
    def _create_intent_classifier(self):
        """Create intent classification component"""
        
        # Intent patterns for rule-based classification
        intent_patterns = {
            Intent.FIND: [
                r'\b(find|search|get|retrieve|show|display)\b',
                r'\bwhat\s+(is|are)\b',
                r'\bwho\s+(is|are)\b',
                r'\bwhich\b'
            ],
            Intent.COUNT: [
                r'\b(count|how many|number of)\b',
                r'\btotal\b.*\b(of|count)\b'
            ],
            Intent.LIST: [
                r'\b(list|enumerate|show all)\b',
                r'\bgive me (all|every)\b'
            ],
            Intent.DESCRIBE: [
                r'\b(describe|explain|tell me about)\b',
                r'\bwhat.*\b(about|regarding)\b'
            ],
            Intent.COMPARE: [
                r'\b(compare|difference|versus|vs|against)\b',
                r'\bbetter than\b',
                r'\bmore.*than\b'
            ],
            Intent.ANALYZE: [
                r'\b(analyze|analysis|examine|study)\b',
                r'\bpattern|trend|relationship\b'
            ],
            Intent.SUMMARIZE: [
                r'\b(summarize|summary|overview)\b',
                r'\bmain points\b'
            ],
            Intent.FILTER: [
                r'\bwhere\b',
                r'\bwith\b.*\b(property|attribute|characteristic)\b',
                r'\bthat\s+(have|has|are|is)\b'
            ],
            Intent.RELATE: [
                r'\b(related|connected|linked|associated)\b',
                r'\brelationship|connection\b'
            ]
        }
        
        return intent_patterns
    
    def _create_entity_recognizer(self):
        """Create entity recognition component"""
        
        # Entity type patterns
        entity_patterns = {
            'Person': [
                r'\b(person|people|individual|human|man|woman|guy|girl)\b',
                r'\b(he|she|they|who)\b',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Name pattern
            ],
            'Organization': [
                r'\b(company|organization|corporation|firm|business|org)\b',
                r'\b(university|college|school|institute)\b',
                r'\b(government|agency|department)\b'
            ],
            'Location': [
                r'\b(place|location|city|country|state|region)\b',
                r'\b(where|address|geographic)\b'
            ],
            'Event': [
                r'\b(event|meeting|conference|celebration)\b',
                r'\b(when|date|time|occurred|happened)\b'
            ],
            'Concept': [
                r'\b(concept|idea|theory|principle)\b',
                r'\b(type|kind|category|class)\b'
            ]
        }
        
        return entity_patterns
    
    def _create_semantic_parser(self):
        """Create semantic parsing component"""
        
        # SPARQL query templates
        sparql_templates = {
            'simple_entity_search': """
                SELECT DISTINCT ?entity ?label WHERE {{
                    ?entity rdf:type {entity_type} .
                    ?entity rdfs:label ?label .
                    {filters}
                }}
                LIMIT {limit}
            """,
            'relationship_query': """
                SELECT DISTINCT ?subject ?predicate ?object WHERE {{
                    ?subject {relation} ?object .
                    {subject_constraints}
                    {object_constraints}
                }}
                LIMIT {limit}
            """,
            'count_query': """
                SELECT (COUNT(DISTINCT ?entity) as ?count) WHERE {{
                    ?entity rdf:type {entity_type} .
                    {filters}
                }}
            """,
            'property_query': """
                SELECT DISTINCT ?entity ?property ?value WHERE {{
                    ?entity rdf:type {entity_type} .
                    ?entity ?property ?value .
                    {filters}
                }}
                LIMIT {limit}
            """
        }
        
        return sparql_templates
    
    def _load_query_templates(self) -> Dict[str, str]:
        """Load query templates for different patterns"""
        
        templates = {
            # Basic patterns
            'find_entities': "SELECT ?entity WHERE { ?entity rdf:type {type} }",
            'count_entities': "SELECT (COUNT(?entity) as ?count) WHERE { ?entity rdf:type {type} }",
            'describe_entity': "DESCRIBE {entity}",
            
            # Relationship patterns
            'find_relations': "SELECT ?subject ?object WHERE { ?subject {relation} ?object }",
            'entity_properties': "SELECT ?property ?value WHERE { {entity} ?property ?value }",
            
            # Complex patterns
            'path_query': "SELECT ?start ?end WHERE { ?start {path}+ ?end }",
            'filter_query': "SELECT ?entity WHERE { ?entity rdf:type {type} . FILTER({condition}) }"
        }
        
        return templates
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity recognition patterns"""
        
        patterns = {
            'person_indicators': ['person', 'people', 'individual', 'human', 'who'],
            'organization_indicators': ['organization', 'company', 'corporation', 'firm'],
            'location_indicators': ['place', 'location', 'city', 'country', 'where'],
            'temporal_indicators': ['when', 'time', 'date', 'year', 'month', 'day'],
            'quantity_indicators': ['how many', 'count', 'number', 'amount', 'total']
        }
        
        return patterns
    
    def _load_relation_patterns(self) -> Dict[str, List[str]]:
        """Load relation recognition patterns"""
        
        patterns = {
            'ownership': ['own', 'owns', 'belong', 'belongs', 'possess', 'has'],
            'employment': ['work', 'works', 'employed', 'job', 'position'],
            'location': ['located', 'in', 'at', 'near', 'from'],
            'temporal': ['before', 'after', 'during', 'when', 'since'],
            'causation': ['cause', 'causes', 'result', 'leads to', 'because'],
            'similarity': ['like', 'similar', 'same', 'equivalent', 'equal']
        }
        
        return patterns
    
    @performance_monitor("nl_query_processing")
    def process_query(self, 
                     natural_query: str,
                     conversation_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process a natural language query and return structured results
        
        Args:
            natural_query: The natural language query
            conversation_id: Optional conversation identifier
            user_id: Optional user identifier
            context: Additional context information
            
        Returns:
            QueryResult with interpretation and execution results
        """
        
        query_id = self._generate_query_id(natural_query)
        start_time = time.time()
        
        logger.info(f"Processing NL query {query_id}: {natural_query[:100]}...")
        
        with PerformanceProfiler("nl_query_processing") as profiler:
            
            profiler.checkpoint("query_preprocessing")
            
            # Preprocess the query
            processed_query = self._preprocess_query(natural_query)
            
            # Get or create conversation context
            conv_context = self._get_conversation_context(conversation_id, user_id)
            
            profiler.checkpoint("intent_classification")
            
            # Classify intent and query type
            interpretation = self._interpret_query(processed_query, conv_context, context)
            
            profiler.checkpoint("query_translation")
            
            # Translate to formal query
            formal_query = self._translate_to_formal_query(interpretation, conv_context)
            
            profiler.checkpoint("query_execution")
            
            # Execute the query
            execution_result = None
            if formal_query and self.federated_engine:
                try:
                    execution_result = self.federated_engine.execute_federated_query(
                        formal_query,
                        query_type='sparql',
                        optimization_level=2
                    )
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
            
            profiler.checkpoint("response_generation")
            
            # Generate natural language response
            response_text = self._generate_response(interpretation, execution_result)
            
            # Generate suggestions and clarifications
            suggestions = self._generate_suggestions(interpretation, execution_result)
            clarifications = self._generate_clarifications(interpretation)
            
            profiler.checkpoint("context_update")
            
            # Update conversation context
            self._update_conversation_context(conv_context, natural_query, interpretation, execution_result)
            
            profiler.checkpoint("result_finalization")
        
        processing_time = time.time() - start_time
        
        # Create result
        result = QueryResult(
            query_id=query_id,
            original_query=natural_query,
            interpretation=interpretation,
            formal_query=formal_query or "",
            execution_result=execution_result,
            response_text=response_text,
            suggestions=suggestions,
            clarifications=clarifications,
            processing_time=processing_time,
            metadata={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'profiler_report': profiler.get_report(),
                'use_transformers': self.use_transformers,
                'confidence_level': self._get_confidence_level(interpretation.confidence)
            }
        )
        
        # Update statistics
        self._update_statistics(result)
        
        # Cache if enabled
        if self.config['enable_caching']:
            self._cache_result(natural_query, interpretation, formal_query)
        
        logger.info(f"NL query {query_id} processed in {processing_time:.3f}s with confidence {interpretation.confidence:.3f}")
        
        return result
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess natural language query"""
        
        # Basic preprocessing
        processed = query.strip().lower()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            processed = processed.replace(contraction, expansion)
        
        # Handle common query patterns
        processed = re.sub(r'\bwho are\b', 'who is', processed)
        processed = re.sub(r'\bwhat are\b', 'what is', processed)
        
        return processed
    
    def _interpret_query(self, 
                        query: str, 
                        context: ConversationContext,
                        additional_context: Optional[Dict[str, Any]] = None) -> QueryInterpretation:
        """Interpret natural language query to extract intent and structure"""
        
        # Check cache first
        cache_key = self._get_interpretation_cache_key(query, context)
        if self.config['enable_caching'] and cache_key in self.interpretation_cache:
            self.stats['cache_hits'] += 1
            return self.interpretation_cache[cache_key]
        
        # Classify intent
        intent = self._classify_intent(query)
        
        # Determine query type
        query_type = self._determine_query_type(query, intent)
        
        # Extract entities
        entities = self._extract_entities(query, context)
        
        # Extract relations
        relations = self._extract_relations(query, context)
        
        # Extract constraints and filters
        constraints = self._extract_constraints(query)
        
        # Extract temporal information
        temporal_info = self._extract_temporal_info(query)
        
        # Calculate confidence
        confidence = self._calculate_interpretation_confidence(
            query, intent, query_type, entities, relations
        )
        
        # Generate explanation
        explanation = self._generate_interpretation_explanation(
            intent, query_type, entities, relations, constraints
        )
        
        interpretation = QueryInterpretation(
            intent=intent,
            query_type=query_type,
            confidence=confidence,
            entities=entities,
            relations=relations,
            constraints=constraints,
            temporal_info=temporal_info,
            explanation=explanation
        )
        
        # Cache the interpretation
        if self.config['enable_caching']:
            self.interpretation_cache[cache_key] = interpretation
        
        return interpretation
    
    def _classify_intent(self, query: str) -> Intent:
        """Classify the intent of the natural language query"""
        
        query_lower = query.lower()
        
        # Use pattern matching for intent classification
        for intent, patterns in self._intent_classifier.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Advanced intent detection using transformers (if available)
        if self.use_transformers and self._nlp_models.get('model'):
            try:
                # This is a simplified approach - in practice, you'd fine-tune
                # a model specifically for intent classification
                intent_scores = self._transformer_intent_classification(query)
                if intent_scores:
                    return max(intent_scores, key=intent_scores.get)
            except Exception as e:
                logger.warning(f"Transformer intent classification failed: {e}")
        
        return Intent.UNKNOWN
    
    def _determine_query_type(self, query: str, intent: Intent) -> QueryType:
        """Determine the type of query based on content and intent"""
        
        query_lower = query.lower()
        
        # Aggregation queries
        if any(word in query_lower for word in ['count', 'how many', 'total', 'sum', 'average']):
            return QueryType.AGGREGATION
        
        # Comparison queries
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'better', 'more than']):
            return QueryType.COMPARISON
        
        # Temporal queries
        if any(word in query_lower for word in ['when', 'before', 'after', 'during', 'since', 'until']):
            return QueryType.TEMPORAL
        
        # Boolean queries
        if any(word in query_lower for word in ['is', 'are', 'was', 'were', 'does', 'do', 'can', 'will']):
            return QueryType.BOOLEAN
        
        # Relationship queries
        if any(word in query_lower for word in ['related', 'connected', 'relationship', 'links']):
            return QueryType.RELATIONSHIP_QUERY
        
        # Entity search (default for find intent)
        if intent == Intent.FIND:
            return QueryType.ENTITY_SEARCH
        
        # Complex queries (multiple clauses, sub-queries)
        if any(word in query_lower for word in ['and', 'or', 'but', 'however', 'also', 'that']):
            return QueryType.COMPLEX
        
        return QueryType.UNKNOWN
    
    def _extract_entities(self, query: str, context: ConversationContext) -> List[EntityMention]:
        """Extract entity mentions from the query"""
        
        entities = []
        
        # Use spaCy for named entity recognition if available
        if self._nlp_models.get('spacy'):
            try:
                doc = self._nlp_models['spacy'](query)
                for ent in doc.ents:
                    entity = EntityMention(
                        text=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        entity_type=ent.label_,
                        confidence=0.8  # SpaCy confidence would be higher
                    )
                    entities.append(entity)
            except Exception as e:
                logger.warning(f"SpaCy entity extraction failed: {e}")
        
        # Rule-based entity extraction as fallback
        entities.extend(self._rule_based_entity_extraction(query))
        
        # Context-based entity resolution
        entities = self._resolve_entities_with_context(entities, context)
        
        return entities
    
    def _rule_based_entity_extraction(self, query: str) -> List[EntityMention]:
        """Rule-based entity extraction as fallback"""
        
        entities = []
        
        # Common entity patterns
        entity_patterns = {
            'Organization': r'\b[A-Z][a-z]+(?:\s+(?:Inc|Corp|LLC|Ltd|Company|University|College))\b',
            'Location': r'\b[A-Z][a-z]+(?:\s+(?:City|State|Country|Street|Avenue|Road))\b',
            'Number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }
        
        # Extract organizations and locations first
        for entity_type, pattern in entity_patterns.items():
            for match in re.finditer(pattern, query):
                entity = EntityMention(
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    entity_type=entity_type,
                    confidence=0.6
                )
                entities.append(entity)
        
        # Then extract person names avoiding already found entities
        used_spans = [(e.start_pos, e.end_pos) for e in entities]
        person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        
        for match in re.finditer(person_pattern, query):
            # Check if this span overlaps with existing entities
            overlap = any(
                not (match.end() <= start or match.start() >= end)
                for start, end in used_spans
            )
            
            if not overlap:
                entity = EntityMention(
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    entity_type='Person',
                    confidence=0.6
                )
                entities.append(entity)
        
        return entities
    
    def _extract_relations(self, query: str, context: ConversationContext) -> List[RelationMention]:
        """Extract relation mentions from the query"""
        
        relations = []
        query_lower = query.lower()
        
        # Relation pattern matching
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                for match in re.finditer(r'\b' + pattern + r'\b', query_lower):
                    relation = RelationMention(
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        relation_type=relation_type,
                        confidence=0.7
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints and filters from the query"""
        
        constraints = {}
        query_lower = query.lower()
        
        # Numeric constraints
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', query)
        if numbers:
            constraints['numeric_values'] = numbers
        
        # Comparison operators
        comparisons = re.findall(r'(greater than|less than|equal to|more than|fewer than|\>|\<|\=)', query_lower)
        if comparisons:
            constraints['comparisons'] = comparisons
        
        # String filters
        quoted_strings = re.findall(r'"([^"]+)"', query)
        if quoted_strings:
            constraints['exact_matches'] = quoted_strings
        
        # Property constraints
        property_patterns = re.findall(r'with\s+(\w+)', query_lower)
        if property_patterns:
            constraints['properties'] = property_patterns
        
        return constraints
    
    def _extract_temporal_info(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from the query"""
        
        temporal_info = {}
        query_lower = query.lower()
        
        # Time expressions
        time_patterns = {
            'year': r'\b(?:19|20)\d{2}\b',  # Fixed to capture full year
            'month': r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            'relative_time': r'\b(today|yesterday|tomorrow|last\s+week|next\s+week|last\s+month|next\s+month|last\s+year|next\s+year)\b',
            'time_range': r'\b(before|after|during|since|until|from|to)\b'
        }
        
        for time_type, pattern in time_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                temporal_info[time_type] = matches
        
        return temporal_info
    
    def _calculate_interpretation_confidence(self, 
                                          query: str, 
                                          intent: Intent, 
                                          query_type: QueryType,
                                          entities: List[EntityMention],
                                          relations: List[RelationMention]) -> float:
        """Calculate confidence score for the interpretation"""
        
        confidence = 0.0
        
        # Base confidence from intent classification
        if intent != Intent.UNKNOWN:
            confidence += 0.3
        
        # Confidence from query type determination
        if query_type != QueryType.UNKNOWN:
            confidence += 0.2
        
        # Confidence from entity extraction
        if entities:
            entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence += entity_confidence * 0.3
        
        # Confidence from relation extraction
        if relations:
            relation_confidence = sum(r.confidence for r in relations) / len(relations)
            confidence += relation_confidence * 0.2
        
        # Penalty for ambiguous queries
        ambiguous_words = ['this', 'that', 'it', 'they', 'something', 'anything']
        ambiguity_penalty = sum(1 for word in ambiguous_words if word in query.lower()) * 0.1
        confidence -= ambiguity_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _translate_to_formal_query(self, 
                                 interpretation: QueryInterpretation,
                                 context: ConversationContext) -> Optional[str]:
        """Translate interpretation to formal query (SPARQL/Cypher)"""
        
        if interpretation.intent == Intent.UNKNOWN or interpretation.confidence < self.config['confidence_threshold']:
            return None
        
        # Use semantic parser templates
        if interpretation.query_type == QueryType.ENTITY_SEARCH:
            return self._generate_entity_search_query(interpretation, context)
        
        elif interpretation.query_type == QueryType.RELATIONSHIP_QUERY:
            return self._generate_relationship_query(interpretation, context)
        
        elif interpretation.query_type == QueryType.AGGREGATION:
            return self._generate_aggregation_query(interpretation, context)
        
        elif interpretation.query_type == QueryType.BOOLEAN:
            return self._generate_boolean_query(interpretation, context)
        
        elif interpretation.query_type == QueryType.COMPARISON:
            return self._generate_comparison_query(interpretation, context)
        
        elif interpretation.query_type == QueryType.TEMPORAL:
            return self._generate_temporal_query(interpretation, context)
        
        else:
            # Fallback to simple pattern matching
            return self._generate_fallback_query(interpretation, context)
    
    def _generate_entity_search_query(self, 
                                    interpretation: QueryInterpretation,
                                    context: ConversationContext) -> str:
        """Generate SPARQL query for entity search"""
        
        # Determine entity type
        entity_type = "?type"
        if interpretation.entities:
            # Use the most confident entity type
            best_entity = max(interpretation.entities, key=lambda e: e.confidence)
            if best_entity.entity_type:
                entity_type = f"<{best_entity.entity_type}>"
        
        # Build filters
        filters = []
        if interpretation.constraints.get('exact_matches'):
            for match in interpretation.constraints['exact_matches']:
                filters.append(f'FILTER(CONTAINS(LCASE(?label), "{match.lower()}"))')
        
        filter_clause = " . ".join(filters) if filters else ""
        
        query = f"""
        SELECT DISTINCT ?entity ?label WHERE {{
            ?entity rdf:type {entity_type} .
            ?entity rdfs:label ?label .
            {filter_clause}
        }}
        LIMIT 10
        """
        
        return query.strip()
    
    def _generate_relationship_query(self, 
                                   interpretation: QueryInterpretation,
                                   context: ConversationContext) -> str:
        """Generate SPARQL query for relationship queries"""
        
        # Basic relationship query template
        query = """
        SELECT DISTINCT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
        """
        
        # Add entity constraints if available
        if interpretation.entities:
            for i, entity in enumerate(interpretation.entities[:2]):  # Limit to 2 entities
                if i == 0:
                    query += f' FILTER(CONTAINS(STR(?subject), "{entity.text}")) .'
                elif i == 1:
                    query += f' FILTER(CONTAINS(STR(?object), "{entity.text}")) .'
        
        query += """
        }
        LIMIT 10
        """
        
        return query.strip()
    
    def _generate_aggregation_query(self, 
                                  interpretation: QueryInterpretation,
                                  context: ConversationContext) -> str:
        """Generate SPARQL query for aggregation queries"""
        
        entity_type = "?type"
        if interpretation.entities:
            best_entity = max(interpretation.entities, key=lambda e: e.confidence)
            if best_entity.entity_type:
                entity_type = f"<{best_entity.entity_type}>"
        
        query = f"""
        SELECT (COUNT(DISTINCT ?entity) as ?count) WHERE {{
            ?entity rdf:type {entity_type} .
        }}
        """
        
        return query.strip()
    
    def _generate_boolean_query(self, 
                              interpretation: QueryInterpretation,
                              context: ConversationContext) -> str:
        """Generate SPARQL query for boolean questions"""
        
        # ASK queries for boolean questions
        if interpretation.entities:
            entity = interpretation.entities[0]
            query = f"""
            ASK WHERE {{
                ?entity rdfs:label "{entity.text}" .
                ?entity rdf:type ?type .
            }}
            """
        else:
            # Fallback boolean query
            query = """
            ASK WHERE {
                ?s ?p ?o .
            }
            """
        
        return query.strip()
    
    def _generate_comparison_query(self, 
                                 interpretation: QueryInterpretation,
                                 context: ConversationContext) -> str:
        """Generate SPARQL query for comparison queries"""
        
        # Basic comparison template
        query = """
        SELECT DISTINCT ?entity1 ?entity2 ?property ?value1 ?value2 WHERE {
            ?entity1 ?property ?value1 .
            ?entity2 ?property ?value2 .
            FILTER(?entity1 != ?entity2)
        }
        LIMIT 10
        """
        
        return query.strip()
    
    def _generate_temporal_query(self, 
                               interpretation: QueryInterpretation,
                               context: ConversationContext) -> str:
        """Generate SPARQL query for temporal queries"""
        
        # Basic temporal query template
        query = """
        SELECT DISTINCT ?entity ?event ?time WHERE {
            ?entity ?relation ?event .
            ?event ?timeProperty ?time .
        }
        ORDER BY ?time
        LIMIT 10
        """
        
        return query.strip()
    
    def _generate_fallback_query(self, 
                                interpretation: QueryInterpretation,
                                context: ConversationContext) -> str:
        """Generate fallback query when specific patterns don't match"""
        
        # Simple subject-predicate-object query
        query = """
        SELECT DISTINCT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
        }
        LIMIT 10
        """
        
        return query.strip()
    
    def _generate_response(self, 
                         interpretation: QueryInterpretation,
                         execution_result: Optional[FederatedQueryResult]) -> str:
        """Generate natural language response from query results"""
        
        if not execution_result or not execution_result.success:
            return self._generate_error_response(interpretation, execution_result)
        
        # Generate response based on intent
        if interpretation.intent == Intent.COUNT:
            return self._generate_count_response(execution_result)
        
        elif interpretation.intent == Intent.FIND:
            return self._generate_find_response(interpretation, execution_result)
        
        elif interpretation.intent == Intent.DESCRIBE:
            return self._generate_describe_response(interpretation, execution_result)
        
        elif interpretation.intent == Intent.LIST:
            return self._generate_list_response(execution_result)
        
        else:
            return self._generate_generic_response(execution_result)
    
    def _generate_count_response(self, result: FederatedQueryResult) -> str:
        """Generate response for count queries"""
        
        if result.data and isinstance(result.data, list) and len(result.data) > 0:
            try:
                count = result.data[0].get('count', len(result.data))
                return f"I found {count} matching results."
            except:
                return f"I found {result.total_rows} matching results."
        
        return "I couldn't determine the count from the query results."
    
    def _generate_find_response(self, 
                              interpretation: QueryInterpretation,
                              result: FederatedQueryResult) -> str:
        """Generate response for find queries"""
        
        if not result.data or result.total_rows == 0:
            return "I couldn't find any matching results for your query."
        
        entity_mentions = [e.text for e in interpretation.entities]
        entity_context = f" related to {', '.join(entity_mentions)}" if entity_mentions else ""
        
        if result.total_rows == 1:
            return f"I found 1 result{entity_context}."
        else:
            return f"I found {result.total_rows} results{entity_context}."
    
    def _generate_describe_response(self, 
                                  interpretation: QueryInterpretation,
                                  result: FederatedQueryResult) -> str:
        """Generate response for describe queries"""
        
        if not result.data:
            return "I couldn't find information to describe the requested entity."
        
        if interpretation.entities:
            entity_name = interpretation.entities[0].text
            return f"Here's what I found about {entity_name}. The query returned {result.total_rows} properties."
        
        return f"I found descriptive information with {result.total_rows} properties."
    
    def _generate_list_response(self, result: FederatedQueryResult) -> str:
        """Generate response for list queries"""
        
        if result.total_rows == 0:
            return "The list is empty - no matching items were found."
        
        return f"Here's the list with {result.total_rows} items."
    
    def _generate_generic_response(self, result: FederatedQueryResult) -> str:
        """Generate generic response"""
        
        if result.total_rows == 0:
            return "No results were found for your query."
        
        return f"Your query returned {result.total_rows} results."
    
    def _generate_error_response(self, 
                               interpretation: QueryInterpretation,
                               result: Optional[FederatedQueryResult]) -> str:
        """Generate response for failed queries"""
        
        if interpretation.confidence < self.config['confidence_threshold']:
            return "I'm not sure I understood your query correctly. Could you rephrase it?"
        
        if result and result.error:
            return f"I encountered an error while processing your query: {result.error}"
        
        return "I'm sorry, I couldn't process your query at this time."
    
    def _generate_suggestions(self, 
                            interpretation: QueryInterpretation,
                            result: Optional[FederatedQueryResult]) -> List[str]:
        """Generate query suggestions for the user"""
        
        suggestions = []
        
        # Suggest refinements based on confidence
        if interpretation.confidence < 0.7:
            suggestions.append("Try being more specific about what you're looking for")
            
        # Suggest alternatives based on entities found
        if interpretation.entities:
            entity_types = set(e.entity_type for e in interpretation.entities if e.entity_type)
            for entity_type in entity_types:
                suggestions.append(f"Search for more {entity_type} entities")
        
        # Suggest related queries
        if interpretation.intent == Intent.FIND:
            suggestions.append("Try counting these entities instead")
            suggestions.append("Ask for relationships between these entities")
        
        return suggestions[:self.config['max_suggestions']]
    
    def _generate_clarifications(self, interpretation: QueryInterpretation) -> List[str]:
        """Generate clarification questions"""
        
        clarifications = []
        
        # Ask for clarification on ambiguous entities
        if interpretation.entities:
            ambiguous_entities = [e for e in interpretation.entities if e.confidence < 0.6]
            for entity in ambiguous_entities:
                clarifications.append(f"Did you mean '{entity.text}' as a {entity.entity_type or 'specific entity'}?")
        
        # Ask for clarification on intent
        if interpretation.confidence < 0.5:
            clarifications.append("What specifically would you like to know?")
        
        return clarifications
    
    def _get_conversation_context(self, 
                                conversation_id: Optional[str],
                                user_id: Optional[str]) -> ConversationContext:
        """Get or create conversation context"""
        
        if not conversation_id:
            conversation_id = self._generate_conversation_id(user_id)
        
        with self.conversation_lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            
            context = self.conversations[conversation_id]
            context.last_interaction = datetime.now()
            return context
    
    def _update_conversation_context(self, 
                                   context: ConversationContext,
                                   query: str,
                                   interpretation: QueryInterpretation,
                                   result: Optional[FederatedQueryResult]):
        """Update conversation context with new interaction"""
        
        with self.conversation_lock:
            # Add to query history
            context.previous_queries.append({
                'query': query,
                'interpretation': interpretation,
                'timestamp': datetime.now(),
                'success': result.success if result else False
            })
            
            # Update entity context
            for entity in interpretation.entities:
                context.entity_context[entity.text] = {
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'last_mentioned': datetime.now()
                }
            
            # Update topic context
            if interpretation.entities:
                entity_types = {e.entity_type for e in interpretation.entities if e.entity_type}
                context.topic_context.update(entity_types)
    
    def _resolve_entities_with_context(self, 
                                     entities: List[EntityMention],
                                     context: ConversationContext) -> List[EntityMention]:
        """Resolve entity mentions using conversation context"""
        
        # Resolve pronouns and references
        resolved_entities = []
        
        for entity in entities:
            if entity.text.lower() in ['it', 'this', 'that', 'they', 'them']:
                # Try to resolve from recent context
                recent_entities = self._get_recent_entities(context)
                if recent_entities:
                    # Use the most recent entity as resolution
                    resolved_entity = recent_entities[0]
                    entity.text = resolved_entity['text']
                    entity.entity_type = resolved_entity['type']
                    entity.confidence *= 0.8  # Reduce confidence for resolved entities
            
            resolved_entities.append(entity)
        
        return resolved_entities
    
    def _get_recent_entities(self, context: ConversationContext) -> List[Dict[str, Any]]:
        """Get recently mentioned entities from context"""
        
        recent_entities = []
        cutoff_time = datetime.now() - timedelta(minutes=self.config['context_window_minutes'])
        
        for entity_text, entity_info in context.entity_context.items():
            if entity_info['last_mentioned'] > cutoff_time:
                recent_entities.append({
                    'text': entity_text,
                    'type': entity_info['type'],
                    'confidence': entity_info['confidence'],
                    'last_mentioned': entity_info['last_mentioned']
                })
        
        # Sort by recency
        recent_entities.sort(key=lambda x: x['last_mentioned'], reverse=True)
        return recent_entities
    
    def _transformer_intent_classification(self, query: str) -> Optional[Dict[Intent, float]]:
        """Use transformer model for intent classification"""
        
        if not self.use_transformers or not self._nlp_models.get('model'):
            return None
        
        try:
            # This is a simplified example - in practice, you'd use a fine-tuned model
            tokenizer = self._nlp_models['tokenizer']
            model = self._nlp_models['model']
            
            inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # This would need a classification head for actual intent classification
                # For now, return None to fall back to rule-based classification
                return None
                
        except Exception as e:
            logger.warning(f"Transformer intent classification failed: {e}")
            return None
    
    def _generate_query_id(self, query: str) -> str:
        """Generate unique query identifier"""
        return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:12]
    
    def _generate_conversation_id(self, user_id: Optional[str]) -> str:
        """Generate conversation identifier"""
        base = f"{user_id}_{datetime.now().isoformat()}" if user_id else f"anon_{datetime.now().isoformat()}"
        return hashlib.md5(base.encode()).hexdigest()[:16]
    
    def _get_interpretation_cache_key(self, query: str, context: ConversationContext) -> str:
        """Generate cache key for interpretation"""
        context_summary = f"{len(context.previous_queries)}_{len(context.entity_context)}"
        return hashlib.md5(f"{query}_{context_summary}".encode()).hexdigest()
    
    def _generate_interpretation_explanation(self,
                                           intent: Intent,
                                           query_type: QueryType,
                                           entities: List[EntityMention],
                                           relations: List[RelationMention],
                                           constraints: Dict[str, Any]) -> str:
        """Generate human-readable explanation of interpretation"""
        
        explanation_parts = []
        
        # Intent explanation
        intent_explanations = {
            Intent.FIND: "Looking for entities or information",
            Intent.COUNT: "Counting occurrences or quantities",
            Intent.LIST: "Listing items in a collection",
            Intent.DESCRIBE: "Describing properties or characteristics",
            Intent.COMPARE: "Comparing multiple entities",
            Intent.ANALYZE: "Performing analysis or computation",
            Intent.SUMMARIZE: "Summarizing information",
            Intent.FILTER: "Filtering based on criteria",
            Intent.RELATE: "Finding relationships between entities"
        }
        
        explanation_parts.append(f"Intent: {intent_explanations.get(intent, 'Unknown intent')}")
        
        # Query type explanation
        if query_type != QueryType.ENTITY_SEARCH:
            explanation_parts.append(f"Query type: {query_type.value}")
        
        # Entities explanation
        if entities:
            entity_types = list(set(e.entity_type for e in entities if e.entity_type))
            if entity_types:
                explanation_parts.append(f"Entities: {', '.join(entity_types)}")
        
        # Relations explanation
        if relations:
            relation_types = list(set(r.relation_type for r in relations if r.relation_type))
            if relation_types:
                explanation_parts.append(f"Relations: {', '.join(relation_types)}")
        
        # Constraints explanation
        if constraints:
            constraint_parts = []
            if 'numeric_values' in constraints and constraints['numeric_values']:
                constraint_parts.append("numeric filters")
            if 'exact_matches' in constraints and constraints['exact_matches']:
                constraint_parts.append("exact matches")
            if 'properties' in constraints and constraints['properties']:
                constraint_parts.append("property filters")
            
            if constraint_parts:
                explanation_parts.append(f"Constraints: {', '.join(constraint_parts)}")
        
        return "; ".join(explanation_parts)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _cache_result(self, query: str, interpretation: QueryInterpretation, formal_query: str):
        """Cache interpretation and query translation results"""
        
        # Cache interpretation
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self.interpretation_cache[cache_key] = interpretation
        
        # Cache formal query translation
        if formal_query:
            self.query_cache[cache_key] = formal_query
    
    def _update_statistics(self, result: QueryResult):
        """Update processing statistics"""
        
        self.stats['queries_processed'] += 1
        
        if result.interpretation.confidence >= self.config['confidence_threshold']:
            self.stats['successful_interpretations'] += 1
        
        # Update intent distribution
        self.stats['intent_distribution'][result.interpretation.intent.value] += 1
        
        # Update query type distribution
        self.stats['query_type_distribution'][result.interpretation.query_type.value] += 1
        
        # Update average confidence
        total_confidence = (self.stats['average_confidence'] * (self.stats['queries_processed'] - 1) + 
                          result.interpretation.confidence)
        self.stats['average_confidence'] = total_confidence / self.stats['queries_processed']
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a specific conversation"""
        
        if conversation_id not in self.conversations:
            return None
        
        context = self.conversations[conversation_id]
        return [
            {
                'query': item['query'],
                'timestamp': item['timestamp'].isoformat(),
                'intent': item['interpretation'].intent.value,
                'confidence': item['interpretation'].confidence,
                'success': item['success']
            }
            for item in context.previous_queries
        ]
    
    def get_interface_statistics(self) -> Dict[str, Any]:
        """Get comprehensive interface statistics"""
        
        return {
            'processing_stats': dict(self.stats),
            'cache_stats': {
                'interpretation_cache_size': len(self.interpretation_cache),
                'query_cache_size': len(self.query_cache),
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['queries_processed'])
            },
            'conversation_stats': {
                'active_conversations': len(self.conversations),
                'total_context_entities': sum(len(ctx.entity_context) for ctx in self.conversations.values())
            },
            'model_stats': {
                'use_transformers': self.use_transformers,
                'available_models': list(self._nlp_models.keys()),
                'spacy_available': self._nlp_models.get('spacy') is not None
            }
        }
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation context"""
        
        with self.conversation_lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                return True
            return False
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversation contexts"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.conversation_lock:
            old_conversations = [
                conv_id for conv_id, context in self.conversations.items()
                if context.last_interaction < cutoff_time
            ]
            
            for conv_id in old_conversations:
                del self.conversations[conv_id]
            
            logger.info(f"Cleaned up {len(old_conversations)} old conversations")
    
    def __repr__(self) -> str:
        stats = self.get_interface_statistics()
        return (f"NaturalLanguageInterface(queries_processed={stats['processing_stats']['queries_processed']}, "
                f"success_rate={stats['processing_stats']['successful_interpretations'] / max(1, stats['processing_stats']['queries_processed']):.2%}, "
                f"avg_confidence={stats['processing_stats']['average_confidence']:.3f})")