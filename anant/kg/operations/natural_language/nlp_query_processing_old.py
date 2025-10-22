"""
NLP Query Processing Operations

Handles query preprocessing, interpretation, intent classification, and query type determination
for the Natural Language Interface system.

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

# Import shared types from parent module
from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext, QueryResult
)

logger = logging.getLogger(__name__)


class NLPQueryProcessing:
    """
    Handles natural language query processing operations including:
    - Query preprocessing and normalization
    - Query interpretation and structure extraction
    - Intent classification
    - Query type determination
    - Confidence calculation
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 nlp_models: Dict[str, Any],
                 intent_classifier: Dict[str, Any],
                 interpretation_cache: Dict[str, Any],
                 stats: Dict[str, Any],
                 use_transformers: bool = False):
        """Initialize NLP query processing operations"""
        self.config = config
        self.nlp_models = nlp_models
        self.intent_classifier = intent_classifier
        self.interpretation_cache = interpretation_cache
        self.stats = stats
        self.use_transformers = use_transformers
        
        # Query preprocessing configuration
        self.contractions = {
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
        
        # Intent patterns for classification
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Query type indicators
        self.query_type_indicators = self._initialize_query_type_indicators()
        
        logger.info("NLPQueryProcessing operations initialized")
    
    def _initialize_intent_patterns(self) -> Dict[Intent, List[str]]:
        """Initialize patterns for intent classification"""
        return {
            Intent.FIND: [
                r'\b(find|search|look\s+for|show\s+me|get|fetch)\b',
                r'\b(what\s+is|who\s+is|where\s+is)\b',
                r'\b(list|display)\b'
            ],
            Intent.COUNT: [
                r'\b(count|how\s+many|number\s+of)\b',
                r'\b(total|sum)\b'
            ],
            Intent.DESCRIBE: [
                r'\b(describe|tell\s+me\s+about|explain|what\s+about)\b',
                r'\b(information\s+about|details\s+of)\b'
            ],
            Intent.COMPARE: [
                r'\b(compare|versus|vs|difference\s+between)\b',
                r'\b(better\s+than|worse\s+than)\b'
            ],
            Intent.LIST: [
                r'\b(list\s+all|show\s+all|all\s+the)\b',
                r'\b(enumerate|itemize)\b'
            ]
        }
    
    def _initialize_query_type_indicators(self) -> Dict[QueryType, List[str]]:
        """Initialize indicators for query type determination"""
        return {
            QueryType.AGGREGATION: [
                'count', 'how many', 'total', 'sum', 'average', 'mean', 'maximum', 'minimum'
            ],
            QueryType.COMPARISON: [
                'compare', 'versus', 'vs', 'difference', 'better', 'more than', 'less than'
            ],
            QueryType.TEMPORAL: [
                'when', 'before', 'after', 'during', 'since', 'until', 'time', 'date'
            ],
            QueryType.BOOLEAN: [
                'is', 'are', 'was', 'were', 'does', 'do', 'can', 'will', 'true', 'false'
            ],
            QueryType.RELATIONSHIP_QUERY: [
                'related', 'connected', 'relationship', 'links', 'associations', 'connections'
            ]
        }
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess natural language query for better analysis"""
        
        # Basic preprocessing
        processed = query.strip().lower()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            processed = processed.replace(contraction, expansion)
        
        # Handle common query patterns
        processed = re.sub(r'\bwho are\b', 'who is', processed)
        processed = re.sub(r'\bwhat are\b', 'what is', processed)
        
        # Remove punctuation at the end
        processed = re.sub(r'[?!.]+$', '', processed)
        
        return processed
    
    def interpret_query(self, 
                       query: str, 
                       context: ConversationContext,
                       additional_context: Optional[Dict[str, Any]] = None,
                       entity_extractor = None,
                       relation_extractor = None,
                       constraint_extractor = None,
                       temporal_extractor = None,
                       cache_manager = None) -> QueryInterpretation:
        """
        Interpret natural language query to extract intent and structure
        
        Note: This method coordinates with other operation modules for complete interpretation
        """
        
        # Check cache first
        if cache_manager and self.config.get('enable_caching', False):
            cache_key = cache_manager.get_interpretation_cache_key(query, context)
            cached_result = self.interpretation_cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] = self.stats.get('cache_hits', 0) + 1
                return cached_result
        
        # Classify intent
        intent = self.classify_intent(query)
        
        # Determine query type
        query_type = self.determine_query_type(query, intent)
        
        # Extract entities (delegated to entity recognition module)
        entities = []
        if entity_extractor:
            entities = entity_extractor.extract_entities(query, context)
        
        # Extract relations (delegated to entity recognition module)
        relations = []
        if relation_extractor:
            relations = relation_extractor.extract_relations(query, context)
        
        # Extract constraints (delegated to constraint processing)
        constraints = {}
        if constraint_extractor:
            constraints = constraint_extractor.extract_constraints(query)
        
        # Extract temporal information (delegated to temporal processing)
        temporal_info = {}
        if temporal_extractor:
            temporal_info = temporal_extractor.extract_temporal_info(query)
        
        # Calculate confidence
        confidence = self.calculate_interpretation_confidence(
            query, intent, query_type, entities, relations
        )
        
        # Generate explanation
        explanation = self.generate_interpretation_explanation(
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
        if cache_manager and self.config.get('enable_caching', False):
            cache_key = cache_manager.get_interpretation_cache_key(query, context)
            self.interpretation_cache[cache_key] = interpretation
        
        return interpretation
    
    def classify_intent(self, query: str) -> Intent:
        """Classify the intent of the natural language query"""
        
        query_lower = query.lower()
        
        # Use pattern matching for intent classification
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Intent {intent.value} matched pattern: {pattern}")
                    return intent
        
        # Use configured intent classifier if available
        for intent_name, patterns in self.intent_classifier.items():
            try:
                intent = Intent(intent_name)
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        logger.debug(f"Intent {intent.value} matched configured pattern: {pattern}")
                        return intent
            except ValueError:
                # Invalid intent name in configuration
                continue
        
        # Advanced intent detection using transformers (if available)
        if self.use_transformers and self.nlp_models.get('model'):
            try:
                intent_scores = self.transformer_intent_classification(query)
                if intent_scores:
                    best_intent = max(intent_scores, key=intent_scores.get)
                    logger.debug(f"Transformer classified intent as: {best_intent.value}")
                    return best_intent
            except Exception as e:
                logger.warning(f"Transformer intent classification failed: {e}")
        
        logger.debug("No intent pattern matched, returning UNKNOWN")
        return Intent.UNKNOWN
    
    def determine_query_type(self, query: str, intent: Intent) -> QueryType:
        """Determine the type of query based on content and intent"""
        
        query_lower = query.lower()
        
        # Check query type indicators
        for query_type, indicators in self.query_type_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                logger.debug(f"Query type {query_type.value} detected via indicators")
                return query_type
        
        # Intent-based query type determination
        if intent == Intent.FIND:
            return QueryType.ENTITY_SEARCH
        elif intent == Intent.COUNT:
            return QueryType.AGGREGATION
        elif intent == Intent.DESCRIBE:
            return QueryType.ENTITY_SEARCH
        elif intent == Intent.COMPARE:
            return QueryType.COMPARISON
        elif intent == Intent.LIST:
            return QueryType.ENTITY_SEARCH
        
        # Complex queries (multiple clauses, sub-queries)
        if any(word in query_lower for word in ['and', 'or', 'but', 'however', 'also', 'that']):
            return QueryType.COMPLEX
        
        logger.debug("No query type pattern matched, returning UNKNOWN")
        return QueryType.UNKNOWN
    
    def calculate_interpretation_confidence(self, 
                                          query: str, 
                                          intent: Intent, 
                                          query_type: QueryType,
                                          entities: List[EntityMention],
                                          relations: List[RelationMention]) -> float:
        """Calculate confidence score for query interpretation"""
        
        confidence = 0.0
        
        # Base confidence from intent classification
        if intent != Intent.UNKNOWN:
            confidence += 0.3
        
        # Confidence from query type determination
        if query_type != QueryType.UNKNOWN:
            confidence += 0.2
        
        # Confidence from entity extraction
        if entities:
            entity_confidence = sum(entity.confidence for entity in entities) / len(entities)
            confidence += 0.3 * entity_confidence
        
        # Confidence from relation extraction
        if relations:
            relation_confidence = sum(relation.confidence for relation in relations) / len(relations)
            confidence += 0.2 * relation_confidence
        
        # Penalty for very short or very long queries
        query_length = len(query.split())
        if query_length < 3:
            confidence *= 0.8
        elif query_length > 20:
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    def generate_interpretation_explanation(self,
                                          intent: Intent,
                                          query_type: QueryType, 
                                          entities: List[EntityMention],
                                          relations: List[RelationMention],
                                          constraints: Dict[str, Any]) -> str:
        """Generate human-readable explanation of query interpretation"""
        
        explanation_parts = []
        
        # Intent explanation
        explanation_parts.append(f"Intent: {intent.value}")
        
        # Query type explanation
        explanation_parts.append(f"Query type: {query_type.value}")
        
        # Entities explanation
        if entities:
            entity_texts = [entity.text for entity in entities]
            explanation_parts.append(f"Entities found: {', '.join(entity_texts)}")
        
        # Relations explanation
        if relations:
            relation_texts = [relation.text for relation in relations]
            explanation_parts.append(f"Relations found: {', '.join(relation_texts)}")
        
        # Constraints explanation
        if constraints:
            constraint_parts = []
            for key, value in constraints.items():
                if isinstance(value, list):
                    constraint_parts.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    constraint_parts.append(f"{key}: {value}")
            
            if constraint_parts:
                explanation_parts.append(f"Constraints: {'; '.join(constraint_parts)}")
        
        return "; ".join(explanation_parts)
    
    def transformer_intent_classification(self, query: str) -> Optional[Dict[Intent, float]]:
        """Advanced intent detection using transformer models"""
        
        if not self.nlp_models.get('model'):
            return None
        
        try:
            # This is a simplified implementation
            # In practice, you would use a fine-tuned model for intent classification
            model = self.nlp_models['model']
            
            # Simple scoring based on query analysis
            intent_scores = {}
            
            # Use basic heuristics for now
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['find', 'search', 'get', 'show']):
                intent_scores[Intent.FIND] = 0.8
            
            if any(word in query_lower for word in ['count', 'how many', 'number']):
                intent_scores[Intent.COUNT] = 0.9
            
            if any(word in query_lower for word in ['describe', 'tell me about', 'explain']):
                intent_scores[Intent.DESCRIBE] = 0.8
            
            if any(word in query_lower for word in ['compare', 'versus', 'difference']):
                intent_scores[Intent.COMPARE] = 0.85
            
            if any(word in query_lower for word in ['list all', 'show all']):
                intent_scores[Intent.LIST] = 0.9
            
            return intent_scores if intent_scores else None
        
        except Exception as e:
            logger.error(f"Transformer intent classification error: {e}")
            return None
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level enum"""
        
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def update_processing_statistics(self, 
                                   query: str, 
                                   intent: Intent, 
                                   query_type: QueryType,
                                   confidence: float):
        """Update processing statistics for monitoring and optimization"""
        
        # Update query processing stats
        self.stats['total_queries'] = self.stats.get('total_queries', 0) + 1
        
        # Intent distribution
        intent_stats = self.stats.setdefault('intent_distribution', {})
        intent_stats[intent.value] = intent_stats.get(intent.value, 0) + 1
        
        # Query type distribution
        type_stats = self.stats.setdefault('query_type_distribution', {})
        type_stats[query_type.value] = type_stats.get(query_type.value, 0) + 1
        
        # Confidence distribution
        confidence_level = self.get_confidence_level(confidence)
        confidence_stats = self.stats.setdefault('confidence_distribution', {})
        confidence_stats[confidence_level.value] = confidence_stats.get(confidence_level.value, 0) + 1
        
        # Query length statistics
        query_length = len(query.split())
        length_stats = self.stats.setdefault('query_length_stats', {})
        length_stats['total_length'] = length_stats.get('total_length', 0) + query_length
        length_stats['query_count'] = length_stats.get('query_count', 0) + 1
        length_stats['average_length'] = length_stats['total_length'] / length_stats['query_count']