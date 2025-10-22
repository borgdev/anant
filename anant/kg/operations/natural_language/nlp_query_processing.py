"""
NLP Query Processing Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles query preprocessing, interpretation, and intent classification.

Core Operations:
- Query preprocessing and normalization
- Intent classification
- Query type determination
- Basic interpretation generation
"""

import logging
from typing import Dict, List, Optional, Any

from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext, QueryResult
)

logger = logging.getLogger(__name__)


class NLPQueryProcessing:
    """
    Handles NLP query processing operations including:
    - Query preprocessing and normalization
    - Intent classification
    - Query type determination
    - Basic interpretation generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Intent patterns for simple classification
        self.intent_patterns = {
            Intent.FIND: ['find', 'show', 'get', 'search', 'look', 'who', 'what', 'where'],
            Intent.COUNT: ['count', 'how many', 'number of', 'total'],
            Intent.DESCRIBE: ['describe', 'tell me about', 'explain', 'what is'],
            Intent.LIST: ['list', 'show all', 'give me all'],
            Intent.ASK: ['is', 'are', 'does', 'can', 'will']
        }
        
        # Query type patterns
        self.query_type_patterns = {
            QueryType.ENTITY_SEARCH: ['find', 'search', 'who', 'what'],
            QueryType.AGGREGATION: ['count', 'how many', 'total', 'sum'],
            QueryType.BOOLEAN: ['is', 'are', 'does', 'can'],
            QueryType.RELATIONSHIP_SEARCH: ['related', 'connected', 'works with']
        }
        
        logger.info("NLPQueryProcessing initialized")
    
    def process_query(self, 
                     query: str,
                     context: Optional[ConversationContext] = None) -> QueryInterpretation:
        """Process a natural language query and return interpretation"""
        
        try:
            # Preprocess the query
            normalized_query = self._preprocess_query(query)
            
            # Classify intent
            intent = self._classify_intent(normalized_query)
            
            # Determine query type
            query_type = self._determine_query_type(normalized_query, intent)
            
            # Calculate confidence
            confidence = self._calculate_confidence(normalized_query, intent, query_type)
            
            # Create basic interpretation
            interpretation = QueryInterpretation(
                original_query=query,
                intent=intent,
                query_type=query_type,
                confidence=confidence,
                entities=[],  # Will be filled by entity recognition
                relations=[], # Will be filled by entity recognition
                constraints={},
                explanation=f"Interpreted as {intent.value} query with {confidence:.2f} confidence"
            )
            
            return interpretation
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryInterpretation(
                original_query=query,
                intent=Intent.UNKNOWN,
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                entities=[],
                relations=[],
                constraints={},
                explanation=f"Processing error: {str(e)}"
            )
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and normalize the query"""
        
        # Basic normalization
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove punctuation at the end
        if normalized.endswith('?') or normalized.endswith('.'):
            normalized = normalized[:-1]
        
        return normalized
    
    def _classify_intent(self, query: str) -> Intent:
        """Classify the intent of the query"""
        
        # Simple pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent
        
        return Intent.UNKNOWN
    
    def _determine_query_type(self, query: str, intent: Intent) -> QueryType:
        """Determine the type of query"""
        
        # Simple pattern matching
        for query_type, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return query_type
        
        # Default based on intent
        if intent == Intent.COUNT:
            return QueryType.AGGREGATION
        elif intent == Intent.FIND:
            return QueryType.ENTITY_SEARCH
        elif intent == Intent.ASK:
            return QueryType.BOOLEAN
        
        return QueryType.UNKNOWN
    
    def _calculate_confidence(self, query: str, intent: Intent, query_type: QueryType) -> float:
        """Calculate confidence score for the interpretation"""
        
        confidence = 0.0
        
        # Intent confidence
        if intent != Intent.UNKNOWN:
            confidence += 0.4
        
        # Query type confidence
        if query_type != QueryType.UNKNOWN:
            confidence += 0.3
        
        # Length and complexity bonus
        if len(query.split()) >= 3:
            confidence += 0.2
        
        # Pattern match bonus
        for intent_patterns in self.intent_patterns.values():
            for pattern in intent_patterns:
                if pattern in query:
                    confidence += 0.1
                    break
        
        return min(1.0, confidence)