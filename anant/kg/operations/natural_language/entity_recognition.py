"""
Entity Recognition Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles entity and relation extraction, recognition, and context resolution.

Core Operations:
- Entity extraction and recognition
- Relation detection
- Context-aware entity resolution
- Interpretation enhancement
"""

import logging
from typing import Dict, List, Optional, Any
import re

from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext
)

logger = logging.getLogger(__name__)


class EntityRecognitionOperations:
    """
    Handles entity recognition operations including:
    - Entity extraction and recognition
    - Relation detection and extraction
    - Context-aware entity resolution
    - Interpretation enhancement with entities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Simple entity patterns
        self.entity_patterns = {
            'PERSON': [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            'ORGANIZATION': [r'\b[A-Z][a-z]+ (Corporation|Company|Inc|Ltd)\b'],
            'NUMBER': [r'\b\d+\b']
        }
        
        # Simple relation patterns
        self.relation_patterns = {
            'works_at': ['works at', 'works for', 'employed by'],
            'located_in': ['located in', 'based in', 'in'],
            'owns': ['owns', 'has', 'possesses']
        }
        
        logger.info("EntityRecognitionOperations initialized")
    
    def enhance_interpretation(self, 
                             interpretation: QueryInterpretation,
                             context: Optional[ConversationContext] = None) -> QueryInterpretation:
        """Enhance interpretation with entity and relation extraction"""
        
        try:
            # Extract entities
            entities = self._extract_entities(interpretation.original_query)
            
            # Extract relations
            relations = self._extract_relations(interpretation.original_query)
            
            # Create enhanced interpretation
            enhanced_interpretation = QueryInterpretation(
                original_query=interpretation.original_query,
                intent=interpretation.intent,
                query_type=interpretation.query_type,
                confidence=interpretation.confidence,
                entities=entities,
                relations=relations,
                constraints=interpretation.constraints,
                explanation=interpretation.explanation
            )
            
            return enhanced_interpretation
        
        except Exception as e:
            logger.error(f"Entity recognition failed: {e}")
            return interpretation
    
    def _extract_entities(self, text: str) -> List[EntityMention]:
        """Extract entities from text using simple patterns"""
        
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = EntityMention(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8,  # Simple confidence
                        normalized_form=match.group().strip()
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_relations(self, text: str) -> List[RelationMention]:
        """Extract relations from text using simple patterns"""
        
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    start_pos = text.lower().find(pattern)
                    end_pos = start_pos + len(pattern)
                    
                    relation = RelationMention(
                        text=pattern,
                        relation_type=relation_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=0.7,  # Simple confidence
                        normalized_form=relation_type
                    )
                    relations.append(relation)
        
        return relations