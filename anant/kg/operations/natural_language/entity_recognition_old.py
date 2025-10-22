"""
Entity Recognition Operations

Handles entity and relation extraction, context-based entity resolution, 
and entity mention processing for the Natural Language Interface system.

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import shared types from parent module
from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext
)

logger = logging.getLogger(__name__)


class EntityRecognitionOperations:
    """
    Handles entity and relation recognition operations including:
    - Named Entity Recognition (NER) using multiple approaches
    - Rule-based entity extraction as fallback
    - Relation extraction from query text
    - Context-based entity resolution
    - Entity mention processing and normalization
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 nlp_models: Dict[str, Any],
                 entity_patterns: Dict[str, List[str]],
                 relation_patterns: Dict[str, List[str]]):
        """Initialize entity recognition operations"""
        self.config = config
        self.nlp_models = nlp_models
        self.entity_patterns = entity_patterns
        self.relation_patterns = relation_patterns
        
        # Entity type patterns for rule-based extraction
        self.entity_type_patterns = {
            'Organization': [
                r'\b[A-Z][a-z]+(?:\s+(?:Inc|Corp|LLC|Ltd|Company|University|College|Institute|Corporation|Foundation))\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Bank|Group|Systems|Technologies|Solutions)\b'
            ],
            'Location': [
                r'\b[A-Z][a-z]+(?:\s+(?:City|State|Country|Street|Avenue|Road|Boulevard|Drive|Lane))\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:District|County|Province|Region)\b'
            ],
            'Person': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            ],
            'Number': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
                r'\b\d+(?:\s*(?:million|billion|thousand|hundred))\b'
            ],
            'Date': [
                r'\b(?:19|20)\d{2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b',
                r'\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b'
            ],
            'Technology': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:API|SDK|Framework|Library|Database|Software)\b',
                r'\b(?:Python|Java|JavaScript|C\+\+|SQL|HTML|CSS|React|Angular|Node\.js)\b'
            ]
        }
        
        # Pronoun resolution mappings
        self.pronouns = {
            'it', 'this', 'that', 'they', 'them', 'these', 'those',
            'he', 'she', 'him', 'her', 'his', 'hers', 'its'
        }
        
        logger.info("EntityRecognitionOperations initialized")
    
    def extract_entities(self, query: str, context: ConversationContext) -> List[EntityMention]:
        """Extract entity mentions from the query using multiple approaches"""
        
        entities = []
        
        # Use spaCy for named entity recognition if available
        entities.extend(self._spacy_entity_extraction(query))
        
        # Rule-based entity extraction as fallback/supplement
        entities.extend(self._rule_based_entity_extraction(query))
        
        # Pattern-based entity extraction from configuration
        entities.extend(self._pattern_based_entity_extraction(query))
        
        # Remove duplicates and merge overlapping entities
        entities = self._deduplicate_entities(entities)
        
        # Context-based entity resolution
        entities = self._resolve_entities_with_context(entities, context)
        
        # Validate and normalize entities
        entities = self._normalize_entities(entities)
        
        logger.debug(f"Extracted {len(entities)} entities from query")
        return entities
    
    def _spacy_entity_extraction(self, query: str) -> List[EntityMention]:
        """Extract entities using spaCy NLP model"""
        
        entities = []
        
        if not self.nlp_models.get('spacy'):
            return entities
        
        try:
            doc = self.nlp_models['spacy'](query)
            for ent in doc.ents:
                entity = EntityMention(
                    text=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    entity_type=ent.label_,
                    confidence=0.8  # SpaCy generally has good confidence
                )
                entities.append(entity)
                logger.debug(f"SpaCy extracted entity: {ent.text} ({ent.label_})")
        
        except Exception as e:
            logger.warning(f"SpaCy entity extraction failed: {e}")
        
        return entities
    
    def _rule_based_entity_extraction(self, query: str) -> List[EntityMention]:
        """Rule-based entity extraction as fallback"""
        
        entities = []
        
        # Extract entities using predefined patterns
        for entity_type, patterns in self.entity_type_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entity = EntityMention(
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        entity_type=entity_type,
                        confidence=0.6  # Lower confidence for rule-based
                    )
                    entities.append(entity)
                    logger.debug(f"Rule-based extracted entity: {entity.text} ({entity_type})")
        
        return entities
    
    def _pattern_based_entity_extraction(self, query: str) -> List[EntityMention]:
        """Extract entities using configured patterns"""
        
        entities = []
        
        if not self.entity_patterns:
            return entities
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, query, re.IGNORECASE):
                        entity = EntityMention(
                            text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            entity_type=entity_type,
                            confidence=0.7  # Medium confidence for configured patterns
                        )
                        entities.append(entity)
                        logger.debug(f"Pattern-based extracted entity: {entity.text} ({entity_type})")
                
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}' for entity type '{entity_type}': {e}")
        
        return entities
    
    def _deduplicate_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Remove duplicate and overlapping entities, keeping the best ones"""
        
        if not entities:
            return entities
        
        # Sort entities by start position
        entities.sort(key=lambda e: e.start_pos)
        
        deduplicated = []
        
        for entity in entities:
            # Check for overlaps with existing entities
            overlapping = False
            
            for existing in deduplicated:
                # Check if entities overlap
                if not (entity.end_pos <= existing.start_pos or entity.start_pos >= existing.end_pos):
                    overlapping = True
                    
                    # Keep the entity with higher confidence or longer text
                    if entity.confidence > existing.confidence or len(entity.text) > len(existing.text):
                        # Replace existing with current entity
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    
                    break
            
            if not overlapping:
                deduplicated.append(entity)
        
        return deduplicated
    
    def extract_relations(self, query: str, context: ConversationContext) -> List[RelationMention]:
        """Extract relation mentions from the query"""
        
        relations = []
        query_lower = query.lower()
        
        # Pattern-based relation extraction
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(r'\b' + pattern + r'\b', query_lower):
                        relation = RelationMention(
                            text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            relation_type=relation_type,
                            confidence=0.7
                        )
                        relations.append(relation)
                        logger.debug(f"Extracted relation: {relation.text} ({relation_type})")
                
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}' for relation type '{relation_type}': {e}")
        
        # Common relation patterns
        common_relations = {
            'is_a': [r'is\s+a\s+', r'are\s+', r'was\s+a\s+', r'were\s+'],
            'has': [r'has\s+', r'have\s+', r'had\s+', r'contains\s+', r'includes\s+'],
            'located_in': [r'in\s+', r'at\s+', r'located\s+in\s+', r'situated\s+in\s+'],
            'works_for': [r'works\s+for\s+', r'employed\s+by\s+', r'at\s+'],
            'related_to': [r'related\s+to\s+', r'connected\s+to\s+', r'associated\s+with\s+'],
            'part_of': [r'part\s+of\s+', r'member\s+of\s+', r'belongs\s+to\s+']
        }
        
        for relation_type, patterns in common_relations.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query_lower):
                    relation = RelationMention(
                        text=match.group().strip(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        relation_type=relation_type,
                        confidence=0.8  # Higher confidence for common patterns
                    )
                    relations.append(relation)
        
        return relations
    
    def _resolve_entities_with_context(self, 
                                     entities: List[EntityMention],
                                     context: ConversationContext) -> List[EntityMention]:
        """Resolve entity mentions using conversation context"""
        
        resolved_entities = []
        
        for entity in entities:
            entity_text_lower = entity.text.lower()
            
            # Check if entity is a pronoun that needs resolution
            if entity_text_lower in self.pronouns:
                resolved_entity = self._resolve_pronoun(entity, context)
                if resolved_entity:
                    resolved_entities.append(resolved_entity)
                else:
                    # Keep original if resolution fails
                    resolved_entities.append(entity)
            else:
                # Check for partial matches in context for entity expansion
                expanded_entity = self._expand_entity_with_context(entity, context)
                resolved_entities.append(expanded_entity)
        
        return resolved_entities
    
    def _resolve_pronoun(self, pronoun_entity: EntityMention, context: ConversationContext) -> Optional[EntityMention]:
        """Resolve pronoun to actual entity using context"""
        
        recent_entities = self._get_recent_entities(context)
        
        if not recent_entities:
            return None
        
        # Simple resolution: use the most recent entity
        # In a more sophisticated system, you'd use grammatical analysis
        recent_entity = recent_entities[0]
        
        resolved_entity = EntityMention(
            text=recent_entity['text'],
            start_pos=pronoun_entity.start_pos,
            end_pos=pronoun_entity.end_pos,
            entity_type=recent_entity['type'],
            confidence=pronoun_entity.confidence * 0.7  # Reduce confidence for resolved entities
        )
        
        logger.debug(f"Resolved pronoun '{pronoun_entity.text}' to '{resolved_entity.text}'")
        return resolved_entity
    
    def _expand_entity_with_context(self, entity: EntityMention, context: ConversationContext) -> EntityMention:
        """Expand entity mention with additional context information"""
        
        # Check if entity has been mentioned before with more complete information
        for entity_text, entity_info in context.entity_context.items():
            if (entity.text.lower() in entity_text.lower() or 
                entity_text.lower() in entity.text.lower()):
                
                # Use the more complete version
                if len(entity_text) > len(entity.text):
                    entity.text = entity_text
                    entity.entity_type = entity_info.get('type', entity.entity_type)
                    # Boost confidence if entity was mentioned before
                    entity.confidence = min(entity.confidence + 0.1, 1.0)
                    
                    logger.debug(f"Expanded entity using context: {entity.text}")
                    break
        
        return entity
    
    def _get_recent_entities(self, context: ConversationContext) -> List[Dict[str, Any]]:
        """Get recently mentioned entities from context"""
        
        recent_entities = []
        cutoff_time = datetime.now() - timedelta(minutes=self.config.get('context_window_minutes', 30))
        
        for entity_text, entity_info in context.entity_context.items():
            if entity_info.get('last_mentioned', datetime.min) > cutoff_time:
                recent_entities.append({
                    'text': entity_text,
                    'type': entity_info.get('type', 'Unknown'),
                    'confidence': entity_info.get('confidence', 0.5),
                    'last_mentioned': entity_info['last_mentioned']
                })
        
        # Sort by recency
        recent_entities.sort(key=lambda x: x['last_mentioned'], reverse=True)
        return recent_entities
    
    def _normalize_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Normalize entity mentions for consistency"""
        
        normalized = []
        
        for entity in entities:
            # Normalize text
            normalized_text = entity.text.strip()
            
            # Remove leading/trailing punctuation
            normalized_text = re.sub(r'^[^\w]+|[^\w]+$', '', normalized_text)
            
            # Skip if text becomes empty
            if not normalized_text:
                continue
            
            # Normalize entity type
            normalized_type = self._normalize_entity_type(entity.entity_type)
            
            # Create normalized entity
            normalized_entity = EntityMention(
                text=normalized_text,
                start_pos=entity.start_pos,
                end_pos=entity.end_pos,
                entity_type=normalized_type,
                confidence=entity.confidence
            )
            
            normalized.append(normalized_entity)
        
        return normalized
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to standard values"""
        
        if not entity_type:
            return 'Unknown'
        
        # Mapping of common variations to standard types
        type_mapping = {
            'PERSON': 'Person',
            'PER': 'Person',
            'ORG': 'Organization',
            'ORGANIZATION': 'Organization',
            'LOC': 'Location',
            'LOCATION': 'Location',
            'GPE': 'Location',  # Geopolitical entity
            'DATE': 'Date',
            'TIME': 'Date',
            'MONEY': 'Number',
            'PERCENT': 'Number',
            'CARDINAL': 'Number',
            'ORDINAL': 'Number'
        }
        
        return type_mapping.get(entity_type.upper(), entity_type)
    
    def extract_constraints_from_entities(self, entities: List[EntityMention]) -> Dict[str, Any]:
        """Extract constraints based on identified entities"""
        
        constraints = {}
        
        # Numeric constraints
        numeric_entities = [e for e in entities if e.entity_type == 'Number']
        if numeric_entities:
            constraints['numeric_values'] = [e.text for e in numeric_entities]
        
        # Date constraints
        date_entities = [e for e in entities if e.entity_type == 'Date']
        if date_entities:
            constraints['temporal_values'] = [e.text for e in date_entities]
        
        # Entity type constraints
        entity_types = list(set(e.entity_type for e in entities if e.entity_type != 'Unknown'))
        if entity_types:
            constraints['entity_types'] = entity_types
        
        return constraints
    
    def update_entity_context(self, 
                            entities: List[EntityMention], 
                            context: ConversationContext):
        """Update conversation context with extracted entities"""
        
        current_time = datetime.now()
        
        for entity in entities:
            # Update entity context
            context.entity_context[entity.text] = {
                'type': entity.entity_type,
                'confidence': entity.confidence,
                'last_mentioned': current_time,
                'mention_count': context.entity_context.get(entity.text, {}).get('mention_count', 0) + 1
            }
            
            # Update topic context
            if entity.entity_type != 'Unknown':
                entity_types = context.topic_context.setdefault('entity_types', set())
                entity_types.add(entity.entity_type)
    
    def get_entity_statistics(self, entities: List[EntityMention]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        
        if not entities:
            return {}
        
        # Entity type distribution
        type_counts = {}
        confidence_sum = 0
        
        for entity in entities:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
            confidence_sum += entity.confidence
        
        return {
            'total_entities': len(entities),
            'unique_types': len(type_counts),
            'type_distribution': type_counts,
            'average_confidence': confidence_sum / len(entities),
            'high_confidence_entities': len([e for e in entities if e.confidence > 0.8])
        }