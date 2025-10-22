"""
Context Management Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles conversation context, history management, and caching operations.

Core Operations:
- Conversation context management
- Query history tracking
- Interpretation caching
- Context-aware entity resolution
"""

import logging
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext
)

logger = logging.getLogger(__name__)


class ContextManagementOperations:
    """
    Handles context management operations including:
    - Conversation context tracking
    - Query history management
    - Interpretation caching
    - Context-aware entity resolution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Context storage
        self.conversations: Dict[str, ConversationContext] = {}
        self.interpretation_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache settings
        self.cache_size_limit = config.get('cache_size_limit', 1000)
        self.cache_ttl_hours = config.get('cache_ttl_hours', 24)
        
        logger.info("ContextManagementOperations initialized")
    
    def get_conversation_context(self, 
                               conversation_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> ConversationContext:
        """Get or create conversation context"""
        
        if not conversation_id:
            conversation_id = self._generate_conversation_id(user_id)
        
        if conversation_id not in self.conversations:
            # Create new conversation context
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                previous_queries=[],
                entity_context={},
                conversation_state={},
                conversation_history=[]
            )
        
        context = self.conversations[conversation_id]
        context.last_updated = datetime.now()
        
        return context
    
    def update_conversation_context(self, 
                                  conversation_id: str,
                                  query: str,
                                  interpretation: QueryInterpretation,
                                  result: Optional[Any] = None):
        """Update conversation context with new query and interpretation"""
        
        context = self.conversations.get(conversation_id)
        if not context:
            return
        
        # Add query to history
        query_entry = {
            'query': query,
            'timestamp': datetime.now(),
            'interpretation': interpretation,
            'result': result
        }
        
        context.previous_queries.append(query_entry)
        
        # Update entity context
        for entity in interpretation.entities:
            entity_key = entity.text.lower()
            context.entity_context[entity_key] = {
                'entity': entity,
                'last_mentioned': datetime.now(),
                'mention_count': context.entity_context.get(entity_key, {}).get('mention_count', 0) + 1
            }
        
        # Update conversation state
        context.conversation_state.update({
            'last_intent': interpretation.intent,
            'last_query_type': interpretation.query_type,
            'last_confidence': interpretation.confidence
        })
        
        context.last_updated = datetime.now()
        
        # Limit history size
        max_history = self.config.get('max_conversation_history', 50)
        if len(context.previous_queries) > max_history:
            context.previous_queries = context.previous_queries[-max_history:]
    
    def get_conversation_history(self, 
                               conversation_id: str,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        
        context = self.conversations.get(conversation_id)
        if not context:
            return []
        
        history = []
        for item in context.previous_queries[-limit:]:
            history.append({
                'query': item['query'],
                'timestamp': item['timestamp'].isoformat(),
                'intent': item['interpretation'].intent.value,
                'confidence': item['interpretation'].confidence
            })
        
        return history
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation context"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
    
    def cache_interpretation(self, 
                           query: str,
                           interpretation: QueryInterpretation):
        """Cache query interpretation"""
        
        cache_key = self._get_interpretation_cache_key(query)
        
        # Check cache size limit
        if len(self.interpretation_cache) >= self.cache_size_limit:
            self._evict_old_cache_entries()
        
        self.interpretation_cache[cache_key] = {
            'interpretation': interpretation,
            'timestamp': datetime.now(),
            'query': query
        }
    
    def get_cached_interpretation(self, query: str) -> Optional[QueryInterpretation]:
        """Get cached interpretation for query"""
        
        cache_key = self._get_interpretation_cache_key(query)
        
        if cache_key in self.interpretation_cache:
            cached = self.interpretation_cache[cache_key]
            
            # Check if cache entry is still valid
            age_hours = (datetime.now() - cached['timestamp']).total_seconds() / 3600
            if age_hours < self.cache_ttl_hours:
                return cached['interpretation']
            else:
                # Remove expired entry
                del self.interpretation_cache[cache_key]
        
        return None
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context management statistics"""
        
        total_conversations = len(self.conversations)
        total_queries = sum(len(ctx.previous_queries) for ctx in self.conversations.values())
        total_entities = sum(len(ctx.entity_context) for ctx in self.conversations.values())
        
        return {
            'active_conversations': total_conversations,
            'total_queries': total_queries,
            'total_context_entities': total_entities,
            'cache_size': len(self.interpretation_cache),
            'average_queries_per_conversation': total_queries / max(1, total_conversations),
            'average_entities_per_conversation': total_entities / max(1, total_conversations)
        }
    
    def _generate_conversation_id(self, user_id: Optional[str] = None) -> str:
        """Generate unique conversation ID"""
        timestamp = datetime.now().isoformat()
        base_string = f"{user_id or 'anonymous'}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    def _get_interpretation_cache_key(self, query: str) -> str:
        """Generate cache key for query interpretation"""
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _evict_old_cache_entries(self):
        """Remove old cache entries to maintain size limit"""
        
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self.interpretation_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest 10% of entries
        entries_to_remove = max(1, len(sorted_entries) // 10)
        
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.interpretation_cache[cache_key]
        
        logger.debug(f"Evicted {entries_to_remove} old cache entries")
    
    def cleanup_resources(self):
        """Clean up resources and old data"""
        
        # Remove old conversations (older than 7 days)
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time.replace(day=cutoff_time.day - 7)
        
        old_conversations = [
            conv_id for conv_id, context in self.conversations.items()
            if context.last_updated < cutoff_time
        ]
        
        for conv_id in old_conversations:
            del self.conversations[conv_id]
        
        # Clean old cache entries
        self._evict_old_cache_entries()
        
        logger.info(f"Cleaned up {len(old_conversations)} old conversations")