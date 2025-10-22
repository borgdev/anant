"""
Response Generation Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles generation of natural language responses from query results.

Core Operations:
- Natural language response generation
- Result formatting and presentation
- Error message generation
- Suggestion and clarification generation
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...natural_language_types import (
    QueryType, Intent, QueryInterpretation, EntityMention, RelationMention
)

logger = logging.getLogger(__name__)


@dataclass
class ResponseTemplate:
    """Template for response generation"""
    intent: Intent
    response_type: str
    template: str
    parameters: List[str]
    fallback: str


class ResponseGenerationOperations:
    """
    Handles response generation operations including:
    - Natural language response generation from query results
    - Result formatting and presentation
    - Error message generation
    - Suggestion and clarification generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Response templates
        self.response_templates = self._initialize_response_templates()
        
        # Response strategies
        self.response_strategies = {
            Intent.FIND: self._generate_find_response,
            Intent.COUNT: self._generate_count_response,
            Intent.DESCRIBE: self._generate_describe_response,
            Intent.LIST: self._generate_list_response,
            Intent.ASK: self._generate_ask_response,
            Intent.UNKNOWN: self._generate_unknown_response
        }
        
        logger.info("ResponseGenerationOperations initialized")
    
    def _initialize_response_templates(self) -> Dict[str, ResponseTemplate]:
        """Initialize response templates"""
        return {
            'find_success': ResponseTemplate(
                intent=Intent.FIND,
                response_type='success',
                template="Found {count} results for '{query}': {results}",
                parameters=['count', 'query', 'results'],
                fallback="Found some results for your query."
            ),
            'count_success': ResponseTemplate(
                intent=Intent.COUNT,
                response_type='success',
                template="There are {count} {entity_type} in total.",
                parameters=['count', 'entity_type'],
                fallback="Here's the count you requested."
            ),
            'no_results': ResponseTemplate(
                intent=Intent.FIND,
                response_type='no_results',
                template="No results found for '{query}'. Try rephrasing your question.",
                parameters=['query'],
                fallback="No results found. Please try a different query."
            )
        }
    
    def generate_response(self, 
                         interpretation: QueryInterpretation,
                         result: Optional[Any] = None,
                         **kwargs) -> str:
        """Generate natural language response from interpretation and results"""
        
        try:
            # Get response strategy based on intent
            strategy = self.response_strategies.get(
                interpretation.intent,
                self._generate_unknown_response
            )
            
            # Generate response using strategy
            response = strategy(interpretation, result, **kwargs)
            
            return response
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_error_response(interpretation, str(e))
    
    def _generate_find_response(self, 
                               interpretation: QueryInterpretation,
                               result: Optional[Any] = None,
                               **kwargs) -> str:
        """Generate response for FIND intent"""
        
        if result and hasattr(result, 'data') and result.data:
            count = len(result.data)
            if count == 1:
                return f"Found 1 result: {self._format_result_item(result.data[0])}"
            else:
                items = [self._format_result_item(item) for item in result.data[:3]]
                response = f"Found {count} results: {', '.join(items)}"
                if count > 3:
                    response += f" and {count - 3} more."
                return response
        else:
            return f"No results found for '{interpretation.original_query}'. Try rephrasing your question."
    
    def _generate_count_response(self, 
                                interpretation: QueryInterpretation,
                                result: Optional[Any] = None,
                                **kwargs) -> str:
        """Generate response for COUNT intent"""
        
        if result and hasattr(result, 'data') and result.data:
            if isinstance(result.data[0], dict) and 'count' in result.data[0]:
                count = result.data[0]['count']
                entity_type = "items"
                if interpretation.entities:
                    entity_type = interpretation.entities[0].entity_type or "items"
                return f"There are {count} {entity_type} in total."
            else:
                count = len(result.data)
                return f"Found {count} items."
        else:
            return "Unable to determine the count. Please try a different query."
    
    def _generate_describe_response(self, 
                                   interpretation: QueryInterpretation,
                                   result: Optional[Any] = None,
                                   **kwargs) -> str:
        """Generate response for DESCRIBE intent"""
        
        if result and hasattr(result, 'data') and result.data:
            item = result.data[0]
            description = self._format_detailed_description(item)
            return f"Here's what I found: {description}"
        else:
            return f"No information found about '{interpretation.original_query}'."
    
    def _generate_list_response(self, 
                               interpretation: QueryInterpretation,
                               result: Optional[Any] = None,
                               **kwargs) -> str:
        """Generate response for LIST intent"""
        
        if result and hasattr(result, 'data') and result.data:
            items = [self._format_list_item(item) for item in result.data[:10]]
            if len(result.data) <= 10:
                return f"Here's the complete list: {', '.join(items)}."
            else:
                return f"Here are the first 10 items: {', '.join(items)}... and {len(result.data) - 10} more."
        else:
            return "No items found to list."
    
    def _generate_ask_response(self, 
                              interpretation: QueryInterpretation,
                              result: Optional[Any] = None,
                              **kwargs) -> str:
        """Generate response for ASK/Boolean intent"""
        
        if result and hasattr(result, 'data'):
            # Boolean result handling
            if isinstance(result.data, bool):
                return "Yes." if result.data else "No."
            elif isinstance(result.data, list) and len(result.data) > 0:
                return "Yes, that's correct."
            else:
                return "No, that doesn't appear to be the case."
        else:
            return "I cannot determine the answer to that question."
    
    def _generate_unknown_response(self, 
                                  interpretation: QueryInterpretation,
                                  result: Optional[Any] = None,
                                  **kwargs) -> str:
        """Generate response for unknown intent"""
        
        return ("I'm not sure how to answer that. Could you please rephrase your question? "
                "You can ask me to find, count, describe, or list information.")
    
    def _generate_error_response(self, 
                                interpretation: QueryInterpretation,
                                error_message: str) -> str:
        """Generate error response"""
        
        return (f"I encountered an error while processing your query '{interpretation.original_query}'. "
                f"Please try rephrasing your question.")
    
    def _format_result_item(self, item: Dict[str, Any]) -> str:
        """Format a single result item for display"""
        
        if isinstance(item, dict):
            if 'name' in item:
                return item['name']
            elif 'label' in item:
                return item['label']
            elif 'entity' in item:
                return item['entity']
            else:
                return str(list(item.values())[0]) if item else "Unknown"
        else:
            return str(item)
    
    def _format_list_item(self, item: Dict[str, Any]) -> str:
        """Format an item for list display"""
        return self._format_result_item(item)
    
    def _format_detailed_description(self, item: Dict[str, Any]) -> str:
        """Format detailed description of an item"""
        
        if isinstance(item, dict):
            details = []
            for key, value in item.items():
                if key not in ['entity', 'uri']:  # Skip technical fields
                    details.append(f"{key}: {value}")
            return "; ".join(details) if details else "No details available"
        else:
            return str(item)
    
    def generate_suggestions(self, 
                           interpretation: QueryInterpretation,
                           **kwargs) -> List[str]:
        """Generate suggestions for improving the query"""
        
        suggestions = []
        
        if interpretation.confidence < 0.5:
            suggestions.append("Try being more specific in your question")
            suggestions.append("Use full names or terms")
        
        if not interpretation.entities:
            suggestions.append("Try mentioning specific entities or names")
        
        if interpretation.intent == Intent.UNKNOWN:
            suggestions.append("Try starting with words like 'find', 'show', 'count', or 'list'")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def generate_clarifications(self, 
                              interpretation: QueryInterpretation,
                              **kwargs) -> List[str]:
        """Generate clarification questions"""
        
        clarifications = []
        
        if len(interpretation.entities) > 1:
            clarifications.append("Which specific entity are you most interested in?")
        
        if interpretation.query_type == QueryType.UNKNOWN:
            clarifications.append("Are you looking to find, count, or describe something?")
        
        return clarifications[:2]  # Limit to 2 clarifications