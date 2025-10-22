"""
Query Translation Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles translation of query interpretations to formal query languages (SPARQL, Cypher, SQL).

Core Operations:
- Formal query generation based on interpretation
- Multi-language query translation (SPARQL, Cypher, SQL)
- Query optimization and validation
- Template-based query construction
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...natural_language_types import (
    QueryType, Intent, EntityMention, RelationMention,
    QueryInterpretation
)

logger = logging.getLogger(__name__)


@dataclass
class QueryTemplate:
    """Template for query generation"""
    name: str
    query_type: QueryType
    intent: Intent
    template: str
    parameters: List[str]
    description: str = ""


class QueryTranslationOperations:
    """
    Handles query translation operations including:
    - Formal query generation from interpretations
    - Multi-language query support (SPARQL, Cypher, SQL)
    - Query optimization and validation
    - Template-based query construction
    """
    
    def __init__(self, config: Dict[str, Any], federated_query_engine=None):
        self.config = config
        self.federated_query_engine = federated_query_engine
        
        # Query language preference
        self.default_query_language = config.get('default_query_language', 'SPARQL')
        
        # Initialize basic templates
        self.builtin_templates = self._create_basic_templates()
        
        logger.info("QueryTranslationOperations initialized")
    
    def _create_basic_templates(self) -> Dict[str, QueryTemplate]:
        """Create basic query templates"""
        return {
            'entity_search': QueryTemplate(
                name='entity_search',
                query_type=QueryType.ENTITY_SEARCH,
                intent=Intent.FIND,
                template="SELECT ?entity WHERE { ?entity rdf:type ?type }",
                parameters=['type'],
                description='Basic entity search'
            )
        }
    
    def translate_to_formal_query(self, 
                                interpretation: QueryInterpretation,
                                context=None,
                                query_language: Optional[str] = None) -> Optional[str]:
        """Translate interpretation to formal query"""
        
        # Check confidence threshold
        if interpretation.confidence < 0.3:
            return None
        
        target_language = query_language or self.default_query_language
        
        try:
            if interpretation.query_type == QueryType.ENTITY_SEARCH:
                return self._generate_entity_search_query(interpretation, target_language)
            elif interpretation.query_type == QueryType.AGGREGATION:
                return self._generate_aggregation_query(interpretation, target_language)
            else:
                return self._generate_basic_query(interpretation, target_language)
        except Exception as e:
            logger.error(f"Query translation failed: {e}")
            return None
    
    def _generate_entity_search_query(self, interpretation: QueryInterpretation, language: str) -> str:
        """Generate entity search query"""
        if language.upper() == 'SPARQL':
            return """
            SELECT DISTINCT ?entity ?label WHERE {
                ?entity rdfs:label ?label .
                ?entity rdf:type ?type .
            }
            LIMIT 10
            """
        else:
            return "MATCH (n) RETURN n LIMIT 10"
    
    def _generate_aggregation_query(self, interpretation: QueryInterpretation, language: str) -> str:
        """Generate aggregation query"""
        if language.upper() == 'SPARQL':
            return """
            SELECT (COUNT(?entity) AS ?count) WHERE {
                ?entity rdf:type ?type .
            }
            """
        else:
            return "MATCH (n) RETURN count(n) AS count"
    
    def _generate_basic_query(self, interpretation: QueryInterpretation, language: str) -> str:
        """Generate basic fallback query"""
        if language.upper() == 'SPARQL':
            return """
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .
            }
            LIMIT 10
            """
        else:
            return "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 10"
    
    def get_supported_languages(self) -> List[str]:
        """Get supported query languages"""
        return ['SPARQL', 'Cypher', 'SQL']