"""
Unified Query Interface
=====================

Provides a unified query interface that works across all graph types:
- Hypergraph
- KnowledgeGraph
- HierarchicalKnowledgeGraph
- Metagraph
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Unified query result format"""
    graph_type: str
    graph_id: str
    nodes: List[Any]
    edges: List[Any]
    metadata: Dict[str, Any]
    execution_time: float


@dataclass  
class UnifiedQuery:
    """Unified query representation"""
    query_type: str  # 'node_search', 'path_search', 'pattern_match', 'aggregate'
    parameters: Dict[str, Any]
    filters: Dict[str, Any]
    output_format: str = 'default'


class QueryTranslator:
    """Translates unified queries to graph-specific query formats"""
    
    def __init__(self):
        """Initialize query translator"""
        self.translation_cache = {}
    
    def translate_for_hypergraph(self, query: UnifiedQuery) -> Dict[str, Any]:
        """Translate query for Hypergraph execution"""
        
        if query.query_type == 'node_search':
            return {
                'operation': 'node_search',
                'criteria': query.parameters.get('criteria', {}),
                'limit': query.parameters.get('limit', 100)
            }
        
        elif query.query_type == 'path_search':
            return {
                'operation': 'path_search',
                'source': query.parameters.get('source'),
                'target': query.parameters.get('target'),
                'max_length': query.parameters.get('max_length', 5)
            }
        
        elif query.query_type == 'pattern_match':
            return {
                'operation': 'pattern_match',
                'pattern': query.parameters.get('pattern'),
                'match_type': query.parameters.get('match_type', 'subgraph')
            }
        
        elif query.query_type == 'aggregate':
            return {
                'operation': 'aggregate',
                'function': query.parameters.get('function', 'count'),
                'group_by': query.parameters.get('group_by')
            }
        
        return {'operation': 'generic', 'parameters': query.parameters}
    
    def translate_for_knowledge_graph(self, query: UnifiedQuery) -> Dict[str, Any]:
        """Translate query for KnowledgeGraph execution"""
        
        if query.query_type == 'node_search':
            # Translate to semantic search
            return {
                'operation': 'semantic_search',
                'query_text': query.parameters.get('query_text', ''),
                'entity_types': query.parameters.get('entity_types'),
                'limit': query.parameters.get('limit', 100)
            }
        
        elif query.query_type == 'path_search':
            return {
                'operation': 'shortest_path',
                'source': query.parameters.get('source'),
                'target': query.parameters.get('target'),
                'relationship_types': query.parameters.get('relationship_types')
            }
        
        elif query.query_type == 'pattern_match':
            # Convert to SPARQL-like query
            pattern = query.parameters.get('pattern', {})
            return {
                'operation': 'sparql_query',
                'query': self._pattern_to_sparql(pattern),
                'variables': query.parameters.get('variables', [])
            }
        
        return {'operation': 'generic', 'parameters': query.parameters}
    
    def translate_for_hierarchical_kg(self, query: UnifiedQuery) -> Dict[str, Any]:
        """Translate query for HierarchicalKnowledgeGraph execution"""
        
        base_translation = self.translate_for_knowledge_graph(query)
        
        # Add hierarchical-specific parameters
        if query.query_type == 'node_search':
            base_translation.update({
                'levels': query.parameters.get('levels'),
                'cross_level_search': query.parameters.get('cross_level_search', True)
            })
        
        elif query.query_type == 'path_search':
            base_translation.update({
                'allow_level_crossing': query.parameters.get('allow_level_crossing', True),
                'prefer_intra_level': query.parameters.get('prefer_intra_level', False)
            })
        
        return base_translation
    
    def translate_for_metagraph(self, query: UnifiedQuery) -> Dict[str, Any]:
        """Translate query for Metagraph execution"""
        
        if query.query_type == 'node_search':
            return {
                'operation': 'meta_search',
                'search_meta_nodes': query.parameters.get('search_meta_nodes', True),
                'search_regular_nodes': query.parameters.get('search_regular_nodes', True),
                'criteria': query.parameters.get('criteria', {})
            }
        
        elif query.query_type == 'pattern_match':
            return {
                'operation': 'meta_pattern_match',
                'pattern': query.parameters.get('pattern'),
                'include_meta_relationships': query.parameters.get('include_meta_relationships', True)
            }
        
        return {'operation': 'generic', 'parameters': query.parameters}
    
    def _pattern_to_sparql(self, pattern: Dict[str, Any]) -> str:
        """Convert a pattern dict to SPARQL query string"""
        # Simplified SPARQL generation
        if 'subject' in pattern and 'predicate' in pattern and 'object' in pattern:
            return f"SELECT * WHERE {{ {pattern['subject']} {pattern['predicate']} {pattern['object']} }}"
        
        return "SELECT * WHERE { ?s ?p ?o }"


class UnifiedQueryInterface:
    """Unified query interface for cross-graph operations"""
    
    def __init__(self):
        """Initialize unified query interface"""
        self.translator = QueryTranslator()
        self.execution_cache = {}
        self.query_stats = defaultdict(int)
        
        logger.info("UnifiedQueryInterface initialized")
    
    def parse_query_string(self, query_string: str) -> UnifiedQuery:
        """Parse a natural language query string into a UnifiedQuery"""
        
        query_lower = query_string.lower()
        
        # Detect query type from keywords
        if any(keyword in query_lower for keyword in ['find', 'search', 'get', 'select']):
            if any(keyword in query_lower for keyword in ['path', 'route', 'connection']):
                query_type = 'path_search'
                
                # Extract source and target
                source_match = re.search(r'from\s+([^\s]+)', query_lower)
                target_match = re.search(r'to\s+([^\s]+)', query_lower)
                
                parameters = {}
                if source_match:
                    parameters['source'] = source_match.group(1)
                if target_match:
                    parameters['target'] = target_match.group(1)
                
            elif 'pattern' in query_lower or 'match' in query_lower:
                query_type = 'pattern_match'
                parameters = {'pattern': {}}  # Would need more sophisticated parsing
                
            else:
                query_type = 'node_search'
                
                # Extract search criteria
                parameters = {'query_text': query_string}
                
                # Look for entity type hints
                if any(keyword in query_lower for keyword in ['person', 'people', 'user']):
                    parameters['entity_types'] = ['Person']
                elif any(keyword in query_lower for keyword in ['organization', 'company', 'org']):
                    parameters['entity_types'] = ['Organization']
        
        elif any(keyword in query_lower for keyword in ['count', 'sum', 'average', 'aggregate']):
            query_type = 'aggregate'
            parameters = {'function': 'count'}
        
        else:
            # Default to node search
            query_type = 'node_search'
            parameters = {'query_text': query_string}
        
        return UnifiedQuery(
            query_type=query_type,
            parameters=parameters,
            filters={}
        )
    
    def execute_query(self, graph: Any, query: Union[str, UnifiedQuery], **kwargs) -> QueryResult:
        """Execute a unified query on a specific graph"""
        import time
        
        start_time = time.time()
        
        # Parse query if it's a string
        if isinstance(query, str):
            query = self.parse_query_string(query)
        
        # Determine graph type
        graph_type = self._get_graph_type(graph)
        graph_id = getattr(graph, 'name', str(id(graph)))
        
        # Translate query for specific graph type
        translated_query = self._translate_query(query, graph_type)
        
        # Execute query
        try:
            result = self._execute_on_graph(graph, translated_query, graph_type)
            
            execution_time = time.time() - start_time
            self.query_stats[f"{graph_type}_{query.query_type}"] += 1
            
            return QueryResult(
                graph_type=graph_type,
                graph_id=graph_id,
                nodes=result.get('nodes', []),
                edges=result.get('edges', []),
                metadata=result.get('metadata', {}),
                execution_time=execution_time
            )
        
        except Exception as e:
            logger.error(f"Query execution failed on {graph_type}: {e}")
            
            return QueryResult(
                graph_type=graph_type,
                graph_id=graph_id,
                nodes=[],
                edges=[],
                metadata={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    def query_multiple_graphs(self, graphs: List[Any], query: Union[str, UnifiedQuery], **kwargs) -> Dict[str, QueryResult]:
        """Execute a unified query across multiple graphs"""
        
        if isinstance(query, str):
            query = self.parse_query_string(query)
        
        results = {}
        
        for i, graph in enumerate(graphs):
            graph_id = getattr(graph, 'name', f"graph_{i}")
            
            try:
                result = self.execute_query(graph, query, **kwargs)
                results[graph_id] = result
            
            except Exception as e:
                logger.error(f"Query failed on graph {graph_id}: {e}")
                results[graph_id] = QueryResult(
                    graph_type=self._get_graph_type(graph),
                    graph_id=graph_id,
                    nodes=[],
                    edges=[],
                    metadata={'error': str(e)},
                    execution_time=0.0
                )
        
        return results
    
    def _get_graph_type(self, graph: Any) -> str:
        """Determine the type of a graph object"""
        
        class_name = graph.__class__.__name__
        
        if 'Hypergraph' in class_name:
            return 'hypergraph'
        elif 'HierarchicalKnowledgeGraph' in class_name:
            return 'hierarchical_kg'
        elif 'KnowledgeGraph' in class_name:
            return 'knowledge_graph'
        elif 'Metagraph' in class_name:
            return 'metagraph'
        else:
            return 'unknown'
    
    def _translate_query(self, query: UnifiedQuery, graph_type: str) -> Dict[str, Any]:
        """Translate query for specific graph type"""
        
        if graph_type == 'hypergraph':
            return self.translator.translate_for_hypergraph(query)
        elif graph_type == 'knowledge_graph':
            return self.translator.translate_for_knowledge_graph(query)
        elif graph_type == 'hierarchical_kg':
            return self.translator.translate_for_hierarchical_kg(query)
        elif graph_type == 'metagraph':
            return self.translator.translate_for_metagraph(query)
        else:
            return {'operation': 'generic', 'parameters': query.parameters}
    
    def _execute_on_graph(self, graph: Any, translated_query: Dict[str, Any], graph_type: str) -> Dict[str, Any]:
        """Execute translated query on specific graph type"""
        
        operation = translated_query.get('operation', 'generic')
        
        if graph_type == 'hypergraph':
            return self._execute_hypergraph_operation(graph, operation, translated_query)
        elif graph_type in ['knowledge_graph', 'hierarchical_kg']:
            return self._execute_kg_operation(graph, operation, translated_query)
        elif graph_type == 'metagraph':
            return self._execute_metagraph_operation(graph, operation, translated_query)
        else:
            return {'nodes': [], 'edges': [], 'metadata': {'error': 'Unknown graph type'}}
    
    def _execute_hypergraph_operation(self, graph: Any, operation: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation on hypergraph"""
        
        try:
            if operation == 'node_search':
                # Simple node search
                criteria = query.get('criteria', {})
                limit = query.get('limit', 100)
                
                matching_nodes = []
                node_count = 0
                
                for node in graph.nodes:
                    if node_count >= limit:
                        break
                    
                    # Simple matching - in practice would be more sophisticated
                    if not criteria or str(node).lower() in str(criteria).lower():
                        matching_nodes.append(node)
                        node_count += 1
                
                return {
                    'nodes': matching_nodes,
                    'edges': [],
                    'metadata': {'operation': operation, 'matches': len(matching_nodes)}
                }
            
            elif operation == 'path_search':
                source = query.get('source')
                target = query.get('target')
                
                if source and target and hasattr(graph, 'shortest_path'):
                    try:
                        path = graph.shortest_path(source, target)
                        return {
                            'nodes': path,
                            'edges': [],
                            'metadata': {'operation': operation, 'path_length': len(path)}
                        }
                    except:
                        pass
                
                return {'nodes': [], 'edges': [], 'metadata': {'operation': operation, 'path_found': False}}
            
            else:
                # Generic operation - return first few nodes
                nodes = list(graph.nodes)[:10]
                return {
                    'nodes': nodes,
                    'edges': [],
                    'metadata': {'operation': 'generic', 'total_nodes': len(nodes)}
                }
        
        except Exception as e:
            return {'nodes': [], 'edges': [], 'metadata': {'error': str(e)}}
    
    def _execute_kg_operation(self, graph: Any, operation: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation on knowledge graph"""
        
        try:
            if operation == 'semantic_search':
                query_text = query.get('query_text', '')
                entity_types = query.get('entity_types')
                limit = query.get('limit', 100)
                
                # Use KG's semantic search if available
                if hasattr(graph, 'semantic_search'):
                    try:
                        results = graph.semantic_search(query_text, entity_types, limit)
                        return {
                            'nodes': results.get('entities', []),
                            'edges': results.get('relationships', []),
                            'metadata': {'operation': operation, 'query': query_text}
                        }
                    except:
                        pass
                
                # Fallback: simple text matching
                matching_nodes = []
                for node in list(graph.nodes)[:limit]:
                    if query_text.lower() in str(node).lower():
                        matching_nodes.append(node)
                
                return {
                    'nodes': matching_nodes,
                    'edges': [],
                    'metadata': {'operation': operation, 'matches': len(matching_nodes)}
                }
            
            elif operation == 'shortest_path':
                source = query.get('source')
                target = query.get('target')
                
                if source and target:
                    # Try to find path
                    if hasattr(graph, 'shortest_path'):
                        try:
                            path = graph.shortest_path(source, target)
                            return {
                                'nodes': path,
                                'edges': [],
                                'metadata': {'operation': operation, 'path_length': len(path)}
                            }
                        except:
                            pass
                
                return {'nodes': [], 'edges': [], 'metadata': {'operation': operation, 'path_found': False}}
            
            else:
                # Generic operation
                nodes = list(graph.nodes)[:10]
                return {
                    'nodes': nodes,
                    'edges': [],
                    'metadata': {'operation': 'generic', 'total_nodes': len(nodes)}
                }
        
        except Exception as e:
            return {'nodes': [], 'edges': [], 'metadata': {'error': str(e)}}
    
    def _execute_metagraph_operation(self, graph: Any, operation: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation on metagraph"""
        
        try:
            if operation == 'meta_search':
                search_meta_nodes = query.get('search_meta_nodes', True)
                criteria = query.get('criteria', {})
                
                results = []
                
                if search_meta_nodes and hasattr(graph, 'meta_nodes'):
                    results.extend(list(graph.meta_nodes)[:5])
                
                if hasattr(graph, 'nodes'):
                    results.extend(list(graph.nodes)[:5])
                
                return {
                    'nodes': results,
                    'edges': [],
                    'metadata': {'operation': operation, 'results': len(results)}
                }
            
            else:
                # Generic operation
                nodes = []
                if hasattr(graph, 'nodes'):
                    nodes.extend(list(graph.nodes)[:5])
                if hasattr(graph, 'meta_nodes'):
                    nodes.extend(list(graph.meta_nodes)[:5])
                
                return {
                    'nodes': nodes,
                    'edges': [],
                    'metadata': {'operation': 'generic', 'total_nodes': len(nodes)}
                }
        
        except Exception as e:
            return {'nodes': [], 'edges': [], 'metadata': {'error': str(e)}}
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about query execution"""
        return {
            'total_queries': sum(self.query_stats.values()),
            'query_breakdown': dict(self.query_stats),
            'cache_size': len(self.execution_cache)
        }