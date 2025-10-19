"""
LLM Query Processor for Metagraph

This module provides natural language query processing capabilities that convert
plain English questions into structured metagraph operations. It serves as the
primary interface for users to interact with the enterprise knowledge graph
using conversational AI.

Enterprise Features:
- Intent classification and entity extraction
- Query decomposition and planning
- Context-aware interpretation
- Multi-step reasoning support
- Security and governance integration

Author: anant development team
Date: October 2025
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Configure logging
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification for natural language processing"""
    SEARCH = "search"
    ANALYZE = "analyze" 
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    RECOMMEND = "recommend"
    EXPLAIN = "explain"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    NAVIGATE = "navigate"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity levels for processing optimization"""
    SIMPLE = "simple"          # Single entity/relationship lookup
    MODERATE = "moderate"      # Multi-entity, single operation
    COMPLEX = "complex"        # Multi-step reasoning required
    ADVANCED = "advanced"      # Cross-layer analysis needed


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language query"""
    intent: QueryIntent
    complexity: QueryComplexity
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    original_query: str = ""
    processing_steps: List[str] = field(default_factory=list)


@dataclass
class QueryExecutionPlan:
    """Execution plan for a parsed query"""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    required_layers: List[str] = field(default_factory=list)
    estimated_complexity: str = "moderate"
    dependencies: List[str] = field(default_factory=list)
    security_checks: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)


class NLPQueryProcessor:
    """
    Natural Language Query Processor for Metagraph
    
    Converts plain English queries into structured metagraph operations
    using various NLP backends (OpenAI, Transformers, rule-based).
    """
    
    def __init__(self,
                 backend: str = "auto",
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 enable_caching: bool = True):
        """
        Initialize the NLP Query Processor
        
        Parameters
        ----------
        backend : str
            NLP backend to use ("openai", "transformers", "rule_based", "auto")
        openai_api_key : str, optional
            OpenAI API key for GPT models
        model_name : str
            Model name for OpenAI or local transformers
        local_model : str
            Local model for sentence embeddings
        enable_caching : bool
            Whether to cache query results
        """
        self.backend = self._select_backend(backend)
        self.enable_caching = enable_caching
        self.query_cache = {} if enable_caching else None
        
        # Initialize chosen backend
        if self.backend == "openai":
            self._init_openai(openai_api_key, model_name)
        elif self.backend == "transformers":
            self._init_transformers(local_model)
        else:
            self._init_rule_based()
        
        # Query patterns and templates
        self.intent_patterns = self._load_intent_patterns()
        self.entity_extractors = self._load_entity_extractors()
        self.query_templates = self._load_query_templates()
        
        logger.info(f"Initialized NLP Query Processor with backend: {self.backend}")
    
    def _select_backend(self, backend: str) -> str:
        """Select the best available NLP backend"""
        if backend == "auto":
            if HAS_OPENAI:
                return "openai"
            elif HAS_TRANSFORMERS:
                return "transformers"
            else:
                return "rule_based"
        
        if backend == "openai" and not HAS_OPENAI:
            logger.warning("OpenAI not available, falling back to transformers")
            return "transformers" if HAS_TRANSFORMERS else "rule_based"
        
        if backend == "transformers" and not HAS_TRANSFORMERS:
            logger.warning("Transformers not available, falling back to rule-based")
            return "rule_based"
        
        return backend
    
    def _init_openai(self, api_key: Optional[str], model_name: str):
        """Initialize OpenAI backend"""
        if api_key:
            openai.api_key = api_key
        self.model_name = model_name
        logger.info(f"OpenAI backend initialized with model: {model_name}")
    
    def _init_transformers(self, model_name: str):
        """Initialize Transformers backend"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = pipeline("text-classification",
                                 model="microsoft/DialoGPT-medium",
                                 return_all_scores=True)
        logger.info(f"Transformers backend initialized with model: {model_name}")
    
    def _init_rule_based(self):
        """Initialize rule-based backend"""
        self.rule_patterns = self._compile_rule_patterns()
        logger.info("Rule-based backend initialized")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ParsedQuery:
        """
        Process a natural language query into structured format
        
        Parameters
        ----------
        query : str
            Natural language query
        context : dict, optional
            Additional context for query interpretation
            
        Returns
        -------
        ParsedQuery
            Structured query representation
        """
        # Check cache first
        if self.enable_caching and query in self.query_cache:
            cached_result = self.query_cache[query]
            logger.debug(f"Using cached result for query: {query[:50]}...")
            return cached_result
        
        # Initialize parsed query
        parsed = ParsedQuery(
            intent=QueryIntent.UNKNOWN,
            complexity=QueryComplexity.SIMPLE,
            original_query=query,
            context=context or {}
        )
        
        try:
            # Step 1: Classify intent
            parsed.intent = self._classify_intent(query)
            parsed.processing_steps.append("intent_classification")
            
            # Step 2: Extract entities and relationships
            entities, relationships = self._extract_entities_relationships(query)
            parsed.entities = entities
            parsed.relationships = relationships
            parsed.processing_steps.append("entity_extraction")
            
            # Step 3: Extract properties and filters
            parsed.properties = self._extract_properties(query)
            parsed.filters = self._extract_filters(query)
            parsed.processing_steps.append("property_extraction")
            
            # Step 4: Extract temporal constraints
            parsed.temporal_constraints = self._extract_temporal_constraints(query)
            parsed.processing_steps.append("temporal_extraction")
            
            # Step 5: Identify aggregations
            parsed.aggregations = self._extract_aggregations(query)
            parsed.processing_steps.append("aggregation_extraction")
            
            # Step 6: Determine complexity
            parsed.complexity = self._determine_complexity(parsed)
            parsed.processing_steps.append("complexity_analysis")
            
            # Step 7: Calculate confidence
            parsed.confidence = self._calculate_confidence(parsed)
            parsed.processing_steps.append("confidence_calculation")
            
            # Cache result
            if self.enable_caching:
                self.query_cache[query] = parsed
            
            logger.info(f"Successfully processed query with intent: {parsed.intent}, "
                       f"confidence: {parsed.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            parsed.confidence = 0.0
            parsed.processing_steps.append(f"error: {str(e)}")
        
        return parsed
    
    def create_execution_plan(self, parsed_query: ParsedQuery) -> QueryExecutionPlan:
        """
        Create an execution plan for a parsed query
        
        Parameters
        ----------
        parsed_query : ParsedQuery
            Structured query representation
            
        Returns
        -------
        QueryExecutionPlan
            Detailed execution plan
        """
        plan = QueryExecutionPlan()
        
        # Determine required layers based on query components
        if parsed_query.entities or parsed_query.relationships:
            plan.required_layers.append("hierarchical")
        
        if parsed_query.properties or parsed_query.filters:
            plan.required_layers.append("metadata")
        
        if "similar" in parsed_query.original_query.lower() or "related" in parsed_query.original_query.lower():
            plan.required_layers.append("semantic")
        
        if parsed_query.temporal_constraints:
            plan.required_layers.append("temporal")
        
        # Add governance checks for sensitive operations
        if parsed_query.intent in [QueryIntent.CREATE, QueryIntent.UPDATE, QueryIntent.DELETE]:
            plan.required_layers.append("governance")
            plan.security_checks.append("write_permission_check")
        
        # Create execution steps based on intent
        if parsed_query.intent == QueryIntent.SEARCH:
            plan.steps.extend(self._create_search_steps(parsed_query))
        elif parsed_query.intent == QueryIntent.ANALYZE:
            plan.steps.extend(self._create_analysis_steps(parsed_query))
        elif parsed_query.intent == QueryIntent.COMPARE:
            plan.steps.extend(self._create_comparison_steps(parsed_query))
        elif parsed_query.intent == QueryIntent.SUMMARIZE:
            plan.steps.extend(self._create_summarization_steps(parsed_query))
        elif parsed_query.intent == QueryIntent.RECOMMEND:
            plan.steps.extend(self._create_recommendation_steps(parsed_query))
        else:
            plan.steps.append({
                "action": "fallback_search",
                "parameters": {"query": parsed_query.original_query}
            })
        
        # Set complexity and add fallbacks
        plan.estimated_complexity = parsed_query.complexity.value
        plan.fallback_strategies = self._create_fallback_strategies(parsed_query)
        
        return plan
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a natural language query"""
        query_lower = query.lower().strip()
        
        # Rule-based intent classification
        intent_keywords = {
            QueryIntent.SEARCH: ["find", "search", "show", "list", "get", "what", "where", "who"],
            QueryIntent.ANALYZE: ["analyze", "analyze", "examine", "study", "investigate", "how"],
            QueryIntent.COMPARE: ["compare", "versus", "vs", "difference", "better", "similar"],
            QueryIntent.SUMMARIZE: ["summarize", "summary", "overview", "describe", "explain"],
            QueryIntent.RECOMMEND: ["recommend", "suggest", "advise", "should", "best"],
            QueryIntent.CREATE: ["create", "add", "insert", "new", "make"],
            QueryIntent.UPDATE: ["update", "modify", "change", "edit", "alter"],
            QueryIntent.DELETE: ["delete", "remove", "drop", "eliminate"],
            QueryIntent.NAVIGATE: ["navigate", "go to", "show me", "take me to"]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent or UNKNOWN
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return QueryIntent.UNKNOWN
    
    def _extract_entities_relationships(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract entities and relationships from query using NER"""
        entities = []
        relationships = []
        
        # Simple pattern-based extraction for now
        # Could be enhanced with NER models
        
        # Common entity patterns
        entity_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'"([^"]+)"',  # Quoted strings
            r'\'([^\']+)\'',  # Single quoted strings
        ]
        
        # Common relationship patterns
        relationship_patterns = [
            r'related to',
            r'connected to',
            r'part of',
            r'belongs to',
            r'contains',
            r'includes'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        for pattern in relationship_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                relationships.append(pattern)
        
        return list(set(entities)), list(set(relationships))
    
    def _extract_properties(self, query: str) -> List[str]:
        """Extract property names from query"""
        # Common property keywords
        property_keywords = [
            "name", "title", "description", "type", "status", "category",
            "value", "price", "cost", "amount", "size", "weight", "color",
            "date", "time", "created", "updated", "modified", "version"
        ]
        
        properties = []
        query_lower = query.lower()
        
        for keyword in property_keywords:
            if keyword in query_lower:
                properties.append(keyword)
        
        return properties
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions from query"""
        filters = {}
        
        # Numeric filters
        numeric_patterns = [
            (r'greater than (\d+)', lambda m: {"operator": ">", "value": int(m.group(1))}),
            (r'less than (\d+)', lambda m: {"operator": "<", "value": int(m.group(1))}),
            (r'equals? (\d+)', lambda m: {"operator": "=", "value": int(m.group(1))}),
            (r'between (\d+) and (\d+)', lambda m: {"operator": "between", "value": [int(m.group(1)), int(m.group(2))]})
        ]
        
        for pattern, extractor in numeric_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters["numeric"] = extractor(match)
                break
        
        # Text filters
        if "contains" in query.lower():
            match = re.search(r'contains ["\']([^"\']+)["\']', query, re.IGNORECASE)
            if match:
                filters["text"] = {"operator": "contains", "value": match.group(1)}
        
        return filters
    
    def _extract_temporal_constraints(self, query: str) -> Dict[str, Any]:
        """Extract temporal constraints from query"""
        constraints = {}
        
        # Time period patterns
        time_patterns = [
            (r'last (\d+) days?', lambda m: {"period": "days", "value": int(m.group(1)), "direction": "past"}),
            (r'last (\d+) weeks?', lambda m: {"period": "weeks", "value": int(m.group(1)), "direction": "past"}),
            (r'last (\d+) months?', lambda m: {"period": "months", "value": int(m.group(1)), "direction": "past"}),
            (r'past year', lambda m: {"period": "years", "value": 1, "direction": "past"}),
            (r'today', lambda m: {"period": "days", "value": 0, "direction": "current"}),
            (r'this week', lambda m: {"period": "weeks", "value": 0, "direction": "current"}),
            (r'this month', lambda m: {"period": "months", "value": 0, "direction": "current"})
        ]
        
        for pattern, extractor in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                constraints.update(extractor(match))
                break
        
        return constraints
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions from query"""
        aggregations = []
        
        agg_keywords = {
            "count": ["count", "number of", "how many"],
            "sum": ["sum", "total", "add up"],
            "average": ["average", "mean", "avg"],
            "min": ["minimum", "smallest", "lowest", "min"],
            "max": ["maximum", "largest", "highest", "max"],
            "group": ["group by", "grouped", "categorize"]
        }
        
        query_lower = query.lower()
        for agg_type, keywords in agg_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                aggregations.append(agg_type)
        
        return aggregations
    
    def _determine_complexity(self, parsed_query: ParsedQuery) -> QueryComplexity:
        """Determine query complexity based on parsed components"""
        complexity_score = 0
        
        # Add points for various complexity factors
        complexity_score += len(parsed_query.entities)
        complexity_score += len(parsed_query.relationships) * 2
        complexity_score += len(parsed_query.properties)
        complexity_score += len(parsed_query.aggregations) * 2
        complexity_score += len(parsed_query.temporal_constraints)
        complexity_score += len(parsed_query.filters)
        
        # Additional complexity for certain intents
        if parsed_query.intent in [QueryIntent.ANALYZE, QueryIntent.COMPARE]:
            complexity_score += 3
        elif parsed_query.intent in [QueryIntent.RECOMMEND]:
            complexity_score += 5
        
        # Classify based on total score
        if complexity_score <= 3:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 8:
            return QueryComplexity.MODERATE
        elif complexity_score <= 15:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ADVANCED
    
    def _calculate_confidence(self, parsed_query: ParsedQuery) -> float:
        """Calculate confidence score for parsed query"""
        confidence = 0.5  # Base confidence
        
        # Add confidence for recognized components
        if parsed_query.intent != QueryIntent.UNKNOWN:
            confidence += 0.2
        
        if parsed_query.entities:
            confidence += 0.1 * min(len(parsed_query.entities), 3)
        
        if parsed_query.properties:
            confidence += 0.05 * min(len(parsed_query.properties), 2)
        
        # Reduce confidence for high complexity
        if parsed_query.complexity == QueryComplexity.ADVANCED:
            confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _create_search_steps(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Create execution steps for search queries"""
        steps = []
        
        if parsed_query.entities:
            steps.append({
                "action": "search_entities",
                "layer": "hierarchical",
                "parameters": {"entities": parsed_query.entities}
            })
        
        if parsed_query.properties:
            steps.append({
                "action": "filter_by_properties",
                "layer": "metadata",
                "parameters": {"properties": parsed_query.properties, "filters": parsed_query.filters}
            })
        
        if parsed_query.temporal_constraints:
            steps.append({
                "action": "apply_temporal_filter",
                "layer": "temporal",
                "parameters": parsed_query.temporal_constraints
            })
        
        return steps
    
    def _create_analysis_steps(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Create execution steps for analysis queries"""
        steps = []
        
        steps.append({
            "action": "gather_entities",
            "layer": "hierarchical",
            "parameters": {"entities": parsed_query.entities}
        })
        
        if parsed_query.aggregations:
            steps.append({
                "action": "compute_aggregations",
                "layer": "metadata",
                "parameters": {"aggregations": parsed_query.aggregations}
            })
        
        steps.append({
            "action": "analyze_patterns",
            "layer": "semantic",
            "parameters": {"analysis_type": "general"}
        })
        
        return steps
    
    def _create_comparison_steps(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Create execution steps for comparison queries"""
        steps = []
        
        steps.append({
            "action": "identify_comparison_targets",
            "layer": "hierarchical",
            "parameters": {"entities": parsed_query.entities}
        })
        
        steps.append({
            "action": "compute_similarities",
            "layer": "semantic",
            "parameters": {"comparison_type": "similarity"}
        })
        
        steps.append({
            "action": "generate_comparison_report",
            "layer": "metadata",
            "parameters": {"properties": parsed_query.properties}
        })
        
        return steps
    
    def _create_summarization_steps(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Create execution steps for summarization queries"""
        steps = []
        
        steps.append({
            "action": "collect_summary_data",
            "layer": "hierarchical",
            "parameters": {"entities": parsed_query.entities}
        })
        
        steps.append({
            "action": "aggregate_metadata",
            "layer": "metadata",
            "parameters": {"summary_type": "comprehensive"}
        })
        
        steps.append({
            "action": "generate_summary",
            "layer": "semantic",
            "parameters": {"format": "natural_language"}
        })
        
        return steps
    
    def _create_recommendation_steps(self, parsed_query: ParsedQuery) -> List[Dict[str, Any]]:
        """Create execution steps for recommendation queries"""
        steps = []
        
        steps.append({
            "action": "analyze_context",
            "layer": "semantic",
            "parameters": {"entities": parsed_query.entities}
        })
        
        steps.append({
            "action": "find_similar_entities",
            "layer": "semantic",
            "parameters": {"similarity_threshold": 0.7}
        })
        
        steps.append({
            "action": "rank_recommendations",
            "layer": "temporal",
            "parameters": {"ranking_method": "relevance_score"}
        })
        
        return steps
    
    def _create_fallback_strategies(self, parsed_query: ParsedQuery) -> List[str]:
        """Create fallback strategies for query execution"""
        strategies = []
        
        if parsed_query.confidence < 0.5:
            strategies.append("fuzzy_search")
            strategies.append("keyword_matching")
        
        if parsed_query.complexity == QueryComplexity.ADVANCED:
            strategies.append("step_by_step_execution")
            strategies.append("partial_result_aggregation")
        
        strategies.append("general_search")
        return strategies
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent classification patterns"""
        # This would normally be loaded from a configuration file
        return {}
    
    def _load_entity_extractors(self) -> Dict[str, Any]:
        """Load entity extraction patterns"""
        # This would normally be loaded from a configuration file
        return {}
    
    def _load_query_templates(self) -> Dict[str, str]:
        """Load query templates for common patterns"""
        # This would normally be loaded from a configuration file
        return {}
    
    def _compile_rule_patterns(self) -> Dict[str, Any]:
        """Compile rule-based patterns for processing"""
        return {}


class LLMQueryExecutor:
    """
    Query executor that interfaces with the metagraph layers
    """
    
    def __init__(self, metagraph_instance):
        """
        Initialize the query executor
        
        Parameters
        ----------
        metagraph_instance : Metagraph
            The metagraph instance to execute queries against
        """
        self.metagraph = metagraph_instance
        self.processor = NLPQueryProcessor()
        
    def execute_natural_language_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a natural language query against the metagraph
        
        Parameters
        ----------
        query : str
            Natural language query
        context : dict, optional
            Additional context for query execution
            
        Returns
        -------
        dict
            Query results with metadata
        """
        try:
            # Parse the query
            parsed_query = self.processor.process_query(query, context)
            
            # Create execution plan
            plan = self.processor.create_execution_plan(parsed_query)
            
            # Execute the plan
            results = self._execute_plan(plan, parsed_query)
            
            # Add metadata
            results["query_metadata"] = {
                "original_query": query,
                "parsed_intent": parsed_query.intent.value,
                "confidence": parsed_query.confidence,
                "complexity": parsed_query.complexity.value,
                "execution_time": datetime.now().isoformat(),
                "layers_used": plan.required_layers
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "error": str(e),
                "query": query,
                "suggestion": "Try rephrasing your query or using simpler terms"
            }
    
    def _execute_plan(self, plan: QueryExecutionPlan, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Execute a query execution plan"""
        results = {"steps_executed": [], "data": {}}
        
        for step in plan.steps:
            try:
                step_result = self._execute_step(step, parsed_query)
                results["steps_executed"].append({
                    "action": step["action"],
                    "success": True,
                    "result_count": len(step_result) if isinstance(step_result, (list, dict)) else 1
                })
                
                # Merge results
                if isinstance(step_result, dict):
                    results["data"].update(step_result)
                elif isinstance(step_result, list):
                    if "items" not in results["data"]:
                        results["data"]["items"] = []
                    results["data"]["items"].extend(step_result)
                
            except Exception as e:
                logger.error(f"Error executing step {step['action']}: {e}")
                results["steps_executed"].append({
                    "action": step["action"],
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def _execute_step(self, step: Dict[str, Any], parsed_query: ParsedQuery) -> Any:
        """Execute a single step in the query plan"""
        action = step["action"]
        layer = step.get("layer")
        parameters = step.get("parameters", {})
        
        if layer == "hierarchical":
            return self._execute_hierarchical_step(action, parameters)
        elif layer == "metadata":
            return self._execute_metadata_step(action, parameters)
        elif layer == "semantic":
            return self._execute_semantic_step(action, parameters)
        elif layer == "temporal":
            return self._execute_temporal_step(action, parameters)
        elif layer == "governance":
            return self._execute_governance_step(action, parameters)
        else:
            return self._execute_general_step(action, parameters)
    
    def _execute_hierarchical_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute hierarchical layer operations"""
        if action == "search_entities":
            entities = parameters.get("entities", [])
            results = []
            for entity in entities:
                # Search in hierarchical store
                found_entities = self.metagraph.hierarchical_store.search_entities(entity)
                results.extend(found_entities)
            return results
        elif action == "gather_entities":
            return self.metagraph.hierarchical_store.get_all_entities()
        else:
            return {}
    
    def _execute_metadata_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute metadata layer operations"""
        if action == "filter_by_properties":
            properties = parameters.get("properties", [])
            filters = parameters.get("filters", {})
            return self.metagraph.metadata_store.query_by_properties(properties, filters)
        elif action == "compute_aggregations":
            aggregations = parameters.get("aggregations", [])
            return self.metagraph.metadata_store.compute_aggregations(aggregations)
        else:
            return {}
    
    def _execute_semantic_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute semantic layer operations"""
        if action == "analyze_patterns":
            return self.metagraph.semantic_layer.analyze_semantic_patterns()
        elif action == "compute_similarities":
            return self.metagraph.semantic_layer.compute_entity_similarities()
        elif action == "find_similar_entities":
            threshold = parameters.get("similarity_threshold", 0.7)
            return self.metagraph.semantic_layer.find_similar_entities(threshold)
        else:
            return {}
    
    def _execute_temporal_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute temporal layer operations"""
        if action == "apply_temporal_filter":
            return self.metagraph.temporal_layer.filter_by_time_range(parameters)
        elif action == "rank_recommendations":
            return self.metagraph.temporal_layer.get_recent_patterns()
        else:
            return {}
    
    def _execute_governance_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute governance layer operations"""
        if action == "check_permissions":
            return self.metagraph.policy_engine.check_access_permissions(parameters)
        else:
            return {}
    
    def _execute_general_step(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute general operations"""
        if action == "fallback_search":
            query = parameters.get("query", "")
            return {"fallback_results": f"General search for: {query}"}
        else:
            return {}


# Export main classes
__all__ = [
    "QueryIntent",
    "QueryComplexity", 
    "ParsedQuery",
    "QueryExecutionPlan",
    "NLPQueryProcessor",
    "LLMQueryExecutor"
]