"""
RAG Context Generator for Enterprise Knowledge Retrieval

This module provides Retrieval-Augmented Generation (RAG) capabilities for the
enterprise metagraph system. It generates relevant context for LLM interactions
by retrieving and summarizing entity information, relationships, and insights
from the knowledge graph.

Features:
- Smart context retrieval from multiple layers
- Entity summarization with LLM enhancement
- Relationship-aware context generation
- Query-specific context optimization
- Enterprise security and access control
- Multi-modal context aggregation

Author: anant development team
Date: October 2025
"""

import json
from datetime import datetime, timedelta
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
    from transformers import pipeline, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Configure logging
logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context for RAG generation"""
    ENTITY_SUMMARY = "entity_summary"
    RELATIONSHIP_MAP = "relationship_map"
    TEMPORAL_CONTEXT = "temporal_context"
    SEMANTIC_CONTEXT = "semantic_context"
    BUSINESS_CONTEXT = "business_context"
    TECHNICAL_CONTEXT = "technical_context"
    COMPLIANCE_CONTEXT = "compliance_context"


class ContextPriority(Enum):
    """Priority levels for context information"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ContextFragment:
    """Individual piece of context information"""
    fragment_id: str
    content: str
    context_type: ContextType
    priority: ContextPriority
    source_layer: str
    entity_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            # Rough token count estimation
            self.token_count = len(self.content.split())


@dataclass
class RAGContext:
    """Complete RAG context for LLM interaction"""
    query: str
    context_fragments: List[ContextFragment] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 4000
    context_summary: str = ""
    entity_focus: List[str] = field(default_factory=list)
    relationship_focus: List[str] = field(default_factory=list)
    temporal_focus: Optional[Dict[str, Any]] = None
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_tokens = sum(fragment.token_count for fragment in self.context_fragments)


class RAGContextGenerator:
    """
    RAG Context Generator for Enterprise Knowledge Retrieval
    
    Generates comprehensive, relevant context for LLM interactions by intelligently
    retrieving and summarizing information from all layers of the metagraph system.
    """
    
    def __init__(self,
                 metagraph_instance,
                 llm_backend: str = "auto",
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-3.5-turbo",
                 max_context_tokens: int = 4000,
                 enable_caching: bool = True):
        """
        Initialize the RAG Context Generator
        
        Parameters
        ----------
        metagraph_instance : Metagraph
            The metagraph instance to retrieve context from
        llm_backend : str
            LLM backend for context enhancement ("auto", "openai", "transformers")
        openai_api_key : str, optional
            OpenAI API key for GPT models
        llm_model : str
            LLM model for context generation
        max_context_tokens : int
            Maximum tokens for generated context
        enable_caching : bool
            Whether to cache generated contexts
        """
        self.metagraph = metagraph_instance
        self.max_context_tokens = max_context_tokens
        self.enable_caching = enable_caching
        
        # Initialize LLM backend
        self.llm_backend = self._select_llm_backend(llm_backend)
        self._init_llm_backend(openai_api_key, llm_model)
        
        # Context caches
        if enable_caching:
            self.context_cache = {}
            self.fragment_cache = {}
        
        # Context generation statistics
        self.generation_stats = {
            "total_contexts_generated": 0,
            "cache_hits": 0,
            "average_generation_time": 0.0,
            "total_tokens_generated": 0
        }
        
        logger.info(f"RAG Context Generator initialized with backend: {self.llm_backend}")
    
    def _select_llm_backend(self, backend: str) -> str:
        """Select the best available LLM backend"""
        if backend == "auto":
            if HAS_OPENAI:
                return "openai"
            elif HAS_TRANSFORMERS:
                return "transformers"
            else:
                return "fallback"
        return backend
    
    def _init_llm_backend(self, api_key: Optional[str], model_name: str):
        """Initialize LLM backend for context enhancement"""
        if self.llm_backend == "openai" and HAS_OPENAI:
            if api_key and 'openai' in globals():
                openai.api_key = api_key  # type: ignore
            self.llm_model = model_name
        elif self.llm_backend == "transformers" and HAS_TRANSFORMERS:
            try:
                if 'pipeline' in globals():
                    self.summarizer = pipeline("summarization",  # type: ignore
                                             model="facebook/bart-large-cnn",
                                             max_length=150)
                    self.llm_model = "transformers_local"
                else:
                    raise ImportError("Pipeline not available")
            except Exception as e:
                logger.warning(f"Failed to initialize transformers pipeline: {e}")
                self.llm_backend = "fallback"
        else:
            self.llm_backend = "fallback"
            logger.info("Using fallback context generation")
    
    def generate_rag_context(self, 
                           query: str,
                           entity_focus: Optional[List[str]] = None,
                           context_types: Optional[List[ContextType]] = None,
                           max_tokens: Optional[int] = None) -> RAGContext:
        """
        Generate comprehensive RAG context for a query
        
        Parameters
        ----------
        query : str
            The query or question requiring context
        entity_focus : List[str], optional
            Specific entities to focus context on
        context_types : List[ContextType], optional
            Types of context to include
        max_tokens : int, optional
            Maximum tokens for this context
            
        Returns
        -------
        RAGContext
            Complete RAG context for LLM interaction
        """
        start_time = datetime.now()
        max_tokens = max_tokens or self.max_context_tokens
        
        # Check cache
        cache_key = self._create_cache_key(query, entity_focus, context_types, max_tokens)
        if self.enable_caching and cache_key in self.context_cache:
            self.generation_stats["cache_hits"] += 1
            logger.debug(f"Using cached context for query: {query[:50]}...")
            return self.context_cache[cache_key]
        
        try:
            # Initialize RAG context
            rag_context = RAGContext(
                query=query,
                max_tokens=max_tokens,
                entity_focus=entity_focus or [],
                metadata={"generation_method": "rag_context_generator"}
            )
            
            # Determine entities of interest
            entities_of_interest = self._identify_entities_of_interest(query, entity_focus)
            rag_context.entity_focus = entities_of_interest
            
            # Determine context types to include
            if context_types is None:
                context_types = self._determine_relevant_context_types(query, entities_of_interest)
            
            # Generate context fragments from each layer
            context_fragments = []
            
            for context_type in context_types:
                fragments = self._generate_context_fragments(
                    context_type, 
                    entities_of_interest, 
                    query
                )
                context_fragments.extend(fragments)
            
            # Prioritize and optimize context fragments
            optimized_fragments = self._optimize_context_fragments(
                context_fragments, 
                max_tokens
            )
            
            rag_context.context_fragments = optimized_fragments
            rag_context.total_tokens = sum(f.token_count for f in optimized_fragments)
            
            # Generate context summary
            rag_context.context_summary = self._generate_context_summary(optimized_fragments)
            
            # Update statistics
            generation_time = (datetime.now() - start_time).total_seconds()
            self._update_generation_stats(rag_context, generation_time)
            
            # Cache result
            if self.enable_caching:
                self.context_cache[cache_key] = rag_context
            
            logger.info(f"Generated RAG context with {len(optimized_fragments)} fragments, "
                       f"{rag_context.total_tokens} tokens")
            
            return rag_context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            # Return minimal fallback context
            return RAGContext(
                query=query,
                context_fragments=[
                    ContextFragment(
                        fragment_id="fallback",
                        content=f"Query context for: {query}",
                        context_type=ContextType.BUSINESS_CONTEXT,
                        priority=ContextPriority.MEDIUM,
                        source_layer="fallback",
                        metadata={"error": str(e)}
                    )
                ],
                max_tokens=max_tokens
            )
    
    def _create_cache_key(self, query: str, entity_focus: Optional[List[str]], 
                         context_types: Optional[List[ContextType]], max_tokens: int) -> str:
        """Create cache key for context caching"""
        entity_str = ",".join(sorted(entity_focus)) if entity_focus else ""
        context_str = ",".join(sorted([ct.value for ct in context_types])) if context_types else ""
        return f"{hash(query)}:{hash(entity_str)}:{hash(context_str)}:{max_tokens}"
    
    def _identify_entities_of_interest(self, query: str, entity_focus: Optional[List[str]]) -> List[str]:
        """Identify entities that are relevant to the query"""
        entities = []
        
        # Start with explicitly provided entities
        if entity_focus:
            entities.extend(entity_focus)
        
        # Extract entities from query using simple keyword matching
        # This could be enhanced with NER
        query_lower = query.lower()
        
        # Search in hierarchical store for entities mentioned in query
        try:
            if hasattr(self.metagraph, 'hierarchical_store'):
                all_entities = self.metagraph.hierarchical_store.get_all_entities()
                for entity in all_entities:
                    entity_lower = entity.lower()
                    if (entity_lower in query_lower or 
                        any(word in entity_lower for word in query_lower.split())):
                        entities.append(entity)
        except Exception as e:
            logger.warning(f"Error searching entities in query: {e}")
        
        # Remove duplicates and limit count
        unique_entities = list(set(entities))[:10]  # Limit to 10 entities
        
        logger.debug(f"Identified {len(unique_entities)} entities of interest")
        return unique_entities
    
    def _determine_relevant_context_types(self, query: str, entities: List[str]) -> List[ContextType]:
        """Determine which types of context are relevant for the query"""
        context_types = []
        query_lower = query.lower()
        
        # Always include entity summary if we have entities
        if entities:
            context_types.append(ContextType.ENTITY_SUMMARY)
        
        # Relationship context for relationship queries
        if any(word in query_lower for word in ["related", "connected", "relationship", "link"]):
            context_types.append(ContextType.RELATIONSHIP_MAP)
        
        # Temporal context for time-based queries
        if any(word in query_lower for word in ["when", "time", "date", "recent", "history", "trend"]):
            context_types.append(ContextType.TEMPORAL_CONTEXT)
        
        # Semantic context for similarity and analysis queries
        if any(word in query_lower for word in ["similar", "like", "analyze", "compare", "pattern"]):
            context_types.append(ContextType.SEMANTIC_CONTEXT)
        
        # Business context for business-related queries
        if any(word in query_lower for word in ["business", "process", "workflow", "operation", "strategy"]):
            context_types.append(ContextType.BUSINESS_CONTEXT)
        
        # Technical context for technical queries
        if any(word in query_lower for word in ["system", "technical", "architecture", "implementation"]):
            context_types.append(ContextType.TECHNICAL_CONTEXT)
        
        # Compliance context for governance queries
        if any(word in query_lower for word in ["compliance", "policy", "governance", "audit", "security"]):
            context_types.append(ContextType.COMPLIANCE_CONTEXT)
        
        # Default context types if none detected
        if not context_types:
            context_types = [ContextType.ENTITY_SUMMARY, ContextType.BUSINESS_CONTEXT]
        
        return context_types
    
    def _generate_context_fragments(self, 
                                   context_type: ContextType, 
                                   entities: List[str], 
                                   query: str) -> List[ContextFragment]:
        """Generate context fragments for a specific context type"""
        fragments = []
        
        try:
            if context_type == ContextType.ENTITY_SUMMARY:
                fragments.extend(self._generate_entity_summary_fragments(entities))
            elif context_type == ContextType.RELATIONSHIP_MAP:
                fragments.extend(self._generate_relationship_fragments(entities))
            elif context_type == ContextType.TEMPORAL_CONTEXT:
                fragments.extend(self._generate_temporal_fragments(entities))
            elif context_type == ContextType.SEMANTIC_CONTEXT:
                fragments.extend(self._generate_semantic_fragments(entities))
            elif context_type == ContextType.BUSINESS_CONTEXT:
                fragments.extend(self._generate_business_fragments(entities, query))
            elif context_type == ContextType.TECHNICAL_CONTEXT:
                fragments.extend(self._generate_technical_fragments(entities))
            elif context_type == ContextType.COMPLIANCE_CONTEXT:
                fragments.extend(self._generate_compliance_fragments(entities))
                
        except Exception as e:
            logger.error(f"Error generating {context_type.value} fragments: {e}")
            # Add error fragment
            fragments.append(
                ContextFragment(
                    fragment_id=f"error_{context_type.value}",
                    content=f"Context generation error for {context_type.value}",
                    context_type=context_type,
                    priority=ContextPriority.LOW,
                    source_layer="error",
                    metadata={"error": str(e)}
                )
            )
        
        return fragments
    
    def _generate_entity_summary_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate entity summary fragments"""
        fragments = []
        
        for entity in entities:
            try:
                # Get entity information from hierarchical store
                entity_info = self._get_entity_summary(entity)
                
                if entity_info:
                    content = self._format_entity_summary(entity, entity_info)
                    
                    fragment = ContextFragment(
                        fragment_id=f"entity_summary_{entity}",
                        content=content,
                        context_type=ContextType.ENTITY_SUMMARY,
                        priority=ContextPriority.HIGH,
                        source_layer="hierarchical",
                        entity_ids=[entity],
                        confidence=0.9
                    )
                    
                    fragments.append(fragment)
                    
            except Exception as e:
                logger.warning(f"Error generating entity summary for {entity}: {e}")
        
        return fragments
    
    def _generate_relationship_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate relationship context fragments"""
        fragments = []
        
        try:
            # Get relationships from semantic layer
            if hasattr(self.metagraph, 'semantic_layer'):
                relationships = self._get_entity_relationships(entities)
                
                if relationships:
                    content = self._format_relationship_summary(relationships)
                    
                    fragment = ContextFragment(
                        fragment_id="relationship_map",
                        content=content,
                        context_type=ContextType.RELATIONSHIP_MAP,
                        priority=ContextPriority.HIGH,
                        source_layer="semantic",
                        entity_ids=entities,
                        confidence=0.8
                    )
                    
                    fragments.append(fragment)
                    
        except Exception as e:
            logger.warning(f"Error generating relationship fragments: {e}")
        
        return fragments
    
    def _generate_temporal_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate temporal context fragments"""
        fragments = []
        
        try:
            # Get temporal information from temporal layer
            if hasattr(self.metagraph, 'temporal_layer'):
                temporal_info = self._get_temporal_context(entities)
                
                if temporal_info:
                    content = self._format_temporal_summary(temporal_info)
                    
                    fragment = ContextFragment(
                        fragment_id="temporal_context",
                        content=content,
                        context_type=ContextType.TEMPORAL_CONTEXT,
                        priority=ContextPriority.MEDIUM,
                        source_layer="temporal",
                        entity_ids=entities,
                        confidence=0.7
                    )
                    
                    fragments.append(fragment)
                    
        except Exception as e:
            logger.warning(f"Error generating temporal fragments: {e}")
        
        return fragments
    
    def _generate_semantic_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate semantic context fragments"""
        fragments = []
        
        try:
            # Get semantic information
            semantic_info = self._get_semantic_context(entities)
            
            if semantic_info:
                content = self._format_semantic_summary(semantic_info)
                
                fragment = ContextFragment(
                    fragment_id="semantic_context",
                    content=content,
                    context_type=ContextType.SEMANTIC_CONTEXT,
                    priority=ContextPriority.MEDIUM,
                    source_layer="semantic",
                    entity_ids=entities,
                    confidence=0.8
                )
                
                fragments.append(fragment)
                
        except Exception as e:
            logger.warning(f"Error generating semantic fragments: {e}")
        
        return fragments
    
    def _generate_business_fragments(self, entities: List[str], query: str) -> List[ContextFragment]:
        """Generate business context fragments"""
        fragments = []
        
        try:
            business_info = self._get_business_context(entities, query)
            
            if business_info:
                content = self._format_business_summary(business_info)
                
                fragment = ContextFragment(
                    fragment_id="business_context",
                    content=content,
                    context_type=ContextType.BUSINESS_CONTEXT,
                    priority=ContextPriority.HIGH,
                    source_layer="metadata",
                    entity_ids=entities,
                    confidence=0.7
                )
                
                fragments.append(fragment)
                
        except Exception as e:
            logger.warning(f"Error generating business fragments: {e}")
        
        return fragments
    
    def _generate_technical_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate technical context fragments"""
        fragments = []
        
        try:
            technical_info = self._get_technical_context(entities)
            
            if technical_info:
                content = self._format_technical_summary(technical_info)
                
                fragment = ContextFragment(
                    fragment_id="technical_context",
                    content=content,
                    context_type=ContextType.TECHNICAL_CONTEXT,
                    priority=ContextPriority.MEDIUM,
                    source_layer="metadata",
                    entity_ids=entities,
                    confidence=0.6
                )
                
                fragments.append(fragment)
                
        except Exception as e:
            logger.warning(f"Error generating technical fragments: {e}")
        
        return fragments
    
    def _generate_compliance_fragments(self, entities: List[str]) -> List[ContextFragment]:
        """Generate compliance context fragments"""
        fragments = []
        
        try:
            # Get compliance information from governance layer
            if hasattr(self.metagraph, 'policy_engine'):
                compliance_info = self._get_compliance_context(entities)
                
                if compliance_info:
                    content = self._format_compliance_summary(compliance_info)
                    
                    fragment = ContextFragment(
                        fragment_id="compliance_context",
                        content=content,
                        context_type=ContextType.COMPLIANCE_CONTEXT,
                        priority=ContextPriority.HIGH,
                        source_layer="governance",
                        entity_ids=entities,
                        confidence=0.9
                    )
                    
                    fragments.append(fragment)
                    
        except Exception as e:
            logger.warning(f"Error generating compliance fragments: {e}")
        
        return fragments
    
    def _optimize_context_fragments(self, 
                                   fragments: List[ContextFragment], 
                                   max_tokens: int) -> List[ContextFragment]:
        """Optimize context fragments for token limit and relevance"""
        # Sort by priority and confidence
        def fragment_score(fragment):
            priority_scores = {
                ContextPriority.CRITICAL: 5,
                ContextPriority.HIGH: 4,
                ContextPriority.MEDIUM: 3,
                ContextPriority.LOW: 2,
                ContextPriority.BACKGROUND: 1
            }
            return priority_scores.get(fragment.priority, 1) * fragment.confidence
        
        sorted_fragments = sorted(fragments, key=fragment_score, reverse=True)
        
        # Select fragments within token limit
        selected_fragments = []
        total_tokens = 0
        
        for fragment in sorted_fragments:
            if total_tokens + fragment.token_count <= max_tokens:
                selected_fragments.append(fragment)
                total_tokens += fragment.token_count
            else:
                # Try to truncate fragment if it's high priority
                if fragment.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 50:  # Minimum useful fragment size
                        truncated_content = self._truncate_content(fragment.content, remaining_tokens)
                        truncated_fragment = ContextFragment(
                            fragment_id=f"{fragment.fragment_id}_truncated",
                            content=truncated_content,
                            context_type=fragment.context_type,
                            priority=fragment.priority,
                            source_layer=fragment.source_layer,
                            entity_ids=fragment.entity_ids,
                            confidence=fragment.confidence * 0.8,  # Reduce confidence for truncation
                            metadata={**fragment.metadata, "truncated": True}
                        )
                        selected_fragments.append(truncated_fragment)
                        break
        
        logger.debug(f"Optimized {len(fragments)} fragments to {len(selected_fragments)} "
                    f"within {max_tokens} token limit")
        
        return selected_fragments
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit"""
        words = content.split()
        if len(words) <= max_tokens:
            return content
        
        truncated_words = words[:max_tokens - 3]  # Reserve space for "..."
        return " ".join(truncated_words) + "..."
    
    def _generate_context_summary(self, fragments: List[ContextFragment]) -> str:
        """Generate a summary of the context"""
        if not fragments:
            return "No context available."
        
        # Count fragments by type
        type_counts = {}
        for fragment in fragments:
            type_counts[fragment.context_type] = type_counts.get(fragment.context_type, 0) + 1
        
        # Generate summary
        summary_parts = []
        for context_type, count in type_counts.items():
            summary_parts.append(f"{count} {context_type.value.replace('_', ' ')} fragment(s)")
        
        total_tokens = sum(f.token_count for f in fragments)
        entity_count = len(set().union(*[f.entity_ids for f in fragments]))
        
        summary = f"Context includes {', '.join(summary_parts)} covering {entity_count} entities in {total_tokens} tokens."
        
        return summary
    
    def _get_entity_summary(self, entity: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive entity summary from metagraph"""
        try:
            # Get information from hierarchical store
            entity_data = {}
            
            if hasattr(self.metagraph, 'hierarchical_store'):
                hierarchy_info = self.metagraph.hierarchical_store.get_entity_info(entity)
                if hierarchy_info:
                    entity_data["hierarchy"] = hierarchy_info
            
            # Get metadata
            if hasattr(self.metagraph, 'metadata_store'):
                metadata = self.metagraph.metadata_store.get_entity_metadata(entity)
                if metadata:
                    entity_data["metadata"] = metadata
            
            return entity_data if entity_data else None
            
        except Exception as e:
            logger.warning(f"Error getting entity summary for {entity}: {e}")
            return None
    
    def _get_entity_relationships(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Get relationships involving the entities"""
        try:
            relationships = {}
            
            if hasattr(self.metagraph, 'semantic_layer'):
                for entity in entities:
                    entity_rels = self.metagraph.semantic_layer.get_entity_relationships(entity)
                    if entity_rels:
                        relationships[entity] = entity_rels
            
            return relationships if relationships else None
            
        except Exception as e:
            logger.warning(f"Error getting entity relationships: {e}")
            return None
    
    def _get_temporal_context(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Get temporal context for entities"""
        try:
            temporal_info = {}
            
            if hasattr(self.metagraph, 'temporal_layer'):
                for entity in entities:
                    entity_temporal = self.metagraph.temporal_layer.get_entity_history(entity)
                    if entity_temporal:
                        temporal_info[entity] = entity_temporal
            
            return temporal_info if temporal_info else None
            
        except Exception as e:
            logger.warning(f"Error getting temporal context: {e}")
            return None
    
    def _get_semantic_context(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Get semantic context for entities"""
        try:
            semantic_info = {}
            
            if hasattr(self.metagraph, 'semantic_layer'):
                for entity in entities:
                    similar_entities = self.metagraph.semantic_layer.find_similar_entities(
                        entity, similarity_threshold=0.7, max_results=5
                    )
                    if similar_entities:
                        semantic_info[entity] = similar_entities
            
            return semantic_info if semantic_info else None
            
        except Exception as e:
            logger.warning(f"Error getting semantic context: {e}")
            return None
    
    def _get_business_context(self, entities: List[str], query: str) -> Optional[Dict[str, Any]]:
        """Get business context for entities"""
        try:
            business_info = {}
            
            # Extract business terms from query and entities
            business_terms = self._extract_business_terms(query, entities)
            if business_terms:
                business_info["terms"] = business_terms
            
            # Get business properties from metadata store
            if hasattr(self.metagraph, 'metadata_store'):
                for entity in entities:
                    business_props = self.metagraph.metadata_store.get_business_properties(entity)
                    if business_props:
                        business_info[entity] = business_props
            
            return business_info if business_info else None
            
        except Exception as e:
            logger.warning(f"Error getting business context: {e}")
            return None
    
    def _get_technical_context(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Get technical context for entities"""
        try:
            technical_info = {}
            
            if hasattr(self.metagraph, 'metadata_store'):
                for entity in entities:
                    tech_props = self.metagraph.metadata_store.get_technical_properties(entity)
                    if tech_props:
                        technical_info[entity] = tech_props
            
            return technical_info if technical_info else None
            
        except Exception as e:
            logger.warning(f"Error getting technical context: {e}")
            return None
    
    def _get_compliance_context(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Get compliance context for entities"""
        try:
            compliance_info = {}
            
            if hasattr(self.metagraph, 'policy_engine'):
                for entity in entities:
                    policies = self.metagraph.policy_engine.get_entity_policies(entity)
                    if policies:
                        compliance_info[entity] = policies
            
            return compliance_info if compliance_info else None
            
        except Exception as e:
            logger.warning(f"Error getting compliance context: {e}")
            return None
    
    def _extract_business_terms(self, query: str, entities: List[str]) -> List[str]:
        """Extract business terms from query and entities"""
        # Simple extraction - could be enhanced with NLP
        business_keywords = [
            "process", "workflow", "procedure", "policy", "standard",
            "customer", "product", "service", "revenue", "cost",
            "strategy", "objective", "goal", "kpi", "metric"
        ]
        
        terms = []
        query_words = query.lower().split()
        
        for keyword in business_keywords:
            if keyword in query_words:
                terms.append(keyword)
        
        return terms
    
    # Formatting methods for different context types
    def _format_entity_summary(self, entity: str, entity_info: Dict[str, Any]) -> str:
        """Format entity summary information"""
        summary_parts = [f"Entity: {entity}"]
        
        if "hierarchy" in entity_info:
            hierarchy = entity_info["hierarchy"]
            if "level" in hierarchy:
                summary_parts.append(f"Level: {hierarchy['level']}")
            if "parent" in hierarchy:
                summary_parts.append(f"Parent: {hierarchy['parent']}")
        
        if "metadata" in entity_info:
            metadata = entity_info["metadata"]
            if "type" in metadata:
                summary_parts.append(f"Type: {metadata['type']}")
            if "description" in metadata:
                summary_parts.append(f"Description: {metadata['description']}")
        
        return ". ".join(summary_parts) + "."
    
    def _format_relationship_summary(self, relationships: Dict[str, Any]) -> str:
        """Format relationship information"""
        summary_parts = ["Relationships:"]
        
        for entity, rels in relationships.items():
            if isinstance(rels, list) and rels:
                rel_count = len(rels)
                rel_types = list(set(rel.get("type", "unknown") for rel in rels))
                summary_parts.append(f"{entity} has {rel_count} relationships of types: {', '.join(rel_types)}")
        
        return " ".join(summary_parts)
    
    def _format_temporal_summary(self, temporal_info: Dict[str, Any]) -> str:
        """Format temporal information"""
        summary_parts = ["Temporal context:"]
        
        for entity, temporal_data in temporal_info.items():
            if "created_at" in temporal_data:
                summary_parts.append(f"{entity} created at {temporal_data['created_at']}")
            if "last_updated" in temporal_data:
                summary_parts.append(f"last updated {temporal_data['last_updated']}")
        
        return " ".join(summary_parts)
    
    def _format_semantic_summary(self, semantic_info: Dict[str, Any]) -> str:
        """Format semantic information"""
        summary_parts = ["Semantic context:"]
        
        for entity, similar_entities in semantic_info.items():
            if similar_entities:
                similar_names = [sim["entity_id"] for sim in similar_entities[:3]]
                summary_parts.append(f"{entity} is similar to {', '.join(similar_names)}")
        
        return " ".join(summary_parts)
    
    def _format_business_summary(self, business_info: Dict[str, Any]) -> str:
        """Format business information"""
        summary_parts = ["Business context:"]
        
        if "terms" in business_info:
            terms = business_info["terms"]
            summary_parts.append(f"Business terms: {', '.join(terms)}")
        
        for entity, props in business_info.items():
            if entity != "terms" and isinstance(props, dict):
                if "business_unit" in props:
                    summary_parts.append(f"{entity} belongs to {props['business_unit']}")
        
        return " ".join(summary_parts)
    
    def _format_technical_summary(self, technical_info: Dict[str, Any]) -> str:
        """Format technical information"""
        summary_parts = ["Technical context:"]
        
        for entity, props in technical_info.items():
            if "system" in props:
                summary_parts.append(f"{entity} is part of {props['system']} system")
            if "technology" in props:
                summary_parts.append(f"uses {props['technology']} technology")
        
        return " ".join(summary_parts)
    
    def _format_compliance_summary(self, compliance_info: Dict[str, Any]) -> str:
        """Format compliance information"""
        summary_parts = ["Compliance context:"]
        
        for entity, policies in compliance_info.items():
            if policies:
                policy_names = [policy.get("name", "unknown") for policy in policies[:3]]
                summary_parts.append(f"{entity} is subject to policies: {', '.join(policy_names)}")
        
        return " ".join(summary_parts)
    
    def _update_generation_stats(self, rag_context: RAGContext, generation_time: float):
        """Update generation statistics"""
        self.generation_stats["total_contexts_generated"] += 1
        self.generation_stats["total_tokens_generated"] += rag_context.total_tokens
        
        # Update average generation time
        current_avg = self.generation_stats["average_generation_time"]
        total_generated = self.generation_stats["total_contexts_generated"]
        
        self.generation_stats["average_generation_time"] = (
            (current_avg * (total_generated - 1) + generation_time) / total_generated
        )
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get context generation statistics"""
        return self.generation_stats.copy()
    
    def clear_cache(self):
        """Clear all caches"""
        if self.enable_caching:
            self.context_cache.clear()
            self.fragment_cache.clear()
        logger.info("Context caches cleared")


# Export main classes
__all__ = [
    "ContextType",
    "ContextPriority",
    "ContextFragment",
    "RAGContext",
    "RAGContextGenerator"
]