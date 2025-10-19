"""
Enhanced Semantic Engine with LLM Integration

This module extends the semantic layer with Large Language Model capabilities
for advanced embedding generation, relationship discovery, and business term
enhancement. It provides AI-powered semantic understanding for enterprise
knowledge graphs.

Features:
- LLM-powered embedding generation
- Intelligent relationship discovery
- Business glossary enhancement
- Semantic similarity computation
- Context-aware term definitions

Author: anant development team
Date: October 2025
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    from sentence_transformers import SentenceTransformer
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


@dataclass
class SemanticEmbedding:
    """Enhanced semantic embedding with LLM-generated features"""
    entity_id: str
    embedding_vector: np.ndarray
    confidence: float
    source_model: str
    generated_at: datetime = field(default_factory=datetime.now)
    context_used: str = ""
    dimensions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.dimensions == 0:
            self.dimensions = len(self.embedding_vector)


@dataclass
class BusinessTerm:
    """Enhanced business term with LLM-generated definitions"""
    term: str
    definition: str
    category: str
    confidence: float
    source: str = "llm_generated"
    created_at: datetime = field(default_factory=datetime.now)
    related_terms: List[str] = field(default_factory=list)
    business_context: str = ""
    usage_examples: List[str] = field(default_factory=list)
    domain_specific: bool = False
    quality_score: float = 0.0


@dataclass
class SemanticRelationship:
    """Enhanced semantic relationship with AI-discovered properties"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    strength: float
    discovery_method: str
    discovered_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"  # pending, validated, rejected


class EnhancedSemanticEngine:
    """
    Enhanced Semantic Engine with LLM Integration
    
    Provides advanced semantic understanding capabilities using Large Language
    Models for embedding generation, relationship discovery, and business
    glossary enhancement.
    """
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_backend: str = "auto",
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-3.5-turbo",
                 enable_caching: bool = True,
                 cache_size: int = 10000):
        """
        Initialize the Enhanced Semantic Engine
        
        Parameters
        ----------
        embedding_model : str
            Sentence transformer model for embeddings
        llm_backend : str
            LLM backend ("openai", "transformers", "auto")
        openai_api_key : str, optional
            OpenAI API key for GPT models
        llm_model : str
            LLM model name
        enable_caching : bool
            Whether to cache embeddings and results
        cache_size : int
            Maximum cache size
        """
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize embedding model
        self._init_embedding_model(embedding_model)
        
        # Initialize LLM backend
        self.llm_backend = self._select_llm_backend(llm_backend)
        self._init_llm_backend(openai_api_key, llm_model)
        
        # Initialize caches
        if enable_caching:
            self.embedding_cache = {}
            self.definition_cache = {}
            self.relationship_cache = {}
        
        # Business glossary and semantic data
        self.business_terms = {}
        self.semantic_relationships = []
        self.entity_embeddings = {}
        
        logger.info(f"Enhanced Semantic Engine initialized with embedding model: {embedding_model}, "
                   f"LLM backend: {self.llm_backend}")
    
    def _init_embedding_model(self, model_name: str):
        """Initialize sentence transformer model"""
        if HAS_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
                self.embedding_dim = 384  # Default dimension
        else:
            logger.warning("Transformers not available, using fallback embedding")
            self.embedding_model = None
            self.embedding_dim = 384
    
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
        """Initialize LLM backend"""
        if self.llm_backend == "openai":
            if api_key:
                openai.api_key = api_key
            self.llm_model = model_name
        elif self.llm_backend == "transformers":
            try:
                self.text_generator = pipeline("text-generation", 
                                             model="microsoft/DialoGPT-medium",
                                             max_length=100)
                self.llm_model = "transformers_local"
            except Exception as e:
                logger.warning(f"Failed to initialize transformers pipeline: {e}")
                self.llm_backend = "fallback"
        else:
            self.llm_backend = "fallback"
            logger.info("Using fallback LLM backend")
    
    def generate_enhanced_embedding(self, 
                                  entity_id: str, 
                                  text_content: str,
                                  context: Optional[str] = None) -> SemanticEmbedding:
        """
        Generate enhanced embedding using LLM-enriched content
        
        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        text_content : str
            Text content to embed
        context : str, optional
            Additional context for embedding generation
            
        Returns
        -------
        SemanticEmbedding
            Enhanced embedding with metadata
        """
        # Check cache first
        cache_key = f"{entity_id}:{hash(text_content)}"
        if self.enable_caching and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Enhance text content with LLM-generated context
            enhanced_content = self._enhance_content_with_llm(text_content, context)
            
            # Generate embedding
            if self.embedding_model:
                embedding_vector = self.embedding_model.encode(enhanced_content)
                confidence = 0.9  # High confidence for transformer models
                source_model = str(type(self.embedding_model).__name__)
            else:
                # Fallback to simple text-based embedding
                embedding_vector = self._generate_fallback_embedding(enhanced_content)
                confidence = 0.5
                source_model = "fallback"
            
            # Create enhanced embedding
            semantic_embedding = SemanticEmbedding(
                entity_id=entity_id,
                embedding_vector=embedding_vector,
                confidence=confidence,
                source_model=source_model,
                context_used=context or "",
                metadata={
                    "original_text_length": len(text_content),
                    "enhanced_text_length": len(enhanced_content),
                    "enhancement_applied": enhanced_content != text_content
                }
            )
            
            # Cache result
            if self.enable_caching:
                self.embedding_cache[cache_key] = semantic_embedding
                self._manage_cache_size()
            
            # Store in entity embeddings
            self.entity_embeddings[entity_id] = semantic_embedding
            
            logger.debug(f"Generated enhanced embedding for entity: {entity_id}")
            return semantic_embedding
            
        except Exception as e:
            logger.error(f"Error generating enhanced embedding: {e}")
            # Return fallback embedding
            fallback_vector = self._generate_fallback_embedding(text_content)
            return SemanticEmbedding(
                entity_id=entity_id,
                embedding_vector=fallback_vector,
                confidence=0.1,
                source_model="error_fallback",
                metadata={"error": str(e)}
            )
    
    def _enhance_content_with_llm(self, content: str, context: Optional[str] = None) -> str:
        """Enhance content using LLM for better semantic understanding"""
        if self.llm_backend == "openai":
            return self._enhance_with_openai(content, context)
        elif self.llm_backend == "transformers":
            return self._enhance_with_transformers(content, context)
        else:
            return content  # No enhancement available
    
    def _enhance_with_openai(self, content: str, context: Optional[str] = None) -> str:
        """Enhance content using OpenAI GPT models"""
        try:
            prompt = f"""
            Enhance the following text for better semantic understanding and embedding generation.
            Add relevant context, expand abbreviations, and clarify technical terms.
            
            Original text: {content}
            {"Context: " + context if context else ""}
            
            Enhanced text:"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            enhanced = response.choices[0].message.content.strip()
            return enhanced if enhanced else content
            
        except Exception as e:
            logger.warning(f"OpenAI enhancement failed: {e}")
            return content
    
    def _enhance_with_transformers(self, content: str, context: Optional[str] = None) -> str:
        """Enhance content using local transformer models"""
        try:
            prompt = f"Expand and clarify: {content}"
            if context:
                prompt += f" (Context: {context})"
            
            generated = self.text_generator(prompt, max_length=len(prompt) + 50)
            enhanced = generated[0]['generated_text'][len(prompt):].strip()
            
            return content + " " + enhanced if enhanced else content
            
        except Exception as e:
            logger.warning(f"Transformers enhancement failed: {e}")
            return content
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple fallback embedding using basic text features"""
        # Simple bag-of-words style embedding
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create fixed-size vector
        vector = np.zeros(self.embedding_dim)
        for i, word in enumerate(words[:self.embedding_dim]):
            vector[i] = hash(word) % 1000 / 1000.0
        
        return vector
    
    def discover_semantic_relationships(self, 
                                      entities: List[str],
                                      similarity_threshold: float = 0.7,
                                      max_relationships: int = 100) -> List[SemanticRelationship]:
        """
        Discover semantic relationships between entities using LLM analysis
        
        Parameters
        ----------
        entities : List[str]
            List of entity IDs to analyze
        similarity_threshold : float
            Minimum similarity score for relationship detection
        max_relationships : int
            Maximum number of relationships to return
            
        Returns
        -------
        List[SemanticRelationship]
            Discovered semantic relationships
        """
        relationships = []
        
        try:
            # Generate embeddings for all entities if not already done
            entity_embeddings = {}
            for entity_id in entities:
                if entity_id in self.entity_embeddings:
                    entity_embeddings[entity_id] = self.entity_embeddings[entity_id]
                else:
                    # Generate embedding with placeholder content
                    embedding = self.generate_enhanced_embedding(entity_id, entity_id)
                    entity_embeddings[entity_id] = embedding
            
            # Compute pairwise similarities
            entity_list = list(entity_embeddings.keys())
            for i, entity1 in enumerate(entity_list):
                for j, entity2 in enumerate(entity_list[i+1:], i+1):
                    similarity = self._compute_similarity(
                        entity_embeddings[entity1].embedding_vector,
                        entity_embeddings[entity2].embedding_vector
                    )
                    
                    if similarity >= similarity_threshold:
                        # Use LLM to determine relationship type
                        relationship_type = self._determine_relationship_type(entity1, entity2)
                        
                        relationship = SemanticRelationship(
                            source_entity=entity1,
                            target_entity=entity2,
                            relationship_type=relationship_type,
                            confidence=similarity,
                            strength=similarity,
                            discovery_method="llm_similarity",
                            context={
                                "similarity_score": similarity,
                                "threshold_used": similarity_threshold
                            }
                        )
                        
                        relationships.append(relationship)
                        
                        if len(relationships) >= max_relationships:
                            break
                
                if len(relationships) >= max_relationships:
                    break
            
            # Store discovered relationships
            self.semantic_relationships.extend(relationships)
            
            logger.info(f"Discovered {len(relationships)} semantic relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Error discovering semantic relationships: {e}")
            return []
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def _determine_relationship_type(self, entity1: str, entity2: str) -> str:
        """Use LLM to determine the type of relationship between entities"""
        if self.llm_backend == "openai":
            return self._determine_relationship_openai(entity1, entity2)
        elif self.llm_backend == "transformers":
            return self._determine_relationship_transformers(entity1, entity2)
        else:
            return "related_to"  # Default relationship type
    
    def _determine_relationship_openai(self, entity1: str, entity2: str) -> str:
        """Determine relationship type using OpenAI"""
        try:
            prompt = f"""
            Analyze the relationship between these two entities and classify it:
            Entity 1: {entity1}
            Entity 2: {entity2}
            
            Choose the most appropriate relationship type from:
            - part_of
            - contains
            - similar_to
            - depends_on
            - created_by
            - manages
            - related_to
            - opposite_of
            - category_of
            
            Relationship type:"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1
            )
            
            relationship_type = response.choices[0].message.content.strip().lower()
            
            # Validate relationship type
            valid_types = [
                "part_of", "contains", "similar_to", "depends_on", "created_by",
                "manages", "related_to", "opposite_of", "category_of"
            ]
            
            return relationship_type if relationship_type in valid_types else "related_to"
            
        except Exception as e:
            logger.warning(f"OpenAI relationship classification failed: {e}")
            return "related_to"
    
    def _determine_relationship_transformers(self, entity1: str, entity2: str) -> str:
        """Determine relationship type using local transformers"""
        try:
            # Simple classification based on entity names
            entity1_lower = entity1.lower()
            entity2_lower = entity2.lower()
            
            # Rule-based classification for now
            if any(word in entity1_lower for word in ["manager", "boss", "lead"]):
                return "manages"
            elif any(word in entity1_lower for word in ["team", "group", "department"]):
                return "contains"
            elif any(word in entity1_lower for word in ["part", "component", "element"]):
                return "part_of"
            else:
                return "related_to"
                
        except Exception as e:
            logger.warning(f"Transformers relationship classification failed: {e}")
            return "related_to"
    
    def enhance_business_glossary(self, 
                                terms: List[str],
                                domain_context: Optional[str] = None) -> List[BusinessTerm]:
        """
        Enhance business glossary using LLM-generated definitions
        
        Parameters
        ----------
        terms : List[str]
            List of business terms to define
        domain_context : str, optional
            Domain or industry context for better definitions
            
        Returns
        -------
        List[BusinessTerm]
            Enhanced business terms with LLM-generated definitions
        """
        enhanced_terms = []
        
        for term in terms:
            try:
                # Check cache first
                cache_key = f"{term}:{domain_context or 'general'}"
                if self.enable_caching and cache_key in self.definition_cache:
                    enhanced_terms.append(self.definition_cache[cache_key])
                    continue
                
                # Generate definition using LLM
                definition_data = self._generate_term_definition(term, domain_context)
                
                business_term = BusinessTerm(
                    term=term,
                    definition=definition_data["definition"],
                    category=definition_data["category"],
                    confidence=definition_data["confidence"],
                    business_context=domain_context or "",
                    related_terms=definition_data.get("related_terms", []),
                    usage_examples=definition_data.get("examples", []),
                    domain_specific=domain_context is not None,
                    quality_score=definition_data.get("quality_score", 0.7)
                )
                
                enhanced_terms.append(business_term)
                
                # Store in business terms and cache
                self.business_terms[term] = business_term
                if self.enable_caching:
                    self.definition_cache[cache_key] = business_term
                
            except Exception as e:
                logger.error(f"Error enhancing term '{term}': {e}")
                # Create fallback term
                fallback_term = BusinessTerm(
                    term=term,
                    definition=f"Business term: {term}",
                    category="general",
                    confidence=0.1,
                    source="fallback"
                )
                enhanced_terms.append(fallback_term)
        
        logger.info(f"Enhanced {len(enhanced_terms)} business terms")
        return enhanced_terms
    
    def _generate_term_definition(self, term: str, domain_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive term definition using LLM"""
        if self.llm_backend == "openai":
            return self._generate_definition_openai(term, domain_context)
        elif self.llm_backend == "transformers":
            return self._generate_definition_transformers(term, domain_context)
        else:
            return self._generate_definition_fallback(term, domain_context)
    
    def _generate_definition_openai(self, term: str, domain_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate term definition using OpenAI"""
        try:
            context_prompt = f" in the context of {domain_context}" if domain_context else ""
            
            prompt = f"""
            Provide a comprehensive definition for the business term "{term}"{context_prompt}.
            
            Format your response as JSON with the following structure:
            {{
                "definition": "Clear, concise definition",
                "category": "Choose from: process, role, system, metric, concept, product, service",
                "confidence": 0.8,
                "related_terms": ["term1", "term2", "term3"],
                "examples": ["usage example 1", "usage example 2"],
                "quality_score": 0.9
            }}
            
            JSON response:"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                definition_data = json.loads(response_text)
                return definition_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "definition": response_text[:200],
                    "category": "concept",
                    "confidence": 0.6,
                    "related_terms": [],
                    "examples": [],
                    "quality_score": 0.5
                }
                
        except Exception as e:
            logger.warning(f"OpenAI definition generation failed: {e}")
            return self._generate_definition_fallback(term, domain_context)
    
    def _generate_definition_transformers(self, term: str, domain_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate term definition using local transformers"""
        try:
            context_prompt = f" in {domain_context}" if domain_context else ""
            prompt = f"Define the business term {term}{context_prompt}:"
            
            generated = self.text_generator(prompt, max_length=100)
            definition = generated[0]['generated_text'][len(prompt):].strip()
            
            return {
                "definition": definition if definition else f"Business term related to {term}",
                "category": "concept",
                "confidence": 0.5,
                "related_terms": [],
                "examples": [],
                "quality_score": 0.4
            }
            
        except Exception as e:
            logger.warning(f"Transformers definition generation failed: {e}")
            return self._generate_definition_fallback(term, domain_context)
    
    def _generate_definition_fallback(self, term: str, domain_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate fallback definition"""
        context_text = f" in the context of {domain_context}" if domain_context else ""
        
        return {
            "definition": f"A business term referring to {term.lower()}{context_text}",
            "category": "concept",
            "confidence": 0.2,
            "related_terms": [],
            "examples": [],
            "quality_score": 0.2
        }
    
    def find_similar_entities(self, 
                            query_entity: str, 
                            similarity_threshold: float = 0.7,
                            max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities similar to a query entity using semantic embeddings
        
        Parameters
        ----------
        query_entity : str
            Entity to find similarities for
        similarity_threshold : float
            Minimum similarity score
        max_results : int
            Maximum number of results to return
            
        Returns
        -------
        List[Dict[str, Any]]
            Similar entities with similarity scores
        """
        if query_entity not in self.entity_embeddings:
            logger.warning(f"No embedding found for query entity: {query_entity}")
            return []
        
        query_embedding = self.entity_embeddings[query_entity].embedding_vector
        similarities = []
        
        for entity_id, embedding_obj in self.entity_embeddings.items():
            if entity_id == query_entity:
                continue
            
            similarity = self._compute_similarity(query_embedding, embedding_obj.embedding_vector)
            
            if similarity >= similarity_threshold:
                similarities.append({
                    "entity_id": entity_id,
                    "similarity_score": similarity,
                    "embedding_confidence": embedding_obj.confidence,
                    "source_model": embedding_obj.source_model
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similarities[:max_results]
    
    def get_semantic_context(self, entity_id: str, context_depth: int = 2) -> Dict[str, Any]:
        """
        Get comprehensive semantic context for an entity
        
        Parameters
        ----------
        entity_id : str
            Entity to get context for
        context_depth : int
            Depth of context to retrieve
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive semantic context
        """
        context = {
            "entity_id": entity_id,
            "embedding_info": {},
            "business_term_info": {},
            "relationships": [],
            "similar_entities": [],
            "context_depth": context_depth
        }
        
        # Add embedding information
        if entity_id in self.entity_embeddings:
            embedding = self.entity_embeddings[entity_id]
            context["embedding_info"] = {
                "confidence": embedding.confidence,
                "source_model": embedding.source_model,
                "dimensions": embedding.dimensions,
                "generated_at": embedding.generated_at.isoformat(),
                "context_used": embedding.context_used
            }
        
        # Add business term information
        if entity_id in self.business_terms:
            term = self.business_terms[entity_id]
            context["business_term_info"] = {
                "definition": term.definition,
                "category": term.category,
                "confidence": term.confidence,
                "related_terms": term.related_terms,
                "usage_examples": term.usage_examples,
                "quality_score": term.quality_score
            }
        
        # Add relationships
        entity_relationships = [
            {
                "target": rel.target_entity,
                "type": rel.relationship_type,
                "confidence": rel.confidence,
                "strength": rel.strength
            }
            for rel in self.semantic_relationships
            if rel.source_entity == entity_id
        ]
        context["relationships"] = entity_relationships
        
        # Add similar entities
        similar_entities = self.find_similar_entities(entity_id, similarity_threshold=0.6, max_results=5)
        context["similar_entities"] = similar_entities
        
        return context
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues"""
        if not self.enable_caching:
            return
        
        # Clear oldest entries if cache is too large
        for cache in [self.embedding_cache, self.definition_cache, self.relationship_cache]:
            if len(cache) > self.cache_size:
                # Remove 20% of oldest entries
                items_to_remove = len(cache) - int(self.cache_size * 0.8)
                for _ in range(items_to_remove):
                    cache.pop(next(iter(cache)))
    
    def export_semantic_data(self) -> Dict[str, Any]:
        """Export all semantic data for persistence"""
        return {
            "entity_embeddings": {
                entity_id: {
                    "embedding_vector": embedding.embedding_vector.tolist(),
                    "confidence": embedding.confidence,
                    "source_model": embedding.source_model,
                    "generated_at": embedding.generated_at.isoformat(),
                    "context_used": embedding.context_used,
                    "metadata": embedding.metadata
                }
                for entity_id, embedding in self.entity_embeddings.items()
            },
            "business_terms": {
                term_id: {
                    "term": term.term,
                    "definition": term.definition,
                    "category": term.category,
                    "confidence": term.confidence,
                    "source": term.source,
                    "created_at": term.created_at.isoformat(),
                    "related_terms": term.related_terms,
                    "business_context": term.business_context,
                    "usage_examples": term.usage_examples,
                    "domain_specific": term.domain_specific,
                    "quality_score": term.quality_score
                }
                for term_id, term in self.business_terms.items()
            },
            "semantic_relationships": [
                {
                    "source_entity": rel.source_entity,
                    "target_entity": rel.target_entity,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                    "strength": rel.strength,
                    "discovery_method": rel.discovery_method,
                    "discovered_at": rel.discovered_at.isoformat(),
                    "context": rel.context,
                    "validation_status": rel.validation_status
                }
                for rel in self.semantic_relationships
            ]
        }


# Export main classes
__all__ = [
    "SemanticEmbedding",
    "BusinessTerm", 
    "SemanticRelationship",
    "EnhancedSemanticEngine"
]