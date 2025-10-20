"""
Semantic Layer for Metagraph - Phase 1
======================================

Manages semantic relationships, embeddings, and business context.
Provides foundation for AI-driven query processing and knowledge discovery.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Literal
from datetime import datetime
import orjson
import uuid
from dataclasses import dataclass

# Type aliases
SimilarityMetric = Literal["cosine", "euclidean", "dot_product", "jaccard"]
RelationshipType = Literal["semantic", "structural", "temporal", "causal", "hierarchical"]
ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]


@dataclass
class SemanticRelationship:
    """Represents a semantic relationship between entities."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    context: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any]


class SemanticLayer:
    """
    Enterprise semantic layer for knowledge graph functionality.
    
    Manages semantic relationships, embeddings, and business context
    with Polars+Parquet backend for high-performance operations.
    """
    
    def __init__(self, 
                 storage_path: str = "./metagraph_semantic",
                 embedding_dimension: int = 768,
                 compression: ParquetCompression = "zstd"):
        """
        Initialize semantic layer with optimized storage.
        
        Args:
            storage_path: Path for storing semantic data
            embedding_dimension: Dimension for semantic embeddings
            compression: Parquet compression algorithm
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dimension
        self.compression: ParquetCompression = compression
        
        # File paths
        self.relationships_file = self.storage_path / "relationships.parquet"
        self.embeddings_file = self.storage_path / "embeddings.parquet"
        self.business_glossary_file = self.storage_path / "business_glossary.parquet"
        
        # Initialize data structures
        self._init_relationships()
        self._init_embeddings()
        self._init_business_glossary()
        
        # Semantic caches
        self._similarity_cache: Dict[str, float] = {}
        self._relationship_cache: Dict[str, List[SemanticRelationship]] = {}
    
    def _init_relationships(self):
        """Initialize relationships DataFrame."""
        if self.relationships_file.exists():
            self.relationships_df = pl.read_parquet(self.relationships_file)
        else:
            self.relationships_df = pl.DataFrame({
                "relationship_id": pl.Series([], dtype=pl.Utf8),
                "source_id": pl.Series([], dtype=pl.Utf8),
                "target_id": pl.Series([], dtype=pl.Utf8),
                "relationship_type": pl.Series([], dtype=pl.Utf8),
                "strength": pl.Series([], dtype=pl.Float64),
                "bidirectional": pl.Series([], dtype=pl.Boolean),
                "context": pl.Series([], dtype=pl.Utf8),  # JSON
                "created_at": pl.Series([], dtype=pl.Datetime),
                "updated_at": pl.Series([], dtype=pl.Datetime),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                "tags": pl.Series([], dtype=pl.List(pl.Utf8)),
                "confidence": pl.Series([], dtype=pl.Float64),
                "validation_status": pl.Series([], dtype=pl.Utf8)
            })
    
    def _init_embeddings(self):
        """Initialize embeddings DataFrame."""
        if self.embeddings_file.exists():
            self.embeddings_df = pl.read_parquet(self.embeddings_file)
        else:
            # Create schema for embeddings
            embedding_cols = {f"emb_{i}": pl.Series([], dtype=pl.Float32) 
                            for i in range(self.embedding_dim)}
            
            self.embeddings_df = pl.DataFrame({
                "entity_id": pl.Series([], dtype=pl.Utf8),
                "embedding_type": pl.Series([], dtype=pl.Utf8),
                "model_name": pl.Series([], dtype=pl.Utf8),
                "created_at": pl.Series([], dtype=pl.Datetime),
                "metadata": pl.Series([], dtype=pl.Utf8),  # JSON
                **embedding_cols
            })
    
    def _init_business_glossary(self):
        """Initialize business glossary DataFrame."""
        if self.business_glossary_file.exists():
            self.business_glossary_df = pl.read_parquet(self.business_glossary_file)
        else:
            self.business_glossary_df = pl.DataFrame({
                "term_id": pl.Series([], dtype=pl.Utf8),
                "term": pl.Series([], dtype=pl.Utf8),
                "definition": pl.Series([], dtype=pl.Utf8),
                "context": pl.Series([], dtype=pl.Utf8),
                "domain": pl.Series([], dtype=pl.Utf8),
                "synonyms": pl.Series([], dtype=pl.List(pl.Utf8)),
                "related_terms": pl.Series([], dtype=pl.List(pl.Utf8)),
                "business_rules": pl.Series([], dtype=pl.Utf8),  # JSON
                "usage_examples": pl.Series([], dtype=pl.List(pl.Utf8)),
                "data_sources": pl.Series([], dtype=pl.List(pl.Utf8)),
                "created_at": pl.Series([], dtype=pl.Datetime),
                "updated_at": pl.Series([], dtype=pl.Datetime),
                "created_by": pl.Series([], dtype=pl.Utf8),
                "status": pl.Series([], dtype=pl.Utf8)
            })
    
    def add_relationship(self, 
                        source_id: str,
                        target_id: str,
                        relationship_type: RelationshipType,
                        strength: float = 1.0,
                        bidirectional: bool = False,
                        context: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None,
                        confidence: float = 1.0) -> str:
        """
        Add semantic relationship between entities.
        
        Args:
            source_id: Source entity identifier
            target_id: Target entity identifier  
            relationship_type: Type of semantic relationship
            strength: Relationship strength (0.0 to 1.0)
            bidirectional: Whether relationship works both ways
            context: Additional context information
            metadata: Custom metadata
            tags: Classification tags
            confidence: Confidence in relationship (0.0 to 1.0)
            
        Returns:
            relationship_id: Unique identifier for the relationship
        """
        relationship_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_relationship = pl.DataFrame({
            "relationship_id": [relationship_id],
            "source_id": [source_id],
            "target_id": [target_id],
            "relationship_type": [relationship_type],
            "strength": [max(0.0, min(1.0, strength))],  # Clamp to [0,1]
            "bidirectional": [bidirectional],
            "context": [orjson.dumps(context or {}).decode()],
            "created_at": [now],
            "updated_at": [now],
            "metadata": [orjson.dumps(metadata or {}).decode()],
            "tags": [tags or []],
            "confidence": [max(0.0, min(1.0, confidence))],
            "validation_status": ["pending"]
        })
        
        self.relationships_df = pl.concat([self.relationships_df, new_relationship])
        
        # Clear cache
        self._relationship_cache.clear()
        
        return relationship_id
    
    def get_relationships(self, 
                         entity_id: str,
                         relationship_types: Optional[List[RelationshipType]] = None,
                         min_strength: float = 0.0,
                         include_bidirectional: bool = True) -> List[SemanticRelationship]:
        """
        Get semantic relationships for an entity.
        
        Args:
            entity_id: Entity to find relationships for
            relationship_types: Filter by relationship types
            min_strength: Minimum relationship strength
            include_bidirectional: Include bidirectional relationships
            
        Returns:
            List of semantic relationships
        """
        cache_key = f"{entity_id}_{relationship_types}_{min_strength}_{include_bidirectional}"
        if cache_key in self._relationship_cache:
            return self._relationship_cache[cache_key]
        
        # Query relationships
        query = self.relationships_df.filter(
            (pl.col("source_id") == entity_id) |
            (include_bidirectional & pl.col("target_id") == entity_id & pl.col("bidirectional"))
        ).filter(pl.col("strength") >= min_strength)
        
        if relationship_types:
            query = query.filter(pl.col("relationship_type").is_in(relationship_types))
        
        relationships = []
        for row in query.iter_rows(named=True):
            relationships.append(SemanticRelationship(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relationship_type=row["relationship_type"],
                strength=row["strength"],
                context=orjson.loads(row["context"]),
                created_at=row["created_at"],
                metadata=orjson.loads(row["metadata"])
            ))
        
        self._relationship_cache[cache_key] = relationships
        return relationships
    
    def add_embedding(self,
                     entity_id: str,
                     embedding: Union[List[float], np.ndarray],
                     embedding_type: str = "semantic",
                     model_name: str = "unknown",
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add semantic embedding for an entity.
        
        Args:
            entity_id: Entity identifier
            embedding: Dense vector embedding
            embedding_type: Type of embedding (semantic, structural, etc.)
            model_name: Name of the model that generated embedding
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
        # Remove existing embedding for this entity and type
        self.embeddings_df = self.embeddings_df.filter(
            ~((pl.col("entity_id") == entity_id) & (pl.col("embedding_type") == embedding_type))
        )
        
        # Create new embedding row
        embedding_data = {
            "entity_id": [entity_id],
            "embedding_type": [embedding_type],
            "model_name": [model_name],
            "created_at": [datetime.now()],
            "metadata": [orjson.dumps(metadata or {}).decode()]
        }
        
        # Add embedding dimensions
        for i, value in enumerate(embedding):
            embedding_data[f"emb_{i}"] = [float(value)]
        
        new_embedding = pl.DataFrame(embedding_data)
        self.embeddings_df = pl.concat([self.embeddings_df, new_embedding])
        
        return True
    
    def get_embedding(self, 
                     entity_id: str,
                     embedding_type: str = "semantic") -> Optional[np.ndarray]:
        """
        Get semantic embedding for an entity.
        
        Args:
            entity_id: Entity identifier
            embedding_type: Type of embedding to retrieve
            
        Returns:
            Embedding vector or None if not found
        """
        result = self.embeddings_df.filter(
            (pl.col("entity_id") == entity_id) & 
            (pl.col("embedding_type") == embedding_type)
        )
        
        if result.height == 0:
            return None
        
        # Extract embedding columns
        embedding_cols = [f"emb_{i}" for i in range(self.embedding_dim)]
        embedding_row = result.select(embedding_cols).row(0)
        
        return np.array(embedding_row, dtype=np.float32)
    
    def find_similar_entities(self,
                             entity_id: str,
                             top_k: int = 10,
                             metric: SimilarityMetric = "cosine",
                             embedding_type: str = "semantic",
                             min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find entities similar to the given entity based on embeddings.
        
        Args:
            entity_id: Reference entity
            top_k: Number of similar entities to return
            metric: Similarity metric to use
            embedding_type: Type of embedding to compare
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        reference_embedding = self.get_embedding(entity_id, embedding_type)
        if reference_embedding is None:
            return []
        
        # Get all embeddings of the same type
        embeddings_query = self.embeddings_df.filter(
            (pl.col("embedding_type") == embedding_type) &
            (pl.col("entity_id") != entity_id)
        )
        
        if embeddings_query.height == 0:
            return []
        
        similarities = []
        embedding_cols = [f"emb_{i}" for i in range(self.embedding_dim)]
        
        for row in embeddings_query.iter_rows(named=True):
            other_embedding = np.array([row[col] for col in embedding_cols], dtype=np.float32)
            similarity = self._calculate_similarity(reference_embedding, other_embedding, metric)
            
            if similarity >= min_similarity:
                similarities.append((row["entity_id"], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_similarity(self, 
                            vec1: np.ndarray, 
                            vec2: np.ndarray, 
                            metric: SimilarityMetric) -> float:
        """Calculate similarity between two vectors."""
        if metric == "cosine":
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        elif metric == "euclidean":
            distance = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + distance))  # Convert distance to similarity
        
        elif metric == "dot_product":
            return float(np.dot(vec1, vec2))
        
        elif metric == "jaccard":
            # For dense vectors, use threshold-based Jaccard
            threshold = 0.5
            set1 = set(np.where(vec1 > threshold)[0])
            set2 = set(np.where(vec2 > threshold)[0])
            
            if len(set1) == 0 and len(set2) == 0:
                return 1.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return float(intersection / union) if union > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def add_business_term(self,
                         term: str,
                         definition: str,
                         context: str = "",
                         domain: str = "general",
                         synonyms: Optional[List[str]] = None,
                         related_terms: Optional[List[str]] = None,
                         business_rules: Optional[Dict[str, Any]] = None,
                         usage_examples: Optional[List[str]] = None,
                         data_sources: Optional[List[str]] = None,
                         created_by: str = "system") -> str:
        """
        Add term to business glossary.
        
        Args:
            term: Business term
            definition: Formal definition
            context: Usage context
            domain: Business domain
            synonyms: Alternative terms
            related_terms: Related business terms
            business_rules: Associated business rules
            usage_examples: Example usage
            data_sources: Related data sources
            created_by: Creator identifier
            
        Returns:
            term_id: Unique identifier for the term
        """
        term_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_term = pl.DataFrame({
            "term_id": [term_id],
            "term": [term],
            "definition": [definition],
            "context": [context],
            "domain": [domain],
            "synonyms": [synonyms or []],
            "related_terms": [related_terms or []],
            "business_rules": [orjson.dumps(business_rules or {}).decode()],
            "usage_examples": [usage_examples or []],
            "data_sources": [data_sources or []],
            "created_at": [now],
            "updated_at": [now],
            "created_by": [created_by],
            "status": ["active"]
        })
        
        self.business_glossary_df = pl.concat([self.business_glossary_df, new_term])
        return term_id
    
    def search_business_terms(self, 
                             query: str,
                             domain: Optional[str] = None,
                             include_synonyms: bool = True) -> List[Dict[str, Any]]:
        """
        Search business glossary terms.
        
        Args:
            query: Search query
            domain: Filter by domain
            include_synonyms: Include synonym matches
            
        Returns:
            List of matching terms
        """
        query_lower = query.lower()
        
        # Base filter
        filter_expr = (
            pl.col("term").str.to_lowercase().str.contains(query_lower) |
            pl.col("definition").str.to_lowercase().str.contains(query_lower) |
            pl.col("context").str.to_lowercase().str.contains(query_lower)
        )
        
        if include_synonyms:
            # Add synonym search (this is a simplified approach)
            filter_expr = filter_expr | pl.col("synonyms").list.eval(
                pl.element().str.to_lowercase().str.contains(query_lower)
            ).list.any()
        
        result = self.business_glossary_df.filter(filter_expr)
        
        if domain:
            result = result.filter(pl.col("domain") == domain)
        
        # Convert to list of dictionaries
        terms = []
        for row in result.iter_rows(named=True):
            terms.append({
                "term_id": row["term_id"],
                "term": row["term"],
                "definition": row["definition"],
                "context": row["context"],
                "domain": row["domain"],
                "synonyms": row["synonyms"],
                "related_terms": row["related_terms"],
                "business_rules": orjson.loads(row["business_rules"]),
                "usage_examples": row["usage_examples"],
                "data_sources": row["data_sources"],
                "created_at": row["created_at"],
                "status": row["status"]
            })
        
        return terms
    
    def get_semantic_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Get comprehensive semantic context for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Semantic context including relationships, embeddings, and business terms
        """
        context = {
            "entity_id": entity_id,
            "relationships": [],
            "similar_entities": [],
            "business_terms": [],
            "embedding_info": {}
        }
        
        # Get relationships
        relationships = self.get_relationships(entity_id)
        context["relationships"] = [
            {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.relationship_type,
                "strength": rel.strength,
                "context": rel.context
            }
            for rel in relationships
        ]
        
        # Get similar entities
        similar = self.find_similar_entities(entity_id, top_k=5)
        context["similar_entities"] = [
            {"entity_id": eid, "similarity": sim}
            for eid, sim in similar
        ]
        
        # Check for embedding info
        embedding = self.get_embedding(entity_id)
        if embedding is not None:
            context["embedding_info"] = {
                "has_embedding": True,
                "dimension": len(embedding),
                "norm": float(np.linalg.norm(embedding))
            }
        else:
            context["embedding_info"] = {"has_embedding": False}
        
        return context
    
    def save(self):
        """Save all semantic data to Parquet files."""
        self.relationships_df.write_parquet(self.relationships_file, compression=self.compression)
        self.embeddings_df.write_parquet(self.embeddings_file, compression=self.compression)
        self.business_glossary_df.write_parquet(self.business_glossary_file, compression=self.compression)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic layer statistics."""
        return {
            "relationships_count": self.relationships_df.height,
            "embeddings_count": self.embeddings_df.height,
            "business_terms_count": self.business_glossary_df.height,
            "unique_entities_with_embeddings": self.embeddings_df["entity_id"].n_unique(),
            "relationship_types": self.relationships_df["relationship_type"].unique().to_list(),
            "embedding_types": self.embeddings_df["embedding_type"].unique().to_list(),
            "business_domains": self.business_glossary_df["domain"].unique().to_list()
        }
    
    @property
    def relationships(self) -> Dict[str, Dict[str, Any]]:
        """Get relationships as a dictionary."""
        if self.relationships_df.is_empty():
            return {}
        
        result = {}
        for row in self.relationships_df.to_dicts():
            rel_id = row["relationship_id"]
            result[rel_id] = {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "relationship_type": row["relationship_type"],
                "strength": row["strength"],
                "metadata": orjson.loads(row["metadata"]) if row["metadata"] else {}
            }
        return result
    
    def add_relationship_type(self, type_name: str, properties: Optional[dict] = None) -> bool:
        """Add a new relationship type."""
        try:
            if properties is None:
                properties = {}
            
            # Create new relationship type record
            new_type = pl.DataFrame({
                'type_name': [type_name],
                'properties': [orjson.dumps(properties).decode()],
                'created_at': [datetime.now()]
            })
            
            # Initialize relationship_types_df if it doesn't exist
            if not hasattr(self, 'relationship_types_df') or self.relationship_types_df is None:
                self.relationship_types_df = new_type
            else:
                self.relationship_types_df = self.relationship_types_df.vstack(new_type)
            
            return True
        except Exception:
            return False
    
    def get_relationship_types(self) -> List[str]:
        """Get all relationship types."""
        try:
            if not hasattr(self, 'relationship_types_df') or self.relationship_types_df is None or self.relationship_types_df.is_empty():
                return []
            
            return self.relationship_types_df.select("type_name").to_series().to_list()
        except Exception:
            return []
    
    def query_relationships(self, relationship_type: Optional[str] = None, source_id: Optional[str] = None, target_id: Optional[str] = None) -> List[dict]:
        """Query relationships with optional filters."""
        try:
            df = self.relationships_df
            
            if relationship_type:
                df = df.filter(pl.col("relationship_type") == relationship_type)
            
            if source_id:
                df = df.filter(pl.col("source_id") == source_id)
            
            if target_id:
                df = df.filter(pl.col("target_id") == target_id)
            
            if df.is_empty():
                return []
            
            return df.to_dicts()
        except Exception:
            return []