"""
Machine Learning Integration for LCG
====================================

Integrates ML capabilities with LayeredContextualGraph:
- Embedding layers for semantic similarity
- Predictive collapse (predict which quantum state will collapse)
- Auto-context detection using ML
- Cross-layer similarity search
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available")

try:
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")
    
    # Provide fallback implementations
    def normalize(X, axis=1, copy=True):
        """Fallback normalize without sklearn"""
        if not NUMPY_AVAILABLE:
            return X
        X = np.array(X)
        norms = np.linalg.norm(X, axis=axis, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return X / norms
    
    def cosine_similarity(X, Y=None):
        """Fallback cosine similarity without sklearn"""
        if not NUMPY_AVAILABLE:
            return np.array([[1.0]])
        X = np.array(X)
        if Y is None:
            Y = X
        else:
            Y = np.array(Y)
        
        # Normalize
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
        
        return np.dot(X_norm, Y_norm.T)
    
    PCA = None
    KMeans = None

from ..core import LayeredContextualGraph, Layer, Context, SuperpositionState, ContextType

logger = logging.getLogger(__name__)


@dataclass
class EntityEmbedding:
    """Embedding for an entity across layers"""
    entity_id: str
    layer_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # layer -> embedding
    aggregated_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def aggregate(self, method: str = "mean"):
        """Aggregate layer embeddings into single vector"""
        if not self.layer_embeddings:
            return None
        
        embeddings = list(self.layer_embeddings.values())
        
        if method == "mean":
            self.aggregated_embedding = np.mean(embeddings, axis=0)
        elif method == "max":
            self.aggregated_embedding = np.max(embeddings, axis=0)
        elif method == "concat":
            self.aggregated_embedding = np.concatenate(embeddings)
        
        return self.aggregated_embedding


class EmbeddingLayer:
    """
    Special layer that stores vector embeddings for entities.
    
    Enables semantic similarity search across layers.
    """
    
    def __init__(
        self,
        name: str,
        embedding_dim: int = 768,
        normalize_embeddings: bool = True
    ):
        self.name = name
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        # Storage
        self.embeddings: Dict[str, np.ndarray] = {}  # entity_id -> embedding
        self.entity_index: List[str] = []  # For efficient similarity search
        self.embedding_matrix: Optional[np.ndarray] = None
        
        logger.info(f"EmbeddingLayer initialized: {name} (dim={embedding_dim})")
    
    def add_embedding(self, entity_id: str, embedding: np.ndarray):
        """Add or update embedding for entity"""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[0]}")
        
        if self.normalize_embeddings:
            embedding = normalize(embedding.reshape(1, -1))[0]
        
        self.embeddings[entity_id] = embedding
        
        # Invalidate index
        self.embedding_matrix = None
    
    def build_index(self):
        """Build matrix for efficient similarity search"""
        if not self.embeddings:
            return
        
        self.entity_index = list(self.embeddings.keys())
        self.embedding_matrix = np.vstack([
            self.embeddings[eid] for eid in self.entity_index
        ])
        
        if self.normalize_embeddings:
            self.embedding_matrix = normalize(self.embedding_matrix)
        
        logger.info(f"Built embedding index: {len(self.entity_index)} entities")
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Find most similar entities"""
        if self.embedding_matrix is None:
            self.build_index()
        
        if self.embedding_matrix is None:
            return []
        
        if self.normalize_embeddings:
            query_embedding = normalize(query_embedding.reshape(1, -1))[0]
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embedding_matrix
        )[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.entity_index[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= min_similarity
        ]
        
        return results
    
    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding for entity"""
        return self.embeddings.get(entity_id)


class MLLayeredGraph(LayeredContextualGraph):
    """
    LayeredContextualGraph with ML capabilities.
    
    Features:
    - Embedding layers for semantic similarity
    - Predictive collapse using ML
    - Auto-context detection
    - Cross-layer similarity search
    """
    
    def __init__(
        self,
        name: str = "ml_layered_graph",
        quantum_enabled: bool = True,
        embedding_dim: int = 768,
        **kwargs
    ):
        super().__init__(name=name, quantum_enabled=quantum_enabled, **kwargs)
        
        self.embedding_dim = embedding_dim
        self.embedding_layers: Dict[str, EmbeddingLayer] = {}
        self.entity_embeddings: Dict[str, EntityEmbedding] = {}
        
        # ML models
        self.collapse_predictor = None  # Trained model for predicting collapses
        self.context_classifier = None  # Model for auto-context detection
        
        logger.info(f"MLLayeredGraph initialized: {name}")
    
    def add_embedding_layer(
        self,
        layer_name: str,
        embedding_dim: Optional[int] = None,
        normalize: bool = True
    ):
        """Add an embedding layer"""
        dim = embedding_dim or self.embedding_dim
        
        emb_layer = EmbeddingLayer(
            name=f"{layer_name}_embeddings",
            embedding_dim=dim,
            normalize_embeddings=normalize
        )
        
        self.embedding_layers[layer_name] = emb_layer
        logger.info(f"Added embedding layer for '{layer_name}'")
    
    def set_entity_embedding(
        self,
        entity_id: str,
        layer_name: str,
        embedding: np.ndarray
    ):
        """Set embedding for entity in specific layer"""
        if layer_name not in self.embedding_layers:
            self.add_embedding_layer(layer_name)
        
        self.embedding_layers[layer_name].add_embedding(entity_id, embedding)
        
        # Update entity embedding record
        if entity_id not in self.entity_embeddings:
            self.entity_embeddings[entity_id] = EntityEmbedding(entity_id=entity_id)
        
        self.entity_embeddings[entity_id].layer_embeddings[layer_name] = embedding
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        layer_name: Optional[str] = None,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search for similar entities across layers or in specific layer.
        
        Returns:
            Dict mapping layer_name to list of (entity_id, similarity) tuples
        """
        results = {}
        
        layers_to_search = [layer_name] if layer_name else self.embedding_layers.keys()
        
        for ln in layers_to_search:
            if ln in self.embedding_layers:
                layer_results = self.embedding_layers[ln].similarity_search(
                    query_embedding,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
                results[ln] = layer_results
        
        return results
    
    def cross_layer_similarity(
        self,
        entity1_id: str,
        entity2_id: str,
        aggregation: str = "mean"
    ) -> float:
        """
        Compute similarity between two entities across all layers.
        
        Args:
            entity1_id: First entity
            entity2_id: Second entity
            aggregation: How to aggregate layer similarities (mean, max, min)
            
        Returns:
            Aggregated similarity score
        """
        if entity1_id not in self.entity_embeddings or entity2_id not in self.entity_embeddings:
            return 0.0
        
        emb1 = self.entity_embeddings[entity1_id]
        emb2 = self.entity_embeddings[entity2_id]
        
        # Compute similarity per layer
        layer_similarities = []
        
        common_layers = set(emb1.layer_embeddings.keys()) & set(emb2.layer_embeddings.keys())
        
        for layer_name in common_layers:
            vec1 = emb1.layer_embeddings[layer_name]
            vec2 = emb2.layer_embeddings[layer_name]
            
            sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
            layer_similarities.append(float(sim))
        
        if not layer_similarities:
            return 0.0
        
        # Aggregate
        if aggregation == "mean":
            return np.mean(layer_similarities)
        elif aggregation == "max":
            return np.max(layer_similarities)
        elif aggregation == "min":
            return np.min(layer_similarities)
        else:
            return np.mean(layer_similarities)
    
    def predict_collapse(
        self,
        entity_id: str,
        context: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Predict which quantum state will collapse given context.
        
        Uses ML model trained on historical collapse patterns.
        
        Returns:
            (predicted_state, confidence)
        """
        if entity_id not in self.superposition_states:
            return None, 0.0
        
        superpos = self.superposition_states[entity_id]
        
        if not superpos.quantum_state or superpos.quantum_state.collapsed:
            return superpos.quantum_state.collapsed_state, 1.0
        
        # If no trained model, use quantum probabilities
        if self.collapse_predictor is None:
            return superpos.quantum_state.get_dominant_state()
        
        # TODO: Use trained model
        # For now, return dominant state
        return superpos.quantum_state.get_dominant_state()
    
    def auto_detect_context(
        self,
        entity_id: str,
        layer_name: Optional[str] = None
    ) -> List[str]:
        """
        Auto-detect applicable contexts for entity using ML.
        
        Analyzes entity's embeddings and suggests relevant contexts.
        
        Returns:
            List of context names
        """
        if entity_id not in self.entity_embeddings:
            return []
        
        # If no trained classifier, use simple heuristics
        if self.context_classifier is None:
            return self._heuristic_context_detection(entity_id, layer_name)
        
        # TODO: Use trained classifier
        return self._heuristic_context_detection(entity_id, layer_name)
    
    def _heuristic_context_detection(
        self,
        entity_id: str,
        layer_name: Optional[str]
    ) -> List[str]:
        """Simple heuristic context detection"""
        applicable_contexts = []
        
        entity_emb = self.entity_embeddings[entity_id]
        
        # Check which layers entity exists in
        entity_layers = set(entity_emb.layer_embeddings.keys())
        
        # Find contexts applicable to those layers
        for context_name, context in self.contexts.items():
            if not context.applicable_layers:
                applicable_contexts.append(context_name)
            elif entity_layers & context.applicable_layers:
                applicable_contexts.append(context_name)
        
        return applicable_contexts
    
    def cluster_entities(
        self,
        layer_name: str,
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[int, List[str]]:
        """
        Cluster entities in a layer based on embeddings.
        
        Returns:
            Dict mapping cluster_id to list of entity_ids
        """
        if layer_name not in self.embedding_layers:
            return {}
        
        emb_layer = self.embedding_layers[layer_name]
        
        if emb_layer.embedding_matrix is None:
            emb_layer.build_index()
        
        if emb_layer.embedding_matrix is None:
            return {}
        
        # Cluster
        if method == "kmeans" and SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(emb_layer.embedding_matrix)
            
            # Group by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(emb_layer.entity_index[idx])
            
            return clusters
        
        return {}
    
    def reduce_dimensionality(
        self,
        layer_name: str,
        n_components: int = 2,
        method: str = "pca"
    ) -> Dict[str, np.ndarray]:
        """
        Reduce embedding dimensionality for visualization.
        
        Returns:
            Dict mapping entity_id to reduced embedding
        """
        if layer_name not in self.embedding_layers:
            return {}
        
        emb_layer = self.embedding_layers[layer_name]
        
        if emb_layer.embedding_matrix is None:
            emb_layer.build_index()
        
        if emb_layer.embedding_matrix is None or not SKLEARN_AVAILABLE:
            return {}
        
        # Reduce
        if method == "pca":
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(emb_layer.embedding_matrix)
            
            return {
                emb_layer.entity_index[idx]: reduced[idx]
                for idx in range(len(emb_layer.entity_index))
            }
        
        return {}


def enable_ml(lcg: LayeredContextualGraph, embedding_dim: int = 768) -> MLLayeredGraph:
    """
    Convert existing LayeredContextualGraph to MLLayeredGraph.
    
    Note: This creates a new instance with ML capabilities.
    """
    ml_lcg = MLLayeredGraph(
        name=lcg.name,
        quantum_enabled=lcg.quantum_enabled,
        embedding_dim=embedding_dim
    )
    
    # Copy layers
    ml_lcg.layers = lcg.layers
    ml_lcg.layer_hierarchy = lcg.layer_hierarchy
    ml_lcg.contexts = lcg.contexts
    ml_lcg.superposition_states = lcg.superposition_states
    ml_lcg.quantum_states = lcg.quantum_states
    
    logger.info(f"Enabled ML for LCG: {lcg.name}")
    return ml_lcg
