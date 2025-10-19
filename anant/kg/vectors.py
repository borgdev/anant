"""
Vector Operations Engine for Knowledge Graphs
============================================

High-performance vector similarity search and operations using FAISS and
other optimized libraries for knowledge graph embeddings.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import pickle

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies
faiss = safe_import('faiss')
sklearn = safe_import('sklearn')

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations"""
    index_type: str = 'HNSW'  # 'HNSW', 'IVF', 'Flat', 'LSH'
    distance_metric: str = 'cosine'  # 'cosine', 'l2', 'ip' (inner product)
    nlist: int = 100  # Number of clusters for IVF
    m: int = 16  # Number of connections for HNSW
    ef_construction: int = 200  # Construction parameter for HNSW
    ef_search: int = 50  # Search parameter for HNSW
    use_gpu: bool = False  # Use GPU acceleration if available
    normalize: bool = True  # Normalize vectors
    

@dataclass
class SearchResult:
    """Result of vector similarity search"""
    entity_id: str
    similarity: float
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class VectorEngine:
    """
    High-Performance Vector Operations Engine
    
    Provides fast similarity search, clustering, and vector operations
    for knowledge graph embeddings using FAISS and other optimized libraries.
    """
    
    SUPPORTED_INDEXES = {
        'Flat': 'Brute-force exact search',
        'HNSW': 'Hierarchical Navigable Small World graphs',
        'IVF': 'Inverted File with quantization',
        'LSH': 'Locality Sensitive Hashing',
        'PQ': 'Product Quantization',
        'IVFPQ': 'IVF with Product Quantization'
    }
    
    def __init__(self, config: Optional[VectorSearchConfig] = None):
        """
        Initialize vector engine
        
        Args:
            config: Vector search configuration
        """
        self.config = config or VectorSearchConfig()
        
        # Vector storage
        self.vectors = None
        self.entity_ids = []
        self.metadata = {}
        self.dimension = None
        
        # Search indexes
        self.index = None
        self.trained = False
        
        # GPU support
        self.gpu_resources = None
        self._setup_gpu()
        
        logger.info(f"Vector Engine initialized with {self.config.index_type} index")
    
    def _setup_gpu(self):
        """Setup GPU resources if available"""
        if not faiss or not self.config.use_gpu:
            return
        
        try:
            # Check if GPU is available
            if faiss.get_num_gpus() > 0:
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info(f"GPU acceleration enabled with {faiss.get_num_gpus()} GPUs")
            else:
                logger.info("GPU requested but not available, falling back to CPU")
                self.config.use_gpu = False
        except Exception as e:
            logger.warning(f"Failed to initialize GPU: {e}")
            self.config.use_gpu = False
    
    @performance_monitor("vector_index_creation")
    def build_index(self, 
                   embeddings: Dict[str, np.ndarray],
                   metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Build vector index from embeddings
        
        Args:
            embeddings: Dictionary mapping entity IDs to embedding vectors
            metadata: Optional metadata for each entity
        """
        
        logger.info(f"Building {self.config.index_type} index for {len(embeddings)} vectors...")
        
        with PerformanceProfiler("vector_index_build") as profiler:
            
            profiler.checkpoint("data_preparation")
            
            # Prepare data
            self.entity_ids = list(embeddings.keys())
            vectors = np.array([embeddings[entity_id] for entity_id in self.entity_ids])
            
            self.dimension = vectors.shape[1]
            self.metadata = metadata or {}
            
            # Normalize vectors if requested
            if self.config.normalize:
                vectors = self._normalize_vectors(vectors)
            
            self.vectors = vectors
            
            profiler.checkpoint("index_creation")
            
            # Create and train index
            self._create_index()
            
            profiler.checkpoint("index_training")
            
            # Train index
            if hasattr(self.index, 'train') and not self.trained and not isinstance(self.index, FallbackIndex):
                self.index.train(vectors)
                self.trained = True
            
            profiler.checkpoint("index_population")
            
            # Add vectors to index
            if self.index is not None:
                self.index.add(vectors)
            
            profiler.checkpoint("build_complete")
        
        build_time = profiler.get_report()['total_execution_time']
        logger.info(f"Index built in {build_time:.2f}s for {len(embeddings)} vectors")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _create_index(self):
        """Create appropriate FAISS index"""
        
        if not faiss or self.dimension is None:
            logger.warning("FAISS not available or dimension not set, using fallback implementation")
            self.index = FallbackIndex(self.dimension or 128)
            return
        
        # Map distance metrics
        metric_map = {
            'cosine': faiss.METRIC_INNER_PRODUCT,  # For normalized vectors
            'l2': faiss.METRIC_L2,
            'ip': faiss.METRIC_INNER_PRODUCT
        }
        metric = metric_map.get(self.config.distance_metric, faiss.METRIC_L2)
        
        # Create index based on type
        if self.config.index_type == 'Flat':
            index = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
        
        elif self.config.index_type == 'HNSW':
            index = faiss.IndexHNSWFlat(self.dimension, self.config.m)
            index.hnsw.efConstruction = self.config.ef_construction
            
        elif self.config.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config.nlist, metric)
            
        elif self.config.index_type == 'IVFPQ':
            quantizer = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            m = min((self.dimension or 128) // 4, 64)  # Number of subvectors
            index = faiss.IndexIVFPQ(quantizer, self.dimension, self.config.nlist, m, 8)
            
        elif self.config.index_type == 'LSH':
            # Use random projection for LSH
            index = faiss.IndexLSH(self.dimension, min((self.dimension or 128) * 2, 1024))
            
        else:
            logger.warning(f"Unknown index type {self.config.index_type}, using Flat")
            index = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
        
        # Move to GPU if requested
        if self.config.use_gpu and self.gpu_resources:
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        
        self.index = index
    
    @performance_monitor("vector_search")
    def search(self, 
              query_vector: np.ndarray,
              k: int = 10,
              ef_search: Optional[int] = None) -> List[SearchResult]:
        """
        Search for k most similar vectors
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            ef_search: Search parameter for HNSW (overrides config)
            
        Returns:
            List of SearchResult objects
        """
        
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Normalize query vector if needed
        if self.config.normalize:
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1))[0]
        
        # Set search parameters for HNSW
        if (self.config.index_type == 'HNSW' and hasattr(self.index, 'hnsw') 
            and not isinstance(self.index, FallbackIndex)):
            ef = ef_search or self.config.ef_search
            self.index.hnsw.efSearch = ef
        
        # Perform search
        if isinstance(self.index, FallbackIndex):
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        else:
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Convert to SearchResult objects
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.entity_ids):  # Valid index
                entity_id = self.entity_ids[idx]
                
                # Convert distance to similarity
                if self.config.distance_metric == 'cosine' or self.config.distance_metric == 'ip':
                    similarity = float(distance)  # Already similarity for inner product
                else:
                    similarity = 1.0 / (1.0 + float(distance))  # Convert L2 distance to similarity
                
                metadata = self.metadata.get(entity_id, {})
                
                results.append(SearchResult(
                    entity_id=entity_id,
                    similarity=similarity,
                    distance=float(distance),
                    metadata=metadata
                ))
        
        return results
    
    @performance_monitor("batch_search")
    def batch_search(self, 
                    query_vectors: np.ndarray,
                    k: int = 10) -> List[List[SearchResult]]:
        """
        Batch search for multiple query vectors
        
        Args:
            query_vectors: Array of query vectors (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Normalize query vectors if needed
        if self.config.normalize:
            query_vectors = self._normalize_vectors(query_vectors)
        
        # Perform batch search
        if isinstance(self.index, FallbackIndex):
            distances, indices = self.index.search(query_vectors, k)
        else:
            distances, indices = self.index.search(query_vectors, k)
        
        # Convert to SearchResult objects
        all_results = []
        for query_idx in range(len(query_vectors)):
            query_results = []
            for i, (distance, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                if idx >= 0 and idx < len(self.entity_ids):
                    entity_id = self.entity_ids[idx]
                    
                    if self.config.distance_metric == 'cosine' or self.config.distance_metric == 'ip':
                        similarity = float(distance)
                    else:
                        similarity = 1.0 / (1.0 + float(distance))
                    
                    metadata = self.metadata.get(entity_id, {})
                    
                    query_results.append(SearchResult(
                        entity_id=entity_id,
                        similarity=similarity,
                        distance=float(distance),
                        metadata=metadata
                    ))
            
            all_results.append(query_results)
        
        return all_results
    
    @performance_monitor("vector_clustering")
    def cluster_vectors(self, 
                       n_clusters: int,
                       algorithm: str = 'kmeans') -> Dict[str, int]:
        """
        Cluster vectors using specified algorithm
        
        Args:
            n_clusters: Number of clusters
            algorithm: Clustering algorithm ('kmeans', 'spectral', 'dbscan')
            
        Returns:
            Dictionary mapping entity IDs to cluster assignments
        """
        
        if self.vectors is None:
            raise ValueError("No vectors available. Call build_index() first.")
        
        logger.info(f"Clustering {len(self.vectors)} vectors into {n_clusters} clusters using {algorithm}")
        
        if algorithm == 'kmeans':
            if faiss:
                # Use FAISS k-means for large-scale clustering
                kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=True)
                kmeans.train(self.vectors)
                
                # Assign vectors to clusters
                _, assignments = kmeans.index.search(self.vectors, 1)
                cluster_assignments = assignments.flatten()
            elif sklearn:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_assignments = kmeans.fit_predict(self.vectors)
            else:
                raise ValueError("Neither FAISS nor sklearn available for clustering")
        
        elif algorithm == 'spectral' and sklearn:
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
            cluster_assignments = spectral.fit_predict(self.vectors)
        
        elif algorithm == 'dbscan' and sklearn:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_assignments = dbscan.fit_predict(self.vectors)
        
        else:
            raise ValueError(f"Clustering algorithm {algorithm} not available or not supported")
        
        # Map to entity IDs
        clusters = {}
        for entity_id, cluster_id in zip(self.entity_ids, cluster_assignments):
            clusters[entity_id] = int(cluster_id)
        
        logger.info(f"Clustering completed: {len(set(cluster_assignments))} clusters created")
        
        return clusters
    
    def save_index(self, filepath: str):
        """Save index to file"""
        
        if not faiss or isinstance(self.index, FallbackIndex):
            # Save fallback data
            data = {
                'vectors': self.vectors,
                'entity_ids': self.entity_ids,
                'metadata': self.metadata,
                'config': self.config.__dict__,
                'dimension': self.dimension
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Save FAISS index
            faiss.write_index(self.index, filepath + '.faiss')
            
            # Save metadata
            metadata = {
                'entity_ids': self.entity_ids,
                'metadata': self.metadata,
                'config': self.config.__dict__,
                'dimension': self.dimension,
                'trained': self.trained
            }
            with open(filepath + '.meta', 'wb') as f:
                pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load index from file"""
        
        try:
            if faiss:
                # Try to load FAISS index
                self.index = faiss.read_index(filepath + '.faiss')
                
                # Load metadata
                with open(filepath + '.meta', 'rb') as f:
                    metadata = pickle.load(f)
                
                self.entity_ids = metadata['entity_ids']
                self.metadata = metadata['metadata']
                self.dimension = metadata['dimension']
                self.trained = metadata.get('trained', True)
                
                # Restore config
                for key, value in metadata['config'].items():
                    setattr(self.config, key, value)
            else:
                # Load fallback data
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.vectors = data['vectors']
                self.entity_ids = data['entity_ids']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                
                # Restore config
                for key, value in data['config'].items():
                    setattr(self.config, key, value)
                
                # Create fallback index
                self.index = FallbackIndex(self.dimension)
                self.index.add(self.vectors)
            
            logger.info(f"Index loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        
        stats = {
            'num_vectors': len(self.entity_ids) if self.entity_ids else 0,
            'dimension': self.dimension,
            'index_type': self.config.index_type,
            'distance_metric': self.config.distance_metric,
            'trained': self.trained,
            'gpu_enabled': self.config.use_gpu
        }
        
        if faiss and self.index and not isinstance(self.index, FallbackIndex):
            stats['faiss_index_size'] = self.index.ntotal
            if hasattr(self.index, 'metric_type'):
                stats['faiss_metric'] = self.index.metric_type
        
        return stats


class FallbackIndex:
    """Fallback vector index when FAISS is not available"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None
        self.ntotal = 0
    
    def add(self, vectors: np.ndarray):
        """Add vectors to index"""
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.ntotal = len(self.vectors)
    
    def search(self, query_vectors: np.ndarray, k: int):
        """Search for k nearest neighbors"""
        if self.vectors is None:
            return np.array([]), np.array([])
        
        # Compute cosine similarities (assuming normalized vectors)
        similarities = np.dot(query_vectors, self.vectors.T)
        
        # Get top k for each query
        distances = []
        indices = []
        
        for i in range(len(query_vectors)):
            query_similarities = similarities[i]
            # Get indices of top k similarities
            top_k_indices = np.argpartition(-query_similarities, min(k, len(query_similarities)-1))[:k]
            # Sort by similarity (descending)
            sorted_indices = top_k_indices[np.argsort(-query_similarities[top_k_indices])]
            
            distances.append(query_similarities[sorted_indices])
            indices.append(sorted_indices)
        
        return np.array(distances), np.array(indices)