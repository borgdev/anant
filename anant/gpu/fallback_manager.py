"""
Fallback Manager - CPU Implementation Fallbacks

Provides optimized CPU implementations for all accelerated operations,
ensuring the library works seamlessly when GPU is unavailable.
"""

import logging
import numpy as np
from typing import Union, Tuple, Any
from scipy import sparse
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class FallbackManager:
    """
    CPU fallback implementations for GPU-accelerated operations.
    
    This class provides optimized CPU implementations that maintain the same
    interface as GPU-accelerated operations, ensuring seamless fallback.
    """
    
    def __init__(self):
        self.logger = logger
        
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        CPU matrix multiplication using optimized BLAS.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Matrix multiplication result
        """
        return np.dot(a, b)
        
    def vector_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray,
                         metric: str = 'cosine') -> np.ndarray:
        """
        CPU vector similarity computation.
        
        Args:
            vectors1: First set of vectors (n x d)
            vectors2: Second set of vectors (m x d)
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity matrix (n x m)
        """
        if metric == 'cosine':
            # Normalize vectors
            v1_norm = vectors1 / (np.linalg.norm(vectors1, axis=1, keepdims=True) + 1e-8)
            v2_norm = vectors2 / (np.linalg.norm(vectors2, axis=1, keepdims=True) + 1e-8)
            return np.dot(v1_norm, v2_norm.T)
        elif metric == 'dot':
            return np.dot(vectors1, vectors2.T)
        elif metric == 'euclidean':
            # Compute pairwise euclidean distances
            distances = np.sqrt(
                np.sum((vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]) ** 2, axis=2)
            )
            # Convert to similarity (inverse distance)
            return 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
    def batch_embedding_lookup(self, embedding_matrix: np.ndarray, 
                              indices: np.ndarray) -> np.ndarray:
        """
        CPU batch embedding lookup.
        
        Args:
            embedding_matrix: Embedding matrix (vocab_size x embedding_dim)
            indices: Indices to lookup (batch_size,)
            
        Returns:
            Looked up embeddings (batch_size x embedding_dim)
        """
        return embedding_matrix[indices]
        
    def sparse_matrix_multiply(self, sparse_matrix: Any, dense_vector: np.ndarray) -> np.ndarray:
        """
        CPU sparse matrix-vector multiplication.
        
        Args:
            sparse_matrix: Sparse matrix (scipy.sparse format)
            dense_vector: Dense vector
            
        Returns:
            Result vector
        """
        if hasattr(sparse_matrix, 'dot'):
            return sparse_matrix.dot(dense_vector)
        elif sparse.issparse(sparse_matrix):
            return sparse_matrix.dot(dense_vector)
        else:
            # Fallback to dense multiplication
            dense_matrix = np.array(sparse_matrix.toarray()) if hasattr(sparse_matrix, 'toarray') else sparse_matrix
            return np.dot(dense_matrix, dense_vector)
            
    def batch_norm(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        CPU batch normalization.
        
        Args:
            data: Input data
            axis: Axis along which to normalize
            
        Returns:
            Normalized data
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + 1e-8)
        
    def kmeans_clustering(self, data: np.ndarray, k: int, max_iters: int = 100,
                         tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        CPU K-means clustering using scikit-learn.
        
        Args:
            data: Input data (n_samples x n_features)
            k: Number of clusters
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (cluster_centers, labels)
        """
        try:
            kmeans = KMeans(n_clusters=k, max_iter=max_iters, tol=tol, random_state=42)
            labels = kmeans.fit_predict(data)
            return kmeans.cluster_centers_, labels
        except Exception as e:
            self.logger.warning(f"Scikit-learn K-means failed: {e}, using manual implementation")
            return self._manual_kmeans(data, k, max_iters, tol)
            
    def _manual_kmeans(self, data: np.ndarray, k: int, max_iters: int = 100,
                      tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """Manual K-means implementation as fallback."""
        n_samples, n_features = data.shape
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(n_samples, k, replace=False)]
        labels = np.zeros(n_samples, dtype=int)  # Initialize labels
        
        for iteration in range(max_iters):
            # Compute distances to centroids
            distances = np.sqrt(np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
            
            # Assign points to closest centroids
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]
                    
            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < tol:
                break
                
            centroids = new_centroids
            
        return centroids, labels