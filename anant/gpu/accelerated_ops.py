"""
Accelerated Operations - GPU-Accelerated Implementations

Provides GPU-accelerated versions of common operations with automatic
fallback to CPU implementations when GPU is unavailable.
"""

import logging
import warnings
from typing import Optional, Union, List, Tuple, Any
import numpy as np

from .gpu_manager import get_gpu_manager, ComputeBackend
from .fallback_manager import FallbackManager

# Optional imports with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class AcceleratedOperations:
    """
    GPU-accelerated operations with automatic CPU fallbacks.
    
    This class provides a unified interface for common computational operations
    that can be accelerated on GPU when available, with seamless fallback to
    optimized CPU implementations.
    """
    
    def __init__(self, gpu_manager=None):
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.fallback_manager = FallbackManager()
        self._device_info = self.gpu_manager.get_device_info()
        
    def _get_backend(self) -> ComputeBackend:
        """Get the current compute backend."""
        return self._device_info.backend
        
    def _ensure_tensor(self, data: Union[np.ndarray, Any], device: Optional[str] = None) -> Any:
        """Convert input to appropriate tensor format for current backend."""
        backend = self._get_backend()
        
        if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
                if device is None:
                    device = f"cuda:{torch.cuda.current_device()}"
                return tensor.to(device)
            elif torch.is_tensor(data):
                if device is None:
                    device = f"cuda:{torch.cuda.current_device()}"
                return data.to(device)
            else:
                tensor = torch.tensor(data)
                if device is None:
                    device = f"cuda:{torch.cuda.current_device()}"
                return tensor.to(device)
        else:
            # CPU fallback
            if isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)
                
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if TORCH_AVAILABLE and torch.is_tensor(data):
            return data.cpu().numpy()
        elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        else:
            return np.asarray(data)
            
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray, 
                       return_numpy: bool = True) -> np.ndarray:
        """
        GPU-accelerated matrix multiplication with CPU fallback.
        
        Args:
            a: First matrix
            b: Second matrix
            return_numpy: Whether to return result as numpy array
            
        Returns:
            Matrix multiplication result
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_matrix_multiply(a, b, return_numpy)
            else:
                return self.fallback_manager.matrix_multiply(a, b)
        except Exception as e:
            logger.warning(f"GPU matrix multiplication failed: {e}, falling back to CPU")
            return self.fallback_manager.matrix_multiply(a, b)
            
    def _cuda_matrix_multiply(self, a: np.ndarray, b: np.ndarray, 
                             return_numpy: bool = True) -> np.ndarray:
        """CUDA implementation of matrix multiplication."""
        a_tensor = self._ensure_tensor(a)
        b_tensor = self._ensure_tensor(b)
        
        result = torch.mm(a_tensor, b_tensor)
        
        if return_numpy:
            return self._to_numpy(result)
        else:
            return result
            
    def vector_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray,
                         metric: str = 'cosine') -> np.ndarray:
        """
        GPU-accelerated vector similarity computation.
        
        Args:
            vectors1: First set of vectors (n x d)
            vectors2: Second set of vectors (m x d) 
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity matrix (n x m)
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_vector_similarity(vectors1, vectors2, metric)
            else:
                return self.fallback_manager.vector_similarity(vectors1, vectors2, metric)
        except Exception as e:
            logger.warning(f"GPU vector similarity failed: {e}, falling back to CPU")
            return self.fallback_manager.vector_similarity(vectors1, vectors2, metric)
            
    def _cuda_vector_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray,
                               metric: str = 'cosine') -> np.ndarray:
        """CUDA implementation of vector similarity."""
        v1 = self._ensure_tensor(vectors1)
        v2 = self._ensure_tensor(vectors2)
        
        if metric == 'cosine':
            # Normalize vectors
            v1_norm = torch.nn.functional.normalize(v1, p=2, dim=1)
            v2_norm = torch.nn.functional.normalize(v2, p=2, dim=1)
            similarity = torch.mm(v1_norm, v2_norm.t())
        elif metric == 'dot':
            similarity = torch.mm(v1, v2.t())
        elif metric == 'euclidean':
            # Compute pairwise euclidean distances
            v1_expanded = v1.unsqueeze(1)  # (n, 1, d)
            v2_expanded = v2.unsqueeze(0)  # (1, m, d)
            distances = torch.norm(v1_expanded - v2_expanded, dim=2)
            # Convert to similarity (inverse distance)
            similarity = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
        return self._to_numpy(similarity)
        
    def batch_embedding_lookup(self, embedding_matrix: np.ndarray, 
                              indices: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated batch embedding lookup.
        
        Args:
            embedding_matrix: Embedding matrix (vocab_size x embedding_dim)
            indices: Indices to lookup (batch_size,)
            
        Returns:
            Looked up embeddings (batch_size x embedding_dim)
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_embedding_lookup(embedding_matrix, indices)
            else:
                return self.fallback_manager.batch_embedding_lookup(embedding_matrix, indices)
        except Exception as e:
            logger.warning(f"GPU embedding lookup failed: {e}, falling back to CPU")
            return self.fallback_manager.batch_embedding_lookup(embedding_matrix, indices)
            
    def _cuda_embedding_lookup(self, embedding_matrix: np.ndarray, 
                              indices: np.ndarray) -> np.ndarray:
        """CUDA implementation of embedding lookup."""
        embeddings = self._ensure_tensor(embedding_matrix)
        idx = self._ensure_tensor(indices, device=embeddings.device).long()
        
        result = torch.index_select(embeddings, 0, idx)
        return self._to_numpy(result)
        
    def sparse_matrix_multiply(self, sparse_matrix: Any, dense_vector: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated sparse matrix-vector multiplication.
        
        Args:
            sparse_matrix: Sparse matrix (scipy.sparse or torch.sparse)
            dense_vector: Dense vector
            
        Returns:
            Result vector
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_sparse_multiply(sparse_matrix, dense_vector)
            else:
                return self.fallback_manager.sparse_matrix_multiply(sparse_matrix, dense_vector)
        except Exception as e:
            logger.warning(f"GPU sparse multiplication failed: {e}, falling back to CPU")
            return self.fallback_manager.sparse_matrix_multiply(sparse_matrix, dense_vector)
            
    def _cuda_sparse_multiply(self, sparse_matrix: Any, dense_vector: np.ndarray) -> np.ndarray:
        """CUDA implementation of sparse matrix multiplication."""
        # Convert to torch sparse if necessary
        if hasattr(sparse_matrix, 'tocoo'):  # scipy sparse
            coo = sparse_matrix.tocoo()
            indices = torch.stack([
                torch.from_numpy(coo.row).long(),
                torch.from_numpy(coo.col).long()
            ])
            values = torch.from_numpy(coo.data).float()
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, coo.shape, dtype=torch.float32
            ).cuda()
        else:
            sparse_tensor = sparse_matrix
            
        vector = self._ensure_tensor(dense_vector).float()
        result = torch.sparse.mm(sparse_tensor, vector.unsqueeze(1)).squeeze(1)
        
        return self._to_numpy(result)
        
    def batch_norm(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        GPU-accelerated batch normalization.
        
        Args:
            data: Input data
            axis: Axis along which to normalize
            
        Returns:
            Normalized data
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_batch_norm(data, axis)
            else:
                return self.fallback_manager.batch_norm(data, axis)
        except Exception as e:
            logger.warning(f"GPU batch norm failed: {e}, falling back to CPU")
            return self.fallback_manager.batch_norm(data, axis)
            
    def _cuda_batch_norm(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """CUDA implementation of batch normalization."""
        tensor = self._ensure_tensor(data)
        
        # Calculate mean and std along specified axis
        mean = torch.mean(tensor, dim=axis, keepdim=True)
        std = torch.std(tensor, dim=axis, keepdim=True)
        
        # Normalize
        normalized = (tensor - mean) / (std + 1e-8)
        
        return self._to_numpy(normalized)
        
    def kmeans_clustering(self, data: np.ndarray, k: int, max_iters: int = 100,
                         tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated K-means clustering.
        
        Args:
            data: Input data (n_samples x n_features)
            k: Number of clusters
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (cluster_centers, labels)
        """
        backend = self._get_backend()
        
        try:
            if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
                return self._cuda_kmeans(data, k, max_iters, tol)
            else:
                return self.fallback_manager.kmeans_clustering(data, k, max_iters, tol)
        except Exception as e:
            logger.warning(f"GPU K-means failed: {e}, falling back to CPU")
            return self.fallback_manager.kmeans_clustering(data, k, max_iters, tol)
            
    def _cuda_kmeans(self, data: np.ndarray, k: int, max_iters: int = 100,
                    tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """CUDA implementation of K-means clustering."""
        X = self._ensure_tensor(data).float()
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        centroids = X[torch.randperm(n_samples)[:k]]
        
        for iteration in range(max_iters):
            # Compute distances to centroids
            distances = torch.cdist(X, centroids)
            
            # Assign points to closest centroids
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = X[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]
                    
            # Check convergence
            centroid_shift = torch.norm(new_centroids - centroids)
            if centroid_shift < tol:
                break
                
            centroids = new_centroids
            
        return self._to_numpy(centroids), self._to_numpy(labels)
        
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        backend = self._get_backend()
        
        if backend == ComputeBackend.CUDA and TORCH_AVAILABLE:
            try:
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                return {
                    "allocated": allocated,
                    "cached": cached,
                    "free": cached - allocated
                }
            except Exception:
                return {"error": "Could not get CUDA memory info"}
        else:
            return {"backend": backend.value, "gpu_memory": "not_applicable"}
            
    def clear_memory_cache(self):
        """Clear GPU memory cache."""
        backend = self._get_backend()
        
        if backend == ComputeBackend.CUDA:
            self.gpu_manager.clear_cache()
            logger.info("GPU memory cache cleared")
        else:
            logger.info("No GPU memory cache to clear")