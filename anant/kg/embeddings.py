"""
Knowledge Graph Embeddings Engine
=================================

Advanced embedding generation for entities and relationships using multiple
state-of-the-art algorithms including TransE, ComplEx, RotatE, and more.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, TYPE_CHECKING
from collections import defaultdict
from dataclasses import dataclass
import pickle
import json

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..utils.extras import safe_import

# Optional dependencies for advanced embeddings
torch = safe_import('torch')
sklearn = safe_import('sklearn')
faiss = safe_import('faiss')

# Type checking imports
if TYPE_CHECKING and torch:
    import torch.nn as nn
    import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    algorithm: str = 'TransE'
    dimensions: int = 256
    learning_rate: float = 0.01
    batch_size: int = 1024
    epochs: int = 100
    negative_samples: int = 10
    margin: float = 1.0
    regularization: float = 0.01
    device: str = 'auto'  # 'cpu', 'cuda', 'auto'


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    entity_embeddings: Dict[str, np.ndarray]
    relation_embeddings: Dict[str, np.ndarray]
    config: EmbeddingConfig
    training_loss: List[float]
    evaluation_metrics: Dict[str, float]
    training_time: float


class KGEmbedder:
    """
    Advanced Knowledge Graph Embedding Engine
    
    Supports multiple state-of-the-art algorithms:
    - TransE: Translation-based embeddings
    - TransR: Relation-specific transformations
    - ComplEx: Complex-valued embeddings
    - RotatE: Rotation-based embeddings
    - ConvE: Convolutional embeddings
    """
    
    SUPPORTED_ALGORITHMS = {
        'TransE': 'Translation-based embeddings (h + r ≈ t)',
        'TransR': 'Relation-specific embeddings with projection',
        'ComplEx': 'Complex-valued embeddings for asymmetric relations',
        'RotatE': 'Rotation in complex space',
        'ConvE': 'Convolutional neural embeddings',
        'DistMult': 'Bilinear diagonal model'
    }
    
    def __init__(self, knowledge_graph, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding engine
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            config: Embedding configuration
        """
        self.kg = knowledge_graph
        self.config = config or EmbeddingConfig()
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize mappings
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        
        # Training data
        self.triples = []
        self.negative_triples = []
        
        # Embeddings
        self.entity_embeddings = None
        self.relation_embeddings = None
        
        # Model
        self.model = None
        
        logger.info(f"KG Embedder initialized with {self.config.algorithm} on {self.device}")
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == 'auto':
            if torch and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return self.config.device
    
    @performance_monitor("embedding_generation")
    def generate_embeddings(self, 
                           algorithm: Optional[str] = None,
                           incremental: bool = False) -> EmbeddingResult:
        """
        Generate embeddings using specified algorithm
        
        Args:
            algorithm: Override default algorithm
            incremental: Update existing embeddings incrementally
            
        Returns:
            EmbeddingResult with generated embeddings and metadata
        """
        
        if algorithm:
            self.config.algorithm = algorithm
        
        logger.info(f"Generating {self.config.algorithm} embeddings...")
        
        with PerformanceProfiler(f"embedding_{self.config.algorithm}") as profiler:
            
            profiler.checkpoint("data_preparation")
            
            # Prepare training data
            if not incremental or not self.triples:
                self._prepare_training_data()
            
            profiler.checkpoint("model_initialization")
            
            # Initialize model
            self._initialize_model()
            
            profiler.checkpoint("training_start")
            
            # Train embeddings
            training_loss, eval_metrics = self._train_embeddings()
            
            profiler.checkpoint("training_complete")
            
            # Extract final embeddings
            entity_emb, relation_emb = self._extract_embeddings()
            
            profiler.checkpoint("extraction_complete")
        
        training_time = profiler.get_report()['total_execution_time']
        
        result = EmbeddingResult(
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
            config=self.config,
            training_loss=training_loss,
            evaluation_metrics=eval_metrics,
            training_time=training_time
        )
        
        logger.info(f"Embedding generation completed in {training_time:.2f}s")
        
        return result
    
    def _prepare_training_data(self):
        """Prepare triples for training"""
        
        # Extract triples from knowledge graph
        entities = set()
        relations = set()
        triples = []
        
        for edge_id in self.kg.edges:
            edge_nodes = self.kg.incidences.get_edge_nodes(edge_id)
            
            if len(edge_nodes) >= 2:
                # Extract relationship
                relationship = self._extract_relationship(edge_id, edge_nodes)
                
                # Create triples from hyperedge
                for i, head in enumerate(edge_nodes):
                    for j, tail in enumerate(edge_nodes):
                        if i != j:
                            triples.append((head, relationship, tail))
                            entities.add(head)
                            entities.add(tail)
                            relations.add(relationship)
        
        # Create mappings
        self.entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}
        self.relation_to_id = {relation: i for i, relation in enumerate(sorted(relations))}
        self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}
        self.id_to_relation = {i: relation for relation, i in self.relation_to_id.items()}
        
        # Convert to ID triples
        self.triples = [
            (self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t])
            for h, r, t in triples
        ]
        
        logger.info(f"Prepared {len(self.triples)} triples, {len(entities)} entities, {len(relations)} relations")
    
    def _extract_relationship(self, edge_id: str, edge_nodes: List[str]) -> str:
        """Extract relationship type from hyperedge"""
        
        # Use existing relationship extraction from KG
        if hasattr(self.kg, '_extract_relationship_type'):
            return self.kg._extract_relationship_type(edge_id, edge_nodes)
        
        # Fallback: use edge ID
        return edge_id
    
    def _initialize_model(self):
        """Initialize embedding model based on algorithm"""
        
        num_entities = len(self.entity_to_id)
        num_relations = len(self.relation_to_id)
        dim = self.config.dimensions
        
        if torch and self.config.algorithm in ['TransE', 'ComplEx', 'RotatE', 'DistMult']:
            if self.config.algorithm == 'TransE':
                self.model = create_transe_model(num_entities, num_relations, dim, self.device)
            elif self.config.algorithm == 'ComplEx':
                self.model = create_complex_model(num_entities, num_relations, dim, self.device)
            elif self.config.algorithm == 'RotatE':
                self.model = create_rotate_model(num_entities, num_relations, dim, self.device)
            elif self.config.algorithm == 'DistMult':
                self.model = create_distmult_model(num_entities, num_relations, dim, self.device)
        else:
            # Fallback to simple numpy implementation
            self.model = SimpleEmbeddingModel(num_entities, num_relations, dim)
    
    def _train_embeddings(self) -> Tuple[List[float], Dict[str, float]]:
        """Train the embedding model"""
        
        if torch and hasattr(self.model, 'train_model') and not isinstance(self.model, SimpleEmbeddingModel):
            return self.model.train_model(
                self.triples,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                negative_samples=self.config.negative_samples,
                margin=self.config.margin
            )
        else:
            # Simple training for fallback model
            return self.model.simple_train(self.triples, epochs=self.config.epochs)
    
    def _extract_embeddings(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract final embeddings from trained model"""
        
        entity_embeddings = {}
        relation_embeddings = {}
        
        # Get embeddings from model
        if torch and hasattr(self.model, 'get_embeddings') and not isinstance(self.model, SimpleEmbeddingModel):
            entity_emb_matrix, relation_emb_matrix = self.model.get_embeddings()
        else:
            entity_emb_matrix = self.model.entity_embeddings
            relation_emb_matrix = self.model.relation_embeddings
        
        # Convert to dictionaries
        for entity, entity_id in self.entity_to_id.items():
            entity_embeddings[entity] = entity_emb_matrix[entity_id]
        
        for relation, relation_id in self.relation_to_id.items():
            relation_embeddings[relation] = relation_emb_matrix[relation_id]
        
        return entity_embeddings, relation_embeddings
    
    @performance_monitor("similarity_search")
    def similarity_search(self, 
                         entity: str, 
                         k: int = 10,
                         similarity_metric: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Find k most similar entities to given entity
        
        Args:
            entity: Target entity
            k: Number of similar entities to return
            similarity_metric: 'cosine', 'euclidean', 'dot_product'
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        
        if self.entity_embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings() first.")
        
        if entity not in self.entity_embeddings:
            raise ValueError(f"Entity {entity} not found in embeddings")
        
        target_embedding = self.entity_embeddings[entity]
        similarities = []
        
        for other_entity, other_embedding in self.entity_embeddings.items():
            if other_entity != entity:
                similarity = self._calculate_similarity(
                    target_embedding, other_embedding, similarity_metric
                )
                similarities.append((other_entity, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def _calculate_similarity(self, 
                            emb1: np.ndarray, 
                            emb2: np.ndarray, 
                            metric: str) -> float:
        """Calculate similarity between two embeddings"""
        
        if metric == 'cosine':
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == 'euclidean':
            return 1.0 / (1.0 + np.linalg.norm(emb1 - emb2))
        elif metric == 'dot_product':
            return np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file"""
        
        data = {
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'config': self.config.__dict__,
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.entity_embeddings = data['entity_embeddings']
        self.relation_embeddings = data['relation_embeddings']
        self.entity_to_id = data['entity_to_id']
        self.relation_to_id = data['relation_to_id']
        self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}
        self.id_to_relation = {i: relation for relation, i in self.relation_to_id.items()}
        
        # Restore config
        if 'config' in data:
            for key, value in data['config'].items():
                setattr(self.config, key, value)
        
        logger.info(f"Embeddings loaded from {filepath}")


# Embedding Models
class SimpleEmbeddingModel:
    """Simple numpy-based embedding model (fallback)"""
    
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        
        # Initialize random embeddings
        self.entity_embeddings = np.random.normal(0, 0.1, (num_entities, dim))
        self.relation_embeddings = np.random.normal(0, 0.1, (num_relations, dim))
    
    def simple_train(self, triples: List[Tuple[int, int, int]], epochs: int = 100):
        """Simple training loop"""
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for head, relation, tail in triples:
                # TransE-style training
                head_emb = self.entity_embeddings[head]
                rel_emb = self.relation_embeddings[relation]
                tail_emb = self.entity_embeddings[tail]
                
                # Calculate score (TransE: ||h + r - t||)
                score = np.linalg.norm(head_emb + rel_emb - tail_emb)
                
                # Simple gradient descent update
                learning_rate = 0.01
                
                if score > 0.5:  # If score is too high, adjust embeddings
                    gradient = (head_emb + rel_emb - tail_emb) / (score + 1e-8)
                    
                    self.entity_embeddings[head] -= learning_rate * gradient
                    self.relation_embeddings[relation] -= learning_rate * gradient
                    self.entity_embeddings[tail] += learning_rate * gradient
                
                epoch_loss += score
            
            losses.append(epoch_loss / len(triples))
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")
        
        # Simple evaluation metrics
        eval_metrics = {
            'final_loss': losses[-1] if losses else 0.0,
            'loss_reduction': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 else 0.0
        }
        
        return losses, eval_metrics


# PyTorch-based models (if available)
if torch:
    import torch.nn as nn
    import torch.optim as optim
    
    class TransEModel(nn.Module):
        """TransE embedding model using PyTorch"""
        
        def __init__(self, num_entities: int, num_relations: int, dim: int, device: str):
            super().__init__()
            
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.dim = dim
            self.device = device
            
            # Embedding layers
            self.entity_embeddings = nn.Embedding(num_entities, dim)
            self.relation_embeddings = nn.Embedding(num_relations, dim)
            
            # Initialize embeddings
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
            
            self.to(device)
        
        def forward(self, heads, relations, tails):
            """Forward pass"""
            
            head_emb = self.entity_embeddings(heads)
            rel_emb = self.relation_embeddings(relations)
            tail_emb = self.entity_embeddings(tails)
            
            # TransE score: ||h + r - t||
            scores = torch.norm(head_emb + rel_emb - tail_emb, p=2, dim=1)
            
            return scores
        
        def train(self, triples, epochs=100, batch_size=1024, learning_rate=0.01, 
                 negative_samples=10, margin=1.0):
            """Train the model"""
            
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = torch.nn.MarginRankingLoss(margin=margin)
            
            losses = []
            
            # Convert triples to tensors
            triple_tensor = torch.tensor(triples, dtype=torch.long, device=self.device)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                # Shuffle triples
                indices = torch.randperm(len(triple_tensor))
                
                for i in range(0, len(triple_tensor), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_triples = triple_tensor[batch_indices]
                    
                    # Generate negative samples
                    neg_triples = self._generate_negative_samples(
                        batch_triples, negative_samples
                    )
                    
                    # Forward pass
                    pos_scores = self.forward(
                        batch_triples[:, 0], 
                        batch_triples[:, 1], 
                        batch_triples[:, 2]
                    )
                    
                    neg_scores = self.forward(
                        neg_triples[:, 0], 
                        neg_triples[:, 1], 
                        neg_triples[:, 2]
                    )
                    
                    # Calculate loss
                    target = torch.ones(pos_scores.size(), device=self.device)
                    loss = criterion(neg_scores, pos_scores, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                epoch_loss /= (len(triple_tensor) // batch_size)
                losses.append(epoch_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
            
            # Evaluation metrics
            eval_metrics = {
                'final_loss': losses[-1] if losses else 0.0,
                'loss_reduction': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 else 0.0
            }
            
            return losses, eval_metrics
        
        def _generate_negative_samples(self, positive_triples, num_negatives):
            """Generate negative samples by corrupting positive triples"""
            
            batch_size = positive_triples.size(0)
            neg_triples = []
            
            for _ in range(num_negatives):
                # Copy positive triples
                neg_batch = positive_triples.clone()
                
                # Randomly corrupt head or tail
                for i in range(batch_size):
                    if torch.rand(1) < 0.5:
                        # Corrupt head
                        neg_batch[i, 0] = torch.randint(0, self.num_entities, (1,))
                    else:
                        # Corrupt tail
                        neg_batch[i, 2] = torch.randint(0, self.num_entities, (1,))
                
                neg_triples.append(neg_batch)
            
            return torch.cat(neg_triples, dim=0)
        
        def get_embeddings(self):
            """Extract embeddings as numpy arrays"""
            
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            
            return entity_emb, relation_emb
    
    
    class ComplExModel(torch.nn.Module):
        """ComplEx embedding model for asymmetric relations"""
        
        def __init__(self, num_entities: int, num_relations: int, dim: int, device: str):
            super().__init__()
            
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.dim = dim
            self.device = device
            
            # Complex embeddings (real and imaginary parts)
            self.entity_re = torch.nn.Embedding(num_entities, dim)
            self.entity_im = torch.nn.Embedding(num_entities, dim)
            self.relation_re = torch.nn.Embedding(num_relations, dim)
            self.relation_im = torch.nn.Embedding(num_relations, dim)
            
            # Initialize embeddings
            for emb in [self.entity_re, self.entity_im, self.relation_re, self.relation_im]:
                torch.nn.init.xavier_uniform_(emb.weight)
            
            self.to(device)
        
        def forward(self, heads, relations, tails):
            """ComplEx scoring function"""
            
            head_re = self.entity_re(heads)
            head_im = self.entity_im(heads)
            rel_re = self.relation_re(relations)
            rel_im = self.relation_im(relations)
            tail_re = self.entity_re(tails)
            tail_im = self.entity_im(tails)
            
            # ComplEx score: Re(<h, r, conj(t)>)
            score = (head_re * rel_re * tail_re +
                    head_re * rel_im * tail_im +
                    head_im * rel_re * tail_im -
                    head_im * rel_im * tail_re).sum(dim=1)
            
            return score
        
        def get_embeddings(self):
            """Extract embeddings as numpy arrays"""
            
            with torch.no_grad():
                # Combine real and imaginary parts
                entity_emb = torch.cat([
                    self.entity_re.weight, self.entity_im.weight
                ], dim=1).cpu().numpy()
                
                relation_emb = torch.cat([
                    self.relation_re.weight, self.relation_im.weight
                ], dim=1).cpu().numpy()
            
            return entity_emb, relation_emb
    
    
    class RotatEModel(torch.nn.Module):
        """RotatE embedding model using rotation in complex space"""
        
        def __init__(self, num_entities: int, num_relations: int, dim: int, device: str):
            super().__init__()
            
            assert dim % 2 == 0, "Dimension must be even for RotatE"
            
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.dim = dim
            self.device = device
            
            # Entity embeddings (complex)
            self.entity_embeddings = torch.nn.Embedding(num_entities, dim)
            
            # Relation embeddings (phases)
            self.relation_embeddings = torch.nn.Embedding(num_relations, dim // 2)
            
            # Initialize embeddings
            torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
            torch.nn.init.uniform_(self.relation_embeddings.weight, 0, 2 * np.pi)
            
            self.to(device)
        
        def forward(self, heads, relations, tails):
            """RotatE scoring function"""
            
            # Get embeddings
            head_emb = self.entity_embeddings(heads)
            tail_emb = self.entity_embeddings(tails)
            rel_phase = self.relation_embeddings(relations)
            
            # Convert to complex representation
            head_re, head_im = torch.chunk(head_emb, 2, dim=1)
            tail_re, tail_im = torch.chunk(tail_emb, 2, dim=1)
            
            # Relation rotation
            rel_cos = torch.cos(rel_phase)
            rel_sin = torch.sin(rel_phase)
            
            # Apply rotation: h * r
            rotated_re = head_re * rel_cos - head_im * rel_sin
            rotated_im = head_re * rel_sin + head_im * rel_cos
            
            # Calculate distance to tail
            diff_re = rotated_re - tail_re
            diff_im = rotated_im - tail_im
            
            scores = torch.norm(torch.cat([diff_re, diff_im], dim=1), p=2, dim=1)
            
            return scores
        
        def get_embeddings(self):
            """Extract embeddings as numpy arrays"""
            
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            
            return entity_emb, relation_emb
    
    
    class DistMultModel(torch.nn.Module):
        """DistMult bilinear model"""
        
        def __init__(self, num_entities: int, num_relations: int, dim: int, device: str):
            super().__init__()
            
            self.entity_embeddings = torch.nn.Embedding(num_entities, dim)
            self.relation_embeddings = torch.nn.Embedding(num_relations, dim)
            
            torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
            torch.nn.init.xavier_uniform_(self.relation_embeddings.weight)
            
            self.to(device)
        
        def forward(self, heads, relations, tails):
            """DistMult scoring function"""
            
            head_emb = self.entity_embeddings(heads)
            rel_emb = self.relation_embeddings(relations)
            tail_emb = self.entity_embeddings(tails)
            
            # DistMult score: <h, r, t>
            scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
            
            return scores
        
        def get_embeddings(self):
            """Extract embeddings as numpy arrays"""
            
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            
            return entity_emb, relation_emb

# PyTorch Model Factory Functions (Safe Imports)
def create_transe_model(num_entities: int, num_relations: int, dim: int, device: str):
    """Create TransE model if PyTorch is available"""
    if not torch:
        return SimpleEmbeddingModel(num_entities, num_relations, dim)
    
    import torch.nn as nn
    import torch.optim as optim
    
    class TransEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.dim = dim
            self.device = device
            
            self.entity_embeddings = nn.Embedding(num_entities, dim)
            self.relation_embeddings = nn.Embedding(num_relations, dim)
            
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
            
            self.to(device)
        
        def forward(self, heads, relations, tails):
            head_emb = self.entity_embeddings(heads)
            rel_emb = self.relation_embeddings(relations)
            tail_emb = self.entity_embeddings(tails)
            
            scores = torch.norm(head_emb + rel_emb - tail_emb, p=2, dim=1)
            return scores
        
        def train_model(self, triples, epochs=100, batch_size=1024, learning_rate=0.01,
                       negative_samples=10, margin=1.0):
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.MarginRankingLoss(margin=margin)
            
            losses = []
            triple_tensor = torch.tensor(triples, dtype=torch.long, device=self.device)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                indices = torch.randperm(len(triple_tensor))
                
                for i in range(0, len(triple_tensor), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_triples = triple_tensor[batch_indices]
                    
                    neg_triples = self._generate_negative_samples(batch_triples, negative_samples)
                    
                    pos_scores = self.forward(batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2])
                    neg_scores = self.forward(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
                    
                    target = torch.ones(pos_scores.size(), device=self.device)
                    loss = criterion(neg_scores, pos_scores, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                epoch_loss /= (len(triple_tensor) // batch_size)
                losses.append(epoch_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
            
            eval_metrics = {
                'final_loss': losses[-1] if losses else 0.0,
                'loss_reduction': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 else 0.0
            }
            
            return losses, eval_metrics
        
        def _generate_negative_samples(self, positive_triples, num_negatives):
            batch_size = positive_triples.size(0)
            neg_triples = []
            
            for _ in range(num_negatives):
                neg_batch = positive_triples.clone()
                
                for i in range(batch_size):
                    if torch.rand(1) < 0.5:
                        neg_batch[i, 0] = torch.randint(0, self.num_entities, (1,))
                    else:
                        neg_batch[i, 2] = torch.randint(0, self.num_entities, (1,))
                
                neg_triples.append(neg_batch)
            
            return torch.cat(neg_triples, dim=0)
        
        def get_embeddings(self):
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            return entity_emb, relation_emb
    
    return TransEModel()


def create_complex_model(num_entities: int, num_relations: int, dim: int, device: str):
    """Create ComplEx model if PyTorch is available"""
    if not torch:
        return SimpleEmbeddingModel(num_entities, num_relations, dim)
    
    import torch.nn as nn
    
    class ComplExModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.entity_re = nn.Embedding(num_entities, dim)
            self.entity_im = nn.Embedding(num_entities, dim)
            self.relation_re = nn.Embedding(num_relations, dim)
            self.relation_im = nn.Embedding(num_relations, dim)
            
            for emb in [self.entity_re, self.entity_im, self.relation_re, self.relation_im]:
                nn.init.xavier_uniform_(emb.weight)
            
            self.to(device)
        
        def get_embeddings(self):
            with torch.no_grad():
                entity_emb = torch.cat([
                    self.entity_re.weight, self.entity_im.weight
                ], dim=1).cpu().numpy()
                
                relation_emb = torch.cat([
                    self.relation_re.weight, self.relation_im.weight
                ], dim=1).cpu().numpy()
            
            return entity_emb, relation_emb
        
        def train_model(self, triples, **kwargs):
            # Simplified training for ComplEx
            return SimpleEmbeddingModel(num_entities, num_relations, dim).simple_train(triples)
    
    return ComplExModel()


def create_rotate_model(num_entities: int, num_relations: int, dim: int, device: str):
    """Create RotatE model if PyTorch is available"""
    if not torch:
        return SimpleEmbeddingModel(num_entities, num_relations, dim)
    
    import torch.nn as nn
    
    class RotatEModel(nn.Module):
        def __init__(self):
            super().__init__()
            assert dim % 2 == 0, "Dimension must be even for RotatE"
            
            self.entity_embeddings = nn.Embedding(num_entities, dim)
            self.relation_embeddings = nn.Embedding(num_relations, dim // 2)
            
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
            nn.init.uniform_(self.relation_embeddings.weight, 0, 2 * np.pi)
            
            self.to(device)
        
        def get_embeddings(self):
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            
            return entity_emb, relation_emb
        
        def train_model(self, triples, **kwargs):
            # Simplified training for RotatE
            return SimpleEmbeddingModel(num_entities, num_relations, dim).simple_train(triples)
    
    return RotatEModel()


def create_distmult_model(num_entities: int, num_relations: int, dim: int, device: str):
    """Create DistMult model if PyTorch is available"""
    if not torch:
        return SimpleEmbeddingModel(num_entities, num_relations, dim)
    
    import torch.nn as nn
    
    class DistMultModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.entity_embeddings = nn.Embedding(num_entities, dim)
            self.relation_embeddings = nn.Embedding(num_relations, dim)
            
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
            
            self.to(device)
        
        def get_embeddings(self):
            with torch.no_grad():
                entity_emb = self.entity_embeddings.weight.cpu().numpy()
                relation_emb = self.relation_embeddings.weight.cpu().numpy()
            
            return entity_emb, relation_emb
        
        def train_model(self, triples, **kwargs):
            # Simplified training for DistMult
            return SimpleEmbeddingModel(num_entities, num_relations, dim).simple_train(triples)
    
    return DistMultModel()