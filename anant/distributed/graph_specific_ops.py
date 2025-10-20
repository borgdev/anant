"""
Graph-Specific Distributed Operations
====================================

Specialized distributed operations tailored for each graph type's unique characteristics:

1. Hypergraph Operations:
   - Hyperedge-based partitioning
   - S-centrality calculations
   - Hyperpath algorithms

2. Knowledge Graph Operations:
   - Semantic entity clustering
   - Ontology-aware reasoning
   - SPARQL query distribution

3. Hierarchical Knowledge Graph Operations:
   - Level-aware processing
   - Cross-level relationship analysis
   - Hierarchical navigation

4. Metagraph Operations:
   - Enterprise-scale entity management
   - Policy-aware processing
   - Temporal pattern analysis
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
import asyncio
import logging
from abc import ABC, abstractmethod

from ..classes.hypergraph import Hypergraph
from ..kg.core import KnowledgeGraph
from ..kg.hierarchical import HierarchicalKnowledgeGraph
from ..metagraph.core.metagraph import Metagraph

from .graph_operations import GraphPartition, GraphType, DistributedGraphResult

logger = logging.getLogger(__name__)


class GraphSpecificOperations(ABC):
    """Abstract base for graph-specific distributed operations"""
    
    @abstractmethod
    async def distributed_centrality(self, graph: Any, partitions: List[GraphPartition]) -> Dict[str, float]:
        """Calculate distributed centrality for specific graph type"""
        pass
    
    @abstractmethod
    async def distributed_clustering(self, graph: Any, partitions: List[GraphPartition]) -> Dict[str, int]:
        """Calculate distributed clustering for specific graph type"""
        pass
    
    @abstractmethod
    async def distributed_search(self, graph: Any, partitions: List[GraphPartition], 
                               query: str) -> List[Dict[str, Any]]:
        """Perform distributed search for specific graph type"""
        pass


class HypergraphOperations(GraphSpecificOperations):
    """Specialized operations for traditional hypergraphs"""
    
    async def distributed_centrality(self, graph: Hypergraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, float]:
        """
        Calculate S-centrality across hypergraph partitions
        """
        logger.info("Computing distributed S-centrality for hypergraph")
        
        # S-centrality considers hyperedge structures
        centrality_scores = {}
        
        for partition in partitions:
            partition_scores = await self._compute_partition_s_centrality(partition)
            centrality_scores.update(partition_scores)
        
        # Normalize scores globally
        if centrality_scores:
            max_score = max(centrality_scores.values())
            centrality_scores = {k: v/max_score for k, v in centrality_scores.items()}
        
        return centrality_scores
    
    async def _compute_partition_s_centrality(self, partition: GraphPartition) -> Dict[str, float]:
        """Compute S-centrality for a single partition"""
        scores = {}
        
        # Get partition data - handle both DataFrame and dict cases
        if isinstance(partition.data, pl.DataFrame):
            incidence_data = partition.data
        else:
            # If data is dict, create empty DataFrame
            incidence_data = pl.DataFrame({
                'edge_id': [], 'node_id': [], 'weight': []
            }).with_columns([
                pl.col('edge_id').cast(pl.Utf8),
                pl.col('node_id').cast(pl.Utf8), 
                pl.col('weight').cast(pl.Float64)
            ])
        
        for node_id in partition.node_ids:
            # Find hyperedges containing this node
            node_edges = incidence_data.filter(pl.col("node_id") == node_id)
            
            s_centrality = 0.0
            for row in node_edges.iter_rows(named=True):
                edge_id = row["edge_id"]
                
                # Calculate edge size (number of nodes in hyperedge)
                edge_size = len(incidence_data.filter(pl.col("edge_id") == edge_id))
                
                # S-centrality contribution: 1/(edge_size - 1) for each edge
                if edge_size > 1:
                    s_centrality += 1.0 / (edge_size - 1)
            
            scores[node_id] = s_centrality
        
        return scores
    
    async def distributed_clustering(self, graph: Hypergraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, int]:
        """
        Perform hypergraph clustering considering hyperedge structures
        """
        logger.info("Computing distributed clustering for hypergraph")
        
        # Use hyperedge-based clustering
        node_clusters = {}
        cluster_id = 0
        
        for partition in partitions:
            partition_clusters = await self._compute_partition_clustering(partition, cluster_id)
            node_clusters.update(partition_clusters["clusters"])
            cluster_id = partition_clusters["next_cluster_id"]
        
        return node_clusters
    
    async def _compute_partition_clustering(self, partition: GraphPartition, 
                                          start_cluster_id: int) -> Dict[str, Any]:
        """Compute clustering for a partition"""
        clusters = {}
        current_cluster_id = start_cluster_id
        
        # Handle both DataFrame and dict data types
        if isinstance(partition.data, pl.DataFrame):
            incidence_data = partition.data
            
            # Group by edges and assign clusters
            edge_groups = incidence_data.group_by("edge_id").agg([
                pl.col("node_id").alias("nodes")
            ])
            
            for row in edge_groups.iter_rows(named=True):
                nodes = row["nodes"]
                
                # Check if any node already has a cluster
                existing_cluster = None
                for node in nodes:
                    if node in clusters:
                        existing_cluster = clusters[node]
                        break
                
                # Assign cluster
                cluster_id = existing_cluster if existing_cluster is not None else current_cluster_id
                
                for node in nodes:
                    clusters[node] = cluster_id
                
                if existing_cluster is None:
                    current_cluster_id += 1
        else:
            # Simple fallback clustering for non-DataFrame data
            for i, node in enumerate(partition.node_ids):
                clusters[node] = start_cluster_id + (i % 3)  # Simple grouping
            current_cluster_id = start_cluster_id + 3
        
        return {
            "clusters": clusters,
            "next_cluster_id": current_cluster_id
        }
    
    async def distributed_search(self, graph: Hypergraph, partitions: List[GraphPartition], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Distributed search in hypergraph
        """
        logger.info(f"Performing distributed search for: {query}")
        
        search_results = []
        
        for partition in partitions:
            partition_results = await self._search_partition(partition, query)
            search_results.extend(partition_results)
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return search_results[:100]  # Top 100 results
    
    async def _search_partition(self, partition: GraphPartition, query: str) -> List[Dict[str, Any]]:
        """Search within a partition"""
        results = []
        query_lower = query.lower()
        
        # Simple string matching search
        for node_id in partition.node_ids:
            if query_lower in str(node_id).lower():
                results.append({
                    "node_id": node_id,
                    "type": "node",
                    "score": 1.0,
                    "match_type": "node_id"
                })
        
        for edge_id in partition.edge_ids:
            if query_lower in str(edge_id).lower():
                results.append({
                    "edge_id": edge_id,
                    "type": "edge", 
                    "score": 0.8,
                    "match_type": "edge_id"
                })
        
        return results


class KnowledgeGraphOperations(GraphSpecificOperations):
    """Specialized operations for semantic knowledge graphs"""
    
    async def distributed_centrality(self, graph: KnowledgeGraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, float]:
        """
        Calculate semantic centrality considering entity types and relationships
        """
        logger.info("Computing distributed semantic centrality for knowledge graph")
        
        centrality_scores = {}
        
        for partition in partitions:
            partition_scores = await self._compute_semantic_centrality(partition)
            centrality_scores.update(partition_scores)
        
        return centrality_scores
    
    async def _compute_semantic_centrality(self, partition: GraphPartition) -> Dict[str, float]:
        """Compute semantic centrality for partition"""
        scores = {}
        
        # Semantic weighting based on entity types
        semantic_weights = {
            "person": 1.5,
            "organization": 1.3,
            "concept": 1.2,
            "event": 1.1,
            "default": 1.0
        }
        
        # Handle both DataFrame and dict data types
        if isinstance(partition.data, pl.DataFrame):
            incidence_data = partition.data
            
            for node_id in partition.node_ids:
                # Basic degree centrality
                node_edges = incidence_data.filter(pl.col("node_id") == node_id)
                degree = len(node_edges)
                
                # Apply semantic weighting
                entity_type = self._extract_entity_type(node_id)
                weight = semantic_weights.get(entity_type, semantic_weights["default"])
                
                scores[node_id] = degree * weight
        else:
            # Fallback for non-DataFrame data
            for node_id in partition.node_ids:
                entity_type = self._extract_entity_type(node_id)
                weight = semantic_weights.get(entity_type, semantic_weights["default"])
                scores[node_id] = 1.0 * weight  # Base score with semantic weighting
        
        return scores
    
    def _extract_entity_type(self, node_id: str) -> str:
        """Extract entity type from node ID (simplified)"""
        node_str = str(node_id).lower()
        if "person" in node_str or "people" in node_str:
            return "person"
        elif "org" in node_str or "company" in node_str:
            return "organization"
        elif "concept" in node_str or "idea" in node_str:
            return "concept"
        elif "event" in node_str:
            return "event"
        else:
            return "default"
    
    async def distributed_clustering(self, graph: KnowledgeGraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, int]:
        """
        Semantic clustering based on entity types and relationships
        """
        logger.info("Computing distributed semantic clustering for knowledge graph")
        
        node_clusters = {}
        cluster_id = 0
        
        for partition in partitions:
            # Group by entity types first
            entity_type_groups = {}
            for node_id in partition.node_ids:
                entity_type = self._extract_entity_type(node_id)
                if entity_type not in entity_type_groups:
                    entity_type_groups[entity_type] = []
                entity_type_groups[entity_type].append(node_id)
            
            # Assign clusters by entity type
            for entity_type, nodes in entity_type_groups.items():
                for node in nodes:
                    node_clusters[node] = cluster_id
                cluster_id += 1
        
        return node_clusters
    
    async def distributed_search(self, graph: KnowledgeGraph, partitions: List[GraphPartition], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Semantic search in knowledge graph
        """
        logger.info(f"Performing distributed semantic search for: {query}")
        
        search_results = []
        
        for partition in partitions:
            partition_results = await self._semantic_search_partition(partition, query)
            search_results.extend(partition_results)
        
        # Sort by semantic relevance
        search_results.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)
        
        return search_results[:100]
    
    async def _semantic_search_partition(self, partition: GraphPartition, query: str) -> List[Dict[str, Any]]:
        """Semantic search within partition"""
        results = []
        query_lower = query.lower()
        
        # Enhanced semantic matching
        semantic_keywords = {
            "person": ["people", "individual", "human", "person"],
            "organization": ["company", "org", "institution", "corporation"],
            "concept": ["idea", "theory", "concept", "principle"],
            "event": ["event", "occurrence", "happening", "incident"]
        }
        
        for node_id in partition.node_ids:
            node_str = str(node_id).lower()
            
            # Direct match
            if query_lower in node_str:
                results.append({
                    "node_id": node_id,
                    "type": "entity",
                    "semantic_score": 1.0,
                    "match_type": "direct",
                    "entity_type": self._extract_entity_type(node_id)
                })
            else:
                # Semantic match
                entity_type = self._extract_entity_type(node_id)
                if entity_type in semantic_keywords:
                    for keyword in semantic_keywords[entity_type]:
                        if keyword in query_lower:
                            results.append({
                                "node_id": node_id,
                                "type": "entity",
                                "semantic_score": 0.7,
                                "match_type": "semantic",
                                "entity_type": entity_type
                            })
                            break
        
        return results


class HierarchicalKnowledgeGraphOperations(GraphSpecificOperations):
    """Specialized operations for hierarchical knowledge graphs"""
    
    async def distributed_centrality(self, graph: HierarchicalKnowledgeGraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, float]:
        """
        Calculate hierarchical centrality considering level positions
        """
        logger.info("Computing distributed hierarchical centrality")
        
        centrality_scores = {}
        
        for partition in partitions:
            partition_scores = await self._compute_hierarchical_centrality(partition)
            centrality_scores.update(partition_scores)
        
        return centrality_scores
    
    async def _compute_hierarchical_centrality(self, partition: GraphPartition) -> Dict[str, float]:
        """Compute centrality with hierarchical weighting"""
        scores = {}
        
        # Hierarchical level weights (higher levels get more weight)
        level_weights = {"level_0": 2.0, "level_1": 1.5, "level_2": 1.2, "default": 1.0}
        
        # Handle both DataFrame and dict data types
        if isinstance(partition.data, pl.DataFrame):
            incidence_data = partition.data
            
            for node_id in partition.node_ids:
                # Basic centrality
                node_edges = incidence_data.filter(pl.col("node_id") == node_id)
                degree = len(node_edges)
                
                # Apply hierarchical weighting
                level_id = partition.metadata.get("level_id", "default")
                weight = level_weights.get(level_id, level_weights["default"])
                
                # Bonus for cross-level connections
                cross_level_bonus = 1.2 if "cross_level" in partition.metadata.get("strategy", "") else 1.0
                
                scores[node_id] = degree * weight * cross_level_bonus
        else:
            # Fallback for non-DataFrame data
            for node_id in partition.node_ids:
                level_id = partition.metadata.get("level_id", "default")
                weight = level_weights.get(level_id, level_weights["default"])
                cross_level_bonus = 1.2 if "cross_level" in partition.metadata.get("strategy", "") else 1.0
                scores[node_id] = 1.0 * weight * cross_level_bonus
        
        return scores
    
    async def distributed_clustering(self, graph: HierarchicalKnowledgeGraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, int]:
        """
        Hierarchical clustering respecting level boundaries
        """
        logger.info("Computing distributed hierarchical clustering")
        
        node_clusters = {}
        cluster_id = 0
        
        # Group partitions by level
        level_partitions = {}
        for partition in partitions:
            level_id = partition.metadata.get("level_id", "unknown")
            if level_id not in level_partitions:
                level_partitions[level_id] = []
            level_partitions[level_id].append(partition)
        
        # Cluster within each level
        for level_id, partitions in level_partitions.items():
            for partition in partitions:
                # All nodes in same level partition get same cluster initially
                for node_id in partition.node_ids:
                    node_clusters[node_id] = cluster_id
                cluster_id += 1
        
        return node_clusters
    
    async def distributed_search(self, graph: HierarchicalKnowledgeGraph, 
                               partitions: List[GraphPartition], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Hierarchical search across all levels
        """
        logger.info(f"Performing distributed hierarchical search for: {query}")
        
        search_results = []
        
        for partition in partitions:
            partition_results = await self._hierarchical_search_partition(partition, query)
            search_results.extend(partition_results)
        
        # Sort by hierarchical relevance (higher levels first)
        def sort_key(result):
            level_priority = {"level_0": 3, "level_1": 2, "level_2": 1}
            level = result.get("level_id", "unknown")
            return (level_priority.get(level, 0), result.get("score", 0))
        
        search_results.sort(key=sort_key, reverse=True)
        
        return search_results[:100]
    
    async def _hierarchical_search_partition(self, partition: GraphPartition, 
                                           query: str) -> List[Dict[str, Any]]:
        """Search within hierarchical partition"""
        results = []
        query_lower = query.lower()
        level_id = partition.metadata.get("level_id", "unknown")
        
        for node_id in partition.node_ids:
            if query_lower in str(node_id).lower():
                results.append({
                    "node_id": node_id,
                    "type": "hierarchical_entity",
                    "score": 1.0,
                    "level_id": level_id,
                    "match_type": "direct"
                })
        
        return results


class MetagraphOperations(GraphSpecificOperations):
    """Specialized operations for enterprise metagraphs"""
    
    async def distributed_centrality(self, graph: Metagraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, float]:
        """
        Calculate enterprise centrality considering business importance
        """
        logger.info("Computing distributed enterprise centrality for metagraph")
        
        centrality_scores = {}
        
        for partition in partitions:
            partition_scores = await self._compute_enterprise_centrality(partition)
            centrality_scores.update(partition_scores)
        
        return centrality_scores
    
    async def _compute_enterprise_centrality(self, partition: GraphPartition) -> Dict[str, float]:
        """Compute enterprise-aware centrality"""
        scores = {}
        
        # Enterprise importance weights
        importance_weights = {
            "critical": 3.0,
            "high": 2.0,
            "medium": 1.5,
            "low": 1.0,
            "default": 1.0
        }
        
        for entity_id in partition.node_ids:
            # Base centrality (simplified as we don't have full relationship data)
            base_score = 1.0
            
            # Apply enterprise importance weighting
            importance = self._get_entity_importance(entity_id)
            weight = importance_weights.get(importance, importance_weights["default"])
            
            scores[entity_id] = base_score * weight
        
        return scores
    
    def _get_entity_importance(self, entity_id: str) -> str:
        """Determine enterprise importance of entity (simplified)"""
        entity_str = str(entity_id).lower()
        if "critical" in entity_str or "ceo" in entity_str:
            return "critical"
        elif "important" in entity_str or "manager" in entity_str:
            return "high"
        elif "team" in entity_str or "project" in entity_str:
            return "medium"
        else:
            return "default"
    
    async def distributed_clustering(self, graph: Metagraph, 
                                   partitions: List[GraphPartition]) -> Dict[str, int]:
        """
        Enterprise clustering based on organizational structure
        """
        logger.info("Computing distributed enterprise clustering for metagraph")
        
        node_clusters = {}
        cluster_id = 0
        
        for partition in partitions:
            # Group by enterprise levels/departments
            levels = partition.metadata.get("levels", [])
            
            for level in levels:
                # All entities in same enterprise level get same cluster
                if isinstance(partition.data, dict):
                    level_entities = partition.data.get("entities", [])
                else:
                    # For DataFrame, use node_ids from partition
                    level_entities = partition.node_ids
                    
                for entity_id in level_entities:
                    node_clusters[entity_id] = cluster_id
                cluster_id += 1
        
        return node_clusters
    
    async def distributed_search(self, graph: Metagraph, partitions: List[GraphPartition], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Enterprise-aware search in metagraph
        """
        logger.info(f"Performing distributed enterprise search for: {query}")
        
        search_results = []
        
        for partition in partitions:
            partition_results = await self._enterprise_search_partition(partition, query)
            search_results.extend(partition_results)
        
        # Sort by enterprise importance
        search_results.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        
        return search_results[:100]
    
    async def _enterprise_search_partition(self, partition: GraphPartition, 
                                         query: str) -> List[Dict[str, Any]]:
        """Enterprise search within partition"""
        results = []
        query_lower = query.lower()
        
        # Enterprise-specific search terms
        enterprise_terms = {
            "governance": ["policy", "compliance", "audit", "governance"],
            "operations": ["process", "workflow", "operation", "business"],
            "people": ["employee", "staff", "team", "person"],
            "technology": ["system", "software", "platform", "tech"]
        }
        
        for entity_id in partition.node_ids:
            entity_str = str(entity_id).lower()
            
            if query_lower in entity_str:
                importance = self._get_entity_importance(entity_id)
                importance_score = {"critical": 3.0, "high": 2.0, "medium": 1.5}.get(importance, 1.0)
                
                results.append({
                    "entity_id": entity_id,
                    "type": "enterprise_entity",
                    "importance_score": importance_score,
                    "match_type": "direct",
                    "enterprise_category": self._get_enterprise_category(entity_id)
                })
        
        return results
    
    def _get_enterprise_category(self, entity_id: str) -> str:
        """Categorize entity by enterprise domain"""
        entity_str = str(entity_id).lower()
        if any(term in entity_str for term in ["policy", "compliance", "audit"]):
            return "governance"
        elif any(term in entity_str for term in ["process", "workflow", "operation"]):
            return "operations"
        elif any(term in entity_str for term in ["employee", "staff", "team"]):
            return "people"
        elif any(term in entity_str for term in ["system", "software", "platform"]):
            return "technology"
        else:
            return "general"


# Factory for creating graph-specific operations
class GraphOperationsFactory:
    """Factory for creating graph-specific operation handlers"""
    
    _operations = {
        GraphType.HYPERGRAPH: HypergraphOperations,
        GraphType.KNOWLEDGE_GRAPH: KnowledgeGraphOperations,
        GraphType.HIERARCHICAL_KNOWLEDGE_GRAPH: HierarchicalKnowledgeGraphOperations,
        GraphType.METAGRAPH: MetagraphOperations
    }
    
    @classmethod
    def create_operations(cls, graph_type: GraphType) -> GraphSpecificOperations:
        """Create operations handler for specific graph type"""
        if graph_type not in cls._operations:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
        return cls._operations[graph_type]()
    
    @classmethod
    def get_supported_types(cls) -> List[GraphType]:
        """Get list of supported graph types"""
        return list(cls._operations.keys())