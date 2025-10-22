"""
Analysis Operations for Hierarchical Knowledge Graph
==================================================

This module handles all analytical operations for hierarchical knowledge graphs,
including centrality measures, connectivity analysis, clustering, and structural metrics.

Key Features:
- Hierarchical centrality measures (betweenness, closeness, eigenvector)
- Cross-level connectivity analysis
- Multi-level clustering and community detection
- Structural balance and stability metrics
- Path analysis and shortest path calculations
- Network topology analysis with hierarchical context
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, deque
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisOperations:
    """
    Handles analytical operations for hierarchical knowledge graphs.
    
    This class provides comprehensive graph analysis capabilities that take
    advantage of the hierarchical structure to provide multi-level insights
    and cross-level analytical metrics.
    
    Features:
    - Hierarchical centrality measures
    - Multi-level connectivity analysis
    - Cross-level clustering and communities
    - Structural stability metrics
    - Path analysis with hierarchical context
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize analysis operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
    
    # =====================================================================
    # CENTRALITY MEASURES
    # =====================================================================
    
    def calculate_hierarchical_centrality(self,
                                        centrality_type: str = "betweenness",
                                        level_ids: Optional[List[str]] = None,
                                        include_cross_level: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate centrality measures with hierarchical context.
        
        Args:
            centrality_type: Type of centrality ("betweenness", "closeness", "eigenvector", "degree")
            level_ids: Specific levels to analyze (None for all)
            include_cross_level: Include cross-level relationships in calculation
            
        Returns:
            Dictionary mapping level_id to entity centrality scores
        """
        levels_to_analyze = level_ids or list(self.hkg.levels.keys())
        centrality_results = {}
        
        for level_id in levels_to_analyze:
            level_centralities = self._calculate_level_centrality(level_id, centrality_type)
            
            # Apply hierarchical weighting
            if include_cross_level:
                level_centralities = self._apply_cross_level_weighting(level_centralities, level_id)
            
            centrality_results[level_id] = level_centralities
        
        return centrality_results
    
    def _calculate_level_centrality(self, level_id: str, centrality_type: str) -> Dict[str, float]:
        """Calculate centrality for entities within a specific level."""
        if level_id not in self.hkg.level_graphs:
            return {}
        
        level_graph = self.hkg.level_graphs[level_id]
        entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
        
        if centrality_type == "degree":
            return self._calculate_degree_centrality(entities, level_graph)
        elif centrality_type == "betweenness":
            return self._calculate_betweenness_centrality(entities, level_graph)
        elif centrality_type == "closeness":
            return self._calculate_closeness_centrality(entities, level_graph)
        elif centrality_type == "eigenvector":
            return self._calculate_eigenvector_centrality(entities, level_graph)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    def _calculate_degree_centrality(self, entities: List[str], level_graph) -> Dict[str, float]:
        """Calculate degree centrality for entities."""
        centralities = {}
        total_possible_edges = len(entities) - 1
        
        for entity in entities:
            # Count connections within this level
            connections = 0
            if hasattr(level_graph, 'get_entity_relationships'):
                relationships = level_graph.get_entity_relationships(entity)
                connections = len(relationships)
            
            # Normalize by maximum possible degree
            centralities[entity] = connections / total_possible_edges if total_possible_edges > 0 else 0.0
        
        return centralities
    
    def _calculate_betweenness_centrality(self, entities: List[str], level_graph) -> Dict[str, float]:
        """Calculate betweenness centrality using shortest paths."""
        centralities = {entity: 0.0 for entity in entities}
        
        # Calculate shortest paths between all pairs
        for source in entities:
            paths = self._single_source_shortest_paths(source, entities, level_graph)
            
            for target in entities:
                if source != target and target in paths:
                    path = paths[target]
                    # Count how many shortest paths pass through each entity
                    for intermediate in path[1:-1]:  # Exclude source and target
                        if intermediate in centralities:
                            centralities[intermediate] += 1.0
        
        # Normalize by number of pairs
        n = len(entities)
        normalization = (n - 1) * (n - 2) / 2 if n > 2 else 1
        for entity in centralities:
            centralities[entity] /= normalization
        
        return centralities
    
    def _calculate_closeness_centrality(self, entities: List[str], level_graph) -> Dict[str, float]:
        """Calculate closeness centrality based on average shortest path length."""
        centralities = {}
        
        for source in entities:
            paths = self._single_source_shortest_paths(source, entities, level_graph)
            
            total_distance = 0
            reachable_count = 0
            
            for target, path in paths.items():
                if target != source and path:
                    total_distance += len(path) - 1  # Path length
                    reachable_count += 1
            
            if reachable_count > 0:
                avg_distance = total_distance / reachable_count
                centralities[source] = 1.0 / avg_distance if avg_distance > 0 else 0.0
            else:
                centralities[source] = 0.0
        
        return centralities
    
    def _calculate_eigenvector_centrality(self, entities: List[str], level_graph) -> Dict[str, float]:
        """Calculate eigenvector centrality using power iteration."""
        # Build adjacency matrix
        entity_to_index = {entity: i for i, entity in enumerate(entities)}
        n = len(entities)
        adj_matrix = [[0.0] * n for _ in range(n)]
        
        # Fill adjacency matrix
        for entity in entities:
            if hasattr(level_graph, 'get_entity_relationships'):
                relationships = level_graph.get_entity_relationships(entity)
                for rel in relationships:
                    target = rel.get('target_entity')
                    if target in entity_to_index:
                        i, j = entity_to_index[entity], entity_to_index[target]
                        adj_matrix[i][j] = 1.0
                        adj_matrix[j][i] = 1.0  # Undirected
        
        # Power iteration
        centralities = {entity: 1.0 for entity in entities}
        
        for iteration in range(100):  # Max iterations
            new_centralities = {}
            for i, entity in enumerate(entities):
                score = sum(adj_matrix[i][j] * centralities[entities[j]] for j in range(n))
                new_centralities[entity] = score
            
            # Normalize
            total_score = sum(new_centralities.values())
            if total_score > 0:
                for entity in new_centralities:
                    new_centralities[entity] /= total_score
            
            # Check convergence
            max_change = max(abs(new_centralities[entity] - centralities[entity]) for entity in entities)
            centralities = new_centralities
            
            if max_change < 1e-6:
                break
        
        return centralities
    
    def _apply_cross_level_weighting(self, centralities: Dict[str, float], level_id: str) -> Dict[str, float]:
        """Apply weighting based on cross-level relationships."""
        weighted_centralities = centralities.copy()
        
        for entity_id, base_centrality in centralities.items():
            # Count cross-level relationships
            cross_level_count = 0
            for relationship in self.hkg.cross_level_relationships:
                if (relationship.get('source_entity') == entity_id or 
                    relationship.get('target_entity') == entity_id):
                    cross_level_count += 1
            
            # Apply cross-level boost
            cross_level_boost = 1.0 + (cross_level_count * 0.1)  # 10% boost per cross-level connection
            weighted_centralities[entity_id] = base_centrality * cross_level_boost
        
        return weighted_centralities
    
    # =====================================================================
    # CONNECTIVITY ANALYSIS
    # =====================================================================
    
    def analyze_connectivity(self,
                           level_ids: Optional[List[str]] = None,
                           include_cross_level: bool = True) -> Dict[str, Any]:
        """
        Analyze connectivity patterns across hierarchical levels.
        
        Args:
            level_ids: Levels to analyze (None for all)
            include_cross_level: Include cross-level connectivity
            
        Returns:
            Comprehensive connectivity analysis results
        """
        levels_to_analyze = level_ids or list(self.hkg.levels.keys())
        
        connectivity_analysis = {
            'timestamp': datetime.now().isoformat(),
            'levels_analyzed': levels_to_analyze,
            'level_connectivity': {},
            'cross_level_connectivity': {},
            'overall_metrics': {}
        }
        
        # Analyze each level
        for level_id in levels_to_analyze:
            level_stats = self._analyze_level_connectivity(level_id)
            connectivity_analysis['level_connectivity'][level_id] = level_stats
        
        # Analyze cross-level connectivity
        if include_cross_level:
            cross_level_stats = self._analyze_cross_level_connectivity()
            connectivity_analysis['cross_level_connectivity'] = cross_level_stats
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_connectivity_metrics(connectivity_analysis)
        connectivity_analysis['overall_metrics'] = overall_metrics
        
        return connectivity_analysis
    
    def _analyze_level_connectivity(self, level_id: str) -> Dict[str, Any]:
        """Analyze connectivity within a specific level."""
        entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
        level_graph = self.hkg.level_graphs.get(level_id)
        
        stats = {
            'entity_count': len(entities),
            'connection_count': 0,
            'density': 0.0,
            'avg_degree': 0.0,
            'clustering_coefficient': 0.0,
            'connected_components': 0,
            'diameter': 0
        }
        
        if not entities or not level_graph:
            return stats
        
        # Count connections and calculate degree distribution
        degrees = []
        total_connections = 0
        
        for entity in entities:
            if hasattr(level_graph, 'get_entity_relationships'):
                relationships = level_graph.get_entity_relationships(entity)
                degree = len(relationships)
                degrees.append(degree)
                total_connections += degree
        
        stats['connection_count'] = total_connections // 2  # Undirected edges
        stats['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
        
        # Calculate density
        max_possible_edges = len(entities) * (len(entities) - 1) // 2
        stats['density'] = stats['connection_count'] / max_possible_edges if max_possible_edges > 0 else 0
        
        # Estimate clustering coefficient
        stats['clustering_coefficient'] = self._estimate_clustering_coefficient(entities, level_graph)
        
        # Find connected components
        components = self._find_connected_components(entities, level_graph)
        stats['connected_components'] = len(components)
        
        # Calculate diameter of largest component
        if components:
            largest_component = max(components, key=len)
            stats['diameter'] = self._calculate_diameter(largest_component, level_graph)
        
        return stats
    
    def _analyze_cross_level_connectivity(self) -> Dict[str, Any]:
        """Analyze connectivity patterns across different levels."""
        cross_level_stats = {
            'total_cross_level_relationships': len(self.hkg.cross_level_relationships),
            'relationships_by_level_pair': defaultdict(int),
            'cross_level_density': 0.0,
            'bridge_entities': []  # Entities with many cross-level connections
        }
        
        # Analyze relationships by level pairs
        for relationship in self.hkg.cross_level_relationships:
            source_entity = relationship.get('source_entity')
            target_entity = relationship.get('target_entity')
            
            source_level = self.hkg.entity_levels.get(source_entity, 'unknown')
            target_level = self.hkg.entity_levels.get(target_entity, 'unknown')
            
            level_pair = tuple(sorted([source_level, target_level]))
            cross_level_stats['relationships_by_level_pair'][level_pair] += 1
        
        # Find bridge entities
        entity_cross_level_count = defaultdict(int)
        for relationship in self.hkg.cross_level_relationships:
            source_entity = relationship.get('source_entity')
            target_entity = relationship.get('target_entity')
            entity_cross_level_count[source_entity] += 1
            entity_cross_level_count[target_entity] += 1
        
        # Top bridge entities
        sorted_bridges = sorted(entity_cross_level_count.items(), key=lambda x: x[1], reverse=True)
        cross_level_stats['bridge_entities'] = sorted_bridges[:10]
        
        return cross_level_stats
    
    def _estimate_clustering_coefficient(self, entities: List[str], level_graph) -> float:
        """Estimate local clustering coefficient."""
        if len(entities) < 3:
            return 0.0
        
        total_clustering = 0.0
        valid_entities = 0
        
        for entity in entities:
            neighbors = self._get_neighbors(entity, level_graph)
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if self._are_connected(neighbor1, neighbor2, level_graph):
                        triangles += 1
            
            local_clustering = triangles / possible_triangles if possible_triangles > 0 else 0
            total_clustering += local_clustering
            valid_entities += 1
        
        return total_clustering / valid_entities if valid_entities > 0 else 0.0
    
    def _find_connected_components(self, entities: List[str], level_graph) -> List[List[str]]:
        """Find connected components using DFS."""
        visited = set()
        components = []
        
        for entity in entities:
            if entity not in visited:
                component = []
                stack = [entity]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        neighbors = self._get_neighbors(current, level_graph)
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if component:
                    components.append(component)
        
        return components
    
    def _calculate_diameter(self, entities: List[str], level_graph) -> int:
        """Calculate diameter (longest shortest path) of a connected component."""
        max_distance = 0
        
        for source in entities:
            distances = self._single_source_distances(source, entities, level_graph)
            for target, distance in distances.items():
                if distance > max_distance:
                    max_distance = distance
        
        return max_distance
    
    # =====================================================================
    # CLUSTERING AND COMMUNITY DETECTION
    # =====================================================================
    
    def detect_communities(self,
                          level_ids: Optional[List[str]] = None,
                          algorithm: str = "modularity",
                          min_community_size: int = 3) -> Dict[str, Any]:
        """
        Detect communities within hierarchical levels.
        
        Args:
            level_ids: Levels to analyze
            algorithm: Community detection algorithm
            min_community_size: Minimum size for valid communities
            
        Returns:
            Community detection results by level
        """
        levels_to_analyze = level_ids or list(self.hkg.levels.keys())
        community_results = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm,
            'min_community_size': min_community_size,
            'communities_by_level': {}
        }
        
        for level_id in levels_to_analyze:
            level_communities = self._detect_level_communities(level_id, algorithm, min_community_size)
            community_results['communities_by_level'][level_id] = level_communities
        
        return community_results
    
    def _detect_level_communities(self, level_id: str, algorithm: str, min_size: int) -> Dict[str, Any]:
        """Detect communities within a specific level."""
        entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
        level_graph = self.hkg.level_graphs.get(level_id)
        
        if len(entities) < min_size:
            return {'communities': [], 'modularity': 0.0, 'coverage': 0.0}
        
        if algorithm == "modularity":
            communities = self._modularity_based_clustering(entities, level_graph)
        else:
            # Fallback to simple connected components
            communities = self._find_connected_components(entities, level_graph)
        
        # Filter by minimum size
        valid_communities = [comm for comm in communities if len(comm) >= min_size]
        
        # Calculate quality metrics
        modularity = self._calculate_modularity(valid_communities, entities, level_graph)
        coverage = sum(len(comm) for comm in valid_communities) / len(entities) if entities else 0
        
        return {
            'communities': valid_communities,
            'community_count': len(valid_communities),
            'modularity': modularity,
            'coverage': coverage,
            'avg_community_size': sum(len(comm) for comm in valid_communities) / len(valid_communities) if valid_communities else 0
        }
    
    def _modularity_based_clustering(self, entities: List[str], level_graph) -> List[List[str]]:
        """Simple modularity-based community detection."""
        # Start with each entity as its own community
        communities = [[entity] for entity in entities]
        
        # Iteratively merge communities to improve modularity
        best_modularity = self._calculate_modularity(communities, entities, level_graph)
        
        for iteration in range(len(entities)):
            best_merge = None
            best_new_modularity = best_modularity
            
            # Try merging each pair of communities
            for i in range(len(communities)):
                for j in range(i + 1, len(communities)):
                    # Test merge
                    test_communities = communities[:i] + communities[i+1:j] + communities[j+1:]
                    test_communities.append(communities[i] + communities[j])
                    
                    modularity = self._calculate_modularity(test_communities, entities, level_graph)
                    if modularity > best_new_modularity:
                        best_new_modularity = modularity
                        best_merge = (i, j)
            
            # Apply best merge if it improves modularity
            if best_merge:
                i, j = best_merge
                merged_community = communities[i] + communities[j]
                communities = communities[:i] + communities[i+1:j] + communities[j+1:] + [merged_community]
                best_modularity = best_new_modularity
            else:
                break  # No improvement found
        
        return communities
    
    def _calculate_modularity(self, communities: List[List[str]], entities: List[str], level_graph) -> float:
        """Calculate modularity score for community partition."""
        if not communities or not entities:
            return 0.0
        
        # Build community membership mapping
        entity_to_community = {}
        for i, community in enumerate(communities):
            for entity in community:
                entity_to_community[entity] = i
        
        # Count edges and calculate modularity
        total_edges = 0
        modularity = 0.0
        
        # Count total edges
        for entity in entities:
            neighbors = self._get_neighbors(entity, level_graph)
            total_edges += len(neighbors)
        total_edges = total_edges // 2  # Undirected
        
        if total_edges == 0:
            return 0.0
        
        # Calculate modularity
        for i, community in enumerate(communities):
            internal_edges = 0
            total_degree = 0
            
            for entity in community:
                neighbors = self._get_neighbors(entity, level_graph)
                total_degree += len(neighbors)
                
                for neighbor in neighbors:
                    if neighbor in community:
                        internal_edges += 1
            
            internal_edges = internal_edges // 2  # Undirected
            expected_internal = (total_degree ** 2) / (4 * total_edges)
            
            modularity += (internal_edges / total_edges) - expected_internal / total_edges
        
        return modularity
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def _single_source_shortest_paths(self, source: str, targets: List[str], level_graph) -> Dict[str, List[str]]:
        """Calculate shortest paths from source to all targets using BFS."""
        paths = {source: [source]}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_path = paths[current]
            
            neighbors = self._get_neighbors(current, level_graph)
            for neighbor in neighbors:
                if neighbor not in paths and neighbor in targets:
                    paths[neighbor] = current_path + [neighbor]
                    queue.append(neighbor)
        
        return paths
    
    def _single_source_distances(self, source: str, targets: List[str], level_graph) -> Dict[str, int]:
        """Calculate shortest path distances from source to all targets."""
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            neighbors = self._get_neighbors(current, level_graph)
            for neighbor in neighbors:
                if neighbor not in distances and neighbor in targets:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        return distances
    
    def _get_neighbors(self, entity: str, level_graph) -> List[str]:
        """Get neighboring entities for a given entity."""
        neighbors = []
        
        if hasattr(level_graph, 'get_entity_relationships'):
            relationships = level_graph.get_entity_relationships(entity)
            for rel in relationships:
                target = rel.get('target_entity')
                source = rel.get('source_entity')
                
                # Add the other endpoint
                if target == entity and source:
                    neighbors.append(source)
                elif source == entity and target:
                    neighbors.append(target)
        
        return neighbors
    
    def _are_connected(self, entity1: str, entity2: str, level_graph) -> bool:
        """Check if two entities are directly connected."""
        neighbors = self._get_neighbors(entity1, level_graph)
        return entity2 in neighbors
    
    def _calculate_overall_connectivity_metrics(self, connectivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall connectivity metrics across all levels."""
        level_connectivity = connectivity_analysis.get('level_connectivity', {})
        
        if not level_connectivity:
            return {}
        
        # Aggregate metrics across levels
        total_entities = sum(stats.get('entity_count', 0) for stats in level_connectivity.values())
        total_connections = sum(stats.get('connection_count', 0) for stats in level_connectivity.values())
        avg_density = sum(stats.get('density', 0) for stats in level_connectivity.values()) / len(level_connectivity)
        avg_clustering = sum(stats.get('clustering_coefficient', 0) for stats in level_connectivity.values()) / len(level_connectivity)
        
        return {
            'total_entities': total_entities,
            'total_connections': total_connections,
            'average_density': avg_density,
            'average_clustering_coefficient': avg_clustering,
            'hierarchy_depth': len(level_connectivity),
            'cross_level_relationships': len(connectivity_analysis.get('cross_level_connectivity', {}).get('total_cross_level_relationships', 0))
        }