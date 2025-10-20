"""
Graph Operations Module
=======================

Core graph operations and algorithms including:
- Graph construction and manipulation
- Traversal algorithms (BFS, DFS, shortest paths)
- Graph metrics and analysis
- Subgraph operations
- Graph transformation and optimization
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set, Callable
from datetime import datetime
import uuid
import logging
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass

from ....exceptions import (
    GraphError, ValidationError, handle_exception,
    require_not_none, require_valid_string, require_valid_dict
)

logger = logging.getLogger(__name__)


@dataclass
class GraphMetrics:
    """Container for graph metrics."""
    node_count: int
    edge_count: int
    density: float
    average_degree: float
    diameter: Optional[int]
    clustering_coefficient: float
    connected_components: int


@dataclass
class PathResult:
    """Container for path finding results."""
    path: List[str]
    distance: float
    exists: bool
    algorithm_used: str


class GraphOperations:
    """
    Handles core graph operations and algorithms for the Metagraph.
    
    Provides graph construction, traversal, analysis, and transformation
    operations with proper error handling and optimization.
    """
    
    def __init__(self, metadata_store, performance_config: Optional[Dict[str, Any]] = None):
        """
        Initialize graph operations.
        
        Args:
            metadata_store: Reference to metadata storage system
            performance_config: Configuration for performance optimization
        """
        self.metadata_store = metadata_store
        self.performance_config = performance_config or {}
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Cache for expensive computations
        self._adjacency_cache = {}
        self._metrics_cache = {}
        self._cache_timestamp = None
        
        # Performance settings
        self.cache_timeout = self.performance_config.get("cache_timeout", 300)  # 5 minutes
        self.max_path_length = self.performance_config.get("max_path_length", 100)
        self.batch_size = self.performance_config.get("batch_size", 1000)
    
    def build_graph_from_entities(self,
                                 entity_filter: Optional[Dict[str, Any]] = None,
                                 include_properties: bool = True,
                                 directed: bool = True) -> Dict[str, Any]:
        """
        Build graph representation from entities and relationships.
        
        Args:
            entity_filter: Optional filter for entities to include
            include_properties: Whether to include entity/edge properties
            directed: Whether to create directed graph
            
        Returns:
            Graph representation with nodes and edges
            
        Raises:
            GraphError: If graph construction fails
        """
        try:
            # Get entities
            all_entities = self.metadata_store.get_all_entities()
            
            # Apply filters
            if entity_filter:
                entities = self._filter_entities_for_graph(all_entities, entity_filter)
            else:
                entities = all_entities
            
            # Get relationships
            all_relationships = self.metadata_store.get_all_relationships()
            entity_ids = {e["entity_id"] for e in entities}
            
            # Filter relationships to only include those between filtered entities
            relationships = [
                rel for rel in all_relationships
                if rel.get("source_entity_id") in entity_ids and 
                   rel.get("target_entity_id") in entity_ids
            ]
            
            # Build graph structure
            graph = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "directed": directed,
                    "node_count": len(entities),
                    "edge_count": len(relationships),
                    "include_properties": include_properties
                },
                "nodes": {},
                "edges": {},
                "adjacency_list": defaultdict(set),
                "reverse_adjacency_list": defaultdict(set) if directed else None
            }
            
            # Add nodes
            for entity in entities:
                node_id = entity["entity_id"]
                node_data = {
                    "id": node_id,
                    "type": entity.get("entity_type", "unknown")
                }
                
                if include_properties:
                    node_data["properties"] = entity.get("properties", {})
                    node_data["created_at"] = entity.get("created_at")
                    node_data["updated_at"] = entity.get("updated_at")
                
                graph["nodes"][node_id] = node_data
            
            # Add edges
            for relationship in relationships:
                source_id = relationship["source_entity_id"]
                target_id = relationship["target_entity_id"]
                edge_id = relationship.get("relationship_id", f"{source_id}-{target_id}")
                
                edge_data = {
                    "id": edge_id,
                    "source": source_id,
                    "target": target_id,
                    "type": relationship.get("relationship_type", "unknown")
                }
                
                if include_properties:
                    edge_data["properties"] = relationship.get("properties", {})
                    edge_data["created_at"] = relationship.get("created_at")
                    edge_data["weight"] = relationship.get("weight", 1.0)
                
                graph["edges"][edge_id] = edge_data
                
                # Update adjacency lists
                graph["adjacency_list"][source_id].add(target_id)
                if directed:
                    graph["reverse_adjacency_list"][target_id].add(source_id)
                else:
                    graph["adjacency_list"][target_id].add(source_id)
            
            # Convert sets to lists for JSON serialization
            graph["adjacency_list"] = {k: list(v) for k, v in graph["adjacency_list"].items()}
            if graph["reverse_adjacency_list"]:
                graph["reverse_adjacency_list"] = {k: list(v) for k, v in graph["reverse_adjacency_list"].items()}
            
            self.logger.info(
                "Graph built successfully",
                extra={
                    "node_count": len(entities),
                    "edge_count": len(relationships),
                    "directed": directed,
                    "include_properties": include_properties
                }
            )
            
            return graph
            
        except Exception as e:
            raise handle_exception("building graph from entities", e, {
                "entity_filter": entity_filter,
                "directed": directed
            })
    
    def find_shortest_path(self,
                          source_id: str,
                          target_id: str,
                          algorithm: str = "dijkstra",
                          weight_property: Optional[str] = None,
                          max_depth: Optional[int] = None) -> PathResult:
        """
        Find shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            algorithm: Algorithm to use (dijkstra, bfs, a_star)
            weight_property: Property to use for edge weights
            max_depth: Maximum search depth
            
        Returns:
            Path result with path, distance, and metadata
            
        Raises:
            GraphError: If path finding fails
        """
        try:
            # Validate inputs
            source_id = require_valid_string(source_id, "source_id")
            target_id = require_valid_string(target_id, "target_id")
            
            if algorithm not in ["dijkstra", "bfs", "a_star"]:
                raise ValidationError(
                    f"Unsupported algorithm: {algorithm}",
                    error_code="UNSUPPORTED_ALGORITHM",
                    context={"algorithm": algorithm}
                )
            
            # Check if nodes exist
            if not self.metadata_store.get_entity(source_id):
                raise GraphError(
                    f"Source entity '{source_id}' not found",
                    error_code="SOURCE_NOT_FOUND",
                    context={"source_id": source_id}
                )
            
            if not self.metadata_store.get_entity(target_id):
                raise GraphError(
                    f"Target entity '{target_id}' not found",
                    error_code="TARGET_NOT_FOUND",
                    context={"target_id": target_id}
                )
            
            # Get adjacency information
            adjacency_data = self._get_adjacency_data()
            
            # Apply max depth limit
            effective_max_depth = min(max_depth or self.max_path_length, self.max_path_length)
            
            # Run algorithm
            if algorithm == "dijkstra":
                result = self._dijkstra(source_id, target_id, adjacency_data, weight_property, effective_max_depth)
            elif algorithm == "bfs":
                result = self._bfs(source_id, target_id, adjacency_data, effective_max_depth)
            elif algorithm == "a_star":
                result = self._a_star(source_id, target_id, adjacency_data, weight_property, effective_max_depth)
            else:
                raise GraphError(f"Algorithm {algorithm} not implemented")
            
            result.algorithm_used = algorithm
            
            self.logger.info(
                "Shortest path search completed",
                extra={
                    "source_id": source_id,
                    "target_id": target_id,
                    "algorithm": algorithm,
                    "path_found": result.exists,
                    "path_length": len(result.path) if result.exists else 0,
                    "distance": result.distance
                }
            )
            
            return result
            
        except (GraphError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"finding path from '{source_id}' to '{target_id}'", e, {
                "source_id": source_id,
                "target_id": target_id,
                "algorithm": algorithm
            })
    
    def graph_traversal(self,
                       start_nodes: List[str],
                       traversal_type: str = "bfs",
                       visit_function: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                       max_depth: Optional[int] = None,
                       filter_condition: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
        """
        Perform graph traversal from starting nodes.
        
        Args:
            start_nodes: List of starting node IDs
            traversal_type: Type of traversal (bfs, dfs)
            visit_function: Optional function to call for each visited node
            max_depth: Maximum traversal depth
            filter_condition: Optional condition to filter visited nodes
            
        Returns:
            Traversal results including visited nodes and metadata
            
        Raises:
            GraphError: If traversal fails
        """
        try:
            # Validate inputs
            if not start_nodes:
                raise ValidationError(
                    "At least one start node must be provided",
                    error_code="NO_START_NODES"
                )
            
            if traversal_type not in ["bfs", "dfs"]:
                raise ValidationError(
                    f"Unsupported traversal type: {traversal_type}",
                    error_code="UNSUPPORTED_TRAVERSAL_TYPE",
                    context={"traversal_type": traversal_type}
                )
            
            # Get adjacency data
            adjacency_data = self._get_adjacency_data()
            
            # Initialize traversal state
            visited = set()
            visit_order = []
            depth_map = {}
            parent_map = {}
            
            # Choose data structure based on traversal type
            queue = deque()
            stack = []
            
            if traversal_type == "bfs":
                for start_node in start_nodes:
                    queue.append((start_node, 0, None))
                    depth_map[start_node] = 0
            else:  # dfs
                for start_node in reversed(start_nodes):
                    stack.append((start_node, 0, None))
                    depth_map[start_node] = 0
            
            # Perform traversal
            nodes_processed = 0
            effective_max_depth = max_depth or self.max_path_length
            
            while (queue if traversal_type == "bfs" else stack) and nodes_processed < 100000:  # Safety limit
                if traversal_type == "bfs":
                    current_node, depth, parent = queue.popleft()
                else:
                    current_node, depth, parent = stack.pop()
                
                if current_node in visited or depth > effective_max_depth:
                    continue
                
                # Get node data
                node_data = self.metadata_store.get_entity(current_node)
                if not node_data:
                    continue
                
                # Apply filter condition
                if filter_condition and not filter_condition(current_node, node_data):
                    continue
                
                # Mark as visited
                visited.add(current_node)
                visit_order.append(current_node)
                if parent:
                    parent_map[current_node] = parent
                nodes_processed += 1
                
                # Call visit function
                if visit_function:
                    should_continue = visit_function(current_node, node_data)
                    if should_continue is False:
                        break
                
                # Add neighbors
                neighbors = adjacency_data.get(current_node, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_depth = depth + 1
                        if neighbor not in depth_map or depth_map[neighbor] > new_depth:
                            depth_map[neighbor] = new_depth
                            
                            if traversal_type == "bfs":
                                queue.append((neighbor, new_depth, current_node))
                            else:
                                stack.append((neighbor, new_depth, current_node))
            
            # Create traversal tree
            traversal_tree = self._build_traversal_tree(visit_order, parent_map, depth_map)
            
            traversal_result = {
                "start_nodes": start_nodes,
                "traversal_type": traversal_type,
                "max_depth": effective_max_depth,
                "visited_nodes": visit_order,
                "visit_count": len(visit_order),
                "max_depth_reached": max(depth_map.values()) if depth_map else 0,
                "traversal_tree": traversal_tree,
                "depth_distribution": self._calculate_depth_distribution(depth_map),
                "completed_at": datetime.now().isoformat()
            }
            
            self.logger.info(
                "Graph traversal completed",
                extra={
                    "start_nodes_count": len(start_nodes),
                    "traversal_type": traversal_type,
                    "nodes_visited": len(visit_order),
                    "max_depth_reached": traversal_result["max_depth_reached"]
                }
            )
            
            return traversal_result
            
        except (GraphError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception("performing graph traversal", e, {
                "start_nodes": start_nodes,
                "traversal_type": traversal_type
            })
    
    def calculate_graph_metrics(self,
                               entity_filter: Optional[Dict[str, Any]] = None,
                               include_advanced: bool = False) -> GraphMetrics:
        """
        Calculate comprehensive graph metrics.
        
        Args:
            entity_filter: Optional filter for entities to include
            include_advanced: Whether to calculate advanced metrics
            
        Returns:
            Graph metrics object
            
        Raises:
            GraphError: If metrics calculation fails
        """
        try:
            # Check cache first
            cache_key = f"metrics_{hash(str(entity_filter))}_{include_advanced}"
            if self._is_cache_valid() and cache_key in self._metrics_cache:
                return self._metrics_cache[cache_key]
            
            # Build graph
            graph = self.build_graph_from_entities(entity_filter, include_properties=False)
            
            nodes = graph["nodes"]
            edges = graph["edges"]
            adjacency_list = graph["adjacency_list"]
            
            node_count = len(nodes)
            edge_count = len(edges)
            
            # Basic metrics
            if node_count == 0:
                metrics = GraphMetrics(
                    node_count=0,
                    edge_count=0,
                    density=0.0,
                    average_degree=0.0,
                    diameter=None,
                    clustering_coefficient=0.0,
                    connected_components=0
                )
            else:
                # Calculate density
                max_edges = node_count * (node_count - 1)
                if not graph["metadata"]["directed"]:
                    max_edges //= 2
                density = edge_count / max_edges if max_edges > 0 else 0.0
                
                # Calculate average degree
                total_degree = sum(len(neighbors) for neighbors in adjacency_list.values())
                average_degree = total_degree / node_count if node_count > 0 else 0.0
                
                # Calculate connected components
                connected_components = self._count_connected_components(adjacency_list, list(nodes.keys()))
                
                # Initialize advanced metrics
                diameter = None
                clustering_coefficient = 0.0
                
                if include_advanced and node_count < 10000:  # Limit for performance
                    diameter = self._calculate_diameter(adjacency_list, list(nodes.keys()))
                    clustering_coefficient = self._calculate_clustering_coefficient(adjacency_list)
                
                metrics = GraphMetrics(
                    node_count=node_count,
                    edge_count=edge_count,
                    density=density,
                    average_degree=average_degree,
                    diameter=diameter,
                    clustering_coefficient=clustering_coefficient,
                    connected_components=connected_components
                )
            
            # Cache result
            self._metrics_cache[cache_key] = metrics
            
            self.logger.info(
                "Graph metrics calculated",
                extra={
                    "node_count": metrics.node_count,
                    "edge_count": metrics.edge_count,
                    "density": metrics.density,
                    "connected_components": metrics.connected_components,
                    "include_advanced": include_advanced
                }
            )
            
            return metrics
            
        except Exception as e:
            raise handle_exception("calculating graph metrics", e, {
                "entity_filter": entity_filter,
                "include_advanced": include_advanced
            })
    
    def extract_subgraph(self,
                        node_ids: List[str],
                        include_neighbors: bool = False,
                        neighbor_depth: int = 1,
                        preserve_connectivity: bool = True) -> Dict[str, Any]:
        """
        Extract subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_neighbors: Whether to include neighboring nodes
            neighbor_depth: Depth of neighbors to include
            preserve_connectivity: Whether to include connecting edges
            
        Returns:
            Subgraph representation
            
        Raises:
            GraphError: If subgraph extraction fails
        """
        try:
            # Validate inputs
            if not node_ids:
                raise ValidationError(
                    "At least one node ID must be provided",
                    error_code="NO_NODE_IDS"
                )
            
            # Get full graph
            full_graph = self.build_graph_from_entities()
            
            # Start with specified nodes
            subgraph_nodes = set(node_ids)
            
            # Add neighbors if requested
            if include_neighbors:
                for depth in range(neighbor_depth):
                    current_nodes = set(subgraph_nodes)
                    for node_id in current_nodes:
                        neighbors = full_graph["adjacency_list"].get(node_id, [])
                        subgraph_nodes.update(neighbors)
            
            # Filter nodes that actually exist
            existing_nodes = {nid for nid in subgraph_nodes if nid in full_graph["nodes"]}
            
            # Extract subgraph
            subgraph = {
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "source_node_count": len(full_graph["nodes"]),
                    "source_edge_count": len(full_graph["edges"]),
                    "requested_nodes": node_ids,
                    "include_neighbors": include_neighbors,
                    "neighbor_depth": neighbor_depth,
                    "preserve_connectivity": preserve_connectivity
                },
                "nodes": {},
                "edges": {},
                "adjacency_list": defaultdict(list)
            }
            
            # Add nodes
            for node_id in existing_nodes:
                subgraph["nodes"][node_id] = full_graph["nodes"][node_id]
            
            # Add edges
            for edge_id, edge_data in full_graph["edges"].items():
                source = edge_data["source"]
                target = edge_data["target"]
                
                # Include edge if both nodes are in subgraph
                if source in existing_nodes and target in existing_nodes:
                    subgraph["edges"][edge_id] = edge_data
                    subgraph["adjacency_list"][source].append(target)
                elif preserve_connectivity:
                    # Check if edge connects components we want to preserve
                    if (source in node_ids and target in existing_nodes) or \
                       (target in node_ids and source in existing_nodes):
                        # Add the missing node if it connects our subgraph
                        if source not in existing_nodes and source in full_graph["nodes"]:
                            subgraph["nodes"][source] = full_graph["nodes"][source]
                            existing_nodes.add(source)
                        if target not in existing_nodes and target in full_graph["nodes"]:
                            subgraph["nodes"][target] = full_graph["nodes"][target]
                            existing_nodes.add(target)
                        
                        subgraph["edges"][edge_id] = edge_data
                        subgraph["adjacency_list"][source].append(target)
            
            # Convert to regular dict
            subgraph["adjacency_list"] = dict(subgraph["adjacency_list"])
            
            # Update metadata with final counts
            subgraph["metadata"]["extracted_node_count"] = len(subgraph["nodes"])
            subgraph["metadata"]["extracted_edge_count"] = len(subgraph["edges"])
            
            self.logger.info(
                "Subgraph extracted successfully",
                extra={
                    "requested_nodes": len(node_ids),
                    "extracted_nodes": len(subgraph["nodes"]),
                    "extracted_edges": len(subgraph["edges"]),
                    "include_neighbors": include_neighbors,
                    "neighbor_depth": neighbor_depth
                }
            )
            
            return subgraph
            
        except (GraphError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception("extracting subgraph", e, {
                "node_ids": node_ids,
                "include_neighbors": include_neighbors,
                "neighbor_depth": neighbor_depth
            })
    
    def _get_adjacency_data(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """Get adjacency list representation with caching."""
        if use_cache and self._is_cache_valid() and "adjacency" in self._adjacency_cache:
            return self._adjacency_cache["adjacency"]
        
        # Build fresh adjacency data
        all_relationships = self.metadata_store.get_all_relationships()
        adjacency = defaultdict(list)
        
        for rel in all_relationships:
            source = rel.get("source_entity_id")
            target = rel.get("target_entity_id")
            if source and target:
                adjacency[source].append(target)
        
        # Convert to regular dict and cache
        adjacency = dict(adjacency)
        self._adjacency_cache["adjacency"] = adjacency
        self._cache_timestamp = datetime.now()
        
        return adjacency
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self.cache_timeout
    
    def _dijkstra(self, source: str, target: str, adjacency: Dict[str, List[str]], 
                  weight_property: Optional[str], max_depth: int) -> PathResult:
        """Dijkstra's shortest path algorithm."""
        distances = {source: 0.0}
        previous = {}
        pq = [(0.0, source)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target:
                # Reconstruct path
                path = []
                node = target
                while node is not None:
                    path.append(node)
                    node = previous.get(node)
                path.reverse()
                
                return PathResult(
                    path=path,
                    distance=current_dist,
                    exists=True,
                    algorithm_used="dijkstra"
                )
            
            if len(path_so_far := self._reconstruct_path(previous, source, current)) > max_depth:
                continue
            
            # Check neighbors
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                
                # Calculate edge weight
                edge_weight = self._get_edge_weight(current, neighbor, weight_property)
                new_distance = current_dist + edge_weight
                
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        # No path found
        return PathResult(path=[], distance=float('inf'), exists=False, algorithm_used="dijkstra")
    
    def _bfs(self, source: str, target: str, adjacency: Dict[str, List[str]], 
             max_depth: int) -> PathResult:
        """Breadth-first search for unweighted shortest path."""
        if source == target:
            return PathResult(path=[source], distance=0.0, exists=True, algorithm_used="bfs")
        
        queue = deque([(source, [source], 0)])
        visited = {source}
        
        while queue:
            current, path, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor == target:
                    return PathResult(
                        path=new_path,
                        distance=float(len(new_path) - 1),
                        exists=True,
                        algorithm_used="bfs"
                    )
                
                visited.add(neighbor)
                queue.append((neighbor, new_path, depth + 1))
        
        return PathResult(path=[], distance=float('inf'), exists=False, algorithm_used="bfs")
    
    def _a_star(self, source: str, target: str, adjacency: Dict[str, List[str]], 
                weight_property: Optional[str], max_depth: int) -> PathResult:
        """A* search algorithm (simplified heuristic)."""
        # For simplicity, fall back to Dijkstra since we don't have spatial coordinates
        # In a real implementation, you'd need a proper heuristic function
        return self._dijkstra(source, target, adjacency, weight_property, max_depth)
    
    def _get_edge_weight(self, source: str, target: str, weight_property: Optional[str]) -> float:
        """Get weight of edge between two nodes."""
        if not weight_property:
            return 1.0
        
        # Find relationship between source and target
        relationships = self.metadata_store.get_all_relationships()
        for rel in relationships:
            if (rel.get("source_entity_id") == source and 
                rel.get("target_entity_id") == target):
                properties = rel.get("properties", {})
                weight = properties.get(weight_property, 1.0)
                try:
                    return float(weight)
                except (ValueError, TypeError):
                    return 1.0
        
        return 1.0
    
    def _reconstruct_path(self, previous: Dict[str, str], source: str, current: str) -> List[str]:
        """Reconstruct path from previous pointers."""
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = previous.get(node)
        path.reverse()
        return path
    
    def _filter_entities_for_graph(self, entities: List[Dict[str, Any]], 
                                  entity_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter entities based on graph-specific criteria."""
        filtered = entities
        
        if "entity_types" in entity_filter:
            types = entity_filter["entity_types"]
            filtered = [e for e in filtered if e.get("entity_type") in types]
        
        if "min_connections" in entity_filter:
            min_conn = entity_filter["min_connections"]
            # This would require relationship analysis - simplified for now
            pass
        
        return filtered
    
    def _count_connected_components(self, adjacency: Dict[str, List[str]], nodes: List[str]) -> int:
        """Count connected components using DFS."""
        visited = set()
        components = 0
        
        for node in nodes:
            if node not in visited:
                components += 1
                self._dfs_mark_component(node, adjacency, visited)
        
        return components
    
    def _dfs_mark_component(self, node: str, adjacency: Dict[str, List[str]], visited: set):
        """Mark all nodes in a connected component."""
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                self._dfs_mark_component(neighbor, adjacency, visited)
    
    def _calculate_diameter(self, adjacency: Dict[str, List[str]], nodes: List[str]) -> Optional[int]:
        """Calculate graph diameter (longest shortest path)."""
        max_distance = 0
        node_sample = nodes[:100]  # Sample for performance
        
        for source in node_sample:
            distances = self._single_source_shortest_paths(source, adjacency)
            if distances:
                max_dist = max(distances.values())
                max_distance = max(max_distance, max_dist)
        
        return max_distance if max_distance > 0 else None
    
    def _single_source_shortest_paths(self, source: str, adjacency: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate shortest paths from a single source using BFS."""
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        return distances
    
    def _calculate_clustering_coefficient(self, adjacency: Dict[str, List[str]]) -> float:
        """Calculate average clustering coefficient."""
        total_clustering = 0.0
        nodes_with_neighbors = 0
        
        for node, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            
            neighbors_set = set(neighbors)
            possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
            actual_edges = 0
            
            # Count edges between neighbors
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if neighbor2 in adjacency.get(neighbor1, []):
                        actual_edges += 1
            
            if possible_edges > 0:
                clustering = actual_edges / possible_edges
                total_clustering += clustering
                nodes_with_neighbors += 1
        
        return total_clustering / nodes_with_neighbors if nodes_with_neighbors > 0 else 0.0
    
    def _build_traversal_tree(self, visit_order: List[str], parent_map: Dict[str, str], 
                             depth_map: Dict[str, int]) -> Dict[str, Any]:
        """Build hierarchical tree structure from traversal results."""
        tree = {"nodes": {}, "root_nodes": []}
        
        for node in visit_order:
            tree["nodes"][node] = {
                "id": node,
                "depth": depth_map.get(node, 0),
                "parent": parent_map.get(node),
                "children": []
            }
        
        # Build parent-child relationships
        for node, node_data in tree["nodes"].items():
            parent = node_data["parent"]
            if parent and parent in tree["nodes"]:
                tree["nodes"][parent]["children"].append(node)
            else:
                tree["root_nodes"].append(node)
        
        return tree
    
    def _calculate_depth_distribution(self, depth_map: Dict[str, int]) -> Dict[int, int]:
        """Calculate distribution of nodes by depth."""
        distribution = defaultdict(int)
        for depth in depth_map.values():
            distribution[depth] += 1
        return dict(distribution)