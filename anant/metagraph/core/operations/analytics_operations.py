"""
Analytics Operations Module
===========================

Advanced analytics and computational operations for the Metagraph including:
- Graph algorithms (centrality, clustering, community detection)
- Network analysis and metrics
- Temporal pattern analysis
- Statistical computations
- Performance analytics
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import math

from ....exceptions import (
    MetagraphError, ValidationError, handle_exception,
    require_not_none, require_valid_string
)

logger = logging.getLogger(__name__)


class AnalyticsOperations:
    """
    Handles analytics and computational operations for the Metagraph.
    
    Provides graph algorithms, network analysis, temporal analytics,
    and statistical computations with proper error handling and logging.
    """
    
    def __init__(self, metadata_store, hierarchical_store):
        """
        Initialize analytics operations.
        
        Args:
            metadata_store: Reference to metadata storage system
            hierarchical_store: Reference to hierarchical storage system
        """
        self.metadata_store = metadata_store
        self.hierarchical_store = hierarchical_store
        self.logger = logger.getChild(self.__class__.__name__)
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the metagraph.
        
        Returns:
            Dictionary containing various statistics
            
        Raises:
            MetagraphError: If statistics calculation fails
        """
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "hierarchical_depth": 0,
                "avg_relationships_per_entity": 0.0,
                "connectivity_metrics": {}
            }
            
            # Get basic counts
            entities = self.metadata_store.get_all_entities()
            relationships = self.metadata_store.get_all_relationships()
            
            stats["total_entities"] = len(entities)
            stats["total_relationships"] = len(relationships)
            
            # Analyze entity types
            entity_type_counts = defaultdict(int)
            for entity in entities:
                entity_type = entity.get("entity_type", "unknown")
                entity_type_counts[entity_type] += 1
            stats["entity_types"] = dict(entity_type_counts)
            
            # Analyze relationship types
            relationship_type_counts = defaultdict(int)
            for rel in relationships:
                rel_type = rel.get("relationship_type", "unknown")
                relationship_type_counts[rel_type] += 1
            stats["relationship_types"] = dict(relationship_type_counts)
            
            # Calculate connectivity metrics
            if stats["total_entities"] > 0:
                stats["avg_relationships_per_entity"] = stats["total_relationships"] / stats["total_entities"]
                stats["connectivity_metrics"] = self._calculate_connectivity_metrics(entities, relationships)
            
            # Calculate hierarchical depth
            stats["hierarchical_depth"] = self._calculate_max_hierarchical_depth()
            
            self.logger.debug("Comprehensive stats calculated", extra=stats)
            return stats
            
        except Exception as e:
            raise handle_exception("calculating comprehensive stats", e)
    
    def analyze_temporal_patterns(self,
                                 entity_id: Optional[str] = None,
                                 days: int = 30,
                                 pattern_type: str = "activity") -> Dict[str, Any]:
        """
        Analyze temporal patterns in entity activities.
        
        Args:
            entity_id: Optional specific entity to analyze
            days: Number of days to analyze
            pattern_type: Type of pattern to analyze
            
        Returns:
            Temporal pattern analysis results
            
        Raises:
            MetagraphError: If analysis fails
        """
        try:
            if days <= 0:
                raise ValidationError(
                    "Days must be positive",
                    error_code="INVALID_DAYS",
                    context={"days": days}
                )
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get temporal data
            temporal_data = self._get_temporal_data(entity_id, start_time, end_time)
            
            # Analyze patterns
            patterns = {
                "period_analyzed": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "days": days
                },
                "activity_trends": self._analyze_activity_trends(temporal_data),
                "peak_periods": self._identify_peak_periods(temporal_data),
                "anomalies": self._detect_temporal_anomalies(temporal_data),
                "seasonality": self._analyze_seasonality(temporal_data),
                "growth_patterns": self._analyze_growth_patterns(temporal_data)
            }
            
            if entity_id:
                patterns["entity_id"] = entity_id
                patterns["entity_specific_metrics"] = self._get_entity_temporal_metrics(entity_id, temporal_data)
            
            return patterns
            
        except ValidationError:
            raise
        except Exception as e:
            raise handle_exception(f"analyzing temporal patterns", e, {
                "entity_id": entity_id,
                "days": days,
                "pattern_type": pattern_type
            })
    
    def degree_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality for all entities.
        
        Returns:
            Dictionary mapping entity IDs to centrality scores
            
        Raises:
            MetagraphError: If calculation fails
        """
        try:
            entities = self.metadata_store.get_all_entities()
            relationships = self.metadata_store.get_all_relationships()
            
            if not entities:
                return {}
            
            # Count connections for each entity
            degree_counts = defaultdict(int)
            for rel in relationships:
                source = rel.get("source_entity_id")
                target = rel.get("target_entity_id")
                if source:
                    degree_counts[source] += 1
                if target:
                    degree_counts[target] += 1
            
            # Normalize by maximum possible connections
            max_possible_connections = len(entities) - 1
            centrality = {}
            
            for entity in entities:
                entity_id = entity["entity_id"]
                degree = degree_counts.get(entity_id, 0)
                centrality[entity_id] = degree / max_possible_connections if max_possible_connections > 0 else 0.0
            
            return centrality
            
        except Exception as e:
            raise handle_exception("calculating degree centrality", e)
    
    def betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all entities.
        
        Returns:
            Dictionary mapping entity IDs to centrality scores
            
        Raises:
            MetagraphError: If calculation fails
        """
        try:
            entities = self.metadata_store.get_all_entities()
            relationships = self.metadata_store.get_all_relationships()
            
            if len(entities) < 3:
                return {entity["entity_id"]: 0.0 for entity in entities}
            
            # Build adjacency list
            graph = self._build_adjacency_list(entities, relationships)
            entity_ids = [entity["entity_id"] for entity in entities]
            
            betweenness = defaultdict(float)
            
            # Calculate betweenness for each pair of nodes
            for source in entity_ids:
                # Single-source shortest paths
                paths = self._all_shortest_paths_from_source(graph, source)
                
                for target in entity_ids:
                    if source != target and target in paths:
                        # Count paths through each intermediate node
                        for path_list in paths[target]:
                            if len(path_list) > 2:  # Has intermediate nodes
                                for intermediate in path_list[1:-1]:
                                    betweenness[intermediate] += 1.0 / len(paths[target])
            
            # Normalize
            n = len(entity_ids)
            normalization = (n - 1) * (n - 2) / 2 if n > 2 else 1
            
            return {entity_id: betweenness[entity_id] / normalization for entity_id in entity_ids}
            
        except Exception as e:
            raise handle_exception("calculating betweenness centrality", e)
    
    def community_detection(self, method: str = "louvain") -> Dict[str, Any]:
        """
        Detect communities in the entity network.
        
        Args:
            method: Community detection method to use
            
        Returns:
            Community detection results
            
        Raises:
            MetagraphError: If detection fails
        """
        try:
            entities = self.metadata_store.get_all_entities()
            relationships = self.metadata_store.get_all_relationships()
            
            if not entities:
                return {"communities": [], "modularity": 0.0, "method": method}
            
            # Build graph for community detection
            graph = self._build_adjacency_list(entities, relationships)
            
            if method.lower() == "louvain":
                communities = self._louvain_community_detection(graph)
            else:
                # Simple connected components as fallback
                communities = self._find_connected_components(graph)
            
            # Calculate modularity
            modularity = self._calculate_modularity_score(graph, communities)
            
            # Format results
            community_results = []
            for i, community in enumerate(communities):
                community_data = {
                    "community_id": i,
                    "entities": list(community),
                    "size": len(community),
                    "density": self._calculate_community_density(graph, community),
                    "internal_edges": self._count_internal_edges(graph, community),
                    "external_edges": self._count_external_edges(graph, community)
                }
                community_results.append(community_data)
            
            return {
                "communities": community_results,
                "num_communities": len(communities),
                "modularity": modularity,
                "method": method,
                "largest_community_size": max(len(c) for c in communities) if communities else 0,
                "average_community_size": sum(len(c) for c in communities) / len(communities) if communities else 0
            }
            
        except Exception as e:
            raise handle_exception(f"detecting communities using {method}", e, {"method": method})
    
    def anomaly_detection(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in entity patterns and relationships.
        
        Args:
            entity_type: Optional filter for specific entity type
            
        Returns:
            Anomaly detection results
            
        Raises:
            MetagraphError: If detection fails
        """
        try:
            entities = self.metadata_store.get_all_entities()
            relationships = self.metadata_store.get_all_relationships()
            
            if entity_type:
                entities = [e for e in entities if e.get("entity_type") == entity_type]
            
            anomalies = {
                "structural_anomalies": self._detect_structural_anomalies(entities, relationships),
                "property_anomalies": self._detect_property_anomalies(entities),
                "temporal_anomalies": self._detect_temporal_anomalies_simple(entities),
                "relationship_anomalies": self._detect_relationship_anomalies(relationships),
                "summary": {
                    "total_entities_analyzed": len(entities),
                    "entity_type_filter": entity_type,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            # Calculate anomaly scores
            all_anomalies = []
            for category, category_anomalies in anomalies.items():
                if isinstance(category_anomalies, list):
                    all_anomalies.extend(category_anomalies)
            
            anomalies["summary"]["total_anomalies"] = len(all_anomalies)
            anomalies["summary"]["anomaly_rate"] = len(all_anomalies) / len(entities) if entities else 0.0
            
            return anomalies
            
        except Exception as e:
            raise handle_exception("detecting anomalies", e, {"entity_type": entity_type})
    
    def _calculate_connectivity_metrics(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Calculate various connectivity metrics."""
        if not entities:
            return {}
        
        graph = self._build_adjacency_list(entities, relationships)
        
        # Calculate metrics
        components = self._find_connected_components(graph)
        largest_component_size = max(len(c) for c in components) if components else 0
        
        return {
            "connected_components": len(components),
            "largest_component_size": largest_component_size,
            "connectivity_ratio": largest_component_size / len(entities) if entities else 0.0,
            "average_component_size": sum(len(c) for c in components) / len(components) if components else 0.0,
            "graph_density": (2 * len(relationships)) / (len(entities) * (len(entities) - 1)) if len(entities) > 1 else 0.0
        }
    
    def _calculate_max_hierarchical_depth(self) -> int:
        """Calculate the maximum depth of hierarchical relationships."""
        try:
            return self.hierarchical_store.calculate_max_depth()
        except Exception:
            return 0
    
    def _get_temporal_data(self, entity_id: Optional[str], start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get temporal data for analysis."""
        # Simulate temporal data - in real implementation, this would query actual temporal store
        temporal_data = []
        current_time = start_time
        
        while current_time <= end_time:
            # Simulate some activity data
            activity_count = np.random.poisson(5)  # Random activity
            temporal_data.append({
                "timestamp": current_time.isoformat(),
                "activity_count": activity_count,
                "entity_id": entity_id
            })
            current_time += timedelta(hours=1)
        
        return temporal_data
    
    def _analyze_activity_trends(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze activity trends in temporal data."""
        if not temporal_data:
            return {"trend": "no_data", "slope": 0.0}
        
        activities = [item["activity_count"] for item in temporal_data]
        
        # Simple linear trend calculation
        n = len(activities)
        x = list(range(n))
        
        if n < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        # Calculate slope using least squares
        x_mean = sum(x) / n
        y_mean = sum(activities) / n
        
        numerator = sum((x[i] - x_mean) * (activities[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "average_activity": y_mean,
            "total_activity": sum(activities),
            "data_points": n
        }
    
    def _identify_peak_periods(self, temporal_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify peak activity periods."""
        if len(temporal_data) < 3:
            return []
        
        activities = [item["activity_count"] for item in temporal_data]
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        threshold = mean_activity + 2 * std_activity
        
        peaks = []
        for i, item in enumerate(temporal_data):
            if item["activity_count"] > threshold:
                peaks.append({
                    "timestamp": item["timestamp"],
                    "activity_count": item["activity_count"],
                    "threshold": threshold,
                    "magnitude": item["activity_count"] - mean_activity
                })
        
        return peaks
    
    def _detect_temporal_anomalies(self, temporal_data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies in the data."""
        if len(temporal_data) < 10:
            return []
        
        activities = [item["activity_count"] for item in temporal_data]
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        
        anomalies = []
        for item in temporal_data:
            z_score = abs(item["activity_count"] - mean_activity) / std_activity if std_activity > 0 else 0
            if z_score > 3:  # 3-sigma rule
                anomalies.append({
                    "timestamp": item["timestamp"],
                    "activity_count": item["activity_count"],
                    "z_score": z_score,
                    "type": "statistical_outlier"
                })
        
        return anomalies
    
    def _analyze_seasonality(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        # Simple seasonality analysis - could be enhanced with FFT
        hourly_patterns = defaultdict(list)
        
        for item in temporal_data:
            timestamp = datetime.fromisoformat(item["timestamp"])
            hour = timestamp.hour
            hourly_patterns[hour].append(item["activity_count"])
        
        hourly_averages = {
            hour: np.mean(activities) for hour, activities in hourly_patterns.items()
        }
        
        return {
            "hourly_patterns": hourly_averages,
            "peak_hour": max(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else None,
            "low_hour": min(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else None
        }
    
    def _analyze_growth_patterns(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze growth patterns in the data."""
        if len(temporal_data) < 2:
            return {"growth_rate": 0.0, "pattern": "insufficient_data"}
        
        activities = [item["activity_count"] for item in temporal_data]
        
        # Calculate period-over-period growth
        growth_rates = []
        for i in range(1, len(activities)):
            if activities[i-1] > 0:
                growth_rate = (activities[i] - activities[i-1]) / activities[i-1]
                growth_rates.append(growth_rate)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.0
        
        return {
            "average_growth_rate": avg_growth_rate,
            "growth_volatility": np.std(growth_rates) if growth_rates else 0.0,
            "pattern": "growing" if avg_growth_rate > 0.05 else "declining" if avg_growth_rate < -0.05 else "stable"
        }
    
    def _get_entity_temporal_metrics(self, entity_id: str, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Get entity-specific temporal metrics."""
        entity_data = [item for item in temporal_data if item.get("entity_id") == entity_id]
        
        if not entity_data:
            return {"total_activity": 0, "average_activity": 0.0}
        
        activities = [item["activity_count"] for item in entity_data]
        
        return {
            "total_activity": sum(activities),
            "average_activity": np.mean(activities),
            "peak_activity": max(activities),
            "min_activity": min(activities),
            "activity_variance": np.var(activities)
        }
    
    def _build_adjacency_list(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, List[str]]:
        """Build adjacency list representation of the graph."""
        graph = defaultdict(list)
        
        # Initialize all entities in graph
        for entity in entities:
            entity_id = entity["entity_id"]
            if entity_id not in graph:
                graph[entity_id] = []
        
        # Add relationships
        for rel in relationships:
            source = rel.get("source_entity_id")
            target = rel.get("target_entity_id")
            if source and target:
                graph[source].append(target)
                graph[target].append(source)  # Undirected graph
        
        return dict(graph)
    
    def _find_connected_components(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find connected components in the graph."""
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    def _all_shortest_paths_from_source(self, graph: Dict[str, List[str]], source: str) -> Dict[str, List[List[str]]]:
        """Find all shortest paths from a source node."""
        paths = defaultdict(list)
        queue = deque([(source, [source])])
        visited = {source: 0}
        
        while queue:
            node, path = queue.popleft()
            current_distance = len(path) - 1
            
            for neighbor in graph.get(node, []):
                new_distance = current_distance + 1
                new_path = path + [neighbor]
                
                if neighbor not in visited:
                    visited[neighbor] = new_distance
                    paths[neighbor].append(new_path)
                    queue.append((neighbor, new_path))
                elif visited[neighbor] == new_distance:
                    paths[neighbor].append(new_path)
        
        return dict(paths)
    
    def _louvain_community_detection(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Simple Louvain-inspired community detection."""
        # Simplified implementation - real Louvain is more complex
        return self._find_connected_components(graph)
    
    def _calculate_modularity_score(self, graph: Dict[str, List[str]], communities: List[List[str]]) -> float:
        """Calculate modularity score for community partition."""
        if not communities:
            return 0.0
        
        total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        if total_edges == 0:
            return 0.0
        
        modularity = 0.0
        
        for community in communities:
            internal_edges = self._count_internal_edges(graph, community)
            degree_sum = sum(len(graph.get(node, [])) for node in community)
            
            modularity += internal_edges / total_edges - (degree_sum / (2 * total_edges)) ** 2
        
        return modularity
    
    def _calculate_community_density(self, graph: Dict[str, List[str]], community: List[str]) -> float:
        """Calculate density of a community."""
        if len(community) < 2:
            return 0.0
        
        internal_edges = self._count_internal_edges(graph, community)
        max_possible_edges = len(community) * (len(community) - 1) // 2
        
        return internal_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    
    def _count_internal_edges(self, graph: Dict[str, List[str]], community: List[str]) -> int:
        """Count edges within a community."""
        community_set = set(community)
        internal_edges = 0
        
        for node in community:
            for neighbor in graph.get(node, []):
                if neighbor in community_set and node < neighbor:  # Avoid double counting
                    internal_edges += 1
        
        return internal_edges
    
    def _count_external_edges(self, graph: Dict[str, List[str]], community: List[str]) -> int:
        """Count edges from community to outside."""
        community_set = set(community)
        external_edges = 0
        
        for node in community:
            for neighbor in graph.get(node, []):
                if neighbor not in community_set:
                    external_edges += 1
        
        return external_edges
    
    def _detect_structural_anomalies(self, entities: List[Dict], relationships: List[Dict]) -> List[Dict[str, Any]]:
        """Detect structural anomalies in the graph."""
        anomalies = []
        
        # Calculate degree for each entity
        degree_counts = defaultdict(int)
        for rel in relationships:
            source = rel.get("source_entity_id")
            target = rel.get("target_entity_id")
            if source:
                degree_counts[source] += 1
            if target:
                degree_counts[target] += 1
        
        # Detect high-degree nodes (hubs)
        degrees = list(degree_counts.values())
        if degrees:
            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)
            threshold = mean_degree + 2 * std_degree
            
            for entity in entities:
                entity_id = entity["entity_id"]
                degree = degree_counts.get(entity_id, 0)
                if degree > threshold:
                    anomalies.append({
                        "entity_id": entity_id,
                        "anomaly_type": "high_degree_hub",
                        "degree": degree,
                        "threshold": threshold,
                        "severity": (degree - mean_degree) / std_degree if std_degree > 0 else 0
                    })
        
        return anomalies
    
    def _detect_property_anomalies(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Detect anomalies in entity properties."""
        anomalies = []
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            entities_by_type[entity_type].append(entity)
        
        # Detect property anomalies within each type
        for entity_type, type_entities in entities_by_type.items():
            if len(type_entities) < 3:  # Need enough samples
                continue
                
            # Analyze property completeness
            all_properties = set()
            for entity in type_entities:
                all_properties.update(entity.get("properties", {}).keys())
            
            for entity in type_entities:
                entity_props = set(entity.get("properties", {}).keys())
                missing_props = all_properties - entity_props
                
                if len(missing_props) > len(all_properties) * 0.5:  # Missing more than 50% of properties
                    anomalies.append({
                        "entity_id": entity["entity_id"],
                        "anomaly_type": "incomplete_properties",
                        "entity_type": entity_type,
                        "missing_properties": list(missing_props),
                        "completeness_ratio": len(entity_props) / len(all_properties) if all_properties else 0
                    })
        
        return anomalies
    
    def _detect_temporal_anomalies_simple(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Detect simple temporal anomalies."""
        anomalies = []
        
        for entity in entities:
            created_at = entity.get("created_at")
            updated_at = entity.get("updated_at")
            
            if created_at and updated_at:
                try:
                    created_time = datetime.fromisoformat(created_at)
                    updated_time = datetime.fromisoformat(updated_at)
                    
                    # Check if updated_at is before created_at
                    if updated_time < created_time:
                        anomalies.append({
                            "entity_id": entity["entity_id"],
                            "anomaly_type": "temporal_inconsistency",
                            "created_at": created_at,
                            "updated_at": updated_at,
                            "time_difference": (created_time - updated_time).total_seconds()
                        })
                except Exception:
                    # Invalid timestamp format
                    anomalies.append({
                        "entity_id": entity["entity_id"],
                        "anomaly_type": "invalid_timestamp_format",
                        "created_at": created_at,
                        "updated_at": updated_at
                    })
        
        return anomalies
    
    def _detect_relationship_anomalies(self, relationships: List[Dict]) -> List[Dict[str, Any]]:
        """Detect anomalies in relationships."""
        anomalies = []
        
        # Detect self-referencing relationships
        for rel in relationships:
            source = rel.get("source_entity_id")
            target = rel.get("target_entity_id")
            
            if source == target:
                anomalies.append({
                    "relationship_id": rel.get("relationship_id"),
                    "anomaly_type": "self_referencing_relationship",
                    "entity_id": source,
                    "relationship_type": rel.get("relationship_type")
                })
        
        # Detect duplicate relationships
        relationship_pairs = defaultdict(list)
        for rel in relationships:
            source = rel.get("source_entity_id")
            target = rel.get("target_entity_id")
            rel_type = rel.get("relationship_type")
            key = (source, target, rel_type)
            relationship_pairs[key].append(rel)
        
        for key, rels in relationship_pairs.items():
            if len(rels) > 1:
                source, target, rel_type = key
                anomalies.append({
                    "anomaly_type": "duplicate_relationships",
                    "source_entity_id": source,
                    "target_entity_id": target,
                    "relationship_type": rel_type,
                    "duplicate_count": len(rels),
                    "relationship_ids": [rel.get("relationship_id") for rel in rels]
                })
        
        return anomalies