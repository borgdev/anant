"""
Metrics Operations for Hierarchical Knowledge Graph
=================================================

This module handles all metrics and statistical analysis operations for hierarchical
knowledge graphs, including performance monitoring, balance metrics, anomaly detection,
and comprehensive health assessments.

Key Features:
- Comprehensive statistics and metrics collection
- Hierarchical balance and stability analysis  
- Anomaly detection and outlier identification
- Performance monitoring and benchmarking
- Health assessment and quality metrics
- Temporal trend analysis and change detection
- Comparative analysis between levels and time periods
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set, Sequence
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import math
import logging
import time

logger = logging.getLogger(__name__)


class MetricsOperations:
    """
    Handles metrics and statistical analysis for hierarchical knowledge graphs.
    
    This class provides comprehensive metrics collection and analysis capabilities
    that leverage the hierarchical structure to provide multi-level insights
    and detect patterns, anomalies, and performance issues.
    
    Features:
    - Multi-level statistical analysis
    - Balance and distribution metrics
    - Anomaly detection and health monitoring
    - Performance benchmarking
    - Temporal trend analysis
    """
    
    def __init__(self, hierarchical_kg):
        """
        Initialize metrics operations.
        
        Args:
            hierarchical_kg: Reference to parent HierarchicalKnowledgeGraph instance
        """
        self.hkg = hierarchical_kg
        self.metric_history = []  # Store historical metrics for trend analysis
    
    # =====================================================================
    # COMPREHENSIVE STATISTICS
    # =====================================================================
    
    def get_comprehensive_statistics(self, include_historical: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the entire hierarchical knowledge graph.
        
        Args:
            include_historical: Include historical trend data
            
        Returns:
            Dictionary containing all statistical metrics
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': self._calculate_basic_metrics(),
            'hierarchical_metrics': self._calculate_hierarchical_metrics(),
            'connectivity_metrics': self._calculate_connectivity_metrics(),
            'distribution_metrics': self._calculate_distribution_metrics(),
            'balance_metrics': self._calculate_balance_metrics(),
            'quality_metrics': self._calculate_quality_metrics()
        }
        
        if include_historical and len(self.metric_history) > 1:
            stats['trend_analysis'] = self._analyze_trends()
        
        # Store in history for trend analysis
        self._store_metrics_snapshot(stats)
        
        return stats
    
    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic graph metrics."""
        return {
            'total_entities': self.hkg.num_nodes(),
            'total_relationships': self.hkg.num_edges(),
            'total_levels': len(self.hkg.levels),
            'cross_level_relationships': len(self.hkg.cross_level_relationships),
            'entities_per_level': {
                level_id: len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
                for level_id in self.hkg.levels.keys()
            },
            'avg_entities_per_level': (
                self.hkg.num_nodes() / len(self.hkg.levels) 
                if len(self.hkg.levels) > 0 else 0
            ),
            'level_utilization': (
                len([level for level in self.hkg.levels.keys() 
                    if len(self.hkg.hierarchy_ops.get_entities_at_level(level)) > 0]) / 
                len(self.hkg.levels) if len(self.hkg.levels) > 0 else 0
            )
        }
    
    def _calculate_hierarchical_metrics(self) -> Dict[str, Any]:
        """Calculate hierarchy-specific metrics."""
        level_depths = list(self.hkg.level_order.values()) if self.hkg.level_order else [0]
        
        # Calculate cross-level relationship patterns
        cross_level_patterns = defaultdict(int)
        for rel in self.hkg.cross_level_relationships:
            source_level = rel.get('source_level', 'unknown')
            target_level = rel.get('target_level', 'unknown')
            source_order = self.hkg.level_order.get(source_level, 0)
            target_order = self.hkg.level_order.get(target_level, 0)
            
            if source_order < target_order:
                cross_level_patterns['downward'] += 1
            elif source_order > target_order:
                cross_level_patterns['upward'] += 1
            else:
                cross_level_patterns['lateral'] += 1
        
        return {
            'hierarchy_depth': max(level_depths) - min(level_depths) + 1 if level_depths else 0,
            'max_level_order': max(level_depths) if level_depths else 0,
            'min_level_order': min(level_depths) if level_depths else 0,
            'cross_level_patterns': dict(cross_level_patterns),
            'cross_level_density': (
                len(self.hkg.cross_level_relationships) / 
                (self.hkg.num_nodes() * (len(self.hkg.levels) - 1))
                if self.hkg.num_nodes() > 0 and len(self.hkg.levels) > 1 else 0
            ),
            'level_connectivity': self._calculate_level_connectivity()
        }
    
    def _calculate_connectivity_metrics(self) -> Dict[str, Any]:
        """Calculate connectivity and network structure metrics."""
        connectivity_metrics = {}
        
        # Analyze connectivity for each level
        for level_id in self.hkg.levels.keys():
            entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            
            if len(entities) < 2:
                continue
            
            # Calculate internal connectivity
            internal_edges = 0
            total_possible_edges = len(entities) * (len(entities) - 1) // 2
            
            # Count edges within this level
            all_relationships = self.hkg.knowledge_graph.get_all_relationships()
            for rel in all_relationships:
                source = rel.get('source_entity')
                target = rel.get('target_entity')
                
                if (source in entities and target in entities and 
                    self.hkg.entity_levels.get(source) == level_id and
                    self.hkg.entity_levels.get(target) == level_id):
                    internal_edges += 1
            
            level_density = internal_edges / total_possible_edges if total_possible_edges > 0 else 0
            
            connectivity_metrics[level_id] = {
                'internal_edges': internal_edges,
                'possible_edges': total_possible_edges,
                'density': level_density,
                'entity_count': len(entities)
            }
        
        # Overall connectivity metrics
        total_internal_edges = sum(m['internal_edges'] for m in connectivity_metrics.values())
        total_possible_edges = sum(m['possible_edges'] for m in connectivity_metrics.values())
        
        return {
            'level_connectivity': connectivity_metrics,
            'overall_internal_density': (
                total_internal_edges / total_possible_edges 
                if total_possible_edges > 0 else 0
            ),
            'cross_to_internal_ratio': (
                len(self.hkg.cross_level_relationships) / total_internal_edges
                if total_internal_edges > 0 else float('inf')
            )
        }
    
    def _calculate_distribution_metrics(self) -> Dict[str, Any]:
        """Calculate distribution and variance metrics."""
        entities_per_level = [
            len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
            for level_id in self.hkg.levels.keys()
        ]
        
        if not entities_per_level:
            return {}
        
        # Entity type distribution
        entity_types = Counter()
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            entity_type = entity_data.get('type', 'unknown') if entity_data else 'unknown'
            entity_types[entity_type] += 1
        
        return {
            'entities_per_level': {
                'values': entities_per_level,
                'mean': statistics.mean(entities_per_level),
                'median': statistics.median(entities_per_level),
                'std_dev': statistics.stdev(entities_per_level) if len(entities_per_level) > 1 else 0,
                'min': min(entities_per_level),
                'max': max(entities_per_level),
                'range': max(entities_per_level) - min(entities_per_level),
                'coefficient_of_variation': (
                    statistics.stdev(entities_per_level) / statistics.mean(entities_per_level)
                    if len(entities_per_level) > 1 and statistics.mean(entities_per_level) > 0 else 0
                )
            },
            'entity_type_distribution': dict(entity_types),
            'entity_type_diversity': len(entity_types),
            'dominant_entity_type': entity_types.most_common(1)[0] if entity_types else None
        }
    
    def _calculate_balance_metrics(self) -> Dict[str, Any]:
        """Calculate balance and stability metrics."""
        # Level balance (how evenly distributed entities are across levels)
        entities_per_level = [
            len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
            for level_id in self.hkg.levels.keys()
        ]
        
        if not entities_per_level:
            return {'level_balance': 0, 'hierarchy_stability': 0}
        
        # Calculate Gini coefficient for level balance
        gini_coefficient = self._calculate_gini_coefficient(entities_per_level)
        
        # Cross-level relationship balance
        cross_level_counts = defaultdict(int)
        for rel in self.hkg.cross_level_relationships:
            source_level = rel.get('source_level', 'unknown')
            target_level = rel.get('target_level', 'unknown')
            cross_level_counts[(source_level, target_level)] += 1
        
        cross_level_balance = (
            self._calculate_gini_coefficient(list(cross_level_counts.values()))
            if cross_level_counts else 0
        )
        
        # Hierarchy stability (measure of structural consistency)
        stability_score = self._calculate_hierarchy_stability()
        
        return {
            'level_balance': 1 - gini_coefficient,  # Higher is more balanced
            'cross_level_balance': 1 - cross_level_balance,
            'hierarchy_stability': stability_score,
            'structural_consistency': self._calculate_structural_consistency(),
            'level_size_variance': statistics.variance(entities_per_level) if len(entities_per_level) > 1 else 0
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality and health metrics."""
        quality_metrics = {
            'completeness': self._calculate_completeness(),
            'consistency': self._calculate_consistency(),
            'connectivity_quality': self._calculate_connectivity_quality(),
            'metadata_richness': self._calculate_metadata_richness(),
            'relationship_quality': self._calculate_relationship_quality()
        }
        
        # Overall quality score (weighted average)
        weights = {
            'completeness': 0.25,
            'consistency': 0.25,
            'connectivity_quality': 0.20,
            'metadata_richness': 0.15,
            'relationship_quality': 0.15
        }
        
        overall_quality = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        quality_metrics['overall_quality'] = overall_quality
        quality_metrics['quality_grade'] = self._get_quality_grade(overall_quality)
        
        return quality_metrics
    
    # =====================================================================
    # ANOMALY DETECTION
    # =====================================================================
    
    def detect_anomalies(self, 
                        sensitivity: float = 2.0,
                        include_statistical: bool = True,
                        include_structural: bool = True) -> Dict[str, Any]:
        """
        Detect anomalies in the hierarchical knowledge graph.
        
        Args:
            sensitivity: Standard deviations for outlier detection
            include_statistical: Include statistical anomalies
            include_structural: Include structural anomalies
            
        Returns:
            Dictionary of detected anomalies by category
        """
        anomalies = {
            'statistical_anomalies': [],
            'structural_anomalies': [],
            'relationship_anomalies': [],
            'level_anomalies': []
        }
        
        if include_statistical:
            anomalies['statistical_anomalies'] = self._detect_statistical_anomalies(sensitivity)
        
        if include_structural:
            anomalies['structural_anomalies'] = self._detect_structural_anomalies()
            anomalies['relationship_anomalies'] = self._detect_relationship_anomalies()
            anomalies['level_anomalies'] = self._detect_level_anomalies()
        
        # Add summary
        total_anomalies = sum(len(category) for category in anomalies.values())
        anomalies['summary'] = {
            'total_anomalies': total_anomalies,
            'severity_distribution': self._categorize_anomaly_severity(anomalies),
            'detection_timestamp': datetime.now().isoformat()
        }
        
        return anomalies
    
    def _detect_statistical_anomalies(self, sensitivity: float) -> List[Dict[str, Any]]:
        """Detect statistical outliers in entity and relationship distributions."""
        anomalies = []
        
        # Entity count per level anomalies
        entities_per_level = [
            len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
            for level_id in self.hkg.levels.keys()
        ]
        
        if len(entities_per_level) > 2:
            mean_entities = statistics.mean(entities_per_level)
            std_entities = statistics.stdev(entities_per_level)
            
            for level_id in self.hkg.levels.keys():
                entity_count = len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
                z_score = abs(entity_count - mean_entities) / std_entities if std_entities > 0 else 0
                
                if z_score > sensitivity:
                    anomalies.append({
                        'type': 'entity_count_outlier',
                        'level_id': level_id,
                        'entity_count': entity_count,
                        'z_score': z_score,
                        'severity': 'high' if z_score > sensitivity * 1.5 else 'medium',
                        'description': f"Level {level_id} has unusually {'high' if entity_count > mean_entities else 'low'} entity count"
                    })
        
        return anomalies
    
    def _detect_structural_anomalies(self) -> List[Dict[str, Any]]:
        """Detect structural anomalies in the hierarchy."""
        anomalies = []
        
        # Isolated entities (no relationships)
        for entity_id in self.hkg.nodes():
            has_relationships = False
            
            # Check regular relationships
            all_relationships = self.hkg.knowledge_graph.get_all_relationships()
            for rel in all_relationships:
                if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                    has_relationships = True
                    break
            
            # Check cross-level relationships
            if not has_relationships:
                for rel in self.hkg.cross_level_relationships:
                    if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                        has_relationships = True
                        break
            
            if not has_relationships:
                anomalies.append({
                    'type': 'isolated_entity',
                    'entity_id': entity_id,
                    'level': self.hkg.entity_levels.get(entity_id, 'unknown'),
                    'severity': 'medium',
                    'description': f"Entity {entity_id} has no relationships"
                })
        
        # Empty levels
        for level_id in self.hkg.levels.keys():
            entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            if len(entities) == 0:
                anomalies.append({
                    'type': 'empty_level',
                    'level_id': level_id,
                    'severity': 'low',
                    'description': f"Level {level_id} has no entities"
                })
        
        return anomalies
    
    def _detect_relationship_anomalies(self) -> List[Dict[str, Any]]:
        """Detect relationship-related anomalies."""
        anomalies = []
        
        # Entities with unusually high degree
        degree_counts = []
        entity_degrees = {}
        
        for entity_id in self.hkg.nodes():
            degree = 0
            
            # Count regular relationships
            all_relationships = self.hkg.knowledge_graph.get_all_relationships()
            for rel in all_relationships:
                if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                    degree += 1
            
            # Count cross-level relationships
            for rel in self.hkg.cross_level_relationships:
                if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                    degree += 1
            
            degree_counts.append(degree)
            entity_degrees[entity_id] = degree
        
        if len(degree_counts) > 2:
            mean_degree = statistics.mean(degree_counts)
            std_degree = statistics.stdev(degree_counts)
            
            for entity_id, degree in entity_degrees.items():
                if std_degree > 0:
                    z_score = abs(degree - mean_degree) / std_degree
                    if z_score > 2.0:  # High degree entities
                        anomalies.append({
                            'type': 'high_degree_entity',
                            'entity_id': entity_id,
                            'degree': degree,
                            'z_score': z_score,
                            'severity': 'medium',
                            'description': f"Entity {entity_id} has unusually high connectivity ({degree} relationships)"
                        })
        
        return anomalies
    
    def _detect_level_anomalies(self) -> List[Dict[str, Any]]:
        """Detect level-specific anomalies."""
        anomalies = []
        
        # Levels with no cross-level connections
        levels_with_cross_connections = set()
        for rel in self.hkg.cross_level_relationships:
            levels_with_cross_connections.add(rel.get('source_level'))
            levels_with_cross_connections.add(rel.get('target_level'))
        
        for level_id in self.hkg.levels.keys():
            entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            
            if len(entities) > 0 and level_id not in levels_with_cross_connections:
                anomalies.append({
                    'type': 'isolated_level',
                    'level_id': level_id,
                    'entity_count': len(entities),
                    'severity': 'high',
                    'description': f"Level {level_id} has no cross-level connections"
                })
        
        return anomalies
    
    # =====================================================================
    # PERFORMANCE MONITORING
    # =====================================================================
    
    def benchmark_operations(self, 
                            operations_to_test: Optional[List[str]] = None,
                            iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark performance of key operations.
        
        Args:
            operations_to_test: List of operations to benchmark
            iterations: Number of iterations for each test
            
        Returns:
            Performance benchmark results
        """
        if operations_to_test is None:
            operations_to_test = [
                'add_entity', 'search', 'navigation', 'centrality', 'export'
            ]
        
        benchmark_results = {}
        
        for operation in operations_to_test:
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                
                if operation == 'add_entity':
                    # Test entity addition
                    test_entity_id = f"bench_entity_{time.time()}"
                    self.hkg.add_entity(test_entity_id, {'type': 'test', 'benchmark': True})
                    self.hkg.remove_entity(test_entity_id)  # Clean up
                
                elif operation == 'search':
                    # Test search operation
                    if hasattr(self.hkg, 'search_ops'):
                        self.hkg.search_ops.semantic_search("test", max_results=5)
                
                elif operation == 'navigation':
                    # Test navigation
                    entities = list(self.hkg.nodes())
                    if entities and hasattr(self.hkg, 'navigation_ops'):
                        self.hkg.navigation_ops.get_children(entities[0])
                
                elif operation == 'centrality':
                    # Test centrality calculation
                    if hasattr(self.hkg, 'analysis_ops'):
                        level_ids = list(self.hkg.levels.keys())[:1]  # Test one level
                        if level_ids:
                            self.hkg.analysis_ops.calculate_hierarchical_centrality(
                                'degree', level_ids
                            )
                
                elif operation == 'export':
                    # Test JSON export
                    if hasattr(self.hkg, 'io_ops'):
                        self.hkg.io_ops.to_json()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            if times:
                benchmark_results[operation] = {
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                    'iterations': iterations
                }
        
        return benchmark_results
    
    def get_performance_health(self) -> Dict[str, Any]:
        """Get overall performance health assessment."""
        health_metrics = {
            'memory_efficiency': self._estimate_memory_efficiency(),
            'structural_efficiency': self._calculate_structural_efficiency(),
            'query_complexity': self._estimate_query_complexity(),
            'scalability_indicators': self._get_scalability_indicators()
        }
        
        # Calculate overall health score
        scores = [v for v in health_metrics.values() if isinstance(v, (int, float))]
        overall_health = statistics.mean(scores) if scores else 0
        
        health_metrics['overall_health'] = overall_health
        health_metrics['health_grade'] = self._get_performance_grade(overall_health)
        
        return health_metrics
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def _calculate_level_connectivity(self) -> Dict[str, float]:
        """Calculate connectivity scores for each level."""
        connectivity = {}
        
        for level_id in self.hkg.levels.keys():
            entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            
            # Count cross-level connections from this level
            cross_connections = 0
            for rel in self.hkg.cross_level_relationships:
                if (rel.get('source_level') == level_id or 
                    rel.get('target_level') == level_id):
                    cross_connections += 1
            
            # Normalize by entity count
            connectivity[level_id] = (
                cross_connections / len(entities) 
                if len(entities) > 0 else 0
            )
        
        return connectivity
    
    def _calculate_gini_coefficient(self, values: Sequence[Union[int, float]]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or len(values) < 2:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum(sorted_values)
        
        if cumsum == 0:
            return 0.0
        
        # Calculate Gini coefficient
        gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * cumsum) - (n + 1) / n
        return gini
    
    def _calculate_hierarchy_stability(self) -> float:
        """Calculate hierarchical stability score."""
        if len(self.hkg.levels) <= 1:
            return 1.0
        
        # Check for consistent level ordering
        level_orders = list(self.hkg.level_order.values())
        if len(set(level_orders)) != len(level_orders):
            return 0.0  # Duplicate orders indicate instability
        
        # Check for reasonable cross-level relationship patterns
        upward_relationships = 0
        downward_relationships = 0
        
        for rel in self.hkg.cross_level_relationships:
            source_order = self.hkg.level_order.get(rel.get('source_level', ''), 0)
            target_order = self.hkg.level_order.get(rel.get('target_level', ''), 0)
            
            if source_order < target_order:
                downward_relationships += 1
            elif source_order > target_order:
                upward_relationships += 1
        
        total_cross_relationships = len(self.hkg.cross_level_relationships)
        if total_cross_relationships == 0:
            return 0.5  # Neutral stability for no cross-relationships
        
        # Prefer more downward relationships (hierarchical flow)
        downward_ratio = downward_relationships / total_cross_relationships
        return min(1.0, downward_ratio * 2)  # Scale to [0, 1]
    
    def _calculate_structural_consistency(self) -> float:
        """Calculate structural consistency score."""
        consistency_factors = []
        
        # Check entity level assignment consistency
        assigned_entities = len([e for e in self.hkg.nodes() if e in self.hkg.entity_levels])
        total_entities = self.hkg.num_nodes()
        assignment_consistency = assigned_entities / total_entities if total_entities > 0 else 1.0
        consistency_factors.append(assignment_consistency)
        
        # Check relationship validity
        valid_relationships = 0
        total_relationships = len(self.hkg.cross_level_relationships)
        
        for rel in self.hkg.cross_level_relationships:
            source_exists = self.hkg.knowledge_graph.has_entity(rel.get('source_entity', ''))
            target_exists = self.hkg.knowledge_graph.has_entity(rel.get('target_entity', ''))
            
            if source_exists and target_exists:
                valid_relationships += 1
        
        relationship_consistency = (
            valid_relationships / total_relationships 
            if total_relationships > 0 else 1.0
        )
        consistency_factors.append(relationship_consistency)
        
        return statistics.mean(consistency_factors) if consistency_factors else 0.0
    
    def _calculate_completeness(self) -> float:
        """Calculate data completeness score."""
        completeness_factors = []
        
        # Entity metadata completeness
        entities_with_metadata = 0
        total_entities = self.hkg.num_nodes()
        
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if entity_data and entity_data.get('properties'):
                entities_with_metadata += 1
        
        metadata_completeness = entities_with_metadata / total_entities if total_entities > 0 else 1.0
        completeness_factors.append(metadata_completeness)
        
        # Level assignment completeness
        assigned_entities = len(self.hkg.entity_levels)
        assignment_completeness = assigned_entities / total_entities if total_entities > 0 else 1.0
        completeness_factors.append(assignment_completeness)
        
        return statistics.mean(completeness_factors) if completeness_factors else 0.0
    
    def _calculate_consistency(self) -> float:
        """Calculate data consistency score."""
        # Use cross-level validation results
        if hasattr(self.hkg, 'cross_level_ops'):
            validation_results = self.hkg.cross_level_ops.validate_cross_level_consistency()
            return 1.0 if validation_results.get('is_consistent', False) else 0.5
        
        return self._calculate_structural_consistency()
    
    def _calculate_connectivity_quality(self) -> float:
        """Calculate connectivity quality score."""
        if self.hkg.num_nodes() <= 1:
            return 1.0
        
        # Measure how well connected the graph is
        connected_entities = 0
        
        for entity_id in self.hkg.nodes():
            has_connections = False
            
            # Check for any relationships
            all_relationships = self.hkg.knowledge_graph.get_all_relationships()
            for rel in all_relationships:
                if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                    has_connections = True
                    break
            
            if not has_connections:
                for rel in self.hkg.cross_level_relationships:
                    if rel.get('source_entity') == entity_id or rel.get('target_entity') == entity_id:
                        has_connections = True
                        break
            
            if has_connections:
                connected_entities += 1
        
        return connected_entities / self.hkg.num_nodes()
    
    def _calculate_metadata_richness(self) -> float:
        """Calculate metadata richness score."""
        if self.hkg.num_nodes() == 0:
            return 1.0
        
        total_properties = 0
        entity_count = 0
        
        for entity_id in self.hkg.nodes():
            entity_data = self.hkg.knowledge_graph.get_entity(entity_id)
            if entity_data:
                properties = entity_data.get('properties', {})
                total_properties += len(properties)
                entity_count += 1
        
        avg_properties_per_entity = total_properties / entity_count if entity_count > 0 else 0
        
        # Normalize to [0, 1] assuming 5 properties per entity is "rich"
        return min(1.0, avg_properties_per_entity / 5.0)
    
    def _calculate_relationship_quality(self) -> float:
        """Calculate relationship quality score."""
        total_relationships = self.hkg.num_edges() + len(self.hkg.cross_level_relationships)
        
        if total_relationships == 0:
            return 0.0 if self.hkg.num_nodes() > 1 else 1.0
        
        # Check for relationship type diversity
        relationship_types = set()
        
        all_relationships = self.hkg.knowledge_graph.get_all_relationships()
        for rel in all_relationships:
            relationship_types.add(rel.get('relationship_type', ''))
        
        for rel in self.hkg.cross_level_relationships:
            relationship_types.add(rel.get('relationship_type', ''))
        
        # More diverse relationship types indicate higher quality
        diversity_score = min(1.0, len(relationship_types) / 5.0)  # Normalize assuming 5 types is diverse
        
        return diversity_score
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to descriptive grade."""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Critical'
    
    def _estimate_memory_efficiency(self) -> float:
        """Estimate memory efficiency (simplified)."""
        # This is a simplified estimation
        entity_count = self.hkg.num_nodes()
        relationship_count = self.hkg.num_edges() + len(self.hkg.cross_level_relationships)
        
        if entity_count == 0:
            return 1.0
        
        # Assume efficient graphs have reasonable relationship-to-entity ratios
        relationship_ratio = relationship_count / entity_count
        
        # Optimal ratio is around 2-4 relationships per entity
        if 2 <= relationship_ratio <= 4:
            return 1.0
        elif relationship_ratio < 2:
            return 0.7  # Sparse graph
        else:
            return max(0.1, 1.0 / (relationship_ratio / 4))  # Dense graph penalty
    
    def _calculate_structural_efficiency(self) -> float:
        """Calculate structural efficiency score."""
        # Measure how efficiently the hierarchy is structured
        level_count = len(self.hkg.levels)
        entity_count = self.hkg.num_nodes()
        
        if level_count == 0 or entity_count == 0:
            return 1.0
        
        # Efficient hierarchies have balanced level distribution
        entities_per_level = [
            len(self.hkg.hierarchy_ops.get_entities_at_level(level_id))
            for level_id in self.hkg.levels.keys()
        ]
        
        if not entities_per_level:
            return 0.0
        
        # Calculate balance using coefficient of variation
        mean_entities = statistics.mean(entities_per_level)
        if mean_entities == 0:
            return 0.0
        
        std_entities = statistics.stdev(entities_per_level) if len(entities_per_level) > 1 else 0
        cv = std_entities / mean_entities
        
        # Lower coefficient of variation indicates better balance
        return max(0.0, 1.0 - cv)
    
    def _estimate_query_complexity(self) -> float:
        """Estimate average query complexity."""
        # Simplified complexity estimation based on graph structure
        entity_count = self.hkg.num_nodes()
        level_count = len(self.hkg.levels)
        
        if entity_count == 0:
            return 1.0
        
        # Complexity increases with size, but hierarchies help
        base_complexity = math.log(entity_count + 1, 2)  # Logarithmic base
        hierarchy_benefit = 1.0 / (level_count + 1)  # More levels = better organization
        
        complexity_score = max(0.0, 1.0 - (base_complexity * hierarchy_benefit) / 10)
        return complexity_score
    
    def _get_scalability_indicators(self) -> Dict[str, Any]:
        """Get indicators of system scalability."""
        return {
            'current_size': self.hkg.num_nodes(),
            'growth_capacity': self._estimate_growth_capacity(),
            'bottleneck_indicators': self._identify_bottlenecks(),
            'recommended_max_size': self._estimate_recommended_max_size()
        }
    
    def _estimate_growth_capacity(self) -> str:
        """Estimate remaining growth capacity."""
        current_size = self.hkg.num_nodes()
        
        if current_size < 1000:
            return 'High'
        elif current_size < 10000:
            return 'Medium'
        else:
            return 'Low'
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        # Check for overly dense levels
        for level_id in self.hkg.levels.keys():
            entities = self.hkg.hierarchy_ops.get_entities_at_level(level_id)
            if len(entities) > 1000:
                bottlenecks.append(f"Level {level_id} has {len(entities)} entities (consider subdivision)")
        
        # Check for high-degree entities
        if hasattr(self.hkg, 'analysis_ops'):
            # This is a simplified check - full implementation would analyze actual degrees
            if len(self.hkg.cross_level_relationships) > self.hkg.num_nodes() * 5:
                bottlenecks.append("High cross-level relationship density may impact performance")
        
        return bottlenecks
    
    def _estimate_recommended_max_size(self) -> int:
        """Estimate recommended maximum size for current structure."""
        level_count = len(self.hkg.levels)
        
        # More levels can handle more entities efficiently
        base_capacity = 1000
        level_multiplier = max(1, level_count)
        
        return base_capacity * level_multiplier
    
    def _categorize_anomaly_severity(self, anomalies: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Categorize anomalies by severity."""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for category_anomalies in anomalies.values():
            if isinstance(category_anomalies, list):
                for anomaly in category_anomalies:
                    severity = anomaly.get('severity', 'low')
                    severity_counts[severity] += 1
        
        return severity_counts
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends from historical metrics."""
        if len(self.metric_history) < 2:
            return {}
        
        # Extract key metrics over time
        entity_counts = []
        relationship_counts = []
        quality_scores = []
        
        for snapshot in self.metric_history[-10:]:  # Last 10 snapshots
            basic_metrics = snapshot.get('basic_metrics', {})
            quality_metrics = snapshot.get('quality_metrics', {})
            
            entity_counts.append(basic_metrics.get('total_entities', 0))
            relationship_counts.append(basic_metrics.get('total_relationships', 0))
            quality_scores.append(quality_metrics.get('overall_quality', 0))
        
        trends = {}
        
        # Calculate growth trends
        if len(entity_counts) > 1:
            entity_trend = 'increasing' if entity_counts[-1] > entity_counts[0] else 'decreasing' if entity_counts[-1] < entity_counts[0] else 'stable'
            trends['entity_growth'] = entity_trend
            trends['entity_growth_rate'] = (entity_counts[-1] - entity_counts[0]) / len(entity_counts)
        
        if len(relationship_counts) > 1:
            rel_trend = 'increasing' if relationship_counts[-1] > relationship_counts[0] else 'decreasing' if relationship_counts[-1] < relationship_counts[0] else 'stable'
            trends['relationship_growth'] = rel_trend
            trends['relationship_growth_rate'] = (relationship_counts[-1] - relationship_counts[0]) / len(relationship_counts)
        
        if len(quality_scores) > 1:
            quality_trend = 'improving' if quality_scores[-1] > quality_scores[0] else 'declining' if quality_scores[-1] < quality_scores[0] else 'stable'
            trends['quality_trend'] = quality_trend
            trends['quality_change_rate'] = (quality_scores[-1] - quality_scores[0]) / len(quality_scores)
        
        return trends
    
    def _store_metrics_snapshot(self, metrics: Dict[str, Any]):
        """Store metrics snapshot for trend analysis."""
        # Keep only last 50 snapshots to manage memory
        if len(self.metric_history) >= 50:
            self.metric_history.pop(0)
        
        # Store simplified snapshot
        snapshot = {
            'timestamp': metrics['timestamp'],
            'basic_metrics': metrics['basic_metrics'],
            'quality_metrics': metrics.get('quality_metrics', {})
        }
        
        self.metric_history.append(snapshot)