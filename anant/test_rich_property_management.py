"""
Comprehensive Test Suite for Rich Property Management System

This module tests all components of the advanced property management system
including property type detection, analysis framework, weight analysis, and
incidence pattern analysis.
"""

import pytest
import polars as pl
import numpy as np
from typing import Dict, List, Any
import tempfile
import os

# Import the Rich Property Management System components
from anant.utils import (
    PropertyTypeManager, PropertyType,
    PropertyAnalysisFramework, CorrelationType, AnomalyType,
    WeightAnalyzer, WeightNormalizationType, WeightDistributionType,
    IncidencePatternAnalyzer, PatternType, MotifSize
)


class TestPropertyTypeManager:
    """Test the PropertyTypeManager component"""
    
    def setup_method(self):
        """Setup test data"""
        self.manager = PropertyTypeManager()
        
        # Create test data with different property types
        self.test_data = pl.DataFrame({
            'categorical_prop': ['A', 'B', 'A', 'C', 'B', 'A'],
            'numerical_prop': [1.5, 2.3, 4.1, 5.8, 3.2, 6.7],
            'integer_prop': [1, 2, 3, 4, 5, 6],
            'temporal_prop': ['2023-01-01', '2023-01-02', '2023-01-03', 
                             '2023-01-04', '2023-01-05', '2023-01-06'],
            'text_prop': ['short text', 'another text sample', 'long text content here',
                         'brief', 'extended text with more words', 'medium length'],
            'vector_prop': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                           [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            'json_prop': ['{"key": "value"}', '{"number": 42}', '{"array": [1,2,3]}',
                         '{"nested": {"inner": "data"}}', '{"bool": true}', '{"float": 3.14}']
        })
        
    def test_property_type_detection(self):
        """Test automatic property type detection"""
        # Test categorical detection
        cat_type = self.manager.detect_property_type(self.test_data['categorical_prop'])
        assert cat_type == PropertyType.CATEGORICAL
        
        # Test numerical detection
        num_type = self.manager.detect_property_type(self.test_data['numerical_prop'])
        assert num_type == PropertyType.NUMERICAL
        
        # Test integer detection (should be numerical)
        int_type = self.manager.detect_property_type(self.test_data['integer_prop'])
        assert int_type == PropertyType.NUMERICAL
        
        # Test text detection
        text_type = self.manager.detect_property_type(self.test_data['text_prop'])
        assert text_type == PropertyType.TEXT
        
        # Test JSON detection
        json_type = self.manager.detect_property_type(self.test_data['json_prop'])
        assert json_type == PropertyType.JSON
        
    def test_batch_type_detection(self):
        """Test batch detection of property types"""
        type_map = self.manager.detect_all_property_types(self.test_data)
        
        expected_types = {
            'categorical_prop': PropertyType.CATEGORICAL,
            'numerical_prop': PropertyType.NUMERICAL,
            'integer_prop': PropertyType.NUMERICAL,
            'text_prop': PropertyType.TEXT,
            'json_prop': PropertyType.JSON
        }
        
        for prop, expected_type in expected_types.items():
            assert type_map[prop] == expected_type
            
    def test_column_optimization(self):
        """Test storage optimization"""
        # Test categorical optimization
        optimized_df = self.manager.optimize_column_storage(
            self.test_data, 'categorical_prop'
        )
        
        # Should be converted to categorical type
        assert optimized_df['categorical_prop'].dtype == pl.Categorical
        
    def test_memory_analysis(self):
        """Test memory usage analysis"""
        analysis = self.manager.analyze_memory_usage(self.test_data)
        
        assert 'total_memory_bytes' in analysis
        assert 'columns' in analysis
        assert analysis['total_memory_bytes'] > 0
        
        # Should have analysis for each column
        for col in self.test_data.columns:
            assert col in analysis['columns']


class TestPropertyAnalysisFramework:
    """Test the PropertyAnalysisFramework component"""
    
    def setup_method(self):
        """Setup test data"""
        self.framework = PropertyAnalysisFramework()
        
        # Create test data with correlations and anomalies
        np.random.seed(42)
        n_samples = 100
        
        x = np.random.normal(0, 1, n_samples)
        y = 0.8 * x + np.random.normal(0, 0.2, n_samples)  # Strong correlation
        z = np.random.uniform(0, 10, n_samples)  # Independent
        
        # Add some anomalies
        anomaly_indices = [10, 25, 60, 85]
        z[anomaly_indices] = [50, 45, 55, 48]  # Clear outliers
        
        self.test_data = pl.DataFrame({
            'prop_x': x,
            'prop_y': y, 
            'prop_z': z,
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'id': range(n_samples)
        })
        
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        correlations = self.framework.analyze_property_correlations(
            self.test_data,
            properties=['prop_x', 'prop_y', 'prop_z'],
            min_correlation=0.1
        )
        
        # Should find strong correlation between prop_x and prop_y
        strong_corr = [c for c in correlations if abs(c.correlation_value) > 0.7]
        assert len(strong_corr) > 0
        
        # Check specific correlation
        xy_corr = [c for c in correlations 
                  if (c.property1 == 'prop_x' and c.property2 == 'prop_y') or
                     (c.property1 == 'prop_y' and c.property2 == 'prop_x')]
        assert len(xy_corr) > 0
        assert abs(xy_corr[0].correlation_value) > 0.7
        
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        anomalies = self.framework.detect_property_anomalies(
            self.test_data,
            'prop_z',
            anomaly_types=[AnomalyType.STATISTICAL_OUTLIER]
        )
        
        assert len(anomalies) > 0
        
        # Should detect the artificially added outliers
        outlier_result = anomalies[0]
        assert len(outlier_result.anomalous_indices) > 0
        
        # Check if detected anomalies include our artificial ones
        detected_anomalies = set(outlier_result.anomalous_indices)
        artificial_anomalies = {10, 25, 60, 85}
        
        # Should detect at least some of the artificial anomalies
        intersection = detected_anomalies & artificial_anomalies
        assert len(intersection) > 0
        
    def test_distribution_analysis(self):
        """Test property distribution analysis"""
        distribution = self.framework.analyze_property_distribution(
            self.test_data, 'prop_x'
        )
        
        assert distribution.property_name == 'prop_x'
        assert distribution.property_type == PropertyType.NUMERICAL
        assert distribution.mean is not None
        assert distribution.std is not None
        assert distribution.quartiles is not None
        
    def test_relationship_discovery(self):
        """Test property relationship discovery"""
        relationships = self.framework.discover_property_relationships(
            self.test_data,
            'prop_x',
            min_correlation=0.3
        )
        
        assert 'target_property' in relationships
        assert relationships['target_property'] == 'prop_x'
        assert 'correlations' in relationships
        assert 'distribution' in relationships
        
    def test_comprehensive_report(self):
        """Test comprehensive analysis report generation"""
        report = self.framework.generate_analysis_report(
            self.test_data,
            properties=['prop_x', 'prop_y', 'prop_z']
        )
        
        assert 'dataset_summary' in report
        assert 'property_distributions' in report
        assert 'correlations' in report
        assert 'anomalies' in report
        
        # Should have distributions for all properties
        for prop in ['prop_x', 'prop_y', 'prop_z']:
            assert prop in report['property_distributions']


class TestWeightAnalyzer:
    """Test the WeightAnalyzer component"""
    
    def setup_method(self):
        """Setup test data"""
        self.analyzer = WeightAnalyzer()
        
        # Create test weight data
        np.random.seed(42)
        n_samples = 100
        
        # Different weight distributions
        normal_weights = np.random.normal(5, 2, n_samples)
        exponential_weights = np.random.exponential(2, n_samples)
        uniform_weights = np.random.uniform(0, 10, n_samples)
        
        # Add some zero weights for sparsity
        sparse_weights = normal_weights.copy()
        sparse_indices = np.random.choice(n_samples, size=30, replace=False)
        sparse_weights[sparse_indices] = 0
        
        self.test_data = pl.DataFrame({
            'normal_weights': normal_weights,
            'exponential_weights': exponential_weights,
            'uniform_weights': uniform_weights,
            'sparse_weights': sparse_weights,
            'entity_id': range(n_samples)
        })
        
    def test_weight_statistics(self):
        """Test weight statistics analysis"""
        stats = self.analyzer.analyze_weight_statistics(
            self.test_data,
            ['normal_weights', 'sparse_weights'],
            entity_type='node'
        )
        
        assert 'normal_weights' in stats
        assert 'sparse_weights' in stats
        
        normal_stats = stats['normal_weights']
        assert normal_stats.entity_type == 'node'
        assert normal_stats.total_count == 100
        assert normal_stats.mean > 0
        assert normal_stats.std > 0
        
        sparse_stats = stats['sparse_weights']
        assert sparse_stats.sparsity > 0.2  # Should have significant sparsity
        
    def test_distribution_detection(self):
        """Test weight distribution detection"""
        # Test normal distribution
        normal_dist = self.analyzer.detect_weight_distribution(
            self.test_data, 'normal_weights'
        )
        
        # Should detect as normal or close to it
        assert normal_dist.distribution_type in [
            WeightDistributionType.NORMAL, 
            WeightDistributionType.UNKNOWN
        ]
        
        # Test uniform distribution
        uniform_dist = self.analyzer.detect_weight_distribution(
            self.test_data, 'uniform_weights'
        )
        
        # Should have some goodness of fit
        assert uniform_dist.goodness_of_fit >= 0
        
    def test_weight_normalization(self):
        """Test weight normalization methods"""
        # Test min-max normalization
        normalized_df = self.analyzer.normalize_weights(
            self.test_data,
            ['normal_weights'],
            WeightNormalizationType.MIN_MAX
        )
        
        normalized_values = normalized_df['normal_weights'].to_numpy()
        assert np.min(normalized_values) >= -0.01  # Allow small numerical errors
        assert np.max(normalized_values) <= 1.01
        
        # Test z-score normalization
        z_normalized_df = self.analyzer.normalize_weights(
            self.test_data,
            ['normal_weights'],
            WeightNormalizationType.Z_SCORE
        )
        
        z_values = z_normalized_df['normal_weights'].to_numpy()
        assert abs(np.mean(z_values)) < 0.01  # Should be approximately 0
        assert abs(np.std(z_values) - 1.0) < 0.01  # Should be approximately 1
        
    def test_weight_clustering(self):
        """Test weight-based clustering"""
        clusters = self.analyzer.cluster_by_weights(
            self.test_data,
            ['normal_weights', 'uniform_weights'],
            n_clusters=3,
            clustering_method='kmeans'
        )
        
        assert len(clusters) <= 3  # Should create up to 3 clusters
        
        if clusters:
            # Check cluster properties
            total_entities = sum(cluster.cluster_size for cluster in clusters)
            assert total_entities <= 100  # Should not exceed data size
            
            for cluster in clusters:
                assert cluster.cluster_size > 0
                assert len(cluster.entity_indices) == cluster.cluster_size
                assert 'normal_weights' in cluster.centroid_weights
                assert 'uniform_weights' in cluster.centroid_weights
                
    def test_weight_correlations(self):
        """Test weight correlation analysis"""
        correlations = self.analyzer.analyze_weight_correlations(
            self.test_data,
            ['normal_weights', 'exponential_weights', 'uniform_weights'],
            min_correlation=0.1
        )
        
        assert 'correlations' in correlations
        assert 'summary' in correlations
        
        summary = correlations['summary']
        assert 'total_pairs' in summary
        assert summary['total_pairs'] == 3  # 3 choose 2 = 3 pairs
        
    def test_storage_optimization(self):
        """Test weight storage optimization"""
        optimization = self.analyzer.optimize_weight_storage(
            self.test_data,
            ['normal_weights', 'sparse_weights'],
            compression_threshold=0.2
        )
        
        assert 'column_results' in optimization
        assert 'summary' in optimization
        
        # Should have results for each weight column
        assert 'normal_weights' in optimization['column_results']
        assert 'sparse_weights' in optimization['column_results']
        
        # Sparse weights should have compression recommendations
        sparse_results = optimization['column_results']['sparse_weights']
        assert sparse_results['sparsity'] > 0.2


class TestIncidencePatternAnalyzer:
    """Test the IncidencePatternAnalyzer component"""
    
    def setup_method(self):
        """Setup test data"""
        self.analyzer = IncidencePatternAnalyzer()
        
        # Create test incidence data with known patterns
        self.incidence_data = pl.DataFrame({
            'node': ['n1', 'n2', 'n3', 'n1', 'n2', 'n4', 'n3', 'n4', 'n5',
                     'n1', 'n5', 'n6', 'n7', 'n8', 'n6', 'n7', 'n8', 'n9'],
            'edge': ['e1', 'e1', 'e1', 'e2', 'e2', 'e2', 'e3', 'e3', 'e3',
                     'e4', 'e4', 'e5', 'e5', 'e5', 'e6', 'e6', 'e6', 'e6']
        })
        
    def test_motif_detection(self):
        """Test incidence motif detection"""
        motifs = self.analyzer.detect_incidence_motifs(
            self.incidence_data,
            node_col='node',
            edge_col='edge',
            min_frequency=1
        )
        
        # Should detect some motifs
        assert len(motifs) > 0
        
        # Check motif properties
        for motif in motifs:
            assert motif.motif_id is not None
            assert motif.pattern_type in PatternType
            assert motif.frequency > 0
            assert motif.significance_score >= 0
            assert len(motif.nodes) > 0
            assert len(motif.edges) > 0
            
    def test_pattern_statistics(self):
        """Test pattern statistics analysis"""
        motifs = self.analyzer.detect_incidence_motifs(
            self.incidence_data,
            min_frequency=1
        )
        
        stats = self.analyzer.analyze_pattern_statistics(motifs)
        
        # Should have statistics for detected pattern types
        for pattern_type, pattern_stats in stats.items():
            assert pattern_stats.total_count > 0
            assert pattern_stats.avg_size > 0
            assert len(pattern_stats.frequency_distribution) > 0
            
    def test_topological_features(self):
        """Test topological features computation"""
        features = self.analyzer.compute_topological_features(
            self.incidence_data,
            node_col='node',
            edge_col='edge'
        )
        
        assert 0 <= features.connectivity <= 1
        assert 0 <= features.clustering_coefficient <= 1
        assert len(features.degree_distribution) > 0
        assert len(features.betweenness_centrality) > 0
        assert len(features.closeness_centrality) > 0
        assert len(features.eigenvector_centrality) > 0
        
    def test_anomaly_detection(self):
        """Test anomalous pattern detection"""
        motifs = self.analyzer.detect_incidence_motifs(
            self.incidence_data,
            min_frequency=1
        )
        
        anomalies = self.analyzer.detect_anomalous_patterns(
            motifs,
            significance_threshold=0.1
        )
        
        # Should be able to detect anomalies (may be 0 if no clear anomalies)
        assert isinstance(anomalies, list)
        
        for anomaly in anomalies:
            assert anomaly.motif_id is not None
            assert anomaly.pattern_type in PatternType
            
    def test_comprehensive_pattern_report(self):
        """Test comprehensive pattern analysis report"""
        report = self.analyzer.generate_pattern_report(
            self.incidence_data,
            node_col='node',
            edge_col='edge'
        )
        
        assert 'dataset_summary' in report
        assert 'motif_analysis' in report
        assert 'pattern_statistics' in report
        assert 'topological_features' in report
        assert 'anomalous_patterns' in report
        
        # Check dataset summary
        summary = report['dataset_summary']
        assert summary['total_nodes'] > 0
        assert summary['total_edges'] > 0
        assert summary['total_incidences'] > 0
        
        # Check motif analysis
        motif_analysis = report['motif_analysis']
        assert 'total_motifs' in motif_analysis
        assert 'motifs_by_type' in motif_analysis
        
        # Check topological features
        topo_features = report['topological_features']
        assert 'connectivity' in topo_features
        assert 'clustering_coefficient' in topo_features


class TestIntegratedPropertyManagement:
    """Test integrated functionality of the Rich Property Management System"""
    
    def setup_method(self):
        """Setup test data for integration tests"""
        self.property_manager = PropertyTypeManager()
        self.analysis_framework = PropertyAnalysisFramework()
        self.weight_analyzer = WeightAnalyzer()
        self.pattern_analyzer = IncidencePatternAnalyzer()
        
        # Create complex test data that combines all aspects
        np.random.seed(42)
        n_nodes = 50
        n_edges = 30
        
        # Node properties
        self.node_data = pl.DataFrame({
            'node_id': [f'n{i}' for i in range(n_nodes)],
            'node_weight': np.random.exponential(2, n_nodes),
            'node_category': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_nodes),
            'node_feature': np.random.normal(5, 2, n_nodes),
            'node_metadata': [f'{{"id": {i}, "active": {str(bool(i % 2)).lower()}}}' for i in range(n_nodes)]
        })
        
        # Edge properties  
        self.edge_data = pl.DataFrame({
            'edge_id': [f'e{i}' for i in range(n_edges)],
            'edge_weight': np.random.gamma(2, 2, n_edges),
            'edge_type': np.random.choice(['connects', 'relates', 'contains'], n_edges),
            'edge_strength': np.random.uniform(0, 1, n_edges)
        })
        
        # Incidence relationships
        incidence_pairs = []
        for edge_idx in range(n_edges):
            # Each edge connects 2-4 random nodes
            n_connections = np.random.randint(2, 5)
            connected_nodes = np.random.choice(n_nodes, n_connections, replace=False)
            for node_idx in connected_nodes:
                incidence_pairs.append((f'n{node_idx}', f'e{edge_idx}'))
                
        self.incidence_data = pl.DataFrame({
            'node': [pair[0] for pair in incidence_pairs],
            'edge': [pair[1] for pair in incidence_pairs]
        })
        
    def test_end_to_end_property_analysis(self):
        """Test complete property analysis workflow"""
        # 1. Detect property types
        node_types = self.property_manager.detect_all_property_types(self.node_data)
        edge_types = self.property_manager.detect_all_property_types(self.edge_data)
        
        # Verify type detection
        assert node_types['node_weight'] == PropertyType.NUMERICAL
        assert node_types['node_category'] == PropertyType.CATEGORICAL
        assert node_types['node_metadata'] == PropertyType.JSON
        
        # 2. Optimize storage
        optimized_nodes = self.property_manager.optimize_dataframe_storage(self.node_data)
        optimized_edges = self.property_manager.optimize_dataframe_storage(self.edge_data)
        
        # Should have optimized categorical columns
        assert optimized_nodes['node_category'].dtype == pl.Categorical
        assert optimized_edges['edge_type'].dtype == pl.Categorical
        
        # 3. Analyze correlations
        node_correlations = self.analysis_framework.analyze_property_correlations(
            optimized_nodes,
            properties=['node_weight', 'node_feature']
        )
        
        # Should find some correlations (may be weak)
        assert isinstance(node_correlations, list)
        
        # 4. Detect anomalies
        weight_anomalies = self.analysis_framework.detect_property_anomalies(
            optimized_nodes,
            'node_weight'
        )
        
        assert isinstance(weight_anomalies, list)
        
    def test_weight_analysis_integration(self):
        """Test integrated weight analysis across node and edge properties"""
        # Combine weight columns from both nodes and edges
        all_weights = pl.DataFrame({
            'entity_id': (
                [f'node_{i}' for i in range(len(self.node_data))] +
                [f'edge_{i}' for i in range(len(self.edge_data))]
            ),
            'weight_value': (
                self.node_data['node_weight'].to_list() +
                self.edge_data['edge_weight'].to_list()
            ),
            'entity_type': (
                ['node'] * len(self.node_data) +
                ['edge'] * len(self.edge_data)
            )
        })
        
        # Analyze weight statistics
        weight_stats = self.weight_analyzer.analyze_weight_statistics(
            all_weights,
            ['weight_value'],
            entity_type='mixed'
        )
        
        assert 'weight_value' in weight_stats
        stats = weight_stats['weight_value']
        assert stats.total_count == len(self.node_data) + len(self.edge_data)
        
        # Detect weight distribution
        distribution = self.weight_analyzer.detect_weight_distribution(
            all_weights,
            'weight_value'
        )
        
        assert distribution.distribution_type in WeightDistributionType
        
        # Normalize weights
        normalized_weights = self.weight_analyzer.normalize_weights(
            all_weights,
            ['weight_value'],
            WeightNormalizationType.MIN_MAX
        )
        
        normalized_values = normalized_weights['weight_value'].to_numpy()
        assert np.min(normalized_values) >= -0.01
        assert np.max(normalized_values) <= 1.01
        
    def test_pattern_analysis_integration(self):
        """Test integrated incidence pattern analysis"""
        # Detect motifs in incidence structure
        motifs = self.pattern_analyzer.detect_incidence_motifs(
            self.incidence_data,
            min_frequency=1
        )
        
        # Analyze pattern statistics
        pattern_stats = self.pattern_analyzer.analyze_pattern_statistics(motifs)
        
        # Compute topological features
        topo_features = self.pattern_analyzer.compute_topological_features(
            self.incidence_data
        )
        
        # Generate comprehensive report
        pattern_report = self.pattern_analyzer.generate_pattern_report(
            self.incidence_data
        )
        
        # Verify all components work together
        assert len(motifs) >= 0
        assert isinstance(pattern_stats, dict)
        assert topo_features.connectivity >= 0
        assert 'dataset_summary' in pattern_report
        
    def test_cross_component_analysis(self):
        """Test analysis that spans multiple components"""
        # 1. Start with property type detection
        node_types = self.property_manager.detect_all_property_types(self.node_data)
        
        # 2. Focus on numerical properties for weight analysis
        numerical_props = [
            prop for prop, prop_type in node_types.items()
            if prop_type == PropertyType.NUMERICAL and prop != 'node_id'
        ]
        
        # 3. Analyze weight properties
        if numerical_props:
            weight_correlations = self.weight_analyzer.analyze_weight_correlations(
                self.node_data,
                numerical_props
            )
            
            assert 'correlations' in weight_correlations
            assert 'summary' in weight_correlations
        
        # 4. Combine with pattern analysis
        pattern_report = self.pattern_analyzer.generate_pattern_report(
            self.incidence_data
        )
        
        # 5. Create integrated analysis summary
        integrated_summary = {
            'property_types': node_types,
            'numerical_properties': numerical_props,
            'pattern_summary': pattern_report['dataset_summary'],
            'topology': {
                'connectivity': pattern_report['topological_features']['connectivity'],
                'clustering': pattern_report['topological_features']['clustering_coefficient']
            }
        }
        
        # Verify integrated analysis
        assert len(integrated_summary['property_types']) > 0
        assert 'connectivity' in integrated_summary['topology']
        assert integrated_summary['pattern_summary']['total_nodes'] > 0


def run_rich_property_management_tests():
    """Run all Rich Property Management System tests"""
    print("🧪 Running Rich Property Management System Tests...")
    
    test_classes = [
        TestPropertyTypeManager,
        TestPropertyAnalysisFramework, 
        TestWeightAnalyzer,
        TestIncidencePatternAnalyzer,
        TestIntegratedPropertyManagement
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance and run setup
                test_instance = test_class()
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test method
                getattr(test_instance, test_method)()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {str(e)}")
    
    # Print summary
    print(f"\n📊 Rich Property Management Test Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print(f"\n🎉 All Rich Property Management System tests passed!")
        return True


if __name__ == "__main__":
    success = run_rich_property_management_tests()
    exit(0 if success else 1)