"""
Intelligent Recommendations Engine

This module provides AI-powered recommendation capabilities for the enterprise
metagraph system. It suggests related entities, potential relationships, data
quality improvements, and optimization opportunities using machine learning
and LLM-powered analysis.

Features:
- Entity relationship recommendations
- Data quality improvement suggestions
- Missing relationship detection
- Anomaly identification and recommendations
- Performance optimization suggestions
- Business insight recommendations

Author: anant development team
Date: October 2025
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of recommendations"""
    ENTITY_RELATIONSHIP = "entity_relationship"
    DATA_QUALITY = "data_quality"
    MISSING_METADATA = "missing_metadata"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUSINESS_INSIGHT = "business_insight"
    SECURITY_IMPROVEMENT = "security_improvement"
    COMPLIANCE_ENHANCEMENT = "compliance_enhancement"


class RecommendationPriority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RecommendationStatus(Enum):
    """Status of recommendations"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


@dataclass
class Recommendation:
    """Individual recommendation with metadata and tracking"""
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    rationale: str
    confidence: float
    entity_ids: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    expected_benefits: List[str] = field(default_factory=list)
    implementation_effort: str = "medium"  # low, medium, high
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: RecommendationStatus = RecommendationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[str] = None
    implementation_notes: Optional[str] = None


@dataclass
class RecommendationBatch:
    """Batch of related recommendations"""
    batch_id: str
    theme: str
    recommendations: List[Recommendation] = field(default_factory=list)
    total_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    analysis_context: Dict[str, Any] = field(default_factory=dict)


class IntelligentRecommendationEngine:
    """
    AI-Powered Recommendation Engine for Metagraph
    
    Analyzes the enterprise knowledge graph to provide intelligent suggestions
    for improvements, optimizations, and insights using machine learning
    and LLM-powered analysis techniques.
    """
    
    def __init__(self,
                 metagraph_instance,
                 llm_backend: str = "auto",
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-3.5-turbo",
                 enable_ml_analysis: bool = True,
                 recommendation_threshold: float = 0.6):
        """
        Initialize the Intelligent Recommendation Engine
        
        Parameters
        ----------
        metagraph_instance : Metagraph
            The metagraph instance to analyze
        llm_backend : str
            LLM backend for generating recommendations
        openai_api_key : str, optional
            OpenAI API key for GPT models
        llm_model : str
            LLM model for recommendation generation
        enable_ml_analysis : bool
            Whether to enable machine learning analysis
        recommendation_threshold : float
            Minimum confidence threshold for recommendations
        """
        self.metagraph = metagraph_instance
        self.recommendation_threshold = recommendation_threshold
        self.enable_ml_analysis = enable_ml_analysis
        
        # Initialize LLM backend
        self.llm_backend = self._select_llm_backend(llm_backend)
        self._init_llm_backend(openai_api_key, llm_model)
        
        # Initialize ML components if available
        if enable_ml_analysis and HAS_SKLEARN:
            self.scaler = StandardScaler()
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            self.ml_available = True
        else:
            self.ml_available = False
        
        # Recommendation storage and tracking
        self.recommendations = {}  # recommendation_id -> Recommendation
        self.recommendation_batches = {}  # batch_id -> RecommendationBatch
        self.recommendation_history = []
        
        # Analysis caches
        self.analysis_cache = {}
        self.entity_analysis_cache = {}
        
        # Statistics
        self.engine_stats = {
            "total_recommendations_generated": 0,
            "accepted_recommendations": 0,
            "implemented_recommendations": 0,
            "average_confidence": 0.0,
            "last_analysis_time": None
        }
        
        logger.info(f"Intelligent Recommendation Engine initialized with LLM backend: {self.llm_backend}, "
                   f"ML analysis: {self.ml_available}")
    
    def _select_llm_backend(self, backend: str) -> str:
        """Select the best available LLM backend"""
        if backend == "auto":
            if HAS_OPENAI:
                return "openai"
            elif HAS_TRANSFORMERS:
                return "transformers"
            else:
                return "fallback"
        return backend
    
    def _init_llm_backend(self, api_key: Optional[str], model_name: str):
        """Initialize LLM backend for recommendation generation"""
        if self.llm_backend == "openai" and HAS_OPENAI:
            if api_key and 'openai' in globals():
                openai.api_key = api_key  # type: ignore
            self.llm_model = model_name
        elif self.llm_backend == "transformers" and HAS_TRANSFORMERS:
            try:
                if 'pipeline' in globals():
                    self.text_generator = pipeline("text-generation",  # type: ignore
                                                  model="microsoft/DialoGPT-medium",
                                                  max_length=200)
                    self.llm_model = "transformers_local"
            except Exception as e:
                logger.warning(f"Failed to initialize transformers pipeline: {e}")
                self.llm_backend = "fallback"
        else:
            self.llm_backend = "fallback"
    
    def generate_comprehensive_recommendations(self, 
                                             analysis_scope: str = "full",
                                             focus_entities: Optional[List[str]] = None,
                                             recommendation_types: Optional[List[RecommendationType]] = None) -> RecommendationBatch:
        """
        Generate comprehensive recommendations for the metagraph
        
        Parameters
        ----------
        analysis_scope : str
            Scope of analysis ("full", "entities", "relationships", "quality")
        focus_entities : List[str], optional
            Specific entities to focus analysis on
        recommendation_types : List[RecommendationType], optional
            Types of recommendations to generate
            
        Returns
        -------
        RecommendationBatch
            Batch of generated recommendations
        """
        start_time = datetime.now()
        logger.info(f"Starting comprehensive recommendation generation with scope: {analysis_scope}")
        
        try:
            # Create recommendation batch
            batch_id = f"batch_{int(start_time.timestamp())}"
            batch = RecommendationBatch(
                batch_id=batch_id,
                theme=f"Comprehensive Analysis - {analysis_scope}",
                analysis_context={
                    "scope": analysis_scope,
                    "focus_entities": focus_entities,
                    "analysis_time": start_time.isoformat(),
                    "ml_enabled": self.ml_available
                }
            )
            
            # Determine recommendation types to generate
            if recommendation_types is None:
                recommendation_types = list(RecommendationType)
            
            # Generate recommendations by type
            for rec_type in recommendation_types:
                try:
                    type_recommendations = self._generate_recommendations_by_type(
                        rec_type, analysis_scope, focus_entities
                    )
                    batch.recommendations.extend(type_recommendations)
                    
                except Exception as e:
                    logger.error(f"Error generating {rec_type.value} recommendations: {e}")
            
            # Calculate batch score and filter recommendations
            batch = self._optimize_recommendation_batch(batch)
            
            # Store batch and update statistics
            self.recommendation_batches[batch_id] = batch
            self._update_engine_stats(batch)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated {len(batch.recommendations)} recommendations in {generation_time:.2f}s")
            
            return batch
            
        except Exception as e:
            logger.error(f"Error in comprehensive recommendation generation: {e}")
            # Return empty batch on error
            return RecommendationBatch(
                batch_id="error_batch",
                theme="Error - No recommendations generated",
                analysis_context={"error": str(e)}
            )
    
    def _generate_recommendations_by_type(self, 
                                        rec_type: RecommendationType,
                                        analysis_scope: str,
                                        focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate recommendations for a specific type"""
        recommendations = []
        
        try:
            if rec_type == RecommendationType.ENTITY_RELATIONSHIP:
                recommendations.extend(self._generate_relationship_recommendations(focus_entities))
            elif rec_type == RecommendationType.DATA_QUALITY:
                recommendations.extend(self._generate_data_quality_recommendations(focus_entities))
            elif rec_type == RecommendationType.MISSING_METADATA:
                recommendations.extend(self._generate_missing_metadata_recommendations(focus_entities))
            elif rec_type == RecommendationType.ANOMALY_DETECTION:
                recommendations.extend(self._generate_anomaly_recommendations(focus_entities))
            elif rec_type == RecommendationType.PERFORMANCE_OPTIMIZATION:
                recommendations.extend(self._generate_performance_recommendations())
            elif rec_type == RecommendationType.BUSINESS_INSIGHT:
                recommendations.extend(self._generate_business_insight_recommendations(focus_entities))
            elif rec_type == RecommendationType.SECURITY_IMPROVEMENT:
                recommendations.extend(self._generate_security_recommendations(focus_entities))
            elif rec_type == RecommendationType.COMPLIANCE_ENHANCEMENT:
                recommendations.extend(self._generate_compliance_recommendations(focus_entities))
            
        except Exception as e:
            logger.warning(f"Error generating {rec_type.value} recommendations: {e}")
        
        return recommendations
    
    def _generate_relationship_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate entity relationship recommendations"""
        recommendations = []
        
        try:
            # Get entities to analyze
            entities_to_analyze = focus_entities or self._get_sample_entities(50)
            
            # Find potential missing relationships
            missing_relationships = self._detect_missing_relationships(entities_to_analyze)
            
            for missing_rel in missing_relationships:
                rec_id = f"rel_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.ENTITY_RELATIONSHIP,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Add relationship between {missing_rel['entity1']} and {missing_rel['entity2']}",
                    description=f"Consider adding a '{missing_rel['suggested_type']}' relationship",
                    rationale=missing_rel['rationale'],
                    confidence=missing_rel['confidence'],
                    entity_ids=[missing_rel['entity1'], missing_rel['entity2']],
                    action_items=[
                        f"Review relationship between {missing_rel['entity1']} and {missing_rel['entity2']}",
                        f"Add {missing_rel['suggested_type']} relationship if appropriate"
                    ],
                    expected_benefits=[
                        "Improved relationship discovery",
                        "Better semantic understanding",
                        "Enhanced navigation"
                    ],
                    implementation_effort="low",
                    metadata=missing_rel
                )
                
                recommendations.append(recommendation)
            
        except Exception as e:
            logger.warning(f"Error generating relationship recommendations: {e}")
        
        return recommendations
    
    def _generate_data_quality_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        try:
            # Analyze data quality issues
            quality_issues = self._analyze_data_quality(focus_entities)
            
            for issue in quality_issues:
                rec_id = f"quality_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                priority = RecommendationPriority.HIGH if issue['severity'] == "high" else RecommendationPriority.MEDIUM
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.DATA_QUALITY,
                    priority=priority,
                    title=f"Fix {issue['issue_type']} in {issue['entity']}",
                    description=issue['description'],
                    rationale=issue['rationale'],
                    confidence=issue['confidence'],
                    entity_ids=[issue['entity']],
                    action_items=issue['action_items'],
                    expected_benefits=[
                        "Improved data reliability",
                        "Better analysis accuracy",
                        "Enhanced user trust"
                    ],
                    implementation_effort=issue['effort'],
                    metadata=issue
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating data quality recommendations: {e}")
        
        return recommendations
    
    def _generate_missing_metadata_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate missing metadata recommendations"""
        recommendations = []
        
        try:
            # Find entities with missing critical metadata
            missing_metadata = self._detect_missing_metadata(focus_entities)
            
            for missing in missing_metadata:
                rec_id = f"metadata_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.MISSING_METADATA,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Add missing {missing['metadata_type']} to {missing['entity']}",
                    description=f"Entity is missing important {missing['metadata_type']} information",
                    rationale=missing['rationale'],
                    confidence=missing['confidence'],
                    entity_ids=[missing['entity']],
                    action_items=[
                        f"Add {missing['metadata_type']} to {missing['entity']}",
                        "Verify data accuracy and completeness"
                    ],
                    expected_benefits=[
                        "Improved entity understanding",
                        "Better search and discovery",
                        "Enhanced completeness"
                    ],
                    implementation_effort="low",
                    metadata=missing
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating missing metadata recommendations: {e}")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate anomaly detection recommendations"""
        recommendations = []
        
        try:
            if not self.ml_available:
                return recommendations
            
            # Detect anomalies using ML analysis
            anomalies = self._detect_anomalies_ml(focus_entities)
            
            for anomaly in anomalies:
                rec_id = f"anomaly_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                priority = RecommendationPriority.HIGH if anomaly['severity'] == "high" else RecommendationPriority.MEDIUM
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.ANOMALY_DETECTION,
                    priority=priority,
                    title=f"Investigate anomaly in {anomaly['entity']}",
                    description=anomaly['description'],
                    rationale=anomaly['rationale'],
                    confidence=anomaly['confidence'],
                    entity_ids=[anomaly['entity']],
                    action_items=[
                        f"Investigate anomaly in {anomaly['entity']}",
                        "Verify data accuracy",
                        "Check for data corruption or errors"
                    ],
                    expected_benefits=[
                        "Improved data quality",
                        "Error prevention",
                        "Better reliability"
                    ],
                    implementation_effort="medium",
                    metadata=anomaly
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating anomaly recommendations: {e}")
        
        return recommendations
    
    def _generate_performance_recommendations(self) -> List[Recommendation]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        try:
            # Analyze performance bottlenecks
            performance_issues = self._analyze_performance_bottlenecks()
            
            for issue in performance_issues:
                rec_id = f"performance_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                    priority=RecommendationPriority.MEDIUM,
                    title=issue['title'],
                    description=issue['description'],
                    rationale=issue['rationale'],
                    confidence=issue['confidence'],
                    action_items=issue['action_items'],
                    expected_benefits=[
                        "Improved system performance",
                        "Faster query responses",
                        "Better user experience"
                    ],
                    implementation_effort=issue['effort'],
                    metadata=issue
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating performance recommendations: {e}")
        
        return recommendations
    
    def _generate_business_insight_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate business insight recommendations using LLM analysis"""
        recommendations = []
        
        try:
            # Generate insights using LLM
            insights = self._generate_business_insights_llm(focus_entities)
            
            for insight in insights:
                rec_id = f"insight_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.BUSINESS_INSIGHT,
                    priority=RecommendationPriority.MEDIUM,
                    title=insight['title'],
                    description=insight['description'],
                    rationale=insight['rationale'],
                    confidence=insight['confidence'],
                    entity_ids=insight.get('entity_ids', []),
                    action_items=insight['action_items'],
                    expected_benefits=[
                        "Business value discovery",
                        "Strategic insights",
                        "Decision support"
                    ],
                    implementation_effort="medium",
                    metadata=insight
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating business insight recommendations: {e}")
        
        return recommendations
    
    def _generate_security_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        try:
            # Analyze security vulnerabilities
            security_issues = self._analyze_security_vulnerabilities(focus_entities)
            
            for issue in security_issues:
                rec_id = f"security_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                priority = RecommendationPriority.CRITICAL if issue['severity'] == "critical" else RecommendationPriority.HIGH
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.SECURITY_IMPROVEMENT,
                    priority=priority,
                    title=issue['title'],
                    description=issue['description'],
                    rationale=issue['rationale'],
                    confidence=issue['confidence'],
                    entity_ids=issue.get('entity_ids', []),
                    action_items=issue['action_items'],
                    expected_benefits=[
                        "Improved security posture",
                        "Risk mitigation",
                        "Compliance enhancement"
                    ],
                    implementation_effort=issue['effort'],
                    metadata=issue
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating security recommendations: {e}")
        
        return recommendations
    
    def _generate_compliance_recommendations(self, focus_entities: Optional[List[str]]) -> List[Recommendation]:
        """Generate compliance enhancement recommendations"""
        recommendations = []
        
        try:
            # Analyze compliance gaps
            compliance_gaps = self._analyze_compliance_gaps(focus_entities)
            
            for gap in compliance_gaps:
                rec_id = f"compliance_{int(datetime.now().timestamp())}_{len(recommendations)}"
                
                recommendation = Recommendation(
                    recommendation_id=rec_id,
                    recommendation_type=RecommendationType.COMPLIANCE_ENHANCEMENT,
                    priority=RecommendationPriority.HIGH,
                    title=gap['title'],
                    description=gap['description'],
                    rationale=gap['rationale'],
                    confidence=gap['confidence'],
                    entity_ids=gap.get('entity_ids', []),
                    action_items=gap['action_items'],
                    expected_benefits=[
                        "Compliance assurance",
                        "Risk reduction",
                        "Regulatory alignment"
                    ],
                    implementation_effort=gap['effort'],
                    metadata=gap
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating compliance recommendations: {e}")
        
        return recommendations
    
    def _optimize_recommendation_batch(self, batch: RecommendationBatch) -> RecommendationBatch:
        """Optimize and filter recommendation batch"""
        # Filter by confidence threshold
        filtered_recommendations = [
            rec for rec in batch.recommendations 
            if rec.confidence >= self.recommendation_threshold
        ]
        
        # Sort by priority and confidence
        def recommendation_score(rec):
            priority_scores = {
                RecommendationPriority.CRITICAL: 5,
                RecommendationPriority.HIGH: 4,
                RecommendationPriority.MEDIUM: 3,
                RecommendationPriority.LOW: 2,
                RecommendationPriority.INFORMATIONAL: 1
            }
            return priority_scores.get(rec.priority, 1) * rec.confidence
        
        sorted_recommendations = sorted(filtered_recommendations, key=recommendation_score, reverse=True)
        
        # Limit to top 50 recommendations
        optimized_recommendations = sorted_recommendations[:50]
        
        # Calculate batch score
        if optimized_recommendations:
            batch.total_score = sum(recommendation_score(rec) for rec in optimized_recommendations) / len(optimized_recommendations)
        else:
            batch.total_score = 0.0
        
        batch.recommendations = optimized_recommendations
        return batch
    
    # Helper methods for analysis (implementations would be more sophisticated in practice)
    
    def _get_sample_entities(self, limit: int) -> List[str]:
        """Get sample entities for analysis"""
        try:
            if hasattr(self.metagraph, 'hierarchical_store'):
                all_entities = self.metagraph.hierarchical_store.get_all_entities()
                return all_entities[:limit]
        except Exception:
            pass
        return ["entity_1", "entity_2", "entity_3"]  # Fallback
    
    def _detect_missing_relationships(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Detect potentially missing relationships between entities"""
        missing_relationships = []
        
        # Simple heuristic: entities with similar names or properties might be related
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                similarity = self._calculate_entity_similarity(entity1, entity2)
                
                if 0.5 < similarity < 0.9:  # Potentially related but not already connected
                    missing_relationships.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'suggested_type': 'related_to',
                        'confidence': similarity,
                        'rationale': f"Entities show {similarity:.2f} similarity but no explicit relationship"
                    })
        
        return missing_relationships[:10]  # Limit results
    
    def _calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate similarity between two entities"""
        # Simple string similarity for now
        common_chars = set(entity1.lower()) & set(entity2.lower())
        total_chars = set(entity1.lower()) | set(entity2.lower())
        return len(common_chars) / len(total_chars) if total_chars else 0.0
    
    def _analyze_data_quality(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Analyze data quality issues"""
        quality_issues = []
        
        entities_to_check = entities or self._get_sample_entities(20)
        
        for entity in entities_to_check:
            # Check for common data quality issues
            if not entity or len(entity.strip()) == 0:
                quality_issues.append({
                    'entity': entity,
                    'issue_type': 'empty_entity',
                    'description': 'Entity identifier is empty or whitespace',
                    'severity': 'high',
                    'confidence': 0.9,
                    'rationale': 'Empty entities can cause system errors',
                    'action_items': ['Remove or fix empty entity identifier'],
                    'effort': 'low'
                })
            
            if len(entity) > 100:  # Suspiciously long entity name
                quality_issues.append({
                    'entity': entity,
                    'issue_type': 'long_entity_name',
                    'description': 'Entity name is unusually long',
                    'severity': 'medium',
                    'confidence': 0.7,
                    'rationale': 'Long entity names may indicate data entry errors',
                    'action_items': ['Review and potentially shorten entity name'],
                    'effort': 'low'
                })
        
        return quality_issues[:5]  # Limit results
    
    def _detect_missing_metadata(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Detect missing metadata for entities"""
        missing_metadata = []
        
        entities_to_check = entities or self._get_sample_entities(15)
        
        for entity in entities_to_check:
            # Simple check for missing common metadata types
            missing_metadata.append({
                'entity': entity,
                'metadata_type': 'description',
                'confidence': 0.6,
                'rationale': 'Entities should have descriptions for better understanding'
            })
        
        return missing_metadata[:10]  # Limit results
    
    def _detect_anomalies_ml(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Detect anomalies using machine learning"""
        anomalies = []
        
        if not self.ml_available:
            return anomalies
        
        # Placeholder ML anomaly detection
        entities_to_check = entities or self._get_sample_entities(10)
        
        for entity in entities_to_check:
            # Simple anomaly detection based on entity name patterns
            if any(char.isdigit() for char in entity) and any(char.isalpha() for char in entity):
                anomalies.append({
                    'entity': entity,
                    'description': 'Entity name contains mixed alphanumeric pattern',
                    'severity': 'medium',
                    'confidence': 0.5,
                    'rationale': 'Mixed patterns might indicate data entry inconsistencies'
                })
        
        return anomalies[:5]  # Limit results
    
    def _analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks"""
        performance_issues = []
        
        # Generic performance recommendations
        performance_issues.append({
            'title': 'Consider indexing frequently queried properties',
            'description': 'Add indexes to improve query performance',
            'confidence': 0.7,
            'rationale': 'Indexing can significantly improve query response times',
            'action_items': ['Identify frequently queried properties', 'Add appropriate indexes'],
            'effort': 'medium'
        })
        
        return performance_issues
    
    def _generate_business_insights_llm(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Generate business insights using LLM"""
        insights = []
        
        # Generate generic business insights
        insights.append({
            'title': 'Entity relationship density analysis',
            'description': 'Analyze the density of relationships in your knowledge graph',
            'confidence': 0.6,
            'rationale': 'Understanding relationship patterns can reveal business insights',
            'action_items': ['Analyze relationship density', 'Identify highly connected entities'],
            'entity_ids': entities[:5] if entities else []
        })
        
        return insights
    
    def _analyze_security_vulnerabilities(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Analyze security vulnerabilities"""
        security_issues = []
        
        # Generic security recommendations
        security_issues.append({
            'title': 'Review access controls for sensitive entities',
            'description': 'Ensure proper access controls are in place',
            'severity': 'high',
            'confidence': 0.8,
            'rationale': 'Proper access controls are essential for data security',
            'action_items': ['Review current access controls', 'Implement role-based access'],
            'effort': 'medium',
            'entity_ids': entities[:3] if entities else []
        })
        
        return security_issues
    
    def _analyze_compliance_gaps(self, entities: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Analyze compliance gaps"""
        compliance_gaps = []
        
        # Generic compliance recommendations
        compliance_gaps.append({
            'title': 'Implement data lineage tracking',
            'description': 'Add comprehensive data lineage tracking for compliance',
            'confidence': 0.7,
            'rationale': 'Data lineage is required for many compliance frameworks',
            'action_items': ['Implement lineage tracking', 'Document data flows'],
            'effort': 'high',
            'entity_ids': entities[:5] if entities else []
        })
        
        return compliance_gaps
    
    def _update_engine_stats(self, batch: RecommendationBatch):
        """Update engine statistics"""
        self.engine_stats["total_recommendations_generated"] += len(batch.recommendations)
        
        if batch.recommendations:
            avg_confidence = sum(rec.confidence for rec in batch.recommendations) / len(batch.recommendations)
            total_generated = self.engine_stats["total_recommendations_generated"]
            current_avg = self.engine_stats["average_confidence"]
            
            # Update rolling average
            self.engine_stats["average_confidence"] = (
                (current_avg * (total_generated - len(batch.recommendations)) + 
                 avg_confidence * len(batch.recommendations)) / total_generated
            )
        
        self.engine_stats["last_analysis_time"] = datetime.now().isoformat()
    
    def get_recommendation_by_id(self, recommendation_id: str) -> Optional[Recommendation]:
        """Get a specific recommendation by ID"""
        return self.recommendations.get(recommendation_id)
    
    def update_recommendation_status(self, 
                                   recommendation_id: str, 
                                   status: RecommendationStatus,
                                   feedback: Optional[str] = None,
                                   implementation_notes: Optional[str] = None):
        """Update the status of a recommendation"""
        if recommendation_id in self.recommendations:
            recommendation = self.recommendations[recommendation_id]
            recommendation.status = status
            recommendation.feedback = feedback
            recommendation.implementation_notes = implementation_notes
            
            # Update statistics
            if status == RecommendationStatus.ACCEPTED:
                self.engine_stats["accepted_recommendations"] += 1
            elif status == RecommendationStatus.IMPLEMENTED:
                self.engine_stats["implemented_recommendations"] += 1
            
            logger.info(f"Updated recommendation {recommendation_id} status to {status.value}")
    
    def get_recommendations_by_type(self, rec_type: RecommendationType) -> List[Recommendation]:
        """Get all recommendations of a specific type"""
        return [rec for rec in self.recommendations.values() if rec.recommendation_type == rec_type]
    
    def get_recommendations_by_priority(self, priority: RecommendationPriority) -> List[Recommendation]:
        """Get all recommendations of a specific priority"""
        return [rec for rec in self.recommendations.values() if rec.priority == priority]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get recommendation engine statistics"""
        return self.engine_stats.copy()
    
    def export_recommendations(self, batch_id: Optional[str] = None) -> Dict[str, Any]:
        """Export recommendations for persistence or analysis"""
        if batch_id and batch_id in self.recommendation_batches:
            batch = self.recommendation_batches[batch_id]
            recommendations_data = [self._recommendation_to_dict(rec) for rec in batch.recommendations]
            return {
                "batch_id": batch_id,
                "batch_theme": batch.theme,
                "total_score": batch.total_score,
                "created_at": batch.created_at.isoformat(),
                "recommendations": recommendations_data
            }
        else:
            # Export all recommendations
            all_recommendations = [self._recommendation_to_dict(rec) for rec in self.recommendations.values()]
            return {
                "total_recommendations": len(all_recommendations),
                "export_time": datetime.now().isoformat(),
                "recommendations": all_recommendations,
                "statistics": self.engine_stats
            }
    
    def _recommendation_to_dict(self, recommendation: Recommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary for export"""
        return {
            "recommendation_id": recommendation.recommendation_id,
            "recommendation_type": recommendation.recommendation_type.value,
            "priority": recommendation.priority.value,
            "title": recommendation.title,
            "description": recommendation.description,
            "rationale": recommendation.rationale,
            "confidence": recommendation.confidence,
            "entity_ids": recommendation.entity_ids,
            "action_items": recommendation.action_items,
            "expected_benefits": recommendation.expected_benefits,
            "implementation_effort": recommendation.implementation_effort,
            "created_at": recommendation.created_at.isoformat(),
            "expires_at": recommendation.expires_at.isoformat() if recommendation.expires_at else None,
            "status": recommendation.status.value,
            "metadata": recommendation.metadata,
            "feedback": recommendation.feedback,
            "implementation_notes": recommendation.implementation_notes
        }


# Export main classes
__all__ = [
    "RecommendationType",
    "RecommendationPriority",
    "RecommendationStatus",
    "Recommendation",
    "RecommendationBatch",
    "IntelligentRecommendationEngine"
]