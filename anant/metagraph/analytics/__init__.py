"""
Advanced Analytics Package for Metagraph

This package provides enterprise-grade analytics capabilities including:
- Pattern Recognition: ML-powered detection of complex relationships and trends
- Predictive Analytics: Forecasting and optimization recommendations  
- Performance Monitoring: Real-time analytics and health monitoring
- Automated Insights: AI-driven discovery of hidden patterns and opportunities

Phase 3 Components:
- Pattern Recognition Engine: Advanced ML algorithms for relationship discovery
- Predictive Analytics: Time-series forecasting and optimization models
- Production Framework: Enterprise deployment and monitoring capabilities
- Governance Automation: Policy enforcement and compliance automation
"""

from typing import TYPE_CHECKING

# Phase 3 Analytics Components
try:
    from .pattern_recognition import (
        PatternRecognitionEngine,
        PatternType,
        DetectedPattern,
        PatternConfidence,
        AnomalyDetector,
        TrendAnalyzer,
        RelationshipDiscoverer
    )
    PATTERN_RECOGNITION_AVAILABLE = True
except ImportError:
    PATTERN_RECOGNITION_AVAILABLE = False

try:
    from .predictive_analytics import (
        PredictiveAnalyticsEngine,
        PredictionType,
        Prediction,
        ForecastModel,
        OptimizationRecommendation,
        TimeSeriesAnalyzer
    )
    PREDICTIVE_ANALYTICS_AVAILABLE = True
except ImportError:
    PREDICTIVE_ANALYTICS_AVAILABLE = False

try:
    from .production_framework import (
        ProductionFramework,
        MonitoringDashboard,
        PerformanceMetrics,
        HealthChecker,
        ScalabilityOptimizer,
        DeploymentManager
    )
    PRODUCTION_FRAMEWORK_AVAILABLE = True
except ImportError:
    PRODUCTION_FRAMEWORK_AVAILABLE = False

try:
    from .governance_automation import (
        GovernanceAutomation,
        PolicyEnforcer,
        ComplianceReporter,
        AutoRemediator,
        RiskAssessment,
        AuditTrail
    )
    GOVERNANCE_AUTOMATION_AVAILABLE = True
except ImportError:
    GOVERNANCE_AUTOMATION_AVAILABLE = False

# Analytics availability status
ANALYTICS_STATUS = {
    'pattern_recognition': PATTERN_RECOGNITION_AVAILABLE,
    'predictive_analytics': PREDICTIVE_ANALYTICS_AVAILABLE,
    'production_framework': PRODUCTION_FRAMEWORK_AVAILABLE,
    'governance_automation': GOVERNANCE_AUTOMATION_AVAILABLE
}

# Export availability checker
def check_analytics_availability():
    """Check which analytics components are available"""
    available = [k for k, v in ANALYTICS_STATUS.items() if v]
    unavailable = [k for k, v in ANALYTICS_STATUS.items() if not v]
    
    print("🔍 Analytics Components Status:")
    for component in available:
        print(f"  ✅ {component}")
    for component in unavailable:
        print(f"  ⚠️  {component} (will be loaded when implemented)")
    
    return ANALYTICS_STATUS

# Convenience imports for available components
__all__ = [
    'check_analytics_availability',
    'ANALYTICS_STATUS'
]

# Add available components to __all__
if PATTERN_RECOGNITION_AVAILABLE:
    __all__.extend([
        'PatternRecognitionEngine',
        'PatternType',
        'DetectedPattern',
        'PatternConfidence',
        'AnomalyDetector',
        'TrendAnalyzer',
        'RelationshipDiscoverer'
    ])

if PREDICTIVE_ANALYTICS_AVAILABLE:
    __all__.extend([
        'PredictiveAnalyticsEngine',
        'PredictionType',
        'Prediction',
        'ForecastModel',
        'OptimizationRecommendation',
        'TimeSeriesAnalyzer'
    ])

if PRODUCTION_FRAMEWORK_AVAILABLE:
    __all__.extend([
        'ProductionFramework',
        'MonitoringDashboard',
        'PerformanceMetrics',
        'HealthChecker',
        'ScalabilityOptimizer',
        'DeploymentManager'
    ])

if GOVERNANCE_AUTOMATION_AVAILABLE:
    __all__.extend([
        'GovernanceAutomation',
        'PolicyEnforcer',
        'ComplianceReporter',
        'AutoRemediator',
        'RiskAssessment',
        'AuditTrail'
    ])