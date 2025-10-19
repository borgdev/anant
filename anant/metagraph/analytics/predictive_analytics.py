"""
Predictive Analytics Engine for Metagraph

This module provides enterprise forecasting and predictive capabilities:
- Entity growth prediction using statistical trends
- Relationship formation forecasting
- Data quality drift prediction
- Optimization opportunity identification

Features robust fallbacks and enterprise-grade reliability.
"""

import asyncio
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions that can be made"""
    ENTITY_GROWTH = "entity_growth"
    RELATIONSHIP_FORMATION = "relationship_formation"
    DATA_QUALITY_DRIFT = "data_quality_drift"
    USAGE_PATTERN = "usage_pattern"
    CAPACITY_REQUIREMENT = "capacity_requirement"
    BUSINESS_METRIC = "business_metric"
    ANOMALY_LIKELIHOOD = "anomaly_likelihood"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"

class ForecastHorizon(Enum):
    """Time horizons for predictions"""
    SHORT_TERM = "short_term"    # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"      # 1-6 months
    STRATEGIC = "strategic"      # 6+ months

@dataclass
class Prediction:
    """Represents a prediction made by the analytics engine"""
    prediction_id: str
    prediction_type: PredictionType
    target_metric: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    forecast_horizon: ForecastHorizon
    prediction_date: datetime
    valid_until: datetime
    feature_importance: Dict[str, float]
    model_used: str
    explanation: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'prediction_type': self.prediction_type.value,
            'target_metric': self.target_metric,
            'predicted_value': self.predicted_value,
            'confidence_interval': self.confidence_interval,
            'confidence_score': self.confidence_score,
            'forecast_horizon': self.forecast_horizon.value,
            'prediction_date': self.prediction_date.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'feature_importance': self.feature_importance,
            'model_used': self.model_used,
            'explanation': self.explanation,
            'metadata': self.metadata or {}
        }

@dataclass
class OptimizationRecommendation:
    """Represents an optimization opportunity identified by predictive analysis"""
    recommendation_id: str
    optimization_type: str
    target_area: str
    current_state: Dict[str, Any]
    predicted_improvement: Dict[str, float]
    implementation_effort: str
    confidence_score: float
    business_impact: str
    timeline: str
    prerequisites: List[str]
    risks: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary"""
        return {
            'recommendation_id': self.recommendation_id,
            'optimization_type': self.optimization_type,
            'target_area': self.target_area,
            'current_state': self.current_state,
            'predicted_improvement': self.predicted_improvement,
            'implementation_effort': self.implementation_effort,
            'confidence_score': self.confidence_score,
            'business_impact': self.business_impact,
            'timeline': self.timeline,
            'prerequisites': self.prerequisites,
            'risks': self.risks,
            'created_at': self.created_at.isoformat()
        }

class ForecastModel:
    """Simple statistical forecasting model"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.historical_data = []
        self.is_trained = False
        self.feature_weights = {}
        
    def train(self, training_data: List[Dict[str, Any]], target_metric: str) -> bool:
        """Train the forecasting model using statistical methods"""
        try:
            self.historical_data = []
            
            # Extract historical values
            for i, data_point in enumerate(training_data):
                value = self._extract_metric_value(data_point, target_metric)
                timestamp = datetime.now() - timedelta(days=len(training_data) - i)
                
                self.historical_data.append({
                    'timestamp': timestamp,
                    'value': value,
                    'features': self._extract_features(data_point)
                })
            
            if len(self.historical_data) >= 3:
                self.is_trained = True
                self._calculate_feature_weights()
                logger.info(f"Model {self.model_name} trained on {len(self.historical_data)} data points")
                return True
            else:
                logger.warning(f"Insufficient training data: {len(self.historical_data)} samples")
                return False
                
        except Exception as e:
            logger.error(f"Error training model {self.model_name}: {e}")
            return False
    
    def predict(self, data: Dict[str, Any], target_metric: str) -> Tuple[float, Tuple[float, float], float]:
        """Make a prediction with confidence interval"""
        if not self.is_trained:
            return 0.0, (0.0, 0.0), 0.0
        
        try:
            # Calculate trend-based prediction
            values = [point['value'] for point in self.historical_data]
            
            if len(values) < 2:
                return values[0] if values else 0.0, (0.0, 0.0), 0.5
            
            # Simple linear trend calculation
            trend_slope = self._calculate_trend_slope(values)
            last_value = values[-1]
            
            # Feature-based adjustment
            current_features = self._extract_features(data)
            feature_adjustment = self._calculate_feature_adjustment(current_features)
            
            # Make prediction
            predicted_value = last_value + trend_slope + feature_adjustment
            
            # Calculate confidence interval
            value_variance = self._calculate_variance(values)
            margin = 1.96 * math.sqrt(value_variance)  # 95% confidence interval
            confidence_interval = (predicted_value - margin, predicted_value + margin)
            
            # Calculate confidence score
            trend_consistency = self._calculate_trend_consistency(values)
            confidence_score = min(1.0, trend_consistency)
            
            return float(predicted_value), confidence_interval, float(confidence_score)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, (0.0, 0.0), 0.0
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> float:
        """Extract metric value from metagraph data"""
        try:
            if metric_name == 'entity_count':
                return float(len(data.get('entities', {})))
            elif metric_name == 'relationship_count':
                return float(len(data.get('relationships', {})))
            elif metric_name == 'connectivity_ratio':
                entity_count = len(data.get('entities', {}))
                relationship_count = len(data.get('relationships', {}))
                if entity_count > 0:
                    return float(relationship_count / entity_count)
                return 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from metagraph data"""
        features = {}
        
        try:
            # Basic structural features
            features['entity_count'] = len(data.get('entities', {}))
            features['relationship_count'] = len(data.get('relationships', {}))
            
            # Entity type diversity
            entity_types = set()
            for entity_data in data.get('entities', {}).values():
                if isinstance(entity_data, dict):
                    entity_types.add(entity_data.get('type', 'unknown'))
            features['entity_type_diversity'] = len(entity_types)
            
            # Relationship type diversity
            rel_types = set()
            for rel_data in data.get('relationships', {}).values():
                if isinstance(rel_data, dict):
                    rel_types.add(rel_data.get('type', 'unknown'))
            features['relationship_type_diversity'] = len(rel_types)
            
            # Connectivity ratio
            if features['entity_count'] > 0:
                features['connectivity_ratio'] = features['relationship_count'] / features['entity_count']
            else:
                features['connectivity_ratio'] = 0.0
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
        
        return features
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        if len(values) < 2:
            return 0.0
        
        try:
            n = len(values)
            x_values = list(range(n))
            
            # Calculate means
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            # Calculate slope
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) < 2:
            return 1.0
        
        try:
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance
        except Exception:
            return 1.0
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate how consistent the trend is (higher = more consistent)"""
        if len(values) < 3:
            return 0.5
        
        try:
            # Calculate differences between consecutive values
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            
            if not diffs:
                return 0.5
            
            # Calculate consistency as inverse of coefficient of variation
            mean_diff = sum(diffs) / len(diffs)
            if mean_diff == 0:
                return 0.8  # Stable trend
            
            diff_variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            diff_std = math.sqrt(diff_variance)
            
            cv = diff_std / abs(mean_diff) if mean_diff != 0 else 1.0
            consistency = 1.0 / (1.0 + cv)
            
            return min(1.0, consistency)
            
        except Exception:
            return 0.5
    
    def _calculate_feature_weights(self) -> None:
        """Calculate weights for features based on correlation with target"""
        try:
            if len(self.historical_data) < 3:
                return
            
            values = [point['value'] for point in self.historical_data]
            
            # Calculate correlation for each feature
            for feature_name in self.historical_data[0]['features'].keys():
                feature_values = [point['features'].get(feature_name, 0) for point in self.historical_data]
                correlation = self._calculate_correlation(values, feature_values)
                self.feature_weights[feature_name] = correlation
                
        except Exception as e:
            logger.error(f"Error calculating feature weights: {e}")
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation between two lists of values"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            n = len(x_values)
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            x_variance = sum((x - x_mean) ** 2 for x in x_values)
            y_variance = sum((y - y_mean) ** 2 for y in y_values)
            
            denominator = math.sqrt(x_variance * y_variance)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return correlation
            
        except Exception:
            return 0.0
    
    def _calculate_feature_adjustment(self, current_features: Dict[str, float]) -> float:
        """Calculate adjustment based on current features"""
        try:
            if not self.feature_weights or not current_features:
                return 0.0
            
            # Get baseline features from historical data
            baseline_features = {}
            for feature_name in self.feature_weights.keys():
                feature_values = [point['features'].get(feature_name, 0) for point in self.historical_data]
                baseline_features[feature_name] = sum(feature_values) / len(feature_values)
            
            # Calculate weighted adjustment
            adjustment = 0.0
            for feature_name, weight in self.feature_weights.items():
                current_val = current_features.get(feature_name, 0)
                baseline_val = baseline_features.get(feature_name, 0)
                
                if baseline_val != 0:
                    feature_change = (current_val - baseline_val) / baseline_val
                    adjustment += weight * feature_change * 0.1  # Scale adjustment
            
            return adjustment
            
        except Exception:
            return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return dict(self.feature_weights)

class TimeSeriesAnalyzer:
    """Analyzes time series patterns in metagraph evolution"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.time_series_data = {}
        
    def add_data_point(self, timestamp: datetime, metric_name: str, value: float) -> None:
        """Add a data point to the time series"""
        if metric_name not in self.time_series_data:
            self.time_series_data[metric_name] = deque(maxlen=self.window_size)
        
        self.time_series_data[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
    
    def detect_seasonality(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns in the time series"""
        if metric_name not in self.time_series_data:
            return None
        
        data = list(self.time_series_data[metric_name])
        if len(data) < 14:
            return None
        
        try:
            values = [point['value'] for point in data]
            
            # Check for weekly patterns (every 7 data points)
            if len(values) >= 14:
                weekly_correlation = self._calculate_autocorrelation(values, 7)
                if weekly_correlation > 0.5:
                    return {
                        'pattern_type': 'weekly',
                        'correlation': weekly_correlation,
                        'period_length': 7
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return None
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at a specific lag"""
        if len(values) <= lag:
            return 0.0
        
        try:
            # Simple autocorrelation calculation
            n = len(values)
            mean_val = sum(values) / n
            
            # Calculate autocorrelation
            numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) 
                          for i in range(n - lag))
            denominator = sum((v - mean_val) ** 2 for v in values)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return abs(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating autocorrelation: {e}")
            return 0.0

class PredictiveAnalyticsEngine:
    """Main engine for predictive analytics and forecasting"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize models
        self.models = {
            'trend_model': ForecastModel('trend_model'),
            'growth_model': ForecastModel('growth_model'),
            'quality_model': ForecastModel('quality_model')
        }
        
        # Time series analyzer
        self.time_series_analyzer = TimeSeriesAnalyzer(
            window_size=self.config.get('time_series_window', 30)
        )
        
        # Prediction storage
        self.predictions = []
        self.prediction_history = []
        self.optimization_recommendations = []
        
        logger.info("Predictive Analytics Engine initialized")
    
    async def train_models(self, historical_data: List[Dict[str, Any]], 
                          target_metrics: Optional[List[str]] = None) -> Dict[str, bool]:
        """Train forecasting models on historical data"""
        if target_metrics is None:
            target_metrics = ['entity_count', 'relationship_count', 'connectivity_ratio']
        
        training_results = {}
        
        try:
            logger.info(f"Training models on {len(historical_data)} data points")
            
            # Train each model for each target metric
            for metric in target_metrics:
                logger.info(f"Training models for metric: {metric}")
                
                for model_name, model in self.models.items():
                    success = model.train(historical_data, metric)
                    training_results[f"{model_name}_{metric}"] = success
                    
                    if success:
                        logger.info(f"✅ {model_name} trained successfully for {metric}")
                    else:
                        logger.warning(f"❌ {model_name} training failed for {metric}")
            
            # Update time series data
            for i, data_point in enumerate(historical_data):
                timestamp = datetime.now() - timedelta(days=len(historical_data) - i)
                
                for metric in target_metrics:
                    value = self._extract_metric_value(data_point, metric)
                    self.time_series_analyzer.add_data_point(timestamp, metric, value)
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
        
        return training_results
    
    async def generate_predictions(self, current_data: Dict[str, Any],
                                 prediction_types: Optional[List[PredictionType]] = None,
                                 forecast_horizons: Optional[List[ForecastHorizon]] = None) -> List[Prediction]:
        """Generate predictions for specified types and horizons"""
        if prediction_types is None:
            prediction_types = [
                PredictionType.ENTITY_GROWTH,
                PredictionType.RELATIONSHIP_FORMATION,
                PredictionType.DATA_QUALITY_DRIFT
            ]
        
        if forecast_horizons is None:
            forecast_horizons = [ForecastHorizon.SHORT_TERM, ForecastHorizon.MEDIUM_TERM]
        
        predictions = []
        
        try:
            # Generate predictions for each type and horizon
            for pred_type in prediction_types:
                for horizon in forecast_horizons:
                    prediction = await self._generate_single_prediction(
                        current_data, pred_type, horizon
                    )
                    if prediction:
                        predictions.append(prediction)
            
            # Store predictions
            self.predictions.extend(predictions)
            self.prediction_history.extend(predictions)
            
            # Maintain history size
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            logger.info(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
        
        return predictions
    
    async def _generate_single_prediction(self, data: Dict[str, Any],
                                        pred_type: PredictionType,
                                        horizon: ForecastHorizon) -> Optional[Prediction]:
        """Generate a single prediction"""
        try:
            # Determine target metric and model based on prediction type
            if pred_type == PredictionType.ENTITY_GROWTH:
                target_metric = 'entity_count'
                model_name = 'growth_model'
            elif pred_type == PredictionType.RELATIONSHIP_FORMATION:
                target_metric = 'relationship_count'
                model_name = 'trend_model'
            elif pred_type == PredictionType.DATA_QUALITY_DRIFT:
                target_metric = 'connectivity_ratio'
                model_name = 'quality_model'
            else:
                target_metric = 'entity_count'
                model_name = 'trend_model'
            
            # Get model
            model = self.models.get(model_name)
            
            if not model or not model.is_trained:
                # Use fallback prediction
                return self._generate_fallback_prediction(data, pred_type, horizon, target_metric)
            
            # Make prediction
            predicted_value, confidence_interval, confidence_score = model.predict(data, target_metric)
            
            # Adjust prediction based on forecast horizon
            horizon_multiplier = self._get_horizon_multiplier(horizon)
            adjusted_value = predicted_value * horizon_multiplier
            adjusted_interval = (
                confidence_interval[0] * horizon_multiplier,
                confidence_interval[1] * horizon_multiplier
            )
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            # Calculate validity period
            valid_until = self._calculate_validity_period(horizon)
            
            # Generate explanation
            explanation = self._generate_prediction_explanation(
                pred_type, target_metric, predicted_value, feature_importance
            )
            
            prediction = Prediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=pred_type,
                target_metric=target_metric,
                predicted_value=adjusted_value,
                confidence_interval=adjusted_interval,
                confidence_score=confidence_score,
                forecast_horizon=horizon,
                prediction_date=datetime.now(),
                valid_until=valid_until,
                feature_importance=feature_importance,
                model_used=model.model_name,
                explanation=explanation
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {pred_type}: {e}")
            return None
    
    def _generate_fallback_prediction(self, data: Dict[str, Any], 
                                    pred_type: PredictionType,
                                    horizon: ForecastHorizon,
                                    target_metric: str) -> Optional[Prediction]:
        """Generate fallback prediction when models are unavailable"""
        try:
            # Extract current value
            current_value = self._extract_metric_value(data, target_metric)
            
            # Simple trend-based prediction
            growth_rates = {
                PredictionType.ENTITY_GROWTH: 0.05,  # 5% growth
                PredictionType.RELATIONSHIP_FORMATION: 0.08,  # 8% growth
                PredictionType.DATA_QUALITY_DRIFT: -0.02  # 2% decline
            }
            
            growth_rate = growth_rates.get(pred_type, 0.03)
            horizon_days = self._get_horizon_days(horizon)
            
            # Calculate predicted value
            predicted_value = current_value * (1 + growth_rate * horizon_days / 30)
            
            # Calculate confidence interval (wider for fallback)
            margin = abs(predicted_value) * 0.2  # 20% margin
            confidence_interval = (predicted_value - margin, predicted_value + margin)
            confidence_score = 0.6  # Lower confidence for fallback
            
            valid_until = self._calculate_validity_period(horizon)
            
            explanation = f"Fallback prediction using {growth_rate*100:.1f}% monthly growth rate"
            
            return Prediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=pred_type,
                target_metric=target_metric,
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                confidence_score=confidence_score,
                forecast_horizon=horizon,
                prediction_date=datetime.now(),
                valid_until=valid_until,
                feature_importance={},
                model_used='fallback_trend',
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error generating fallback prediction: {e}")
            return None
    
    async def identify_optimization_opportunities(self, current_data: Dict[str, Any],
                                                predictions: List[Prediction]) -> List[OptimizationRecommendation]:
        """Identify optimization opportunities based on predictions"""
        recommendations = []
        
        try:
            # Analyze predictions for optimization opportunities
            entity_predictions = [p for p in predictions if p.prediction_type == PredictionType.ENTITY_GROWTH]
            
            # Capacity optimization
            if entity_predictions:
                for pred in entity_predictions:
                    current_count = self._extract_metric_value(current_data, 'entity_count')
                    if pred.predicted_value > current_count * 1.5:
                        recommendation = OptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            optimization_type='capacity_scaling',
                            target_area='entity_storage',
                            current_state={
                                'entity_count': current_count,
                                'predicted_growth': (pred.predicted_value - current_count) / current_count
                            },
                            predicted_improvement={
                                'capacity_increase': pred.predicted_value * 0.2,
                                'performance_improvement': 0.15
                            },
                            implementation_effort='medium',
                            confidence_score=pred.confidence_score,
                            business_impact='Prevent performance degradation from capacity constraints',
                            timeline='2-4 weeks',
                            prerequisites=['Infrastructure assessment', 'Budget approval'],
                            risks=['Temporary service impact during scaling'],
                            created_at=datetime.now()
                        )
                        recommendations.append(recommendation)
            
            # Store recommendations
            self.optimization_recommendations.extend(recommendations)
            
            logger.info(f"Identified {len(recommendations)} optimization opportunities")
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
        
        return recommendations
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> float:
        """Extract metric value from metagraph data"""
        try:
            if metric_name == 'entity_count':
                return float(len(data.get('entities', {})))
            elif metric_name == 'relationship_count':
                return float(len(data.get('relationships', {})))
            elif metric_name == 'connectivity_ratio':
                entity_count = len(data.get('entities', {}))
                relationship_count = len(data.get('relationships', {}))
                if entity_count > 0:
                    return float(relationship_count / entity_count)
                return 0.0
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_horizon_multiplier(self, horizon: ForecastHorizon) -> float:
        """Get multiplier for forecast horizon"""
        multipliers = {
            ForecastHorizon.SHORT_TERM: 1.0,
            ForecastHorizon.MEDIUM_TERM: 1.1,
            ForecastHorizon.LONG_TERM: 1.3,
            ForecastHorizon.STRATEGIC: 1.5
        }
        return multipliers.get(horizon, 1.0)
    
    def _get_horizon_days(self, horizon: ForecastHorizon) -> int:
        """Get number of days for forecast horizon"""
        days = {
            ForecastHorizon.SHORT_TERM: 7,
            ForecastHorizon.MEDIUM_TERM: 28,
            ForecastHorizon.LONG_TERM: 90,
            ForecastHorizon.STRATEGIC: 180
        }
        return days.get(horizon, 7)
    
    def _calculate_validity_period(self, horizon: ForecastHorizon) -> datetime:
        """Calculate how long the prediction remains valid"""
        validity_days = {
            ForecastHorizon.SHORT_TERM: 3,
            ForecastHorizon.MEDIUM_TERM: 7,
            ForecastHorizon.LONG_TERM: 14,
            ForecastHorizon.STRATEGIC: 30
        }
        days = validity_days.get(horizon, 3)
        return datetime.now() + timedelta(days=days)
    
    def _generate_prediction_explanation(self, pred_type: PredictionType,
                                       target_metric: str,
                                       predicted_value: float,
                                       feature_importance: Dict[str, float]) -> str:
        """Generate human-readable explanation for the prediction"""
        try:
            # Get top contributing features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if pred_type == PredictionType.ENTITY_GROWTH:
                base_text = f"Predicted entity count: {predicted_value:.0f}"
            elif pred_type == PredictionType.RELATIONSHIP_FORMATION:
                base_text = f"Predicted relationship count: {predicted_value:.0f}"
            elif pred_type == PredictionType.DATA_QUALITY_DRIFT:
                base_text = f"Predicted connectivity ratio: {predicted_value:.2f}"
            else:
                base_text = f"Predicted {target_metric}: {predicted_value:.2f}"
            
            if top_features:
                feature_text = ", ".join([f"{feat} ({imp:.2f})" for feat, imp in top_features])
                explanation = f"{base_text}. Key factors: {feature_text}"
            else:
                explanation = f"{base_text}. Based on historical trends."
            
            return explanation
            
        except Exception:
            return f"Predicted {target_metric}: {predicted_value:.2f}"
    
    def get_predictions_by_type(self, pred_type: PredictionType) -> List[Prediction]:
        """Get all predictions of a specific type"""
        return [p for p in self.predictions if p.prediction_type == pred_type]
    
    def get_predictions_by_horizon(self, horizon: ForecastHorizon) -> List[Prediction]:
        """Get all predictions for a specific time horizon"""
        return [p for p in self.predictions if p.forecast_horizon == horizon]
    
    def export_predictions(self, format: str = 'dict') -> Union[List[Dict], Dict]:
        """Export predictions in various formats"""
        try:
            if format == 'dict':
                return [pred.to_dict() for pred in self.predictions]
            elif format == 'summary':
                summary = {
                    'total_predictions': len(self.predictions),
                    'by_type': {},
                    'by_horizon': {},
                    'average_confidence': 0.0,
                    'recent_predictions': len([p for p in self.predictions 
                                             if (datetime.now() - p.prediction_date).days < 1])
                }
                
                # Count by type and horizon
                for pred in self.predictions:
                    pred_type = pred.prediction_type.value
                    horizon = pred.forecast_horizon.value
                    
                    summary['by_type'][pred_type] = summary['by_type'].get(pred_type, 0) + 1
                    summary['by_horizon'][horizon] = summary['by_horizon'].get(horizon, 0) + 1
                
                # Calculate average confidence
                if self.predictions:
                    total_confidence = sum(p.confidence_score for p in self.predictions)
                    summary['average_confidence'] = total_confidence / len(self.predictions)
                
                return summary
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
            return []
    
    def clear_expired_predictions(self) -> int:
        """Remove expired predictions and return count removed"""
        now = datetime.now()
        initial_count = len(self.predictions)
        
        self.predictions = [p for p in self.predictions if p.valid_until > now]
        
        removed_count = initial_count - len(self.predictions)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired predictions")
        
        return removed_count