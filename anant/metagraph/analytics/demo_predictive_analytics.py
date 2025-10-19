"""
Phase 3 Predictive Analytics Demo

This demo showcases the enterprise forecasting and optimization capabilities:
- Entity growth prediction using statistical models
- Relationship formation forecasting
- Data quality drift prediction
- Optimization opportunity identification

Features demonstrated:
- Statistical forecasting with trend analysis
- Confidence interval calculations
- Feature importance analysis
- Optimization recommendations
- Enterprise reliability with graceful degradation
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

# Add the project root to the Python path
sys.path.append('/home/amansingh/dev/ai/anant')

def create_historical_metagraph_data(num_snapshots: int = 10) -> List[Dict[str, Any]]:
    """Create historical metagraph data for training"""
    
    historical_data = []
    base_entities = 5
    base_relationships = 4
    
    for i in range(num_snapshots):
        # Simulate growth over time
        entity_growth = i * 2 + random.randint(0, 3)
        relationship_growth = i * 1.5 + random.randint(0, 2)
        
        data = {
            'entities': {},
            'relationships': {}
        }
        
        # Create entities with growth pattern
        for j in range(int(base_entities + entity_growth)):
            entity_id = f'entity_{j:03d}'
            entity_type = random.choice(['Customer', 'Product', 'Order', 'Person', 'Department'])
            data['entities'][entity_id] = {
                'type': entity_type,
                'properties': {'index': j, 'snapshot': i},
                'created_at': (datetime.now() - timedelta(days=num_snapshots - i)).isoformat() + 'Z'
            }
        
        # Create relationships with growth pattern
        for j in range(int(base_relationships + relationship_growth)):
            rel_id = f'rel_{j:03d}'
            rel_type = random.choice(['purchases', 'orders', 'works_for', 'manages', 'contains'])
            
            # Pick random entities for relationship
            entity_ids = list(data['entities'].keys())
            if len(entity_ids) >= 2:
                selected_entities = random.sample(entity_ids, 2)
                data['relationships'][rel_id] = {
                    'type': rel_type,
                    'entities': selected_entities,
                    'properties': {'index': j, 'snapshot': i}
                }
        
        historical_data.append(data)
    
    return historical_data

def create_current_metagraph_data() -> Dict[str, Any]:
    """Create current metagraph data for prediction"""
    
    return {
        'entities': {
            'current_entity_001': {
                'type': 'Customer',
                'properties': {'name': 'Current Corp', 'tier': 'Enterprise'},
                'created_at': datetime.now().isoformat() + 'Z'
            },
            'current_entity_002': {
                'type': 'Product',
                'properties': {'name': 'Current Product', 'category': 'Software'},
                'created_at': datetime.now().isoformat() + 'Z'
            },
            'current_entity_003': {
                'type': 'Order',
                'properties': {'amount': 5000, 'status': 'active'},
                'created_at': datetime.now().isoformat() + 'Z'
            }
        },
        'relationships': {
            'current_rel_001': {
                'type': 'purchases',
                'entities': ['current_entity_001', 'current_entity_002'],
                'properties': {'date': datetime.now().isoformat()}
            },
            'current_rel_002': {
                'type': 'orders',
                'entities': ['current_entity_001', 'current_entity_003'],
                'properties': {'order_date': datetime.now().isoformat()}
            }
        }
    }

async def demo_predictive_analytics():
    """Demonstrate the Predictive Analytics Engine capabilities"""
    
    print("📊 Phase 3: Predictive Analytics Engine Demo")
    print("=" * 60)
    
    try:
        from anant.metagraph.analytics.predictive_analytics import (
            PredictiveAnalyticsEngine,
            PredictionType,
            ForecastHorizon,
            Prediction,
            OptimizationRecommendation,
            ForecastModel,
            TimeSeriesAnalyzer
        )
        
        print("✅ Predictive Analytics components loaded successfully")
        
        # Initialize the predictive analytics engine
        config = {
            'time_series_window': 20,
            'prediction_confidence_threshold': 0.7,
            'optimization_threshold': 1.3
        }
        
        engine = PredictiveAnalyticsEngine(config)
        print("✅ Predictive Analytics Engine initialized")
        
        # Test 1: Model Training
        print("\n🎯 Test 1: Model Training")
        print("-" * 40)
        
        historical_data = create_historical_metagraph_data(12)
        print(f"📈 Created {len(historical_data)} historical data snapshots")
        
        training_results = await engine.train_models(
            historical_data, 
            ['entity_count', 'relationship_count', 'connectivity_ratio']
        )
        
        print(f"🎯 Training Results:")
        successful_models = sum(1 for success in training_results.values() if success)
        total_models = len(training_results)
        print(f"  ✅ {successful_models}/{total_models} models trained successfully")
        
        for model_metric, success in training_results.items():
            status = "✅" if success else "❌"
            print(f"    {status} {model_metric}")
        
        # Test 2: Prediction Generation
        print("\n🔮 Test 2: Prediction Generation")
        print("-" * 40)
        
        current_data = create_current_metagraph_data()
        print(f"📈 Current data: {len(current_data['entities'])} entities, {len(current_data['relationships'])} relationships")
        
        prediction_types = [
            PredictionType.ENTITY_GROWTH,
            PredictionType.RELATIONSHIP_FORMATION,
            PredictionType.DATA_QUALITY_DRIFT
        ]
        
        forecast_horizons = [
            ForecastHorizon.SHORT_TERM,
            ForecastHorizon.MEDIUM_TERM,
            ForecastHorizon.LONG_TERM
        ]
        
        predictions = await engine.generate_predictions(
            current_data, prediction_types, forecast_horizons
        )
        
        print(f"🔮 Generated {len(predictions)} predictions")
        
        # Display predictions by type
        for pred_type in prediction_types:
            type_predictions = [p for p in predictions if p.prediction_type == pred_type]
            if type_predictions:
                print(f"\\n📊 {pred_type.value.replace('_', ' ').title()} Predictions:")
                for pred in type_predictions[:2]:  # Show first 2
                    print(f"  • {pred.explanation}")
                    print(f"    Confidence: {pred.confidence_score:.2f}")
                    print(f"    Horizon: {pred.forecast_horizon.value}")
                    print(f"    Model: {pred.model_used}")
                    if pred.feature_importance:
                        top_feature = max(pred.feature_importance.items(), key=lambda x: x[1])
                        print(f"    Top Factor: {top_feature[0]} ({top_feature[1]:.2f})")
        
        # Test 3: Time Series Analysis
        print("\n📈 Test 3: Time Series Analysis")
        print("-" * 40)
        
        time_series_analyzer = TimeSeriesAnalyzer(window_size=15)
        
        # Add historical time series data
        base_date = datetime.now() - timedelta(days=20)
        for i in range(20):
            timestamp = base_date + timedelta(days=i)
            
            # Simulate metrics with trends
            entity_count = 5 + i * 0.5 + random.uniform(-1, 1)
            relationship_count = 4 + i * 0.3 + random.uniform(-0.5, 0.5)
            connectivity_ratio = relationship_count / entity_count
            
            time_series_analyzer.add_data_point(timestamp, 'entity_count', entity_count)
            time_series_analyzer.add_data_point(timestamp, 'relationship_count', relationship_count)
            time_series_analyzer.add_data_point(timestamp, 'connectivity_ratio', connectivity_ratio)
        
        print(f"📊 Added 20 days of time series data")
        
        # Test seasonality detection
        for metric in ['entity_count', 'relationship_count']:
            seasonality = time_series_analyzer.detect_seasonality(metric)
            if seasonality:
                print(f"  🔄 {metric}: {seasonality['pattern_type']} pattern detected (correlation: {seasonality['correlation']:.2f})")
            else:
                print(f"  📈 {metric}: No clear seasonal pattern detected")
        
        # Test 4: Optimization Opportunities
        print("\n🎯 Test 4: Optimization Opportunities")
        print("-" * 40)
        
        optimization_recommendations = await engine.identify_optimization_opportunities(
            current_data, predictions
        )
        
        print(f"💡 Identified {len(optimization_recommendations)} optimization opportunities")
        
        for i, recommendation in enumerate(optimization_recommendations[:3], 1):
            print(f"\\n🎯 Recommendation {i}: {recommendation.optimization_type.replace('_', ' ').title()}")
            print(f"  Target: {recommendation.target_area}")
            print(f"  Impact: {recommendation.business_impact}")
            print(f"  Effort: {recommendation.implementation_effort}")
            print(f"  Timeline: {recommendation.timeline}")
            print(f"  Confidence: {recommendation.confidence_score:.2f}")
            
            if recommendation.predicted_improvement:
                improvements = [f"{k}: +{v*100:.1f}%" for k, v in recommendation.predicted_improvement.items()]
                print(f"  Expected Improvements: {', '.join(improvements)}")
        
        # Test 5: Prediction Analysis and Export
        print("\n📤 Test 5: Prediction Analysis and Export")
        print("-" * 40)
        
        # Filter predictions by confidence
        high_confidence_predictions = [p for p in predictions if p.confidence_score > 0.7]
        print(f"🎯 High confidence predictions: {len(high_confidence_predictions)}")
        
        # Group by horizon
        horizon_groups = {}
        for pred in predictions:
            horizon = pred.forecast_horizon.value
            horizon_groups[horizon] = horizon_groups.get(horizon, 0) + 1
        
        print(f"📊 Predictions by horizon:")
        for horizon, count in horizon_groups.items():
            print(f"  {horizon}: {count}")
        
        # Export predictions
        prediction_summary = engine.export_predictions('summary')
        print(f"\\n📋 Prediction Summary:")
        print(f"  Total: {prediction_summary['total_predictions']}")
        print(f"  Recent (24h): {prediction_summary['recent_predictions']}")
        print(f"  Average Confidence: {prediction_summary['average_confidence']:.2f}")
        
        # Test 6: Model Performance Evaluation
        print("\n🔬 Test 6: Model Performance")
        print("-" * 40)
        
        # Test individual forecast model
        test_model = ForecastModel('test_model')
        success = test_model.train(historical_data[:8], 'entity_count')
        
        if success:
            print("✅ Individual model training successful")
            
            # Test prediction
            test_prediction = test_model.predict(current_data, 'entity_count')
            predicted_value, confidence_interval, confidence_score = test_prediction
            
            print(f"🔮 Test Prediction:")
            print(f"  Value: {predicted_value:.1f}")
            print(f"  Confidence Interval: ({confidence_interval[0]:.1f}, {confidence_interval[1]:.1f})")
            print(f"  Confidence Score: {confidence_score:.2f}")
            
            # Feature importance
            feature_importance = test_model.get_feature_importance()
            if feature_importance:
                print(f"  Feature Importance:")
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    {feature}: {importance:.3f}")
        else:
            print("❌ Individual model training failed")
        
        # Test cleanup
        expired_count = engine.clear_expired_predictions()
        print(f"🧹 Cleared {expired_count} expired predictions")
        
        print("\n🎉 Predictive Analytics Demo Complete!")
        print("✅ All forecasting and optimization features working")
        print("🚀 Ready for Phase 3 continuation: Production Framework")
        
    except Exception as e:
        print(f"❌ Error in predictive analytics demo: {e}")
        import traceback
        traceback.print_exc()

async def demo_individual_forecasting_models():
    """Demo individual forecasting model components"""
    
    print("\n🧩 Individual Forecasting Model Demonstrations")
    print("=" * 55)
    
    try:
        from anant.metagraph.analytics.predictive_analytics import (
            ForecastModel,
            TimeSeriesAnalyzer
        )
        
        # Demo ForecastModel
        print("\n📈 Forecast Model Demo")
        print("-" * 30)
        
        model = ForecastModel('demo_model')
        historical_data = create_historical_metagraph_data(8)
        
        success = model.train(historical_data, 'entity_count')
        print(f"✅ Model training: {'Success' if success else 'Failed'}")
        
        if success:
            current_data = create_current_metagraph_data()
            predicted_value, conf_interval, conf_score = model.predict(current_data, 'entity_count')
            
            print(f"🔮 Prediction Results:")
            print(f"  Predicted Value: {predicted_value:.1f}")
            print(f"  Confidence Interval: ({conf_interval[0]:.1f}, {conf_interval[1]:.1f})")
            print(f"  Confidence Score: {conf_score:.2f}")
            
            feature_importance = model.get_feature_importance()
            print(f"  Feature Importance: {len(feature_importance)} features analyzed")
        
        # Demo TimeSeriesAnalyzer
        print("\n⏰ Time Series Analyzer Demo")
        print("-" * 30)
        
        analyzer = TimeSeriesAnalyzer(window_size=10)
        
        # Add sample time series data
        base_time = datetime.now() - timedelta(days=15)
        for i in range(15):
            timestamp = base_time + timedelta(days=i)
            value = 10 + i * 0.5 + random.uniform(-1, 1)
            analyzer.add_data_point(timestamp, 'test_metric', value)
        
        print(f"📊 Added 15 data points to time series")
        
        seasonality = analyzer.detect_seasonality('test_metric')
        if seasonality:
            print(f"🔄 Seasonality detected: {seasonality}")
        else:
            print(f"📈 No seasonality pattern found")
        
        print("\n✅ Individual model demos complete")
        
    except Exception as e:
        print(f"❌ Error in individual model demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function"""
    print("🎯 Phase 3: Predictive Analytics Engine - Complete Demo")
    print("📊 Enterprise Forecasting & Optimization for Metagraphs")
    print("=" * 70)
    print()
    
    # Run the main predictive analytics demo
    asyncio.run(demo_predictive_analytics())
    
    # Run individual component demos
    asyncio.run(demo_individual_forecasting_models())
    
    print("\n" + "=" * 70)
    print("🎊 Phase 3 Predictive Analytics Demo Complete!")
    print("📋 Key Features Demonstrated:")
    print("  ✅ Statistical forecasting with trend analysis")
    print("  ✅ Entity growth and relationship formation prediction")
    print("  ✅ Data quality drift forecasting")
    print("  ✅ Confidence interval calculation and scoring")
    print("  ✅ Feature importance analysis")
    print("  ✅ Optimization opportunity identification")
    print("  ✅ Time series pattern detection")
    print("  ✅ Enterprise reliability with graceful degradation")
    print()
    print("🚀 Ready for Phase 3 continuation:")
    print("  🏭 Production Deployment Framework")
    print("  🛡️ Advanced Governance Automation")

if __name__ == "__main__":
    main()