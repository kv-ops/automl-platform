"""
Example Integration - Complete MLOps Workflow
==============================================
Place in: automl_platform/examples/mlops_integration.py

Demonstrates complete MLOps workflow with model training, registration,
A/B testing, retraining, and export.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging

# Import AutoML and MLOps components
from automl_platform.config import AutoMLConfig
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.mlflow_registry import MLflowRegistry, ABTestingService, ModelStage
from automl_platform.retraining_service import RetrainingService, RetrainingConfig
from automl_platform.export_service import ModelExporter, ExportConfig
from automl_platform.storage import StorageService
from automl_platform.monitoring import ModelMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLOpsWorkflow:
    """Complete MLOps workflow example"""
    
    def __init__(self):
        # Initialize configuration
        self.config = AutoMLConfig()
        self.config.mlflow_tracking_uri = "http://localhost:5000"
        self.config.environment = "production"
        
        # Initialize services
        self.orchestrator = AutoMLOrchestrator(self.config)
        self.registry = MLflowRegistry(self.config)
        self.exporter = ModelExporter()
        self.ab_testing = ABTestingService(self.registry)
        self.storage = StorageService(self.config)
        self.monitor = ModelMonitor(self.config)
        self.retraining = RetrainingService(
            self.config, 
            self.registry, 
            self.monitor,
            self.storage
        )
    
    def generate_sample_data(self, n_samples=1000, n_features=20):
        """Generate sample data for demonstration"""
        
        # Generate features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        # Generate target (binary classification)
        # Create a pattern for the model to learn
        y = pd.Series(
            (X['feature_0'] + X['feature_1'] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        )
        
        return X, y
    
    async def run_complete_workflow(self):
        """Run complete MLOps workflow"""
        
        logger.info("=" * 80)
        logger.info("Starting Complete MLOps Workflow")
        logger.info("=" * 80)
        
        # ====================================================================
        # Step 1: Initial Model Training
        # ====================================================================
        logger.info("\n📊 Step 1: Training Initial Model")
        
        X_train, y_train = self.generate_sample_data(1000, 20)
        
        # Train model with AutoML
        self.orchestrator.fit(
            X_train, 
            y_train,
            task="classification",
            register_best_model=True,
            model_name="customer_churn_model"
        )
        
        # Get leaderboard
        leaderboard = self.orchestrator.get_leaderboard()
        logger.info(f"Trained {len(leaderboard)} models")
        logger.info(f"Best model: {leaderboard.iloc[0]['model']} with CV score: {leaderboard.iloc[0]['cv_score']:.4f}")
        
        # ====================================================================
        # Step 2: Model Registration and Versioning
        # ====================================================================
        logger.info("\n📝 Step 2: Model Registration in MLflow")
        
        # Model is already registered in fit(), get the info
        model_name = "customer_churn_model"
        model_info = self.orchestrator.training_metadata.get("registered_model", {})
        
        if model_info:
            logger.info(f"Model registered: {model_info['name']} v{model_info['version']}")
            
            # Promote to staging
            self.registry.promote_model(
                model_name,
                model_info['version'],
                ModelStage.STAGING
            )
            logger.info(f"Promoted to STAGING")
            
            # After validation, promote to production
            self.registry.promote_model(
                model_name,
                model_info['version'],
                ModelStage.PRODUCTION
            )
            logger.info(f"Promoted to PRODUCTION")
        
        # ====================================================================
        # Step 3: Model Export for Deployment
        # ====================================================================
        logger.info("\n📦 Step 3: Exporting Model for Deployment")
        
        # Export to ONNX
        onnx_result = self.orchestrator.export_best_model(
            format="onnx",
            sample_data=X_train.head(10)
        )
        
        if onnx_result.get("success"):
            logger.info(f"✅ ONNX export successful:")
            logger.info(f"   - Size: {onnx_result['size_mb']} MB")
            if "quantized_size_mb" in onnx_result:
                logger.info(f"   - Quantized size: {onnx_result['quantized_size_mb']} MB")
                logger.info(f"   - Size reduction: {onnx_result['size_reduction']}")
        
        # Export for edge deployment
        edge_result = self.orchestrator.export_best_model(
            format="edge",
            sample_data=X_train.head(10)
        )
        
        if edge_result.get("exports"):
            logger.info(f"✅ Edge deployment package created:")
            for format_name, format_info in edge_result["exports"].items():
                if format_info.get("success"):
                    logger.info(f"   - {format_name}: {format_info.get('size_mb', 'N/A')} MB")
        
        # ====================================================================
        # Step 4: A/B Testing New Model
        # ====================================================================
        logger.info("\n🔬 Step 4: A/B Testing with Challenger Model")
        
        # Train a challenger model with different configuration
        X_train_new, y_train_new = self.generate_sample_data(1200, 20)
        
        # Create a new config for challenger
        challenger_config = AutoMLConfig()
        challenger_config.algorithms = ["XGBClassifier", "LGBMClassifier"]
        challenger_config.hpo_n_iter = 30
        
        challenger_orchestrator = AutoMLOrchestrator(challenger_config)
        challenger_orchestrator.fit(
            X_train_new,
            y_train_new,
            task="classification"
        )
        
        # Register challenger model
        challenger_version = self.registry.register_model(
            model=challenger_orchestrator.best_pipeline,
            model_name=model_name,
            metrics=challenger_orchestrator.leaderboard[0]['metrics'],
            params=challenger_orchestrator.leaderboard[0]['params'],
            description="Challenger model for A/B test",
            tags={"type": "challenger", "experiment": "xgboost_test"}
        )
        
        # Create A/B test
        test_id = self.ab_testing.create_ab_test(
            model_name=model_name,
            champion_version=1,  # Original model
            challenger_version=challenger_version.version,
            traffic_split=0.2,  # 20% to challenger
            min_samples=50
        )
        
        logger.info(f"Created A/B test: {test_id}")
        
        # Simulate predictions and record results
        logger.info("Simulating A/B test traffic...")
        
        X_test, y_test = self.generate_sample_data(200, 20)
        
        for i in range(len(X_test)):
            # Route prediction
            model_type, version = self.ab_testing.route_prediction(test_id)
            
            # Simulate prediction success (would be actual model prediction)
            success = np.random.random() > 0.2  # 80% success rate
            
            # Add slight advantage to challenger for demo
            if model_type == "challenger":
                success = np.random.random() > 0.15  # 85% success rate
            
            # Record result
            self.ab_testing.record_result(test_id, model_type, success)
        
        # Get test results
        test_results = self.ab_testing.get_test_results(test_id)
        
        logger.info(f"\n📊 A/B Test Results:")
        logger.info(f"   Champion success rate: {test_results['champion']['success_rate']:.2%}")
        logger.info(f"   Challenger success rate: {test_results['challenger']['success_rate']:.2%}")
        logger.info(f"   Improvement: {test_results['improvement']:.1f}%")
        
        if "statistical_significance" in test_results:
            logger.info(f"   P-value: {test_results['statistical_significance']['p_value']:.4f}")
            logger.info(f"   Significant: {test_results['statistical_significance']['significant_at_95']}")
        
        # Conclude test
        conclusion = self.ab_testing.conclude_test(test_id, promote_winner=True)
        logger.info(f"   Winner: {conclusion['winner']}")
        logger.info(f"   Promoted: {conclusion.get('promoted', False)}")
        
        # ====================================================================
        # Step 5: Automated Retraining Check
        # ====================================================================
        logger.info("\n♻️ Step 5: Checking for Retraining Needs")
        
        # Check if model needs retraining
        should_retrain, reason, metrics = self.retraining.should_retrain(model_name)
        
        logger.info(f"Retraining needed: {should_retrain}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Metrics: {metrics}")
        
        if should_retrain:
            # Trigger retraining
            logger.info("Triggering model retraining...")
            
            X_retrain, y_retrain = self.generate_sample_data(1500, 20)
            
            retrain_result = await self.retraining.retrain_model(
                model_name,
                X_retrain,
                y_retrain,
                reason=reason
            )
            
            logger.info(f"Retraining status: {retrain_result['status']}")
            
            if retrain_result['status'] == 'success':
                logger.info(f"New version created: v{retrain_result['new_version']}")
        
        # ====================================================================
        # Step 6: Setup Automated Retraining Schedule
        # ====================================================================
        logger.info("\n⏰ Step 6: Setting up Automated Retraining")
        
        # Configure retraining
        self.retraining.retrain_config = RetrainingConfig(
            drift_threshold=0.3,
            performance_degradation_threshold=0.05,
            min_data_points=500,
            check_frequency="daily",
            retrain_hour=2,
            notify_on_drift=True,
            notify_on_retrain=True
        )
        
        # Create schedule (Airflow or Prefect)
        schedule = self.retraining.create_retraining_schedule()
        
        if schedule:
            logger.info("✅ Automated retraining schedule created")
            logger.info("   - Frequency: daily at 2 AM")
            logger.info("   - Drift threshold: 0.3")
            logger.info("   - Performance threshold: 5% degradation")
        
        # ====================================================================
        # Step 7: Model Comparison and History
        # ====================================================================
        logger.info("\n📈 Step 7: Model Version Comparison")
        
        # Get model history
        history = self.registry.get_model_history(model_name, limit=5)
        
        logger.info(f"Model version history ({len(history)} versions):")
        for version_info in history:
            logger.info(f"   v{version_info['version']}: {version_info['stage']} - Created: {version_info['created_at']}")
        
        # Compare versions if we have multiple
        if len(history) >= 2:
            comparison = self.registry.compare_models(
                model_name,
                history[0]['version'],
                history[1]['version']
            )
            
            logger.info(f"\nComparison v{history[0]['version']} vs v{history[1]['version']}:")
            
            if "metric_diff" in comparison:
                for metric, diff in comparison["metric_diff"].items():
                    logger.info(f"   {metric}: {diff['relative']:.1f}% change")
        
        # ====================================================================
        # Summary
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("✅ MLOps Workflow Complete!")
        logger.info("=" * 80)
        
        summary = {
            "models_trained": len(self.orchestrator.leaderboard),
            "best_model": self.orchestrator.leaderboard[0]['model'] if self.orchestrator.leaderboard else None,
            "versions_created": len(history),
            "ab_tests_run": 1,
            "exports_created": ["onnx", "edge"],
            "retraining_scheduled": schedule is not None
        }
        
        logger.info("\n📊 Summary:")
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")
        
        return summary


async def main():
    """Main execution function"""
    
    workflow = MLOpsWorkflow()
    
    try:
        results = await workflow.run_complete_workflow()
        logger.info("\n✅ All MLOps operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        raise


if __name__ == "__main__":
    # Run the complete workflow
    asyncio.run(main())
