"""Main training pipeline script."""

import sys
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs, load_config
from src.data_ingestion.loader import DataLoader, load_train_test_split
from src.feature_engineering.preprocessor import prepare_data_for_training
from src.modeling.trainer import TrainingPipeline
from src.evaluation.metrics import ModelEvaluator, CrossValidationEvaluator
from src.eda.analyzer import EDAAnalyzer
from src.decisioning.engine import DecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("STARTING RISK ASSESSMENT MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load configurations
    logger.info("\n1. Loading configurations...")
    configs = load_all_configs("configs")
    model_config = configs['model_config']
    
    print("\nLoaded configurations:")
    print(f"  - Models to train: {model_config['models_to_train']}")
    print(f"  - CV folds: {model_config['cross_validation']['n_splits']}")
    
    # Load and validate data
    logger.info("\n2. Loading data from data/raw/loan_risk_prediction_dataset.csv...")
    loader = DataLoader("data/raw/loan_risk_prediction_dataset.csv")
    df = loader.load_and_validate()
    
    summary = loader.get_data_summary(df)
    print(f"\nData Summary:")
    print(f"  - Shape: {summary['shape']}")
    print(f"  - Target distribution: {summary['target_distribution']}")
    print(f"  - Target ratio (approved): {summary['target_ratio']:.2%}")
    
    # Exploratory Data Analysis
    logger.info("\n3. Performing Exploratory Data Analysis...")
    """eda_analyzer = EDAAnalyzer(output_dir="reports")
    
    numerical_cols = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience']
    categorical_cols = ['Gender', 'Education', 'City', 'EmploymentType']
    
    eda_report = eda_analyzer.generate_summary_report(
        df,
        target_col='LoanApproved',
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        save=True
    )
    
    print(f"\nEDA Summary:")
    print(f"  - Feature signal strength (Information Value)")
    print(eda_report['information_value'].head(3).to_string())
    print(f"\n  - Feature importance (Univariate ROC-AUC)")
    print(eda_report['roc_auc_per_feature'].head(3).to_string())"""
    
    # Split data
    logger.info("\n3. Splitting data into train/test (80/20)...")
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['LoanApproved'])
    y = df['LoanApproved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    
    print(f"  - Train size: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"  - Test size: {len(X_test)} ({len(X_test)/len(X):.1%})")
    print(f"  - Train positive ratio: {y_train.sum()/len(y_train):.2%}")
    print(f"  - Test positive ratio: {y_test.sum()/len(y_test):.2%}")
    
    # Feature engineering
    logger.info("\n4. Preprocessing features...")
    X_train_processed, X_test_processed, preprocessor = prepare_data_for_training(
        X_train, X_test,
        handle_missing=True,
        remove_outliers=False
    )
    
    print(f"\nFeature Engineering:")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - Processed features: {X_train_processed.shape[1]}")
    print(f"  - Feature names: {preprocessor.get_feature_names()[:5]}... (showing first 5)")
    
    # Train models
    logger.info("\n5. Training models with cross-validation...")
    
    # Filter models to train from config
    models_to_train = model_config['models_to_train']
    model_configs_filtered = {
        model_type: model_config[model_type]
        for model_type in models_to_train
        if model_type in model_config
    }
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        model_configs=model_configs_filtered,
        random_state=model_config['training']['random_seed'],
        cv_folds=model_config['cross_validation']['n_splits']
    )
    
    # Train all models with CV
    cv_results = pipeline.train_all_models(X_train_processed, y_train, use_cv=True)
    
    # Also train on full data for comparison and deployment
    logger.info("\n6. Training models on full training data...")
    train_test_results = {}
    for model_type in models_to_train:
        if model_type in pipeline.trainers:
            trainer = pipeline.trainers[model_type]
            result = trainer.train_and_evaluate(
                X_train_processed, y_train,
                X_test_processed, y_test
            )
            train_test_results[model_type] = result
            
            print(f"\n{model_type} - Test Set Performance:")
            ModelEvaluator.print_metrics(result['metrics'], model_type)
    
    # Find best model
    logger.info("\n7. Identifying best model...")
    best_model_name, best_trainer = pipeline.get_best_model(metric='roc_auc')
    print(f"\nBest performing model: {best_model_name}")
    if 'aggregated_metrics' in cv_results[best_model_name]:
        best_roc_auc = cv_results[best_model_name]['aggregated_metrics']['roc_auc']['mean']
        print(f"  - CV ROC-AUC: {best_roc_auc:.4f}")
    
    # Save models
    logger.info("\n8. Saving trained models...")
    output_dir = "models"
    Path(output_dir).mkdir(exist_ok=True)
    
    for model_type, trainer in pipeline.trainers.items():
        trainer.save_model(f"{output_dir}/{model_type}_model.pkl")
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
    logger.info(f"Saved preprocessor to {output_dir}/preprocessor.pkl")
    
    # Get feature importance for best model
    logger.info("\n9. Computing feature importance for best model...")
    feature_names = preprocessor.get_feature_names()
    importance_df = best_trainer.get_feature_importance(feature_names)
    
    print(f"\nTop 10 features for {best_model_name}:")
    print(importance_df.head(10).to_string(index=False))
    
    # Decision Engine Analysis
    logger.info("\n10. Applying Decision Engine...")
    decision_engine = DecisionEngine(output_dir="reports")
    
    # Get predictions on test set for decision analysis
    best_model_obj = train_test_results[best_model_name]['model']
    y_pred_proba_test = best_model_obj.predict_proba(X_test_processed)[:, 1]
    
    decision_report = decision_engine.generate_decision_report(
        y_true=y_test.values,
        y_pred_proba=y_pred_proba_test,
        loan_amounts=None,
        approve_threshold=0.30,
        reject_threshold=0.70,
        save=True
    )
    
    print(f"\nDecision Engine Results (Test Set):")
    metrics_de = decision_report['business_metrics']
    print(f"  - Approval Rate: {metrics_de['approval_rate']:.2%}")
    print(f"  - Default Rate in Approved: {metrics_de['default_rate_in_approved']:.2%}")
    print(f"  - False Negative Rate: {metrics_de['false_negative_rate']:.2%}")
    print(f"  - Rejected Defaults Caught: {metrics_de['rejected_defaults']}")
    
    # Save summary report
    logger.info("\n11. Saving summary report...")
    save_training_summary(
        cv_results=cv_results,
        train_test_results=train_test_results,
        best_model=best_model_name,
        importance_df=importance_df,
        decision_report=decision_report,
        output_dir="reports"
    )
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return pipeline, best_trainer, preprocessor


def save_training_summary(
    cv_results: dict,
    train_test_results: dict,
    best_model: str,
    importance_df,
    decision_report: dict = None,
    output_dir: str = "reports"
) -> None:
    """
    Save training summary report.
    
    Args:
        cv_results: Cross-validation results
        train_test_results: Train/test evaluation results
        best_model: Best model name
        importance_df: Feature importance dataframe
        decision_report: Decision engine report (optional)
        output_dir: Output directory
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save CV results
    cv_report_path = output_dir / "cv_results.txt"
    with open(cv_report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for model_type, result in cv_results.items():
            f.write(f"\n{model_type.upper()}\n")
            f.write("-"*80 + "\n")
            
            if 'aggregated_metrics' in result:
                metrics = result['aggregated_metrics']
                f.write(f"{'Metric':<20} {'Mean':<12} {'Std':<12}\n")
                f.write("-"*44 + "\n")
                for metric, stats in metrics.items():
                    if metric != 'threshold':
                        f.write(f"{metric:<20} {stats['mean']:>11.4f} {stats['std']:>11.4f}\n")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    # Save training summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Best Model: {best_model}\n\n")
        
        f.write("Train/Test Results:\n")
        f.write("-"*80 + "\n")
        for model_type, result in train_test_results.items():
            f.write(f"\n{model_type}:\n")
            metrics = result['metrics']
            f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Top 10 Feature Importance\n")
        f.write("="*80 + "\n")
        f.write(importance_df.head(10).to_string(index=False))
    
    logger.info(f"Saved training report to {summary_path}")


if __name__ == "__main__":
    pipeline, best_trainer, preprocessor = main()
