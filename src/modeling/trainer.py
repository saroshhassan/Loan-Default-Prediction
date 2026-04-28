"""Model training pipeline with cross-validation and evaluation."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict 
import logging

from .models import ModelFactory
from ..evaluation.metrics import ModelEvaluator, CrossValidationEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate models with cross-validation."""
    
    def __init__(
        self,
        model_type: str,
        params: Dict[str, Any],
        random_state: int = 42,
        cv_folds: int = 5,
        stratified: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('baseline', 'xgboost', 'lightgbm')
            params: Model hyperparameters
            random_state: Random seed
            cv_folds: Number of cross-validation folds
            stratified: Whether to use stratified K-fold
        """
        self.model_type = model_type
        self.params = params.copy()
        self.params['random_state'] = random_state
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.stratified = stratified
        
        self.model = None
        self.cv_results = None
        self.cv_predictions = None
        self.cv_probabilities = None
        self.trained_models = []
    
    def train_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Train model on full training data (no CV).
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Self for method chaining
        """
        self.model = ModelFactory.create(self.model_type, self.params)
        self.model.fit(X_train, y_train)
        
        logger.info(f"Trained {self.model_type} model on {len(X_train)} samples")
        return self
    
    def train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with CV results
        """
        # Ensure y is numpy array (not pandas Series)
        if hasattr(y, 'values'):
            y = y.values
        
        if self.stratified:
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        
        # Create base model
        base_model = ModelFactory.create(self.model_type, self.params)
        
        # Scoring metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        # Cross-validate
        cv_scores = cross_validate(
            base_model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Get probability predictions for detailed evaluation
        self.cv_probabilities = cross_val_predict(
            base_model, X, y, 
            cv=cv_splitter,
            method='predict_proba'
        )[:, 1]
        
        # Get binary predictions
        self.cv_predictions = (self.cv_probabilities >= 0.5).astype(int)
        
        # Evaluate each fold in detail
        fold_evaluations = []
        for fold_idx in range(self.cv_folds):
            # Get fold mask
            fold_test_indices = list(cv_splitter.split(X, y))[fold_idx][1]
            y_fold_test = y[fold_test_indices]
            y_fold_pred_proba = self.cv_probabilities[fold_test_indices]
            y_fold_pred = self.cv_predictions[fold_test_indices]
            
            # Evaluate fold
            fold_metrics = ModelEvaluator.evaluate_binary_classification(
                y_fold_test,
                y_fold_pred_proba,
                y_fold_pred
            )
            fold_evaluations.append(fold_metrics)
        
        # Store results
        self.cv_results = {
            'sklearn_scores': cv_scores,
            'detailed_metrics': fold_evaluations,
            'aggregated_metrics': CrossValidationEvaluator.aggregate_cv_scores(fold_evaluations)
        }
        
        logger.info(f"Completed {self.cv_folds}-fold CV for {self.model_type}")
        
        return self.cv_results
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train on train set and evaluate on test set.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with training and evaluation results
        """
        # Train
        self.train_single(X_train, y_train)
        
        # Predict
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        metrics = ModelEvaluator.evaluate_binary_classification(
            y_test,
            y_pred_proba,
            y_pred
        )
        
        results = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model': self.model,
            'metrics': metrics,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        logger.info(f"Train/Test evaluation for {self.model_type}: ROC-AUC = {metrics['roc_auc']:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_single() or train_cv() first.")
        
        if self.model_type == 'baseline':
            # Logistic Regression coefficients
            if hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])
                names = feature_names or [f"Feature_{i}" for i in range(len(importances))]
            else:
                raise ValueError("Model does not have feature importance")
        elif self.model_type in ['xgboost', 'lightgbm']:
            # Tree-based feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                names = feature_names or [f"Feature_{i}" for i in range(len(importances))]
            else:
                raise ValueError("Model does not have feature importance")
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")
        
        importance_df = pd.DataFrame({
            'feature': names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def save_model(self, output_path: str) -> None:
        """
        Save trained model.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, output_path)
        logger.info(f"Saved model to {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")


class TrainingPipeline:
    """Complete training pipeline orchestrator."""
    
    def __init__(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        random_state: int = 42,
        cv_folds: int = 5
    ):
        """
        Initialize pipeline.
        
        Args:
            model_configs: Dictionary of model configurations
            random_state: Random seed
            cv_folds: Number of CV folds
        """
        self.model_configs = model_configs
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.trainers = {}
        self.results = {}
    
    def train_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X: Features
            y: Labels
            use_cv: Whether to use cross-validation
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        for model_type, config in self.model_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_type}")
            logger.info(f"{'='*60}")
            
            # Create trainer
            trainer = ModelTrainer(
                model_type=config['type'],
                params=config['params'],
                random_state=self.random_state,
                cv_folds=self.cv_folds
            )
            
            # Train
            if use_cv:
                cv_results = trainer.train_cv(X, y)
                results[model_type] = {
                    'trainer': trainer,
                    'cv_results': cv_results,
                    'aggregated_metrics': cv_results['aggregated_metrics']
                }
            else:
                trainer.train_single(X, y)
                results[model_type] = {
                    'trainer': trainer,
                    'model': trainer.model
                }
            
            self.trainers[model_type] = trainer
            
            # Print results
            if use_cv and 'aggregated_metrics' in results[model_type]:
                CrossValidationEvaluator.print_cv_results(
                    results[model_type]['aggregated_metrics'],
                    model_type
                )
        
        self.results = results
        return results
    
    def save_all_models(self, output_dir: str) -> None:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        
        for model_type, trainer in self.trainers.items():
            model_path = output_dir / f"{model_type}_model.pkl"
            trainer.save_model(str(model_path))
        
        logger.info(f"Saved all models to {output_dir}")
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, ModelTrainer]:
        """
        Get best performing model based on metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, trainer)
        """
        if not self.results:
            raise ValueError("No training results. Run train_all_models() first.")
        
        best_model_name = None
        best_score = -1
        
        for model_type, result in self.results.items():
            if 'aggregated_metrics' in result:
                score = result['aggregated_metrics'][metric]['mean']
                if score > best_score:
                    best_score = score
                    best_model_name = model_type
        
        if best_model_name is None:
            raise ValueError(f"Could not find best model for metric: {metric}")
        
        return best_model_name, self.trainers[best_model_name]
