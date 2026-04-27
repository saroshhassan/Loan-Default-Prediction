"""Model evaluation metrics and utilities."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, matthews_corrcoef, log_loss
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    @staticmethod
    def evaluate_binary_classification(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_binary: np.ndarray = None,
        threshold: float = 0.5
    ) -> dict:
        """
        Evaluate binary classification model.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            y_pred_binary: Predicted binary labels (if None, created from y_pred_proba)
            threshold: Probability threshold for binary prediction
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Probability-based metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        log_loss_val = log_loss(y_true, y_pred_proba)
        
        # Binary classification metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Precision-Recall curve
        pr_auc = auc(*precision_recall_curve(y_true, y_pred_proba)[:2])
        
        metrics = {
            # Core metrics
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': log_loss_val,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'specificity': specificity,
            'npv': npv,
            
            # Confusion matrix components
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            
            # Thresholds
            'threshold': threshold
        }
        
        return metrics
    
    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
        """
        Get ROC curve coordinates.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_pred_proba)
    
    @staticmethod
    def get_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
        """
        Get precision-recall curve coordinates.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_pred_proba)
    
    @staticmethod
    def get_calibration_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> tuple:
        """
        Get calibration curve (expected vs actual probabilities).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (prob_true, prob_pred)
        """
        return calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    @staticmethod
    def print_metrics(metrics: dict, model_name: str = "Model") -> None:
        """
        Print formatted evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of model for display
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics: {model_name}")
        print(f"{'='*60}")
        
        print(f"\nProbability Metrics:")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
        print(f"  Log Loss:          {metrics['log_loss']:.4f}")
        
        print(f"\nClassification Metrics (threshold={metrics['threshold']}):")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        print(f"  F1-Score:          {metrics['f1']:.4f}")
        print(f"  MCC:               {metrics['mcc']:.4f}")
        print(f"  NPV:               {metrics['npv']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['tp']:>5}  |  FP: {metrics['fp']:>5}")
        print(f"  FN: {metrics['fn']:>5}  |  TN: {metrics['tn']:>5}")
        print(f"{'='*60}\n")


class CrossValidationEvaluator:
    """Cross-validation evaluation utilities."""
    
    @staticmethod
    def aggregate_cv_scores(cv_scores_list: list) -> dict:
        """
        Aggregate cross-validation scores.
        
        Args:
            cv_scores_list: List of metric dictionaries from each fold
            
        Returns:
            Dictionary with aggregated statistics
        """
        metrics_names = cv_scores_list[0].keys()
        aggregated = {}
        
        for metric in metrics_names:
            values = [scores[metric] for scores in cv_scores_list]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'folds': values
            }
        
        return aggregated
    
    @staticmethod
    def print_cv_results(aggregated_scores: dict, model_name: str = "Model") -> None:
        """
        Print cross-validation results.
        
        Args:
            aggregated_scores: Aggregated CV scores dictionary
            model_name: Name of model
        """
        print(f"\n{'='*70}")
        print(f"Cross-Validation Results: {model_name}")
        print(f"{'='*70}\n")
        
        print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 70)
        
        for metric, stats in aggregated_scores.items():
            if metric != 'threshold':
                print(f"{metric:<20} {stats['mean']:>11.4f} {stats['std']:>11.4f} "
                      f"{stats['min']:>11.4f} {stats['max']:>11.4f}")
        
        print(f"{'='*70}\n")
