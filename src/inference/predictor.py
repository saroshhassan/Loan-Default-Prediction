"""Inference utilities for making predictions with trained models."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Make predictions with trained models."""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        X_processed = self._preprocess(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions for both classes (n_samples, 2)
        """
        X_processed = self._preprocess(X)
        return self.model.predict_proba(X_processed)
    
    def predict_proba_positive(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probability for positive class only.
        
        Args:
            X: Input features
            
        Returns:
            Probability for positive class (1D array)
        """
        return self.predict_proba(X)[:, 1]
    
    def _preprocess(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess features.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features
        """
        if isinstance(X, np.ndarray):
            # If already numpy array, assume already in correct format
            return X
        elif isinstance(X, pd.DataFrame):
            return self.preprocessor.transform(X)
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")


def batch_predict(
    model_path: str,
    preprocessor_path: str,
    data_path: str,
    output_path: str = None,
    probability_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Make predictions on a dataset and save results.
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        data_path: Path to input data (CSV or Parquet)
        output_path: Path to save predictions (if None, not saved)
        probability_threshold: Threshold for binary prediction
        
    Returns:
        DataFrame with predictions and probabilities
    """
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format")
    
    # Remove target if present
    X = df.drop(columns=['LoanApproved'], errors='ignore')
    
    # Make predictions
    predictor = ModelPredictor(model_path, preprocessor_path)
    y_pred = predictor.predict(X)
    y_pred_proba = predictor.predict_proba_positive(X)
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': y_pred,
        'probability': y_pred_proba,
        'risk_level': pd.cut(
            y_pred_proba,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    })
    
    # Save if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
    
    return results


def explain_prediction(
    model_path: str,
    preprocessor_path: str,
    X_single: Union[pd.DataFrame, dict],
    feature_names: list = None,
    n_top_features: int = 5
) -> dict:
    """
    Explain a single prediction (for baseline models).
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        X_single: Single sample to explain
        feature_names: List of feature names
        n_top_features: Number of top features to show
        
    Returns:
        Dictionary with explanation
    """
    if isinstance(X_single, dict):
        X_single = pd.DataFrame([X_single])
    elif isinstance(X_single, pd.DataFrame):
        if len(X_single) > 1:
            X_single = X_single.iloc[[0]]
    
    predictor = ModelPredictor(model_path, preprocessor_path)
    model = predictor.model
    
    # Get prediction
    y_pred = predictor.predict(X_single)[0]
    y_pred_proba = predictor.predict_proba_positive(X_single)[0]
    
    explanation = {
        'prediction': int(y_pred),
        'probability': float(y_pred_proba),
        'risk_level': 'Low Risk' if y_pred_proba < 0.3 else ('High Risk' if y_pred_proba > 0.7 else 'Medium Risk')
    }
    
    # Get feature importance if available
    if hasattr(model, 'coef_'):
        # Logistic Regression
        coefficients = np.abs(model.coef_[0])
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
        top_features_idx = np.argsort(coefficients)[-n_top_features:][::-1]
        explanation['top_features'] = [
            {'feature': feature_names[i], 'coefficient': float(model.coef_[0][i])}
            for i in top_features_idx
        ]
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        top_features_idx = np.argsort(importances)[-n_top_features:][::-1]
        explanation['top_features'] = [
            {'feature': feature_names[i], 'importance': float(importances[i])}
            for i in top_features_idx
        ]
    
    return explanation
