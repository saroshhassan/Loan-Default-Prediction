"""Model definitions and factory."""

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating different model types."""
    
    @staticmethod
    def create_baseline(params: Dict[str, Any]) -> LogisticRegression:
        """
        Create Logistic Regression baseline model.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            LogisticRegression model
        """
        # For sklearn 1.8+ compatibility, don't specify deprecated penalty/n_jobs
        model = LogisticRegression(
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42),
            class_weight=params.get('class_weight', 'balanced')
        )
        logger.info("Created LogisticRegression baseline model")
        return model
    
    @staticmethod
    def create_xgboost(params: Dict[str, Any]) -> XGBClassifier:
        """
        Create XGBoost model.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            XGBClassifier model
        """
        model = XGBClassifier(
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=params.get('n_estimators', 100),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0),
            reg_lambda=params.get('reg_lambda', 1),
            scale_pos_weight=params.get('scale_pos_weight', None),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            verbosity=0
        )
        logger.info("Created XGBoost model")
        return model
    
    @staticmethod
    def create_lightgbm(params: Dict[str, Any]) -> LGBMClassifier:
        """
        Create LightGBM model.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            LGBMClassifier model
        """
        model = LGBMClassifier(
            max_depth=params.get('max_depth', 7),
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=params.get('n_estimators', 100),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            lambda_l1=params.get('lambda_l1', 0),
            lambda_l2=params.get('lambda_l2', 1),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            verbose=-1
        )
        logger.info("Created LightGBM model")
        return model
    
    @staticmethod
    def create(model_type: str, params: Dict[str, Any]):
        """
        Create model by type.
        
        Args:
            model_type: Type of model ('baseline', 'xgboost', 'lightgbm')
            params: Model hyperparameters
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model_type not recognized
        """
        if model_type == 'baseline' or model_type == 'logistic_regression':
            return ModelFactory.create_baseline(params)
        elif model_type == 'xgboost':
            return ModelFactory.create_xgboost(params)
        elif model_type == 'lightgbm':
            return ModelFactory.create_lightgbm(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def get_available_models() -> Dict[str, str]:
    """
    Get dictionary of available models.
    
    Returns:
        Dictionary with model names and descriptions
    """
    return {
        'baseline': 'Logistic Regression - Interpretable baseline model',
        'xgboost': 'XGBoost - High-performance gradient boosting',
        'lightgbm': 'LightGBM - Fast gradient boosting'
    }


def get_model_type_from_config(config: Dict[str, Any]) -> str:
    """
    Get model type from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Model type string
    """
    return config.get('type', 'baseline')
