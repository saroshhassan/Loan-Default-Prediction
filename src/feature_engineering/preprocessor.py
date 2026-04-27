"""Feature engineering and preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Preprocess and engineer features for model training."""
    
    NUMERICAL_FEATURES = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience']
    CATEGORICAL_FEATURES = ['Gender', 'Education', 'City', 'EmploymentType']
    
    def __init__(self):
        """Initialize preprocessor."""
        self.preprocessor = None
        self.feature_names = None
    
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build preprocessing pipeline.
        
        Returns:
            ColumnTransformer for preprocessing
        """
        # Numerical features: standardization
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical features: one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.NUMERICAL_FEATURES),
                ('cat', categorical_transformer, self.CATEGORICAL_FEATURES)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training features DataFrame
            
        Returns:
            Self for method chaining
        """
        if self.preprocessor is None:
            self.build_preprocessor()
        
        self.preprocessor.fit(X)
        
        # Get feature names after transformation
        self._set_feature_names(X)
        
        logger.info(f"Fitted preprocessor. Total features: {len(self.feature_names)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features array
        """
        self.fit(X)
        return self.transform(X)
    
    def _set_feature_names(self, X: pd.DataFrame) -> None:
        """
        Set feature names after transformation.
        
        Args:
            X: Original features DataFrame
        """
        feature_names = []
        
        # Numerical feature names
        feature_names.extend(self.NUMERICAL_FEATURES)
        
        # Categorical feature names (after one-hot encoding)
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES)
        feature_names.extend(cat_features)
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """
        Get transformed feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names


class DataCleaner:
    """Clean and handle missing values."""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # For numerical columns: impute median
        numerical_cols = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience']
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                logger.info(f"Imputed missing values in {col} with median")
        
        # For categorical columns: impute mode
        categorical_cols = ['Gender', 'Education', 'City', 'EmploymentType']
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                logger.info(f"Imputed missing values in {col} with mode")
        
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            df: Input DataFrame
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        from scipy import stats
        
        df = df.copy()
        numerical_cols = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience']
        
        z_scores = np.abs(stats.zscore(df[numerical_cols]))
        mask = (z_scores < z_threshold).all(axis=1)
        
        removed_count = len(df) - mask.sum()
        df = df[mask]
        
        logger.info(f"Removed {removed_count} outliers (z-score > {z_threshold})")
        
        return df


def prepare_data_for_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    handle_missing: bool = True,
    remove_outliers: bool = False
) -> Tuple[np.ndarray, np.ndarray, FeaturePreprocessor]:
    """
    Complete data preparation pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        handle_missing: Whether to handle missing values
        remove_outliers: Whether to remove outliers
        
    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor)
    """
    # Handle missing values
    if handle_missing:
        X_train = DataCleaner.handle_missing_values(X_train)
        X_test = DataCleaner.handle_missing_values(X_test)
    
    # Remove outliers (only from training data)
    if remove_outliers:
        X_train = DataCleaner.remove_outliers(X_train)
    
    # Fit preprocessor on training data, transform both
    preprocessor = FeaturePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Data preparation complete. Processed shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, preprocessor
