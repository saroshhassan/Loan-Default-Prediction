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
        # Set sparse_output=False to ensure dense output and avoid NaN issues
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(
                drop='first', 
                sparse_output=False,
                handle_unknown='ignore',
                min_frequency=1,
                feature_name_combiner='concat'
            ))
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
        
        # Ensure all input columns are present and valid
        for col in self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES:
            if col not in X.columns:
                raise ValueError(f"Required column '{col}' not found in X")
        
        self.preprocessor.fit(X)
        
        # Get feature names after transformation
        self._set_feature_names(X)
        
        logger.info(f"Fitted preprocessor. Total features: {len(self.feature_names) if self.feature_names else 'unknown'}")
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
        
        X_transformed = self.preprocessor.transform(X)
        
        # Ensure no NaNs in output
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate output
        if np.isnan(X_transformed).any():
            logger.warning(f"Found NaN values in transformed data after nan_to_num")
            X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features array
        """
        self.fit(X)
        X_transformed = self.transform(X)
        
        return X_transformed
    
    def _set_feature_names(self, X: pd.DataFrame) -> None:
        """
        Set feature names after transformation.
        
        Args:
            X: Original features DataFrame
        """
        try:
            feature_names = []
            
            # Numerical feature names
            feature_names.extend(self.NUMERICAL_FEATURES)
            
            # Categorical feature names (after one-hot encoding)
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES)
            feature_names.extend(cat_features)
            
            self.feature_names = feature_names
            logger.info(f"Set {len(feature_names)} feature names after transformation")
        except Exception as e:
            logger.warning(f"Error setting feature names: {e}. Falling back to default names.")
            # Generate default names if we can't get them from the encoder
            n_features = len(self.NUMERICAL_FEATURES)
            # Estimate categorical features after one-hot encoding
            n_features += len(self.CATEGORICAL_FEATURES) * 5  # rough estimate
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
    
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
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed missing values in {col} with median")
        
        # For categorical columns: impute mode
        categorical_cols = ['Gender', 'Education', 'City', 'EmploymentType']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
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
    logger.info(f"Starting data preparation. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Initial validation
    logger.info(f"X_train NaN count before cleaning: {X_train.isna().sum().sum()}")
    logger.info(f"X_test NaN count before cleaning: {X_test.isna().sum().sum()}")
    
    # Handle missing values
    if handle_missing:
        X_train = DataCleaner.handle_missing_values(X_train)
        X_test = DataCleaner.handle_missing_values(X_test)
        
        logger.info(f"X_train NaN count after cleaning: {X_train.isna().sum().sum()}")
        logger.info(f"X_test NaN count after cleaning: {X_test.isna().sum().sum()}")
    
    # Remove outliers (only from training data)
    if remove_outliers:
        X_train = DataCleaner.remove_outliers(X_train)
    
    # Ensure no NaN values remain before preprocessing
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_test.median(numeric_only=True))
    
    for col in ['Gender', 'Education', 'City', 'EmploymentType']:
        if col in X_train.columns and X_train[col].isnull().any():
            mode_val = X_train[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            X_train[col] = X_train[col].fillna(fill_val)
        if col in X_test.columns and X_test[col].isnull().any():
            mode_val = X_test[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            X_test[col] = X_test[col].fillna(fill_val)
    
    logger.info(f"Final X_train NaN before preprocessing: {X_train.isna().sum().sum()}")
    logger.info(f"Final X_test NaN before preprocessing: {X_test.isna().sum().sum()}")
    
    # Fit preprocessor on training data, transform both
    preprocessor = FeaturePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"X_train_processed shape: {X_train_processed.shape}, NaN count: {np.isnan(X_train_processed).sum()}")
    logger.info(f"X_test_processed shape: {X_test_processed.shape}, NaN count: {np.isnan(X_test_processed).sum()}")
    
    logger.info(f"Data preparation complete. Processed shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, preprocessor
