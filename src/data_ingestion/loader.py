"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate loan dataset."""
    
    REQUIRED_COLUMNS = [
        'Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience',
        'Gender', 'Education', 'City', 'EmploymentType', 'LoanApproved'
    ]
    
    NUMERICAL_COLUMNS = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'YearsExperience']
    CATEGORICAL_COLUMNS = ['Gender', 'Education', 'City', 'EmploymentType']
    TARGET_COLUMN = 'LoanApproved'
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to CSV or Parquet file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV or Parquet.
        
        Returns:
            DataFrame with data
            
        Raises:
            ValueError: If file format unsupported or required columns missing
        """
        if self.data_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate data structure and content.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types
        for col in self.NUMERICAL_COLUMNS:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column '{col}' should be numeric, got {df[col].dtype}")
        
        # Check target variable
        if not set(df[self.TARGET_COLUMN].unique()).issubset({0, 1}):
            raise ValueError(f"Target '{self.TARGET_COLUMN}' must contain only 0 and 1")
        
        logger.info("Data validation passed")
    
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load and validate data.
        
        Returns:
            Validated DataFrame
        """
        df = self.load()
        self.validate(df)
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics about the data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'target_distribution': df[self.TARGET_COLUMN].value_counts().to_dict(),
            'target_ratio': (df[self.TARGET_COLUMN].sum() / len(df)),
            'column_dtypes': df.dtypes.to_dict()
        }


def load_train_test_split(
    data_path: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and perform train-test split.
    
    Args:
        data_path: Path to data file
        test_size: Proportion of test data (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    loader = DataLoader(data_path)
    df = loader.load_and_validate()
    
    X = df.drop(columns=[DataLoader.TARGET_COLUMN])
    y = df[DataLoader.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train/Test split: {X_train.shape} / {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
