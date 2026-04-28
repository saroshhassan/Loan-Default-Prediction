"""Exploratory Data Analysis module with visualization generation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """Comprehensive exploratory data analysis."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize EDA analyzer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"EDA output directory: {self.output_dir}")
    
    def plot_distributions(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        save: bool = True
    ) -> None:
        """
        Plot distributions for numerical and categorical features.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            save: Whether to save plots
        """
        # Numerical features
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(df[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
            
            # Add KDE plot
            ax2 = axes[idx].twinx()
            df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Hide unused subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plot_path = self.output_dir / "01_numerical_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        plt.show()
        
        # Categorical features
        n_cat = len(categorical_cols)
        fig, axes = plt.subplots(2, (n_cat + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            counts = df[col].value_counts()
            axes[idx].bar(range(len(counts)), counts.values, color='coral', alpha=0.7, edgecolor='black')
            axes[idx].set_xticks(range(len(counts)))
            axes[idx].set_xticklabels(counts.index, rotation=45, ha='right')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plot_path = self.output_dir / "02_categorical_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        plt.show()
    
    def plot_approval_comparison(
        self,
        df: pd.DataFrame,
        target_col: str = 'LoanApproved',
        numerical_cols: List[str] = None,
        categorical_cols: List[str] = None,
        save: bool = True
    ) -> None:
        """
        Compare features between approved and rejected applicants.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            numerical_cols: Numerical features to compare
            categorical_cols: Categorical features to compare
            save: Whether to save plots
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [c for c in numerical_cols if c != target_col]
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Numerical features comparison
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        approved = df[df[target_col] == 1]
        rejected = df[df[target_col] == 0]
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(rejected[col], bins=25, alpha=0.6, label='Rejected', color='red', edgecolor='black')
            axes[idx].hist(approved[col], bins=25, alpha=0.6, label='Approved', color='green', edgecolor='black')
            axes[idx].set_title(f'{col}: Approved vs Rejected', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plot_path = self.output_dir / "03_approval_comparison_numerical.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        plt.show()
        
        # Categorical features comparison
        n_cat = len(categorical_cols)
        fig, axes = plt.subplots(2, (n_cat + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            cross_tab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            cross_tab.plot(kind='bar', ax=axes[idx], color=['red', 'green'], alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{col}: Approval Rate by Category', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].set_xlabel(col)
            axes[idx].legend(['Rejected', 'Approved'], loc='best')
            axes[idx].grid(alpha=0.3, axis='y')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plot_path = self.output_dir / "04_approval_comparison_categorical.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        plt.show()
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Compute and plot correlation matrix.
        
        Args:
            df: Input DataFrame
            numerical_cols: Numerical columns to include
            save: Whether to save plot
            
        Returns:
            Correlation matrix
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / "05_correlation_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        plt.show()
        
        return corr_matrix
    
    def compute_vif(self, df: pd.DataFrame, numerical_cols: List[str] = None) -> pd.DataFrame:
        """
        Compute Variance Inflation Factor (multicollinearity check).
        
        Args:
            df: Input DataFrame
            numerical_cols: Numerical columns to analyze
            
        Returns:
            DataFrame with VIF values
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Clean data: remove NaN and infinite values
        df_clean = df[numerical_cols].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            logger.warning("No valid data for VIF computation after cleaning")
            return pd.DataFrame({'Feature': numerical_cols, 'VIF': [np.nan] * len(numerical_cols)})
        
        vif_results = []
        
        # Compute VIF for each feature
        for i, col in enumerate(numerical_cols):
            try:
                X = df_clean[numerical_cols].values
                # Add constant for proper OLS regression
                X_with_const = add_constant(X)
                vif = variance_inflation_factor(X_with_const, i + 1)  # +1 because of constant term
                vif_results.append({'Feature': col, 'VIF': float(vif)})
            except Exception as e:
                logger.warning(f"Could not compute VIF for {col}: {str(e)}")
                vif_results.append({'Feature': col, 'VIF': np.nan})
        
        vif_data = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
        
        logger.info("\nVariance Inflation Factor (VIF):")
        logger.info(vif_data.to_string(index=False))
        
        return vif_data
    
    def compute_information_value(
        self,
        df: pd.DataFrame,
        target_col: str = 'LoanApproved',
        numerical_cols: List[str] = None,
        categorical_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute Information Value (IV) for feature signal strength.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            numerical_cols: Numerical features
            categorical_cols: Categorical features
            
        Returns:
            DataFrame with IV values
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [c for c in numerical_cols if c != target_col]
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        iv_results = []
        
        # IV for numerical features (binned into deciles)
        for col in numerical_cols:
            df_temp = df[[col, target_col]].copy()
            df_temp[col] = pd.qcut(df_temp[col], q=10, duplicates='drop')
            
            iv = self._compute_iv_for_feature(df_temp, col, target_col)
            iv_results.append({'Feature': col, 'IV': iv})
        
        # IV for categorical features
        for col in categorical_cols:
            iv = self._compute_iv_for_feature(df, col, target_col)
            iv_results.append({'Feature': col, 'IV': iv})
        
        iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
        
        logger.info("\nInformation Value (Signal Strength):")
        logger.info(iv_df.to_string(index=False))
        
        return iv_df
    
    @staticmethod
    def _compute_iv_for_feature(df: pd.DataFrame, feature: str, target: str) -> float:
        """Compute IV for a single feature."""
        crosstab = pd.crosstab(df[feature], df[target])
        
        if crosstab.shape[1] != 2:
            return 0.0
        
        n_events = crosstab.iloc[:, 1]
        n_non_events = crosstab.iloc[:, 0]
        
        total_events = n_events.sum()
        total_non_events = n_non_events.sum()
        
        if total_events == 0 or total_non_events == 0:
            return 0.0
        
        pct_events = n_events / total_events
        pct_non_events = n_non_events / total_non_events
        
        # Avoid log(0)
        pct_events = pct_events.replace(0, 1e-10)
        pct_non_events = pct_non_events.replace(0, 1e-10)
        
        iv = ((pct_events - pct_non_events) * np.log(pct_events / pct_non_events)).sum()
        
        return float(iv)
    
    def compute_roc_auc_per_feature(
        self,
        df: pd.DataFrame,
        target_col: str = 'LoanApproved',
        numerical_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute univariate ROC-AUC for each numerical feature.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            numerical_cols: Numerical features to analyze
            
        Returns:
            DataFrame with ROC-AUC values
        """
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [c for c in numerical_cols if c != target_col]
        
        roc_auc_results = []
        
        for col in numerical_cols:
            try:
                roc_auc = roc_auc_score(df[target_col], df[col])
            except:
                roc_auc = 0.5
            
            roc_auc_results.append({'Feature': col, 'ROC_AUC': roc_auc})
        
        roc_auc_df = pd.DataFrame(roc_auc_results).sort_values('ROC_AUC', ascending=False)
        
        logger.info("\nUnivariate ROC-AUC (Feature Signal Strength):")
        logger.info(roc_auc_df.to_string(index=False))
        
        return roc_auc_df
    
    def generate_summary_report(
        self,
        df: pd.DataFrame,
        target_col: str = 'LoanApproved',
        numerical_cols: List[str] = None,
        categorical_cols: List[str] = None,
        save: bool = True
    ) -> Dict:
        """
        Generate comprehensive EDA summary report.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            numerical_cols: Numerical features
            categorical_cols: Categorical features
            save: Whether to save report
            
        Returns:
            Dictionary with all analysis results
        """
        # Clean data first
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [c for c in numerical_cols if c != target_col]
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info("\n" + "="*80)
        logger.info("EXPLORATORY DATA ANALYSIS SUMMARY")
        logger.info("="*80)
        
        # Generate all plots
        logger.info("\nGenerating distribution plots...")
        self.plot_distributions(df, numerical_cols, categorical_cols, save=save)
        
        logger.info("\nGenerating approval comparison plots...")
        self.plot_approval_comparison(df, target_col, numerical_cols, categorical_cols, save=save)
        
        logger.info("\nGenerating correlation matrix...")
        corr_matrix = self.plot_correlation_matrix(df, numerical_cols, save=save)
        
        # Compute statistical analyses
        logger.info("\nComputing VIF (multicollinearity)...")
        vif = self.compute_vif(df, numerical_cols)
        
        logger.info("\nComputing Information Value (signal strength)...")
        iv = self.compute_information_value(df, target_col, numerical_cols, categorical_cols)
        
        logger.info("\nComputing univariate ROC-AUC...")
        roc_auc = self.compute_roc_auc_per_feature(df, target_col, numerical_cols)
        
        # Compile results
        report = {
            'data_shape': df.shape,
            'target_distribution': df[target_col].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'correlation_matrix': corr_matrix,
            'vif': vif,
            'information_value': iv,
            'roc_auc_per_feature': roc_auc
        }
        
        if save:
            report_path = self.output_dir / "eda_summary_report.txt"
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Dataset Shape: {report['data_shape']}\n")
                f.write(f"Target Distribution:\n{pd.Series(report['target_distribution']).to_string()}\n\n")
                
                f.write("Missing Values:\n")
                f.write(pd.Series(report['missing_values']).to_string() + "\n\n")
                
                f.write("Top 10 Features by Information Value (Signal Strength):\n")
                f.write(report['information_value'].head(10).to_string() + "\n\n")
                
                f.write("Top 10 Features by Univariate ROC-AUC:\n")
                f.write(report['roc_auc_per_feature'].head(10).to_string() + "\n\n")
                
                f.write("Variance Inflation Factor (Multicollinearity Check):\n")
                f.write(report['vif'].to_string() + "\n")
            
            logger.info(f"Saved report to: {report_path}")
        
        logger.info("\n" + "="*80)
        logger.info("EDA COMPLETE")
        logger.info("="*80)
        
        return report
