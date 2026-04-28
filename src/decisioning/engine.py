"""Decision engine for converting risk scores to actionable decisions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Decision(Enum):
    """Decision categories."""
    APPROVE = 1
    REVIEW = 0
    REJECT = -1


class DecisionEngine:
    """Convert risk scores to decisions and simulate business impact."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize decision engine.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Decision engine output directory: {self.output_dir}")
    
    @staticmethod
    def apply_decision_policy(
        risk_scores: np.ndarray,
        approve_threshold: float = 0.30,
        reject_threshold: float = 0.70
    ) -> pd.Series:
        """
        Apply decision policy based on risk scores.
        
        Args:
            risk_scores: Array of risk scores (0-1, probability of default)
            approve_threshold: Score below which to APPROVE
            reject_threshold: Score above which to REJECT
            
        Returns:
            Series of Decision enums
        """
        decisions = []
        
        for score in risk_scores:
            if score < approve_threshold:
                decisions.append(Decision.APPROVE)
            elif score > reject_threshold:
                decisions.append(Decision.REJECT)
            else:
                decisions.append(Decision.REVIEW)
        
        return pd.Series(decisions)
    
    @staticmethod
    def simulate_business_impact(
        decisions: pd.Series,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        loan_amounts: np.ndarray = None
    ) -> Dict:
        """
        Simulate business outcomes from decision policy.
        
        Args:
            decisions: Series of Decision enums
            y_true: True labels (1=default, 0=good)
            y_pred_proba: Predicted probability of default
            loan_amounts: Loan amounts (for expected loss calculation)
            
        Returns:
            Dictionary with business metrics
        """
        # Get approved decisions
        approved_mask = decisions != Decision.REJECT
        review_mask = decisions == Decision.REVIEW
        reject_mask = decisions == Decision.REJECT
        approve_only_mask = decisions == Decision.APPROVE
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        total_applicants = len(decisions)
        approved_count = approved_mask.sum()
        reviewed_count = review_mask.sum()
        rejected_count = reject_mask.sum()
        
        # For approved applicants
        if approved_count > 0:
            approved_defaults = y_true[approved_mask].sum()
            approval_rate = approved_count / total_applicants
            default_rate_in_approved = approved_defaults / approved_count if approved_count > 0 else 0
            false_negative_rate = approved_defaults / y_true.sum() if y_true.sum() > 0 else 0
        else:
            approved_defaults = 0
            approval_rate = 0
            default_rate_in_approved = 0
            false_negative_rate = 0
        
        # For rejected applicants
        if rejected_count > 0:
            rejected_defaults = y_true[reject_mask].sum()
        else:
            rejected_defaults = 0
        
        # Expected loss calculation
        if loan_amounts is not None:
            loan_amounts = np.array(loan_amounts)
            expected_loss_approved = (y_pred_proba[approved_mask] * loan_amounts[approved_mask]).sum()
            expected_loss_total = (y_pred_proba * loan_amounts).sum()
        else:
            expected_loss_approved = None
            expected_loss_total = None
        
        metrics = {
            'total_applicants': int(total_applicants),
            'approved_count': int(approved_count),
            'reviewed_count': int(reviewed_count),
            'rejected_count': int(rejected_count),
            'approval_rate': float(approval_rate),
            'approved_defaults': int(approved_defaults),
            'default_rate_in_approved': float(default_rate_in_approved),
            'false_negative_rate': float(false_negative_rate),
            'rejected_defaults': int(rejected_defaults),
            'expected_loss_approved': float(expected_loss_approved) if expected_loss_approved is not None else None,
            'expected_loss_total': float(expected_loss_total) if expected_loss_total is not None else None
        }
        
        return metrics
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        loan_amounts: np.ndarray = None,
        metric: str = 'approval_rate'
    ) -> Dict:
        """
        Optimize decision thresholds by sweeping parameter space.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            loan_amounts: Loan amounts (optional)
            metric: Metric to optimize ('approval_rate', 'default_rate', 'expected_loss')
            
        Returns:
            Dictionary with optimal thresholds and metrics
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for approve_th in thresholds:
            for reject_th in thresholds:
                if approve_th >= reject_th:
                    continue
                
                decisions = self.apply_decision_policy(y_pred_proba, approve_th, reject_th)
                metrics = self.simulate_business_impact(decisions, y_true, y_pred_proba, loan_amounts)
                metrics['approve_threshold'] = approve_th
                metrics['reject_threshold'] = reject_th
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Find optimal based on metric
        if metric == 'approval_rate':
            optimal_idx = results_df['approval_rate'].idxmax()
        elif metric == 'default_rate':
            optimal_idx = results_df['default_rate_in_approved'].idxmin()
        elif metric == 'expected_loss' and 'expected_loss_approved' in results_df.columns:
            optimal_idx = results_df['expected_loss_approved'].idxmin()
        else:
            optimal_idx = 0
        
        optimal_metrics = results_df.loc[optimal_idx].to_dict()
        
        logger.info(f"\nOptimized thresholds (metric={metric}):")
        logger.info(f"  Approve threshold: {optimal_metrics['approve_threshold']:.2f}")
        logger.info(f"  Reject threshold: {optimal_metrics['reject_threshold']:.2f}")
        logger.info(f"  Approval rate: {optimal_metrics['approval_rate']:.2%}")
        logger.info(f"  Default rate in approved: {optimal_metrics['default_rate_in_approved']:.2%}")
        
        return {
            'optimal_metrics': optimal_metrics,
            'all_results': results_df
        }
    
    def generate_tradeoff_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        loan_amounts: np.ndarray = None,
        save: bool = True
    ) -> None:
        """
        Generate tradeoff curves between business metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            loan_amounts: Loan amounts (optional)
            save: Whether to save plots
        """
        # Get percentiles of actual predictions to understand their distribution
        p10 = np.percentile(y_pred_proba, 10)
        p25 = np.percentile(y_pred_proba, 25)
        p50 = np.percentile(y_pred_proba, 50)
        p75 = np.percentile(y_pred_proba, 75)
        p90 = np.percentile(y_pred_proba, 90)
        
        logger.info(f"\nPrediction distribution percentiles:")
        logger.info(f"  10th: {p10:.4f}, 25th: {p25:.4f}, 50th: {p50:.4f}, 75th: {p75:.4f}, 90th: {p90:.4f}")
        
        # Create thresholds that span from very strict (low percentile) to lenient (high percentile)
        # This ensures we capture the range of actual predictions
        approve_thresholds = np.linspace(np.min(y_pred_proba), p75, 25)
        
        results = []
        for approve_th in approve_thresholds:
            # Set reject threshold higher than approve threshold
            reject_th = max(p90, approve_th + (p75 - p10) * 0.1)
            
            decisions = self.apply_decision_policy(y_pred_proba, approve_th, reject_th)
            metrics = self.simulate_business_impact(decisions, y_true, y_pred_proba, loan_amounts)
            metrics['approve_threshold'] = approve_th
            metrics['reject_threshold'] = reject_th
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"\nThreshold sweep results:")
        logger.info(f"  Approval rate range: {results_df['approval_rate'].min():.2%} to {results_df['approval_rate'].max():.2%}")
        logger.info(f"  Default rate range: {results_df['default_rate_in_approved'].min():.2%} to {results_df['default_rate_in_approved'].max():.2%}")
        logger.info(f"  False negative rate range: {results_df['false_negative_rate'].min():.2%} to {results_df['false_negative_rate'].max():.2%}")
        
        # Plot tradeoffs
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Approval rate vs Default rate
        axes[0, 0].plot(results_df['approve_threshold'], results_df['approval_rate'], 
                       marker='o', label='Approval Rate', linewidth=2.5, markersize=7, color='#3498db')
        axes[0, 0].set_xlabel('Approve Threshold (Risk Score)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Approval Rate', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Approval Rate vs Threshold', fontweight='bold', fontsize=12)
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        axes[0, 0].legend(fontsize=10, loc='best')
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Default rate in approved
        axes[0, 1].plot(results_df['approve_threshold'], results_df['default_rate_in_approved'],
                       marker='s', label='Default Rate', color='#e74c3c', linewidth=2.5, markersize=7)
        axes[0, 1].set_xlabel('Approve Threshold (Risk Score)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Default Rate in Approved Portfolio', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Portfolio Risk vs Threshold', fontweight='bold', fontsize=12)
        axes[0, 1].grid(alpha=0.3, linestyle='--')
        axes[0, 1].legend(fontsize=10, loc='best')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
        
        # False Negative Rate (Missed Defaults)
        axes[1, 0].plot(results_df['approve_threshold'], results_df['false_negative_rate'],
                       marker='^', label='False Negative Rate', color='#f39c12', linewidth=2.5, markersize=7)
        axes[1, 0].set_xlabel('Approve Threshold (Risk Score)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('False Negative Rate', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Missed Defaults vs Threshold', fontweight='bold', fontsize=12)
        axes[1, 0].grid(alpha=0.3, linestyle='--')
        axes[1, 0].legend(fontsize=10, loc='best')
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
        
        # Approval count
        axes[1, 1].plot(results_df['approve_threshold'], results_df['approved_count'],
                       marker='d', label='Approved Count', color='#2ecc71', linewidth=2.5, markersize=7)
        axes[1, 1].set_xlabel('Approve Threshold (Risk Score)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Approvals', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Approval Volume vs Threshold', fontweight='bold', fontsize=12)
        axes[1, 1].grid(alpha=0.3, linestyle='--')
        axes[1, 1].legend(fontsize=10, loc='best')
        
        plt.suptitle('Decision Threshold Tradeoff Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / "06_decision_tradeoffs.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        
        plt.show()
    
    def segment_by_risk(
        self,
        y_pred_proba: np.ndarray,
        segment_thresholds: List[float] = None,
        y_true: np.ndarray = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Segment applicants into risk categories.
        
        Args:
            y_pred_proba: Predicted probabilities
            segment_thresholds: Thresholds for risk segments
            y_true: True labels (for segment performance)
            save: Whether to save plot
            
        Returns:
            DataFrame with risk segments
        """
        if segment_thresholds is None:
            segment_thresholds = [0.2, 0.4, 0.6, 0.8]
        
        # Create risk segments
        segments = []
        for score in y_pred_proba:
            if score < segment_thresholds[0]:
                segments.append('Very Low Risk')
            elif score < segment_thresholds[1]:
                segments.append('Low Risk')
            elif score < segment_thresholds[2]:
                segments.append('Medium Risk')
            elif score < segment_thresholds[3]:
                segments.append('High Risk')
            else:
                segments.append('Very High Risk')
        
        segment_df = pd.DataFrame({
            'risk_score': y_pred_proba,
            'risk_segment': segments
        })
        
        if y_true is not None:
            segment_df['actual_default'] = y_true
        
        # Statistics
        segment_stats = segment_df.groupby('risk_segment').agg({
            'risk_score': ['count', 'mean', 'min', 'max']
        })
        
        if y_true is not None:
            segment_df_with_default = segment_df.groupby('risk_segment')['actual_default'].agg(['sum', 'count', 'mean'])
            segment_stats['actual_default_rate'] = segment_df_with_default['mean']
        
        logger.info("\nRisk Segmentation:")
        logger.info(segment_stats)
        
        # Plot segments
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Segment distribution
        segment_counts = segment_df['risk_segment'].value_counts()
        segment_order = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        segment_counts = segment_counts.reindex([s for s in segment_order if s in segment_counts.index])
        
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
        axes[0].bar(range(len(segment_counts)), segment_counts.values, color=colors[:len(segment_counts)], 
                   alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(segment_counts)))
        axes[0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Applicant Distribution by Risk Segment', fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Segment risk scores
        for idx, segment in enumerate(segment_order):
            if segment in segment_df['risk_segment'].values:
                segment_scores = segment_df[segment_df['risk_segment'] == segment]['risk_score']
                axes[1].scatter([idx] * len(segment_scores), segment_scores, 
                              alpha=0.5, s=50, color=colors[idx], label=segment)
        
        axes[1].set_xticks(range(len(segment_order)))
        axes[1].set_xticklabels(segment_order, rotation=45, ha='right')
        axes[1].set_ylabel('Risk Score')
        axes[1].set_title('Risk Score Distribution by Segment', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / "07_risk_segments.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {plot_path}")
        
        plt.show()
        
        return segment_df
    
    def generate_decision_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        loan_amounts: np.ndarray = None,
        approve_threshold: float = 0.30,
        reject_threshold: float = 0.70,
        save: bool = True
    ) -> Dict:
        """
        Generate comprehensive decision analysis report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            loan_amounts: Loan amounts (optional)
            approve_threshold: Approval threshold
            reject_threshold: Rejection threshold
            save: Whether to save report
            
        Returns:
            Dictionary with report data
        """
        logger.info("\n" + "="*80)
        logger.info("DECISION ENGINE ANALYSIS")
        logger.info("="*80)
        
        # Apply policy
        decisions = self.apply_decision_policy(y_pred_proba, approve_threshold, reject_threshold)
        
        # Simulate impact
        metrics = self.simulate_business_impact(decisions, y_true, y_pred_proba, loan_amounts)
        
        # Generate visualizations
        logger.info("\nGenerating tradeoff curves...")
        self.generate_tradeoff_curves(y_true, y_pred_proba, loan_amounts, save=save)
        
        logger.info("\nSegmenting by risk...")
        segment_df = self.segment_by_risk(y_pred_proba, y_true=y_true, save=save)
        
        report = {
            'thresholds': {
                'approve': approve_threshold,
                'reject': reject_threshold
            },
            'business_metrics': metrics,
            'segment_analysis': segment_df
        }
        
        if save:
            report_path = self.output_dir / "decision_report.txt"
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("DECISION ENGINE REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Decision Thresholds:\n")
                f.write(f"  Approve if Risk Score < {approve_threshold:.2f}\n")
                f.write(f"  Reject if Risk Score > {reject_threshold:.2f}\n")
                f.write(f"  Review if Risk Score in [{approve_threshold:.2f}, {reject_threshold:.2f}]\n\n")
                
                f.write("Business Impact Metrics:\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if 'rate' in key or 'approval' in key.lower():
                            f.write(f"  {key}: {value:.2%}\n")
                        else:
                            f.write(f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            logger.info(f"Saved report to: {report_path}")
        
        logger.info("\n" + "="*80)
        logger.info("DECISION ENGINE ANALYSIS COMPLETE")
        logger.info("="*80)
        
        return report
