"""
Error analysis for understanding model limitations and failure modes.

This module provides comprehensive error analysis including:
- Error distribution and statistical properties
- Worst prediction analysis
- Error patterns by features
- Residual diagnostics
- Prediction interval estimation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


def calculate_errors(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate various error metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with error arrays
    """
    errors = y_true.values - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    percentage_errors = (abs_errors / (y_true.values + 1e-10)) * 100

    return {
        'errors': errors,
        'abs_errors': abs_errors,
        'squared_errors': squared_errors,
        'percentage_errors': percentage_errors
    }


def analyze_error_distribution(
    errors: np.ndarray,
    abs_errors: np.ndarray
) -> Dict[str, float]:
    """
    Analyze statistical properties of errors.

    Args:
        errors: Prediction errors
        abs_errors: Absolute errors

    Returns:
        Dictionary with error statistics
    """
    stats_dict = {
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'std_error': errors.std(),
        'mean_abs_error': abs_errors.mean(),
        'median_abs_error': np.median(abs_errors),
        'p90_abs_error': np.percentile(abs_errors, 90),
        'p95_abs_error': np.percentile(abs_errors, 95),
        'p99_abs_error': np.percentile(abs_errors, 99),
        'max_abs_error': abs_errors.max()
    }

    # Test for normality
    _, p_value = stats.shapiro(errors[:5000] if len(errors) > 5000 else errors)
    stats_dict['normality_p_value'] = p_value
    stats_dict['is_normal'] = p_value > 0.05

    print("\nError Distribution Statistics:")
    print(f"  Mean error: {stats_dict['mean_error']:.3f} (should be ~0)")
    print(f"  Median error: {stats_dict['median_error']:.3f}")
    print(f"  Std error: {stats_dict['std_error']:.3f}")
    print(f"  Mean absolute error: {stats_dict['mean_abs_error']:.3f}")
    print(f"  Median absolute error: {stats_dict['median_abs_error']:.3f}")
    print(f"  90th percentile error: {stats_dict['p90_abs_error']:.3f}")
    print(f"  95th percentile error: {stats_dict['p95_abs_error']:.3f}")
    print(f"  Max error: {stats_dict['max_abs_error']:.3f}")
    print(f"  Normality test: {'Normal' if stats_dict['is_normal'] else 'Non-normal'} (p={p_value:.4f})")

    return stats_dict


def find_worst_predictions(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    abs_errors: np.ndarray,
    n_worst: int = 20
) -> pd.DataFrame:
    """
    Identify and analyze worst predictions.

    Args:
        X: Features
        y_true: True values
        y_pred: Predictions
        abs_errors: Absolute errors
        n_worst: Number of worst cases to return

    Returns:
        DataFrame with worst predictions
    """
    # Get worst indices
    worst_indices = np.argsort(abs_errors)[::-1][:n_worst]

    worst_df = pd.DataFrame({
        'actual': y_true.iloc[worst_indices].values,
        'predicted': y_pred[worst_indices],
        'error': y_true.iloc[worst_indices].values - y_pred[worst_indices],
        'abs_error': abs_errors[worst_indices]
    })

    # Add key features if available
    feature_cols = ['grid_position']
    for col in ['circuit_name', 'team', 'driver_name', 'year']:
        if col in X.columns:
            feature_cols.append(col)

    for col in feature_cols:
        if col in X.columns:
            worst_df[col] = X.iloc[worst_indices][col].values

    print(f"\nWorst {n_worst} predictions:")
    print(worst_df.to_string())

    return worst_df


def analyze_error_patterns(
    X: pd.DataFrame,
    errors: np.ndarray,
    abs_errors: np.ndarray,
    categorical_features: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Analyze error patterns by features.

    Args:
        X: Features
        errors: Prediction errors
        abs_errors: Absolute errors
        categorical_features: List of categorical features to analyze

    Returns:
        Dictionary with pattern analysis DataFrames
    """
    patterns = {}

    # Grid position error pattern
    if 'grid_position' in X.columns:
        grid_errors = pd.DataFrame({
            'grid': X['grid_position'].values,
            'error': errors,
            'abs_error': abs_errors
        })

        grid_stats = grid_errors.groupby('grid')['abs_error'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(3)
        grid_stats.columns = ['count', 'mean_error', 'std_error', 'median_error']

        patterns['grid_position'] = grid_stats

        print("\nError by Grid Position (top 10):")
        print(grid_stats.head(10))

    # Categorical feature patterns
    if categorical_features:
        for feature in categorical_features:
            if feature in X.columns:
                cat_errors = pd.DataFrame({
                    'category': X[feature].values,
                    'abs_error': abs_errors
                })

                cat_stats = cat_errors.groupby('category')['abs_error'].agg([
                    'count', 'mean', 'std'
                ]).round(3)
                cat_stats.columns = ['count', 'mean_error', 'std_error']
                cat_stats = cat_stats.sort_values('mean_error', ascending=False)

                patterns[feature] = cat_stats

                print(f"\nError by {feature} (top 5 worst):")
                print(cat_stats.head())

    return patterns


def plot_error_distribution(
    errors: np.ndarray,
    abs_errors: np.ndarray,
    save_path: str = None
) -> None:
    """
    Create comprehensive error distribution plots.

    Args:
        errors: Prediction errors
        abs_errors: Absolute errors
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of errors
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 0].set_xlabel('Prediction Error (Actual - Predicted)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Error Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Q-Q plot for normality
    stats.probplot(errors, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Absolute errors by percentile
    percentiles = np.arange(0, 101, 5)
    error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]

    axes[1, 0].plot(percentiles, error_percentiles, linewidth=2, color='green', marker='o')
    axes[1, 0].axhline(np.median(abs_errors), color='orange', linestyle='--',
                       label=f'Median: {np.median(abs_errors):.2f}')
    axes[1, 0].axhline(np.percentile(abs_errors, 90), color='red', linestyle='--',
                       label=f'90th %ile: {np.percentile(abs_errors, 90):.2f}')
    axes[1, 0].set_xlabel('Percentile', fontsize=10)
    axes[1, 0].set_ylabel('Absolute Error', fontsize=10)
    axes[1, 0].set_title('Error Percentiles', fontsize=11, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Box plot of absolute errors
    axes[1, 1].boxplot(abs_errors, vert=True)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=10)
    axes[1, 1].set_title('Absolute Error Distribution', fontsize=11, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error distribution plot to {save_path}")

    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors: np.ndarray,
    save_path: str = None
) -> None:
    """
    Create residual diagnostic plots.

    Args:
        y_true: True values
        y_pred: Predictions
        errors: Errors
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Position', fontsize=10)
    axes[0].set_ylabel('Predicted Position', fontsize=10)
    axes[0].set_title('Predicted vs Actual', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residuals vs Predicted
    axes[1].scatter(y_pred, errors, alpha=0.5, s=20, color='coral')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].set_xlabel('Predicted Position', fontsize=10)
    axes[1].set_ylabel('Residual', fontsize=10)
    axes[1].set_title('Residual Plot', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residual plots to {save_path}")

    plt.close()


def plot_error_by_features(
    X: pd.DataFrame,
    abs_errors: np.ndarray,
    patterns: Dict[str, pd.DataFrame],
    save_path: str = None
) -> None:
    """
    Plot error patterns by features.

    Args:
        X: Features
        abs_errors: Absolute errors
        patterns: Pattern analysis results
        save_path: Path to save figure
    """
    n_plots = min(3, len(patterns))
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Grid position errors
    if 'grid_position' in patterns and plot_idx < n_plots:
        grid_stats = patterns['grid_position']
        if len(grid_stats) > 0:
            axes[plot_idx].bar(grid_stats.index[:20], grid_stats['mean_error'][:20],
                              color='steelblue', alpha=0.7)
            axes[plot_idx].errorbar(grid_stats.index[:20], grid_stats['mean_error'][:20],
                                   yerr=grid_stats['std_error'][:20],
                                   fmt='none', color='black', alpha=0.5, capsize=3)
            axes[plot_idx].set_xlabel('Grid Position', fontsize=10)
            axes[plot_idx].set_ylabel('Mean Absolute Error', fontsize=10)
            axes[plot_idx].set_title('Error by Starting Position', fontsize=11, fontweight='bold')
            axes[plot_idx].grid(alpha=0.3)
            plot_idx += 1

    # Circuit errors
    if 'circuit_name' in patterns and plot_idx < n_plots:
        circuit_stats = patterns['circuit_name'].head(15)
        if len(circuit_stats) > 0:
            axes[plot_idx].barh(range(len(circuit_stats)), circuit_stats['mean_error'],
                               color='coral', alpha=0.7)
            axes[plot_idx].set_yticks(range(len(circuit_stats)))
            axes[plot_idx].set_yticklabels(circuit_stats.index, fontsize=8)
            axes[plot_idx].set_xlabel('Mean Absolute Error', fontsize=10)
            axes[plot_idx].set_title('Error by Circuit (Top 15)', fontsize=11, fontweight='bold')
            axes[plot_idx].invert_yaxis()
            axes[plot_idx].grid(axis='x', alpha=0.3)
            plot_idx += 1

    # Team errors
    if 'team' in patterns and plot_idx < n_plots:
        team_stats = patterns['team'].head(10)
        if len(team_stats) > 0:
            axes[plot_idx].barh(range(len(team_stats)), team_stats['mean_error'],
                               color='green', alpha=0.7)
            axes[plot_idx].set_yticks(range(len(team_stats)))
            axes[plot_idx].set_yticklabels(team_stats.index, fontsize=8)
            axes[plot_idx].set_xlabel('Mean Absolute Error', fontsize=10)
            axes[plot_idx].set_title('Error by Team (Top 10)', fontsize=11, fontweight='bold')
            axes[plot_idx].invert_yaxis()
            axes[plot_idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature error plots to {save_path}")

    plt.close()


def estimate_prediction_intervals(
    model,
    X: pd.DataFrame,
    y_true: pd.Series,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Estimate prediction intervals for tree-based models.

    Args:
        model: Trained model
        X: Features
        y_true: True values
        confidence: Confidence level

    Returns:
        Dictionary with interval estimates
    """
    print(f"\nEstimating {confidence:.0%} prediction intervals...")

    # Check if model has estimators (ensemble)
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

        pred_mean = tree_predictions.mean(axis=0)
        pred_std = tree_predictions.std(axis=0)

        # Calculate intervals
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower_bound = pred_mean - z_score * pred_std
        upper_bound = pred_mean + z_score * pred_std

        # Calculate coverage
        coverage = ((y_true.values >= lower_bound) & (y_true.values <= upper_bound)).mean()

        # Average interval width
        avg_width = (upper_bound - lower_bound).mean()

        print(f"  Coverage: {coverage:.1%} (target: {confidence:.0%})")
        print(f"  Average interval width: {avg_width:.2f} positions")

        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'coverage': coverage,
            'avg_width': avg_width
        }
    else:
        print("  Model does not support prediction intervals (no estimators)")
        return None


def analyze_errors(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str = 'results/figures',
    categorical_features: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive error analysis.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        output_dir: Directory to save plots
        categorical_features: Categorical features to analyze

    Returns:
        Dictionary with error analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get predictions
    y_pred = model.predict(X_val)

    # Calculate errors
    error_dict = calculate_errors(y_val, y_pred)
    errors = error_dict['errors']
    abs_errors = error_dict['abs_errors']

    # Analyze distribution
    error_stats = analyze_error_distribution(errors, abs_errors)

    # Find worst predictions
    worst_df = find_worst_predictions(X_val, y_val, y_pred, abs_errors, n_worst=20)

    # Analyze patterns
    if categorical_features is None:
        categorical_features = ['circuit_name', 'team']

    patterns = analyze_error_patterns(X_val, errors, abs_errors, categorical_features)

    # Create visualizations
    plot_error_distribution(
        errors,
        abs_errors,
        save_path=output_path / 'error_distribution.png'
    )

    plot_residuals(
        y_val.values,
        y_pred,
        errors,
        save_path=output_path / 'residual_plots.png'
    )

    plot_error_by_features(
        X_val,
        abs_errors,
        patterns,
        save_path=output_path / 'error_by_features.png'
    )

    # Prediction intervals
    intervals = estimate_prediction_intervals(model, X_val, y_val)

    # Summary
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n80% of predictions within {np.percentile(abs_errors, 80):.2f} positions")
    print(f"90% of predictions within {np.percentile(abs_errors, 90):.2f} positions")

    if 'circuit_name' in patterns:
        worst_circuit = patterns['circuit_name'].index[0]
        worst_mae = patterns['circuit_name'].iloc[0]['mean_error']
        print(f"\nWorst circuit: {worst_circuit} (MAE={worst_mae:.2f})")

    # Check for systematic bias
    if abs(error_stats['mean_error']) > 0.1:
        bias_direction = 'over-predicting' if error_stats['mean_error'] > 0 else 'under-predicting'
        print(f"\nSystematic bias detected: {bias_direction} by {abs(error_stats['mean_error']):.2f} positions")

    return {
        'error_stats': error_stats,
        'worst_predictions': worst_df,
        'patterns': patterns,
        'intervals': intervals,
        'errors': error_dict
    }
