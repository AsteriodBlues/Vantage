"""
SHAP (SHapley Additive exPlanations) analysis for model interpretation.

This module provides SHAP-based model interpretation including:
- SHAP value calculation for different model types
- Summary plots showing feature impact
- Dependence plots for feature interactions
- Waterfall plots for individual predictions
- Force plots for detailed explanations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle


def create_shap_explainer(model, model_type: str, X_background: pd.DataFrame = None):
    """
    Create appropriate SHAP explainer for model type.

    Args:
        model: Trained model
        model_type: Type of model ('xgboost', 'random_forest', 'stacking', etc.)
        X_background: Background data for KernelExplainer (optional)

    Returns:
        SHAP explainer object
    """
    print(f"Creating SHAP explainer for {model_type}...")

    if model_type in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
        # TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
    elif model_type == 'stacking':
        # For ensembles, use a sample as background
        if X_background is None:
            raise ValueError("Background data required for stacking models")
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict, background)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return explainer


def calculate_shap_values(
    explainer,
    X: pd.DataFrame,
    model_type: str,
    sample_size: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Calculate SHAP values for dataset.

    Args:
        explainer: SHAP explainer object
        X: Features to explain
        model_type: Type of model
        sample_size: Maximum samples to use (for performance)
        random_state: Random seed for sampling

    Returns:
        Tuple of (shap_values array, expected_value)
    """
    print(f"Calculating SHAP values for {len(X)} samples...")

    # Sample if dataset is large
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} for computational efficiency")
    else:
        X_sample = X

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle list output (some models return list)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Get expected value
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = explainer.expected_value[0]
    else:
        expected_value = explainer.expected_value

    print(f"SHAP values calculated. Shape: {shap_values.shape}")

    return shap_values, expected_value, X_sample


def verify_shap_values(
    shap_values: np.ndarray,
    expected_value: float,
    predictions: np.ndarray,
    sample_idx: int = 0
) -> bool:
    """
    Verify SHAP values sum to prediction.

    Args:
        shap_values: Calculated SHAP values
        expected_value: Model's expected value
        predictions: Model predictions
        sample_idx: Index to check

    Returns:
        True if verification passes
    """
    shap_sum = shap_values[sample_idx].sum() + expected_value
    actual_pred = predictions[sample_idx]

    difference = abs(shap_sum - actual_pred)
    print(f"\nSHAP Verification (sample {sample_idx}):")
    print(f"  SHAP sum: {shap_sum:.4f}")
    print(f"  Actual prediction: {actual_pred:.4f}")
    print(f"  Difference: {difference:.6f}")

    if difference < 0.01:
        print("  ✓ SHAP values verified!")
        return True
    else:
        print("  ⚠ Large difference detected")
        return False


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    save_path: str = None
) -> None:
    """
    Create SHAP summary plot showing feature importance and impact direction.

    Args:
        shap_values: SHAP values
        X: Feature data
        max_display: Maximum features to display
        save_path: Path to save figure
    """
    print(f"Creating SHAP summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to {save_path}")

    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    features: List[str],
    save_path: str = None
) -> None:
    """
    Create SHAP dependence plots for specified features.

    Args:
        shap_values: SHAP values
        X: Feature data
        features: List of features to plot
        save_path: Path to save figure
    """
    print(f"Creating SHAP dependence plots for {len(features)} features...")

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if idx < len(axes):
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                interaction_index='auto',
                ax=axes[idx],
                show=False
            )
            axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=10, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP dependence plots to {save_path}")

    plt.close()


def create_shap_waterfall(
    shap_values: np.ndarray,
    expected_value: float,
    X: pd.DataFrame,
    instance_idx: int,
    max_display: int = 15,
    save_path: str = None
) -> None:
    """
    Create waterfall plot for individual prediction explanation.

    Args:
        shap_values: SHAP values
        expected_value: Model expected value
        X: Feature data
        instance_idx: Index of instance to explain
        max_display: Maximum features to display
        save_path: Path to save figure
    """
    print(f"Creating waterfall plot for instance {instance_idx}...")

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=expected_value,
        data=X.iloc[instance_idx].values,
        feature_names=X.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved waterfall plot to {save_path}")

    plt.close()


def create_shap_force_plot(
    shap_values: np.ndarray,
    expected_value: float,
    X: pd.DataFrame,
    instance_idx: int,
    save_path: str = None
) -> None:
    """
    Create force plot for individual prediction explanation.

    Args:
        shap_values: SHAP values
        expected_value: Model expected value
        X: Feature data
        instance_idx: Index of instance to explain
        save_path: Path to save figure
    """
    print(f"Creating force plot for instance {instance_idx}...")

    # Force plot (returns HTML/JS visualization)
    force_plot = shap.force_plot(
        expected_value,
        shap_values[instance_idx],
        X.iloc[instance_idx],
        matplotlib=True,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved force plot to {save_path}")

    plt.close()


def find_interesting_predictions(
    X: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray
) -> Dict[str, int]:
    """
    Find interesting predictions to explain with SHAP.

    Args:
        X: Feature data
        y_true: True target values
        predictions: Model predictions

    Returns:
        Dictionary with indices of interesting cases
    """
    interesting = {}

    # Pole position winner
    pole_winners = X[(X['grid_position'] == 1) & (y_true == 1)]
    if len(pole_winners) > 0:
        interesting['pole_winner'] = pole_winners.index[0]

    # Pole position but didn't win
    pole_not_win = X[(X['grid_position'] == 1) & (y_true > 1)]
    if len(pole_not_win) > 0:
        interesting['pole_not_winner'] = pole_not_win.index[0]

    # Back of grid to points
    back_to_points = X[(X['grid_position'] > 15) & (y_true <= 10)]
    if len(back_to_points) > 0:
        interesting['back_to_points'] = back_to_points.index[0]

    # Large prediction error
    errors = np.abs(y_true.values - predictions)
    worst_idx = np.argmax(errors)
    interesting['worst_prediction'] = y_true.index[worst_idx]

    # Best prediction
    best_idx = np.argmin(errors)
    interesting['best_prediction'] = y_true.index[best_idx]

    print("\nInteresting predictions found:")
    for case_name, idx in interesting.items():
        actual = y_true.loc[idx] if idx in y_true.index else y_true.iloc[idx]
        pred = predictions[y_true.index.get_loc(idx)] if idx in y_true.index else predictions[idx]
        print(f"  {case_name}: Actual={actual:.1f}, Predicted={pred:.1f}")

    return interesting


def analyze_shap_values(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = 'xgboost',
    X_train: pd.DataFrame = None,
    output_dir: str = 'results/figures',
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Comprehensive SHAP analysis for model interpretation.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        model_type: Type of model
        X_train: Training data (for background)
        output_dir: Directory to save plots
        sample_size: Maximum samples for SHAP calculation

    Returns:
        Dictionary with SHAP results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Create explainer
    if model_type == 'stacking' and X_train is not None:
        explainer = create_shap_explainer(model, model_type, X_train)
    else:
        explainer = create_shap_explainer(model, model_type)

    # Calculate SHAP values
    shap_values, expected_value, X_sample = calculate_shap_values(
        explainer, X_val, model_type, sample_size
    )

    # Get corresponding predictions and targets
    y_sample = y_val.loc[X_sample.index]
    predictions = model.predict(X_sample)

    # Verify SHAP values
    verify_shap_values(shap_values, expected_value, predictions, sample_idx=0)

    results['shap_values'] = shap_values
    results['expected_value'] = expected_value
    results['X_sample'] = X_sample

    # Create summary plot
    plot_shap_summary(
        shap_values,
        X_sample,
        max_display=20,
        save_path=output_path / 'shap_summary.png'
    )

    # Get top features for dependence plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = [X_sample.columns[i] for i in top_feature_indices]

    # Create dependence plots
    plot_shap_dependence(
        shap_values,
        X_sample,
        top_features,
        save_path=output_path / 'shap_dependence.png'
    )

    # Find interesting predictions
    interesting = find_interesting_predictions(X_sample, y_sample, predictions)

    # Create waterfall plots for interesting cases
    waterfall_cases = ['pole_winner', 'back_to_points', 'worst_prediction']
    for case_name in waterfall_cases:
        if case_name in interesting:
            idx = interesting[case_name]
            # Convert to iloc position
            iloc_pos = X_sample.index.get_loc(idx)

            create_shap_waterfall(
                shap_values,
                expected_value,
                X_sample,
                iloc_pos,
                save_path=output_path / f'shap_waterfall_{case_name}.png'
            )

    results['interesting_cases'] = interesting
    results['top_features'] = top_features

    # Print summary
    print("\n" + "="*60)
    print("SHAP ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nExpected value (base prediction): {expected_value:.2f}")
    print(f"\nTop 5 features by mean |SHAP|:")
    for i, feature in enumerate(top_features, 1):
        mean_impact = mean_abs_shap[X_sample.columns.get_loc(feature)]
        print(f"  {i}. {feature}: {mean_impact:.4f}")

    print(f"\nGenerated SHAP visualizations:")
    print(f"  - Summary plot")
    print(f"  - Dependence plots for top 5 features")
    print(f"  - Waterfall plots for {len(waterfall_cases)} interesting cases")

    return results
