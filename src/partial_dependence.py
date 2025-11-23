"""
Partial Dependence Plot (PDP) analysis for model interpretation.

This module provides PDP and ICE (Individual Conditional Expectation) analysis:
- One-way PDPs showing marginal feature effects
- Two-way PDPs for interaction visualization
- ICE plots showing individual prediction curves
- Automated identification of non-linear patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def select_pdp_features(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 5,
    include_features: List[str] = None
) -> List[str]:
    """
    Select features for PDP analysis based on importance.

    Args:
        X: Feature data
        importance_df: DataFrame with feature importance
        top_n: Number of top features to include
        include_features: Additional features to include

    Returns:
        List of feature names for PDP
    """
    # Get top features from importance
    top_features = importance_df.head(top_n)['feature'].tolist()

    # Add any explicitly requested features
    if include_features:
        for feature in include_features:
            if feature in X.columns and feature not in top_features:
                top_features.append(feature)

    # Ensure features exist in data
    pdp_features = [f for f in top_features if f in X.columns]

    print(f"Selected {len(pdp_features)} features for PDP analysis:")
    for i, feature in enumerate(pdp_features, 1):
        print(f"  {i}. {feature}")

    return pdp_features


def plot_one_way_pdp(
    model,
    X: pd.DataFrame,
    features: List[str],
    grid_resolution: int = 50,
    save_path: str = None
) -> None:
    """
    Create one-way partial dependence plots for features.

    Args:
        model: Trained model
        X: Feature data
        features: List of features to plot
        grid_resolution: Number of grid points
        save_path: Path to save figure
    """
    print(f"Creating one-way PDPs for {len(features)} features...")

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_features > 1 else axes

    # Create PDP display
    pdp_display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features,
        kind='average',
        grid_resolution=grid_resolution,
        ax=axes[:n_features]
    )

    # Customize each subplot
    for idx, feature in enumerate(features):
        axes[idx].set_title(f'PDP: {feature}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Predicted Position', fontsize=9)
        axes[idx].grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved one-way PDP to {save_path}")

    plt.close()


def plot_two_way_pdp(
    model,
    X: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]],
    grid_resolution: int = 20,
    save_path: str = None
) -> None:
    """
    Create two-way partial dependence plots for feature interactions.

    Args:
        model: Trained model
        X: Feature data
        feature_pairs: List of (feature1, feature2) tuples
        grid_resolution: Number of grid points per feature
        save_path: Path to save figure
    """
    print(f"Creating two-way PDPs for {len(feature_pairs)} feature pairs...")

    n_pairs = len(feature_pairs)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_pairs > 1 else axes

    for idx, (feat1, feat2) in enumerate(feature_pairs):
        if idx < len(axes):
            # Create 2D PDP
            pdp_display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=[(feat1, feat2)],
                kind='average',
                grid_resolution=grid_resolution,
                ax=axes[idx]
            )

            axes[idx].set_title(f'{feat1} × {feat2}', fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved two-way PDP to {save_path}")

    plt.close()


def plot_ice_plots(
    model,
    X: pd.DataFrame,
    features: List[str],
    n_samples: int = 100,
    grid_resolution: int = 50,
    save_path: str = None
) -> None:
    """
    Create ICE (Individual Conditional Expectation) plots.

    Args:
        model: Trained model
        X: Feature data
        features: List of features to plot
        n_samples: Number of samples to show
        grid_resolution: Number of grid points
        save_path: Path to save figure
    """
    print(f"Creating ICE plots for {len(features)} features...")

    # Sample data for clarity
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_features > 1 else axes

    # Create ICE plots
    pdp_display = PartialDependenceDisplay.from_estimator(
        model,
        X_sample,
        features=features,
        kind='both',  # Both PDP and ICE
        grid_resolution=grid_resolution,
        ax=axes[:n_features],
        ice_lines_kw={'alpha': 0.3, 'linewidth': 0.5},
        pd_line_kw={'color': 'red', 'linewidth': 2}
    )

    # Customize each subplot
    for idx, feature in enumerate(features):
        axes[idx].set_title(f'ICE Plot: {feature}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Predicted Position', fontsize=9)
        axes[idx].grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ICE plots to {save_path}")

    plt.close()


def identify_feature_interactions(
    X: pd.DataFrame,
    top_features: List[str],
    max_pairs: int = 3
) -> List[Tuple[str, str]]:
    """
    Identify likely feature interactions for 2D PDP analysis.

    Args:
        X: Feature data
        top_features: List of important features
        max_pairs: Maximum number of pairs to return

    Returns:
        List of (feature1, feature2) tuples
    """
    interaction_pairs = []

    # Define logical interaction pairs
    interaction_patterns = [
        ('grid', 'circuit'),
        ('grid', 'team'),
        ('driver', 'circuit'),
        ('team', 'year'),
        ('grid', 'driver')
    ]

    # Match features to patterns
    for pattern1, pattern2 in interaction_patterns:
        if len(interaction_pairs) >= max_pairs:
            break

        feat1 = next((f for f in top_features if pattern1 in f.lower()), None)
        feat2 = next((f for f in top_features if pattern2 in f.lower()), None)

        if feat1 and feat2 and feat1 != feat2:
            interaction_pairs.append((feat1, feat2))

    # If not enough pairs, use top features
    if len(interaction_pairs) < max_pairs:
        for i in range(min(3, len(top_features))):
            for j in range(i + 1, min(3, len(top_features))):
                if len(interaction_pairs) >= max_pairs:
                    break
                pair = (top_features[i], top_features[j])
                if pair not in interaction_pairs:
                    interaction_pairs.append(pair)

    print(f"\nIdentified {len(interaction_pairs)} feature interaction pairs:")
    for feat1, feat2 in interaction_pairs:
        print(f"  {feat1} × {feat2}")

    return interaction_pairs


def analyze_pdp_insights(
    model,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50
) -> Dict[str, Any]:
    """
    Extract insights from PDP curve.

    Args:
        model: Trained model
        X: Feature data
        feature: Feature to analyze
        grid_resolution: Number of grid points

    Returns:
        Dictionary with insights
    """
    from sklearn.inspection import partial_dependence

    # Calculate PDP
    pdp_result = partial_dependence(
        model,
        X,
        features=[feature],
        grid_resolution=grid_resolution,
        kind='average'
    )

    grid_values = pdp_result['grid_values'][0]
    pdp_values = pdp_result['average'][0]

    # Analyze curve
    insights = {}

    # Overall trend
    if pdp_values[-1] > pdp_values[0]:
        insights['trend'] = 'increasing'
    elif pdp_values[-1] < pdp_values[0]:
        insights['trend'] = 'decreasing'
    else:
        insights['trend'] = 'flat'

    # Linearity check
    x_scaled = (grid_values - grid_values.min()) / (grid_values.max() - grid_values.min())
    y_scaled = (pdp_values - pdp_values.min()) / (pdp_values.max() - pdp_values.min() + 1e-10)

    linear_fit = np.polyfit(x_scaled, y_scaled, 1)
    linear_pred = np.polyval(linear_fit, x_scaled)
    r_squared = 1 - (np.sum((y_scaled - linear_pred) ** 2) /
                     (np.sum((y_scaled - y_scaled.mean()) ** 2) + 1e-10))

    insights['linearity'] = 'linear' if r_squared > 0.95 else 'non-linear'
    insights['r_squared'] = r_squared

    # Find inflection points (where derivative changes significantly)
    derivatives = np.diff(pdp_values)
    if len(derivatives) > 1:
        deriv_changes = np.diff(np.sign(derivatives))
        n_inflections = np.sum(np.abs(deriv_changes) > 0)
        insights['n_inflections'] = n_inflections
    else:
        insights['n_inflections'] = 0

    # Effect size
    effect_size = pdp_values.max() - pdp_values.min()
    insights['effect_size'] = effect_size

    return insights


def analyze_partial_dependence(
    model,
    X_val: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: str = 'results/figures',
    top_n_features: int = 6,
    ice_n_samples: int = 100
) -> Dict[str, Any]:
    """
    Comprehensive partial dependence analysis.

    Args:
        model: Trained model
        X_val: Validation features
        importance_df: Feature importance DataFrame
        output_dir: Directory to save plots
        top_n_features: Number of top features for analysis
        ice_n_samples: Number of samples for ICE plots

    Returns:
        Dictionary with PDP results and insights
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Select features for PDP
    pdp_features = select_pdp_features(X_val, importance_df, top_n=top_n_features)
    results['pdp_features'] = pdp_features

    # Create one-way PDPs
    plot_one_way_pdp(
        model,
        X_val,
        pdp_features,
        grid_resolution=50,
        save_path=output_path / 'pdp_one_way.png'
    )

    # Identify and plot feature interactions
    interaction_pairs = identify_feature_interactions(X_val, pdp_features, max_pairs=3)
    results['interaction_pairs'] = interaction_pairs

    plot_two_way_pdp(
        model,
        X_val,
        interaction_pairs,
        grid_resolution=20,
        save_path=output_path / 'pdp_two_way.png'
    )

    # Create ICE plots for top features
    ice_features = pdp_features[:4]  # Limit to top 4 for clarity
    plot_ice_plots(
        model,
        X_val,
        ice_features,
        n_samples=ice_n_samples,
        grid_resolution=50,
        save_path=output_path / 'ice_plots.png'
    )

    # Analyze insights for each feature
    feature_insights = {}
    for feature in pdp_features[:5]:  # Top 5
        insights = analyze_pdp_insights(model, X_val, feature)
        feature_insights[feature] = insights

    results['feature_insights'] = feature_insights

    # Print summary
    print("\n" + "="*60)
    print("PARTIAL DEPENDENCE ANALYSIS SUMMARY")
    print("="*60)

    print("\nFeature Effects:")
    for feature, insights in feature_insights.items():
        print(f"\n{feature}:")
        print(f"  Trend: {insights['trend']}")
        print(f"  Relationship: {insights['linearity']} (R²={insights['r_squared']:.3f})")
        print(f"  Effect size: {insights['effect_size']:.3f} positions")
        print(f"  Inflection points: {insights['n_inflections']}")

    print(f"\nGenerated visualizations:")
    print(f"  - One-way PDPs for {len(pdp_features)} features")
    print(f"  - Two-way PDPs for {len(interaction_pairs)} feature pairs")
    print(f"  - ICE plots for {len(ice_features)} features")

    return results
