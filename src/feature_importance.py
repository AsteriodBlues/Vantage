"""
Feature importance analysis for F1 finish position prediction models.

This module provides comprehensive feature importance analysis using multiple methods:
- Built-in tree importance (gain, weight, cover)
- Permutation importance
- Grouped importance by feature category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from typing import Dict, List, Any, Tuple
import pickle
from pathlib import Path


def extract_xgboost_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract multiple importance types from XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with different importance measures
    """
    # Get built-in importance
    importance_gain = model.feature_importances_

    # Get importance scores from booster
    booster = model.get_booster()
    importance_weight = booster.get_score(importance_type='weight')
    importance_gain_dict = booster.get_score(importance_type='gain')
    importance_cover = booster.get_score(importance_type='cover')

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gain': importance_gain
    })

    # Add other importance types (need to map feature names)
    feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}

    importance_df['weight'] = importance_df['feature'].apply(
        lambda x: importance_weight.get(f'f{feature_names.index(x)}', 0)
    )
    importance_df['cover'] = importance_df['feature'].apply(
        lambda x: importance_cover.get(f'f{feature_names.index(x)}', 0)
    )

    # Sort by gain
    importance_df = importance_df.sort_values('gain', ascending=False)

    return importance_df


def calculate_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance for model features.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with permutation importance statistics
    """
    print("Calculating permutation importance...")

    perm_importance = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    perm_df = pd.DataFrame({
        'feature': X_val.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)

    return perm_df


def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Group features by category for analysis.

    Args:
        feature_names: List of all feature names

    Returns:
        Dictionary mapping category names to feature lists
    """
    categories = {
        'Grid': [],
        'Circuit': [],
        'Team': [],
        'Driver': [],
        'Temporal': [],
        'Interaction': []
    }

    for feature in feature_names:
        if 'grid' in feature.lower():
            categories['Grid'].append(feature)
        elif 'circuit' in feature.lower() or 'track' in feature.lower():
            categories['Circuit'].append(feature)
        elif 'team' in feature.lower() or 'constructor' in feature.lower():
            categories['Team'].append(feature)
        elif 'driver' in feature.lower():
            categories['Driver'].append(feature)
        elif any(x in feature.lower() for x in ['year', 'race_number', 'round']):
            categories['Temporal'].append(feature)
        elif '_x_' in feature.lower() or 'interaction' in feature.lower():
            categories['Interaction'].append(feature)
        else:
            # Default to most likely category based on content
            if 'avg' in feature.lower() or 'championship' in feature.lower():
                categories['Team'].append(feature)
            else:
                categories['Driver'].append(feature)

    return categories


def calculate_category_importance(
    importance_df: pd.DataFrame,
    categories: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Aggregate feature importance by category.

    Args:
        importance_df: DataFrame with feature importance
        categories: Dictionary mapping categories to features

    Returns:
        DataFrame with category-level importance
    """
    category_importance = []

    for category, features in categories.items():
        cat_features = [f for f in features if f in importance_df['feature'].values]
        if cat_features:
            total_importance = importance_df[
                importance_df['feature'].isin(cat_features)
            ]['gain'].sum()

            category_importance.append({
                'category': category,
                'importance': total_importance,
                'n_features': len(cat_features)
            })

    cat_df = pd.DataFrame(category_importance).sort_values('importance', ascending=False)
    return cat_df


def plot_top_features(
    builtin_importance: pd.DataFrame,
    perm_importance: pd.DataFrame,
    top_n: int = 20,
    save_path: str = None
) -> None:
    """
    Create side-by-side plots of top features by different importance measures.

    Args:
        builtin_importance: DataFrame with built-in importance
        perm_importance: DataFrame with permutation importance
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Built-in importance
    top_builtin = builtin_importance.head(top_n)
    axes[0].barh(range(top_n), top_builtin['gain'].values, color='steelblue')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_builtin['feature'].values, fontsize=9)
    axes[0].set_xlabel('Importance (Gain)', fontsize=11)
    axes[0].set_title(f'Top {top_n} Features - Built-in Importance', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Permutation importance with error bars
    top_perm = perm_importance.head(top_n)
    axes[1].barh(range(top_n), top_perm['importance_mean'].values, color='coral')
    axes[1].errorbar(
        top_perm['importance_mean'].values,
        range(top_n),
        xerr=top_perm['importance_std'].values,
        fmt='none',
        color='black',
        alpha=0.5,
        capsize=3
    )
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_perm['feature'].values, fontsize=9)
    axes[1].set_xlabel('Importance (MAE Increase)', fontsize=11)
    axes[1].set_title(f'Top {top_n} Features - Permutation Importance', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved top features plot to {save_path}")

    plt.close()


def plot_category_importance(
    category_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Create visualization of feature importance by category.

    Args:
        category_df: DataFrame with category importance
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = plt.cm.Set3(range(len(category_df)))
    axes[0].barh(range(len(category_df)), category_df['importance'].values, color=colors)
    axes[0].set_yticks(range(len(category_df)))
    axes[0].set_yticklabels(category_df['category'].values, fontsize=11)
    axes[0].set_xlabel('Total Importance', fontsize=11)
    axes[0].set_title('Feature Importance by Category', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Pie chart
    axes[1].pie(
        category_df['importance'].values,
        labels=category_df['category'].values,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    axes[1].set_title('Feature Category Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved category importance plot to {save_path}")

    plt.close()


def plot_cumulative_importance(
    importance_df: pd.DataFrame,
    save_path: str = None
) -> Tuple[int, int]:
    """
    Plot cumulative importance to show how many features matter.

    Args:
        importance_df: DataFrame with feature importance
        save_path: Path to save figure

    Returns:
        Tuple of (n_features_90, n_features_95)
    """
    # Calculate cumulative importance
    cumsum_importance = importance_df['gain'].cumsum() / importance_df['gain'].sum()

    # Find thresholds
    n_features_90 = (cumsum_importance <= 0.90).sum() + 1
    n_features_95 = (cumsum_importance <= 0.95).sum() + 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(cumsum_importance) + 1), cumsum_importance.values,
            linewidth=2, color='steelblue')
    ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=1.5,
               label=f'90% at {n_features_90} features')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5,
               label=f'95% at {n_features_95} features')
    ax.axvline(x=n_features_90, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=n_features_95, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Number of Features', fontsize=11)
    ax.set_ylabel('Cumulative Importance', fontsize=11)
    ax.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative importance plot to {save_path}")

    plt.close()

    return n_features_90, n_features_95


def analyze_feature_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = 'xgboost',
    output_dir: str = 'results/figures'
) -> Dict[str, Any]:
    """
    Comprehensive feature importance analysis.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        model_type: Type of model ('xgboost', 'random_forest', etc.)
        output_dir: Directory to save plots

    Returns:
        Dictionary with importance results and insights
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = list(X_val.columns)
    results = {}

    # Extract built-in importance
    print("Extracting built-in feature importance...")
    if model_type == 'xgboost':
        builtin_importance = extract_xgboost_importance(model, feature_names)
    else:
        builtin_importance = pd.DataFrame({
            'feature': feature_names,
            'gain': model.feature_importances_
        }).sort_values('gain', ascending=False)

    results['builtin_importance'] = builtin_importance

    # Calculate permutation importance
    perm_importance = calculate_permutation_importance(model, X_val, y_val)
    results['permutation_importance'] = perm_importance

    # Categorize features
    categories = categorize_features(feature_names)
    category_importance = calculate_category_importance(builtin_importance, categories)
    results['category_importance'] = category_importance

    # Create visualizations
    plot_top_features(
        builtin_importance,
        perm_importance,
        top_n=20,
        save_path=output_path / 'top_features.png'
    )

    plot_category_importance(
        category_importance,
        save_path=output_path / 'category_importance.png'
    )

    n_90, n_95 = plot_cumulative_importance(
        builtin_importance,
        save_path=output_path / 'cumulative_importance.png'
    )

    results['n_features_90'] = n_90
    results['n_features_95'] = n_95

    # Generate insights
    top_5 = builtin_importance.head(5)['feature'].tolist()
    top_5_importance = builtin_importance.head(5)['gain'].sum() / builtin_importance['gain'].sum()

    meaningful_features = (builtin_importance['gain'] / builtin_importance['gain'].sum() > 0.01).sum()

    results['insights'] = {
        'top_5_features': top_5,
        'top_5_contribution': f"{top_5_importance:.1%}",
        'meaningful_features': meaningful_features,
        'features_for_90_percent': n_90,
        'features_for_95_percent': n_95,
        'dominant_category': category_importance.iloc[0]['category'],
        'dominant_category_importance': f"{category_importance.iloc[0]['importance'] / category_importance['importance'].sum():.1%}"
    }

    # Print summary
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTop 5 Most Important Features:")
    for i, feature in enumerate(top_5, 1):
        imp = builtin_importance[builtin_importance['feature'] == feature]['gain'].values[0]
        print(f"  {i}. {feature}: {imp:.4f}")

    print(f"\nTop 5 features explain {top_5_importance:.1%} of model decisions")
    print(f"{meaningful_features} features contribute >1% importance")
    print(f"{n_90} features needed for 90% cumulative importance")
    print(f"{n_95} features needed for 95% cumulative importance")

    print(f"\nDominant Feature Category: {category_importance.iloc[0]['category']}")
    print(f"  Contributes {results['insights']['dominant_category_importance']} of total importance")

    print("\nCategory Breakdown:")
    for _, row in category_importance.iterrows():
        pct = row['importance'] / category_importance['importance'].sum()
        print(f"  {row['category']}: {pct:.1%} ({row['n_features']} features)")

    return results
