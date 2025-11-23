"""
Cross-validation analysis for model reliability assessment.

This module provides comprehensive cross-validation strategies:
- Standard K-Fold cross-validation
- Time series cross-validation
- Leave-one-circuit-out validation
- Overfitting and stability analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Any, Tuple
from pathlib import Path


def perform_kfold_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-Fold cross-validation with multiple metrics.

    Args:
        model: Model to evaluate
        X: Features
        y: Target
        n_splits: Number of folds
        shuffle: Whether to shuffle data
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    print(f"Performing {n_splits}-Fold cross-validation...")

    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Process results
    results = {
        'train_mae': -cv_results['train_mae'],
        'test_mae': -cv_results['test_mae'],
        'train_rmse': -cv_results['train_rmse'],
        'test_rmse': -cv_results['test_rmse'],
        'train_r2': cv_results['train_r2'],
        'test_r2': cv_results['test_r2']
    }

    # Calculate statistics
    stats = {
        'mae_mean': results['test_mae'].mean(),
        'mae_std': results['test_mae'].std(),
        'rmse_mean': results['test_rmse'].mean(),
        'rmse_std': results['test_rmse'].std(),
        'r2_mean': results['test_r2'].mean(),
        'r2_std': results['test_r2'].std(),
        'overfitting_mae': results['test_mae'].mean() - results['train_mae'].mean(),
        'overfitting_rmse': results['test_rmse'].mean() - results['train_rmse'].mean()
    }

    print(f"\nK-Fold CV Results:")
    print(f"  MAE: {stats['mae_mean']:.3f} ± {stats['mae_std']:.3f}")
    print(f"  RMSE: {stats['rmse_mean']:.3f} ± {stats['rmse_std']:.3f}")
    print(f"  R²: {stats['r2_mean']:.3f} ± {stats['r2_std']:.3f}")
    print(f"  Overfitting (MAE): {stats['overfitting_mae']:.3f}")

    results['stats'] = stats

    return results


def perform_timeseries_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform time series cross-validation.

    Args:
        model: Model to evaluate
        X: Features
        y: Target
        n_splits: Number of splits

    Returns:
        Dictionary with time series CV results
    """
    print(f"\nPerforming Time Series {n_splits}-Split cross-validation...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        # Evaluate
        train_pred = fold_model.predict(X_train_fold)
        val_pred = fold_model.predict(X_val_fold)

        train_mae = mean_absolute_error(y_train_fold, train_pred)
        val_mae = mean_absolute_error(y_val_fold, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        val_r2 = r2_score(y_val_fold, val_pred)

        fold_scores.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        })

        print(f"  Fold {fold + 1}: MAE={val_mae:.3f}, RMSE={val_rmse:.3f}, R²={val_r2:.3f}")

    fold_df = pd.DataFrame(fold_scores)

    stats = {
        'mae_mean': fold_df['val_mae'].mean(),
        'mae_std': fold_df['val_mae'].std(),
        'rmse_mean': fold_df['val_rmse'].mean(),
        'rmse_std': fold_df['val_rmse'].std(),
        'r2_mean': fold_df['val_r2'].mean(),
        'r2_std': fold_df['val_r2'].std(),
        'performance_trend': 'improving' if fold_df['val_mae'].iloc[-1] < fold_df['val_mae'].iloc[0] else 'declining'
    }

    print(f"\nTime Series CV Summary:")
    print(f"  MAE: {stats['mae_mean']:.3f} ± {stats['mae_std']:.3f}")
    print(f"  Performance trend: {stats['performance_trend']}")

    return {
        'fold_scores': fold_df,
        'stats': stats
    }


def perform_leave_one_circuit_out_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    circuit_column: str = 'circuit_name'
) -> Dict[str, Any]:
    """
    Perform leave-one-circuit-out cross-validation.

    Args:
        model: Model to evaluate
        X: Features
        y: Target
        circuit_column: Name of circuit identifier column

    Returns:
        Dictionary with LOCO CV results
    """
    print(f"\nPerforming Leave-One-Circuit-Out cross-validation...")

    if circuit_column not in X.columns:
        print(f"Warning: {circuit_column} not found in features. Skipping LOCO CV.")
        return None

    circuits = X[circuit_column].unique()
    print(f"  Testing on {len(circuits)} circuits...")

    loco_scores = []

    for circuit in circuits:
        # Split by circuit
        train_mask = X[circuit_column] != circuit
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[~train_mask]
        y_val = y[~train_mask]

        if len(X_val) == 0:
            continue

        # Train model
        loco_model = clone(model)
        loco_model.fit(X_train, y_train)

        # Evaluate
        val_pred = loco_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        loco_scores.append({
            'circuit': circuit,
            'n_samples': len(X_val),
            'mae': val_mae,
            'rmse': val_rmse
        })

    loco_df = pd.DataFrame(loco_scores).sort_values('mae')

    stats = {
        'mae_mean': loco_df['mae'].mean(),
        'mae_std': loco_df['mae'].std(),
        'mae_min': loco_df['mae'].min(),
        'mae_max': loco_df['mae'].max(),
        'easiest_circuit': loco_df.iloc[0]['circuit'],
        'hardest_circuit': loco_df.iloc[-1]['circuit']
    }

    print(f"\nLOCO CV Results:")
    print(f"  Average MAE: {stats['mae_mean']:.3f} ± {stats['mae_std']:.3f}")
    print(f"  Best circuit: {stats['easiest_circuit']} (MAE={stats['mae_min']:.3f})")
    print(f"  Worst circuit: {stats['hardest_circuit']} (MAE={stats['mae_max']:.3f})")

    return {
        'circuit_scores': loco_df,
        'stats': stats
    }


def plot_cv_results(
    kfold_results: Dict[str, Any],
    timeseries_results: Dict[str, Any],
    save_path: str = None
) -> None:
    """
    Visualize cross-validation results.

    Args:
        kfold_results: K-Fold CV results
        timeseries_results: Time series CV results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # K-Fold train vs test
    ax = axes[0, 0]
    x = np.arange(len(kfold_results['test_mae']))
    width = 0.35

    ax.bar(x - width/2, kfold_results['train_mae'], width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, kfold_results['test_mae'], width, label='Test', color='coral', alpha=0.7)

    ax.set_xlabel('Fold', fontsize=10)
    ax.set_ylabel('MAE', fontsize=10)
    ax.set_title('K-Fold CV: Train vs Test MAE', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in x], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # K-Fold metric comparison
    ax = axes[0, 1]
    metrics = ['MAE', 'RMSE']
    train_scores = [kfold_results['train_mae'].mean(), kfold_results['train_rmse'].mean()]
    test_scores = [kfold_results['test_mae'].mean(), kfold_results['test_rmse'].mean()]
    test_stds = [kfold_results['stats']['mae_std'], kfold_results['stats']['rmse_std']]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, train_scores, width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, test_scores, width, yerr=test_stds, label='Test',
           color='coral', alpha=0.7, capsize=5)

    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('K-Fold CV: Average Metrics', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Time series performance over folds
    if timeseries_results:
        ax = axes[1, 0]
        fold_df = timeseries_results['fold_scores']

        ax.plot(fold_df['fold'], fold_df['train_mae'], 'o-', label='Train MAE',
                color='steelblue', linewidth=2)
        ax.plot(fold_df['fold'], fold_df['val_mae'], 's-', label='Val MAE',
                color='coral', linewidth=2)

        ax.set_xlabel('Fold', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.set_title('Time Series CV: Performance Over Time', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Time series R² over folds
        ax = axes[1, 1]
        ax.plot(fold_df['fold'], fold_df['val_r2'], 'o-', color='green', linewidth=2)
        ax.set_xlabel('Fold', fontsize=10)
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title('Time Series CV: R² Over Time', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CV results plot to {save_path}")

    plt.close()


def plot_loco_results(
    loco_results: Dict[str, Any],
    save_path: str = None
) -> None:
    """
    Visualize leave-one-circuit-out CV results.

    Args:
        loco_results: LOCO CV results
        save_path: Path to save figure
    """
    circuit_df = loco_results['circuit_scores']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by MAE
    circuit_df = circuit_df.sort_values('mae')

    colors = ['green' if mae < circuit_df['mae'].mean() else 'orange'
              for mae in circuit_df['mae']]

    ax.barh(range(len(circuit_df)), circuit_df['mae'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(circuit_df)))
    ax.set_yticklabels(circuit_df['circuit'], fontsize=8)
    ax.set_xlabel('MAE (positions)', fontsize=10)
    ax.set_title('Leave-One-Circuit-Out CV: Prediction Difficulty by Circuit',
                 fontsize=12, fontweight='bold')
    ax.axvline(circuit_df['mae'].mean(), color='red', linestyle='--',
               label=f"Mean: {circuit_df['mae'].mean():.2f}")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved LOCO results plot to {save_path}")

    plt.close()


def analyze_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str = 'results/figures',
    n_folds: int = 5,
    circuit_column: str = None
) -> Dict[str, Any]:
    """
    Comprehensive cross-validation analysis.

    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training target
        output_dir: Directory to save plots
        n_folds: Number of CV folds
        circuit_column: Column name for circuit identifier

    Returns:
        Dictionary with all CV results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # K-Fold CV
    kfold_results = perform_kfold_cv(model, X_train, y_train, n_splits=n_folds)
    results['kfold'] = kfold_results

    # Time Series CV
    timeseries_results = perform_timeseries_cv(model, X_train, y_train, n_splits=n_folds)
    results['timeseries'] = timeseries_results

    # Plot standard CV results
    plot_cv_results(
        kfold_results,
        timeseries_results,
        save_path=output_path / 'cv_results.png'
    )

    # Leave-One-Circuit-Out CV
    if circuit_column and circuit_column in X_train.columns:
        loco_results = perform_leave_one_circuit_out_cv(model, X_train, y_train, circuit_column)
        if loco_results:
            results['loco'] = loco_results
            plot_loco_results(loco_results, save_path=output_path / 'loco_results.png')

    # Overall summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"\nK-Fold CV ({n_folds} folds):")
    print(f"  MAE: {kfold_results['stats']['mae_mean']:.3f} ± {kfold_results['stats']['mae_std']:.3f}")
    print(f"  RMSE: {kfold_results['stats']['rmse_mean']:.3f} ± {kfold_results['stats']['rmse_std']:.3f}")
    print(f"  R²: {kfold_results['stats']['r2_mean']:.3f} ± {kfold_results['stats']['r2_std']:.3f}")
    print(f"  Overfitting gap: {kfold_results['stats']['overfitting_mae']:.3f} positions")

    print(f"\nTime Series CV ({n_folds} splits):")
    print(f"  MAE: {timeseries_results['stats']['mae_mean']:.3f} ± {timeseries_results['stats']['mae_std']:.3f}")
    print(f"  Trend: {timeseries_results['stats']['performance_trend']}")

    if 'loco' in results:
        print(f"\nLeave-One-Circuit-Out CV:")
        print(f"  MAE: {results['loco']['stats']['mae_mean']:.3f} ± {results['loco']['stats']['mae_std']:.3f}")
        print(f"  Range: {results['loco']['stats']['mae_min']:.3f} - {results['loco']['stats']['mae_max']:.3f}")
        print(f"  Hardest: {results['loco']['stats']['hardest_circuit']}")

    return results
