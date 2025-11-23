"""
Ensemble models combining multiple base regressors.
Implements stacking, voting, and custom ensemble strategies.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def evaluate_ensemble(model, X, y_true, set_name: str = "") -> Dict[str, float]:
    """
    Evaluate ensemble model performance.

    Args:
        model: Trained ensemble
        X: Feature matrix
        y_true: True target values
        set_name: Dataset name

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    abs_errors = np.abs(y_true - y_pred)
    median_error = np.median(abs_errors)
    p90_error = np.percentile(abs_errors, 90)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'median_error': median_error,
        'p90_error': p90_error
    }

    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"  MAE:    {mae:.3f} positions")
        print(f"  RMSE:   {rmse:.3f}")
        print(f"  R²:     {r2:.3f}")
        print(f"  Median: {median_error:.3f}")
        print(f"  90th:   {p90_error:.3f}")

    return metrics


def create_stacking_ensemble(base_models: List[Tuple],
                            final_estimator=None,
                            passthrough: bool = False,
                            cv: int = 5):
    """
    Create stacking regressor ensemble.

    Args:
        base_models: List of (name, model) tuples
        final_estimator: Meta-learner (default: RidgeCV)
        passthrough: Include original features in meta-model
        cv: Cross-validation folds for meta-features

    Returns:
        StackingRegressor instance
    """
    if final_estimator is None:
        final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1
    )

    return ensemble


def train_stacking_ensemble(base_models: List[Tuple],
                           X_train, y_train, X_val, y_val,
                           final_estimator=None,
                           passthrough: bool = False):
    """Train stacking ensemble and evaluate."""
    print("\n" + "=" * 60)
    print("STACKING ENSEMBLE")
    print("=" * 60)

    print(f"\nBase models: {len(base_models)}")
    for name, _ in base_models:
        print(f"  - {name}")

    print(f"Passthrough original features: {passthrough}")

    ensemble = create_stacking_ensemble(
        base_models,
        final_estimator=final_estimator,
        passthrough=passthrough
    )

    start = time.time()
    ensemble.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate
    train_metrics = evaluate_ensemble(ensemble, X_train, y_train, "Training Set")
    val_metrics = evaluate_ensemble(ensemble, X_val, y_val, "Validation Set")

    # Meta-model analysis
    if hasattr(ensemble.final_estimator_, 'coef_'):
        print(f"\nMeta-model coefficients:")
        for i, (name, _) in enumerate(base_models):
            coef = ensemble.final_estimator_.coef_[i]
            print(f"  {name:<20} {coef:>10.4f}")

    return {
        'model': ensemble,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time
    }


def create_voting_ensemble(base_models: List[Tuple], weights=None):
    """
    Create voting regressor ensemble.

    Args:
        base_models: List of (name, model) tuples
        weights: Optional weights for each model

    Returns:
        VotingRegressor instance
    """
    ensemble = VotingRegressor(
        estimators=base_models,
        weights=weights,
        n_jobs=-1
    )

    return ensemble


def train_voting_ensemble(base_models: List[Tuple],
                         X_train, y_train, X_val, y_val,
                         weights=None):
    """Train voting ensemble and evaluate."""
    print("\n" + "=" * 60)
    if weights is None:
        print("VOTING ENSEMBLE (EQUAL WEIGHTS)")
    else:
        print("VOTING ENSEMBLE (WEIGHTED)")
    print("=" * 60)

    print(f"\nBase models: {len(base_models)}")
    for i, (name, _) in enumerate(base_models):
        weight = weights[i] if weights is not None else 1.0 / len(base_models)
        print(f"  - {name:<20} weight: {weight:.3f}")

    ensemble = create_voting_ensemble(base_models, weights=weights)

    start = time.time()
    ensemble.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate
    train_metrics = evaluate_ensemble(ensemble, X_train, y_train, "Training Set")
    val_metrics = evaluate_ensemble(ensemble, X_val, y_val, "Validation Set")

    return {
        'model': ensemble,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'weights': weights
    }


def calculate_optimal_weights(models: List, X_val, y_val):
    """
    Calculate optimal weights based on individual validation performance.

    Args:
        models: List of trained models
        X_val, y_val: Validation data

    Returns:
        Normalized weights (sum to 1)
    """
    print("\nCalculating optimal weights based on validation MAE...")

    maes = []
    for model in models:
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
        print(f"  Model MAE: {mae:.3f}")

    # Inverse MAE weights (better models get higher weight)
    inv_maes = 1.0 / np.array(maes)
    weights = inv_maes / inv_maes.sum()

    print(f"\nOptimal weights:")
    for i, w in enumerate(weights):
        print(f"  Model {i+1}: {w:.3f}")

    return weights.tolist()


def analyze_prediction_diversity(models: List[Tuple], X_val, y_val):
    """
    Analyze how different base model predictions are.

    Args:
        models: List of (name, model) tuples
        X_val, y_val: Validation data

    Returns:
        DataFrame with correlation matrix and disagreement stats
    """
    print("\n" + "=" * 60)
    print("PREDICTION DIVERSITY ANALYSIS")
    print("=" * 60)

    predictions = {}
    for name, model in models:
        predictions[name] = model.predict(X_val)

    pred_df = pd.DataFrame(predictions)

    # Correlation matrix
    correlation = pred_df.corr()
    print("\nPrediction Correlations:")
    print(correlation.to_string())

    # Prediction variance
    pred_variance = pred_df.var(axis=1)
    print(f"\nPrediction Variance Statistics:")
    print(f"  Mean: {pred_variance.mean():.3f}")
    print(f"  Std:  {pred_variance.std():.3f}")
    print(f"  Max:  {pred_variance.max():.3f}")

    # Find high disagreement cases
    high_disagreement = pred_variance.nlargest(10)
    print(f"\nTop 10 Cases with Highest Model Disagreement:")
    for idx in high_disagreement.index:
        print(f"  Index {idx}: Actual={y_val.iloc[idx]:.1f}, Variance={pred_variance[idx]:.3f}")
        for name in pred_df.columns:
            print(f"    {name}: {pred_df.loc[idx, name]:.2f}")

    return correlation, pred_variance


def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table for all ensemble methods."""
    comparison_data = []

    for ensemble_name, result in results_dict.items():
        if 'val_metrics' in result:
            val_metrics = result['val_metrics']
            train_metrics = result.get('train_metrics', {})

            row = {
                'Ensemble': ensemble_name,
                'Train MAE': train_metrics.get('mae', np.nan),
                'Val MAE': val_metrics['mae'],
                'Val RMSE': val_metrics['rmse'],
                'Val R²': val_metrics['r2'],
                'Median Error': val_metrics['median_error'],
                'P90 Error': val_metrics['p90_error'],
                'Train Time (s)': result.get('train_time', 0),
                'Overfitting': train_metrics.get('mae', np.nan) - val_metrics['mae']
            }
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data).sort_values('Val MAE')
    return df


def blend_predictions(models: List[Tuple], X, weights=None):
    """
    Simple blending of predictions (without fitting meta-model).

    Args:
        models: List of (name, model) tuples
        X: Features
        weights: Optional weights for each model

    Returns:
        Blended predictions
    """
    predictions = np.array([model.predict(X) for _, model in models])

    if weights is None:
        # Equal weights
        return predictions.mean(axis=0)
    else:
        # Weighted average
        weights = np.array(weights).reshape(-1, 1)
        return (predictions * weights).sum(axis=0)


def confidence_weighted_ensemble(models: List[Tuple], X, y_true=None):
    """
    Ensemble with confidence weighting (higher weight to more confident predictions).

    This is a simple implementation using prediction variance as uncertainty proxy.

    Args:
        models: List of (name, model) tuples
        X: Features
        y_true: Optional true values for validation

    Returns:
        Weighted predictions
    """
    predictions = np.array([model.predict(X) for _, model in models])

    # Use inverse of prediction std as confidence
    # Lower variance across similar predictions = higher confidence
    pred_std = predictions.std(axis=0)
    # Avoid division by zero
    confidences = 1.0 / (pred_std + 1e-6)

    # Weighted average by confidence
    weights = confidences / confidences.sum()

    final_pred = (predictions.T * weights).sum(axis=1)

    if y_true is not None:
        mae = mean_absolute_error(y_true, final_pred)
        print(f"\nConfidence-weighted ensemble MAE: {mae:.3f}")

    return final_pred, weights
