"""
Baseline models for F1 race finish position prediction.
Includes dummy baselines, linear models, and tree-based models.
"""

import numpy as np
import pandas as pd
import pickle
import time
import warnings
from typing import Dict, Tuple, Any

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

warnings.filterwarnings('ignore')


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def load_modeling_data(data_dir: str = 'data/processed') -> Tuple:
    """
    Load train/val/test splits from CSV files.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    # Target is always 'Position'
    target = 'Position'

    # Separate features and target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return X_train, X_val, X_test, y_train, y_val, y_test


def verify_data_integrity(X_train, X_val, X_test, y_train, y_val, y_test):
    """Check data quality and consistency."""
    print("=" * 60)
    print("DATA INTEGRITY CHECK")
    print("=" * 60)

    # Check shapes
    print(f"\nDataset Shapes:")
    print(f"  Train: {X_train.shape} features, {y_train.shape} samples")
    print(f"  Val:   {X_val.shape} features, {y_val.shape} samples")
    print(f"  Test:  {X_test.shape} features, {y_test.shape} samples")

    # Check for NaN values
    train_nans = X_train.isna().sum().sum()
    val_nans = X_val.isna().sum().sum()
    test_nans = X_test.isna().sum().sum()

    print(f"\nMissing Values:")
    print(f"  Train: {train_nans} NaN values")
    print(f"  Val:   {val_nans} NaN values")
    print(f"  Test:  {test_nans} NaN values")

    # Check feature consistency
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)

    if train_cols == val_cols == test_cols:
        print(f"\n✓ Feature consistency: All splits have {len(train_cols)} identical features")
    else:
        print("\n✗ Feature inconsistency detected!")

    # Check target distribution
    print(f"\nTarget Distribution:")
    print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, Range: [{y_train.min()}, {y_train.max()}]")
    print(f"  Val   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}, Range: [{y_val.min()}, {y_val.max()}]")
    print(f"  Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, Range: [{y_test.min()}, {y_test.max()}]")

    print("=" * 60)


def evaluate_model(model, X, y_true, set_name: str = "") -> Dict[str, float]:
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        model: Trained model
        X: Feature matrix
        y_true: True target values
        set_name: Name of dataset (train/val/test)

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Calculate error percentiles
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


def train_dummy_baseline(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train dummy baseline models."""
    results = {}

    print("\n" + "=" * 60)
    print("DUMMY BASELINES")
    print("=" * 60)

    # Grid position baseline (predict finish = grid)
    if 'GridPosition' in X_train.columns:
        grid_mae = mean_absolute_error(y_val, X_val['GridPosition'])
        print(f"\nGrid Position Baseline:")
        print(f"  MAE: {grid_mae:.3f} positions")
        print(f"  Interpretation: On average, drivers finish {grid_mae:.1f} positions from where they started")

        results['grid_baseline'] = {
            'model': None,
            'predictions': X_val['GridPosition'].values,
            'mae': grid_mae
        }

    # Mean baseline
    dummy_mean = DummyRegressor(strategy='mean')
    start = time.time()
    dummy_mean.fit(X_train, y_train)
    train_time = time.time() - start

    metrics = evaluate_model(dummy_mean, X_val, y_val, "Dummy (Mean)")

    results['dummy_mean'] = {
        'model': dummy_mean,
        'metrics': metrics,
        'train_time': train_time
    }

    # Median baseline
    dummy_median = DummyRegressor(strategy='median')
    dummy_median.fit(X_train, y_train)
    metrics = evaluate_model(dummy_median, X_val, y_val, "Dummy (Median)")

    results['dummy_median'] = {
        'model': dummy_median,
        'metrics': metrics,
        'train_time': train_time
    }

    return results


def train_mean_baseline(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train historical mean baseline."""
    print("\n" + "=" * 60)
    print("HISTORICAL MEAN BASELINE")
    print("=" * 60)

    results = {}

    # Grid position mean
    if 'GridPosition' in X_train.columns:
        # Calculate average finish position for each grid position
        grid_lookup = {}
        for grid_pos in range(1, 21):
            mask = X_train['GridPosition'] == grid_pos
            if mask.sum() > 0:
                grid_lookup[grid_pos] = y_train[mask].mean()
            else:
                grid_lookup[grid_pos] = grid_pos  # fallback to grid position

        # Make predictions
        y_pred = X_val['GridPosition'].map(grid_lookup)
        mae = mean_absolute_error(y_val, y_pred)

        print(f"\nGrid Position Lookup Table:")
        print(f"  MAE: {mae:.3f} positions")
        print(f"  Improvement over dummy: {(results.get('dummy_mean', {}).get('metrics', {}).get('mae', 0) - mae):.3f}")

        # Show some lookup values
        print(f"\n  Sample grid→finish mappings:")
        for pos in [1, 5, 10, 15, 20]:
            if pos in grid_lookup:
                print(f"    Grid P{pos} → Avg finish: {grid_lookup[pos]:.2f}")

        results['grid_mean'] = {
            'model': grid_lookup,
            'predictions': y_pred.values,
            'mae': mae
        }

    return results


def train_linear_regression(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train linear regression model."""
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)

    model = LinearRegression()

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate on train and val
    train_metrics = evaluate_model(model, X_train, y_train, "Training Set")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation Set")

    # Analyze coefficients
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nTop 10 Most Important Features (by coefficient magnitude):")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:>10.4f}")

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def train_ridge_regression(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train ridge regression with hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("RIDGE REGRESSION")
    print("=" * 60)

    # Hyperparameter search
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    print(f"\nTuning alpha with {len(alphas)} candidates...")

    grid_search = GridSearchCV(
        Ridge(random_state=42),
        param_grid={'alpha': alphas},
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"Training time: {train_time:.2f}s")
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate
    train_metrics = evaluate_model(best_model, X_train, y_train, "Training Set")
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation Set")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': best_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:>10.4f}")

    # CV scores
    cv_results = pd.DataFrame(grid_search.cv_results_)

    return {
        'model': best_model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance,
        'cv_results': cv_results,
        'best_params': grid_search.best_params_
    }


def train_random_forest(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train random forest baseline."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Training Set")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation Set")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>10.4f}")

    # Calculate cumulative importance
    feature_importance['cumulative'] = feature_importance['importance'].cumsum()
    n_features_90 = (feature_importance['cumulative'] <= 0.90).sum()

    print(f"\nFeatures needed for 90% importance: {n_features_90}/{len(X_train.columns)}")

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def save_models(results: Dict, output_dir: str = 'results/models'):
    """Save trained models and results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save model objects
    with open(f'{output_dir}/baseline_models.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModels saved to {output_dir}/baseline_models.pkl")


def create_results_summary(results: Dict) -> pd.DataFrame:
    """Create summary table of all model results."""
    summary_data = []

    # Add each model's results
    for model_name, model_results in results.items():
        if 'val_metrics' in model_results:
            metrics = model_results['val_metrics']
            row = {
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'Median Error': metrics['median_error'],
                'P90 Error': metrics['p90_error'],
                'Train Time (s)': model_results.get('train_time', 0)
            }
            summary_data.append(row)

    df = pd.DataFrame(summary_data).sort_values('MAE')
    return df
