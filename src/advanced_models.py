"""
Advanced gradient boosting models for F1 race finish position prediction.
Includes XGBoost, LightGBM, and CatBoost with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, Tuple, Any, Optional

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


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


def train_xgboost_default(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """Train XGBoost with default parameters."""
    print("\n" + "=" * 60)
    print("XGBOOST - DEFAULT PARAMETERS")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1
    )

    start = time.time()
    model.fit(X_train, y_train, verbose=False)
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

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def create_xgboost_objective(X_train, y_train, n_folds: int = 5):
    """Create Optuna objective function for XGBoost hyperparameter tuning."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }

        model = XGBRegressor(**params)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=n_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        mae = -cv_scores.mean()
        return mae

    return objective


def tune_xgboost_optuna(X_train, y_train, n_trials: int = 100) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna."""
    print("\n" + "=" * 60)
    print("XGBOOST - HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 60)
    print(f"\nRunning {n_trials} optimization trials...")
    print("This may take 30-60 minutes...\n")

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )

    # Create objective
    objective = create_xgboost_objective(X_train, y_train)

    # Optimize with progress tracking
    start = time.time()

    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Best MAE = {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)

    total_time = time.time() - start

    print(f"\nOptimization complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best MAE: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return {
        'study': study,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'n_trials': n_trials,
        'optimization_time': total_time
    }


def train_xgboost_tuned(X_train, y_train, X_val, y_val, best_params: Dict) -> Dict[str, Any]:
    """Train XGBoost with tuned hyperparameters."""
    print("\n" + "=" * 60)
    print("XGBOOST - TUNED MODEL")
    print("=" * 60)

    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)

    start = time.time()
    model.fit(X_train, y_train, verbose=False)
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

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance,
        'params': best_params
    }


def create_lightgbm_objective(X_train, y_train, n_folds: int = 5):
    """Create Optuna objective function for LightGBM hyperparameter tuning."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        model = LGBMRegressor(**params)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=n_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        mae = -cv_scores.mean()
        return mae

    return objective


def train_lightgbm(X_train, y_train, X_val, y_val,
                   tune: bool = False, n_trials: int = 50) -> Dict[str, Any]:
    """Train LightGBM model with optional hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("LIGHTGBM" + (" - HYPERPARAMETER TUNING" if tune else " - DEFAULT PARAMETERS"))
    print("=" * 60)

    if tune:
        print(f"\nRunning {n_trials} optimization trials...")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )

        objective = create_lightgbm_objective(X_train, y_train)

        start_opt = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        opt_time = time.time() - start_opt

        print(f"\nOptimization complete! Time: {opt_time/60:.1f} minutes")
        print(f"Best MAE: {study.best_value:.4f}")

        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    else:
        best_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

    # Train final model
    model = LGBMRegressor(**best_params)

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

    result = {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance,
        'params': best_params
    }

    if tune:
        result['study'] = study
        result['optimization_time'] = opt_time

    return result


def create_catboost_objective(X_train, y_train, n_folds: int = 5):
    """Create Optuna objective function for CatBoost hyperparameter tuning."""

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostRegressor(**params)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=n_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        mae = -cv_scores.mean()
        return mae

    return objective


def train_catboost(X_train, y_train, X_val, y_val,
                   tune: bool = False, n_trials: int = 50) -> Dict[str, Any]:
    """Train CatBoost model with optional hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("CATBOOST" + (" - HYPERPARAMETER TUNING" if tune else " - DEFAULT PARAMETERS"))
    print("=" * 60)

    if tune:
        print(f"\nRunning {n_trials} optimization trials...")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )

        objective = create_catboost_objective(X_train, y_train)

        start_opt = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        opt_time = time.time() - start_opt

        print(f"\nOptimization complete! Time: {opt_time/60:.1f} minutes")
        print(f"Best MAE: {study.best_value:.4f}")

        best_params = study.best_params
        best_params.update({'random_seed': 42, 'verbose': False})
    else:
        best_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False
        }

    # Train final model
    model = CatBoostRegressor(**best_params)

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

    result = {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance,
        'params': best_params
    }

    if tune:
        result['study'] = study
        result['optimization_time'] = opt_time

    return result


def create_model_comparison(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Create comprehensive comparison table of all models."""
    comparison_data = []

    for model_name, result in results_dict.items():
        if 'val_metrics' in result:
            val_metrics = result['val_metrics']
            train_metrics = result.get('train_metrics', {})

            row = {
                'Model': model_name,
                'Train MAE': train_metrics.get('mae', np.nan),
                'Val MAE': val_metrics['mae'],
                'Val RMSE': val_metrics['rmse'],
                'Val R²': val_metrics['r2'],
                'Median Error': val_metrics['median_error'],
                'P90 Error': val_metrics['p90_error'],
                'Train Time (s)': result.get('train_time', 0),
                'Overfitting': train_metrics.get('mae', np.nan) - val_metrics['mae'] if 'train_metrics' in result else np.nan
            }
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data).sort_values('Val MAE')
    return df
