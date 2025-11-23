"""
Classification models for F1 race predictions.
Includes win probability and podium probability classifiers with imbalance handling.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, Tuple, Any

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


def prepare_classification_targets(y_train, y_val, y_test, task='win'):
    """
    Prepare binary classification targets.

    Args:
        y_train, y_val, y_test: Position arrays
        task: 'win' (position==1) or 'podium' (position<=3)

    Returns:
        Binary target arrays and class statistics
    """
    if task == 'win':
        y_train_class = (y_train == 1).astype(int)
        y_val_class = (y_val == 1).astype(int)
        y_test_class = (y_test == 1).astype(int)
    elif task == 'podium':
        y_train_class = (y_train <= 3).astype(int)
        y_val_class = (y_val <= 3).astype(int)
        y_test_class = (y_test <= 3).astype(int)
    else:
        raise ValueError("task must be 'win' or 'podium'")

    # Class statistics
    stats = {
        'train_positive_rate': y_train_class.mean(),
        'val_positive_rate': y_val_class.mean(),
        'test_positive_rate': y_test_class.mean(),
        'train_counts': np.bincount(y_train_class),
        'val_counts': np.bincount(y_val_class)
    }

    print(f"\n{task.upper()} Classification - Class Balance:")
    print(f"  Training:   {stats['train_positive_rate']:.2%} positive ({stats['train_counts'][1]}/{len(y_train_class)})")
    print(f"  Validation: {stats['val_positive_rate']:.2%} positive ({stats['val_counts'][1]}/{len(y_val_class)})")

    return y_train_class, y_val_class, y_test_class, stats


def evaluate_classifier(model, X, y_true, set_name: str = "", threshold: float = 0.5) -> Dict:
    """
    Comprehensive classifier evaluation.

    Args:
        model: Trained classifier
        X: Features
        y_true: True labels
        set_name: Dataset name for printing
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'avg_precision': average_precision_score(y_true, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'probabilities': y_pred_proba
    }

    if set_name:
        print(f"\n{set_name} Metrics (threshold={threshold}):")
        print(f"  Accuracy:    {metrics['accuracy']:.3f}")
        print(f"  Precision:   {metrics['precision']:.3f}")
        print(f"  Recall:      {metrics['recall']:.3f}")
        print(f"  F1 Score:    {metrics['f1']:.3f}")
        print(f"  ROC AUC:     {metrics['roc_auc']:.3f}")
        print(f"  Avg Prec:    {metrics['avg_precision']:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {metrics['confusion_matrix'][0,0]:3d}  FP: {metrics['confusion_matrix'][0,1]:3d}")
        print(f"    FN: {metrics['confusion_matrix'][1,0]:3d}  TP: {metrics['confusion_matrix'][1,1]:3d}")

    return metrics


def train_win_classifier(X_train, y_train_class, X_val, y_val_class, model_type='xgboost'):
    """Train classifier for win prediction."""
    print("\n" + "=" * 60)
    print(f"WIN PROBABILITY - {model_type.upper()}")
    print("=" * 60)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_class)
    scale_pos_weight = class_weights[1] / class_weights[0]

    print(f"\nClass imbalance handling:")
    print(f"  Scale positive weight: {scale_pos_weight:.2f}")

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train
    start = time.time()
    model.fit(X_train, y_train_class)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate
    train_metrics = evaluate_classifier(model, X_train, y_train_class, "Training Set")
    val_metrics = evaluate_classifier(model, X_val, y_val_class, "Validation Set")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Features for Win Prediction:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>10.4f}")
    else:
        feature_importance = None

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def train_podium_classifier(X_train, y_train_class, X_val, y_val_class, model_type='xgboost'):
    """Train classifier for podium prediction."""
    print("\n" + "=" * 60)
    print(f"PODIUM PROBABILITY - {model_type.upper()}")
    print("=" * 60)

    # Calculate class weights (less extreme than win)
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_class)
    scale_pos_weight = class_weights[1] / class_weights[0]

    print(f"\nClass imbalance handling:")
    print(f"  Scale positive weight: {scale_pos_weight:.2f}")

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,  # Can be deeper, more positive examples
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train
    start = time.time()
    model.fit(X_train, y_train_class)
    train_time = time.time() - start

    print(f"\nTraining time: {train_time:.2f}s")

    # Evaluate
    train_metrics = evaluate_classifier(model, X_train, y_train_class, "Training Set")
    val_metrics = evaluate_classifier(model, X_val, y_val_class, "Validation Set")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Features for Podium Prediction:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>10.4f}")
    else:
        feature_importance = None

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: 'f1', 'precision', or 'recall'

    Returns:
        Optimal threshold and corresponding metric value
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        results.append({
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score, pd.DataFrame(results)


def calibrate_classifier(model, X_train, y_train, method='isotonic'):
    """
    Calibrate classifier probabilities.

    Args:
        model: Trained classifier
        X_train, y_train: Training data
        method: 'isotonic' or 'sigmoid'

    Returns:
        Calibrated classifier
    """
    print(f"\nCalibrating classifier using {method} method...")

    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=3
    )

    start = time.time()
    calibrated.fit(X_train, y_train)
    calibration_time = time.time() - start

    print(f"Calibration complete in {calibration_time:.2f}s")

    return calibrated
