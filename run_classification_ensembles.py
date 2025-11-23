"""
Train classification models and ensemble regressors.
Comprehensive analysis of win/podium prediction and model ensembles.
"""

import sys
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import load_modeling_data, set_random_seeds
from src.classification_models import (
    prepare_classification_targets,
    train_win_classifier,
    train_podium_classifier,
    find_optimal_threshold,
    evaluate_classifier
)
from src.ensemble_models import (
    train_stacking_ensemble,
    train_voting_ensemble,
    calculate_optimal_weights,
    analyze_prediction_diversity,
    create_comparison_table
)

# Import base models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
set_random_seeds(42)

os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*80)
print("CLASSIFICATION MODELS AND ENSEMBLES")
print("="*80)

# Load data
print("\nLoading data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_modeling_data()

# ============================================================================
# PART 1: WIN PROBABILITY CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("PART 1: WIN PROBABILITY CLASSIFICATION")
print("="*80)

y_win_train, y_win_val, y_win_test, win_stats = prepare_classification_targets(
    y_train, y_val, y_test, task='win'
)

# Train multiple classifiers
win_xgb_results = train_win_classifier(
    X_train, y_win_train, X_val, y_win_val, model_type='xgboost'
)

win_catboost_results = train_win_classifier(
    X_train, y_win_train, X_val, y_win_val, model_type='catboost'
)

# Find optimal threshold
print("\n" + "-"*60)
print("OPTIMAL THRESHOLD ANALYSIS")
print("-"*60)

best_threshold, best_f1, threshold_df = find_optimal_threshold(
    y_win_val,
    win_xgb_results['val_metrics']['probabilities'],
    metric='f1'
)

print(f"\nOptimal threshold for F1: {best_threshold:.3f}")
print(f"Best F1 score: {best_f1:.3f}")

# Evaluate with optimal threshold
_ = evaluate_classifier(
    win_xgb_results['model'],
    X_val,
    y_win_val,
    "Validation (Optimized Threshold)",
    threshold=best_threshold
)

# ============================================================================
# PART 2: PODIUM PROBABILITY CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("PART 2: PODIUM PROBABILITY CLASSIFICATION")
print("="*80)

y_podium_train, y_podium_val, y_podium_test, podium_stats = prepare_classification_targets(
    y_train, y_val, y_test, task='podium'
)

podium_xgb_results = train_podium_classifier(
    X_train, y_podium_train, X_val, y_podium_val, model_type='xgboost'
)

podium_catboost_results = train_podium_classifier(
    X_train, y_podium_train, X_val, y_podium_val, model_type='catboost'
)

# ============================================================================
# PART 3: LOAD BEST REGRESSION MODELS FOR ENSEMBLES
# ============================================================================
print("\n" + "="*80)
print("PART 3: PREPARING ENSEMBLE BASE MODELS")
print("="*80)

# Load best parameters from previous work
with open('results/models/best_params.json', 'r') as f:
    import json
    best_params = json.load(f)

print("\nLoading best model configurations...")
print(f"  XGBoost params: {best_params['xgboost']['n_estimators']} estimators")
print(f"  LightGBM params: {best_params['lightgbm']['n_estimators']} estimators")
print(f"  CatBoost params: {best_params['catboost']['iterations']} iterations")

# Load trained models
with open('results/models/advanced_models.pkl', 'rb') as f:
    trained_models = pickle.load(f)

# Use the best models as base estimators
base_models = [
    ('catboost', trained_models['catboost_tuned']),
    ('lightgbm', trained_models['lightgbm_default']),
    ('xgboost', trained_models['xgboost_tuned'])
]

print(f"\n{len(base_models)} base models prepared for ensembling")

# ============================================================================
# PART 4: VOTING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("PART 4: VOTING ENSEMBLE")
print("="*80)

# Equal weights voting
voting_equal_results = train_voting_ensemble(
    base_models, X_train, y_train, X_val, y_val, weights=None
)

# Calculate optimal weights from individual validation performance
model_list = [model for _, model in base_models]
optimal_weights = calculate_optimal_weights(model_list, X_val, y_val)

# Weighted voting
voting_weighted_results = train_voting_ensemble(
    base_models, X_train, y_train, X_val, y_val, weights=optimal_weights
)

# ============================================================================
# PART 5: STACKING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("PART 5: STACKING ENSEMBLE")
print("="*80)

# Stacking without passthrough
stacking_results = train_stacking_ensemble(
    base_models, X_train, y_train, X_val, y_val,
    passthrough=False
)

# Stacking with passthrough (includes original features)
stacking_passthrough_results = train_stacking_ensemble(
    base_models, X_train, y_train, X_val, y_val,
    passthrough=True
)

# ============================================================================
# PART 6: DIVERSITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 6: MODEL DIVERSITY ANALYSIS")
print("="*80)

correlation_matrix, pred_variance = analyze_prediction_diversity(
    base_models, X_val, y_val
)

# ============================================================================
# PART 7: ENSEMBLE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 7: ENSEMBLE COMPARISON")
print("="*80)

# Include individual base models for comparison
individual_results = {}
for name, model in base_models:
    y_pred = model.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - y_val.mean())**2)

    individual_results[f'{name} (individual)'] = {
        'val_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'median_error': np.median(np.abs(y_val - y_pred)),
            'p90_error': np.percentile(np.abs(y_val - y_pred), 90)
        }
    }

all_results = {
    **individual_results,
    'Voting (equal)': voting_equal_results,
    'Voting (weighted)': voting_weighted_results,
    'Stacking': stacking_results,
    'Stacking (passthrough)': stacking_passthrough_results
}

comparison = create_comparison_table(all_results)
print("\n" + comparison.to_string(index=False))

# Save comparison
comparison.to_csv('results/models/ensemble_comparison.csv', index=False)
print("\nSaved: results/models/ensemble_comparison.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Classification ROC curves
from sklearn.metrics import roc_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Win ROC
for name, results in [('XGBoost', win_xgb_results), ('CatBoost', win_catboost_results)]:
    y_proba = results['val_metrics']['probabilities']
    fpr, tpr, _ = roc_curve(y_win_val, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Win Prediction - ROC Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Podium ROC
for name, results in [('XGBoost', podium_xgb_results), ('CatBoost', podium_catboost_results)]:
    y_proba = results['val_metrics']['probabilities']
    fpr, tpr, _ = roc_curve(y_podium_val, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Podium Prediction - ROC Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/classification_roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/classification_roc_curves.png")
plt.close()

# 2. Ensemble performance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAE comparison
ensemble_only = comparison[~comparison['Ensemble'].str.contains('individual')]
axes[0].barh(ensemble_only['Ensemble'], ensemble_only['Val MAE'], alpha=0.7)
axes[0].set_xlabel('Validation MAE (positions)')
axes[0].set_title('Ensemble Performance Comparison')
axes[0].invert_yaxis()
for i, v in enumerate(ensemble_only['Val MAE']):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center')

# R² comparison
axes[1].barh(ensemble_only['Ensemble'], ensemble_only['Val R²'], alpha=0.7, color='green')
axes[1].set_xlabel('R² Score')
axes[1].set_title('Variance Explained')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('results/figures/ensemble_performance.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/ensemble_performance.png")
plt.close()

# 3. Prediction correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=1.0, vmin=0.9, vmax=1.0, square=True)
plt.title('Base Model Prediction Correlations')
plt.tight_layout()
plt.savefig('results/figures/model_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/model_correlation.png")
plt.close()

# 4. Threshold analysis for win prediction
plt.figure(figsize=(12, 6))
plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', linewidth=2)
plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', linewidth=2)
plt.plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score', linewidth=2)
plt.axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Win Prediction: Threshold Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/threshold_optimization.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/threshold_optimization.png")
plt.close()

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

classification_models = {
    'win_xgboost': win_xgb_results['model'],
    'win_catboost': win_catboost_results['model'],
    'podium_xgboost': podium_xgb_results['model'],
    'podium_catboost': podium_catboost_results['model']
}

with open('results/models/classification_models.pkl', 'wb') as f:
    pickle.dump(classification_models, f)
print("Saved: results/models/classification_models.pkl")

ensemble_models = {
    'voting_equal': voting_equal_results['model'],
    'voting_weighted': voting_weighted_results['model'],
    'stacking': stacking_results['model'],
    'stacking_passthrough': stacking_passthrough_results['model']
}

with open('results/models/ensemble_models.pkl', 'wb') as f:
    pickle.dump(ensemble_models, f)
print("Saved: results/models/ensemble_models.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nClassification Performance:")
print(f"  Win Prediction (XGBoost):")
print(f"    ROC-AUC: {win_xgb_results['val_metrics']['roc_auc']:.3f}")
print(f"    Avg Precision: {win_xgb_results['val_metrics']['avg_precision']:.3f}")
print(f"  Podium Prediction (XGBoost):")
print(f"    ROC-AUC: {podium_xgb_results['val_metrics']['roc_auc']:.3f}")
print(f"    Avg Precision: {podium_xgb_results['val_metrics']['avg_precision']:.3f}")

print(f"\nEnsemble Performance:")
best_ensemble = comparison.iloc[0]
print(f"  Best Ensemble: {best_ensemble['Ensemble']}")
print(f"    Validation MAE: {best_ensemble['Val MAE']:.3f} positions")
print(f"    R²: {best_ensemble['Val R²']:.3f}")

# Compare to best individual
best_individual_mae = min([r['val_metrics']['mae'] for r in individual_results.values()])
improvement = ((best_individual_mae - best_ensemble['Val MAE']) / best_individual_mae) * 100

print(f"\n  Best Individual MAE: {best_individual_mae:.3f}")
print(f"  Improvement: {improvement:.1f}%")

print(f"\nModel Diversity:")
print(f"  Average correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
print(f"  Average prediction variance: {pred_variance.mean():.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
