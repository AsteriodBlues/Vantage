"""
Train and compare advanced gradient boosting models.
Includes XGBoost, LightGBM, and CatBoost with hyperparameter optimization.
"""

import sys
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import load_modeling_data, set_random_seeds
from src.advanced_models import (
    train_xgboost_default,
    tune_xgboost_optuna,
    train_xgboost_tuned,
    train_lightgbm,
    train_catboost,
    create_model_comparison
)

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
set_random_seeds(42)

# Create output directories
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*80)
print("ADVANCED GRADIENT BOOSTING MODELS")
print("="*80)

# Load data
print("\nLoading data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_modeling_data()
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Features: {X_train.shape[1]}")

# ============================================================================
# XGBOOST - DEFAULT
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: XGBOOST WITH DEFAULT PARAMETERS")
print("="*80)

xgb_default_results = train_xgboost_default(X_train, y_train, X_val, y_val)

# ============================================================================
# XGBOOST - HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: XGBOOST HYPERPARAMETER OPTIMIZATION")
print("="*80)

# Run optimization (fewer trials for faster execution, increase to 100 for production)
tuning_results = tune_xgboost_optuna(X_train, y_train, n_trials=50)

# Train model with best parameters
xgb_tuned_results = train_xgboost_tuned(
    X_train, y_train, X_val, y_val,
    tuning_results['best_params']
)

# ============================================================================
# LIGHTGBM
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: LIGHTGBM")
print("="*80)

# Default LightGBM
lgb_default_results = train_lightgbm(
    X_train, y_train, X_val, y_val,
    tune=False
)

# Tuned LightGBM
lgb_tuned_results = train_lightgbm(
    X_train, y_train, X_val, y_val,
    tune=True,
    n_trials=30
)

# ============================================================================
# CATBOOST
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: CATBOOST")
print("="*80)

# Default CatBoost
cb_default_results = train_catboost(
    X_train, y_train, X_val, y_val,
    tune=False
)

# Tuned CatBoost
cb_tuned_results = train_catboost(
    X_train, y_train, X_val, y_val,
    tune=True,
    n_trials=30
)

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

all_results = {
    'XGBoost (default)': xgb_default_results,
    'XGBoost (tuned)': xgb_tuned_results,
    'LightGBM (default)': lgb_default_results,
    'LightGBM (tuned)': lgb_tuned_results,
    'CatBoost (default)': cb_default_results,
    'CatBoost (tuned)': cb_tuned_results
}

comparison = create_model_comparison(all_results)
print("\n" + comparison.to_string(index=False))

# Save comparison
comparison.to_csv('results/models/advanced_models_comparison.csv', index=False)
print("\nComparison saved to: results/models/advanced_models_comparison.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Performance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE comparison
axes[0].barh(comparison['Model'], comparison['Val MAE'], alpha=0.7)
axes[0].set_xlabel('Validation MAE (positions)')
axes[0].set_title('Model Performance Comparison')
axes[0].invert_yaxis()
for i, v in enumerate(comparison['Val MAE']):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

# R² comparison
axes[1].barh(comparison['Model'], comparison['Val R²'], alpha=0.7, color='green')
axes[1].set_xlabel('R² Score')
axes[1].set_title('Variance Explained')
axes[1].invert_yaxis()

# Training time vs performance
axes[2].scatter(comparison['Train Time (s)'], comparison['Val MAE'], s=150, alpha=0.6)
for idx, row in comparison.iterrows():
    axes[2].annotate(row['Model'].split()[0],
                    (row['Train Time (s)'], row['Val MAE']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[2].set_xlabel('Training Time (seconds)')
axes[2].set_ylabel('Validation MAE')
axes[2].set_title('Efficiency: Speed vs Accuracy')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/advanced_models_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/advanced_models_comparison.png")
plt.close()

# 2. Feature importance comparison (top models)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

models_to_compare = [
    ('XGBoost (tuned)', xgb_tuned_results),
    ('LightGBM (tuned)', lgb_tuned_results),
    ('CatBoost (tuned)', cb_tuned_results)
]

for idx, (name, result) in enumerate(models_to_compare):
    top_features = result['feature_importance'].head(15)
    axes[idx].barh(range(len(top_features)), top_features['importance'], alpha=0.7)
    axes[idx].set_yticks(range(len(top_features)))
    axes[idx].set_yticklabels(top_features['feature'], fontsize=8)
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title(f'{name}\nTop 15 Features')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig('results/figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/feature_importance_comparison.png")
plt.close()

# 3. Overfitting analysis
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35

train_mae = comparison['Train MAE']
val_mae = comparison['Val MAE']

ax.bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.7)
ax.bar(x + width/2, val_mae, width, label='Val MAE', alpha=0.7)
ax.set_xlabel('Model')
ax.set_ylabel('MAE (positions)')
ax.set_title('Train vs Validation MAE (Overfitting Check)')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/overfitting_analysis.png")
plt.close()

# 4. XGBoost optimization history
if 'study' in tuning_results:
    study = tuning_results['study']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Optimization history
    trials = study.trials
    best_values = [min([t.value for t in trials[:i+1]]) for i in range(len(trials))]

    axes[0].plot(best_values, linewidth=2)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Best MAE')
    axes[0].set_title('XGBoost Optimization Progress')
    axes[0].grid(True, alpha=0.3)

    # Parameter importance
    try:
        param_importances = optuna.importance.get_param_importances(study)
        params = list(param_importances.keys())
        importances = list(param_importances.values())

        axes[1].barh(params, importances, alpha=0.7)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('XGBoost Parameter Importance')
        axes[1].invert_yaxis()
    except:
        axes[1].text(0.5, 0.5, 'Parameter importance\nnot available',
                    ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('results/figures/xgboost_optimization.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/xgboost_optimization.png")
    plt.close()

# 5. Best model predictions
best_model_name = comparison.iloc[0]['Model']
best_result = all_results[best_model_name]
y_pred = best_result['model'].predict(X_val)

plt.figure(figsize=(10, 10))
scatter = plt.scatter(y_pred, y_val, c=X_val['GridPosition'],
                     cmap='RdYlGn_r', alpha=0.6, s=50)
plt.plot([0, 20], [0, 20], 'k--', lw=2, label='Perfect prediction')
plt.plot([0, 20], [1, 21], 'r--', lw=1, alpha=0.5, label='±1 position')
plt.plot([0, 20], [-1, 19], 'r--', lw=1, alpha=0.5)
plt.plot([0, 20], [3, 23], 'orange', linestyle='--', lw=1, alpha=0.5, label='±3 positions')
plt.plot([0, 20], [-3, 17], 'orange', linestyle='--', lw=1, alpha=0.5)
plt.colorbar(scatter, label='Grid Position')
plt.xlabel('Predicted Finish Position')
plt.ylabel('Actual Finish Position')
plt.title(f'{best_model_name}: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.savefig('results/figures/best_model_predictions.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/best_model_predictions.png")
plt.close()

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save all model objects
models_to_save = {
    'xgboost_default': xgb_default_results['model'],
    'xgboost_tuned': xgb_tuned_results['model'],
    'lightgbm_default': lgb_default_results['model'],
    'lightgbm_tuned': lgb_tuned_results['model'],
    'catboost_default': cb_default_results['model'],
    'catboost_tuned': cb_tuned_results['model']
}

with open('results/models/advanced_models.pkl', 'wb') as f:
    pickle.dump(models_to_save, f)
print("Models saved to: results/models/advanced_models.pkl")

# Save best parameters
best_params = {
    'xgboost': xgb_tuned_results['params'],
    'lightgbm': lgb_tuned_results['params'],
    'catboost': cb_tuned_results['params']
}

with open('results/models/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("Best parameters saved to: results/models/best_params.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

best_row = comparison.iloc[0]
baseline_mae = 4.254  # Grid position baseline from previous analysis

print(f"\nBest Model: {best_row['Model']}")
print(f"  Validation MAE: {best_row['Val MAE']:.3f} positions")
print(f"  Validation RMSE: {best_row['Val RMSE']:.3f}")
print(f"  R² Score: {best_row['Val R²']:.3f}")
print(f"  Training Time: {best_row['Train Time (s)']:.2f}s")

print(f"\nImprovement over baseline:")
print(f"  Grid baseline MAE: {baseline_mae:.3f}")
print(f"  Improvement: {((baseline_mae - best_row['Val MAE']) / baseline_mae * 100):.1f}%")

# Calculate prediction accuracy
error_df = pd.DataFrame({
    'actual': y_val.values,
    'predicted': y_pred,
    'abs_error': np.abs(y_val.values - y_pred)
})

within_1 = (error_df['abs_error'] <= 1).mean() * 100
within_3 = (error_df['abs_error'] <= 3).mean() * 100
within_5 = (error_df['abs_error'] <= 5).mean() * 100

print(f"\nPrediction Accuracy ({best_row['Model']}):")
print(f"  Within ±1 position: {within_1:.1f}%")
print(f"  Within ±3 positions: {within_3:.1f}%")
print(f"  Within ±5 positions: {within_5:.1f}%")

print(f"\nTop 5 Most Important Features:")
for idx, row in best_result['feature_importance'].head(5).iterrows():
    print(f"  {row['feature']:<35} {row['importance']:>8.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nAll results saved to results/models/")
print("All figures saved to results/figures/")
