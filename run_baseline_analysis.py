"""
Run comprehensive baseline model analysis with visualizations.
Saves all results and figures to results/ directory.
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
from scipy import stats

from src.models import (
    set_random_seeds,
    load_modeling_data,
    verify_data_integrity,
    train_dummy_baseline,
    train_mean_baseline,
    train_linear_regression,
    train_ridge_regression,
    train_random_forest,
    create_results_summary,
)

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
set_random_seeds(42)

# Create output directories
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*80)
print("F1 RACE PREDICTION - BASELINE MODELS ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_modeling_data()
verify_data_integrity(X_train, X_val, X_test, y_train, y_val, y_test)

# Train all models
print("\n" + "="*80)
print("TRAINING BASELINE MODELS")
print("="*80)

dummy_results = train_dummy_baseline(X_train, y_train, X_val, y_val)
mean_results = train_mean_baseline(X_train, y_train, X_val, y_val)
lr_results = train_linear_regression(X_train, y_train, X_val, y_val)
ridge_results = train_ridge_regression(X_train, y_train, X_val, y_val)
rf_results = train_random_forest(X_train, y_train, X_val, y_val)

# Combine results
all_results = {
    'Linear Regression': lr_results,
    'Ridge Regression': ridge_results,
    'Random Forest': rf_results
}

# Create summary
summary = create_results_summary(all_results)
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(summary.to_string(index=False))
print("="*80)

# Save summary
summary.to_csv('results/models/baseline_results.csv', index=False)
print("\nResults saved to: results/models/baseline_results.csv")

# Save model objects
with open('results/models/baseline_models.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("Models saved to: results/models/baseline_models.pkl")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].barh(summary['Model'], summary['MAE'], alpha=0.7, color='steelblue')
axes[0].set_xlabel('MAE (positions)')
axes[0].set_title('Model Comparison: MAE')
axes[0].invert_yaxis()

axes[1].barh(summary['Model'], summary['R²'], alpha=0.7, color='forestgreen')
axes[1].set_xlabel('R² Score')
axes[1].set_title('Model Comparison: R²')
axes[1].invert_yaxis()

axes[2].scatter(summary['Train Time (s)'], summary['MAE'], s=150, alpha=0.6)
for idx, row in summary.iterrows():
    axes[2].annotate(row['Model'], (row['Train Time (s)'], row['MAE']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[2].set_xlabel('Training Time (seconds)')
axes[2].set_ylabel('MAE (positions)')
axes[2].set_title('Efficiency: Training Time vs Performance')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/model_comparison.png")
plt.close()

# 2. Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Ridge coefficients
top_ridge = ridge_results['feature_importance'].head(20)
axes[0].barh(range(len(top_ridge)), top_ridge['coefficient'], alpha=0.7,
            color=['red' if c < 0 else 'green' for c in top_ridge['coefficient']])
axes[0].set_yticks(range(len(top_ridge)))
axes[0].set_yticklabels(top_ridge['feature'])
axes[0].set_xlabel('Coefficient Value')
axes[0].set_title('Ridge Regression: Top 20 Features')
axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].invert_yaxis()

# Random Forest importance
top_rf = rf_results['feature_importance'].head(20)
axes[1].barh(range(len(top_rf)), top_rf['importance'], alpha=0.7, color='forestgreen')
axes[1].set_yticks(range(len(top_rf)))
axes[1].set_yticklabels(top_rf['feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Random Forest: Top 20 Features')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/feature_importance.png")
plt.close()

# 3. Predictions vs Actual for best model
best_model = rf_results['model']
y_pred = best_model.predict(X_val)

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
plt.title('Random Forest: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.savefig('results/figures/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/predictions_vs_actual.png")
plt.close()

# 4. Error analysis
error_df = pd.DataFrame({
    'actual': y_val.values,
    'predicted': y_pred,
    'error': y_val.values - y_pred,
    'abs_error': np.abs(y_val.values - y_pred),
    'grid_position': X_val['GridPosition'].values
})

error_by_grid = error_df.groupby('grid_position')['abs_error'].mean().sort_index()

plt.figure(figsize=(12, 6))
plt.bar(error_by_grid.index, error_by_grid.values, alpha=0.7, color='coral')
plt.xlabel('Grid Position')
plt.ylabel('Average Absolute Error')
plt.title('Prediction Error by Grid Position')
plt.xticks(range(1, 21))
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('results/figures/error_by_grid.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/error_by_grid.png")
plt.close()

# 5. Linear regression diagnostics
y_pred_lr = lr_results['model'].predict(X_val)
residuals = y_val - y_pred_lr

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_pred_lr, y_val, alpha=0.5)
axes[0, 0].plot([0, 20], [0, 20], 'r--', lw=2)
axes[0, 0].set_xlabel('Predicted Position')
axes[0, 0].set_ylabel('Actual Position')
axes[0, 0].set_title('Linear Regression: Predicted vs Actual')

axes[0, 1].scatter(y_pred_lr, residuals, alpha=0.5)
axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Position')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Predicted')

axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residual')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution')

stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('results/figures/linear_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/linear_diagnostics.png")
plt.close()

# 6. Ridge hyperparameter tuning
cv_results = ridge_results['cv_results']
alphas = cv_results['param_alpha'].values
mean_scores = -cv_results['mean_test_score'].values

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, mean_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Cross-Validation MAE')
plt.title('Ridge Regression: Hyperparameter Tuning')
plt.grid(True, alpha=0.3)
best_alpha = ridge_results['best_params']['alpha']
plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best: α={best_alpha}')
plt.legend()
plt.savefig('results/figures/ridge_tuning.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/ridge_tuning.png")
plt.close()

# ============================================================================
# FINAL STATISTICS
# ============================================================================

print("\n" + "="*80)
print("FINAL STATISTICS")
print("="*80)

grid_mae = dummy_results.get('grid_baseline', {}).get('mae', 0)
best_mae = summary['MAE'].min()

within_1 = (error_df['abs_error'] <= 1).mean() * 100
within_3 = (error_df['abs_error'] <= 3).mean() * 100
within_5 = (error_df['abs_error'] <= 5).mean() * 100

print(f"\nBaseline Comparisons:")
print(f"  Grid Position Baseline: {grid_mae:.3f} MAE")
print(f"  Historical Mean Baseline: {mean_results['grid_mean']['mae']:.3f} MAE")
print(f"  Best Model (Random Forest): {best_mae:.3f} MAE")
print(f"  Improvement over grid: {((grid_mae - best_mae) / grid_mae * 100):.1f}%")

print(f"\nRandom Forest Prediction Accuracy:")
print(f"  Within 1 position: {within_1:.1f}%")
print(f"  Within 3 positions: {within_3:.1f}%")
print(f"  Within 5 positions: {within_5:.1f}%")

print(f"\nTop 5 Most Important Features (Random Forest):")
for idx, row in rf_results['feature_importance'].head(5).iterrows():
    print(f"  {row['feature']:<35} {row['importance']:>8.4f}")

print("\n" + "="*80)
print("BASELINE ANALYSIS COMPLETE")
print("="*80)
print("\nAll results saved to results/models/")
print("All figures saved to results/figures/")
