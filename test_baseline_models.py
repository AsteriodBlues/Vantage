"""Quick test to verify baseline models work correctly."""

import sys
import warnings
warnings.filterwarnings('ignore')

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

# Set seeds
set_random_seeds(42)

print("Loading data...")
X_train, X_val, X_test, y_train, y_val, y_test = load_modeling_data()

print("\nVerifying data integrity...")
verify_data_integrity(X_train, X_val, X_test, y_train, y_val, y_test)

# Train models
print("\n" + "="*60)
print("TRAINING BASELINE MODELS")
print("="*60)

print("\n1. Training dummy baselines...")
dummy_results = train_dummy_baseline(X_train, y_train, X_val, y_val)

print("\n2. Training mean baseline...")
mean_results = train_mean_baseline(X_train, y_train, X_val, y_val)

print("\n3. Training Linear Regression...")
lr_results = train_linear_regression(X_train, y_train, X_val, y_val)

print("\n4. Training Ridge Regression...")
ridge_results = train_ridge_regression(X_train, y_train, X_val, y_val)

print("\n5. Training Random Forest...")
rf_results = train_random_forest(X_train, y_train, X_val, y_val)

# Create summary
all_results = {
    'Linear Regression': lr_results,
    'Ridge Regression': ridge_results,
    'Random Forest': rf_results
}

summary = create_results_summary(all_results)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(summary.to_string(index=False))
print("="*80)

# Calculate improvements
grid_mae = dummy_results.get('grid_baseline', {}).get('mae', 0)
best_mae = summary['MAE'].min()

if grid_mae > 0:
    improvement = ((grid_mae - best_mae) / grid_mae) * 100
    print(f"\nImprovement over grid baseline: {improvement:.1f}%")
    print(f"Grid baseline MAE: {grid_mae:.3f}")
    print(f"Best model MAE: {best_mae:.3f}")

print("\n✓ All models trained successfully!")
