# Baseline Models - Summary Report

## Completed Tasks

### 1. Modeling Framework Setup ✓
- Created `src/models.py` with reusable modeling functions
- Implemented data loading with automatic feature filtering
- Fixed critical data leakage (removed `position_change` feature)
- Added comprehensive evaluation metrics (MAE, RMSE, R², percentiles)

### 2. Dummy Baselines ✓
**Grid Position Baseline**: 4.254 MAE
- Simply predicts drivers finish where they started
- On average, drivers finish 4.3 positions from their starting grid position

**Mean/Median Baselines**: ~4.95 MAE
- Always predict the average/median finish position
- Slightly worse than grid position baseline

### 3. Historical Mean Baseline ✓
**Grid Lookup Table**: 3.900 MAE
- Maps each grid position to historical average finish
- Example: P1 → 3.89, P5 → 7.56, P10 → 11.22
- 8.3% improvement over grid baseline

### 4. Linear Regression ✓
**Performance**: 0.906 MAE, R² = 0.955
- Training: 0.433 MAE (very low overfitting)
- Validation: 0.906 MAE
- Training time: 0.002s

**Top Features**:
1. TeamName_freq (-5.56)
2. frontrow_x_correlation (1.25)
3. front_row (-1.17)
4. pole_at_processional (-0.91)
5. driver_vs_car_potential (-0.90)

**Diagnostics**:
- Residuals show some non-linearity
- Q-Q plot suggests slight deviation from normality
- Predictions reasonable but not optimal

### 5. Ridge Regression ✓
**Performance**: 0.828 MAE, R² = 0.968
- Training: 0.481 MAE
- Validation: 0.828 MAE
- Best alpha: 100.0 (strong regularization)
- Training time: 1.28s

**Top Features** (after regularization):
1. driver_vs_car_potential (-0.87)
2. team_last5_avg_finish (0.33)
3. team_last3_avg_finish (0.24)
4. TeamName_target_enc (0.24)
5. team_momentum (0.20)

**Insights**:
- Regularization improved validation performance
- Coefficients more stable than linear regression
- 8.6% better than linear regression

### 6. Random Forest Baseline ✓
**Performance**: 0.770 MAE, R² = 0.974 (BEST MODEL)
- Training: 0.361 MAE
- Validation: 0.770 MAE
- Training time: 0.15s
- **81.9% improvement over grid baseline**

**Feature Importance**:
1. driver_vs_car_potential (60.79%) - Dominates predictions!
2. TeamName_target_enc (22.87%)
3. team_avg_finish (1.59%)
4. team_last5_avg_finish (1.51%)
5. driver_vs_teammate (1.34%)

**Key Finding**: Only 7 features needed for 90% importance

**Prediction Accuracy**:
- Within ±1 position: 69.8%
- Within ±3 positions: 100%
- Within ±5 positions: 100%

### 7. Comprehensive Analysis & Visualizations ✓

Generated 6 publication-ready figures:
1. **model_comparison.png**: MAE, R², efficiency comparison
2. **feature_importance.png**: Ridge vs RF top features
3. **predictions_vs_actual.png**: Scatter with grid position coloring
4. **error_by_grid.png**: Error varies by starting position
5. **linear_diagnostics.png**: Residual analysis
6. **ridge_tuning.png**: Hyperparameter CV curve

## Key Insights

### What Works:
1. **driver_vs_car_potential is king** - Accounts for 60% of RF importance
2. **Team performance metrics crucial** - Last 3-5 races highly predictive
3. **Grid position matters** - But context (team, driver) matters more
4. **Tree models dominate** - RF beats linear by 15% in MAE

### What Doesn't Work:
1. **Simple baselines** - Grid position alone gives 4.25 MAE
2. **Pure statistics** - Mean/median worse than grid baseline
3. **Linear assumptions** - Relationships are non-linear

### Error Patterns:
- Errors higher for midfield positions (8-15)
- Front row predictions very accurate (pole advantage clear)
- Back markers more predictable than expected

## Model Comparison

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|----|-----------|
| Grid Baseline | 4.254 | - | - | - |
| Historical Mean | 3.900 | - | - | - |
| Linear Regression | 0.906 | 1.216 | 0.955 | 0.002s |
| Ridge Regression | 0.828 | 1.030 | 0.968 | 1.28s |
| **Random Forest** | **0.770** | **0.928** | **0.974** | **0.15s** |

## Files Created

### Code:
- `src/models.py` - Modeling framework (500+ lines)
- `notebooks/baseline_models.ipynb` - Interactive analysis
- `test_baseline_models.py` - Quick verification script
- `run_baseline_analysis.py` - Full analysis pipeline

### Results:
- `results/models/baseline_models.pkl` - Trained models
- `results/models/baseline_results.csv` - Performance table
- `results/figures/*.png` - 6 visualization files

## Next Steps (Day 12)

The baseline Random Forest achieves 0.77 MAE with default parameters. Tomorrow's goals:

1. **XGBoost** - Gradient boosting should improve further
2. **LightGBM** - Faster training for larger experiments
3. **CatBoost** - Better categorical handling
4. **Hyperparameter Tuning** - Optuna for Bayesian optimization
5. **Target**: Sub-0.70 MAE (10% improvement)

## Commits

1. `e665458` - Add baseline modeling framework and notebook
2. `16798e5` - Fix data leakage in baseline models
3. `788b05b` - Add comprehensive baseline analysis with visualizations

All changes pushed to `main` branch.

---

**Summary**: Day 11 successfully established strong baseline performance with Random Forest achieving 0.77 MAE (81.9% improvement over simple baselines). The modeling framework is production-ready and the feature importance analysis provides clear direction for future improvements.
