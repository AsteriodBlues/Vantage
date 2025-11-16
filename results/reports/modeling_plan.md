# Machine Learning Modeling Plan

## Overview

Based on comprehensive EDA of F1 race data (2018-2024), this document outlines the complete machine learning strategy for predicting race outcomes from grid positions.

---

## 1. Problem Definition

### Primary Task: Finish Position Prediction

**Type:** Regression
**Target Variable:** `Position` (finish position, 1-20)
**Input Features:** Grid position, circuit characteristics, team performance, temporal features

**Success Criteria:**
- MAE < 2.5 positions (better than naive baseline)
- R² > 0.70 (explain 70%+ of variance)
- Consistent performance across circuits and teams

### Secondary Tasks

1. **Win Probability Prediction**
   - Type: Binary classification
   - Target: Position == 1 (yes/no)
   - Evaluation: ROC-AUC, precision-recall

2. **Podium Probability Prediction**
   - Type: Binary classification
   - Target: Position <= 3 (yes/no)
   - Evaluation: ROC-AUC, F1-score

3. **Points Probability Prediction**
   - Type: Binary classification
   - Target: Position <= 10 (yes/no)
   - Evaluation: Accuracy, balanced accuracy

4. **Position Change Magnitude**
   - Type: Regression
   - Target: `position_change` (GridPosition - Position)
   - Evaluation: MAE, RMSE

---

## 2. Feature Engineering Plan

### 2.1 Grid Position Features

**Raw Features:**
- `GridPosition` (1-20)

**Engineered Features:**
- `grid_squared`: GridPosition² (capture non-linear advantage)
- `grid_cubed`: GridPosition³ (diminishing returns at back)
- `grid_log`: log(GridPosition) (alternative non-linear transform)
- `front_row`: Binary (GridPosition <= 2)
- `top_three`: Binary (GridPosition <= 3)
- `top_ten`: Binary (GridPosition <= 10)
- `grid_side`: Categorical ('clean' for odd, 'dirty' for even)

**Rationale:** EDA shows non-linear relationship between grid and finish. Polynomial terms capture this effectively.

### 2.2 Circuit Features

**From EDA Analysis:**
- `overtaking_difficulty_index`: 0-100 scale (calculated from circuit stats)
- `circuit_pole_win_rate`: Historical pole-to-win % at circuit
- `circuit_avg_pos_change`: Mean position change at circuit
- `circuit_pos_change_variance`: Variance of position changes
- `circuit_dnf_rate`: Historical DNF rate
- `circuit_correlation`: Grid-finish correlation coefficient

**Circuit Type Encoding:**
- `circuit_type`: Categorical (street/permanent/high_speed)
- One-hot encode or use target encoding

**Rationale:** Circuit characteristics explain 10-15% additional variance beyond grid position.

### 2.3 Team Features

**Current Performance:**
- `team_avg_grid`: Season average grid position
- `team_avg_finish`: Season average finish position
- `team_performance_delta`: Grid - Finish (over/under-performing)
- `team_dnf_rate`: Season DNF percentage

**Rolling Performance (Momentum):**
- `team_last3_avg_finish`: Rolling 3-race average finish
- `team_last5_avg_finish`: Rolling 5-race average finish
- `team_form_trend`: Slope of last 5 finishes (improving/declining)

**Historical:**
- `team_circuit_history`: Average finish at specific circuit
- `team_circuit_wins`: Number of wins at circuit

**Rationale:** Team performance explains 5-10% variance. Rolling features capture momentum and form.

### 2.4 Driver Features (If Available)

- `driver_experience`: Total races completed
- `driver_circuit_experience`: Races at specific circuit
- `driver_avg_finish`: Career average finish
- `driver_last5_avg`: Recent form
- `driver_circuit_best`: Best finish at circuit

### 2.5 Temporal Features

**Season Context:**
- `race_number_in_season`: 1-24 (position in calendar)
- `season_progress`: Normalized 0-1 (early/mid/late season)

**Era Indicators:**
- `post_2022`: Binary (regulation change)
- `year`: Numerical (2018-2024)

**Recent Results:**
- `last_race_finish`: Previous race result
- `last_race_pos_change`: Previous race position change

### 2.6 Interaction Features

**Key Interactions (from EDA):**
- `grid_x_circuit_difficulty`: GridPosition × Overtaking Index
- `grid_x_team_delta`: GridPosition × Team Performance Delta
- `top3_x_circuit_type`: Top 3 grid × Circuit Type
- `team_x_circuit`: Team × Circuit (track-specific advantages)

**Rationale:** ANOVA suggested significant interaction effects. Tree models handle these automatically, but explicit features may help.

### 2.7 Feature Summary

| Feature Category | Count | Example Features |
|-----------------|-------|------------------|
| Grid Position | 7 | grid, grid², grid³, grid_log, front_row, top_three, grid_side |
| Circuit | 8 | overtaking_index, pole_win_rate, circuit_type, dnf_rate |
| Team | 10 | team_delta, team_last3_avg, team_dnf_rate, team_circuit_history |
| Driver | 5 | driver_experience, driver_circuit_exp, driver_last5_avg |
| Temporal | 4 | race_number, season_progress, post_2022, year |
| Interactions | 4 | grid×circuit, grid×team, team×circuit, top3×type |
| **Total** | **38** | Comprehensive feature set |

---

## 3. Model Selection & Rationale

### 3.1 Recommended Models

#### Primary: XGBoost Regressor

**Justification:**
- Handles non-linear relationships effectively (grid position effects)
- Automatically captures feature interactions
- Robust to outliers (DNFs, exceptional performances)
- Excellent performance on tabular data
- Built-in regularization prevents overfitting
- Feature importance for interpretability

**Expected Performance:** MAE 2.0-2.5, R² 0.72-0.78

**Hyperparameters to Tune:**
- `max_depth`: 4-8 (control complexity)
- `learning_rate`: 0.01-0.1 (slower = better generalization)
- `n_estimators`: 100-1000 (early stopping)
- `subsample`: 0.7-1.0 (row sampling)
- `colsample_bytree`: 0.7-1.0 (column sampling)
- `min_child_weight`: 1-5 (regularization)
- `gamma`: 0-5 (pruning threshold)

#### Alternative: LightGBM

**Justification:**
- Faster training than XGBoost
- Handles categorical features natively
- Lower memory usage
- Similar performance to XGBoost

**Use Case:** If training time becomes bottleneck

#### Baseline: Random Forest

**Justification:**
- Simple, interpretable baseline
- Robust and stable
- Good for comparison

**Expected Performance:** MAE 2.3-2.8, R² 0.68-0.74

#### Not Recommended: Linear Regression

**Why not:**
- EDA shows clear non-linearity (pole advantage, diminishing returns)
- Requires extensive manual feature engineering for interactions
- Expected performance: MAE 3.0+, R² 0.55-0.60 (insufficient)

### 3.2 Model Ensemble Strategy

**Stacking Approach:**

Level 0 (Base Models):
1. XGBoost (primary)
2. LightGBM (fast alternative)
3. Random Forest (diversity)
4. Gradient Boosting (sklearn)

Level 1 (Meta-Model):
- Ridge Regression or Light XGBoost
- Combines predictions from base models
- Expected improvement: +2-3% R²

**Voting Ensemble:**
- Simple average of top 3 models
- Weighted by validation performance
- More robust, less prone to overfitting

---

## 4. Data Preprocessing

### 4.1 Train/Validation/Test Split

**Time-Based Split (Recommended):**
- **Training:** 2018-2022 (5 years, ~75% data)
- **Validation:** 2023 (1 year, ~12% data)
- **Test:** 2024 (1 year, ~13% data)

**Rationale:**
- Respects temporal nature of data
- Prevents data leakage
- Tests on most recent regulations
- Validates on near-term predictions

**Alternative: K-Fold Time Series CV:**
- 5 folds with temporal ordering preserved
- Each fold: train on past, test on future
- More robust performance estimates

### 4.2 Feature Scaling

**Required for:**
- Distance-based models (if used)
- Neural networks (if explored)

**Not required for:**
- Tree-based models (XGBoost, RF, LightGBM)

**Method (if needed):**
- StandardScaler for most features
- MinMaxScaler for bounded features (0-1)
- RobustScaler if outliers present

### 4.3 Categorical Encoding

**One-Hot Encoding:**
- `circuit_type` (3 categories)
- `grid_side` (2 categories)
- Low cardinality, safe for tree models

**Target Encoding:**
- `TeamName` (10-12 categories)
- `circuit` (20+ categories)
- Higher cardinality, risk of overfitting
- Use cross-validated target encoding

**Label Encoding:**
- For tree models only (XGBoost can handle)
- Ordinal features if any

### 4.4 Handling Missing Data

**Current Status:** Minimal missing data after cleaning

**Strategy:**
- Grid position: Never missing (required field)
- Circuit features: Impute with median/mode
- Team rolling features: Forward fill (first races have no history)
- Driver features: Impute with 0 or create "missing" flag

### 4.5 Outlier Handling

**Approach:** Keep outliers for tree models
- Tree models are robust to outliers
- Outliers (exceptional wins) are valid data points
- May cap extreme values for linear models (if used)

---

## 5. Expected Challenges & Solutions

### Challenge 1: Class Imbalance (Win Prediction)

**Problem:** Only ~5% of races are wins (1st place)

**Solutions:**
1. **Class Weighting:** Use `scale_pos_weight` in XGBoost
2. **SMOTE:** Synthetic oversampling of minority class
3. **Threshold Tuning:** Adjust decision threshold based on validation
4. **Different Metric:** Use ROC-AUC instead of accuracy
5. **Stratified Sampling:** Ensure balanced folds in CV

**Expected Impact:** Improve recall for wins from 40% → 65%

### Challenge 2: Temporal Concept Drift

**Problem:** Team performance changes between seasons (e.g., Mercedes decline, Red Bull rise)

**Solutions:**
1. **Recent Data Weighting:** Give higher weight to 2022-2024 data
2. **Rolling Features:** Use last-N-races instead of season averages
3. **Separate Models:** Train circuit-specific or year-specific models
4. **Online Learning:** Update model as new data arrives
5. **Monitoring:** Track performance degradation over time

**Expected Impact:** Maintain R² within 5% across seasons

### Challenge 3: Limited Circuit Data

**Problem:** Only 2-7 races per circuit (small sample)

**Solutions:**
1. **Circuit Grouping:** Cluster similar circuits (street/permanent/high-speed)
2. **Meta-Features:** Use circuit characteristics instead of one-hot encoding
3. **Hierarchical Models:** Global model + circuit-specific adjustments
4. **Regularization:** Prevent overfitting to specific circuits
5. **Leave-One-Circuit-Out CV:** Validate generalization

**Expected Impact:** Robust predictions for unseen circuits

### Challenge 4: DNF Handling

**Problem:** DNFs have undefined finish position (or assigned to last)

**Solutions:**
1. **Separate DNF Model:** Binary classifier (DNF yes/no) → regression if finish
2. **Multi-Task Learning:** Joint prediction of (DNF, finish_position)
3. **Censored Regression:** Treat DNFs as censored data (survival analysis)
4. **Exclude from Training:** Train only on finishers, predict DNF separately
5. **DNF Probability Feature:** Add predicted DNF risk as feature

**Recommended:** Option 1 (two-stage model)

**Expected Impact:** Improve MAE by 0.2-0.3 positions

### Challenge 5: Feature Leakage

**Problem:** Risk of using future information in features

**Prevention:**
1. **Strict Temporal Split:** Never use future data
2. **Rolling Windows:** Only use past N races
3. **No Global Stats:** Avoid season-wide aggregates (use cumulative)
4. **Validation Checks:** Monitor train vs test performance gap

---

## 6. Evaluation Metrics

### Regression (Finish Position)

**Primary Metric:**
- **MAE (Mean Absolute Error):** Average positions off
  - Target: < 2.5 positions
  - Interpretable: "On average, predictions are off by X positions"

**Secondary Metrics:**
- **RMSE (Root Mean Squared Error):** Penalizes large errors
  - Target: < 3.5 positions
- **R² Score:** Variance explained
  - Target: > 0.70 (70% variance explained)
- **Median Absolute Error:** Robust to outliers
  - Target: < 2.0 positions
- **Within-1-Position Accuracy:** % predictions within ±1 position
  - Target: > 40%
- **Within-3-Positions Accuracy:** % predictions within ±3 positions
  - Target: > 75%

### Classification (Win/Podium/Points)

**Primary Metrics:**
- **ROC-AUC:** Area under ROC curve (class imbalance robust)
  - Target: > 0.85 for wins, > 0.80 for podiums
- **Precision-Recall AUC:** Better for imbalanced classes
  - Target: > 0.70 for wins

**Secondary Metrics:**
- **F1-Score:** Balance of precision and recall
- **Precision:** When we predict win, how often correct?
- **Recall:** What % of actual wins do we predict?
- **Balanced Accuracy:** Accounts for class imbalance

### Per-Segment Analysis

Evaluate separately for:
- **By Grid Position:** P1-P3, P4-P10, P11-P20
- **By Circuit Type:** Street, permanent, high-speed
- **By Team Tier:** Top, midfield, backmarkers
- **By Season:** 2022 vs 2023 vs 2024 (test drift)

---

## 7. Implementation Timeline

### Phase 1: Feature Engineering (Days 8-9)
- Implement all engineered features
- Create feature pipeline
- Validate feature calculations
- Save processed dataset

### Phase 2: Baseline Models (Day 10)
- Linear regression (baseline)
- Random forest (strong baseline)
- Document baseline performance

### Phase 3: XGBoost Development (Days 11-12)
- Hyperparameter tuning (grid search / Bayesian optimization)
- Feature importance analysis
- Error analysis by segment

### Phase 4: Model Refinement (Day 13)
- LightGBM comparison
- Ensemble methods (stacking/voting)
- Two-stage DNF handling

### Phase 5: Evaluation (Day 14-15)
- Comprehensive test set evaluation
- Per-segment performance analysis
- Feature importance interpretation (SHAP values)
- Error case studies

### Phase 6: Secondary Tasks (Day 16)
- Win probability classifier
- Podium probability classifier
- Points probability classifier

### Phase 7: Documentation & Deployment Prep (Days 17-18)
- Model versioning and serialization
- Prediction pipeline code
- Performance reports
- API design (if applicable)

---

## 8. Success Criteria

### Minimum Viable Model
- MAE < 3.0 positions (better than random guessing)
- R² > 0.60 (explains majority of variance)
- Generalizes to test set (train-test gap < 10%)

### Target Performance
- MAE < 2.5 positions
- R² > 0.70
- Within-3-positions accuracy > 75%
- Consistent across circuits and teams

### Stretch Goals
- MAE < 2.0 positions
- R² > 0.75
- Win prediction ROC-AUC > 0.90
- Deployed production model

---

## 9. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting to training data | Medium | High | Cross-validation, regularization, simple models |
| Concept drift in test period | Medium | Medium | Recent data weighting, monitoring, model updates |
| Poor generalization to circuits | Low | Medium | Circuit features, leave-one-out CV, grouping |
| Feature leakage | Low | High | Strict temporal splits, careful feature review |
| Class imbalance hurts classifiers | High | Medium | SMOTE, class weights, threshold tuning |
| Insufficient data for complex model | Low | Medium | Regularization, simpler models, feature selection |

---

## 10. Next Actions

1. ✅ Complete EDA and statistical analysis
2. ✅ Document modeling plan
3. ⏳ Implement feature engineering pipeline
4. ⏳ Create train/validation/test splits
5. ⏳ Train baseline models
6. ⏳ Develop XGBoost model
7. ⏳ Perform hyperparameter tuning
8. ⏳ Evaluate and interpret results
9. ⏳ Build deployment pipeline

---

*Document Version: 1.0*
*Last Updated: [Date]*
*Next Review: After baseline model training*
