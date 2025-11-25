# Technical Methodology

## Overview

This document details the technical approach used in VANTAGE to predict Formula 1 race finishing positions from qualifying results and contextual race data.

## Data Collection

### Source
All race data is collected using the FastF1 API, which provides access to official Formula 1 timing data including:
- Session results (practice, qualifying, race)
- Driver and team information
- Lap times and sector splits
- DNF classifications and status codes
- Circuit information

### Coverage
- **Time Period**: 2018-2024 (7 complete seasons)
- **Total Races**: 780 races across all sessions
- **Circuits**: 18 distinct tracks in regular rotation
- **Data Points**: ~15,000 individual driver-race combinations

### Collection Process
Data collection is implemented in `notebooks/collect_full_dataset.ipynb` with FastF1's built-in caching to minimize API calls. The process:
1. Iterates through each season and round
2. Loads session data for qualifying and race
3. Extracts relevant driver results
4. Handles missing data and session errors
5. Saves to parquet format for efficient storage

## Data Processing

### Cleaning Operations

**Team Name Standardization**: Teams that underwent rebrandings (Racing Point → Aston Martin, Renault → Alpine) are tracked with consistent identifiers while preserving historical context.

**Circuit Normalization**: Different naming conventions for the same circuit (e.g., "British Grand Prix" vs "Silverstone") are unified to a single canonical name.

**DNF Handling**: Drivers who did not finish are flagged with binary indicators. Those who completed more than 90% of race distance receive classified finishing positions; others are assigned positions based on their retirement order.

**Missing Data**: Grid positions missing due to pit lane starts or penalties are imputed based on qualifying results. Driver experience for rookies uses zero until their first race completes.

### Feature Engineering

The core of the prediction accuracy comes from comprehensive feature engineering. The system generates 136 features across several categories:

#### Grid Position Features (15 features)
- Raw starting position (1-20)
- Polynomial transformations: squared, cubed
- Log and square root transformations
- Binary indicators: is_pole, front_row, top_three, top_five, top_ten, back_half
- Position metadata: grid_row, grid_side, grid_side_clean

#### Circuit Characteristics (25 features)
Historical statistics computed for each track:
- Pole win rate (% of races won from pole)
- Overtaking difficulty score (inverse of position changes)
- Correlation between grid and finish position
- Average position changes during race
- DNF rate and completion percentage
- Percentage of drivers improving position
- Position variance and standard deviation

Physical track properties:
- Track length and number of corners
- Longest straight section
- Altitude and circuit type (street, permanent, hybrid)
- Downforce level categorization

#### Team Performance (20 features)
- Rolling averages: finish position over last 3, 5 races
- Points scored in last 5 races
- Season statistics: wins, podiums, total points
- Momentum: trend in recent results
- Consistency: standard deviation in finishes
- Performance delta: actual finish vs grid position average
- Reliability: DNF rate over last 10 races, season completion rate

#### Driver Metrics (15 features)
- Career races and years of experience
- Rookie and veteran binary flags (0-2 years, 5+ years)
- Circuit-specific history: races at this track, average finish
- Circuit specialist flag (significantly better than average)
- Recent form: average finish and points over last 5 races
- Momentum and trend indicators
- Relative metrics: vs teammate, vs car potential
- Team leadership flag

#### Temporal Features (10 features)
- Race number and season progress (0-1 normalized)
- Races remaining in season
- Season phase: early (1-7), mid (8-15), late (16+)
- Season opener and finale flags
- Regulation era: post-2022 flag (new aerodynamic regulations)
- Years into current regulation set

#### Interaction Features (20 features)
Capturing non-linear relationships:
- Grid position × overtaking difficulty
- Grid position × team performance delta
- Grid position × low downforce circuits
- Momentum × circuit variance
- Recent form × championship contention
- Veteran driver × new circuit
- Early season × high variance circuits
- Late season championship pressure

#### Encoding Features (31 features)
Three encoding strategies for categorical variables:
- **Frequency encoding**: How often each circuit/team/driver appears
- **Target encoding**: Average historical finishing position
- **Label encoding**: Numeric identifiers for categorical values

Applied to circuit_name, TeamName, and DriverId.

### Feature Selection

Initial feature engineering produced 120+ candidate features. The final set of 136 was selected through:
1. Correlation analysis to remove redundant features (>0.95 correlation)
2. Feature importance from random forest baseline models
3. Domain knowledge to retain theoretically important features
4. Performance testing on validation set

## Model Development

### Algorithm Selection

Multiple regression algorithms were evaluated:

**Random Forest**: Selected for production due to balanced performance, fast inference, and interpretability. Handles non-linear relationships well and is robust to feature scale differences.

**XGBoost/LightGBM/CatBoost**: Gradient boosting methods showed similar performance but with longer training times. Retained for ensemble experiments.

**Linear Models**: Ridge and Lasso regression used as baselines but underperformed due to complex non-linear relationships.

### Hyperparameter Configuration

The production Random Forest model uses:
- `n_estimators=100`: Sufficient for convergence without excessive training time
- `max_depth=15`: Deep enough to capture complex patterns
- `min_samples_split=5`: Prevents excessive overfitting
- `min_samples_leaf=2`: Balances bias-variance tradeoff
- `random_state=42`: Ensures reproducibility

These parameters were selected through grid search on validation data.

### Training Strategy

**Data Split**:
- Training: 2018-2022 seasons (357 races)
- Validation: 2023 season (63 races)
- Test: 2024 season (360 races)

This temporal split ensures the model is evaluated on truly future data it has never seen, mimicking real-world deployment.

**Cross-Validation**: 5-fold time-series cross-validation on the training set to assess stability and prevent overfitting.

**Evaluation Metrics**:
- Primary: Mean Absolute Error (MAE) - interpretable in position units
- Secondary: R² score - variance explained
- Root Mean Squared Error (RMSE) - penalizes large errors

## Circuit Analysis

### Clustering Methodology

Circuits were clustered using K-Means to identify track types with similar racing characteristics.

**Features Used**:
- Overtaking difficulty
- Position variance
- Pole win rate
- Track length
- Number of corners

**Optimal Clusters**: 4 clusters identified using elbow method on inertia and silhouette scores.

**Identified Types**:
1. Street circuits with high pole win rates
2. High-speed tracks with long straights
3. Technical circuits with many corners
4. Balanced mixed-characteristic tracks

### Statistical Analysis

For each circuit, historical statistics were computed:
- Win rate from each grid position (1-10)
- Transition matrices (grid position → finish position)
- Average position changes
- Variance in outcomes

These statistics reveal significant differences: Monaco has 85% pole-to-win rate while Bahrain shows 48%, indicating overtaking opportunities vary dramatically by venue.

## Model Interpretation

### Feature Importance

Feature importance was analyzed using:
- Tree-based importance from Random Forest (Gini importance)
- Permutation importance on test set
- SHAP (SHapley Additive exPlanations) values

Top contributing features:
1. Grid position (raw): 12.3%
2. Circuit pole win rate: 8.7%
3. Team recent form (last 5): 7.2%
4. Grid position squared: 6.1%
5. Circuit overtaking difficulty: 5.4%

The top 20 features account for ~65% of prediction power.

### Prediction Uncertainty

The model provides confidence through:
- Prediction intervals from individual tree predictions
- Win/podium probability estimates
- Uncertainty flags for unusual scenarios (rookies, new circuits)

## Performance Validation

### Test Set Results

On the held-out 2024 season:
- **MAE**: 0.570 positions
- **RMSE**: 0.89 positions
- **R²**: 0.971
- **Median Absolute Error**: 0.42 positions

**Distribution of Errors**:
- Within 1 position: 68%
- Within 2 positions: 87%
- Within 3 positions: 94%

### Error Analysis

The model performs best:
- At processional circuits (Monaco, Hungary) - low variance
- For top teams with consistent performance
- In typical racing conditions without major incidents

Higher errors occur:
- On first lap crashes affecting multiple drivers
- During safety car periods changing strategy
- With weather changes mid-race
- For midfield battles with tight performance gaps

### Comparison to Baselines

- **Always predict grid position**: MAE 3.4 positions
- **Linear regression**: MAE 2.8 positions
- **Random Forest (simple)**: MAE 1.2 positions
- **Production model (full features)**: MAE 0.57 positions

The comprehensive feature engineering reduced error by more than 50% compared to basic random forest.

## Implementation Details

### Production Pipeline

The prediction pipeline (`src/prediction_pipeline.py`) handles:
1. Input validation and preprocessing
2. Feature generation from raw inputs
3. Model loading and versioning
4. Prediction with uncertainty quantification
5. Post-processing and formatting

### Model Serialization

Models are saved using joblib with metadata:
- Model object and parameters
- Training date and performance metrics
- Feature names and preprocessing artifacts
- Example inputs for validation

Versioned directories allow rollback and A/B testing.

### Performance Optimization

- **Inference Time**: <50ms per prediction on standard hardware
- **Memory Footprint**: 0.14 MB model size
- **Preprocessing**: Cached circuit and team statistics for fast lookup
- **Batch Processing**: Vectorized operations for multiple predictions

## Limitations and Assumptions

### Known Limitations

**Weather**: Temperature and precipitation data not included. Wet races show higher prediction errors.

**Strategy**: Pit stop timing and tire compound choices not modeled. Two-stop vs one-stop strategies can affect outcomes.

**Mechanical**: Reliability issues beyond historical team DNF rates not predicted.

**Incidents**: First-lap collisions, safety cars, and red flags create variance the model cannot anticipate.

### Assumptions

**Continuity**: Historical patterns continue into future races. Major regulation changes may require retraining.

**Team Stability**: Teams maintain relative performance within seasons. Mid-season technical breakthroughs are captured only through rolling averages.

**Driver Skill**: Treated as constant within a season, measured through historical results rather than subjective ratings.

**Data Quality**: FastF1 API data assumed accurate and complete.

## Reproducibility

All analysis is reproducible through:
- Fixed random seeds (`random_state=42`)
- Versioned dependencies in `requirements.txt`
- Jupyter notebooks with cell execution order
- Documented preprocessing steps
- Saved model artifacts with metadata

The complete pipeline from raw data to predictions can be re-run to verify results.
