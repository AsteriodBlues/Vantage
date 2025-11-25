# Model Performance Summary

## Model Architecture
- **Type**: Random Forest Regressor
- **Training Data**: 2018-2024 F1 races
- **Total Features**: 136 engineered features
- **Target**: Race finish position (1-20)

## Performance Overview

### Production Model
- **Model Type**: RandomForestRegressor
- **Training Samples**: 357 races
- **Validation Samples**: 63 races
- **Test Samples**: 360 races
- **Model Size**: 0.14 MB
- **Inference Time**: <50ms per prediction

### Performance Metrics
- **Training MAE**: 0.210 positions
- **Validation MAE**: 0.342 positions
- **Test MAE**: 0.570 positions
- **Training R²**: 0.996
- **Validation R²**: 0.989
- **Test R²**: 0.971

## Feature Categories (136 total)

### Core Features
- **Grid Position**: Starting position and transformations (squared, cubed, log, sqrt)
- **Position Indicators**: front_row, top_three, top_five, top_ten, back_half
- **Grid Metadata**: grid_side, grid_side_clean, grid_row

### Temporal Features
- **Season Progress**: race_number, season_progress, races_remaining
- **Season Phase**: early_season, mid_season, late_season
- **Special Races**: is_season_opener, is_season_finale
- **Regulation Era**: post_2022, years_into_regulations

### Circuit Features
- **Track Characteristics**: pole_win_rate, overtaking_difficulty, correlation
- **Statistics**: avg_pos_change, dnf_rate, improved_pct
- **Physical**: track_length, num_turns, altitude, longest_straight
- **Type**: circuit_type, downforce_level, is_street

### Team Features
- **Recent Form**: avg_finish_last_5, avg_finish_last_3, points_last_5
- **Season Stats**: wins_season, podiums_season, points_total
- **Performance**: momentum, consistency, vs_average_grid
- **Reliability**: dnf_rate_last_10, completion_rate_season

### Driver Features
- **Experience**: career_races, years_experience, is_rookie, is_veteran
- **Track History**: races_at_circuit, avg_finish_at_circuit, is_specialist
- **Recent Form**: avg_finish_last_5, points_last_5, momentum
- **Relative**: vs_teammate, vs_car_potential, is_team_leader

### Interaction Features
- **Grid Interactions**: grid_x_overtaking, grid_x_team_delta, grid_x_low_df
- **Context**: momentum_x_variance, form_x_contention, veteran_new_circuit
- **Strategic**: early_x_variance, late_contention_pressure

### Categorical Encodings
- **Frequency**: circuit_freq, TeamName_freq, DriverId_freq
- **Target**: circuit_target_enc, TeamName_target_enc, DriverId_target_enc
- **Label**: circuit_encoded, TeamName_encoded, DriverId_encoded

## Model Strengths
- **Accurate**: Test MAE of 0.57 positions (predicts within ~1 position)
- **Fast**: Sub-50ms inference time
- **Generalizable**: Strong R² on unseen test data (0.971)
- **Comprehensive**: 136 features covering all race aspects
- **Interpretable**: Tree-based model with clear feature importance

## Usage Examples

### Command Line Interface

```bash
# Single driver prediction
python src/predict_cli.py single \
    --driver "Max Verstappen" \
    --team "Red Bull" \
    --circuit "Monaco" \
    --grid 1 \
    --race-number 7

# Full grid prediction
python src/predict_cli.py grid \
    --grid-file examples/example_grid.json

# Interactive mode
python src/predict_cli.py interactive
```

### Python API
```python
from src.prediction_pipeline import F1PredictionPipeline

# Initialize pipeline
pipeline = F1PredictionPipeline()

# Single prediction
result = pipeline.predict(
    grid_position=1,
    circuit_name="Monaco",
    team="Ferrari",
    driver="Charles Leclerc",
    year=2024,
    race_number=7
)

print(f"Predicted finish: P{result['predicted_finish_rounded']}")
print(f"Win probability: {result['probabilities']['win']:.1%}")
```

### Model Files
```
models/
├── production/
│   ├── simple_predictor_latest/ (symlink)
│   └── simple_predictor_20251123_180915/
│       ├── model.pkl (0.14 MB)
│       └── metadata.json
└── preprocessing/
    ├── feature_names.pkl
    ├── circuit_statistics.pkl
    ├── team_baselines.pkl
    └── driver_statistics.pkl
```

## Hyperparameters
- **n_estimators**: 100 trees
- **max_depth**: 15 levels
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **random_state**: 42

## Deployment Checklist
- Model serialized and versioned
- Preprocessing artifacts saved
- Metadata documented
- Example data provided
- API specification created
- CLI tool implemented
- Python prediction pipeline
- Comprehensive documentation
- Performance metrics validated

## Model Limitations
- **Data Coverage**: Limited to 2018-2024 seasons
- **Unknown Drivers**: New drivers use default statistics until building race history
- **Track Changes**: Circuit modifications or resurfacing not immediately reflected
- **Weather**: No weather condition or track temperature features
- **Strategy**: Pit stop strategies and tire choices not modeled
- **Incidents**: First-lap crashes, safety cars, and red flags create unpredictable variance
