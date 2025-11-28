# VANTAGE F1

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![FastF1](https://img.shields.io/badge/FastF1-E10600?style=flat&logo=formula1&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=microsoft&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-4051B5?style=flat&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-7db0bc?style=flat&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**V**aluating **A**dvantage **N**umerically **T**hrough **A**nalysis of **G**rid **E**ffects

A machine learning system that predicts Formula 1 race finishing positions based on qualifying results, historical performance data, and circuit characteristics.

## Overview

Starting grid position is one of the strongest predictors of race outcome in Formula 1, but its impact varies significantly across different circuits and conditions. VANTAGE quantifies these relationships using 7 years of race data (2018-2024) to forecast race results with high accuracy.

The system achieves a mean absolute error of **0.57 positions** on test data, meaning predictions are typically within one position of the actual finishing order. This level of precision is achieved through comprehensive feature engineering that captures team performance trends, driver experience, circuit-specific characteristics, and temporal factors like regulation changes.

## Key Results

- **Test MAE**: 0.57 positions (predicts within ~1 position on average)
- **Test R²**: 0.971 (explains 97% of variance in finishing positions)
- **Training Set**: 357 races from 2018-2022
- **Test Set**: 360 races from 2023-2024
- **Feature Count**: 136 engineered features
- **Circuits Analyzed**: 18 distinct tracks

## Installation

```bash
git clone https://github.com/yourusername/Vantage.git
cd Vantage
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Interactive Dashboard

The Streamlit dashboard provides a web interface for predictions and analysis:

```bash
./run_dashboard.sh
# Opens at http://localhost:8501
```

### Command Line Interface

```bash
# Predict a single driver's finish
python src/predict_cli.py single \
    --driver "Max Verstappen" \
    --team "Red Bull" \
    --circuit "Monaco" \
    --grid 1

# Simulate an entire race from a grid configuration
python src/predict_cli.py grid \
    --grid-file examples/example_grid.json

# Interactive prediction mode
python src/predict_cli.py interactive
```

### Python API

```python
from src.prediction_pipeline import F1PredictionPipeline

pipeline = F1PredictionPipeline()

result = pipeline.predict(
    grid_position=3,
    circuit_name="Silverstone",
    team="Ferrari",
    driver="Charles Leclerc",
    year=2024,
    race_number=10
)

print(f"Predicted finish: P{result['predicted_finish_rounded']}")
print(f"Win probability: {result['probabilities']['win']:.1%}")
```

## Project Structure

```
Vantage/
├── app/                          # Streamlit web dashboard
│   └── dashboard.py
├── src/                          # Core source code
│   ├── prediction_pipeline.py   # Main prediction interface
│   ├── predict_cli.py           # Command-line tool
│   └── model_deployment.py      # Model loading utilities
├── models/                       # Trained models & preprocessors
│   ├── production/              # Versioned production models
│   └── preprocessing/           # Feature encoders & statistics
├── data/                         # Raw and processed datasets
├── notebooks/                    # Jupyter analysis notebooks
├── results/                      # Generated figures & reports
├── docs/                         # Documentation
└── examples/                     # Sample input files
```

## How It Works

### Data Collection

Race data is sourced from the FastF1 API, which provides official Formula 1 timing and telemetry information. The dataset includes:
- Qualifying and race results for all sessions from 2018-2024
- Driver and team information
- Circuit layouts and characteristics
- DNF (Did Not Finish) classifications

### Feature Engineering

The model uses 136 features organized into several categories:

**Grid Position Features**: Raw starting position plus transformations (squared, cubed, log, square root) and categorical indicators (pole, front row, top 5, etc.)

**Circuit Characteristics**: Historical statistics for each track including pole win rate, average position changes, overtaking difficulty, DNF rates, and physical track properties (length, corners, altitude, longest straight).

**Team Performance**: Rolling averages of recent finishes, points scored, wins and podiums this season, momentum metrics, reliability statistics, and performance relative to grid position.

**Driver Metrics**: Career experience, races at specific circuit, historical performance at the track, recent form trends, and comparison to teammate.

**Temporal Context**: Race number, season progress, regulation era (pre/post 2022 changes), championship standings phase.

**Interaction Terms**: Combined features like grid position × overtaking difficulty, team form × championship pressure, and veteran driver on new circuit.

### Model Architecture

The production model is a Random Forest Regressor with 100 trees, trained on 2018-2022 data and validated on 2023-2024 seasons. The random forest approach was selected for its:
- Strong performance on tabular data with mixed feature types
- Inherent handling of non-linear relationships
- Resistance to overfitting through ensemble averaging
- Fast inference time (<50ms per prediction)
- Clear feature importance interpretability

### Circuit Analysis

One interesting finding from this project is the significant variation in grid position advantage across circuits. Monaco shows an 85% pole win rate, making qualifying position extremely important, while tracks like Bahrain and Spa show much lower pole win rates (45-50%) due to long straights enabling overtaking.

Circuit clustering analysis identified four distinct track types based on overtaking difficulty, position volatility, and physical characteristics. This clustering helps the model understand which historical patterns apply to similar circuits.

## Use Cases

**Race Weekend Predictions**: After qualifying, predict the likely race outcome and win/podium probabilities for each driver.

**Strategy Analysis**: Quantify the value of grid position at different circuits to inform qualifying vs race setup trade-offs.

**Historical Comparison**: Analyze how regulation changes (particularly 2022 aerodynamic rules) affected racing and position changes.

**What-If Scenarios**: Simulate hypothetical grids to understand how different qualifying outcomes might affect the race.

## Technical Stack

- **Data Processing**: pandas, numpy, FastF1
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna
- **Interpretation**: SHAP
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Streamlit
- **Development**: Jupyter notebooks

## Documentation

- [Model Performance Details](docs/model_performance.md) - Comprehensive metrics and benchmarks
- [API Specification](docs/api_specification.md) - Python API and CLI usage
- [Data Collection Strategy](docs/data_collection_strategy.md) - Data sources and processing

## Model Limitations

The model performs best under typical race conditions and has some known limitations:

- **Weather**: No weather or track temperature features included
- **Strategy**: Pit stop strategies and tire choices not modeled
- **Incidents**: First-lap collisions and safety cars create unpredictable variance
- **New Drivers**: Rookies use default statistics until history builds up
- **Track Changes**: Circuit modifications or resurfacing not immediately reflected

Despite these limitations, the model captures the fundamental dynamics of grid position advantage and produces reliable predictions for normal race scenarios.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built using race data from the FastF1 API. This project is for educational and analytical purposes only.
