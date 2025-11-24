# VANTAGE F1 Dashboard

Interactive web dashboard for F1 race predictions and analysis.

## Features

- **🏠 Home**: Project overview and key statistics
- **🔮 Predictions**: Single position and full grid race simulation
- **🏎️ Circuit Analysis**: Circuit characteristics and comparisons
- **📈 Visualizations**: Interactive charts and historical trends
- **ℹ️ About**: Project information and model details

## Running the Dashboard

### Local Development

```bash
# From project root
streamlit run app/dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Production Deployment

The dashboard can be deployed to Streamlit Cloud or any hosting service that supports Streamlit apps.

## Usage

### Making Predictions

1. Navigate to the **Predictions** tab
2. Select **Single Position** mode
3. Choose:
   - Grid position (1-20)
   - Circuit
   - Team
   - Optional: Driver name
4. Click **Predict Finish Position**

### Full Grid Simulation

1. Go to **Predictions** → **Full Grid Simulation**
2. Use the example grid or enter custom grid
3. Select circuit and race details
4. Click **Simulate Race**

### Circuit Analysis

1. Visit the **Circuit Analysis** tab
2. Select a metric to analyze
3. View circuit rankings and characteristics

## Requirements

All dependencies are listed in the project's main `requirements.txt`:

- streamlit >= 1.37.0
- pandas >= 2.2.0
- plotly >= 5.20.0
- scikit-learn >= 1.5.0
- joblib >= 1.4.0

## Model Integration

The dashboard automatically loads the production model from:
```
models/production/simple_predictor_latest/
```

If the model is not available, the dashboard runs in demo mode with simulated predictions.

## Data Sources

- Circuit statistics: `models/preprocessing/circuit_statistics.pkl`
- Historical data: `data/processed/train.csv`
- Model artifacts: `models/production/`

## Customization

### Adding New Circuits

Update the circuit lists in `dashboard.py`:
```python
circuits = [
    'Monaco', 'Monza', 'Spa-Francorchamps', ...
]
```

### Modifying Teams

Update the teams list in prediction functions:
```python
teams = [
    'Red Bull', 'Mercedes', 'Ferrari', ...
]
```

## Troubleshooting

**Model not loading:**
- Ensure models are properly trained and saved
- Check `models/production/simple_predictor_latest/` exists
- Dashboard will fall back to demo mode if model unavailable

**Data not displaying:**
- Verify preprocessing files exist in `models/preprocessing/`
- Check historical data exists in `data/processed/`

**Import errors:**
- Install all requirements: `pip install -r requirements.txt`
- Ensure `src/` module is in Python path
