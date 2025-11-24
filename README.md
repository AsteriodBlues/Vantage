# VANTAGE F1

**V**aluating **A**dvantage **N**umerically **T**hrough **A**nalysis of **G**rid **E**ffects

Machine learning model for predicting Formula 1 race outcomes based on starting grid positions and comprehensive race analysis.

## 🏆 Features

- **Accurate Predictions**: MAE of 0.57 positions (within ~1 position accuracy)
- **Interactive Dashboard**: Streamlit web interface for predictions and analysis
- **Command Line Tool**: CLI for batch predictions and automation
- **136 Engineered Features**: Comprehensive race data modeling
- **Circuit Intelligence**: Analysis of 18 historic F1 circuits

## 🚀 Quick Start

### Interactive Dashboard

```bash
# Launch the dashboard
./run_dashboard.sh

# Or manually
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501`

### Command Line Predictions

```bash
# Single driver prediction
python src/predict_cli.py single \
    --driver "Max Verstappen" \
    --team "Red Bull" \
    --circuit "Monaco" \
    --grid 1

# Full grid simulation
python src/predict_cli.py grid \
    --grid-file examples/example_grid.json
```

## 📊 Model Performance

- **Test MAE**: 0.57 positions
- **Test R²**: 0.971
- **Training Data**: 780 races (2018-2024)
- **Inference Time**: <50ms per prediction

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/AsteriodBlues/Vantage.git
cd Vantage

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
Vantage/
├── app/                    # Streamlit dashboard
│   ├── dashboard.py
│   └── README.md
├── src/                    # Source code
│   ├── prediction_pipeline.py
│   ├── predict_cli.py
│   └── model_deployment.py
├── models/                 # Trained models
│   ├── production/
│   └── preprocessing/
├── data/                   # Data files
│   ├── processed/
│   └── raw/
├── docs/                   # Documentation
│   ├── model_performance.md
│   └── api_specification.md
└── examples/              # Example inputs
```

## 📖 Documentation

- [Dashboard Guide](app/README.md)
- [Model Performance](docs/model_performance.md)
- [API Specification](docs/api_specification.md)

## 🎯 Use Cases

- **Race Prediction**: Forecast finish positions from qualifying results
- **Strategy Planning**: Analyze grid position advantages per circuit
- **Historical Analysis**: Compare circuit characteristics and trends
- **What-If Scenarios**: Simulate race outcomes with different grids

## 🔧 Development

### Training a New Model

```bash
python train_simple_model.py
```

### Building Feature Database

```bash
python build_feature_database.py
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

MIT License

## 🙏 Acknowledgments

Data sourced from historical F1 race results and statistics.
