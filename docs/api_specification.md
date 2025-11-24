# F1 Prediction API Specification

## Base URL
```
http://api.vantage-f1.com/v1
```

## Authentication
API Key required in header:
```
X-API-Key: <your-api-key>
```

## Endpoints

### 1. Single Position Prediction
**POST** `/predict/position`

Predict finish position for a single driver.

#### Request Body
```json
{
  "grid_position": 5,
  "circuit_name": "Monaco",
  "team": "Mercedes",
  "driver": "Lewis Hamilton",
  "year": 2024,
  "race_number": 7
}
```

#### Response
```json
{
  "status": "success",
  "data": {
    "predicted_finish": 3.7,
    "predicted_finish_rounded": 4,
    "confidence_interval": {
      "lower": 2.1,
      "upper": 5.3
    },
    "position_change": 1.3,
    "probabilities": {
      "win": 0.12,
      "podium": 0.45,
      "points": 0.89
    }
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_time_ms": 45
  }
}
```

### 2. Full Grid Prediction
**POST** `/predict/race`

Predict full race result from starting grid.

#### Request Body
```json
{
  "circuit_name": "Monaco",
  "year": 2024,
  "race_number": 7,
  "grid": [
    {
      "position": 1,
      "driver": "Max Verstappen",
      "team": "Red Bull"
    },
    {
      "position": 2,
      "driver": "Charles Leclerc",
      "team": "Ferrari"
    }
  ]
}
```

#### Response
```json
{
  "status": "success",
  "data": {
    "predicted_result": [
      {
        "position": 1,
        "driver": "Max Verstappen",
        "team": "Red Bull",
        "grid_position": 1,
        "position_change": 0,
        "confidence": 0.78
      }
    ],
    "race_metrics": {
      "expected_position_changes": 3.2,
      "overtaking_difficulty": 0.82,
      "dnf_probability": 0.15
    }
  }
}
```

### 3. Win Probability
**POST** `/predict/win-probability`

Calculate win probability from grid position.

#### Request Body
```json
{
  "grid_positions": [1, 3, 5, 10],
  "circuit_name": "Monza",
  "team": "Ferrari"
}
```

#### Response
```json
{
  "status": "success",
  "data": [
    {"grid": 1, "win_probability": 0.52},
    {"grid": 3, "win_probability": 0.18},
    {"grid": 5, "win_probability": 0.08},
    {"grid": 10, "win_probability": 0.01}
  ]
}
```

### 4. Circuit Analysis
**GET** `/circuits/{circuit_name}`

Get circuit characteristics and statistics.

#### Response
```json
{
  "status": "success",
  "data": {
    "circuit_name": "Monaco",
    "cluster": "Street Circuit",
    "characteristics": {
      "pole_win_rate": 0.78,
      "overtaking_score": 0.12,
      "avg_position_changes": 1.8,
      "dnf_rate": 0.18
    },
    "grid_advantages": {
      "P1": {"avg_finish": 1.7, "win_rate": 0.78},
      "P2": {"avg_finish": 3.1, "win_rate": 0.15}
    }
  }
}
```

### 5. Model Information
**GET** `/model/info`

Get information about current model.

#### Response
```json
{
  "status": "success",
  "data": {
    "model_version": "1.0.0",
    "model_type": "RandomForest",
    "training_date": "2024-11",
    "performance_metrics": {
      "validation_mae": 1.84,
      "test_mae": 1.86
    },
    "data_coverage": {
      "years": "2018-2024",
      "circuits": 18,
      "races": 154
    }
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "status": "error",
  "error": {
    "code": "INVALID_INPUT",
    "message": "Grid position must be between 1 and 20"
  }
}
```

### 404 Not Found
```json
{
  "status": "error",
  "error": {
    "code": "CIRCUIT_NOT_FOUND",
    "message": "Circuit 'InvalidName' not in database"
  }
}
```

### 500 Internal Server Error
```json
{
  "status": "error",
  "error": {
    "code": "PREDICTION_ERROR",
    "message": "Model prediction failed"
  }
}
```

## Rate Limiting
- 100 requests per minute per API key
- 1000 requests per hour per API key

## Versioning
API version specified in URL: `/v1/`, `/v2/`, etc.

## Response Times
- Single predictions: ~50ms
- Full grid predictions: ~200ms
- Circuit analysis: ~10ms (cached)

## Data Freshness
- Model updated: Monthly
- Circuit statistics: After each race
- Team baselines: Weekly during season
