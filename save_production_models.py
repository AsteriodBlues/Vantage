"""
Save trained models for production deployment.

Exports best models with metadata and preprocessing artifacts.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.model_deployment import (
    save_model_pipeline,
    create_symlink_latest,
    save_preprocessing_artifacts
)


def main():
    """Save production models."""
    print("="*80)
    print("PRODUCTION MODEL EXPORT")
    print("="*80)

    # Check for trained models
    model_path = Path('results/models/best_model.pkl')
    if not model_path.exists():
        print(f"\nError: No trained model found at {model_path}")
        print("Please train a model first using the training scripts")
        return

    # Load best model
    print("\nLoading best model...")
    best_model = joblib.load(model_path)
    print(f"Model type: {type(best_model).__name__}")

    # Load training data for feature names
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')

    # Identify feature columns (exclude target and metadata)
    exclude_cols = [
        'Position', 'circuit', 'year', 'round', 'race_name', 'date',
        'DriverNumber', 'BroadcastName', 'Abbreviation', 'DriverId',
        'TeamName', 'TeamColor', 'TeamId', 'FirstName', 'LastName',
        'FullName', 'HeadshotUrl', 'CountryCode', 'ClassifiedPosition',
        'Q1', 'Q2', 'Q3', 'Time', 'Status', 'Points', 'Laps'
    ]

    feature_cols = [col for col in train.columns if col not in exclude_cols]
    print(f"\nIdentified {len(feature_cols)} features")

    # Prepare model metadata
    metadata = {
        'model_type': type(best_model).__name__,
        'target': 'finish_position',
        'features': feature_cols,
        'n_features': len(feature_cols),
        'performance': {
            'training_samples': len(train),
            'validation_samples': len(val)
        },
        'training_date': '2024-11',
        'data_years': '2018-2024',
        'notes': 'Production model for race finish position prediction'
    }

    # Calculate performance metrics if possible
    try:
        X_val = val[feature_cols]
        y_val = val['Position']

        predictions = best_model.predict(X_val)
        mae = np.mean(np.abs(predictions - y_val))
        rmse = np.sqrt(np.mean((predictions - y_val)**2))

        metadata['performance']['validation_mae'] = float(mae)
        metadata['performance']['validation_rmse'] = float(rmse)

        print(f"\nValidation Performance:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")

    except Exception as e:
        print(f"\nWarning: Could not calculate metrics: {e}")

    # Save main model
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")

    model_dir = save_model_pipeline(
        best_model,
        'finish_position_predictor',
        metadata
    )

    # Create latest symlink
    create_symlink_latest(model_dir)

    # Save preprocessing artifacts
    print(f"\n{'='*80}")
    print("SAVING PREPROCESSING ARTIFACTS")
    print(f"{'='*80}")

    artifacts = {
        'feature_names': feature_cols,
        'feature_config': {
            'all_features': feature_cols,
            'n_features': len(feature_cols)
        }
    }

    # Save circuit metadata
    circuits = train['circuit'].unique()
    circuit_stats = {}
    for circuit in circuits:
        circuit_data = train[train['circuit'] == circuit]
        circuit_stats[circuit] = {
            'total_races': len(circuit_data),
            'avg_finish': float(circuit_data['Position'].mean()),
            'dnf_rate': float((circuit_data['Position'] > 20).mean())
        }

    artifacts['circuit_statistics'] = circuit_stats

    # Save team baselines
    teams = train['TeamName'].unique()
    team_stats = {}
    for team in teams:
        team_data = train[train['TeamName'] == team]
        team_stats[team] = {
            'avg_finish': float(team_data['Position'].mean()),
            'avg_grid': float(team_data['GridPosition'].mean())
        }

    artifacts['team_baselines'] = team_stats

    save_preprocessing_artifacts(artifacts)

    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"\nModel saved to: {model_dir}")
    print(f"Preprocessing saved to: models/preprocessing/")
    print(f"\nFiles created:")
    print(f"  • {model_dir}/model.pkl")
    print(f"  • {model_dir}/metadata.json")
    print(f"  • models/preprocessing/feature_names.pkl")
    print(f"  • models/preprocessing/circuit_statistics.pkl")
    print(f"  • models/preprocessing/team_baselines.pkl")


if __name__ == "__main__":
    main()
