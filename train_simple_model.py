"""
Train a simplified model without cluster features for easier deployment.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


def main():
    """Train simplified model."""
    # load data
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')

    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")

    # select features (no cluster features, only numeric)
    feature_cols = [col for col in train.columns
                   if col not in ['Position', 'Position_raw', 'circuit_cluster', 'cluster_1', 'cluster_0',
                                  'circuit', 'TeamName', 'DriverId', 'url', 'driver_url', 'team_url']
                   and train[col].dtype in ['int64', 'float64']
                   and not col.endswith('_url')
                   and 'url' not in col.lower()]

    X_train = train[feature_cols]
    y_train = train['Position_raw']

    X_val = val[feature_cols]
    y_val = val['Position_raw']

    X_test = test[feature_cols]
    y_test = test['Position_raw']

    print(f"\nFeatures: {len(feature_cols)}")

    # train model
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nPerformance:")
    print(f"Train MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    print(f"Val MAE:   {val_mae:.3f}, R²: {val_r2:.3f}")
    print(f"Test MAE:  {test_mae:.3f}, R²: {test_r2:.3f}")

    # save model
    from src.model_deployment import save_model_pipeline, create_symlink_latest, save_preprocessing_artifacts

    metadata = {
        'model_type': 'RandomForestRegressor',
        'training_samples': len(train),
        'validation_samples': len(val),
        'test_samples': len(test),
        'num_features': len(feature_cols),
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'val_r2': float(val_r2),
        'test_r2': float(test_r2),
        'features': feature_cols
    }

    model_dir = save_model_pipeline(model, 'simple_predictor', metadata)
    create_symlink_latest(model_dir, 'models/production')

    # save preprocessing artifacts
    artifacts = {
        'feature_names': feature_cols
    }
    save_preprocessing_artifacts(artifacts)

    print(f"\nModel saved to: {model_dir}")
    print("Symlink created: models/production/simple_predictor_latest")


if __name__ == '__main__':
    main()
