"""
Apply circuit clustering to improve model predictions.

Implements cluster-based features and cluster-specific models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def add_cluster_features(
    df: pd.DataFrame,
    cluster_assignments: pd.DataFrame
) -> pd.DataFrame:
    """
    Add circuit cluster features to race data.

    Args:
        df: Race data
        cluster_assignments: Circuit cluster mapping

    Returns:
        DataFrame with cluster features added
    """
    print("Adding cluster features to race data...")

    # Create cluster mapping dictionary
    cluster_map = dict(zip(
        cluster_assignments['circuit'],
        cluster_assignments['cluster']
    ))

    # Add cluster ID
    df['circuit_cluster'] = df['circuit'].map(cluster_map)

    # One-hot encode clusters
    cluster_dummies = pd.get_dummies(
        df['circuit_cluster'],
        prefix='cluster',
        dtype=int
    )
    df = pd.concat([df, cluster_dummies], axis=1)

    print(f"Added cluster features: {cluster_dummies.columns.tolist()}")
    print(f"Cluster distribution:\n{df['circuit_cluster'].value_counts().sort_index()}")

    return df


def train_model_with_clusters(
    train_path: str,
    val_path: str,
    cluster_path: str,
    output_dir: str = 'results/clustering'
) -> Dict:
    """
    Train model with circuit cluster features and compare to baseline.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        cluster_path: Path to cluster assignments
        output_dir: Directory to save results

    Returns:
        Dictionary with model and metrics
    """
    print("\n" + "="*60)
    print("TRAINING MODEL WITH CLUSTER FEATURES")
    print("="*60)

    # Load data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    clusters = pd.read_csv(cluster_path)

    # Add cluster features
    train = add_cluster_features(train, clusters)
    val = add_cluster_features(val, clusters)

    # Define feature sets
    baseline_features = [
        'GridPosition', 'driver_encoded', 'circuit_encoded',
        'season_phase', 'team_encoded', 'recent_form'
    ]

    cluster_features = [
        col for col in train.columns
        if col.startswith('cluster_') or col == 'circuit_cluster'
    ]

    # Train baseline model
    print("\nTraining baseline model...")
    X_train_base = train[baseline_features]
    y_train = train['Position']
    X_val_base = val[baseline_features]
    y_val = val['Position']

    baseline_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    baseline_model.fit(X_train_base, y_train)

    baseline_pred = baseline_model.predict(X_val_base)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    baseline_r2 = r2_score(y_val, baseline_pred)

    print(f"Baseline Performance:")
    print(f"  RMSE: {baseline_rmse:.4f}")
    print(f"  MAE:  {baseline_mae:.4f}")
    print(f"  R²:   {baseline_r2:.4f}")

    # Train model with clusters
    print("\nTraining model with cluster features...")
    all_features = baseline_features + cluster_features
    X_train_cluster = train[all_features]
    X_val_cluster = val[all_features]

    cluster_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    cluster_model.fit(X_train_cluster, y_train)

    cluster_pred = cluster_model.predict(X_val_cluster)
    cluster_rmse = np.sqrt(mean_squared_error(y_val, cluster_pred))
    cluster_mae = mean_absolute_error(y_val, cluster_pred)
    cluster_r2 = r2_score(y_val, cluster_pred)

    print(f"Cluster Model Performance:")
    print(f"  RMSE: {cluster_rmse:.4f}")
    print(f"  MAE:  {cluster_mae:.4f}")
    print(f"  R²:   {cluster_r2:.4f}")

    # Calculate improvement
    rmse_improvement = ((baseline_rmse - cluster_rmse) / baseline_rmse) * 100
    mae_improvement = ((baseline_mae - cluster_mae) / baseline_mae) * 100
    r2_improvement = ((cluster_r2 - baseline_r2) / abs(baseline_r2)) * 100

    print(f"\nImprovement:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE:  {mae_improvement:+.2f}%")
    print(f"  R²:   {r2_improvement:+.2f}%")

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': cluster_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    # Save results
    output_path = Path(output_dir)
    joblib.dump(cluster_model, output_path / 'model_with_clusters.pkl')

    results = pd.DataFrame({
        'model': ['Baseline', 'With Clusters'],
        'rmse': [baseline_rmse, cluster_rmse],
        'mae': [baseline_mae, cluster_mae],
        'r2': [baseline_r2, cluster_r2]
    })
    results.to_csv(output_path / 'cluster_model_comparison.csv', index=False)
    feature_importance.to_csv(output_path / 'cluster_feature_importance.csv', index=False)

    return {
        'baseline_model': baseline_model,
        'cluster_model': cluster_model,
        'baseline_metrics': {
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2
        },
        'cluster_metrics': {
            'rmse': cluster_rmse,
            'mae': cluster_mae,
            'r2': cluster_r2
        },
        'improvement': {
            'rmse': rmse_improvement,
            'mae': mae_improvement,
            'r2': r2_improvement
        },
        'feature_importance': feature_importance
    }


def train_cluster_specific_models(
    train_path: str,
    val_path: str,
    cluster_path: str,
    output_dir: str = 'results/clustering'
) -> Dict:
    """
    Train separate models for each circuit cluster.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        cluster_path: Path to cluster assignments
        output_dir: Directory to save results

    Returns:
        Dictionary with models and metrics per cluster
    """
    print("\n" + "="*60)
    print("TRAINING CLUSTER-SPECIFIC MODELS")
    print("="*60)

    # Load data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    clusters = pd.read_csv(cluster_path)

    # Add cluster features
    train = add_cluster_features(train, clusters)
    val = add_cluster_features(val, clusters)

    # Features for cluster-specific models
    features = [
        'GridPosition', 'driver_encoded', 'season_phase',
        'team_encoded', 'recent_form'
    ]

    cluster_models = {}
    cluster_metrics = {}
    predictions = []

    for cluster_id in sorted(train['circuit_cluster'].unique()):
        print(f"\nTraining model for Cluster {cluster_id}...")

        # Split by cluster
        train_cluster = train[train['circuit_cluster'] == cluster_id]
        val_cluster = val[val['circuit_cluster'] == cluster_id]

        if len(val_cluster) == 0:
            print(f"  No validation data for cluster {cluster_id}, skipping")
            continue

        print(f"  Training samples: {len(train_cluster)}")
        print(f"  Validation samples: {len(val_cluster)}")

        # Train model
        X_train = train_cluster[features]
        y_train = train_cluster['Position']
        X_val = val_cluster[features]
        y_val = val_cluster['Position']

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        cluster_models[cluster_id] = model
        cluster_metrics[cluster_id] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(train_cluster),
            'n_val': len(val_cluster)
        }

        # Store predictions
        val_cluster_pred = val_cluster.copy()
        val_cluster_pred['predicted'] = y_pred
        predictions.append(val_cluster_pred)

    # Combine predictions and evaluate overall
    all_predictions = pd.concat(predictions, ignore_index=True)
    overall_rmse = np.sqrt(mean_squared_error(
        all_predictions['Position'],
        all_predictions['predicted']
    ))
    overall_mae = mean_absolute_error(
        all_predictions['Position'],
        all_predictions['predicted']
    )
    overall_r2 = r2_score(
        all_predictions['Position'],
        all_predictions['predicted']
    )

    print(f"\nOverall Cluster-Specific Performance:")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  MAE:  {overall_mae:.4f}")
    print(f"  R²:   {overall_r2:.4f}")

    # Save models and metrics
    output_path = Path(output_dir)
    for cluster_id, model in cluster_models.items():
        joblib.dump(model, output_path / f'model_cluster_{cluster_id}.pkl')

    metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient='index')
    metrics_df.to_csv(output_path / 'cluster_specific_metrics.csv')

    print(f"\nSaved {len(cluster_models)} cluster-specific models")

    return {
        'models': cluster_models,
        'metrics': cluster_metrics,
        'overall_metrics': {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'r2': overall_r2
        }
    }


def predict_new_circuit_cluster(
    circuit_features: pd.DataFrame,
    clustering_model_path: str,
    scaler_path: str = None
) -> int:
    """
    Predict cluster for a new unseen circuit based on similarity.

    Args:
        circuit_features: Features of new circuit
        clustering_model_path: Path to trained K-Means model
        scaler_path: Path to feature scaler

    Returns:
        Predicted cluster ID
    """
    print("\nPredicting cluster for new circuit...")

    # Load clustering artifacts
    clustering_results = joblib.load(clustering_model_path)
    kmeans = clustering_results['kmeans']

    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        circuit_features_scaled = scaler.transform(circuit_features)
    else:
        circuit_features_scaled = circuit_features

    # Predict cluster
    cluster = kmeans.predict(circuit_features_scaled)[0]

    # Get distance to cluster center
    distances = kmeans.transform(circuit_features_scaled)[0]
    confidence = 1 - (distances[cluster] / distances.sum())

    print(f"Predicted Cluster: {cluster}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Distances to cluster centers: {distances}")

    return cluster


def find_similar_circuits(
    circuit_id: int,
    clustering_features: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    n_similar: int = 3
) -> pd.DataFrame:
    """
    Find most similar circuits based on clustering features.

    Args:
        circuit_id: Circuit to find similarities for
        clustering_features: Feature matrix used for clustering
        cluster_assignments: Cluster assignments
        n_similar: Number of similar circuits to return

    Returns:
        DataFrame with similar circuits and similarity scores
    """
    print(f"\nFinding circuits similar to circuit {circuit_id}...")

    # Get circuit's features
    circuit_features = clustering_features.loc[circuit_id].values.reshape(1, -1)

    # Calculate Euclidean distances to all other circuits
    distances = []
    for idx in clustering_features.index:
        if idx != circuit_id:
            other_features = clustering_features.loc[idx].values.reshape(1, -1)
            dist = np.linalg.norm(circuit_features - other_features)
            distances.append({
                'circuit': idx,
                'distance': dist
            })

    # Sort by distance
    similar_df = pd.DataFrame(distances).sort_values('distance')
    similar_df = similar_df.head(n_similar)

    # Add cluster info
    similar_df = similar_df.merge(
        cluster_assignments[['circuit', 'cluster', 'cluster_name']],
        on='circuit',
        how='left'
    )

    # Calculate similarity score (0-1, where 1 is identical)
    max_dist = similar_df['distance'].max()
    similar_df['similarity'] = 1 - (similar_df['distance'] / max_dist)

    print(f"\nTop {n_similar} similar circuits:")
    print(similar_df[['circuit', 'cluster_name', 'similarity']].to_string(index=False))

    return similar_df
