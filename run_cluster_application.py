"""
Apply circuit clustering to improve race prediction models.

Tests cluster-based features and cluster-specific models.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.cluster_application import (
    train_model_with_clusters,
    train_cluster_specific_models,
    find_similar_circuits
)
from src.alternative_clustering import compare_clustering_methods
from src.circuit_clustering import scale_features


def main():
    """Run cluster application analysis."""
    print("="*80)
    print("CIRCUIT CLUSTERING APPLICATION TO RACE PREDICTIONS")
    print("="*80)

    # Step 1: Train model with cluster features
    print("\n" + "="*80)
    print("STEP 1: MODEL WITH CLUSTER FEATURES")
    print("="*80)

    cluster_results = train_model_with_clusters(
        train_path='data/processed/train.csv',
        val_path='data/processed/val.csv',
        cluster_path='results/clustering/circuit_clusters.csv',
        output_dir='results/clustering'
    )

    # Step 2: Train cluster-specific models
    print("\n" + "="*80)
    print("STEP 2: CLUSTER-SPECIFIC MODELS")
    print("="*80)

    specific_results = train_cluster_specific_models(
        train_path='data/processed/train.csv',
        val_path='data/processed/val.csv',
        cluster_path='results/clustering/circuit_clusters.csv',
        output_dir='results/clustering'
    )

    # Step 3: Circuit similarity analysis
    print("\n" + "="*80)
    print("STEP 3: CIRCUIT SIMILARITY ANALYSIS")
    print("="*80)

    # Load clustering features
    clustering_features = pd.read_csv('results/clustering/clustering_features.csv', index_col=0)
    cluster_assignments = pd.read_csv('results/clustering/circuit_clusters.csv')

    # Find similar circuits for a few examples
    example_circuits = clustering_features.index[:3].tolist()

    for circuit_id in example_circuits:
        similar = find_similar_circuits(
            circuit_id,
            clustering_features,
            cluster_assignments,
            n_similar=3
        )

    # Step 4: Alternative clustering methods
    print("\n" + "="*80)
    print("STEP 4: ALTERNATIVE CLUSTERING METHODS")
    print("="*80)

    # Load K-Means results for comparison
    kmeans_clusters = pd.read_csv('results/clustering/circuit_clusters.csv')

    # Scale features for alternative methods
    X_scaled, _ = scale_features(clustering_features)

    # Compare methods
    comparison = compare_clustering_methods(
        X_scaled,
        clustering_features,
        kmeans_clusters['cluster'].values,
        output_dir='results/clustering'
    )

    # Summary
    print("\n" + "="*80)
    print("CLUSTER APPLICATION SUMMARY")
    print("="*80)

    print("\nModel Performance:")
    print(f"  Baseline RMSE: {cluster_results['baseline_metrics']['rmse']:.4f}")
    print(f"  With Clusters RMSE: {cluster_results['cluster_metrics']['rmse']:.4f}")
    print(f"  Improvement: {cluster_results['improvement']['rmse']:+.2f}%")

    print(f"\nCluster-Specific Models:")
    print(f"  Overall RMSE: {specific_results['overall_metrics']['rmse']:.4f}")
    print(f"  Number of models: {len(specific_results['models'])}")

    print("\nAlternative Clustering Methods:")
    print(comparison[['Method', 'N_Clusters', 'Silhouette']].to_string(index=False))

    print(f"\nAll results saved to: results/clustering/")
    print(f"  - cluster_model_comparison.csv")
    print(f"  - cluster_feature_importance.csv")
    print(f"  - cluster_specific_metrics.csv")
    print(f"  - model_with_clusters.pkl")
    print(f"  - model_cluster_*.pkl")
    print(f"  - dbscan_clusters.csv")
    print(f"  - gmm_clusters.csv")
    print(f"  - clustering_comparison.csv")

    print("\n" + "="*80)
    print("CLUSTER APPLICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
