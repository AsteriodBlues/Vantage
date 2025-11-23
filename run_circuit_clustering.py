"""
Circuit Clustering and Similarity Analysis Pipeline

Complete workflow for identifying and analyzing circuit groups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.circuit_features import create_circuit_feature_matrix
from src.circuit_clustering import run_clustering_analysis, scale_features
from src.circuit_pca import run_pca_analysis


def main():
    """Run complete circuit clustering pipeline."""
    print("="*80)
    print("CIRCUIT CLUSTERING AND SIMILARITY ANALYSIS")
    print("="*80)

    # Step 1: Create circuit feature matrix
    print("\n" + "="*80)
    print("STEP 1: CIRCUIT FEATURE MATRIX PREPARATION")
    print("="*80)

    circuit_features, clustering_features = create_circuit_feature_matrix(
        'data/processed/train.csv'
    )

    print(f"\nCircuit feature matrix created:")
    print(f"  Total features: {circuit_features.shape[1]}")
    print(f"  Clustering features: {clustering_features.shape[1]}")
    print(f"  Circuits: {len(clustering_features)}")

    # Step 2: Run clustering analysis
    print("\n" + "="*80)
    print("STEP 2: K-MEANS CLUSTERING")
    print("="*80)

    clustering_results = run_clustering_analysis(
        clustering_features,
        output_dir='results/clustering',
        n_clusters=None  # Auto-determine optimal k
    )

    print(f"\nClustering complete:")
    print(f"  Number of clusters: {len(np.unique(clustering_results['labels']))}")
    print(f"  Cluster names:")
    for cluster_id, name in clustering_results['cluster_names'].items():
        n_circuits = (clustering_results['labels'] == cluster_id).sum()
        print(f"    Cluster {cluster_id} ({name}): {n_circuits} circuits")

    # Step 3: PCA visualization
    print("\n" + "="*80)
    print("STEP 3: PCA DIMENSIONALITY REDUCTION")
    print("="*80)

    # Get scaled features from clustering
    X_scaled, _ = scale_features(clustering_features)

    pca_results = run_pca_analysis(
        X_scaled,
        clustering_features,
        cluster_labels=clustering_results['labels'],
        output_dir='results/clustering'
    )

    # Step 4: Summary statistics
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*80)

    # Load cluster assignments
    cluster_df = pd.read_csv('results/clustering/circuit_clusters.csv')

    print(f"\nCircuit Cluster Assignments:")
    for _, row in cluster_df.iterrows():
        print(f"  {row['circuit']:30s} -> {row['cluster_name']}")

    # PCA variance
    total_var = pca_results['pca_model'].explained_variance_ratio_.sum()
    print(f"\nPCA Analysis:")
    print(f"  Total variance (3 components): {total_var:.1%}")
    print(f"  PC1: {pca_results['pca_model'].explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca_results['pca_model'].explained_variance_ratio_[1]:.1%}")
    print(f"  PC3: {pca_results['pca_model'].explained_variance_ratio_[2]:.1%}")

    # Top PC1 loadings
    print(f"\nTop features driving PC1:")
    pc1_loadings = pca_results['loadings']['PC1'].abs().nlargest(3)
    for feat, loading in pc1_loadings.items():
        print(f"  {feat}: {loading:.3f}")

    print(f"\nAll results saved to: results/clustering/")
    print(f"  - circuit_clusters.csv: Cluster assignments")
    print(f"  - cluster_profiles.csv: Cluster characteristics")
    print(f"  - elbow_analysis.png: Optimal k determination")
    print(f"  - dendrogram.png: Hierarchical clustering")
    print(f"  - clusters_2d.png: 2D cluster visualization")
    print(f"  - pca_2d.png: 2D PCA projection")
    print(f"  - pca_3d.png: 3D PCA projection")
    print(f"  - pca_loadings.csv: Component loadings")
    print(f"  - scree_plot.png: Variance explained")

    print("\n" + "="*80)
    print("CIRCUIT CLUSTERING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
