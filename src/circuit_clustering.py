"""
Circuit clustering using K-Means and hierarchical methods.

Identifies groups of similar circuits based on characteristics and performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features for clustering.

    Args:
        X: Feature matrix

    Returns:
        Tuple of (scaled_array, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def determine_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 10),
    random_state: int = 42
) -> Dict[str, List]:
    """
    Determine optimal number of clusters using multiple methods.

    Args:
        X_scaled: Scaled feature matrix
        k_range: Range of k values to test
        random_state: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    print("Determining optimal number of clusters...")

    results = {
        'k_values': list(k_range),
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': []
    }

    for k in k_range:
        print(f"  Testing k={k}...")

        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Calculate metrics
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))

    # Find optimal k
    silhouette_optimal = k_range[np.argmax(results['silhouette'])]
    db_optimal = k_range[np.argmin(results['davies_bouldin'])]

    print(f"\nOptimal k by silhouette score: {silhouette_optimal}")
    print(f"Optimal k by Davies-Bouldin: {db_optimal}")

    return results


def plot_elbow_analysis(
    results: Dict[str, List],
    save_path: str = None
) -> None:
    """
    Create elbow and silhouette plots.

    Args:
        results: Results from determine_optimal_k
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow plot
    axes[0].plot(results['k_values'], results['inertia'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[0].set_ylabel('Inertia', fontsize=10)
    axes[0].set_title('Elbow Method', fontsize=11, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Silhouette plot
    axes[1].plot(results['k_values'], results['silhouette'], 'go-', linewidth=2)
    optimal_k = results['k_values'][np.argmax(results['silhouette'])]
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7,
                    label=f'Optimal k={optimal_k}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[1].set_ylabel('Silhouette Score', fontsize=10)
    axes[1].set_title('Silhouette Analysis', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Davies-Bouldin plot (lower is better)
    axes[2].plot(results['k_values'], results['davies_bouldin'], 'ro-', linewidth=2)
    optimal_k_db = results['k_values'][np.argmin(results['davies_bouldin'])]
    axes[2].axvline(x=optimal_k_db, color='g', linestyle='--', alpha=0.7,
                    label=f'Optimal k={optimal_k_db}')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=10)
    axes[2].set_title('Davies-Bouldin Index (lower=better)', fontsize=11, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved elbow analysis to {save_path}")

    plt.close()


def apply_kmeans(
    X_scaled: np.ndarray,
    X_original: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Apply K-Means clustering.

    Args:
        X_scaled: Scaled features
        X_original: Original feature DataFrame
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (fitted_model, cluster_labels)
    """
    print(f"\nApplying K-Means with k={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate final metrics
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_

    print(f"  Silhouette score: {silhouette:.3f}")
    print(f"  Inertia: {inertia:.2f}")

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        circuits = X_original.index[labels == cluster].tolist()
        print(f"  Cluster {cluster}: {count} circuits")
        circuit_names = [str(c) for c in circuits[:5]]
        print(f"    {', '.join(circuit_names)}{'...' if count > 5 else ''}")

    return kmeans, labels


def analyze_cluster_characteristics(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_top_features: int = 5
) -> pd.DataFrame:
    """
    Analyze what characterizes each cluster.

    Args:
        X: Feature matrix
        labels: Cluster labels
        n_top_features: Number of top features to show per cluster

    Returns:
        DataFrame with cluster characteristics
    """
    print("\nAnalyzing cluster characteristics...")

    cluster_profiles = []

    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        cluster_data = X[cluster_mask]

        # Calculate cluster center (mean)
        cluster_mean = cluster_data.mean()

        # Compare to overall mean
        overall_mean = X.mean()
        difference = cluster_mean - overall_mean

        # Get top distinguishing features
        top_features = difference.abs().nlargest(n_top_features)

        cluster_profiles.append({
            'cluster': cluster,
            'size': cluster_mask.sum(),
            'circuits': ', '.join([str(c) for c in X.index[cluster_mask].tolist()]),
            'top_features': ', '.join(top_features.index.tolist()),
            'feature_values': ', '.join([f"{v:.2f}" for v in cluster_mean[top_features.index]])
        })

    profile_df = pd.DataFrame(cluster_profiles)

    print(profile_df[['cluster', 'size', 'top_features']])

    return profile_df


def create_hierarchical_dendrogram(
    X_scaled: np.ndarray,
    circuit_names: List[str],
    save_path: str = None
) -> np.ndarray:
    """
    Create hierarchical clustering dendrogram.

    Args:
        X_scaled: Scaled features
        circuit_names: Circuit names for labels
        save_path: Path to save figure

    Returns:
        Linkage matrix
    """
    print("\nCreating hierarchical clustering dendrogram...")

    # Calculate distance matrix
    distance_matrix = pdist(X_scaled, metric='euclidean')

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Create dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=circuit_names,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=10
    )
    plt.title('Circuit Similarity Dendrogram (Ward Linkage)', fontsize=12, fontweight='bold')
    plt.xlabel('Circuit', fontsize=10)
    plt.ylabel('Distance', fontsize=10)
    plt.axhline(y=8, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Suggested cut')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dendrogram to {save_path}")

    plt.close()

    return linkage_matrix


def visualize_clusters_2d(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_x: str = None,
    feature_y: str = None,
    save_path: str = None
) -> None:
    """
    Visualize clusters in 2D feature space.

    Args:
        X: Feature matrix
        labels: Cluster labels
        feature_x: Feature for x-axis (auto-select if None)
        feature_y: Feature for y-axis (auto-select if None)
        save_path: Path to save figure
    """
    # Auto-select features if not provided
    if feature_x is None:
        feature_x = X.columns[0]
    if feature_y is None:
        feature_y = X.columns[1] if len(X.columns) > 1 else X.columns[0]

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = plt.scatter(
        X[feature_x],
        X[feature_y],
        c=labels,
        cmap='viridis',
        s=200,
        alpha=0.6,
        edgecolors='black',
        linewidth=1
    )

    # Add circuit labels
    for idx, circuit in enumerate(X.index):
        plt.annotate(
            circuit,
            (X.iloc[idx][feature_x], X.iloc[idx][feature_y]),
            fontsize=8,
            ha='center',
            va='center'
        )

    plt.xlabel(feature_x.replace('_', ' ').title(), fontsize=10)
    plt.ylabel(feature_y.replace('_', ' ').title(), fontsize=10)
    plt.title(f'Circuit Clusters: {feature_x} vs {feature_y}', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization to {save_path}")

    plt.close()


def name_clusters(
    X: pd.DataFrame,
    labels: np.ndarray
) -> Dict[int, str]:
    """
    Automatically name clusters based on characteristics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary mapping cluster ID to name
    """
    cluster_names = {}

    for cluster in np.unique(labels):
        cluster_data = X[labels == cluster]
        cluster_mean = cluster_data.mean()

        # Determine name based on dominant characteristics
        if 'overtaking_score' in X.columns:
            if cluster_mean['overtaking_score'] < -0.5:
                name = "Processional"
            elif cluster_mean['overtaking_score'] > 0.5:
                name = "High Overtaking"
            else:
                name = "Balanced"
        else:
            name = f"Type {cluster}"

        # Add more specific descriptors
        if 'chaos_factor' in X.columns and cluster_mean['chaos_factor'] > 0.5:
            name += " Chaotic"
        if 'pole_win_rate' in X.columns and cluster_mean['pole_win_rate'] > 0.5:
            name += " Grid-Dominant"

        cluster_names[cluster] = name.strip()

    return cluster_names


def run_clustering_analysis(
    X: pd.DataFrame,
    output_dir: str = 'results/clustering',
    n_clusters: int = None
) -> Dict:
    """
    Complete clustering analysis pipeline.

    Args:
        X: Feature matrix
        output_dir: Directory to save results
        n_clusters: Number of clusters (None for auto-determine)

    Returns:
        Dictionary with clustering results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CIRCUIT CLUSTERING ANALYSIS")
    print("="*60)

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Determine optimal k if not specified
    if n_clusters is None:
        opt_results = determine_optimal_k(X_scaled)
        plot_elbow_analysis(opt_results, save_path=output_path / 'elbow_analysis.png')

        # Use silhouette score to choose k
        n_clusters = opt_results['k_values'][np.argmax(opt_results['silhouette'])]
        print(f"\nAuto-selected k={n_clusters} based on silhouette score")

    # Apply K-Means
    kmeans, labels = apply_kmeans(X_scaled, X, n_clusters)

    # Analyze clusters
    cluster_profiles = analyze_cluster_characteristics(X, labels)

    # Name clusters
    cluster_names = name_clusters(X, labels)
    print(f"\nCluster names:")
    for cluster_id, name in cluster_names.items():
        print(f"  Cluster {cluster_id}: {name}")

    # Create dendrogram
    linkage_matrix = create_hierarchical_dendrogram(
        X_scaled,
        X.index.tolist(),
        save_path=output_path / 'dendrogram.png'
    )

    # Visualize in 2D - use available features
    vis_features = [f for f in ['overtaking_score', 'pole_win_rate'] if f in X.columns]
    if len(vis_features) >= 2:
        visualize_clusters_2d(
            X, labels, vis_features[0], vis_features[1],
            save_path=output_path / 'clusters_2d.png'
        )
    else:
        # Use any two available features
        visualize_clusters_2d(
            X, labels,
            save_path=output_path / 'clusters_2d.png'
        )

    # Save results
    results_df = pd.DataFrame({
        'circuit': X.index,
        'cluster': labels,
        'cluster_name': [cluster_names[l] for l in labels]
    })
    results_df.to_csv(output_path / 'circuit_clusters.csv', index=False)

    cluster_profiles.to_csv(output_path / 'cluster_profiles.csv', index=False)

    print(f"\nSaved clustering results to {output_path}")

    return {
        'kmeans': kmeans,
        'labels': labels,
        'cluster_names': cluster_names,
        'scaler': scaler,
        'profiles': cluster_profiles
    }
