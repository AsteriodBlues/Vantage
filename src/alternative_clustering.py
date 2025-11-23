"""
Alternative clustering methods for circuit similarity analysis.

Implements DBSCAN and Gaussian Mixture Models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
from typing import Dict, Tuple


def apply_dbscan(
    X_scaled: np.ndarray,
    X_original: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 2
) -> Tuple[DBSCAN, np.ndarray]:
    """
    Apply DBSCAN clustering to identify outliers and dense regions.

    Args:
        X_scaled: Scaled feature matrix
        X_original: Original feature DataFrame
        eps: Maximum distance between samples
        min_samples: Minimum samples in a neighborhood

    Returns:
        Tuple of (fitted_model, cluster_labels)
    """
    print("\n" + "="*60)
    print("DBSCAN CLUSTERING")
    print("="*60)
    print(f"Parameters: eps={eps}, min_samples={min_samples}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Analyze results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nResults:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")

    # Print cluster compositions
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        if label == -1:
            continue
        cluster_mask = labels == label
        circuits = X_original.index[cluster_mask].tolist()
        print(f"\n  Cluster {label}: {len(circuits)} circuits")
        print(f"    {', '.join([str(c) for c in circuits])}")

    # Outliers
    if n_noise > 0:
        outliers = X_original.index[labels == -1].tolist()
        print(f"\n  Outliers ({n_noise} circuits):")
        print(f"    {', '.join([str(c) for c in outliers])}")

    # Calculate metrics (excluding noise points)
    if n_clusters > 1:
        valid_mask = labels != -1
        if valid_mask.sum() > n_clusters:
            silhouette = silhouette_score(
                X_scaled[valid_mask],
                labels[valid_mask]
            )
            db_score = davies_bouldin_score(
                X_scaled[valid_mask],
                labels[valid_mask]
            )
            print(f"\n  Silhouette score: {silhouette:.3f}")
            print(f"  Davies-Bouldin score: {db_score:.3f}")

    return dbscan, labels


def optimize_dbscan_params(
    X_scaled: np.ndarray,
    eps_range: np.ndarray = None,
    min_samples_range: range = None
) -> Dict:
    """
    Find optimal DBSCAN parameters.

    Args:
        X_scaled: Scaled feature matrix
        eps_range: Range of eps values to test
        min_samples_range: Range of min_samples to test

    Returns:
        Dictionary with optimization results
    """
    print("\nOptimizing DBSCAN parameters...")

    if eps_range is None:
        eps_range = np.arange(0.3, 2.0, 0.1)
    if min_samples_range is None:
        min_samples_range = range(2, 5)

    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Skip if no clusters or all noise
            if n_clusters == 0 or n_noise == len(labels):
                continue

            # Calculate silhouette for valid configurations
            valid_mask = labels != -1
            if n_clusters > 1 and valid_mask.sum() > n_clusters:
                try:
                    silhouette = silhouette_score(
                        X_scaled[valid_mask],
                        labels[valid_mask]
                    )
                except:
                    silhouette = -1
            else:
                silhouette = -1

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Find best by silhouette
        best = results_df.loc[results_df['silhouette'].idxmax()]
        print(f"\nBest parameters (by silhouette):")
        print(f"  eps: {best['eps']:.2f}")
        print(f"  min_samples: {int(best['min_samples'])}")
        print(f"  n_clusters: {int(best['n_clusters'])}")
        print(f"  silhouette: {best['silhouette']:.3f}")

        return {
            'results': results_df,
            'best_params': {
                'eps': best['eps'],
                'min_samples': int(best['min_samples'])
            }
        }
    else:
        print("No valid DBSCAN configurations found")
        return {'results': results_df, 'best_params': None}


def apply_gmm(
    X_scaled: np.ndarray,
    X_original: pd.DataFrame,
    n_components: int = 3,
    covariance_type: str = 'full'
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Apply Gaussian Mixture Model clustering.

    Args:
        X_scaled: Scaled feature matrix
        X_original: Original feature DataFrame
        n_components: Number of mixture components
        covariance_type: Type of covariance parameters

    Returns:
        Tuple of (fitted_model, cluster_labels, probabilities)
    """
    print("\n" + "="*60)
    print("GAUSSIAN MIXTURE MODEL CLUSTERING")
    print("="*60)
    print(f"Parameters: n_components={n_components}, covariance={covariance_type}")

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        n_init=10
    )
    labels = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    # Metrics
    silhouette = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    print(f"\nResults:")
    print(f"  Silhouette score: {silhouette:.3f}")
    print(f"  Davies-Bouldin score: {db_score:.3f}")
    print(f"  BIC: {bic:.2f}")
    print(f"  AIC: {aic:.2f}")

    # Print cluster compositions
    print(f"\nCluster assignments:")
    for cluster in range(n_components):
        cluster_mask = labels == cluster
        circuits = X_original.index[cluster_mask].tolist()
        avg_prob = probabilities[cluster_mask, cluster].mean()

        print(f"\n  Cluster {cluster}: {len(circuits)} circuits (avg prob: {avg_prob:.2f})")
        print(f"    {', '.join([str(c) for c in circuits])}")

    # Show uncertain assignments (low max probability)
    max_probs = probabilities.max(axis=1)
    uncertain_mask = max_probs < 0.7
    if uncertain_mask.sum() > 0:
        print(f"\n  Uncertain assignments ({uncertain_mask.sum()} circuits):")
        for idx in X_original.index[uncertain_mask]:
            idx_pos = X_original.index.get_loc(idx)
            probs = probabilities[idx_pos]
            print(f"    Circuit {idx}: {dict(enumerate(probs.round(2)))}")

    return gmm, labels, probabilities


def determine_optimal_gmm_components(
    X_scaled: np.ndarray,
    n_range: range = range(2, 8)
) -> Dict:
    """
    Determine optimal number of GMM components.

    Args:
        X_scaled: Scaled feature matrix
        n_range: Range of component counts to test

    Returns:
        Dictionary with evaluation results
    """
    print("\nDetermining optimal number of GMM components...")

    results = {
        'n_components': [],
        'bic': [],
        'aic': [],
        'silhouette': []
    }

    for n in n_range:
        print(f"  Testing n={n}...")

        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        labels = gmm.fit_predict(X_scaled)

        results['n_components'].append(n)
        results['bic'].append(gmm.bic(X_scaled))
        results['aic'].append(gmm.aic(X_scaled))
        results['silhouette'].append(silhouette_score(X_scaled, labels))

    # Find optimal
    bic_optimal = results['n_components'][np.argmin(results['bic'])]
    aic_optimal = results['n_components'][np.argmin(results['aic'])]
    silhouette_optimal = results['n_components'][np.argmax(results['silhouette'])]

    print(f"\nOptimal components:")
    print(f"  By BIC: {bic_optimal}")
    print(f"  By AIC: {aic_optimal}")
    print(f"  By Silhouette: {silhouette_optimal}")

    return results


def plot_gmm_selection(
    results: Dict,
    save_path: str = None
) -> None:
    """
    Plot GMM model selection criteria.

    Args:
        results: Results from determine_optimal_gmm_components
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    n_components = results['n_components']

    # BIC
    axes[0].plot(n_components, results['bic'], 'bo-', linewidth=2)
    optimal_n = n_components[np.argmin(results['bic'])]
    axes[0].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.7,
                    label=f'Optimal n={optimal_n}')
    axes[0].set_xlabel('Number of Components', fontsize=10)
    axes[0].set_ylabel('BIC', fontsize=10)
    axes[0].set_title('Bayesian Information Criterion (lower=better)',
                      fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AIC
    axes[1].plot(n_components, results['aic'], 'go-', linewidth=2)
    optimal_n = n_components[np.argmin(results['aic'])]
    axes[1].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.7,
                    label=f'Optimal n={optimal_n}')
    axes[1].set_xlabel('Number of Components', fontsize=10)
    axes[1].set_ylabel('AIC', fontsize=10)
    axes[1].set_title('Akaike Information Criterion (lower=better)',
                      fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Silhouette
    axes[2].plot(n_components, results['silhouette'], 'ro-', linewidth=2)
    optimal_n = n_components[np.argmax(results['silhouette'])]
    axes[2].axvline(x=optimal_n, color='g', linestyle='--', alpha=0.7,
                    label=f'Optimal n={optimal_n}')
    axes[2].set_xlabel('Number of Components', fontsize=10)
    axes[2].set_ylabel('Silhouette Score', fontsize=10)
    axes[2].set_title('Silhouette Score (higher=better)',
                      fontsize=11, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved GMM selection plot to {save_path}")

    plt.close()


def compare_clustering_methods(
    X_scaled: np.ndarray,
    X_original: pd.DataFrame,
    kmeans_labels: np.ndarray,
    output_dir: str = 'results/clustering'
) -> pd.DataFrame:
    """
    Compare different clustering methods.

    Args:
        X_scaled: Scaled feature matrix
        X_original: Original feature DataFrame
        kmeans_labels: K-Means cluster labels
        output_dir: Directory to save results

    Returns:
        DataFrame comparing methods
    """
    print("\n" + "="*60)
    print("COMPARING CLUSTERING METHODS")
    print("="*60)

    output_path = Path(output_dir)

    # K-Means metrics
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
    kmeans_clusters = len(set(kmeans_labels))

    print(f"\nK-Means:")
    print(f"  Clusters: {kmeans_clusters}")
    print(f"  Silhouette: {kmeans_silhouette:.3f}")
    print(f"  Davies-Bouldin: {kmeans_db:.3f}")

    # DBSCAN
    dbscan_opt = optimize_dbscan_params(X_scaled)
    if dbscan_opt['best_params']:
        dbscan, dbscan_labels = apply_dbscan(
            X_scaled, X_original,
            eps=dbscan_opt['best_params']['eps'],
            min_samples=dbscan_opt['best_params']['min_samples']
        )

        dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        dbscan_noise = list(dbscan_labels).count(-1)

        valid_mask = dbscan_labels != -1
        if dbscan_clusters > 1 and valid_mask.sum() > dbscan_clusters:
            dbscan_silhouette = silhouette_score(
                X_scaled[valid_mask],
                dbscan_labels[valid_mask]
            )
            dbscan_db = davies_bouldin_score(
                X_scaled[valid_mask],
                dbscan_labels[valid_mask]
            )
        else:
            dbscan_silhouette = np.nan
            dbscan_db = np.nan
    else:
        dbscan_clusters = 0
        dbscan_noise = 0
        dbscan_silhouette = np.nan
        dbscan_db = np.nan
        dbscan_labels = None

    # GMM
    gmm_results = determine_optimal_gmm_components(X_scaled)
    plot_gmm_selection(gmm_results, save_path=output_path / 'gmm_selection.png')

    optimal_n = gmm_results['n_components'][np.argmin(gmm_results['bic'])]
    gmm, gmm_labels, gmm_probs = apply_gmm(X_scaled, X_original, n_components=optimal_n)

    gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
    gmm_db = davies_bouldin_score(X_scaled, gmm_labels)
    gmm_bic = gmm.bic(X_scaled)

    # Create comparison table
    comparison = pd.DataFrame({
        'Method': ['K-Means', 'DBSCAN', 'GMM'],
        'N_Clusters': [kmeans_clusters, dbscan_clusters, optimal_n],
        'Silhouette': [kmeans_silhouette, dbscan_silhouette, gmm_silhouette],
        'Davies_Bouldin': [kmeans_db, dbscan_db, gmm_db],
        'Notes': [
            'Hard clustering',
            f'{dbscan_noise} outliers detected',
            f'Soft clustering (BIC={gmm_bic:.1f})'
        ]
    })

    print("\n" + "="*60)
    print("CLUSTERING METHOD COMPARISON")
    print("="*60)
    print(comparison.to_string(index=False))

    # Save results
    comparison.to_csv(output_path / 'clustering_comparison.csv', index=False)

    # Save alternative labels
    if dbscan_labels is not None:
        dbscan_df = pd.DataFrame({
            'circuit': X_original.index,
            'cluster': dbscan_labels
        })
        dbscan_df.to_csv(output_path / 'dbscan_clusters.csv', index=False)

    gmm_df = pd.DataFrame({
        'circuit': X_original.index,
        'cluster': gmm_labels
    })
    for i in range(optimal_n):
        gmm_df[f'prob_cluster_{i}'] = gmm_probs[:, i]
    gmm_df.to_csv(output_path / 'gmm_clusters.csv', index=False)

    print(f"\nSaved alternative clustering results to {output_path}")

    return comparison
