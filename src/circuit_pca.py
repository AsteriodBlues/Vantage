"""
PCA dimensionality reduction for circuit similarity visualization.

Creates 2D and 3D visualizations of circuit relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional
from pathlib import Path


def apply_pca(
    X_scaled: np.ndarray,
    n_components: int = 3
) -> Tuple[PCA, np.ndarray]:
    """
    Apply PCA to scaled features.

    Args:
        X_scaled: Scaled feature matrix
        n_components: Number of components

    Returns:
        Tuple of (fitted_pca, transformed_data)
    """
    print(f"Applying PCA with {n_components} components...")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Print variance explained
    print(f"\nVariance explained by components:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.1%}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    return pca, X_pca


def analyze_component_loadings(
    pca: PCA,
    feature_names: list,
    n_top: int = 5
) -> pd.DataFrame:
    """
    Analyze what features drive each principal component.

    Args:
        pca: Fitted PCA object
        feature_names: List of feature names
        n_top: Number of top features to show per component

    Returns:
        DataFrame with component loadings
    """
    print("\nAnalyzing component loadings...")

    n_components = pca.n_components_
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )

    print("\nTop features per component:")
    for i in range(n_components):
        pc_name = f'PC{i+1}'
        top_features = loadings[pc_name].abs().nlargest(n_top)

        print(f"\n{pc_name} (explains {pca.explained_variance_ratio_[i]:.1%}):")
        for feat, loading in top_features.items():
            direction = "+" if loadings.loc[feat, pc_name] > 0 else "-"
            print(f"  {direction} {feat}: {abs(loading):.3f}")

    return loadings


def plot_pca_2d(
    X_pca: np.ndarray,
    circuit_names: list,
    cluster_labels: Optional[np.ndarray],
    pca_model: PCA,
    save_path: str = None
) -> None:
    """
    Create 2D PCA visualization.

    Args:
        X_pca: PCA-transformed data
        circuit_names: Circuit names
        cluster_labels: Cluster assignments (optional)
        pca_model: Fitted PCA model
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use cluster colors if available
    if cluster_labels is not None:
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap='viridis',
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )
        plt.colorbar(scatter, label='Cluster', ax=ax)

        # Draw cluster ellipses
        for cluster in np.unique(cluster_labels):
            cluster_points = X_pca[cluster_labels == cluster, :2]
            if len(cluster_points) > 2:
                cov = np.cov(cluster_points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                ellipse = Ellipse(
                    cluster_points.mean(axis=0),
                    2 * np.sqrt(eigenvalues[0]),
                    2 * np.sqrt(eigenvalues[1]),
                    angle=angle,
                    alpha=0.2,
                    facecolor=plt.cm.viridis(cluster / np.max(cluster_labels))
                )
                ax.add_patch(ellipse)
    else:
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )

    # Add circuit labels
    for i, circuit in enumerate(circuit_names):
        ax.annotate(
            circuit,
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=9,
            ha='center',
            va='center',
            fontweight='bold'
        )

    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)',
                  fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)',
                  fontsize=11)
    ax.set_title('Circuit Similarity Map (PCA)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D PCA plot to {save_path}")

    plt.close()


def plot_pca_3d(
    X_pca: np.ndarray,
    circuit_names: list,
    cluster_labels: Optional[np.ndarray],
    pca_model: PCA,
    save_path: str = None
) -> None:
    """
    Create 3D PCA visualization.

    Args:
        X_pca: PCA-transformed data (must have 3+ components)
        circuit_names: Circuit names
        cluster_labels: Cluster assignments (optional)
        pca_model: Fitted PCA model
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Use cluster colors if available
    if cluster_labels is not None:
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            X_pca[:, 2],
            c=cluster_labels,
            cmap='viridis',
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )
        plt.colorbar(scatter, label='Cluster', ax=ax, shrink=0.7)
    else:
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            X_pca[:, 2],
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5
        )

    # Add circuit labels
    for i, circuit in enumerate(circuit_names):
        ax.text(
            X_pca[i, 0],
            X_pca[i, 1],
            X_pca[i, 2],
            circuit,
            fontsize=8,
            ha='center'
        )

    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]:.1%})', fontsize=10)
    ax.set_title('3D Circuit Similarity Space', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D PCA plot to {save_path}")

    plt.close()


def plot_loading_vectors(
    pca: PCA,
    feature_names: list,
    save_path: str = None
) -> None:
    """
    Create PCA loading plot showing feature contributions.

    Args:
        pca: Fitted PCA model
        feature_names: List of feature names
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot loading vectors
    for i, feature in enumerate(feature_names):
        ax.arrow(
            0, 0,
            pca.components_[0, i] * 3,
            pca.components_[1, i] * 3,
            head_width=0.08,
            head_length=0.08,
            fc='red',
            ec='red',
            alpha=0.6
        )
        ax.text(
            pca.components_[0, i] * 3.2,
            pca.components_[1, i] * 3.2,
            feature.replace('_', ' '),
            fontsize=8,
            ha='center',
            va='center'
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f'PC1 Loading ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 Loading ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('PCA Loading Plot', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loading plot to {save_path}")

    plt.close()


def plot_scree(
    pca: PCA,
    save_path: str = None
) -> None:
    """
    Create scree plot showing variance explained.

    Args:
        pca: Fitted PCA model
        save_path: Path to save figure
    """
    n_components = len(pca.explained_variance_ratio_)
    components = range(1, n_components + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Individual variance
    axes[0].bar(components, pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=10)
    axes[0].set_ylabel('Variance Explained', fontsize=10)
    axes[0].set_title('Scree Plot', fontsize=11, fontweight='bold')
    axes[0].set_xticks(components)
    axes[0].grid(axis='y', alpha=0.3)

    # Cumulative variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(components, cumsum_var, 'o-', linewidth=2, color='green')
    axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    axes[1].set_xlabel('Principal Component', fontsize=10)
    axes[1].set_ylabel('Cumulative Variance Explained', fontsize=10)
    axes[1].set_title('Cumulative Variance', fontsize=11, fontweight='bold')
    axes[1].set_xticks(components)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scree plot to {save_path}")

    plt.close()


def run_pca_analysis(
    X_scaled: np.ndarray,
    X_original: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
    output_dir: str = 'results/clustering'
) -> Dict:
    """
    Complete PCA analysis pipeline.

    Args:
        X_scaled: Scaled feature matrix
        X_original: Original unscaled DataFrame
        cluster_labels: Cluster assignments (optional)
        output_dir: Directory to save results

    Returns:
        Dictionary with PCA results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PCA DIMENSIONALITY REDUCTION")
    print("="*60)

    # Apply PCA
    pca_model, X_pca = apply_pca(X_scaled, n_components=3)

    # Analyze loadings
    loadings = analyze_component_loadings(
        pca_model,
        X_original.columns.tolist(),
        n_top=5
    )

    # Save loadings
    loadings.to_csv(output_path / 'pca_loadings.csv')

    # Create visualizations
    plot_scree(pca_model, save_path=output_path / 'scree_plot.png')

    plot_pca_2d(
        X_pca,
        X_original.index.tolist(),
        cluster_labels,
        pca_model,
        save_path=output_path / 'pca_2d.png'
    )

    plot_pca_3d(
        X_pca,
        X_original.index.tolist(),
        cluster_labels,
        pca_model,
        save_path=output_path / 'pca_3d.png'
    )

    plot_loading_vectors(
        pca_model,
        X_original.columns.tolist(),
        save_path=output_path / 'pca_loadings_plot.png'
    )

    print(f"\nSaved PCA results to {output_path}")

    return {
        'pca_model': pca_model,
        'X_pca': X_pca,
        'loadings': loadings
    }
