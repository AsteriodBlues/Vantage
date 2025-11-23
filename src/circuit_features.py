"""
Circuit feature matrix preparation for clustering analysis.

Aggregates circuit-level statistics and characteristics for similarity analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_circuit_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate circuit-level performance statistics.

    Args:
        df: Race data with features

    Returns:
        DataFrame with one row per circuit containing aggregated stats
    """
    print("Calculating circuit statistics...")

    # Create helper columns for aggregations
    df['is_pole_win'] = ((df['GridPosition'] == 1) & (df['Position'] == 1)).astype(int)
    df['is_dnf'] = (df['Position'] > 20).astype(int)
    df['is_points'] = (df['Position'] <= 10).astype(int)
    df['position_change'] = df['GridPosition'] - df['Position']

    # Aggregate by circuit
    circuit_stats = df.groupby('circuit').agg({
        # Performance metrics
        'Position': ['mean', 'std', 'count'],
        'position_change': ['mean', 'std', 'min', 'max'],
        'GridPosition': 'mean',

        # Race characteristics
        'is_dnf': 'mean',
        'is_pole_win': 'mean',
        'is_points': 'mean',

        # Years of data
        'year_first_raced': 'first'
    }).round(3)

    # Flatten column names
    circuit_stats.columns = ['_'.join(str(col)).strip('_') for col in circuit_stats.columns]

    # Rename for clarity
    circuit_stats.rename(columns={
        'Position_mean': 'avg_finish_position',
        'Position_std': 'position_std',
        'Position_count': 'n_races',
        'position_change_mean': 'avg_position_change',
        'position_change_std': 'position_change_std',
        'position_change_min': 'max_positions_lost',
        'position_change_max': 'max_positions_gained',
        'GridPosition_mean': 'avg_grid_position',
        'is_dnf_mean': 'dnf_rate',
        'is_pole_win_mean': 'pole_win_rate',
        'is_points_mean': 'points_rate',
        'year_first_raced_first': 'first_year'
    }, inplace=True)

    print(f"Calculated statistics for {len(circuit_stats)} circuits")

    return circuit_stats


def calculate_grid_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate grid position advantage metrics per circuit.

    Args:
        df: Race data

    Returns:
        DataFrame with grid-based metrics
    """
    grid_metrics = {}

    for circuit in df['circuit'].unique():
        circuit_data = df[df['circuit'] == circuit]

        # Front row dominance
        front_row = circuit_data[circuit_data['GridPosition'] <= 2]
        if len(front_row) > 0:
            front_row_avg = front_row['Position'].mean()
        else:
            front_row_avg = np.nan

        # Pole advantage
        pole = circuit_data[circuit_data['GridPosition'] == 1]
        if len(pole) > 0:
            pole_avg = pole['Position'].mean()
        else:
            pole_avg = np.nan

        # Midfield volatility
        midfield = circuit_data[
            (circuit_data['GridPosition'] >= 7) &
            (circuit_data['GridPosition'] <= 14)
        ]
        if len(midfield) > 5:
            midfield_volatility = midfield['position_change'].std()
        else:
            midfield_volatility = np.nan

        # Points from back
        back_grid = circuit_data[circuit_data['GridPosition'] >= 15]
        if len(back_grid) > 0:
            points_from_back = (back_grid['Position'] <= 10).mean()
        else:
            points_from_back = np.nan

        grid_metrics[circuit] = {
            'front_row_dominance': front_row_avg,
            'pole_advantage': pole_avg,
            'midfield_volatility': midfield_volatility,
            'points_from_back': points_from_back
        }

    return pd.DataFrame(grid_metrics).T


def calculate_derived_features(circuit_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived clustering features from base statistics.

    Args:
        circuit_stats: Base circuit statistics

    Returns:
        DataFrame with additional derived features
    """
    print("Calculating derived features...")

    # Overtaking metrics
    circuit_stats['overtaking_score'] = circuit_stats['position_change_std']
    circuit_stats['processional_score'] = 1 / (circuit_stats['position_change_std'] + 0.1)

    # Chaos factor
    circuit_stats['chaos_factor'] = (
        circuit_stats['position_std'] /
        (circuit_stats['avg_finish_position'] + 1)
    )

    # Predictability
    circuit_stats['predictability'] = 1 / (circuit_stats['position_std'] + 0.1)

    return circuit_stats


def merge_circuit_info(
    circuit_stats: pd.DataFrame,
    circuit_info_path: str = 'data/processed/circuit_statistics.csv'
) -> pd.DataFrame:
    """
    Merge calculated statistics with external circuit information.

    Args:
        circuit_stats: Calculated statistics
        circuit_info_path: Path to circuit info CSV

    Returns:
        Merged DataFrame
    """
    print(f"Loading circuit info from {circuit_info_path}...")

    if not Path(circuit_info_path).exists():
        print(f"Warning: {circuit_info_path} not found, using only calculated stats")
        return circuit_stats

    circuit_info = pd.read_csv(circuit_info_path)

    # Ensure circuit names match
    circuit_info = circuit_info.rename(columns={'circuit': 'circuit_name'})

    # Merge
    merged = circuit_stats.merge(
        circuit_info,
        left_index=True,
        right_on='circuit_name',
        how='left'
    )

    print(f"Merged data shape: {merged.shape}")

    return merged


def prepare_clustering_features(
    circuit_features: pd.DataFrame,
    feature_list: List[str] = None
) -> pd.DataFrame:
    """
    Select and prepare features for clustering.

    Args:
        circuit_features: Full circuit feature DataFrame
        feature_list: List of features to use (None for auto-select)

    Returns:
        DataFrame ready for clustering
    """
    if feature_list is None:
        # Auto-select numerical features
        feature_list = [
            # Performance characteristics
            'overtaking_score', 'processional_score', 'chaos_factor',
            'pole_win_rate', 'avg_position_change', 'position_change_std',
            'dnf_rate', 'front_row_dominance', 'midfield_volatility',
            'points_from_back', 'predictability',

            # Physical characteristics (if available)
            'track_length_km', 'num_turns', 'longest_straight_m',
            'altitude_m', 'drs_zones'
        ]

    # Select available features
    available_features = [f for f in feature_list if f in circuit_features.columns]

    print(f"Using {len(available_features)} features for clustering:")
    for feat in available_features:
        print(f"  - {feat}")

    # Create feature matrix
    X = circuit_features[available_features].copy()

    # Fill missing values with column mean
    X = X.fillna(X.mean())

    # Remove any remaining NaN
    X = X.dropna()

    print(f"Clustering feature matrix: {X.shape}")

    return X


def create_circuit_feature_matrix(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to create circuit feature matrix.

    Args:
        data_path: Path to processed race data

    Returns:
        Tuple of (full_features_df, clustering_features_df)
    """
    print("="*60)
    print("CIRCUIT FEATURE MATRIX CREATION")
    print("="*60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} race results")

    # Calculate base statistics
    circuit_stats = calculate_circuit_statistics(df)

    # Calculate grid metrics
    grid_metrics = calculate_grid_metrics(df)
    circuit_stats = circuit_stats.join(grid_metrics, how='left')

    # Calculate derived features
    circuit_stats = calculate_derived_features(circuit_stats)

    # Merge with external circuit info
    circuit_features = merge_circuit_info(circuit_stats)

    # Prepare clustering features
    clustering_features = prepare_clustering_features(circuit_features)

    # Save
    output_dir = Path('results/clustering')
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit_features.to_csv(output_dir / 'circuit_features.csv')
    clustering_features.to_csv(output_dir / 'clustering_features.csv')

    print(f"\nSaved circuit features to {output_dir}")
    print(f"Full features: {circuit_features.shape}")
    print(f"Clustering features: {clustering_features.shape}")

    return circuit_features, clustering_features


if __name__ == "__main__":
    # Run feature creation
    circuit_features, clustering_features = create_circuit_feature_matrix(
        'data/processed/train.csv'
    )

    print("\nCircuit feature creation complete!")
