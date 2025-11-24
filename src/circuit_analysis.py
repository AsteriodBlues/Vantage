"""
Comprehensive circuit-level analysis functions.

Analyzes individual circuit characteristics, grid position dynamics,
and historical performance patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional


def create_transition_matrix(circuit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create grid position to finish position transition matrix.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        DataFrame with transition probabilities
    """
    transition_matrix = np.zeros((20, 20))
    transition_counts = np.zeros((20, 20))

    # Only use finished races
    finished_data = circuit_data[circuit_data['is_dnf'] == 0].copy()

    for _, row in finished_data.iterrows():
        grid = int(row['GridPosition']) - 1
        finish = int(row['Position']) - 1

        if 0 <= grid < 20 and 0 <= finish < 20:
            transition_counts[grid, finish] += 1

    # Normalize to probabilities
    for i in range(20):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_counts[i] / row_sum

    return pd.DataFrame(
        transition_matrix,
        index=[f'P{i+1}' for i in range(20)],
        columns=[f'P{i+1}' for i in range(20)]
    )


def calculate_grid_statistics(circuit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed statistics for each grid position.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        DataFrame with grid position statistics
    """
    stats_list = []

    for grid_pos in range(1, 21):
        grid_data = circuit_data[circuit_data['GridPosition'] == grid_pos]
        finished_data = grid_data[grid_data['is_dnf'] == 0]

        if len(finished_data) > 0:
            stats = {
                'grid_position': grid_pos,
                'sample_size': len(grid_data),
                'avg_finish': finished_data['Position'].mean(),
                'median_finish': finished_data['Position'].median(),
                'std_finish': finished_data['Position'].std(),
                'best_finish': finished_data['Position'].min(),
                'worst_finish': finished_data['Position'].max(),
                'dnf_rate': grid_data['is_dnf'].mean(),
                'win_rate': (finished_data['Position'] == 1).mean(),
                'podium_rate': (finished_data['Position'] <= 3).mean(),
                'points_rate': (finished_data['Position'] <= 10).mean(),
                'avg_position_change': (grid_pos - finished_data['Position']).mean(),
                'improvement_rate': (finished_data['Position'] < grid_pos).mean()
            }
            stats_list.append(stats)

    return pd.DataFrame(stats_list)


def analyze_position_changes(circuit_data: pd.DataFrame) -> Dict:
    """
    Analyze position change patterns.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        Dictionary with position change statistics
    """
    finished_data = circuit_data[circuit_data['is_dnf'] == 0]

    if len(finished_data) == 0:
        return {
            'mean_change': 0.0,
            'std_change': 0.0,
            'max_gain': 0,
            'max_loss': 0
        }

    position_changes = finished_data['position_change']

    return {
        'mean_change': position_changes.mean(),
        'std_change': position_changes.std(),
        'max_gain': position_changes.max(),
        'max_loss': position_changes.min(),
        'median_change': position_changes.median(),
        'gains_rate': (position_changes > 0).mean(),
        'losses_rate': (position_changes < 0).mean()
    }


def calculate_success_probabilities(circuit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win, podium, and points probabilities from each grid position.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        DataFrame with success probabilities
    """
    probabilities = {}

    for grid_pos in range(1, 21):
        grid_data = circuit_data[
            (circuit_data['GridPosition'] == grid_pos) &
            (circuit_data['is_dnf'] == 0)
        ]

        if len(grid_data) > 0:
            probabilities[f'P{grid_pos}'] = {
                'win_prob': (grid_data['Position'] == 1).mean(),
                'podium_prob': (grid_data['Position'] <= 3).mean(),
                'points_prob': (grid_data['Position'] <= 10).mean(),
                'top5_prob': (grid_data['Position'] <= 5).mean(),
                'sample_size': len(grid_data)
            }

    return pd.DataFrame(probabilities).T


def analyze_historical_trends(circuit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how circuit characteristics have changed over time.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        DataFrame with yearly trends
    """
    yearly_stats = circuit_data.groupby('year').agg({
        'Position': ['mean', 'std'],
        'position_change': ['mean', 'std'],
        'is_dnf': 'mean',
        'GridPosition': 'mean'
    }).round(3)

    # Flatten column names
    yearly_stats.columns = ['_'.join(map(str, col)).strip() for col in yearly_stats.columns]

    # Add pole win rate by year
    pole_wins_by_year = []
    for year in circuit_data['year'].unique():
        year_data = circuit_data[circuit_data['year'] == year]
        pole_data = year_data[year_data['GridPosition'] == 1]
        if len(pole_data) > 0:
            pole_win_rate = (pole_data['Position'] == 1).mean()
        else:
            pole_win_rate = np.nan
        pole_wins_by_year.append({'year': year, 'pole_win_rate': pole_win_rate})

    pole_wins_df = pd.DataFrame(pole_wins_by_year).set_index('year')
    yearly_stats = yearly_stats.join(pole_wins_df)

    # Calculate trend
    if len(yearly_stats) > 2:
        years = yearly_stats.index.values
        position_std = yearly_stats['Position_std'].values

        mask = ~np.isnan(position_std)
        if mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                years[mask], position_std[mask]
            )

            yearly_stats['overtaking_trend'] = 'increasing' if slope > 0 else 'decreasing'
            yearly_stats['trend_significance'] = p_value

    return yearly_stats


def analyze_team_performance(circuit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze team-specific performance at circuit.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        DataFrame with team performance metrics
    """
    team_stats = circuit_data.groupby('TeamName').agg({
        'Position': ['mean', 'std', 'min'],
        'GridPosition': 'mean',
        'position_change': 'mean',
        'is_dnf': 'mean',
        'Points': 'sum'
    }).round(3)

    team_stats.columns = ['_'.join(map(str, col)).strip() for col in team_stats.columns]
    team_stats = team_stats.sort_values('Position_mean')

    return team_stats


def extract_notable_statistics(circuit_data: pd.DataFrame) -> Dict:
    """
    Extract notable statistics and records for the circuit.

    Args:
        circuit_data: Race data for specific circuit

    Returns:
        Dictionary with notable stats
    """
    finished_data = circuit_data[circuit_data['is_dnf'] == 0]

    # Pole statistics
    pole_data = circuit_data[circuit_data['GridPosition'] == 1]
    pole_win_rate = (pole_data['Position'] == 1).mean() if len(pole_data) > 0 else 0.0

    # Front row statistics
    front_row_data = circuit_data[circuit_data['GridPosition'] <= 2]
    front_row_win_rate = (front_row_data['Position'] == 1).mean() if len(front_row_data) > 0 else 0.0

    # Top 3 grid
    top3_grid_data = circuit_data[circuit_data['GridPosition'] <= 3]
    top3_win_rate = (top3_grid_data['Position'] == 1).mean() if len(top3_grid_data) > 0 else 0.0

    # Winners from grid positions
    winners = finished_data[finished_data['Position'] == 1]
    lowest_winner_grid = int(winners['GridPosition'].max()) if len(winners) > 0 else 0

    # Podium from grid
    podiums = finished_data[finished_data['Position'] <= 3]
    highest_podium_grid = int(podiums['GridPosition'].max()) if len(podiums) > 0 else 0

    # Position changes
    max_positions_gained = int(finished_data['position_change'].max()) if len(finished_data) > 0 else 0

    # Overtaking score based on position variance
    overtaking_score = finished_data['Position'].std() / 5.0 if len(finished_data) > 0 else 0.0

    return {
        'pole_win_rate': pole_win_rate,
        'front_row_win_rate': front_row_win_rate,
        'top3_win_rate': top3_win_rate,
        'lowest_winner_grid': lowest_winner_grid,
        'highest_podium_grid': highest_podium_grid,
        'max_positions_gained': max_positions_gained,
        'dnf_rate': circuit_data['is_dnf'].mean(),
        'overtaking_score': overtaking_score,
        'total_finishers': len(finished_data),
        'total_dnfs': circuit_data['is_dnf'].sum()
    }


def analyze_circuit(
    df: pd.DataFrame,
    circuit_name: str,
    cluster_info: Optional[pd.DataFrame] = None
) -> Optional[Dict]:
    """
    Comprehensive analysis for a single circuit.

    Args:
        df: Full race dataset
        circuit_name: Name of circuit to analyze
        cluster_info: Optional cluster assignment data

    Returns:
        Dictionary with all circuit metrics and analysis
    """
    circuit_data = df[df['circuit'] == circuit_name].copy()

    if len(circuit_data) == 0:
        print(f"No data found for circuit: {circuit_name}")
        return None

    # Initialize results
    results = {
        'circuit_name': circuit_name,
        'total_races': len(circuit_data['year'].unique()),
        'total_entries': len(circuit_data),
        'years_present': sorted(circuit_data['year'].unique().tolist())
    }

    # Run all analyses
    results['transition_matrix'] = create_transition_matrix(circuit_data)
    results['grid_statistics'] = calculate_grid_statistics(circuit_data)
    results['position_change_stats'] = analyze_position_changes(circuit_data)
    results['success_probabilities'] = calculate_success_probabilities(circuit_data)
    results['historical_trends'] = analyze_historical_trends(circuit_data)
    results['team_performance'] = analyze_team_performance(circuit_data)
    results['notable_stats'] = extract_notable_statistics(circuit_data)

    # Add cluster info if available
    if cluster_info is not None:
        circuit_encoded = df[df['circuit'] == circuit_name]['circuit_encoded'].iloc[0]
        cluster_row = cluster_info[cluster_info['circuit'] == circuit_encoded]
        if len(cluster_row) > 0:
            results['notable_stats']['cluster_name'] = cluster_row['cluster_name'].iloc[0]
            results['notable_stats']['cluster_id'] = int(cluster_row['cluster'].iloc[0])

    return results
