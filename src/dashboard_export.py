"""
Export circuit analysis data for dashboard and visualization tools.

Prepares consolidated data structures for interactive dashboards.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict


def export_dashboard_data(all_analyses: Dict, output_path: str = 'results/data/dashboard_circuit_data.json'):
    """
    Export circuit analysis data in dashboard-friendly format.

    Args:
        all_analyses: Dictionary of all circuit analyses
        output_path: Path to save JSON file
    """
    dashboard_data = {}

    for circuit, analysis in all_analyses.items():
        dashboard_data[circuit] = {
            # Basic info
            'name': circuit,
            'cluster': analysis['notable_stats'].get('cluster_name', 'Unknown'),
            'cluster_id': int(analysis['notable_stats'].get('cluster_id', -1)),
            'total_races': int(analysis['total_races']),
            'years': analysis['years_present'],

            # Quick stats for display
            'quick_stats': {
                'pole_win_rate': float(analysis['notable_stats']['pole_win_rate']),
                'front_row_win_rate': float(analysis['notable_stats']['front_row_win_rate']),
                'overtaking_score': float(analysis['notable_stats']['overtaking_score']),
                'avg_position_change': float(analysis['position_change_stats']['mean_change']),
                'position_std': float(analysis['position_change_stats']['std_change']),
                'dnf_rate': float(analysis['notable_stats']['dnf_rate']),
                'lowest_winner_grid': int(analysis['notable_stats']['lowest_winner_grid']),
                'highest_podium_grid': int(analysis['notable_stats']['highest_podium_grid']),
                'max_positions_gained': int(analysis['notable_stats']['max_positions_gained'])
            },

            # Grid statistics
            'grid_data': analysis['grid_statistics'].to_dict('records'),

            # Transition matrix
            'transition_matrix': analysis['transition_matrix'].values.tolist(),

            # Success probabilities
            'success_probs': analysis['success_probabilities'].to_dict('records'),

            # Historical trends
            'trends': analysis['historical_trends'].reset_index().to_dict('records') if len(analysis['historical_trends']) > 0 else []
        }

    # Save to JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Exported dashboard data to {output_path}")
    print(f"  Circuits: {len(dashboard_data)}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return dashboard_data


def export_circuit_metadata(all_analyses: Dict, output_path: str = 'results/data/circuit_metadata.csv'):
    """
    Export circuit metadata as CSV for easy reference.

    Args:
        all_analyses: Dictionary of all circuit analyses
        output_path: Path to save CSV file
    """
    metadata = []

    for circuit, analysis in all_analyses.items():
        row = {
            'circuit': circuit,
            'cluster': analysis['notable_stats'].get('cluster_name', 'Unknown'),
            'total_races': analysis['total_races'],
            'first_year': min(analysis['years_present']),
            'last_year': max(analysis['years_present']),
            'pole_win_rate': analysis['notable_stats']['pole_win_rate'],
            'overtaking_score': analysis['notable_stats']['overtaking_score'],
            'position_std': analysis['position_change_stats']['std_change'],
            'dnf_rate': analysis['notable_stats']['dnf_rate']
        }
        metadata.append(row)

    metadata_df = pd.DataFrame(metadata)
    metadata_df = metadata_df.sort_values('circuit')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"Exported metadata to {output_path}")

    return metadata_df
