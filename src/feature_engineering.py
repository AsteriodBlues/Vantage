"""
Feature engineering functions for F1 race prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical circuit features from race data.

    Features include:
    - Overtaking difficulty index
    - Historical pole win rate
    - Average position change
    - Position change variance
    - DNF rate per circuit
    - Grid-finish correlation
    """
    circuit_stats = []

    for circuit in df['circuit'].unique():
        circuit_data = df[df['circuit'] == circuit].copy()

        # only look at finished races for correlation
        finished = circuit_data[circuit_data['completed_race'] == True]

        total_entries = len(circuit_data)
        finished_entries = len(finished)

        if finished_entries == 0:
            continue

        # position change stats
        avg_pos_change = finished['position_change'].mean()
        std_pos_change = finished['position_change'].std()
        var_pos_change = finished['position_change'].var()

        # pole win rate
        pole_starts = circuit_data[circuit_data['GridPosition'] == 1]
        pole_wins = pole_starts[pole_starts['Position'] == 1]
        pole_win_rate = len(pole_wins) / len(pole_starts) * 100 if len(pole_starts) > 0 else 0

        # top 3 grid win rate
        top3_starts = circuit_data[circuit_data['GridPosition'] <= 3]
        top3_wins = top3_starts[top3_starts['Position'] == 1]
        top3_win_rate = len(top3_wins) / len(top3_starts) * 100 if len(top3_starts) > 0 else 0

        # grid-finish correlation
        if len(finished) > 5:
            correlation = finished['GridPosition'].corr(finished['Position'])
        else:
            correlation = np.nan

        # dnf rate
        dnf_rate = circuit_data['is_dnf'].sum() / total_entries * 100

        # improved percentage (gained positions)
        improved = (finished['position_change'] > 0).sum()
        improved_pct = improved / finished_entries * 100

        # calculate overtaking difficulty index (0-100, higher = harder to overtake)
        # combines correlation, pole win rate, and position variance
        if not np.isnan(correlation):
            corr_component = correlation * 30  # 0-30
            pole_component = pole_win_rate * 0.3  # 0-30
            var_component = max(0, 40 - var_pos_change)  # 0-40 (lower variance = harder)
            overtaking_index = corr_component + pole_component + var_component
            overtaking_index = np.clip(overtaking_index, 0, 100)
        else:
            overtaking_index = 50  # default middle value

        circuit_stats.append({
            'circuit': circuit,
            'circuit_pole_win_rate': pole_win_rate,
            'circuit_top3_win_rate': top3_win_rate,
            'circuit_avg_pos_change': avg_pos_change,
            'circuit_std_pos_change': std_pos_change,
            'circuit_var_pos_change': var_pos_change,
            'circuit_correlation': correlation,
            'circuit_dnf_rate': dnf_rate,
            'circuit_improved_pct': improved_pct,
            'overtaking_difficulty_index': overtaking_index
        })

    return pd.DataFrame(circuit_stats)


def create_grid_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grid position derived features.

    Features include:
    - Polynomial terms (squared, cubed, log)
    - Binary indicators (front row, top 3, top 10)
    - Grid side (clean/dirty line)
    """
    df = df.copy()

    # polynomial features
    df['grid_squared'] = df['GridPosition'] ** 2
    df['grid_cubed'] = df['GridPosition'] ** 3
    df['grid_log'] = np.log(df['GridPosition'].clip(lower=1))
    df['grid_sqrt'] = np.sqrt(df['GridPosition'])

    # binary indicators
    df['front_row'] = (df['GridPosition'] <= 2).astype(int)
    df['top_three'] = (df['GridPosition'] <= 3).astype(int)
    df['top_five'] = (df['GridPosition'] <= 5).astype(int)
    df['top_ten'] = (df['GridPosition'] <= 10).astype(int)
    df['back_half'] = (df['GridPosition'] > 10).astype(int)

    # grid side (odd = clean/racing line, even = dirty/off-line)
    df['grid_side'] = df['GridPosition'].apply(lambda x: 'clean' if x % 2 == 1 else 'dirty')
    df['grid_side_clean'] = (df['GridPosition'] % 2 == 1).astype(int)

    # grid row (pair positions together)
    df['grid_row'] = ((df['GridPosition'] - 1) // 2) + 1

    return df


def load_circuit_info(filepath: str = 'data/raw/circuit_info.csv') -> pd.DataFrame:
    """
    Load external circuit characteristics data.
    """
    circuit_info = pd.read_csv(filepath)

    # create additional derived features
    circuit_info['high_downforce'] = (circuit_info['downforce_level'] == 'high').astype(int)
    circuit_info['low_downforce'] = (circuit_info['downforce_level'] == 'low').astype(int)
    circuit_info['is_street'] = (circuit_info['circuit_type'] == 'street').astype(int)
    circuit_info['high_altitude'] = (circuit_info['altitude_m'] > 500).astype(int)
    circuit_info['many_drs_zones'] = (circuit_info['drs_zones'] >= 3).astype(int)
    circuit_info['long_straight'] = (circuit_info['longest_straight_m'] > 1000).astype(int)

    return circuit_info


def merge_features(race_data: pd.DataFrame,
                   circuit_features: pd.DataFrame,
                   circuit_info: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all circuit and grid features with main race data.
    """
    # merge historical circuit stats
    merged = race_data.merge(circuit_features, on='circuit', how='left')

    # merge external circuit info
    merged = merged.merge(circuit_info, on='circuit', how='left')

    return merged


def create_all_features(race_data_path: str = 'data/processed/processed_race_data.csv',
                        circuit_info_path: str = 'data/raw/circuit_info.csv',
                        output_path: str = 'data/processed/race_data_with_features.csv') -> pd.DataFrame:
    """
    Main function to create all features and save processed dataset.
    """
    # load data
    print("Loading race data...")
    race_data = pd.read_csv(race_data_path)

    print("Loading circuit info...")
    circuit_info = load_circuit_info(circuit_info_path)

    # calculate historical circuit features
    print("Calculating circuit features...")
    circuit_features = calculate_circuit_features(race_data)

    # create grid position features
    print("Creating grid position features...")
    race_data = create_grid_position_features(race_data)

    # merge everything
    print("Merging features...")
    final_data = merge_features(race_data, circuit_features, circuit_info)

    # save output
    print(f"Saving to {output_path}...")
    final_data.to_csv(output_path, index=False)

    print(f"Done! Created {len(final_data.columns)} features for {len(final_data)} records")

    return final_data


if __name__ == '__main__':
    df = create_all_features()
    print("\nFeature summary:")
    print(f"Total columns: {len(df.columns)}")
    print(f"New grid features: 12")
    print(f"Circuit features: 10")
    print(f"External circuit info: 12")
