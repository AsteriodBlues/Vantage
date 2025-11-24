"""
Build comprehensive feature database for predictions.
Extracts all necessary lookup tables from training data.
"""

import pandas as pd
import joblib
from pathlib import Path


def main():
    """Build and save feature lookup tables."""
    # load training data
    train = pd.read_csv('data/processed/train.csv')
    print(f"Loading {len(train)} training samples...")

    # circuit statistics
    circuit_features = {}
    for circuit in train['circuit'].unique():
        circuit_data = train[train['circuit'] == circuit].iloc[0]
        circuit_features[circuit] = {
            'circuit_pole_win_rate': circuit_data.get('circuit_pole_win_rate', 50.0),
            'circuit_top3_win_rate': circuit_data.get('circuit_top3_win_rate', 50.0),
            'circuit_avg_pos_change': circuit_data.get('circuit_avg_pos_change', 0.0),
            'circuit_std_pos_change': circuit_data.get('circuit_std_pos_change', 2.0),
            'circuit_var_pos_change': circuit_data.get('circuit_var_pos_change', 4.0),
            'circuit_correlation': circuit_data.get('circuit_correlation', 0.7),
            'circuit_dnf_rate': circuit_data.get('circuit_dnf_rate', 15.0),
            'circuit_improved_pct': circuit_data.get('circuit_improved_pct', 40.0),
            'overtaking_difficulty_index': circuit_data.get('overtaking_difficulty_index', 50.0),
            'track_length_km': circuit_data.get('track_length_km', 5.0),
            'num_turns': circuit_data.get('num_turns', 15),
            'circuit_type': circuit_data.get('circuit_type', 1),
            'downforce_level': circuit_data.get('downforce_level', 2),
            'drs_zones': circuit_data.get('drs_zones', 2),
            'longest_straight_m': circuit_data.get('longest_straight_m', 500),
            'year_first_raced': circuit_data.get('year_first_raced', 2000),
            'altitude_m': circuit_data.get('altitude_m', 100),
            'typical_lap_time_s': circuit_data.get('typical_lap_time_s', 90),
            'direction': circuit_data.get('direction', 1),
            'high_downforce': circuit_data.get('high_downforce', 0),
            'low_downforce': circuit_data.get('low_downforce', 0),
            'is_street': circuit_data.get('is_street', 0),
            'high_altitude': circuit_data.get('high_altitude', 0),
            'many_drs_zones': circuit_data.get('many_drs_zones', 0),
            'long_straight': circuit_data.get('long_straight', 0),
            'downforce_ordinal': circuit_data.get('downforce_ordinal', 1),
            'circuit_freq': circuit_data.get('circuit_freq', 20),
            'circuit_target_enc': circuit_data.get('circuit_target_enc', 10.0),
            'circuit_encoded': circuit_data.get('circuit_encoded', 0),
        }

    # team statistics
    team_features = {}
    for team in train['TeamName'].unique():
        team_data = train[train['TeamName'] == team].iloc[-5:]  # last 5 races

        team_features[team] = {
            'avg_finish': team_data['Position_raw'].mean(),
            'avg_points': team_data.get('team_points_last_5', pd.Series([5.0])).mean(),
            'consistency': team_data['Position_raw'].std(),
            'win_rate': (team_data['Position_raw'] == 1).sum() / len(team_data) * 100,
            'podium_rate': (team_data['Position_raw'] <= 3).sum() / len(team_data) * 100,
            'points_rate': (team_data['Position_raw'] <= 10).sum() / len(team_data) * 100,
            'momentum': team_data.get('team_momentum', pd.Series([0.0])).mean(),
            'TeamName_freq': train[train['TeamName'] == team]['TeamName_freq'].iloc[0] if 'TeamName_freq' in train.columns else 20,
            'TeamName_target_enc': train[train['TeamName'] == team]['TeamName_target_enc'].iloc[0] if 'TeamName_target_enc' in train.columns else 10.0,
            'TeamName_encoded': train[train['TeamName'] == team]['TeamName_encoded'].iloc[0] if 'TeamName_encoded' in train.columns else 0,
        }

    # driver statistics (aggregate from training data)
    driver_features = {}
    for driver in train['DriverId'].unique():
        driver_data = train[train['DriverId'] == driver]

        driver_features[driver] = {
            'career_races': len(driver_data),
            'career_wins': (driver_data['Position_raw'] == 1).sum(),
            'career_podiums': (driver_data['Position_raw'] <= 3).sum(),
            'years_experience': driver_data.get('driver_years_experience', pd.Series([5])).iloc[-1],
            'avg_finish': driver_data['Position_raw'].mean(),
            'DriverId_freq': driver_data['DriverId_freq'].iloc[0] if 'DriverId_freq' in driver_data.columns else 20,
            'DriverId_target_enc': driver_data['DriverId_target_enc'].iloc[0] if 'DriverId_target_enc' in driver_data.columns else 10.0,
            'DriverId_encoded': driver_data['DriverId_encoded'].iloc[0] if 'DriverId_encoded' in driver_data.columns else 0,
        }

    # save all lookup tables
    output_dir = Path('models/preprocessing')
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(circuit_features, output_dir / 'circuit_statistics.pkl')
    joblib.dump(team_features, output_dir / 'team_baselines.pkl')
    joblib.dump(driver_features, output_dir / 'driver_statistics.pkl')

    print(f"\nSaved feature databases:")
    print(f"  - {len(circuit_features)} circuits")
    print(f"  - {len(team_features)} teams")
    print(f"  - {len(driver_features)} drivers")

    # also save mean values for unknown entities
    defaults = {
        'circuit_default': {k: 50.0 if 'rate' in k else 0.0 for k in circuit_features[list(circuit_features.keys())[0]].keys()},
        'team_default': {k: 10.0 if 'avg' in k else 0.0 for k in team_features[list(team_features.keys())[0]].keys()},
        'driver_default': {k: 5.0 if 'avg' in k or 'experience' in k else 0.0 for k in driver_features[list(driver_features.keys())[0]].keys()}
    }
    joblib.dump(defaults, output_dir / 'feature_defaults.pkl')
    print(f"  - Default values for unknown entities")


if __name__ == '__main__':
    main()
