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


def create_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team performance features.

    Includes season averages, performance delta, and reliability metrics.
    Uses cumulative stats to avoid data leakage.
    """
    df = df.copy()
    df = df.sort_values(['year', 'round', 'TeamName']).reset_index(drop=True)

    # group by team and year for season stats
    team_season_stats = []

    for (team, year), group in df.groupby(['TeamName', 'year']):
        group = group.sort_values('round')

        # cumulative averages (only using past races)
        cum_grid = group['GridPosition'].expanding().mean().shift(1)
        cum_finish = group['Position'].expanding().mean().shift(1)
        cum_dnf_rate = group['is_dnf'].expanding().mean().shift(1) * 100

        # performance delta (positive = gaining positions on average)
        cum_delta = cum_grid - cum_finish

        for idx, row in group.iterrows():
            race_num = row['round']
            team_season_stats.append({
                'idx': idx,
                'team_avg_grid': cum_grid.loc[idx] if not pd.isna(cum_grid.loc[idx]) else row['GridPosition'],
                'team_avg_finish': cum_finish.loc[idx] if not pd.isna(cum_finish.loc[idx]) else row['Position'],
                'team_performance_delta': cum_delta.loc[idx] if not pd.isna(cum_delta.loc[idx]) else 0,
                'team_dnf_rate': cum_dnf_rate.loc[idx] if not pd.isna(cum_dnf_rate.loc[idx]) else 0
            })

    team_df = pd.DataFrame(team_season_stats).set_index('idx')

    for col in ['team_avg_grid', 'team_avg_finish', 'team_performance_delta', 'team_dnf_rate']:
        df[col] = team_df[col]

    return df


def create_team_rolling_features(df: pd.DataFrame, windows: list = [3, 5]) -> pd.DataFrame:
    """
    Create rolling window features for team momentum.

    Calculates last N race averages for finish position and position change.
    """
    df = df.copy()
    df = df.sort_values(['TeamName', 'year', 'round']).reset_index(drop=True)

    for window in windows:
        finish_col = f'team_last{window}_avg_finish'
        change_col = f'team_last{window}_avg_change'

        df[finish_col] = np.nan
        df[change_col] = np.nan

        for team in df['TeamName'].unique():
            team_mask = df['TeamName'] == team
            team_data = df[team_mask].copy()

            # rolling mean with shift to avoid leakage
            rolling_finish = team_data['Position'].rolling(window=window, min_periods=1).mean().shift(1)
            rolling_change = team_data['position_change'].rolling(window=window, min_periods=1).mean().shift(1)

            df.loc[team_mask, finish_col] = rolling_finish.values
            df.loc[team_mask, change_col] = rolling_change.values

        # fill first race with season average or default
        df[finish_col] = df[finish_col].fillna(df['Position'])
        df[change_col] = df[change_col].fillna(0)

    # team form trend (slope of last 5 finishes)
    df['team_form_trend'] = 0.0

    for team in df['TeamName'].unique():
        team_mask = df['TeamName'] == team
        team_data = df[team_mask].copy()

        trends = []
        positions = team_data['Position'].tolist()

        for i in range(len(positions)):
            if i < 2:
                trends.append(0)
            else:
                # simple trend: compare last position to average of previous
                window_size = min(5, i)
                recent = positions[max(0, i-window_size):i]
                if len(recent) >= 2:
                    # negative trend = improving (lower positions are better)
                    trend = (recent[-1] - recent[0]) / len(recent)
                    trends.append(trend)
                else:
                    trends.append(0)

        df.loc[team_mask, 'team_form_trend'] = trends

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.

    Includes race number in season, season progress, and era indicators.
    """
    df = df.copy()

    # race number in season (already have 'round')
    df['race_number'] = df['round']

    # season progress (0-1 scale)
    max_rounds = df.groupby('year')['round'].transform('max')
    df['season_progress'] = df['round'] / max_rounds

    # era indicator (2022 regulation change)
    df['post_2022'] = (df['year'] >= 2022).astype(int)

    # early/mid/late season
    df['early_season'] = (df['season_progress'] <= 0.33).astype(int)
    df['mid_season'] = ((df['season_progress'] > 0.33) & (df['season_progress'] <= 0.66)).astype(int)
    df['late_season'] = (df['season_progress'] > 0.66).astype(int)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between key features.

    These capture non-linear relationships that tree models might miss.
    """
    df = df.copy()

    # grid position interactions
    if 'overtaking_difficulty_index' in df.columns:
        df['grid_x_overtaking'] = df['GridPosition'] * df['overtaking_difficulty_index']

    if 'team_performance_delta' in df.columns:
        df['grid_x_team_delta'] = df['GridPosition'] * df['team_performance_delta']

    # top 3 interactions with circuit type
    if 'is_street' in df.columns:
        df['top3_x_street'] = df['top_three'] * df['is_street']

    if 'high_downforce' in df.columns:
        df['top3_x_high_df'] = df['top_three'] * df['high_downforce']

    # front row advantage on processional tracks
    if 'circuit_correlation' in df.columns:
        df['frontrow_x_correlation'] = df['front_row'] * df['circuit_correlation']

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

    # create team performance features
    print("Creating team performance features...")
    race_data = create_team_features(race_data)

    # create team rolling/momentum features
    print("Creating team rolling features...")
    race_data = create_team_rolling_features(race_data)

    # create temporal features
    print("Creating temporal features...")
    race_data = create_temporal_features(race_data)

    # merge circuit features
    print("Merging circuit features...")
    final_data = merge_features(race_data, circuit_features, circuit_info)

    # create interaction features (after merge so we have all columns)
    print("Creating interaction features...")
    final_data = create_interaction_features(final_data)

    # save output
    print(f"Saving to {output_path}...")
    final_data.to_csv(output_path, index=False)

    print(f"Done! Created {len(final_data.columns)} features for {len(final_data)} records")

    return final_data


if __name__ == '__main__':
    df = create_all_features()
    print("\nFeature summary:")
    print(f"Total columns: {len(df.columns)}")

    # count feature categories
    grid_features = ['grid_squared', 'grid_cubed', 'grid_log', 'grid_sqrt', 'front_row',
                     'top_three', 'top_five', 'top_ten', 'back_half', 'grid_side',
                     'grid_side_clean', 'grid_row']
    team_features = ['team_avg_grid', 'team_avg_finish', 'team_performance_delta',
                     'team_dnf_rate', 'team_last3_avg_finish', 'team_last3_avg_change',
                     'team_last5_avg_finish', 'team_last5_avg_change', 'team_form_trend']
    temporal_features = ['race_number', 'season_progress', 'post_2022',
                         'early_season', 'mid_season', 'late_season']
    interaction_features = ['grid_x_overtaking', 'grid_x_team_delta', 'top3_x_street',
                           'top3_x_high_df', 'frontrow_x_correlation']

    print(f"Grid position features: {len(grid_features)}")
    print(f"Team features: {len(team_features)}")
    print(f"Temporal features: {len(temporal_features)}")
    print(f"Interaction features: {len(interaction_features)}")
    print(f"Circuit features: 10")
    print(f"External circuit info: 12")
