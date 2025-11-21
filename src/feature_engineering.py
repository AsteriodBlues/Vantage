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
    Calculate comprehensive team performance features.

    Includes rolling metrics, season stats, reliability, and relative performance.
    Uses only past data to avoid leakage.
    """
    df = df.copy()
    df = df.sort_values(['date', 'TeamName']).reset_index(drop=True)

    # initialize all team feature columns
    team_cols = [
        'team_avg_finish_last_5', 'team_avg_finish_last_3', 'team_avg_grid_last_5',
        'team_best_finish_last_5', 'team_worst_finish_last_5', 'team_points_last_5',
        'team_position_change_avg_5', 'team_dnf_rate_last_10', 'team_completion_rate_season',
        'team_wins_season', 'team_podiums_season', 'team_points_total',
        'team_races_since_podium', 'team_vs_average_grid', 'team_momentum',
        'team_consistency', 'team_avg_grid', 'team_avg_finish', 'team_performance_delta'
    ]

    for col in team_cols:
        df[col] = 0.0

    # process by race to avoid intra-race leakage
    df = df.sort_values(['date', 'TeamName', 'DriverId']).reset_index(drop=True)

    for team in df['TeamName'].unique():
        team_mask = df['TeamName'] == team
        team_data = df[team_mask].copy()

        finishes = []
        grids = []
        points_list = []
        pos_changes = []
        dnfs = []
        season_wins = 0
        season_podiums = 0
        season_points = 0
        last_podium_idx = -1
        current_year = None
        current_race = None
        race_buffer = []  # buffer to collect all entries from same race

        for i, (idx, row) in enumerate(team_data.iterrows()):
            # check if we've moved to a new race - if so, process buffered entries
            race_key = (row['year'], row['round'])
            if race_key != current_race:
                # process previous race's buffered entries (only if same year)
                for buffered_row in race_buffer:
                    # only add to history if same year (don't let last year leak into this year)
                    if buffered_row['year'] == current_year:
                        b_pos = buffered_row['Position']
                        season_finishes.append(b_pos)

                        if b_pos == 1:
                            season_wins += 1
                        if b_pos <= 3:
                            season_podiums += 1
                            last_podium_idx = len(finishes) - 1
                        season_points += buffered_row['Points']

                    # always add to rolling history regardless of year
                    b_pos = buffered_row['Position']
                    finishes.append(b_pos)
                    grids.append(buffered_row['GridPosition'])
                    points_list.append(buffered_row['Points'])
                    pos_changes.append(buffered_row['position_change'])
                    dnfs.append(1 if buffered_row['is_dnf'] else 0)

                race_buffer = []
                current_race = race_key

            # reset season stats on new year (after processing buffer)
            if row['year'] != current_year:
                current_year = row['year']
                season_wins = 0
                season_podiums = 0
                season_points = 0
                season_finishes = []

            # rolling metrics from past races
            if len(finishes) >= 5:
                df.loc[idx, 'team_avg_finish_last_5'] = np.mean(finishes[-5:])
                df.loc[idx, 'team_avg_grid_last_5'] = np.mean(grids[-5:])
                df.loc[idx, 'team_best_finish_last_5'] = min(finishes[-5:])
                df.loc[idx, 'team_worst_finish_last_5'] = max(finishes[-5:])
                df.loc[idx, 'team_points_last_5'] = sum(points_list[-5:])
                df.loc[idx, 'team_position_change_avg_5'] = np.mean(pos_changes[-5:])
                df.loc[idx, 'team_consistency'] = np.std(finishes[-5:])
            elif len(finishes) >= 3:
                df.loc[idx, 'team_avg_finish_last_5'] = np.mean(finishes)
                df.loc[idx, 'team_avg_grid_last_5'] = np.mean(grids)
                df.loc[idx, 'team_best_finish_last_5'] = min(finishes)
                df.loc[idx, 'team_worst_finish_last_5'] = max(finishes)
                df.loc[idx, 'team_points_last_5'] = sum(points_list)
                df.loc[idx, 'team_position_change_avg_5'] = np.mean(pos_changes)
                df.loc[idx, 'team_consistency'] = np.std(finishes) if len(finishes) > 1 else 0
            else:
                # default for first races
                df.loc[idx, 'team_avg_finish_last_5'] = 10.0
                df.loc[idx, 'team_avg_grid_last_5'] = 10.0
                df.loc[idx, 'team_best_finish_last_5'] = 10.0
                df.loc[idx, 'team_worst_finish_last_5'] = 10.0

            if len(finishes) >= 3:
                df.loc[idx, 'team_avg_finish_last_3'] = np.mean(finishes[-3:])
            else:
                df.loc[idx, 'team_avg_finish_last_3'] = np.mean(finishes) if finishes else 10.0

            # dnf rate from last 10
            if len(dnfs) >= 10:
                df.loc[idx, 'team_dnf_rate_last_10'] = np.mean(dnfs[-10:]) * 100
            elif dnfs:
                df.loc[idx, 'team_dnf_rate_last_10'] = np.mean(dnfs) * 100

            # season stats
            df.loc[idx, 'team_wins_season'] = season_wins
            df.loc[idx, 'team_podiums_season'] = season_podiums
            df.loc[idx, 'team_points_total'] = season_points

            if len(season_finishes) > 0:
                df.loc[idx, 'team_completion_rate_season'] = (1 - np.mean([1 if f > 20 else 0 for f in season_finishes])) * 100
                df.loc[idx, 'team_avg_grid'] = np.mean(grids[-len(season_finishes):]) if grids else 10.0
                df.loc[idx, 'team_avg_finish'] = np.mean(season_finishes)
                df.loc[idx, 'team_performance_delta'] = df.loc[idx, 'team_avg_grid'] - df.loc[idx, 'team_avg_finish']

            # races since last podium
            if last_podium_idx >= 0:
                df.loc[idx, 'team_races_since_podium'] = i - last_podium_idx
            else:
                df.loc[idx, 'team_races_since_podium'] = i

            # momentum: compare last 3 to season average
            if len(finishes) >= 5:
                recent_avg = np.mean(finishes[-3:])
                season_avg = np.mean(finishes)
                df.loc[idx, 'team_momentum'] = season_avg - recent_avg  # positive = improving

            # buffer current race entry (will be processed when we move to next race)
            race_buffer.append(row)

        # process final race buffer
        for buffered_row in race_buffer:
            b_pos = buffered_row['Position']
            finishes.append(b_pos)
            grids.append(buffered_row['GridPosition'])
            points_list.append(buffered_row['Points'])
            pos_changes.append(buffered_row['position_change'])
            dnfs.append(1 if buffered_row['is_dnf'] else 0)

    # relative to field average
    for year in df['year'].unique():
        year_mask = df['year'] == year
        field_avg_grid = df.loc[year_mask, 'GridPosition'].mean()
        df.loc[year_mask, 'team_vs_average_grid'] = df.loc[year_mask, 'team_avg_grid'] - field_avg_grid

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


def create_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate driver-specific performance features.

    Includes experience, circuit specialization, recent form, and relative metrics.
    """
    df = df.copy()
    df = df.sort_values(['date', 'DriverId']).reset_index(drop=True)

    # initialize driver columns
    driver_cols = [
        'driver_career_races', 'driver_career_wins', 'driver_career_podiums',
        'driver_years_experience', 'is_rookie', 'is_veteran',
        'driver_races_at_circuit', 'driver_avg_finish_at_circuit',
        'driver_best_finish_at_circuit', 'is_circuit_specialist',
        'driver_avg_finish_last_5', 'driver_avg_position_change_5',
        'driver_points_last_5', 'driver_best_finish_last_5', 'driver_consistency_last_5',
        'driver_momentum', 'driver_hot_streak', 'driver_points_season', 'driver_avg_grid_season',
        'driver_avg_finish_season', 'driver_vs_teammate', 'driver_vs_car_potential'
    ]

    for col in driver_cols:
        df[col] = 0.0

    for driver in df['DriverId'].unique():
        driver_mask = df['DriverId'] == driver
        driver_data = df[driver_mask].copy()

        career_races = 0
        career_wins = 0
        career_podiums = 0
        seasons_seen = set()
        circuit_history = {}  # circuit -> list of finishes
        recent_finishes = []
        recent_pos_changes = []
        recent_points = []
        consecutive_points = 0
        season_points = 0
        season_grids = []
        season_finishes = []
        current_year = None

        for i, (idx, row) in enumerate(driver_data.iterrows()):
            circuit = row['circuit']

            # reset season stats
            if row['year'] != current_year:
                current_year = row['year']
                season_points = 0
                season_grids = []
                season_finishes = []

            # career stats (at time of race)
            df.loc[idx, 'driver_career_races'] = career_races
            df.loc[idx, 'driver_career_wins'] = career_wins
            df.loc[idx, 'driver_career_podiums'] = career_podiums
            df.loc[idx, 'driver_years_experience'] = len(seasons_seen)
            df.loc[idx, 'is_rookie'] = 1 if career_races < 20 else 0
            df.loc[idx, 'is_veteran'] = 1 if career_races > 100 else 0

            # circuit specific history
            if circuit in circuit_history and len(circuit_history[circuit]) > 0:
                circuit_avg = np.mean(circuit_history[circuit])
                df.loc[idx, 'driver_races_at_circuit'] = len(circuit_history[circuit])
                df.loc[idx, 'driver_avg_finish_at_circuit'] = circuit_avg
                df.loc[idx, 'driver_best_finish_at_circuit'] = min(circuit_history[circuit])

                # circuit specialist: significantly better here than overall
                overall_avg = np.mean(recent_finishes) if recent_finishes else 10.0
                df.loc[idx, 'is_circuit_specialist'] = 1 if (overall_avg - circuit_avg) > 2 else 0
            else:
                df.loc[idx, 'driver_races_at_circuit'] = 0
                df.loc[idx, 'driver_avg_finish_at_circuit'] = 10.0
                df.loc[idx, 'driver_best_finish_at_circuit'] = 20.0
                df.loc[idx, 'is_circuit_specialist'] = 0

            # hot streak (consecutive points finishes)
            df.loc[idx, 'driver_hot_streak'] = consecutive_points

            # recent form (last 5)
            if len(recent_finishes) >= 5:
                df.loc[idx, 'driver_avg_finish_last_5'] = np.mean(recent_finishes[-5:])
                df.loc[idx, 'driver_avg_position_change_5'] = np.mean(recent_pos_changes[-5:])
                df.loc[idx, 'driver_points_last_5'] = sum(recent_points[-5:])
                df.loc[idx, 'driver_best_finish_last_5'] = min(recent_finishes[-5:])
                df.loc[idx, 'driver_consistency_last_5'] = np.std(recent_finishes[-5:])
            elif len(recent_finishes) > 0:
                df.loc[idx, 'driver_avg_finish_last_5'] = np.mean(recent_finishes)
                df.loc[idx, 'driver_avg_position_change_5'] = np.mean(recent_pos_changes)
                df.loc[idx, 'driver_points_last_5'] = sum(recent_points)
                df.loc[idx, 'driver_best_finish_last_5'] = min(recent_finishes)
                df.loc[idx, 'driver_consistency_last_5'] = np.std(recent_finishes) if len(recent_finishes) > 1 else 0
            else:
                df.loc[idx, 'driver_avg_finish_last_5'] = 10.0
                df.loc[idx, 'driver_best_finish_last_5'] = 10.0

            # momentum
            if len(recent_finishes) >= 5:
                recent_avg = np.mean(recent_finishes[-3:])
                overall_avg = np.mean(recent_finishes)
                df.loc[idx, 'driver_momentum'] = overall_avg - recent_avg

            # season stats
            df.loc[idx, 'driver_points_season'] = season_points
            if season_grids:
                df.loc[idx, 'driver_avg_grid_season'] = np.mean(season_grids)
            if season_finishes:
                df.loc[idx, 'driver_avg_finish_season'] = np.mean(season_finishes)

            # update history
            career_races += 1
            seasons_seen.add(row['year'])

            if row['Position'] == 1:
                career_wins += 1
            if row['Position'] <= 3:
                career_podiums += 1

            # update hot streak
            if row['Position'] <= 10:
                consecutive_points += 1
            else:
                consecutive_points = 0

            if circuit not in circuit_history:
                circuit_history[circuit] = []
            circuit_history[circuit].append(row['Position'])

            recent_finishes.append(row['Position'])
            recent_pos_changes.append(row['position_change'])
            recent_points.append(row['Points'])
            season_points += row['Points']
            season_grids.append(row['GridPosition'])
            season_finishes.append(row['Position'])

    # calculate vs teammate and teammate comparison features (requires second pass)
    df['driver_vs_teammate'] = 0.0
    df['teammate_avg_grid_diff'] = 0.0
    df['teammate_avg_finish_diff'] = 0.0
    df['is_team_leader'] = 0

    # track teammate history for rolling comparisons
    teammate_grid_history = {}  # (team, year) -> {driver: [grids]}
    teammate_finish_history = {}  # (team, year) -> {driver: [finishes]}

    df_sorted = df.sort_values(['year', 'round']).reset_index(drop=True)

    for (team, year, race), group in df_sorted.groupby(['TeamName', 'year', 'round']):
        if len(group) == 2:
            drivers = group['DriverId'].values
            positions = group['Position'].values
            grids = group['GridPosition'].values
            indices = group.index.values

            # current race comparison
            df.loc[indices[0], 'driver_vs_teammate'] = positions[1] - positions[0]
            df.loc[indices[1], 'driver_vs_teammate'] = positions[0] - positions[1]

            # rolling teammate comparison
            key = (team, year)
            if key not in teammate_grid_history:
                teammate_grid_history[key] = {drivers[0]: [], drivers[1]: []}
                teammate_finish_history[key] = {drivers[0]: [], drivers[1]: []}

            for i, driver in enumerate(drivers):
                other = drivers[1 - i]
                idx = indices[i]

                # calculate average difference from past races
                my_grids = teammate_grid_history[key].get(driver, [])
                other_grids = teammate_grid_history[key].get(other, [])
                my_finishes = teammate_finish_history[key].get(driver, [])
                other_finishes = teammate_finish_history[key].get(other, [])

                if my_grids and other_grids:
                    df.loc[idx, 'teammate_avg_grid_diff'] = np.mean(my_grids) - np.mean(other_grids)
                    df.loc[idx, 'teammate_avg_finish_diff'] = np.mean(my_finishes) - np.mean(other_finishes)
                    # team leader has better average grid
                    df.loc[idx, 'is_team_leader'] = 1 if np.mean(my_grids) < np.mean(other_grids) else 0

            # update history after calculating
            for i, driver in enumerate(drivers):
                if driver not in teammate_grid_history[key]:
                    teammate_grid_history[key][driver] = []
                    teammate_finish_history[key][driver] = []
                teammate_grid_history[key][driver].append(grids[i])
                teammate_finish_history[key][driver].append(positions[i])

    # driver vs car potential (driver finish vs team average finish)
    for idx, row in df.iterrows():
        team_avg = df.loc[idx, 'team_avg_finish']
        if team_avg > 0:
            df.loc[idx, 'driver_vs_car_potential'] = team_avg - row['Position']

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.

    Includes race number, season progress, era indicators, and championship context.
    """
    df = df.copy()

    # race number in season
    df['race_number'] = df['round']

    # season progress (0-1 scale)
    max_rounds = df.groupby('year')['round'].transform('max')
    df['season_progress'] = df['round'] / max_rounds
    df['races_remaining'] = max_rounds - df['round']

    # era indicator (2022 regulation change)
    df['post_2022'] = (df['year'] >= 2022).astype(int)
    df['years_into_regulations'] = df['year'].apply(lambda x: x - 2022 if x >= 2022 else x - 2017)

    # season phase indicators
    df['is_season_opener'] = (df['round'] == 1).astype(int)
    df['is_season_finale'] = (df['round'] == max_rounds).astype(int)
    df['early_season'] = (df['season_progress'] <= 0.33).astype(int)
    df['mid_season'] = ((df['season_progress'] > 0.33) & (df['season_progress'] <= 0.66)).astype(int)
    df['late_season'] = (df['season_progress'] > 0.66).astype(int)

    # covid affected seasons
    df['covid_affected'] = df['year'].isin([2020, 2021]).astype(int)

    # championship context
    df['championship_gap_leader'] = 0.0
    df['driver_in_contention'] = 0

    for year in df['year'].unique():
        year_mask = df['year'] == year
        year_data = df[year_mask].copy()

        for race_round in year_data['round'].unique():
            round_mask = (df['year'] == year) & (df['round'] == race_round)

            # get points up to previous race
            prev_races = df[(df['year'] == year) & (df['round'] < race_round)]

            if len(prev_races) > 0:
                driver_points = prev_races.groupby('DriverId')['Points'].sum()
                leader_points = driver_points.max() if len(driver_points) > 0 else 0

                for idx in df[round_mask].index:
                    driver = df.loc[idx, 'DriverId']
                    current_points = driver_points.get(driver, 0)
                    gap = leader_points - current_points
                    df.loc[idx, 'championship_gap_leader'] = gap

                    # in contention if within 50 points (about 2 race wins)
                    races_left = df.loc[idx, 'races_remaining']
                    max_possible = races_left * 25
                    df.loc[idx, 'driver_in_contention'] = 1 if gap <= max_possible else 0

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sophisticated interaction terms between key features.

    Captures grid × circuit, team × circuit, driver × conditions relationships.
    """
    df = df.copy()

    # grid × circuit interactions
    if 'overtaking_difficulty_index' in df.columns:
        df['grid_x_overtaking'] = df['GridPosition'] * df['overtaking_difficulty_index']

    if 'circuit_pole_win_rate' in df.columns:
        df['grid_x_pole_win_rate'] = df['GridPosition'] * df['circuit_pole_win_rate'] / 100

    if 'circuit_var_pos_change' in df.columns:
        # midfield advantage at chaotic circuits
        df['midfield_x_variance'] = ((df['GridPosition'] > 5) & (df['GridPosition'] <= 15)).astype(int) * df['circuit_var_pos_change']

        # back of grid at high variance circuits
        df['back_x_variance'] = (df['GridPosition'] > 15).astype(int) * df['circuit_var_pos_change']

    # pole/front row at processional tracks
    if 'circuit_correlation' in df.columns:
        df['frontrow_x_correlation'] = df['front_row'] * df['circuit_correlation']
        df['pole_at_processional'] = ((df['GridPosition'] == 1) & (df['circuit_correlation'] > 0.8)).astype(int)

    # top 3 interactions with circuit type
    if 'is_street' in df.columns:
        df['top3_x_street'] = df['top_three'] * df['is_street']

    if 'high_downforce' in df.columns:
        df['top3_x_high_df'] = df['top_three'] * df['high_downforce']

    if 'low_downforce' in df.columns:
        df['grid_x_low_df'] = df['GridPosition'] * df['low_downforce']

    # team × circuit interactions
    if 'team_performance_delta' in df.columns:
        df['grid_x_team_delta'] = df['GridPosition'] * df['team_performance_delta']

    if 'team_momentum' in df.columns and 'circuit_var_pos_change' in df.columns:
        df['momentum_x_variance'] = df['team_momentum'] * df['circuit_var_pos_change']

    # team reliability at harsh circuits
    if 'team_dnf_rate_last_10' in df.columns and 'circuit_dnf_rate' in df.columns:
        df['reliability_x_circuit_dnf'] = df['team_dnf_rate_last_10'] * df['circuit_dnf_rate'] / 100

    # driver experience interactions
    if 'driver_career_races' in df.columns:
        # veteran advantage at new circuits (less data for them)
        if 'driver_races_at_circuit' in df.columns:
            df['veteran_new_circuit'] = (df['driver_career_races'] > 100).astype(int) * (df['driver_races_at_circuit'] == 0).astype(int)

        # experience matters more in chaotic conditions
        if 'circuit_var_pos_change' in df.columns:
            df['experience_x_chaos'] = df['driver_career_races'] * df['circuit_var_pos_change'] / 100

    # form × context
    if 'driver_momentum' in df.columns and 'driver_in_contention' in df.columns:
        df['form_x_contention'] = df['driver_momentum'] * df['driver_in_contention']

    # season phase interactions
    if 'early_season' in df.columns:
        # early season uncertainty (less predictable)
        df['early_x_variance'] = df['early_season'] * df.get('circuit_var_pos_change', 0)

    if 'late_season' in df.columns and 'driver_in_contention' in df.columns:
        # pressure in late season title fight
        df['late_contention_pressure'] = df['late_season'] * df['driver_in_contention']

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


def create_feature_summary(df: pd.DataFrame, output_path: str = 'data/processed/feature_summary.csv') -> pd.DataFrame:
    """
    Generate feature quality summary for documentation and selection.

    Calculates correlation with target, missing %, variance, and categorizes features.
    """
    summary_data = []

    # define feature categories
    grid_features = ['GridPosition', 'grid_squared', 'grid_cubed', 'grid_log', 'grid_sqrt',
                     'front_row', 'top_three', 'top_five', 'top_ten', 'back_half', 'grid_side_clean', 'grid_row']

    team_features = [col for col in df.columns if col.startswith('team_')]
    driver_features = [col for col in df.columns if col.startswith('driver_') or col in ['is_rookie', 'is_veteran']]
    circuit_features = [col for col in df.columns if col.startswith('circuit_') or col.startswith('overtaking')]
    temporal_features = ['race_number', 'season_progress', 'races_remaining', 'post_2022',
                         'years_into_regulations', 'early_season', 'mid_season', 'late_season',
                         'is_season_opener', 'is_season_finale', 'covid_affected',
                         'championship_gap_leader', 'driver_in_contention']
    interaction_features = [col for col in df.columns if '_x_' in col or 'pole_at_' in col or
                           'veteran_new' in col or 'late_contention' in col]

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in ['Position', 'DriverNumber', 'year', 'round', 'Laps']:
            continue

        # determine category
        if col in grid_features:
            category = 'grid'
        elif col in team_features:
            category = 'team'
        elif col in driver_features:
            category = 'driver'
        elif col in circuit_features:
            category = 'circuit'
        elif col in temporal_features:
            category = 'temporal'
        elif col in interaction_features:
            category = 'interaction'
        else:
            category = 'other'

        # calculate metrics
        missing_pct = df[col].isna().sum() / len(df) * 100
        variance = df[col].var()

        # correlation with target
        valid_mask = df[col].notna() & df['Position'].notna()
        if valid_mask.sum() > 10:
            corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, 'Position'])
        else:
            corr = np.nan

        summary_data.append({
            'feature_name': col,
            'feature_type': category,
            'missing_pct': round(missing_pct, 2),
            'variance': round(variance, 4) if not np.isnan(variance) else 0,
            'correlation_with_target': round(corr, 4) if not np.isnan(corr) else 0,
            'abs_correlation': abs(corr) if not np.isnan(corr) else 0
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('abs_correlation', ascending=False)

    # save to csv
    summary_df.to_csv(output_path, index=False)
    print(f"Feature summary saved to {output_path}")

    return summary_df


def validate_no_leakage(df: pd.DataFrame) -> dict:
    """
    Validate that no data leakage exists in features.

    Checks that rolling/cumulative features only use past data.
    Returns dict with validation results.
    """
    results = {
        'passed': True,
        'issues': []
    }

    # check first race of each season for each team
    # Season stats should be zero at start of season (team_wins_season, team_points_total are season-specific)
    # But rolling stats (last 5 races) can carry over from previous season - that's expected
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        first_round = year_data['round'].min()
        first_race = year_data[year_data['round'] == first_round]

        for team in first_race['TeamName'].unique():
            team_first = first_race[first_race['TeamName'] == team]

            for idx, row in team_first.iterrows():
                # first race of season should have zero season-specific stats
                # (wins, points accumulated in CURRENT season only)
                # Rolling stats from last 5 races can carry from previous season - that's fine
                if row['team_wins_season'] > 0:
                    results['passed'] = False
                    results['issues'].append(f"Team {team} in {year} R{first_round}: has {row['team_wins_season']} season wins at first race")

                if row['team_points_total'] > 0:
                    results['passed'] = False
                    results['issues'].append(f"Team {team} in {year} R{first_round}: has {row['team_points_total']} season points at first race")

    # check that driver career races increases over time
    for driver in df['DriverId'].unique():
        driver_data = df[df['DriverId'] == driver].sort_values('date')
        career_races = driver_data['driver_career_races'].values

        for i in range(1, len(career_races)):
            if career_races[i] < career_races[i-1]:
                results['passed'] = False
                results['issues'].append(f"Driver {driver} career races decreased")
                break

    return results


def test_edge_cases(df: pd.DataFrame) -> dict:
    """
    Test edge cases in the data.

    Validates handling of:
    - First race of season
    - Rookie's first race
    - First race at new circuit
    - Teams with only one driver
    """
    results = {
        'first_race_season': {'tested': 0, 'passed': 0},
        'rookie_first_race': {'tested': 0, 'passed': 0},
        'first_at_circuit': {'tested': 0, 'passed': 0},
        'single_driver_team': {'tested': 0, 'passed': 0}
    }

    # test first race of season
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        first_round = year_data['round'].min()
        first_race = year_data[year_data['round'] == first_round]

        for idx, row in first_race.iterrows():
            results['first_race_season']['tested'] += 1

            # should have reasonable defaults, not NaN
            if pd.notna(row['team_avg_finish_last_5']) and pd.notna(row['driver_avg_finish_last_5']):
                results['first_race_season']['passed'] += 1

    # test rookie first races
    rookies = df[df['driver_career_races'] == 0]
    for idx, row in rookies.iterrows():
        results['rookie_first_race']['tested'] += 1

        if row['is_rookie'] == 1 and pd.notna(row['driver_avg_finish_last_5']):
            results['rookie_first_race']['passed'] += 1

    # test first at circuit
    first_at_circuit = df[df['driver_races_at_circuit'] == 0]
    for idx, row in first_at_circuit.iterrows():
        results['first_at_circuit']['tested'] += 1

        if pd.notna(row['driver_avg_finish_at_circuit']):
            results['first_at_circuit']['passed'] += 1

    # test teams with potentially one driver (check for sensible teammate features)
    for (team, year, race), group in df.groupby(['TeamName', 'year', 'round']):
        if len(group) == 1:
            results['single_driver_team']['tested'] += 1
            row = group.iloc[0]

            # should have default values, not errors
            if row['driver_vs_teammate'] == 0:
                results['single_driver_team']['passed'] += 1

    return results


def run_validation(df: pd.DataFrame) -> None:
    """
    Run all validation checks and print results.
    """
    print("\n" + "=" * 50)
    print("Running validation checks...")
    print("=" * 50)

    # check for data leakage
    leakage = validate_no_leakage(df)
    if leakage['passed']:
        print("✓ No data leakage detected")
    else:
        print("✗ Data leakage issues found:")
        for issue in leakage['issues'][:5]:
            print(f"  - {issue}")

    # test edge cases
    edge = test_edge_cases(df)
    print("\nEdge case tests:")
    for test_name, result in edge.items():
        if result['tested'] > 0:
            pct = result['passed'] / result['tested'] * 100
            status = "✓" if pct >= 90 else "✗"
            print(f"  {status} {test_name}: {result['passed']}/{result['tested']} ({pct:.1f}%)")
        else:
            print(f"  - {test_name}: No cases to test")

    print("=" * 50 + "\n")


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

    # create driver features
    print("Creating driver features...")
    race_data = create_driver_features(race_data)

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

    # generate feature summary
    print("Generating feature summary...")
    create_feature_summary(final_data)

    # run validation
    run_validation(final_data)

    print(f"Done! Created {len(final_data.columns)} features for {len(final_data)} records")

    return final_data


if __name__ == '__main__':
    df = create_all_features()
    print("\nFeature breakdown:")
    print(f"Total columns: {len(df.columns)}")

    # count by category
    grid_count = len([c for c in df.columns if c.startswith('grid_') or c in ['front_row', 'top_three', 'top_five', 'top_ten', 'back_half']])
    team_count = len([c for c in df.columns if c.startswith('team_')])
    driver_count = len([c for c in df.columns if c.startswith('driver_') or c in ['is_rookie', 'is_veteran']])
    circuit_count = len([c for c in df.columns if c.startswith('circuit_') or c.startswith('overtaking')])
    temporal_count = len([c for c in df.columns if c in ['race_number', 'season_progress', 'races_remaining', 'post_2022',
                                                          'years_into_regulations', 'early_season', 'mid_season', 'late_season',
                                                          'is_season_opener', 'is_season_finale', 'covid_affected',
                                                          'championship_gap_leader', 'driver_in_contention']])
    interaction_count = len([c for c in df.columns if '_x_' in c or 'pole_at_' in c or 'veteran_new' in c or 'late_contention' in c])

    print(f"Grid position features: {grid_count}")
    print(f"Team features: {team_count}")
    print(f"Driver features: {driver_count}")
    print(f"Circuit features: {circuit_count}")
    print(f"Temporal features: {temporal_count}")
    print(f"Interaction features: {interaction_count}")

    # show top correlated features
    summary = pd.read_csv('data/processed/feature_summary.csv')
    print("\nTop 10 features by correlation with finish position:")
    for _, row in summary.head(10).iterrows():
        print(f"  {row['feature_name']}: {row['correlation_with_target']:.3f}")


# =============================================================================
# CATEGORICAL ENCODING FUNCTIONS
# =============================================================================

"""
Encoding Strategy Decision Rationale:

1. circuit (21 unique):
   - One-hot: creates 21 columns, manageable
   - Target encoding: useful for tree models, reduces dimensions
   - Decision: implement both, let model selection decide

2. TeamName/TeamId (12 unique):
   - One-hot: only 12 columns, acceptable overhead
   - Teams have strong predictive power for performance
   - Decision: one-hot for linear models, label encoding for trees

3. DriverId (26 unique):
   - One-hot: 26 columns is borderline acceptable
   - Driver skill already captured in driver_* features
   - Decision: target encoding preferred, skip one-hot to avoid curse of dimensionality

4. circuit_type (2 unique: permanent, street):
   - Binary, simple one-hot (1 column after drop_first)
   - Decision: one-hot encoding

5. downforce_level (3 unique: high, medium, low):
   - Ordinal nature (low < medium < high)
   - Decision: ordinal encoding (0, 1, 2) AND one-hot for flexibility

6. grid_side (2 unique: clean, dirty):
   - Already binary encoded as grid_side_clean
   - Decision: keep existing binary column
"""


def create_label_encodings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Create label encodings for categorical variables.

    Returns dataframe with encoded columns and mapping dict for inverse transform.
    Useful for tree-based models that handle integers well.
    """
    df = df.copy()
    label_mappings = {}

    categorical_cols = ['circuit', 'TeamName', 'TeamId', 'DriverId']

    for col in categorical_cols:
        if col not in df.columns:
            continue

        unique_vals = sorted(df[col].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        inverse_mapping = {idx: val for val, idx in mapping.items()}

        df[f'{col}_encoded'] = df[col].map(mapping)
        label_mappings[col] = {
            'to_int': mapping,
            'to_label': inverse_mapping
        }

    return df, label_mappings


def create_onehot_features(df: pd.DataFrame,
                           columns: list = None,
                           drop_first: bool = True) -> pd.DataFrame:
    """
    Create one-hot encoded features for specified categorical columns.

    Default columns are those suitable for one-hot encoding:
    - circuit: 21 values, captures track characteristics
    - TeamName: 12 values, captures constructor performance
    - circuit_type: 2 values
    - downforce_level: 3 values

    Skips DriverId by default (use target encoding instead).
    """
    df = df.copy()

    if columns is None:
        columns = ['circuit', 'TeamName', 'circuit_type', 'downforce_level']

    # filter to columns that exist
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        df = pd.concat([df, dummies], axis=1)

    return df


def create_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ordinal encodings for variables with natural ordering.

    Currently handles:
    - downforce_level: low=0, medium=1, high=2
    """
    df = df.copy()

    # downforce level has natural ordering
    if 'downforce_level' in df.columns:
        downforce_order = {'low': 0, 'medium': 1, 'high': 2}
        df['downforce_ordinal'] = df['downforce_level'].map(downforce_order)

    return df


def create_target_encoding(df: pd.DataFrame,
                           target_col: str = 'Position',
                           columns: list = None,
                           n_folds: int = 5,
                           smoothing: float = 10.0) -> pd.DataFrame:
    """
    Create target-encoded features using cross-validation to prevent leakage.

    Each category is encoded as its mean target value, computed using
    out-of-fold predictions to avoid overfitting.

    Args:
        df: input dataframe with target and categorical columns
        target_col: name of target variable
        columns: list of columns to encode (defaults to high-cardinality cats)
        n_folds: number of CV folds for out-of-fold encoding
        smoothing: regularization parameter (higher = more shrinkage to global mean)

    The smoothing formula is:
        encoded = (count * category_mean + smoothing * global_mean) / (count + smoothing)

    This shrinks small categories toward the global mean.
    """
    from sklearn.model_selection import KFold

    df = df.copy()

    if columns is None:
        # default to high-cardinality categoricals
        columns = ['circuit', 'TeamName', 'DriverId']

    columns = [c for c in columns if c in df.columns]
    global_mean = df[target_col].mean()

    for col in columns:
        encoded_col = f'{col}_target_enc'
        df[encoded_col] = np.nan

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(df):
            train_data = df.iloc[train_idx]

            # calculate smoothed mean for each category using training fold only
            category_stats = train_data.groupby(col)[target_col].agg(['mean', 'count'])

            smoothed_means = (
                (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) /
                (category_stats['count'] + smoothing)
            )

            # apply to validation fold
            df.loc[df.index[val_idx], encoded_col] = df.iloc[val_idx][col].map(smoothed_means)

        # fill any missing (unseen categories) with global mean
        df[encoded_col] = df[encoded_col].fillna(global_mean)

    return df


def create_frequency_encoding(df: pd.DataFrame,
                              columns: list = None) -> pd.DataFrame:
    """
    Encode categories by their frequency in the dataset.

    Useful as a simple feature that captures how common certain
    circuits/teams/drivers are without leaking target info.
    """
    df = df.copy()

    if columns is None:
        columns = ['circuit', 'TeamName', 'DriverId']

    columns = [c for c in columns if c in df.columns]

    for col in columns:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    return df


# =============================================================================
# FEATURE PREPARATION PIPELINE
# =============================================================================

def get_feature_columns(df: pd.DataFrame, include_onehot: bool = False) -> dict:
    """
    Get lists of feature columns by category.

    Returns dict with keys: numeric, categorical, target, id_columns
    """
    # columns to exclude from features
    id_cols = ['DriverNumber', 'BroadcastName', 'Abbreviation', 'DriverId',
               'TeamName', 'TeamColor', 'TeamId', 'FirstName', 'LastName',
               'FullName', 'HeadshotUrl', 'CountryCode', 'race_name', 'date',
               'Q1', 'Q2', 'Q3', 'Time', 'Status', 'ClassifiedPosition',
               'GridPosition_raw', 'Position_raw', 'circuit', 'direction']

    target_col = 'Position'
    leakage_cols = ['Points', 'Laps', 'position_change', 'is_dnf', 'completed_race']

    # categoricals that need encoding
    categorical_raw = ['circuit', 'TeamName', 'DriverId', 'circuit_type',
                       'downforce_level', 'grid_side']

    # get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # remove target, leakage, and id columns
    exclude = set(id_cols + [target_col] + leakage_cols + ['year', 'round', 'month'])
    numeric_features = [c for c in numeric_cols if c not in exclude]

    # add one-hot columns if requested
    onehot_cols = []
    if include_onehot:
        onehot_cols = [c for c in df.columns if any(
            c.startswith(f'{cat}_') and c not in numeric_features
            for cat in ['circuit', 'TeamName', 'circuit_type', 'downforce_level']
        )]

    return {
        'numeric': numeric_features,
        'categorical_raw': categorical_raw,
        'onehot': onehot_cols,
        'target': target_col,
        'id_columns': id_cols,
        'leakage': leakage_cols
    }


def prepare_features_for_training(df: pd.DataFrame,
                                  encoding_type: str = 'target',
                                  include_interactions: bool = True) -> tuple[pd.DataFrame, list]:
    """
    Prepare feature matrix for model training.

    Args:
        df: dataframe with all features
        encoding_type: 'target', 'onehot', 'label', or 'mixed'
        include_interactions: whether to include interaction features

    Returns:
        Tuple of (processed dataframe, list of feature column names)
    """
    df = df.copy()

    # apply ordinal encoding (always useful)
    df = create_ordinal_encoding(df)

    # apply frequency encoding (always useful, no leakage)
    df = create_frequency_encoding(df)

    if encoding_type == 'target':
        df = create_target_encoding(df, target_col='Position')
        df, _ = create_label_encodings(df)

    elif encoding_type == 'onehot':
        df = create_onehot_features(df)
        df, _ = create_label_encodings(df)

    elif encoding_type == 'label':
        df, _ = create_label_encodings(df)

    elif encoding_type == 'mixed':
        # target for driver (high cardinality), onehot for circuit/team
        df = create_target_encoding(df, columns=['DriverId'])
        df = create_onehot_features(df, columns=['circuit', 'TeamName', 'circuit_type', 'downforce_level'])
        df, _ = create_label_encodings(df)

    # get feature columns
    feature_info = get_feature_columns(df, include_onehot=(encoding_type in ['onehot', 'mixed']))

    feature_cols = feature_info['numeric'] + feature_info['onehot']

    # add encoded columns
    encoded_cols = [c for c in df.columns if c.endswith('_encoded') or
                    c.endswith('_target_enc') or c.endswith('_freq') or
                    c == 'downforce_ordinal']
    feature_cols = list(set(feature_cols + encoded_cols))

    # remove any remaining leakage columns
    feature_cols = [c for c in feature_cols if c not in feature_info['leakage']]

    # filter to interaction features if not wanted
    if not include_interactions:
        feature_cols = [c for c in feature_cols if '_x_' not in c]

    return df, sorted(feature_cols)


def split_by_time(df: pd.DataFrame,
                  test_year: int = None,
                  val_ratio: float = 0.15) -> tuple:
    """
    Split data respecting temporal ordering.

    Uses most recent year as test set, and a portion of remaining as validation.
    This prevents future data leakage.

    Args:
        df: full dataframe
        test_year: year to use as test set (defaults to most recent)
        val_ratio: fraction of training data to use for validation

    Returns:
        (train_df, val_df, test_df)
    """
    df = df.sort_values(['year', 'round', 'GridPosition']).reset_index(drop=True)

    if test_year is None:
        test_year = df['year'].max()

    # test set is the specified year
    test_df = df[df['year'] == test_year].copy()
    train_val_df = df[df['year'] < test_year].copy()

    # split train/val by time within remaining data
    train_val_df = train_val_df.sort_values(['year', 'round'])
    n_train = int(len(train_val_df) * (1 - val_ratio))

    train_df = train_val_df.iloc[:n_train].copy()
    val_df = train_val_df.iloc[n_train:].copy()

    return train_df, val_df, test_df


def create_model_ready_dataset(input_path: str = 'data/processed/race_data_with_features.csv',
                               output_dir: str = 'data/processed',
                               encoding_type: str = 'target') -> dict:
    """
    Create final model-ready datasets with train/val/test splits.

    Saves processed data to CSV and returns summary statistics.
    """
    import os

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # prepare features
    print(f"Applying {encoding_type} encoding...")
    df, feature_cols = prepare_features_for_training(df, encoding_type=encoding_type)

    # split data
    print("Splitting by time...")
    train_df, val_df, test_df = split_by_time(df)

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    features_path = os.path.join(output_dir, 'feature_columns.txt')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val: {val_path} ({len(val_df)} rows)")
    print(f"  Test: {test_path} ({len(test_df)} rows)")
    print(f"  Features: {features_path} ({len(feature_cols)} features)")

    return {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'train_years': sorted(train_df['year'].unique().tolist()),
        'val_years': sorted(val_df['year'].unique().tolist()),
        'test_years': sorted(test_df['year'].unique().tolist())
    }
