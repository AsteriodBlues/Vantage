"""
Prediction pipeline for F1 race outcome predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class F1PredictionPipeline:
    """
    Complete pipeline for F1 race predictions.
    Handles feature engineering from raw inputs and generates predictions.
    """

    def __init__(self, model_dir: str = 'models/production/finish_position_predictor_latest'):
        """Initialize pipeline with model and preprocessing artifacts."""
        model_path = Path(model_dir)

        # load model
        self.model = joblib.load(model_path / 'model.pkl')
        with open(model_path / 'metadata.json', 'r') as f:
            import json
            self.metadata = json.load(f)

        # load preprocessing artifacts
        preprocessing_dir = Path('models/preprocessing')
        self.feature_names = joblib.load(preprocessing_dir / 'feature_names.pkl')
        self.circuit_stats = joblib.load(preprocessing_dir / 'circuit_statistics.pkl')
        self.team_baselines = joblib.load(preprocessing_dir / 'team_baselines.pkl')

        print(f"Loaded model: {self.metadata['model_type']}")
        print(f"Features: {len(self.feature_names)}")

    def create_features(self,
                       grid_position: int,
                       circuit_name: str,
                       team: str,
                       driver: str,
                       year: int,
                       race_number: int) -> pd.DataFrame:
        """
        Create all required features from raw inputs.

        Args:
            grid_position: Starting position (1-20)
            circuit_name: Name of circuit
            team: Team name
            driver: Driver name
            year: Race year
            race_number: Race number in season

        Returns:
            DataFrame with all engineered features
        """
        # base features
        features = {
            'GridPosition': grid_position,
            'GridPosition_raw': grid_position,
            'year': year,
            'race_number': race_number,
        }

        # grid position transformations
        features['grid_squared'] = grid_position ** 2
        features['grid_cubed'] = grid_position ** 3
        features['grid_log'] = np.log(grid_position)
        features['grid_sqrt'] = np.sqrt(grid_position)

        # grid position indicators
        features['front_row'] = 1 if grid_position <= 2 else 0
        features['top_three'] = 1 if grid_position <= 3 else 0
        features['top_five'] = 1 if grid_position <= 5 else 0
        features['top_ten'] = 1 if grid_position <= 10 else 0
        features['back_half'] = 1 if grid_position > 10 else 0

        # grid side (odd = clean line, even = dirty)
        features['grid_side'] = grid_position % 2
        features['grid_side_clean'] = 1 if grid_position % 2 == 1 else 0
        features['grid_row'] = (grid_position + 1) // 2

        # season progress
        features['season_progress'] = race_number / 24.0
        features['season_phase_early'] = 1 if race_number <= 8 else 0
        features['season_phase_mid'] = 1 if 8 < race_number <= 16 else 0
        features['season_phase_late'] = 1 if race_number > 16 else 0

        # month (estimate based on race number)
        month_mapping = {1: 3, 2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 6, 8: 6,
                        9: 7, 10: 7, 11: 8, 12: 8, 13: 9, 14: 9, 15: 10,
                        16: 10, 17: 10, 18: 11, 19: 11, 20: 11, 21: 11, 22: 12, 23: 12, 24: 12}
        features['month'] = month_mapping.get(race_number, 6)

        # circuit features
        circuit_data = self.circuit_stats.get(circuit_name, {})
        features['circuit_pole_win_rate'] = circuit_data.get('circuit_pole_win_rate', 50.0)
        features['circuit_top3_win_rate'] = circuit_data.get('circuit_top3_win_rate', 50.0)
        features['circuit_avg_pos_change'] = circuit_data.get('circuit_avg_pos_change', 0.0)
        features['circuit_std_pos_change'] = circuit_data.get('circuit_std_pos_change', 2.0)
        features['circuit_var_pos_change'] = circuit_data.get('circuit_var_pos_change', 4.0)
        features['circuit_correlation'] = circuit_data.get('circuit_correlation', 0.7)
        features['circuit_dnf_rate'] = circuit_data.get('circuit_dnf_rate', 15.0)
        features['circuit_improved_pct'] = circuit_data.get('circuit_improved_pct', 40.0)
        features['overtaking_difficulty_index'] = circuit_data.get('overtaking_difficulty_index', 50.0)

        # team features
        team_data = self.team_baselines.get(team, {})
        features['team_avg_finish_last_5'] = team_data.get('avg_finish', 10.0)
        features['team_points_last_5'] = team_data.get('avg_points', 5.0)
        features['team_consistency'] = team_data.get('consistency', 5.0)
        features['team_win_rate'] = team_data.get('win_rate', 0.0)
        features['team_podium_rate'] = team_data.get('podium_rate', 0.0)
        features['team_points_rate'] = team_data.get('points_rate', 50.0)
        features['team_momentum'] = team_data.get('momentum', 0.0)

        # interaction features
        features['grid_x_overtaking'] = grid_position * features['overtaking_difficulty_index']
        features['grid_x_team_avg'] = grid_position * features['team_avg_finish_last_5']
        features['grid_x_dnf_rate'] = grid_position * features['circuit_dnf_rate']
        features['team_x_circuit_correlation'] = features['team_avg_finish_last_5'] * features['circuit_correlation']

        # position-based expectations
        features['expected_finish_from_grid'] = grid_position * features['circuit_correlation']
        features['underdog_potential'] = max(0, 10 - grid_position) * (100 - features['overtaking_difficulty_index']) / 100
        features['pole_advantage'] = (1 if grid_position == 1 else 0) * features['circuit_pole_win_rate']

        # add dummy values for remaining features to match training
        df = pd.DataFrame([features])

        # fill in missing features with defaults
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # ensure correct column order
        df = df[self.feature_names]

        return df

    def predict(self,
               grid_position: int,
               circuit_name: str,
               team: str,
               driver: str,
               year: int = 2024,
               race_number: int = 1) -> Dict:
        """
        Predict finish position for a single driver.

        Returns:
            Dictionary with prediction, confidence intervals, and probabilities
        """
        # create features
        X = self.create_features(grid_position, circuit_name, team, driver, year, race_number)

        # make prediction
        prediction = self.model.predict(X)[0]

        # estimate confidence interval using tree predictions
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            lower = np.percentile(tree_predictions, 10)
            upper = np.percentile(tree_predictions, 90)
        else:
            # fallback for non-ensemble models
            lower = max(1, prediction - 2)
            upper = min(20, prediction + 2)

        # calculate probabilities
        win_prob = max(0, (3 - prediction) / 3) if prediction <= 3 else 0.01
        podium_prob = max(0, (5 - prediction) / 5) if prediction <= 5 else 0.05
        points_prob = max(0, (12 - prediction) / 12) if prediction <= 12 else 0.1

        return {
            'predicted_finish': round(prediction, 2),
            'predicted_finish_rounded': int(round(prediction)),
            'confidence_interval': {
                'lower': round(lower, 1),
                'upper': round(upper, 1)
            },
            'position_change': round(grid_position - prediction, 1),
            'probabilities': {
                'win': round(win_prob, 3),
                'podium': round(podium_prob, 3),
                'points': round(points_prob, 3)
            },
            'inputs': {
                'grid_position': grid_position,
                'circuit': circuit_name,
                'team': team,
                'driver': driver
            }
        }

    def predict_race_result(self,
                           circuit_name: str,
                           year: int,
                           race_number: int,
                           grid: List[Dict]) -> Dict:
        """
        Predict full race result from starting grid.

        Args:
            circuit_name: Circuit name
            year: Race year
            race_number: Race number in season
            grid: List of dicts with 'position', 'driver', 'team'

        Returns:
            Dictionary with predicted results and race metrics
        """
        predictions = []

        for entry in grid:
            pred = self.predict(
                grid_position=entry['position'],
                circuit_name=circuit_name,
                team=entry['team'],
                driver=entry['driver'],
                year=year,
                race_number=race_number
            )

            predictions.append({
                'driver': entry['driver'],
                'team': entry['team'],
                'grid_position': entry['position'],
                'predicted_finish': pred['predicted_finish'],
                'position_change': pred['position_change'],
                'confidence_lower': pred['confidence_interval']['lower'],
                'confidence_upper': pred['confidence_interval']['upper']
            })

        # sort by predicted finish
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_finish')
        predictions_df['predicted_position'] = range(1, len(predictions_df) + 1)

        # calculate race metrics
        avg_position_change = abs(predictions_df['position_change']).mean()

        circuit_data = self.circuit_stats.get(circuit_name, {})
        overtaking_difficulty = circuit_data.get('overtaking_difficulty_index', 50.0) / 100
        dnf_probability = circuit_data.get('circuit_dnf_rate', 15.0) / 100

        result = {
            'predicted_result': predictions_df.to_dict('records'),
            'race_metrics': {
                'expected_position_changes': round(avg_position_change, 2),
                'overtaking_difficulty': round(overtaking_difficulty, 2),
                'dnf_probability': round(dnf_probability, 2)
            },
            'metadata': {
                'circuit': circuit_name,
                'year': year,
                'race_number': race_number,
                'grid_size': len(grid)
            }
        }

        return result
