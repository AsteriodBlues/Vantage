"""
Prediction pipeline for F1 race outcome predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class F1PredictionPipeline:
    """
    Complete pipeline for F1 race predictions.
    Handles feature engineering from raw inputs and generates predictions.
    """

    def __init__(self, model_dir: str = 'models/production/simple_predictor_latest'):
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
        self.driver_stats = joblib.load(preprocessing_dir / 'driver_statistics.pkl')

        # load training data for default feature values
        self.train_data = pd.read_csv('data/processed/train.csv')

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
        Uses training data as template and modifies key features.
        """
        # get a sample row from training data matching circuit/team if possible
        circuit_data = self.train_data[self.train_data['circuit'] == circuit_name]

        if len(circuit_data) > 0:
            # use most recent race at this circuit
            template = circuit_data.iloc[-1:].copy()
        else:
            # fallback to any circuit
            template = self.train_data.iloc[-1:].copy()

        # override key features with provided inputs
        template['GridPosition'] = grid_position
        template['GridPosition_raw'] = grid_position
        template['race_number'] = race_number
        template['year'] = year

        # recalculate grid transformations
        template['grid_squared'] = grid_position ** 2
        template['grid_cubed'] = grid_position ** 3
        template['grid_log'] = np.log(grid_position)
        template['grid_sqrt'] = np.sqrt(grid_position)

        # grid position indicators
        template['front_row'] = 1 if grid_position <= 2 else 0
        template['top_three'] = 1 if grid_position <= 3 else 0
        template['top_five'] = 1 if grid_position <= 5 else 0
        template['top_ten'] = 1 if grid_position <= 10 else 0
        template['back_half'] = 1 if grid_position > 10 else 0
        template['grid_side'] = grid_position % 2
        template['grid_side_clean'] = 1 if grid_position % 2 == 1 else 0
        template['grid_row'] = (grid_position + 1) // 2

        # update season progress features
        template['season_progress'] = race_number / 24.0
        template['races_remaining'] = 24 - race_number
        template['early_season'] = 1 if race_number <= 8 else 0
        template['mid_season'] = 1 if 8 < race_number <= 16 else 0
        template['late_season'] = 1 if race_number > 16 else 0
        template['is_season_opener'] = 1 if race_number == 1 else 0
        template['is_season_finale'] = 1 if race_number == 24 else 0

        # update team stats if available
        if team in self.team_baselines:
            team_info = self.team_baselines[team]
            template['team_avg_finish'] = team_info.get('avg_finish', 10.0)
            template['TeamName_freq'] = team_info.get('TeamName_freq', 0.05)
            template['TeamName_target_enc'] = team_info.get('TeamName_target_enc', 10.0)
            template['TeamName_encoded'] = team_info.get('TeamName_encoded', 0)

        # update driver stats if available
        if driver in self.driver_stats:
            driver_info = self.driver_stats[driver]
            template['driver_career_races'] = driver_info.get('career_races', 50)
            template['driver_career_wins'] = driver_info.get('career_wins', 0)
            template['driver_career_podiums'] = driver_info.get('career_podiums', 0)
            template['driver_years_experience'] = driver_info.get('years_experience', 5)
            template['DriverId_freq'] = driver_info.get('DriverId_freq', 0.05)
            template['DriverId_target_enc'] = driver_info.get('DriverId_target_enc', 10.0)
            template['DriverId_encoded'] = driver_info.get('DriverId_encoded', 0)

        # update circuit stats if available
        if circuit_name in self.circuit_stats:
            circuit_info = self.circuit_stats[circuit_name]
            for key, value in circuit_info.items():
                if key in template.columns:
                    template[key] = value

        # ensure all required features exist and are in correct order
        for feature in self.feature_names:
            if feature not in template.columns:
                template[feature] = 0

        return template[self.feature_names]

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

        circuit_info = self.circuit_stats.get(circuit_name, {})
        overtaking_difficulty = circuit_info.get('overtaking_difficulty_index', 50.0) / 100
        dnf_probability = circuit_info.get('circuit_dnf_rate', 15.0) / 100

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
