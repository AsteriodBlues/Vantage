"""
Data processing and cleaning utilities for F1 race data.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles cleaning and standardization of F1 race data.
    """

    # Circuit name standardization mapping
    CIRCUIT_MAPPINGS = {
        'Great Britain': 'Silverstone',
        'British Grand Prix': 'Silverstone',
        'Italy': 'Monza',
        'Italian Grand Prix': 'Monza',
        'Belgium': 'Spa-Francorchamps',
        'Belgian Grand Prix': 'Spa-Francorchamps',
        'Spain': 'Barcelona',
        'Spanish Grand Prix': 'Barcelona',
        'Monaco': 'Monte Carlo',
        'Abu Dhabi': 'Yas Marina',
        'United States': 'Austin',
        'USA': 'Austin',
        'Mexico': 'Mexico City',
        'Brazil': 'Interlagos',
        'Japan': 'Suzuka',
        'Singapore': 'Marina Bay',
        'Australia': 'Melbourne',
        'Bahrain': 'Sakhir',
        'China': 'Shanghai',
        'Azerbaijan': 'Baku',
        'Canada': 'Montreal',
        'France': 'Paul Ricard',
        'Austria': 'Red Bull Ring',
        'Hungary': 'Hungaroring',
        'Netherlands': 'Zandvoort',
        'Russia': 'Sochi',
        'Turkey': 'Istanbul',
        'Portugal': 'Portimao',
        'Saudi Arabia': 'Jeddah',
        'Qatar': 'Losail',
        'Miami': 'Miami',
        'Las Vegas': 'Las Vegas',
    }

    # Team name standardization (historical to current where applicable)
    TEAM_MAPPINGS = {
        # Aston Martin lineage
        'Racing Point': 'Racing Point',  # Keep historical for 2019-2020
        'Force India': 'Force India',     # Keep historical pre-2019

        # Alpine lineage
        'Renault': 'Renault',             # Keep historical 2018-2020

        # AlphaTauri lineage
        'Toro Rosso': 'Toro Rosso',       # Keep historical 2018-2019
        'Scuderia Toro Rosso': 'Toro Rosso',

        # Alfa Romeo variations
        'Alfa Romeo Racing': 'Alfa Romeo',
        'Alfa Romeo Racing ORLEN': 'Alfa Romeo',

        # Haas variations
        'Haas F1 Team': 'Haas',

        # Williams variations
        'Williams Racing': 'Williams',

        # McLaren variations
        'McLaren F1 Team': 'McLaren',

        # Mercedes variations
        'Mercedes-AMG Petronas F1 Team': 'Mercedes',
        'Mercedes AMG Petronas F1 Team': 'Mercedes',

        # Ferrari variations
        'Scuderia Ferrari': 'Ferrari',

        # Red Bull variations
        'Red Bull Racing': 'Red Bull',
        'Red Bull Racing Honda': 'Red Bull',
        'Red Bull Racing-TAG Heuer': 'Red Bull',
    }

    def __init__(self, remove_dnfs=False, standardize_names=True):
        """
        Initialize the data processor.

        Args:
            remove_dnfs (bool): Whether to remove DNF records (not recommended)
            standardize_names (bool): Standardize circuit and team names
        """
        self.remove_dnfs = remove_dnfs
        self.standardize_names = standardize_names
        self.cleaning_stats = {}

    def clean_data(self, data):
        """
        Comprehensive data cleaning and standardization.

        Args:
            data (pd.DataFrame): Raw race data

        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning process")
        df = data.copy()

        original_shape = df.shape
        self.cleaning_stats['original_records'] = len(df)

        # Convert data types
        df = self._convert_data_types(df)

        # Standardize names if requested
        if self.standardize_names:
            df = self._standardize_circuit_names(df)
            df = self._standardize_team_names(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove obvious errors
        df = self._remove_errors(df)

        # Add derived columns
        df = self._add_derived_columns(df)

        # Check data consistency
        self._check_consistency(df)

        final_shape = df.shape
        self.cleaning_stats['final_records'] = len(df)
        self.cleaning_stats['records_removed'] = original_shape[0] - final_shape[0]

        logger.info(f"Cleaning complete: {original_shape[0]} -> {final_shape[0]} records")

        return df

    def _convert_data_types(self, df):
        """Convert columns to appropriate data types."""
        logger.info("Converting data types")

        # Handle GridPosition
        if 'GridPosition' in df.columns:
            # Keep original for reference
            df['GridPosition_raw'] = df['GridPosition']

            # Convert to numeric, handling pit lane starts
            df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')

            # Count conversions
            self.cleaning_stats['gridposition_converted'] = df['GridPosition'].notna().sum()

        # Handle Position (finish)
        if 'Position' in df.columns:
            df['Position_raw'] = df['Position']
            df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            self.cleaning_stats['position_converted'] = df['Position'].notna().sum()

        # Handle Points
        if 'Points' in df.columns:
            df['Points'] = pd.to_numeric(df['Points'], errors='coerce')

        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    def _standardize_circuit_names(self, df):
        """Standardize circuit names across years."""
        if 'circuit' not in df.columns:
            return df

        logger.info("Standardizing circuit names")

        original_unique = df['circuit'].nunique()

        # Apply mappings
        df['circuit'] = df['circuit'].replace(self.CIRCUIT_MAPPINGS)

        final_unique = df['circuit'].nunique()
        self.cleaning_stats['circuits_before'] = original_unique
        self.cleaning_stats['circuits_after'] = final_unique

        logger.info(f"Circuit names: {original_unique} -> {final_unique} unique")

        return df

    def _standardize_team_names(self, df):
        """Standardize team names across years."""
        if 'TeamName' not in df.columns:
            return df

        logger.info("Standardizing team names")

        original_unique = df['TeamName'].nunique()

        # Apply mappings
        df['TeamName'] = df['TeamName'].replace(self.TEAM_MAPPINGS)

        final_unique = df['TeamName'].nunique()
        self.cleaning_stats['teams_before'] = original_unique
        self.cleaning_stats['teams_after'] = final_unique

        logger.info(f"Team names: {original_unique} -> {final_unique} unique")

        return df

    def _handle_missing_values(self, df):
        """Handle missing values appropriately."""
        logger.info("Handling missing values")

        # Log missing values in critical columns
        critical_cols = ['year', 'round', 'DriverNumber', 'Position', 'GridPosition']
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    logger.warning(f"{col}: {missing} missing values")

        # For DNFs, Position might be NaN - this is acceptable
        # Keep them for now, will handle in handle_dnfs method

        return df

    def _remove_errors(self, df):
        """Remove obviously erroneous records."""
        logger.info("Checking for data errors")

        errors_removed = 0

        # Check for invalid grid positions (should be 0-26)
        if 'GridPosition' in df.columns:
            invalid_grid = (df['GridPosition'] < 0) | (df['GridPosition'] > 26)
            errors_count = invalid_grid.sum()
            if errors_count > 0:
                logger.warning(f"Removing {errors_count} records with invalid grid positions")
                df = df[~invalid_grid]
                errors_removed += errors_count

        # Check for invalid finish positions
        if 'Position' in df.columns:
            # Position can be NaN for DNFs, but if present should be reasonable
            invalid_pos = ((df['Position'].notna()) &
                          ((df['Position'] < 1) | (df['Position'] > 25)))
            errors_count = invalid_pos.sum()
            if errors_count > 0:
                logger.warning(f"Removing {errors_count} records with invalid positions")
                df = df[~invalid_pos]
                errors_removed += errors_count

        self.cleaning_stats['errors_removed'] = errors_removed

        return df

    def _add_derived_columns(self, df):
        """Add useful derived columns."""
        logger.info("Adding derived columns")

        # Position change (grid to finish)
        if 'GridPosition' in df.columns and 'Position' in df.columns:
            df['position_change'] = df['GridPosition'] - df['Position']

        # Add month if date column exists
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month

        return df

    def _check_consistency(self, df):
        """Check for data consistency issues."""
        logger.info("Checking data consistency")

        # Check drivers per race
        drivers_per_race = df.groupby(['year', 'round']).size()
        unusual_counts = drivers_per_race[(drivers_per_race < 18) | (drivers_per_race > 22)]

        if len(unusual_counts) > 0:
            logger.warning(f"Found {len(unusual_counts)} races with unusual driver counts")
            self.cleaning_stats['unusual_race_counts'] = len(unusual_counts)

    def handle_dnfs(self, df):
        """
        Add DNF flags and handling logic.

        This uses the recommended approach: flag DNFs but keep all records.
        Allows flexibility in analysis to include/exclude DNFs as needed.

        Args:
            df (pd.DataFrame): Cleaned data

        Returns:
            pd.DataFrame: Data with DNF flags
        """
        logger.info("Processing DNF data")

        # Create DNF flag based on Status
        if 'Status' in df.columns:
            df['is_dnf'] = df['Status'] != 'Finished'
            df['completed_race'] = ~df['is_dnf']

            # Count DNFs
            dnf_count = df['is_dnf'].sum()
            dnf_rate = (dnf_count / len(df)) * 100

            logger.info(f"DNF Analysis:")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Finished: {df['completed_race'].sum()}")
            logger.info(f"  DNFs: {dnf_count} ({dnf_rate:.1f}%)")

            self.cleaning_stats['dnf_count'] = dnf_count
            self.cleaning_stats['dnf_rate'] = dnf_rate

            # DNF by year
            dnf_by_year = df.groupby('year')['is_dnf'].agg(['sum', 'count'])
            dnf_by_year['rate'] = (dnf_by_year['sum'] / dnf_by_year['count']) * 100
            logger.info("\nDNF rate by year:")
            for year, row in dnf_by_year.iterrows():
                logger.info(f"  {year}: {row['rate']:.1f}%")

        # If remove_dnfs flag is set, filter them out
        if self.remove_dnfs:
            original_len = len(df)
            df = df[df['completed_race']].copy()
            removed = original_len - len(df)
            logger.info(f"Removed {removed} DNF records (remove_dnfs=True)")

        return df

    def get_cleaning_summary(self):
        """
        Get summary of cleaning operations performed.

        Returns:
            dict: Cleaning statistics
        """
        return self.cleaning_stats
