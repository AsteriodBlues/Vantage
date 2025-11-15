"""
Data collection utilities for F1 race results.
"""

import fastf1
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import logging

from config import CACHE_DIR, START_YEAR, END_YEAR, MAX_RETRIES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(CACHE_DIR))
logger.info(f"FastF1 cache enabled at: {CACHE_DIR}")


class F1DataCollector:
    """
    Handles collection of F1 race data using the FastF1 API.
    """

    def __init__(self, start_year=START_YEAR, end_year=END_YEAR):
        """
        Initialize the data collector.

        Args:
            start_year (int): First year to collect data from
            end_year (int): Last year to collect data from
        """
        self.start_year = start_year
        self.end_year = end_year
        logger.info(f"Initialized collector for years {start_year}-{end_year}")

    def collect_single_race(self, year, round_number):
        """
        Collect results from a single race.

        Args:
            year (int): Season year
            round_number (int): Race round number

        Returns:
            pd.DataFrame: Race results with driver positions and details
        """
        try:
            # Load the race session
            session = fastf1.get_session(year, round_number, 'R')
            session.load()

            # Extract results
            results = session.results

            # Add metadata
            results['year'] = year
            results['round'] = round_number
            results['race_name'] = session.event['EventName']
            results['circuit'] = session.event['Location']
            results['date'] = session.event['EventDate']

            logger.info(f"Collected {year} Round {round_number}: {session.event['EventName']}")

            return results

        except Exception as e:
            logger.error(f"Failed to collect {year} Round {round_number}: {str(e)}")
            return None

    def collect_season_data(self, year, include_sprints=False):
        """
        Collect all race results from a single season.

        Args:
            year (int): Season year
            include_sprints (bool): Whether to include sprint race results

        Returns:
            pd.DataFrame: Combined results for entire season
        """
        logger.info(f"Starting collection for {year} season")

        # Get the season schedule
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            logger.error(f"Failed to get schedule for {year}: {str(e)}")
            return None

        # Filter for race events only
        race_schedule = schedule[schedule['EventFormat'] != 'testing']

        all_results = []
        failed_races = []

        # Collect each race with progress bar
        for idx, event in tqdm(race_schedule.iterrows(),
                              total=len(race_schedule),
                              desc=f"{year} Season"):

            round_number = event['RoundNumber']

            # Skip sprint-only events if not including sprints
            if not include_sprints and 'Sprint' in str(event['EventName']):
                continue

            # Try to collect the race
            race_results = self.collect_single_race(year, round_number)

            if race_results is not None:
                all_results.append(race_results)
            else:
                failed_races.append({
                    'year': year,
                    'round': round_number,
                    'name': event['EventName']
                })

            # Be respectful to the API
            time.sleep(0.5)

        # Report any failures
        if failed_races:
            logger.warning(f"Failed to collect {len(failed_races)} races from {year}")
            for race in failed_races:
                logger.warning(f"  - Round {race['round']}: {race['name']}")

        # Combine all results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            logger.info(f"Collected {len(combined)} driver results from {len(all_results)} races in {year}")
            return combined
        else:
            logger.error(f"No data collected for {year}")
            return None

    def collect_race_data(self, save_intermediate=True):
        """
        Main method to collect race data across all configured years.

        This orchestrates the full multi-year collection process with validation,
        intermediate saves, and comprehensive reporting.

        Args:
            save_intermediate (bool): Save progress after each year

        Returns:
            pd.DataFrame: Combined race results for all years
        """
        from config import RAW_DATA_DIR

        all_years_data = []
        failed_years = []
        collection_stats = {
            'total_races': 0,
            'total_records': 0,
            'years_collected': []
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting full data collection: {self.start_year}-{self.end_year}")
        logger.info(f"{'='*70}\n")

        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {year} season")
            logger.info(f"{'='*60}")

            season_data = self.collect_season_data(year)

            if season_data is not None:
                # Track statistics
                num_races = season_data['race_name'].nunique()
                num_records = len(season_data)

                all_years_data.append(season_data)
                collection_stats['total_races'] += num_races
                collection_stats['total_records'] += num_records
                collection_stats['years_collected'].append(year)

                logger.info(f"Year {year} summary: {num_races} races, {num_records} records")

                # Save intermediate results
                if save_intermediate:
                    output_file = RAW_DATA_DIR / f"races_{year}.csv"
                    season_data.to_csv(output_file, index=False)
                    logger.info(f"Saved to {output_file}")
            else:
                failed_years.append(year)
                logger.error(f"Failed to collect data for {year}")

        # Combine all years
        if not all_years_data:
            logger.error("No data collected from any year")
            return None

        combined = pd.concat(all_years_data, ignore_index=True)

        # Data validation
        logger.info(f"\n{'='*70}")
        logger.info("Data Validation")
        logger.info(f"{'='*70}")

        # Check for duplicates
        duplicates = combined.duplicated(subset=['year', 'round', 'DriverNumber']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate entries")

        # Verify record counts
        expected_avg = 20 * 22  # 20 drivers, ~22 races per year
        actual_avg = collection_stats['total_records'] / len(collection_stats['years_collected'])
        logger.info(f"Average records per year: {actual_avg:.0f} (expected ~{expected_avg})")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("Collection Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Years collected: {collection_stats['years_collected']}")
        logger.info(f"Total races: {collection_stats['total_races']}")
        logger.info(f"Total records: {collection_stats['total_records']}")
        logger.info(f"Unique drivers: {combined['DriverNumber'].nunique()}")
        logger.info(f"Unique circuits: {combined['circuit'].nunique()}")

        if failed_years:
            logger.warning(f"Failed years: {failed_years}")

        logger.info(f"{'='*70}\n")

        return combined

    def save_data(self, data, filename, create_backup=True):
        """
        Save collected data to CSV and pickle formats with optional backup.

        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename (without extension)
            create_backup (bool): Create backup copy
        """
        from config import RAW_DATA_DIR
        import shutil

        if data is None or len(data) == 0:
            logger.warning("No data to save")
            return

        # Remove extension if provided
        base_filename = filename.replace('.csv', '').replace('.pkl', '')

        # Save as CSV
        csv_path = RAW_DATA_DIR / f"{base_filename}.csv"
        data.to_csv(csv_path, index=False)
        csv_size = csv_path.stat().st_size / 1024

        # Save as pickle for faster loading
        pkl_path = RAW_DATA_DIR / f"{base_filename}.pkl"
        data.to_pickle(pkl_path)
        pkl_size = pkl_path.stat().st_size / 1024

        logger.info(f"Saved {len(data)} records:")
        logger.info(f"  CSV: {csv_path} ({csv_size:.2f} KB)")
        logger.info(f"  Pickle: {pkl_path} ({pkl_size:.2f} KB)")

        # Create backup
        if create_backup:
            backup_dir = RAW_DATA_DIR / "backup"
            backup_dir.mkdir(exist_ok=True)

            csv_backup = backup_dir / f"{base_filename}.csv"
            pkl_backup = backup_dir / f"{base_filename}.pkl"

            shutil.copy2(csv_path, csv_backup)
            shutil.copy2(pkl_path, pkl_backup)

            logger.info(f"Backup created in {backup_dir}")
