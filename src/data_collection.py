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

    def collect_multi_year_data(self, save_intermediate=True):
        """
        Collect data across multiple years.

        Args:
            save_intermediate (bool): Save results after each year

        Returns:
            pd.DataFrame: Combined results for all years
        """
        all_years_data = []

        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {year} season")
            logger.info(f"{'='*60}")

            season_data = self.collect_season_data(year)

            if season_data is not None:
                all_years_data.append(season_data)

                # Save intermediate results
                if save_intermediate:
                    from config import RAW_DATA_DIR
                    output_file = RAW_DATA_DIR / f"races_{year}.csv"
                    season_data.to_csv(output_file, index=False)
                    logger.info(f"Saved {year} data to {output_file}")

        # Combine all years
        if all_years_data:
            combined = pd.concat(all_years_data, ignore_index=True)
            logger.info(f"\nTotal collection: {len(combined)} results from {len(all_years_data)} seasons")
            return combined
        else:
            logger.error("No data collected")
            return None

    def save_data(self, data, filename):
        """
        Save collected data to CSV.

        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        from config import RAW_DATA_DIR

        if data is None or len(data) == 0:
            logger.warning("No data to save")
            return

        output_path = RAW_DATA_DIR / filename
        data.to_csv(output_path, index=False)
        logger.info(f"Saved {len(data)} records to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
