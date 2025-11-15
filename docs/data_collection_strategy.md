# Data Collection Strategy for 20 Years (2005-2024)

## Overview

Collecting 20 years of Formula 1 data requires a robust, resumable strategy to handle ~380 races and 7,600+ driver results.

## Data Availability

Based on FastF1 library testing:
- **Full data**: 2018-2024 (~7 seasons, 150+ races)
- **Limited data**: 2005-2017 (~13 seasons, may require alternative sources)

## Collection Approach

### Chunked Collection
- Collect in 5-year chunks to manage memory and allow interruption recovery
- Chunk 1: 2020-2024 (most recent, complete data)
- Chunk 2: 2015-2019 (hybrid era)
- Chunk 3: 2010-2014 (late V8/early hybrid)
- Chunk 4: 2005-2009 (V10/V8 transition)

### Caching Strategy
- Enable FastF1 cache to avoid re-downloading data
- Estimated cache size: 2-4 GB for 20 years
- Cache location: `data/cache/`
- First download: 30-60 minutes per 5-year chunk
- Subsequent loads: seconds (from cache)

### Error Handling
- Wrap each race collection in try-except
- Log missing or failed races
- Continue collection even if individual races fail
- Save progress after each completed year

### Resumability
- Save intermediate results after each year/chunk
- Check for existing data before re-downloading
- Allow script to resume from last successful collection

## Target Data Points

### Essential Columns
- `GridPosition`: Qualifying result
- `Position`: Race finish position
- `DriverNumber`: Driver identifier
- `FullName`: Driver name
- `TeamName`: Constructor
- `Points`: Championship points awarded
- `Status`: Race completion status (Finished, DNF reason)

### Derived Fields
- `circuit_name`: Track identifier
- `race_date`: Event date
- `year`: Season
- `round`: Race number in season
- `is_dnf`: Boolean for Did Not Finish

## Data Volume Estimates

- **Years**: 2005-2024 (20 seasons)
- **Races per year**: 16-23 (average ~19)
- **Total races**: ~380
- **Drivers per race**: 20-24 (average ~22)
- **Total records**: ~8,360 driver-race combinations
- **Storage**: ~10-20 MB raw CSV, ~50-100 MB with features

## Progress Tracking

- Use `tqdm` for progress bars
- Print summary after each year:
  - Races collected
  - Records added
  - Missing/failed races
  - Time elapsed
  - Estimated time remaining

## Missing Data Handling

### For older years (2005-2017):
1. Attempt FastF1 first
2. If unavailable, document missing years
3. Consider alternative sources:
   - Ergast API (historical F1 data)
   - Manual CSV compilation from Wikipedia/official sources
4. Focus analysis on 2018-2024 if older data too sparse

## File Organization

```
data/
├── raw/
│   ├── races_2020_2024.csv
│   ├── races_2015_2019.csv
│   ├── races_2010_2014.csv
│   └── races_2005_2009.csv
├── processed/
│   └── all_races_combined.csv
└── cache/
    └── (FastF1 cache files)
```

## Implementation Steps

1. Create `F1DataCollector` class in `src/data_collection.py`
2. Implement methods:
   - `collect_single_race(year, round)`
   - `collect_season(year)`
   - `collect_year_range(start, end)`
   - `save_progress(data, filename)`
   - `load_existing_data(filename)`
3. Add comprehensive logging
4. Create progress tracking with tqdm
5. Test with single year before full collection
6. Run full collection in chunks

## Estimated Timeline

- Setup and testing: 1 hour
- First full collection (2018-2024): 30-45 minutes
- Extended collection (2005-2017): 1-2 hours (if data available)
- Total first run: 2-3 hours
- Subsequent runs: <5 minutes (cached)

## Next Steps

1. Implement `F1DataCollector` class
2. Test with 2024 season only
3. Validate data quality
4. Run full collection
5. Merge and clean data
6. Begin exploratory analysis
