#!/usr/bin/env python3
"""
Command-line interface for F1 race predictions.
"""

import argparse
import json
import sys
from pathlib import Path
from prediction_pipeline import F1PredictionPipeline


def predict_single(args):
    """Predict finish position for a single driver."""
    pipeline = F1PredictionPipeline(args.model_dir)

    result = pipeline.predict(
        grid_position=args.grid,
        circuit_name=args.circuit,
        team=args.team,
        driver=args.driver,
        year=args.year,
        race_number=args.race_number
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{args.driver} ({args.team})")
        print(f"{'='*50}")
        print(f"Starting Position: P{args.grid}")
        print(f"Predicted Finish:  P{result['predicted_finish_rounded']} ({result['predicted_finish']:.2f})")
        print(f"Position Change:   {result['position_change']:+.1f}")
        print(f"Confidence Range:  P{result['confidence_interval']['lower']:.1f} - P{result['confidence_interval']['upper']:.1f}")
        print(f"\nProbabilities:")
        print(f"  Win:    {result['probabilities']['win']*100:5.1f}%")
        print(f"  Podium: {result['probabilities']['podium']*100:5.1f}%")
        print(f"  Points: {result['probabilities']['points']*100:5.1f}%")
        print()


def predict_grid(args):
    """Predict race result from full starting grid."""
    pipeline = F1PredictionPipeline(args.model_dir)

    # load grid from file
    with open(args.grid_file, 'r') as f:
        data = json.load(f)

    result = pipeline.predict_race_result(
        circuit_name=data.get('circuit', args.circuit),
        year=data.get('year', args.year),
        race_number=data.get('race_number', args.race_number),
        grid=data['grid']
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{data.get('circuit', args.circuit)} - Race {data.get('race_number', args.race_number)}")
        print(f"{'='*80}")
        print(f"{'Pos':<5} {'Driver':<20} {'Team':<15} {'Grid':<6} {'Change':<8} {'Confidence'}")
        print(f"{'-'*80}")

        for entry in result['predicted_result']:
            driver = entry['driver']
            team = entry['team']
            grid = entry['grid_position']
            pred_pos = entry['predicted_position']
            change = entry['position_change']
            conf_range = f"P{entry['confidence_lower']:.0f}-{entry['confidence_upper']:.0f}"

            change_str = f"{change:+.1f}" if change != 0 else "  -"
            print(f"P{pred_pos:<4} {driver:<20} {team:<15} P{grid:<5} {change_str:<8} {conf_range}")

        print(f"\nRace Metrics:")
        print(f"  Expected position changes: {result['race_metrics']['expected_position_changes']:.1f}")
        print(f"  Overtaking difficulty:     {result['race_metrics']['overtaking_difficulty']:.2f}")
        print(f"  DNF probability:           {result['race_metrics']['dnf_probability']:.1%}")
        print()


def interactive_mode(args):
    """Interactive prediction mode."""
    pipeline = F1PredictionPipeline(args.model_dir)

    print("\nF1 Race Prediction - Interactive Mode")
    print("="*50)
    print("Enter driver details (or 'quit' to exit)")

    while True:
        print()
        try:
            circuit = input("Circuit name: ").strip()
            if circuit.lower() == 'quit':
                break

            driver = input("Driver name: ").strip()
            if driver.lower() == 'quit':
                break

            team = input("Team name: ").strip()
            if team.lower() == 'quit':
                break

            grid = int(input("Grid position (1-20): ").strip())
            if grid < 1 or grid > 20:
                print("Grid position must be between 1 and 20")
                continue

            year = int(input("Year (default 2024): ").strip() or "2024")
            race_num = int(input("Race number (default 1): ").strip() or "1")

            result = pipeline.predict(
                grid_position=grid,
                circuit_name=circuit,
                team=team,
                driver=driver,
                year=year,
                race_number=race_num
            )

            print(f"\n{'-'*50}")
            print(f"Predicted Finish:  P{result['predicted_finish_rounded']} ({result['predicted_finish']:.2f})")
            print(f"Position Change:   {result['position_change']:+.1f}")
            print(f"Confidence Range:  P{result['confidence_interval']['lower']:.1f} - P{result['confidence_interval']['upper']:.1f}")
            print(f"Win Probability:   {result['probabilities']['win']*100:.1f}%")
            print(f"{'-'*50}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='F1 Race Prediction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single driver
  python predict_cli.py single --driver "Max Verstappen" --team "Red Bull" \\
      --circuit "Monaco" --grid 1

  # Predict full race from grid file
  python predict_cli.py grid --grid-file examples/example_grid.json

  # Interactive mode
  python predict_cli.py interactive
        """
    )

    parser.add_argument('--model-dir', default='models/production/simple_predictor_latest',
                       help='Path to model directory')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # single prediction
    single_parser = subparsers.add_parser('single', help='Predict single driver finish position')
    single_parser.add_argument('--driver', required=True, help='Driver name')
    single_parser.add_argument('--team', required=True, help='Team name')
    single_parser.add_argument('--circuit', required=True, help='Circuit name')
    single_parser.add_argument('--grid', type=int, required=True, help='Grid position (1-20)')
    single_parser.add_argument('--year', type=int, default=2024, help='Race year')
    single_parser.add_argument('--race-number', type=int, default=1, help='Race number in season')

    # full grid prediction
    grid_parser = subparsers.add_parser('grid', help='Predict full race result from grid')
    grid_parser.add_argument('--grid-file', required=True, help='Path to grid JSON file')
    grid_parser.add_argument('--circuit', help='Override circuit from file')
    grid_parser.add_argument('--year', type=int, default=2024, help='Race year')
    grid_parser.add_argument('--race-number', type=int, default=1, help='Race number in season')

    # interactive mode
    subparsers.add_parser('interactive', help='Interactive prediction mode')

    args = parser.parse_args()

    if args.command == 'single':
        predict_single(args)
    elif args.command == 'grid':
        predict_grid(args)
    elif args.command == 'interactive':
        interactive_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
