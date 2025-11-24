"""
Generate comprehensive circuit analysis reports.

Analyzes all circuits in the dataset and creates detailed reports
with visualizations and statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.circuit_analysis import analyze_circuit
from src.circuit_reporting import (
    save_circuit_report,
    create_circuit_visualizations,
    rank_circuits_by_metric
)


def main():
    """Generate reports for all circuits."""
    print("="*80)
    print("CIRCUIT ANALYSIS REPORT GENERATION")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv('data/processed/train.csv')
    print(f"Loaded {len(train)} race results")

    # Load cluster info
    cluster_path = Path('results/clustering/circuit_clusters.csv')
    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        print(f"Loaded cluster assignments for {len(clusters)} circuits")
    else:
        clusters = None
        print("No cluster assignments found")

    # Get unique circuits
    circuits = train['circuit'].unique()
    print(f"\nAnalyzing {len(circuits)} circuits...")

    # Create output directories
    Path('results/reports/circuits').mkdir(parents=True, exist_ok=True)
    Path('results/figures/circuits').mkdir(parents=True, exist_ok=True)

    # Analyze all circuits
    all_analyses = {}

    for circuit in circuits:
        print(f"\n{'='*60}")
        print(f"Analyzing: {circuit}")
        print(f"{'='*60}")

        analysis = analyze_circuit(train, circuit, clusters)

        if analysis is not None:
            all_analyses[circuit] = analysis

            # Save reports
            save_circuit_report(analysis, circuit)

            # Create visualizations
            create_circuit_visualizations(train, analysis, circuit)

    print(f"\n{'='*80}")
    print("CIRCUIT ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAnalyzed {len(all_analyses)} circuits")
    print(f"Reports saved to: results/reports/circuits/")
    print(f"Figures saved to: results/figures/circuits/")

    # Generate rankings
    print(f"\n{'='*80}")
    print("CIRCUIT RANKINGS")
    print(f"{'='*80}")

    # Most processional circuits (highest pole win rate)
    print("\nMost Processional Circuits (High Pole Win Rate):")
    processional = rank_circuits_by_metric(all_analyses, 'pole_win_rate', top_n=10)
    print(processional.to_string(index=False))

    # Best overtaking circuits (high position variance)
    print("\nBest Overtaking Circuits (High Position Variance):")
    overtaking = rank_circuits_by_metric(all_analyses, 'std_change', top_n=10)
    print(overtaking.to_string(index=False))

    # Most chaotic circuits
    print("\nMost Unpredictable Circuits (Position Change StdDev):")
    chaotic = rank_circuits_by_metric(all_analyses, 'std_change', top_n=10)
    print(chaotic.to_string(index=False))

    # Save rankings
    rankings_dir = Path('results/reports/rankings')
    rankings_dir.mkdir(parents=True, exist_ok=True)

    processional.to_csv(rankings_dir / 'processional_circuits.csv', index=False)
    overtaking.to_csv(rankings_dir / 'overtaking_circuits.csv', index=False)
    chaotic.to_csv(rankings_dir / 'chaotic_circuits.csv', index=False)

    # Calculate overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    avg_pole_win = np.mean([a['notable_stats']['pole_win_rate'] for a in all_analyses.values()])
    avg_pos_change = np.mean([abs(a['position_change_stats']['mean_change']) for a in all_analyses.values()])
    avg_dnf = np.mean([a['notable_stats']['dnf_rate'] for a in all_analyses.values()])

    print(f"\nOverall Pole-to-Win Rate: {avg_pole_win:.1%}")
    print(f"Average Position Changes: ±{avg_pos_change:.2f}")
    print(f"Average DNF Rate: {avg_dnf:.1%}")

    # Find extremes
    most_processional = max(all_analyses.items(), key=lambda x: x[1]['notable_stats']['pole_win_rate'])
    most_chaotic = max(all_analyses.items(), key=lambda x: x[1]['position_change_stats']['std_change'])
    safest = min(all_analyses.items(), key=lambda x: x[1]['notable_stats']['dnf_rate'])

    print(f"\nMost Processional: {most_processional[0]} ({most_processional[1]['notable_stats']['pole_win_rate']:.1%} pole wins)")
    print(f"Most Chaotic: {most_chaotic[0]} (σ={most_chaotic[1]['position_change_stats']['std_change']:.2f})")
    print(f"Safest (Lowest DNF): {safest[0]} ({safest[1]['notable_stats']['dnf_rate']:.1%} DNF rate)")

    print(f"\n{'='*80}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
