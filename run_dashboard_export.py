"""
Export circuit analysis data for dashboards and visualization.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.circuit_analysis import analyze_circuit
from src.dashboard_export import export_dashboard_data, export_circuit_metadata


def main():
    """Export dashboard data."""
    print("="*80)
    print("DASHBOARD DATA EXPORT")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv('data/processed/train.csv')

    # Load cluster info
    try:
        clusters = pd.read_csv('results/clustering/circuit_clusters.csv')
    except:
        clusters = None

    # Analyze all circuits
    circuits = train['circuit'].unique()
    print(f"Analyzing {len(circuits)} circuits...")

    all_analyses = {}
    for circuit in circuits:
        analysis = analyze_circuit(train, circuit, clusters)
        if analysis is not None:
            all_analyses[circuit] = analysis

    print(f"Successfully analyzed {len(all_analyses)} circuits")

    # Export dashboard data
    print(f"\n{'='*80}")
    print("EXPORTING DASHBOARD DATA")
    print(f"{'='*80}")

    dashboard_data = export_dashboard_data(
        all_analyses,
        output_path='results/data/dashboard_circuit_data.json'
    )

    # Export metadata
    metadata = export_circuit_metadata(
        all_analyses,
        output_path='results/data/circuit_metadata.csv'
    )

    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  • results/data/dashboard_circuit_data.json")
    print("  • results/data/circuit_metadata.csv")


if __name__ == "__main__":
    main()
