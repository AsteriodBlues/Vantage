"""
Create circuit comparison visualizations and archetype analysis.

Compares circuits across different dimensions and identifies archetypes.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.circuit_analysis import analyze_circuit
from src.circuit_comparison import (
    create_comparison_dashboard,
    export_comparison_table,
    create_historical_evolution_plot,
    identify_circuit_archetypes
)


def main():
    """Generate circuit comparisons and archetypes."""
    print("="*80)
    print("CIRCUIT COMPARISON ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv('data/processed/train.csv')

    # Load cluster info
    try:
        clusters = pd.read_csv('results/clustering/circuit_clusters.csv')
    except:
        clusters = None

    # Get unique circuits
    circuits = train['circuit'].unique()

    # Analyze all circuits
    print(f"\nAnalyzing {len(circuits)} circuits...")
    all_analyses = {}

    for circuit in circuits:
        analysis = analyze_circuit(train, circuit, clusters)
        if analysis is not None:
            all_analyses[circuit] = analysis

    print(f"Successfully analyzed {len(all_analyses)} circuits")

    # Select interesting circuits for detailed comparison
    selected_circuits = [
        'Monte Carlo',      # Processional street circuit
        'Monza',            # High-speed, low downforce
        'Spa-Francorchamps', # Classic, weather variable
        'Sakhir',           # Modern, good racing
        'Marina Bay',       # Street with overtaking
        'Silverstone'       # High-speed classic
    ]

    # Filter to available circuits
    selected_circuits = [c for c in selected_circuits if c in all_analyses]

    print(f"\n{'='*80}")
    print("CREATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}")

    # Create comparison dashboard
    create_comparison_dashboard(
        all_analyses,
        selected_circuits,
        output_path='results/figures/circuit_comparison_dashboard.png'
    )

    # Export comparison table
    comparison_df = export_comparison_table(
        all_analyses,
        output_path='results/reports/circuit_comparison_table.csv'
    )

    print(f"\n{'='*80}")
    print("CIRCUIT COMPARISON TABLE (Top 10 by Pole Win Rate)")
    print(f"{'='*80}")
    print(comparison_df.head(10)[['Circuit', 'Pole_Win_Rate', 'Pos_Change_StdDev', 'DNF_Rate']].to_string(index=False))

    # Create historical evolution plot
    create_historical_evolution_plot(
        all_analyses,
        output_path='results/figures/circuit_evolution.png'
    )

    # Identify archetypes
    print(f"\n{'='*80}")
    print("CIRCUIT ARCHETYPES")
    print(f"{'='*80}")

    archetypes = identify_circuit_archetypes(all_analyses)

    for archetype_name, circuit_list in archetypes.items():
        if len(circuit_list) > 0:
            print(f"\n{archetype_name}:")
            for circuit in circuit_list:
                analysis = all_analyses[circuit]
                print(f"  • {circuit}")
                print(f"    Pole Win: {analysis['notable_stats']['pole_win_rate']:.1%}, " +
                      f"Variance: {analysis['position_change_stats']['std_change']:.2f}, " +
                      f"DNF: {analysis['notable_stats']['dnf_rate']:.1%}")

    # Save archetypes
    archetype_data = []
    for archetype_name, circuit_list in archetypes.items():
        for circuit in circuit_list:
            archetype_data.append({
                'circuit': circuit,
                'archetype': archetype_name
            })

    if len(archetype_data) > 0:
        archetype_df = pd.DataFrame(archetype_data)
        archetype_df.to_csv('results/reports/circuit_archetypes.csv', index=False)
        print(f"\nSaved archetypes to results/reports/circuit_archetypes.csv")

    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  • results/figures/circuit_comparison_dashboard.png")
    print("  • results/figures/circuit_evolution.png")
    print("  • results/reports/circuit_comparison_table.csv")
    print("  • results/reports/circuit_archetypes.csv")


if __name__ == "__main__":
    main()
