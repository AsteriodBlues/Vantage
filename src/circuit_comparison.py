"""
Circuit comparison and ranking utilities.

Tools for comparing circuits and analyzing their characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def create_comparison_dashboard(
    all_analyses: Dict,
    selected_circuits: List[str],
    output_path: str = 'results/figures/circuit_comparison.png'
):
    """
    Create comprehensive comparison dashboard for selected circuits.

    Args:
        all_analyses: Dictionary of all circuit analyses
        selected_circuits: List of circuit names to compare
        output_path: Path to save figure
    """
    # Filter to selected circuits
    analyses = {c: all_analyses[c] for c in selected_circuits if c in all_analyses}

    if len(analyses) < 2:
        print("Need at least 2 circuits for comparison")
        return

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    circuit_names = list(analyses.keys())
    n_circuits = len(circuit_names)

    # 1. Pole Win Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    pole_wins = [analyses[c]['notable_stats']['pole_win_rate'] for c in circuit_names]
    bars = ax1.barh(circuit_names, pole_wins, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Pole Win Rate', fontsize=10)
    ax1.set_title('Grid Position Advantage', fontsize=11, fontweight='bold')
    ax1.set_xlim([0, 1])

    for i, (name, val) in enumerate(zip(circuit_names, pole_wins)):
        ax1.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=9)

    # 2. Position Change Variance
    ax2 = fig.add_subplot(gs[0, 1])
    variances = [analyses[c]['position_change_stats']['std_change'] for c in circuit_names]
    bars = ax2.barh(circuit_names, variances, color='coral', edgecolor='black')
    ax2.set_xlabel('Position Change StdDev', fontsize=10)
    ax2.set_title('Overtaking Opportunities', fontsize=11, fontweight='bold')

    for i, (name, val) in enumerate(zip(circuit_names, variances)):
        ax2.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=9)

    # 3. DNF Rates
    ax3 = fig.add_subplot(gs[0, 2])
    dnf_rates = [analyses[c]['notable_stats']['dnf_rate'] for c in circuit_names]
    bars = ax3.barh(circuit_names, dnf_rates, color='crimson', edgecolor='black')
    ax3.set_xlabel('DNF Rate', fontsize=10)
    ax3.set_title('Circuit Safety', fontsize=11, fontweight='bold')
    ax3.set_xlim([0, max(dnf_rates) * 1.2])

    for i, (name, val) in enumerate(zip(circuit_names, dnf_rates)):
        ax3.text(val + 0.01, i, f'{val:.1%}', va='center', fontsize=9)

    # 4. Grid Position vs Finish (first circuit)
    ax4 = fig.add_subplot(gs[1, 0])
    first_circuit = analyses[circuit_names[0]]
    grid_stats = first_circuit['grid_statistics']
    ax4.plot(grid_stats['grid_position'], grid_stats['avg_finish'],
            'bo-', linewidth=2, markersize=6)
    ax4.plot([1, 20], [1, 20], 'r--', alpha=0.5)
    ax4.set_xlabel('Grid Position', fontsize=10)
    ax4.set_ylabel('Avg Finish', fontsize=10)
    ax4.set_title(f'{circuit_names[0]}: Grid vs Finish', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Grid Position vs Finish (second circuit)
    ax5 = fig.add_subplot(gs[1, 1])
    if len(circuit_names) > 1:
        second_circuit = analyses[circuit_names[1]]
        grid_stats = second_circuit['grid_statistics']
        ax5.plot(grid_stats['grid_position'], grid_stats['avg_finish'],
                'go-', linewidth=2, markersize=6)
        ax5.plot([1, 20], [1, 20], 'r--', alpha=0.5)
        ax5.set_xlabel('Grid Position', fontsize=10)
        ax5.set_ylabel('Avg Finish', fontsize=10)
        ax5.set_title(f'{circuit_names[1]}: Grid vs Finish', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 6. Success Probabilities Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    for circuit_name in circuit_names[:3]:  # Limit to first 3 for readability
        success_probs = analyses[circuit_name]['success_probabilities']
        x_pos = range(1, min(11, len(success_probs) + 1))
        podium_probs = success_probs['podium_prob'].values[:10]
        ax6.plot(x_pos, podium_probs, 'o-', label=circuit_name, linewidth=2, markersize=4)

    ax6.set_xlabel('Grid Position', fontsize=10)
    ax6.set_ylabel('Podium Probability', fontsize=10)
    ax6.set_title('Podium Chances by Grid', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Radar Chart Comparison
    ax7 = fig.add_subplot(gs[2, :], projection='polar')

    categories = ['Pole Win', 'Overtaking', 'Safety', 'Predictability', 'Grid Impact']
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for circuit_name in circuit_names[:4]:  # Limit to first 4 for readability
        analysis = analyses[circuit_name]

        values = [
            analysis['notable_stats']['pole_win_rate'],
            analysis['position_change_stats']['std_change'] / 10,  # Normalize
            1 - analysis['notable_stats']['dnf_rate'],  # Invert so higher is better
            1 / (analysis['position_change_stats']['std_change'] + 1),  # Predictability
            analysis['grid_statistics']['avg_finish'].corr(
                analysis['grid_statistics']['grid_position']
            )
        ]
        values += values[:1]

        ax7.plot(angles, values, 'o-', linewidth=2, label=circuit_name, markersize=6)
        ax7.fill(angles, values, alpha=0.15)

    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories, fontsize=9)
    ax7.set_ylim(0, 1)
    ax7.set_title('Circuit Characteristics Comparison', fontsize=12, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax7.grid(True)

    plt.suptitle('Circuit Comparison Dashboard', fontsize=16, fontweight='bold')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison dashboard to {output_path}")


def export_comparison_table(
    all_analyses: Dict,
    output_path: str = 'results/reports/circuit_comparison_table.csv'
):
    """
    Export comprehensive comparison table for all circuits.

    Args:
        all_analyses: Dictionary of all circuit analyses
        output_path: Path to save CSV
    """
    comparison_data = []

    for circuit, analysis in all_analyses.items():
        row = {
            'Circuit': circuit,
            'Cluster': analysis['notable_stats'].get('cluster_name', 'Unknown'),
            'Total_Races': analysis['total_races'],
            'Pole_Win_Rate': analysis['notable_stats']['pole_win_rate'],
            'Front_Row_Win_Rate': analysis['notable_stats']['front_row_win_rate'],
            'Overtaking_Score': analysis['notable_stats']['overtaking_score'],
            'Avg_Pos_Change': analysis['position_change_stats']['mean_change'],
            'Pos_Change_StdDev': analysis['position_change_stats']['std_change'],
            'DNF_Rate': analysis['notable_stats']['dnf_rate'],
            'Lowest_Winner_Grid': analysis['notable_stats']['lowest_winner_grid'],
            'Highest_Podium_Grid': analysis['notable_stats']['highest_podium_grid'],
            'Max_Positions_Gained': analysis['notable_stats']['max_positions_gained']
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Pole_Win_Rate', ascending=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False, float_format='%.3f')

    print(f"Saved comparison table to {output_path}")
    return comparison_df


def create_historical_evolution_plot(
    all_analyses: Dict,
    output_path: str = 'results/figures/circuit_evolution.png'
):
    """
    Plot how circuit characteristics have evolved over time.

    Args:
        all_analyses: Dictionary of all circuit analyses
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect data across all circuits
    all_years = set()
    for analysis in all_analyses.values():
        if len(analysis['historical_trends']) > 0:
            all_years.update(analysis['historical_trends'].index.tolist())

    all_years = sorted(list(all_years))

    # 1. Pole Win Rate Evolution
    ax1 = axes[0, 0]
    for circuit, analysis in all_analyses.items():
        trends = analysis['historical_trends']
        if len(trends) > 1 and 'pole_win_rate' in trends.columns:
            ax1.plot(trends.index, trends['pole_win_rate'], 'o-',
                    label=circuit, alpha=0.6, markersize=4)

    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Pole Win Rate', fontsize=10)
    ax1.set_title('Pole Advantage Evolution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))

    # 2. Position Variance Evolution
    ax2 = axes[0, 1]
    for circuit, analysis in all_analyses.items():
        trends = analysis['historical_trends']
        if len(trends) > 1 and 'Position_std' in trends.columns:
            ax2.plot(trends.index, trends['Position_std'], 'o-',
                    label=circuit, alpha=0.6, markersize=4)

    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Position StdDev', fontsize=10)
    ax2.set_title('Racing Quality Evolution', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. DNF Rate Evolution
    ax3 = axes[1, 0]
    for circuit, analysis in all_analyses.items():
        trends = analysis['historical_trends']
        if len(trends) > 1 and 'is_dnf_mean' in trends.columns:
            ax3.plot(trends.index, trends['is_dnf_mean'], 'o-',
                    label=circuit, alpha=0.6, markersize=4)

    ax3.set_xlabel('Year', fontsize=10)
    ax3.set_ylabel('DNF Rate', fontsize=10)
    ax3.set_title('Reliability Evolution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Average Position Change Evolution
    ax4 = axes[1, 1]
    for circuit, analysis in all_analyses.items():
        trends = analysis['historical_trends']
        if len(trends) > 1 and 'position_change_mean' in trends.columns:
            ax4.plot(trends.index, trends['position_change_mean'], 'o-',
                    label=circuit, alpha=0.6, markersize=4)

    ax4.set_xlabel('Year', fontsize=10)
    ax4.set_ylabel('Avg Position Change', fontsize=10)
    ax4.set_title('Overtaking Evolution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Circuit Characteristics Evolution Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved evolution plot to {output_path}")


def identify_circuit_archetypes(all_analyses: Dict) -> Dict[str, List[str]]:
    """
    Group circuits into archetypes based on characteristics.

    Args:
        all_analyses: Dictionary of all circuit analyses

    Returns:
        Dictionary mapping archetype names to circuit lists
    """
    archetypes = {
        'Processional (High Grid Advantage)': [],
        'Overtaking-Friendly (Low Grid Advantage)': [],
        'Chaotic (High Variance)': [],
        'Predictable (Low Variance)': [],
        'Dangerous (High DNF)': [],
        'Safe (Low DNF)': []
    }

    for circuit, analysis in all_analyses.items():
        pole_win = analysis['notable_stats']['pole_win_rate']
        variance = analysis['position_change_stats']['std_change']
        dnf_rate = analysis['notable_stats']['dnf_rate']

        if pole_win > 0.6:
            archetypes['Processional (High Grid Advantage)'].append(circuit)
        elif pole_win < 0.3:
            archetypes['Overtaking-Friendly (Low Grid Advantage)'].append(circuit)

        if variance > 4.0:
            archetypes['Chaotic (High Variance)'].append(circuit)
        elif variance < 2.5:
            archetypes['Predictable (Low Variance)'].append(circuit)

        if dnf_rate > 0.6:
            archetypes['Dangerous (High DNF)'].append(circuit)
        elif dnf_rate < 0.4:
            archetypes['Safe (Low DNF)'].append(circuit)

    return archetypes
