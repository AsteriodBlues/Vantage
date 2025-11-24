"""
Circuit analysis reporting and visualization.

Generates detailed reports and visualizations for individual circuits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict


def save_circuit_report(analysis: Dict, circuit_name: str, output_dir: str = 'results/reports/circuits'):
    """
    Save circuit analysis to CSV files.

    Args:
        analysis: Circuit analysis dictionary
        circuit_name: Name of circuit
        output_dir: Base output directory
    """
    circuit_dir = Path(output_dir) / circuit_name.replace(" ", "_").replace("/", "_")
    circuit_dir.mkdir(parents=True, exist_ok=True)

    # Save each component
    analysis['transition_matrix'].to_csv(circuit_dir / 'transition_matrix.csv')
    analysis['grid_statistics'].to_csv(circuit_dir / 'grid_statistics.csv', index=False)
    analysis['success_probabilities'].to_csv(circuit_dir / 'success_probabilities.csv')
    analysis['historical_trends'].to_csv(circuit_dir / 'historical_trends.csv')
    analysis['team_performance'].to_csv(circuit_dir / 'team_performance.csv')

    # Save summary statistics
    summary = {
        'metric': [
            'Total Races', 'Total Entries', 'Years Active', 'Pole Win Rate',
            'Avg Position Change', 'DNF Rate', 'Overtaking Score',
            'Front Row Win Rate', 'Lowest Winner Grid'
        ],
        'value': [
            analysis['total_races'],
            analysis['total_entries'],
            len(analysis['years_present']),
            f"{analysis['notable_stats']['pole_win_rate']:.3f}",
            f"{analysis['position_change_stats']['mean_change']:.3f}",
            f"{analysis['notable_stats']['dnf_rate']:.3f}",
            f"{analysis['notable_stats']['overtaking_score']:.3f}",
            f"{analysis['notable_stats']['front_row_win_rate']:.3f}",
            analysis['notable_stats']['lowest_winner_grid']
        ]
    }
    pd.DataFrame(summary).to_csv(circuit_dir / 'summary.csv', index=False)

    print(f"Saved reports for {circuit_name}")


def create_circuit_visualizations(
    df: pd.DataFrame,
    analysis: Dict,
    circuit_name: str,
    output_dir: str = 'results/figures/circuits'
):
    """
    Create comprehensive visualizations for a circuit.

    Args:
        df: Full dataset
        analysis: Circuit analysis dictionary
        circuit_name: Name of circuit
        output_dir: Output directory for figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    circuit_data = df[df['circuit'] == circuit_name]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Transition Matrix Heatmap
    ax1 = fig.add_subplot(gs[0:2, 0])
    sns.heatmap(
        analysis['transition_matrix'].values,
        cmap='RdYlGn_r',
        center=0,
        annot=False,
        cbar_kws={'label': 'Probability'},
        ax=ax1,
        vmin=0,
        vmax=1
    )
    ax1.set_title(f'{circuit_name}\nGrid to Finish Heatmap', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Finish Position')
    ax1.set_ylabel('Grid Position')
    ax1.set_xticklabels([f'{i+1}' for i in range(20)], rotation=45)
    ax1.set_yticklabels([f'{i+1}' for i in range(20)], rotation=0)

    # 2. Average Finish by Grid Position
    ax2 = fig.add_subplot(gs[0, 1])
    grid_stats = analysis['grid_statistics']
    ax2.plot(grid_stats['grid_position'], grid_stats['avg_finish'], 'b-', linewidth=2, marker='o')
    ax2.fill_between(
        grid_stats['grid_position'],
        grid_stats['avg_finish'] - grid_stats['std_finish'],
        grid_stats['avg_finish'] + grid_stats['std_finish'],
        alpha=0.3
    )
    ax2.plot([1, 20], [1, 20], 'r--', alpha=0.5, label='No change')
    ax2.set_xlabel('Grid Position', fontsize=10)
    ax2.set_ylabel('Average Finish Position', fontsize=10)
    ax2.set_title('Grid vs Finish Position', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Success Probabilities
    ax3 = fig.add_subplot(gs[0, 2])
    success_probs = analysis['success_probabilities']
    x_pos = range(1, len(success_probs) + 1)

    ax3.plot(x_pos, success_probs['win_prob'], 'go-', label='Win', linewidth=2, markersize=4)
    ax3.plot(x_pos, success_probs['podium_prob'], 'bo-', label='Podium', linewidth=2, markersize=4)
    ax3.plot(x_pos, success_probs['points_prob'], 'ro-', label='Points', linewidth=2, markersize=4)

    ax3.set_xlabel('Grid Position', fontsize=10)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.set_title('Success Probabilities', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # 4. Position Change Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    position_changes = circuit_data[circuit_data['is_dnf'] == 0]['position_change']
    if len(position_changes) > 0:
        ax4.hist(position_changes, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax4.axvline(position_changes.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {position_changes.mean():.1f}')
        ax4.set_xlabel('Position Change', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Position Change Distribution', fontsize=11, fontweight='bold')
        ax4.legend()

    # 5. Historical Trends
    ax5 = fig.add_subplot(gs[1, 2])
    trends = analysis['historical_trends']
    if len(trends) > 0:
        ax5.plot(trends.index, trends['pole_win_rate'], 'g^-', label='Pole Win Rate',
                markersize=8, linewidth=2)
        ax5.plot(trends.index, trends['is_dnf_mean'], 'rs-', label='DNF Rate',
                markersize=6, linewidth=2)
        ax5.set_xlabel('Year', fontsize=10)
        ax5.set_ylabel('Rate', fontsize=10)
        ax5.set_title('Historical Trends', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. DNF Rate by Grid Position
    ax6 = fig.add_subplot(gs[2, 0])
    dnf_by_grid = circuit_data.groupby('GridPosition')['is_dnf'].mean()
    if len(dnf_by_grid) > 0:
        ax6.bar(dnf_by_grid.index, dnf_by_grid.values, color='coral', edgecolor='black')
        ax6.set_xlabel('Grid Position', fontsize=10)
        ax6.set_ylabel('DNF Rate', fontsize=10)
        ax6.set_title('DNF Rate by Starting Position', fontsize=11, fontweight='bold')
        ax6.axhline(dnf_by_grid.mean(), color='red', linestyle='--', alpha=0.5,
                   label=f'Avg: {dnf_by_grid.mean():.2f}')
        ax6.legend()

    # 7. Team Performance
    ax7 = fig.add_subplot(gs[2, 1])
    team_perf = analysis['team_performance'].head(10)

    x = range(len(team_perf))
    width = 0.35

    if 'GridPosition_mean' in team_perf.columns and 'Position_mean' in team_perf.columns:
        ax7.barh([i - width/2 for i in x], team_perf['GridPosition_mean'],
                width, label='Avg Grid', color='skyblue', edgecolor='black')
        ax7.barh([i + width/2 for i in x], team_perf['Position_mean'],
                width, label='Avg Finish', color='lightcoral', edgecolor='black')
        ax7.set_yticks(x)
        ax7.set_yticklabels(team_perf.index, fontsize=8)
        ax7.set_xlabel('Position', fontsize=10)
        ax7.set_title('Team Performance (Top 10)', fontsize=11, fontweight='bold')
        ax7.legend()
        ax7.invert_xaxis()

    # 8. Win/Podium/Points by Grid
    ax8 = fig.add_subplot(gs[2, 2])
    grid_stats_plot = analysis['grid_statistics'].head(15)
    x_grid = grid_stats_plot['grid_position']

    ax8.plot(x_grid, grid_stats_plot['win_rate'] * 100, 'go-', label='Win %',
            linewidth=2, markersize=6)
    ax8.plot(x_grid, grid_stats_plot['podium_rate'] * 100, 'bo-', label='Podium %',
            linewidth=2, markersize=6)
    ax8.plot(x_grid, grid_stats_plot['points_rate'] * 100, 'ro-', label='Points %',
            linewidth=2, markersize=6)

    ax8.set_xlabel('Grid Position', fontsize=10)
    ax8.set_ylabel('Success Rate (%)', fontsize=10)
    ax8.set_title('Success Rates by Grid', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Key Statistics Box
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    cluster_name = analysis['notable_stats'].get('cluster_name', 'Unknown')
    stats_text = f"""
KEY STATISTICS: {circuit_name.upper()}

Cluster: {cluster_name}
Total Races: {analysis['total_races']} ({analysis['years_present'][0]}-{analysis['years_present'][-1]})

OVERTAKING METRICS:
  • Overtaking Score: {analysis['notable_stats']['overtaking_score']:.2f}
  • Avg Position Changes: ±{abs(analysis['position_change_stats']['mean_change']):.2f}
  • Position Change StdDev: {analysis['position_change_stats']['std_change']:.2f}

GRID ADVANTAGE:
  • Pole Win Rate: {analysis['notable_stats']['pole_win_rate']:.1%}
  • Front Row Win Rate: {analysis['notable_stats']['front_row_win_rate']:.1%}
  • Top 3 Grid Win Rate: {analysis['notable_stats']['top3_win_rate']:.1%}

NOTABLE RECORDS:
  • Lowest Winner Grid: P{analysis['notable_stats']['lowest_winner_grid']}
  • Highest Podium Grid: P{analysis['notable_stats']['highest_podium_grid']}
  • Max Positions Gained: +{analysis['notable_stats']['max_positions_gained']}
  • DNF Rate: {analysis['notable_stats']['dnf_rate']:.1%}
    """

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Circuit Analysis: {circuit_name}', fontsize=16, fontweight='bold')

    output_file = output_path / f"{circuit_name.replace(' ', '_').replace('/', '_')}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization for {circuit_name}")


def rank_circuits_by_metric(
    all_analyses: Dict,
    metric: str = 'overtaking_score',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Rank circuits by a specific metric.

    Args:
        all_analyses: Dictionary of all circuit analyses
        metric: Metric to rank by
        top_n: Number of top circuits to return

    Returns:
        DataFrame with rankings
    """
    rankings = []

    for circuit, analysis in all_analyses.items():
        if metric in analysis['notable_stats']:
            value = analysis['notable_stats'][metric]
        elif metric in analysis['position_change_stats']:
            value = analysis['position_change_stats'][metric]
        else:
            continue

        rankings.append({
            'circuit': circuit,
            'value': value,
            'cluster': analysis['notable_stats'].get('cluster_name', 'Unknown')
        })

    rankings_df = pd.DataFrame(rankings).sort_values('value', ascending=False).head(top_n)
    return rankings_df


def compare_circuits(
    circuit1: str,
    circuit2: str,
    all_analyses: Dict
) -> pd.DataFrame:
    """
    Create side-by-side comparison of two circuits.

    Args:
        circuit1: First circuit name
        circuit2: Second circuit name
        all_analyses: Dictionary of all circuit analyses

    Returns:
        Comparison DataFrame
    """
    if circuit1 not in all_analyses or circuit2 not in all_analyses:
        return pd.DataFrame()

    comparison = pd.DataFrame({
        circuit1: {
            'Pole Win Rate': all_analyses[circuit1]['notable_stats']['pole_win_rate'],
            'Overtaking Score': all_analyses[circuit1]['notable_stats']['overtaking_score'],
            'Avg Pos Change': all_analyses[circuit1]['position_change_stats']['mean_change'],
            'Pos Change StdDev': all_analyses[circuit1]['position_change_stats']['std_change'],
            'DNF Rate': all_analyses[circuit1]['notable_stats']['dnf_rate'],
            'Front Row Win %': all_analyses[circuit1]['notable_stats']['front_row_win_rate'],
            'Lowest Winner Grid': all_analyses[circuit1]['notable_stats']['lowest_winner_grid']
        },
        circuit2: {
            'Pole Win Rate': all_analyses[circuit2]['notable_stats']['pole_win_rate'],
            'Overtaking Score': all_analyses[circuit2]['notable_stats']['overtaking_score'],
            'Avg Pos Change': all_analyses[circuit2]['position_change_stats']['mean_change'],
            'Pos Change StdDev': all_analyses[circuit2]['position_change_stats']['std_change'],
            'DNF Rate': all_analyses[circuit2]['notable_stats']['dnf_rate'],
            'Front Row Win %': all_analyses[circuit2]['notable_stats']['front_row_win_rate'],
            'Lowest Winner Grid': all_analyses[circuit2]['notable_stats']['lowest_winner_grid']
        }
    }).T

    return comparison
