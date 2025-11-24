"""
VANTAGE F1 Prediction Dashboard
Interactive web interface for F1 race predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.prediction_pipeline import F1PredictionPipeline
except ImportError:
    F1PredictionPipeline = None

# Page configuration
st.set_page_config(
    page_title="VANTAGE F1 - Grid Position Advantage Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    h1 {
        color: #1976D2;
        font-family: 'Arial Black', sans-serif;
    }
    .stButton > button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_pipeline' not in st.session_state:
    st.session_state.prediction_pipeline = None
    st.session_state.predictions_history = []
    st.session_state.circuit_data = {}
    st.session_state.model_loaded = False


@st.cache_resource
def load_prediction_pipeline():
    """Load the ML model pipeline"""
    if F1PredictionPipeline is None:
        return None

    try:
        pipeline = F1PredictionPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_circuit_data():
    """Load circuit statistics"""
    try:
        circuit_stats = joblib.load('models/preprocessing/circuit_statistics.pkl')
        return circuit_stats
    except:
        return {}


@st.cache_data
def load_historical_data():
    """Load historical race data"""
    try:
        df = pd.read_csv('data/processed/train.csv')
        return df
    except:
        return pd.DataFrame()


def main():
    """Main application entry point"""

    # Sidebar
    with st.sidebar:
        st.markdown("### 🏎️ VANTAGE F1")
        st.markdown("**V**aluating **A**dvantage **N**umerically **T**hrough **A**nalysis of **G**rid **E**ffects")

        st.markdown("---")
        st.markdown("### 📊 Project Stats")
        st.metric("Races Analyzed", "780")
        st.metric("Years Covered", "2018-2024")
        st.metric("Model Accuracy (MAE)", "0.57 positions")
        st.metric("Test R²", "0.971")

        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[GitHub Repository](https://github.com/AsteriodBlues/Vantage)")
        st.markdown("[API Documentation](../docs/api_specification.md)")
        st.markdown("[Model Performance](../docs/model_performance.md)")

    # Main content
    st.title("🏁 VANTAGE F1 Prediction Dashboard")
    st.markdown("**Quantifying Grid Position Advantage in Formula 1 Racing**")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "🔮 Predictions",
        "🏎️ Circuit Analysis",
        "📈 Visualizations",
        "ℹ️ About"
    ])

    with tab1:
        home_page()

    with tab2:
        prediction_page()

    with tab3:
        circuit_analysis_page()

    with tab4:
        visualization_page()

    with tab5:
        about_page()


def home_page():
    """Create the home/landing page"""

    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2>Welcome to VANTAGE F1</h2>
            <p style='font-size: 18px;'>
                Advanced machine learning for Formula 1 race prediction based on
                starting grid positions and comprehensive race analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics row
    st.markdown("### 🏆 Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>Prediction Accuracy</h4>
            <h2 style='color: #1976D2;'>±0.57</h2>
            <p>Average positions error</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>Within 1 Position</h4>
            <h2 style='color: #388E3C;'>72%</h2>
            <p>Prediction success rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>R² Score</h4>
            <h2 style='color: #7B1FA2;'>0.971</h2>
            <p>Variance explained</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h4>Features</h4>
            <h2 style='color: #F57C00;'>136</h2>
            <p>Engineered features</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature showcase
    st.markdown("### 🎯 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Capabilities")
        st.markdown("""
        - **Position Prediction**: Forecast finish position from any grid slot
        - **Win Probability**: Calculate chances of victory
        - **Podium Probability**: Estimate top-3 finish likelihood
        - **Full Grid Simulation**: Predict entire race result
        - **Confidence Intervals**: Uncertainty quantification
        """)

        # Quick demo chart
        demo_data = pd.DataFrame({
            'Grid Position': range(1, 21),
            'Average Finish': [2.8, 4.1, 4.9, 6.2, 7.1, 8.3, 9.2, 10.1,
                              10.8, 11.5, 12.1, 12.7, 13.3, 13.8,
                              14.3, 14.8, 15.2, 15.6, 16.1, 16.5]
        })

        fig = px.line(demo_data, x='Grid Position', y='Average Finish',
                     title='Grid Position vs Average Finish',
                     markers=True)
        fig.add_trace(go.Scatter(x=[1, 20], y=[1, 20],
                                mode='lines',
                                line=dict(dash='dash', color='red'),
                                name='No Change Line'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Circuit Intelligence")
        st.markdown("""
        - **18 Circuits Analyzed**: Historic F1 calendar
        - **Circuit Clustering**: Automatic track categorization
        - **Overtaking Difficulty**: Quantified for each circuit
        - **Historical Patterns**: 7 years of race data
        - **Regulation Impact**: Pre/post 2022 analysis
        """)

        # Circuit cluster pie chart
        cluster_data = pd.DataFrame({
            'Type': ['Street Circuits', 'High-Speed', 'Technical', 'Balanced'],
            'Count': [4, 5, 5, 4]
        })

        fig = px.pie(cluster_data, values='Count', names='Type',
                    title='Circuit Classification',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recent predictions
    if st.session_state.predictions_history:
        st.markdown("### 📊 Recent Predictions")
        recent_df = pd.DataFrame(st.session_state.predictions_history[-5:])
        st.dataframe(recent_df, use_container_width=True)

    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Step 1: Make a Prediction**

        Navigate to the Predictions tab and enter:
        - Grid position (1-20)
        - Circuit name
        - Team
        - Optional: Driver details
        """)

    with col2:
        st.info("""
        **Step 2: Analyze Circuits**

        Explore circuit characteristics:
        - View overtaking difficulty
        - Check pole win rates
        - Compare circuit types
        """)

    with col3:
        st.info("""
        **Step 3: Explore Visualizations**

        Discover patterns with:
        - Interactive charts
        - Historical trends
        - Feature importance
        """)


def prediction_page():
    """Interactive prediction tool page"""

    st.markdown("## 🔮 Race Position Predictor")

    # Load model if not loaded
    if st.session_state.prediction_pipeline is None and not st.session_state.model_loaded:
        with st.spinner("Loading prediction model..."):
            st.session_state.prediction_pipeline = load_prediction_pipeline()
            st.session_state.model_loaded = True

    use_demo_mode = st.session_state.prediction_pipeline is None

    if use_demo_mode:
        st.warning("⚠️ Model not available. Using demonstration mode with simulated predictions.")

    # Prediction mode selector
    mode = st.radio(
        "Select Prediction Mode",
        ["Single Position", "Full Grid Simulation"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Single Position":
        single_position_prediction(use_demo_mode)
    else:
        full_grid_simulation(use_demo_mode)


def single_position_prediction(demo_mode=False):
    """Single position prediction interface"""

    st.markdown("### Single Position Prediction")
    st.markdown("Predict the finishing position for a single driver")

    # Input columns
    col1, col2, col3 = st.columns(3)

    with col1:
        grid_position = st.slider(
            "Grid Position",
            min_value=1,
            max_value=20,
            value=5,
            help="Starting position on the grid (1 = Pole)"
        )

        # Visual grid position indicator
        grid_visual = create_grid_visual(grid_position)
        st.markdown(grid_visual, unsafe_allow_html=True)

    with col2:
        circuits = [
            'Monaco', 'Monza', 'Spa-Francorchamps', 'Silverstone',
            'Sakhir', 'Marina Bay', 'Baku', 'Sao Paulo', 'Austin',
            'Suzuka', 'Barcelona', 'Budapest', 'Spielberg', 'Montreal',
            'Melbourne', 'Shanghai', 'Mexico City', 'Abu Dhabi'
        ]

        circuit = st.selectbox(
            "Circuit",
            circuits,
            index=0,
            help="Select the race circuit"
        )

        # Show circuit characteristics if available
        circuit_data = load_circuit_data()
        if circuit in circuit_data:
            stats = circuit_data[circuit]
            pole_wr = stats.get('circuit_pole_win_rate', 0) * 100
            overtake = stats.get('overtaking_difficulty_index', 50)
            st.caption(f"Pole Win Rate: {pole_wr:.1f}%")
            st.caption(f"Overtaking: {'Easy' if overtake < 50 else 'Difficult'}")

    with col3:
        teams = [
            'Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine',
            'AlphaTauri', 'Aston Martin', 'Williams', 'Alfa Romeo', 'Haas',
            'Renault', 'Racing Point', 'Toro Rosso', 'Sauber', 'Force India'
        ]

        team = st.selectbox(
            "Team",
            teams,
            index=0,
            help="Select the team"
        )

        driver = st.text_input(
            "Driver Name (Optional)",
            placeholder="e.g., Max Verstappen",
            help="Optional: Enter driver name for record keeping"
        )

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=2018, max_value=2024, value=2024)
        with col2:
            race_number = st.slider("Race Number in Season", 1, 24, 10)

    # Prediction button
    if st.button("🏁 Predict Finish Position", type="primary", use_container_width=True):
        make_single_prediction(
            grid_position, circuit, team, driver, year,
            race_number, demo_mode
        )


def make_single_prediction(grid, circuit, team, driver, year, race_num, demo_mode):
    """Make and display single prediction"""

    if demo_mode:
        # Demo calculations
        import random
        random.seed(grid + year + race_num)
        base_finish = grid + random.gauss(0, 2.5)
        predicted_finish = max(1, min(20, base_finish))
        predicted_rounded = round(predicted_finish)
        confidence_lower = max(1, predicted_finish - 2)
        confidence_upper = min(20, predicted_finish + 2)

        # Demo probabilities
        win_prob = max(0, min(1, 2.0 / max(1, predicted_finish))) if predicted_finish < 3 else 0.05 / max(1, predicted_finish)
        podium_prob = max(0, min(1, 4.0 / max(1, predicted_finish))) if predicted_finish < 6 else 0.2 / max(1, predicted_finish)
        points_prob = max(0, min(1, 11.0 / max(1, predicted_finish))) if predicted_finish < 15 else 0.5 / max(1, predicted_finish)

    else:
        # Real prediction
        with st.spinner("Calculating prediction..."):
            result = st.session_state.prediction_pipeline.predict(
                grid_position=grid,
                circuit_name=circuit,
                team=team,
                driver=driver if driver else f"Driver {grid}",
                year=year,
                race_number=race_num
            )

        predicted_finish = result['predicted_finish']
        predicted_rounded = result['predicted_finish_rounded']
        confidence_lower = result['confidence_interval']['lower']
        confidence_upper = result['confidence_interval']['upper']
        win_prob = result['probabilities']['win']
        podium_prob = result['probabilities']['podium']
        points_prob = result['probabilities']['points']

    # Display results
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # Main prediction display
    col1, col2, col3 = st.columns([2, 3, 2])

    with col2:
        position_change = grid - predicted_rounded
        change_emoji = "📈" if position_change > 0 else "📉" if position_change < 0 else "➡️"

        st.markdown(f"""
        <div class='prediction-box' style='text-align: center;'>
            <h1 style='margin: 0;'>P{predicted_rounded}</h1>
            <p style='font-size: 20px; margin: 5px;'>Predicted Finish</p>
            <p style='font-size: 16px;'>
                Starting P{grid} → Finishing P{predicted_rounded}
            </p>
            <h3>{change_emoji} {position_change:+d} positions</h3>
        </div>
        """, unsafe_allow_html=True)

    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Exact Prediction",
            f"P{predicted_finish:.2f}",
            f"±{(confidence_upper - confidence_lower)/2:.1f}"
        )

    with col2:
        st.metric(
            "Win Probability",
            f"{win_prob*100:.1f}%"
        )

    with col3:
        st.metric(
            "Podium Probability",
            f"{podium_prob*100:.1f}%"
        )

    with col4:
        st.metric(
            "Points Probability",
            f"{points_prob*100:.1f}%"
        )

    # Confidence interval visualization
    st.markdown("### Confidence Interval")

    fig_confidence = go.Figure()

    # Add confidence band
    fig_confidence.add_trace(go.Scatter(
        x=[confidence_lower, predicted_finish, confidence_upper],
        y=[1, 1, 1],
        mode='markers+lines',
        marker=dict(size=[10, 15, 10], color=['lightblue', 'blue', 'lightblue']),
        line=dict(color='lightblue', width=5),
        showlegend=False
    ))

    # Add grid position
    fig_confidence.add_trace(go.Scatter(
        x=[grid],
        y=[1.1],
        mode='markers+text',
        marker=dict(size=12, color='green', symbol='diamond'),
        text=['Start'],
        textposition='top center',
        name='Grid Position',
        showlegend=True
    ))

    fig_confidence.update_layout(
        xaxis=dict(
            range=[0.5, 20.5],
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Position"
        ),
        yaxis=dict(
            range=[0.8, 1.3],
            showticklabels=False,
            title=""
        ),
        height=200,
        title=f"95% Confidence: P{confidence_lower:.1f} - P{confidence_upper:.1f}"
    )

    st.plotly_chart(fig_confidence, use_container_width=True)

    # Save to history
    st.session_state.predictions_history.append({
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Circuit': circuit,
        'Team': team,
        'Grid': f"P{grid}",
        'Predicted': f"P{predicted_rounded}",
        'Change': f"{position_change:+d}",
        'Win%': f"{win_prob*100:.1f}%"
    })

    st.success("✅ Prediction added to history!")


def create_grid_visual(position):
    """Create visual representation of grid position"""

    grid_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px; max-width: 200px;'>"

    for i in range(1, 21):
        if i == position:
            color = '#2196F3'
            border = '3px solid #1565C0'
            text_color = 'white'
        else:
            color = '#e0e0e0'
            border = '1px solid #bdbdbd'
            text_color = '#666'

        grid_html += f"""
        <div style='
            width: 45px;
            height: 35px;
            background-color: {color};
            border: {border};
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: {text_color};
        '>
            P{i}
        </div>
        """

        if i % 2 == 0:
            grid_html += "<div style='width: 100%; height: 0;'></div>"

    grid_html += "</div>"

    return grid_html


def full_grid_simulation(demo_mode=False):
    """Full grid simulation interface"""

    st.markdown("### Full Grid Simulation")
    st.markdown("Simulate an entire race result from starting grid")

    st.info("📝 Enter the starting grid for all 20 positions")

    # Circuit and race info
    col1, col2 = st.columns(2)

    with col1:
        circuits = ['Monaco', 'Monza', 'Spa-Francorchamps', 'Silverstone', 'Sakhir', 'Marina Bay']
        circuit = st.selectbox("Circuit", circuits)

    with col2:
        year = st.number_input("Year", 2018, 2024, 2024)
        race_number = st.slider("Race Number", 1, 24, 10)

    # Quick grid setup
    teams_list = ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine',
                  'AlphaTauri', 'Aston Martin', 'Williams', 'Alfa Romeo', 'Haas']

    # Use example grid or custom
    use_example = st.checkbox("Use example grid (2024 Monaco)", value=True)

    if use_example:
        grid = [
            {"position": 1, "driver": "Charles Leclerc", "team": "Ferrari"},
            {"position": 2, "driver": "Oscar Piastri", "team": "McLaren"},
            {"position": 3, "driver": "Carlos Sainz", "team": "Ferrari"},
            {"position": 4, "driver": "Lando Norris", "team": "McLaren"},
            {"position": 5, "driver": "Max Verstappen", "team": "Red Bull"},
            {"position": 6, "driver": "George Russell", "team": "Mercedes"},
            {"position": 7, "driver": "Lewis Hamilton", "team": "Mercedes"},
            {"position": 8, "driver": "Yuki Tsunoda", "team": "AlphaTauri"},
            {"position": 9, "driver": "Alex Albon", "team": "Williams"},
            {"position": 10, "driver": "Pierre Gasly", "team": "Alpine"},
            {"position": 11, "driver": "Esteban Ocon", "team": "Alpine"},
            {"position": 12, "driver": "Daniel Ricciardo", "team": "AlphaTauri"},
            {"position": 13, "driver": "Fernando Alonso", "team": "Aston Martin"},
            {"position": 14, "driver": "Lance Stroll", "team": "Aston Martin"},
            {"position": 15, "driver": "Valtteri Bottas", "team": "Alfa Romeo"},
            {"position": 16, "driver": "Zhou Guanyu", "team": "Alfa Romeo"},
            {"position": 17, "driver": "Kevin Magnussen", "team": "Haas"},
            {"position": 18, "driver": "Nico Hulkenberg", "team": "Haas"},
            {"position": 19, "driver": "Logan Sargeant", "team": "Williams"},
            {"position": 20, "driver": "Sergio Perez", "team": "Red Bull"}
        ]

        # Display grid
        grid_df = pd.DataFrame(grid)
        st.dataframe(grid_df, use_container_width=True, hide_index=True)

    if st.button("🏁 Simulate Race", type="primary", use_container_width=True):
        if use_example or 'grid' in locals():
            with st.spinner("Simulating race..."):
                simulate_full_race(grid, circuit, year, race_number, demo_mode)
        else:
            st.error("Please set up the grid first")


def simulate_full_race(grid, circuit, year, race_num, demo_mode):
    """Simulate full race and display results"""

    predictions = []

    for entry in grid:
        if demo_mode:
            import random
            random.seed(entry['position'] + year + race_num)
            pred_finish = entry['position'] + random.gauss(0, 2.5)
            pred_finish = max(1, min(20, pred_finish))
        else:
            result = st.session_state.prediction_pipeline.predict(
                grid_position=entry['position'],
                circuit_name=circuit,
                team=entry['team'],
                driver=entry['driver'],
                year=year,
                race_number=race_num
            )
            pred_finish = result['predicted_finish']

        predictions.append({
            'Driver': entry['driver'],
            'Team': entry['team'],
            'Grid': entry['position'],
            'Predicted Finish': pred_finish,
            'Change': entry['position'] - pred_finish
        })

    # Sort by predicted finish
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('Predicted Finish')
    pred_df['Final Position'] = range(1, len(pred_df) + 1)

    # Display results
    st.markdown("---")
    st.markdown("### 🏁 Race Simulation Results")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_change = abs(pred_df['Change']).mean()
        st.metric("Avg Position Changes", f"{avg_change:.1f}")

    with col2:
        top3_changes = pred_df.head(3)['Change'].abs().sum()
        st.metric("Podium Shake-up", f"{top3_changes:.0f} positions")

    with col3:
        biggest_gainer = pred_df['Change'].max()
        st.metric("Biggest Gain", f"+{biggest_gainer:.0f}" if biggest_gainer > 0 else "0")

    # Results table with styling
    st.markdown("### Final Classification")

    # Color code the changes
    def highlight_change(val):
        if val > 2:
            return 'background-color: #c8e6c9'
        elif val < -2:
            return 'background-color: #ffcdd2'
        return ''

    display_df = pred_df[['Final Position', 'Driver', 'Team', 'Grid', 'Predicted Finish', 'Change']].copy()
    display_df['Change'] = display_df['Change'].apply(lambda x: f"+{x:.0f}" if x > 0 else f"{x:.0f}")

    st.dataframe(
        display_df.style.applymap(lambda x: highlight_change(float(x.replace('+', '')) if isinstance(x, str) and ('+' in x or '-' in x) else 0),
                                  subset=['Change']),
        use_container_width=True,
        hide_index=True
    )

    # Visualization
    fig = go.Figure()

    for idx, row in pred_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Grid'], row['Final Position']],
            y=['Grid', 'Finish'],
            mode='lines+markers',
            name=row['Driver'],
            line=dict(width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Position Changes (Grid → Finish)",
        xaxis_title="Position",
        yaxis_title="Stage",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def circuit_analysis_page():
    """Circuit analysis and comparison page"""

    st.markdown("## 🏎️ Circuit Analysis")

    circuit_data = load_circuit_data()

    if not circuit_data:
        st.warning("Circuit data not available")
        st.info("Circuit statistics are loaded from the model preprocessing files")
        return

    # Circuit rankings
    st.markdown("### Circuit Characteristics")

    rankings = []
    for circuit, stats in circuit_data.items():
        rankings.append({
            'Circuit': circuit,
            'Pole Win Rate': stats.get('circuit_pole_win_rate', 0) * 100,
            'Overtaking Difficulty': stats.get('overtaking_difficulty_index', 50),
            'DNF Rate': stats.get('circuit_dnf_rate', 0),
            'Avg Position Change': stats.get('circuit_avg_pos_change', 0)
        })

    rankings_df = pd.DataFrame(rankings)

    # Visualization selector
    metric = st.selectbox(
        "Select Metric to Visualize",
        ['Pole Win Rate', 'Overtaking Difficulty', 'DNF Rate', 'Avg Position Change']
    )

    # Create bar chart
    fig = px.bar(
        rankings_df.sort_values(metric, ascending=False).head(15),
        x=metric,
        y='Circuit',
        orientation='h',
        title=f'Top 15 Circuits by {metric}',
        color=metric,
        color_continuous_scale='RdYlGn_r' if metric in ['Overtaking Difficulty', 'DNF Rate'] else 'RdYlGn'
    )

    fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    with st.expander("View Full Circuit Data"):
        st.dataframe(rankings_df.sort_values(metric, ascending=False), use_container_width=True, hide_index=True)


def visualization_page():
    """Visualizations and analysis page"""

    st.markdown("## 📈 Data Visualizations")

    df = load_historical_data()

    if df.empty:
        st.warning("Historical data not available for visualization")
        return

    # Grid vs Finish scatter
    st.markdown("### Grid Position vs Finish Position")

    if 'GridPosition' in df.columns and 'Position_raw' in df.columns:
        sample_df = df.sample(min(500, len(df)))

        fig = px.scatter(
            sample_df,
            x='GridPosition',
            y='Position_raw',
            title='Historical Grid vs Finish Positions',
            labels={'GridPosition': 'Grid Position', 'Position_raw': 'Finish Position'},
            opacity=0.6
        )

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[1, 20],
            y=[1, 20],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='No Change Line'
        ))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.markdown("### Feature Distributions")

    col1, col2 = st.columns(2)

    with col1:
        if 'position_change' in df.columns:
            fig = px.histogram(
                df,
                x='position_change',
                nbins=40,
                title='Distribution of Position Changes',
                labels={'position_change': 'Position Change'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'team_momentum' in df.columns:
            fig = px.box(
                df,
                x='TeamName' if 'TeamName' in df.columns else None,
                y='team_momentum',
                title='Team Momentum Distribution'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def about_page():
    """About page with project information"""

    st.markdown("## ℹ️ About VANTAGE F1")

    st.markdown("""
    ### Project Overview

    VANTAGE F1 is a comprehensive machine learning project for predicting Formula 1 race outcomes
    based on starting grid positions and other race factors.

    **V**aluating **A**dvantage **N**umerically **T**hrough **A**nalysis of **G**rid **E**ffects

    ### Model Details

    - **Algorithm**: Random Forest Regressor
    - **Features**: 136 engineered features
    - **Training Data**: 357 races from 2018-2024
    - **Test Performance**: MAE 0.57 positions, R² 0.971
    - **Inference Time**: <50ms per prediction

    ### Feature Categories

    1. **Grid Position Features**: Transformations and indicators
    2. **Temporal Features**: Season progress and race timing
    3. **Circuit Features**: Track characteristics and statistics
    4. **Team Features**: Recent form and historical performance
    5. **Driver Features**: Experience and track-specific stats
    6. **Interaction Features**: Complex relationships
    7. **Categorical Encodings**: Frequency, target, and label encoding

    ### Technology Stack

    - **ML Framework**: scikit-learn
    - **Data Processing**: pandas, numpy
    - **Visualization**: plotly, streamlit
    - **Deployment**: Streamlit Cloud

    ### Links

    - [GitHub Repository](https://github.com/AsteriodBlues/Vantage)
    - [Model Documentation](../docs/model_performance.md)
    - [API Specification](../docs/api_specification.md)

    ### Acknowledgments

    Data sourced from historical F1 race results and statistics.
    """)


if __name__ == "__main__":
    main()
