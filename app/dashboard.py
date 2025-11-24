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

# Modern CSS styling with dark mode support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(180deg, #fafbfc 0%, #f4f6f8 100%);
    }

    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #fafbfc 0%, #f4f6f8 100%);
    }

    /* Modern glass morphism tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(250, 251, 252, 0.8);
        backdrop-filter: blur(10px);
        padding: 8px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.03);
    }
    .stTabs [data-baseweb="tab-list"] button {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(25, 118, 210, 0.1);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Modern metric cards with soft neumorphism */
    .metric-card {
        background: #f8f9fa;
        padding: 28px;
        border-radius: 20px;
        box-shadow:
            8px 8px 16px rgba(174, 174, 192, 0.2),
            -8px -8px 16px rgba(255, 255, 255, 0.8);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0, 0, 0, 0.03);
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow:
            10px 10px 20px rgba(174, 174, 192, 0.25),
            -10px -10px 20px rgba(255, 255, 255, 0.9),
            0 0 30px rgba(102, 126, 234, 0.15);
    }

    /* Glassmorphism prediction box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.06) 0%, rgba(118, 75, 162, 0.06) 100%);
        backdrop-filter: blur(16px);
        padding: 28px;
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin: 20px 0;
        box-shadow:
            0 4px 16px rgba(102, 126, 234, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    /* Modern alert boxes */
    .warning-box {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF3E5 100%);
        padding: 24px;
        border-radius: 16px;
        border-left: 4px solid #FF9800;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.08);
    }

    .success-box {
        background: linear-gradient(135deg, #F1F8F4 0%, #E8F5E9 100%);
        padding: 24px;
        border-radius: 16px;
        border-left: 4px solid #4CAF50;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.08);
    }

    /* Modern typography */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        letter-spacing: -0.02em;
        text-shadow: none;
    }
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 2rem;
        border-bottom: none;
        padding-bottom: 8px;
        position: relative;
        letter-spacing: -0.01em;
    }
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.5rem;
    }

    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 15px;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5);
        transform: translateY(-3px);
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Modern download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        box-shadow: 0 4px 16px rgba(72, 187, 120, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        box-shadow: 0 8px 24px rgba(72, 187, 120, 0.5);
        transform: translateY(-3px);
    }

    /* Modern metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #718096;
        font-size: 0.9rem;
    }

    /* Modern dataframes */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* Modern expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        font-weight: 600;
        padding: 16px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s;
    }
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    /* Modern inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Modern slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #f4f6f8 100%);
        border-right: 1px solid #e8eaed;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 32px 20px;
        color: #718096;
        background: linear-gradient(135deg, #f8f9fa 0%, #f4f6f8 100%);
        border-top: 1px solid #e8eaed;
        margin-top: 60px;
        border-radius: 20px 20px 0 0;
        font-size: 14px;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .footer a:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    /* Smooth animations */
    * {
        transition: background-color 0.3s ease;
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home",
        "🔮 Predictions",
        "🏎️ Circuit Analysis",
        "📈 Visualizations",
        "🧠 Model Insights",
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
        model_insights_page()

    with tab6:
        about_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p><strong>VANTAGE F1</strong> - Valuating Advantage Numerically Through Analysis of Grid Effects</p>
        <p>Powered by Random Forest ML | Data: 2018-2024 F1 Seasons | Accuracy: 0.57 MAE</p>
        <p><a href='https://github.com/AsteriodBlues/Vantage' target='_blank'>GitHub</a> | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


def home_page():
    """Create the home/landing page"""

    # Modern hero section
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px 40px 20px;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                border-radius: 24px; margin-bottom: 40px; border: 1px solid rgba(102, 126, 234, 0.1);'>
        <div style='font-size: 4rem; margin-bottom: 20px;'>🏁</div>
        <h1 style='font-size: 3.5rem; margin-bottom: 16px; letter-spacing: -0.03em;'>VANTAGE F1</h1>
        <p style='font-size: 1.3rem; color: #4a5568; font-weight: 600; margin-bottom: 12px;'>
            Valuating Advantage Numerically Through Analysis of Grid Effects
        </p>
        <p style='font-size: 1.1rem; color: #718096; max-width: 700px; margin: 0 auto; line-height: 1.6; font-weight: 500;'>
            Advanced machine learning for Formula 1 race prediction based on starting grid positions
            and comprehensive race analysis.
        </p>
        <div style='margin-top: 32px; display: inline-flex; gap: 12px; flex-wrap: wrap; justify-content: center;'>
            <span style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 8px 20px; border-radius: 20px;
                         font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
                ⚡ 0.57 MAE
            </span>
            <span style='background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                         color: white; padding: 8px 20px; border-radius: 20px;
                         font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(72, 187, 120, 0.3);'>
                📊 0.971 R²
            </span>
            <span style='background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
                         color: white; padding: 8px 20px; border-radius: 20px;
                         font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(237, 137, 54, 0.3);'>
                🚀 136 Features
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    """Make and display single prediction with error handling"""

    try:
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
            # Real prediction with error handling
            with st.spinner("Calculating prediction..."):
                try:
                    result = st.session_state.prediction_pipeline.predict(
                        grid_position=grid,
                        circuit_name=circuit,
                        team=team,
                        driver=driver if driver else f"Driver {grid}",
                        year=year,
                        race_number=race_num
                    )
                except ValueError as e:
                    st.error(f"Invalid input: {str(e)}")
                    return
                except KeyError as e:
                    st.error(f"Missing data for: {str(e)}")
                    return
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    return

            predicted_finish = result['predicted_finish']
            predicted_rounded = result['predicted_finish_rounded']
            confidence_lower = result['confidence_interval']['lower']
            confidence_upper = result['confidence_interval']['upper']
            win_prob = result['probabilities']['win']
            podium_prob = result['probabilities']['podium']
            points_prob = result['probabilities']['points']

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return

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
    prediction_record = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Circuit': circuit,
        'Team': team,
        'Driver': driver if driver else f"Driver {grid}",
        'Grid': f"P{grid}",
        'Predicted': f"P{predicted_rounded}",
        'Change': f"{position_change:+d}",
        'Win%': f"{win_prob*100:.1f}%",
        'Podium%': f"{podium_prob*100:.1f}%",
        'Points%': f"{points_prob*100:.1f}%"
    }
    st.session_state.predictions_history.append(prediction_record)

    # Export options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.success("✅ Prediction added to history!")

    with col2:
        # Export as JSON
        export_data = {
            'prediction': {
                'grid_position': grid,
                'circuit': circuit,
                'team': team,
                'driver': driver if driver else f"Driver {grid}",
                'year': year,
                'race_number': race_num
            },
            'results': {
                'predicted_finish': float(predicted_finish),
                'predicted_finish_rounded': predicted_rounded,
                'position_change': position_change,
                'confidence_interval': {
                    'lower': float(confidence_lower),
                    'upper': float(confidence_upper)
                },
                'probabilities': {
                    'win': float(win_prob),
                    'podium': float(podium_prob),
                    'points': float(points_prob)
                }
            }
        }
        st.download_button(
            label="📥 Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"prediction_{circuit}_{grid}.json",
            mime="application/json"
        )

    with col3:
        # Export as CSV
        csv_data = pd.DataFrame([prediction_record]).to_csv(index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"prediction_{circuit}_{grid}.csv",
            mime="text/csv"
        )


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

    # Export options for grid simulation
    st.markdown("### 📥 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # Export as CSV
        csv_export = pred_df[['Final Position', 'Driver', 'Team', 'Grid', 'Predicted Finish', 'Change']].copy()
        csv_data = csv_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name=f"race_simulation_{circuit}_{year}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Export as JSON
        json_export = {
            'race_info': {
                'circuit': circuit,
                'year': year,
                'race_number': race_num
            },
            'results': pred_df.to_dict('records')
        }
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(json_export, indent=2),
            file_name=f"race_simulation_{circuit}_{year}.json",
            mime="application/json",
            use_container_width=True
        )


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
    """Comprehensive visualizations and analysis page"""

    st.markdown("## 📈 Data Visualizations & Insights")

    df = load_historical_data()

    if df.empty:
        st.warning("Historical data not available for visualization")
        return

    # Create tabs for different visualization categories
    viz_tabs = st.tabs([
        "📊 Grid Analysis",
        "🏎️ Circuit Patterns",
        "📈 Performance Trends",
        "🎯 Win Probability"
    ])

    with viz_tabs[0]:
        grid_analysis_viz(df)

    with viz_tabs[1]:
        circuit_patterns_viz(df)

    with viz_tabs[2]:
        performance_trends_viz(df)

    with viz_tabs[3]:
        win_probability_viz(df)


def grid_analysis_viz(df):
    """Grid position advantage analysis visualizations"""

    st.markdown("### 🎯 Grid Position Advantage Analysis")

    # Calculate grid statistics
    if 'GridPosition' in df.columns and 'Position_raw' in df.columns:
        grid_stats = df.groupby('GridPosition').agg({
            'Position_raw': ['mean', 'std', 'min', 'max'],
            'position_change': 'mean'
        }).round(2)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Main advantage plot with confidence bands
            fig = go.Figure()

            grid_positions = sorted(df['GridPosition'].unique())
            avg_finish = [grid_stats.loc[p, ('Position_raw', 'mean')] for p in grid_positions if p in grid_stats.index]
            std_finish = [grid_stats.loc[p, ('Position_raw', 'std')] for p in grid_positions if p in grid_stats.index]

            # Average line
            fig.add_trace(go.Scatter(
                x=grid_positions[:len(avg_finish)],
                y=avg_finish,
                mode='lines+markers',
                name='Average Finish',
                line=dict(color='#2196F3', width=3),
                marker=dict(size=8)
            ))

            # No change line
            fig.add_trace(go.Scatter(
                x=[1, 20],
                y=[1, 20],
                mode='lines',
                line=dict(dash='dash', color='red', width=2),
                name='No Change'
            ))

            fig.update_layout(
                title='Grid Position vs Average Finish',
                xaxis_title='Grid Position',
                yaxis_title='Average Finish Position',
                height=500,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Key Statistics")

            pole_data = df[df['GridPosition'] == 1]
            if len(pole_data) > 0:
                pole_win_rate = (pole_data['Position_raw'] == 1).mean() * 100
                st.metric("Pole Win Rate", f"{pole_win_rate:.1f}%")

            top3_data = df[df['GridPosition'] <= 3]
            if len(top3_data) > 0:
                top3_win = (top3_data['Position_raw'] == 1).mean() * 100
                st.metric("Top 3 Win Rate", f"{top3_win:.1f}%")

            if 'position_change' in df.columns:
                avg_change = df['position_change'].mean()
                st.metric("Avg Position Change", f"{avg_change:+.2f}")

    # Position change distribution
    st.markdown("---")
    st.markdown("### Position Change Patterns")

    col1, col2 = st.columns(2)

    with col1:
        if 'position_change' in df.columns:
            fig = px.histogram(
                df,
                x='position_change',
                nbins=40,
                title='Distribution of Position Changes',
                labels={'position_change': 'Position Change'},
                color_discrete_sequence=['#667eea']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'GridPosition' in df.columns and 'Position_raw' in df.columns:
            # Scatter plot
            sample_df = df.sample(min(500, len(df)))
            fig = px.scatter(
                sample_df,
                x='GridPosition',
                y='Position_raw',
                opacity=0.4,
                title='Grid vs Finish Scatter',
                labels={'GridPosition': 'Grid Position', 'Position_raw': 'Finish Position'}
            )
            fig.add_trace(go.Scatter(x=[1, 20], y=[1, 20], mode='lines',
                                    line=dict(dash='dash', color='red'), name='No Change'))
            st.plotly_chart(fig, use_container_width=True)


def circuit_patterns_viz(df):
    """Circuit-specific pattern visualizations"""

    st.markdown("### 🏎️ Circuit Characteristics")

    circuit_data = load_circuit_data()

    if circuit_data:
        # Prepare circuit comparison data
        circuits_list = []
        for name, data in list(circuit_data.items())[:15]:
            circuits_list.append({
                'Circuit': name,
                'Pole Win Rate': data.get('circuit_pole_win_rate', 0) * 100,
                'Overtaking Difficulty': data.get('overtaking_difficulty_index', 50),
                'DNF Rate': data.get('circuit_dnf_rate', 0),
                'Avg Changes': abs(data.get('circuit_avg_pos_change', 0))
            })

        circuits_df = pd.DataFrame(circuits_list)

        col1, col2 = st.columns(2)

        with col1:
            # Pole win rate comparison
            fig = px.bar(
                circuits_df.sort_values('Pole Win Rate', ascending=False).head(10),
                x='Pole Win Rate',
                y='Circuit',
                orientation='h',
                title='Top 10 Circuits by Pole Win Rate',
                color='Pole Win Rate',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Overtaking difficulty
            fig = px.bar(
                circuits_df.sort_values('Overtaking Difficulty', ascending=False).head(10),
                x='Overtaking Difficulty',
                y='Circuit',
                orientation='h',
                title='Most Difficult Circuits for Overtaking',
                color='Overtaking Difficulty',
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Circuit characteristics radar
        st.markdown("---")
        if len(circuits_df) >= 5:
            selected_circuits = st.multiselect(
                "Select circuits to compare (max 5):",
                circuits_df['Circuit'].tolist(),
                default=circuits_df['Circuit'].head(3).tolist()[:3]
            )

            if selected_circuits:
                fig = go.Figure()

                for circuit in selected_circuits[:5]:
                    circuit_row = circuits_df[circuits_df['Circuit'] == circuit].iloc[0]
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            circuit_row['Pole Win Rate'] / 100,
                            circuit_row['Overtaking Difficulty'] / 100,
                            circuit_row['DNF Rate'] / 20,
                            circuit_row['Avg Changes'] / 5
                        ],
                        theta=['Pole Win %', 'Overtaking', 'DNF Risk', 'Position Changes'],
                        fill='toself',
                        name=circuit
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Circuit Characteristics Comparison"
                )

                st.plotly_chart(fig, use_container_width=True)


def performance_trends_viz(df):
    """Performance trends over time"""

    st.markdown("### 📈 Historical Performance Trends")

    if 'year' not in df.columns:
        st.info("Year data not available for trend analysis")
        return

    # Yearly statistics
    yearly_stats = df.groupby('year').agg({
        'Position_raw': 'mean',
        'position_change': 'std',
        'GridPosition': 'count'
    }).reset_index()
    yearly_stats.columns = ['Year', 'Avg Finish', 'Position Variance', 'Races']

    col1, col2 = st.columns(2)

    with col1:
        # Position variance over time
        fig = px.line(
            yearly_stats,
            x='Year',
            y='Position Variance',
            title='Position Change Variance by Year',
            markers=True
        )
        fig.update_layout(yaxis_title='Standard Deviation')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Race count by year
        fig = px.bar(
            yearly_stats,
            x='Year',
            y='Races',
            title='Races Analyzed by Year',
            color='Races',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Team performance evolution
    if 'TeamName' in df.columns:
        st.markdown("---")
        st.markdown("### Team Performance Over Time")

        top_teams = df['TeamName'].value_counts().head(6).index.tolist()
        team_yearly = df[df['TeamName'].isin(top_teams)].groupby(['year', 'TeamName']).agg({
            'Position_raw': 'mean'
        }).reset_index()

        fig = px.line(
            team_yearly,
            x='year',
            y='Position_raw',
            color='TeamName',
            title='Average Finish Position by Team',
            markers=True
        )
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Average Finish Position',
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig, use_container_width=True)


def win_probability_viz(df):
    """Win probability analysis"""

    st.markdown("### 🎯 Win Probability Analysis")

    if 'GridPosition' not in df.columns or 'Position_raw' not in df.columns:
        st.info("Insufficient data for win probability analysis")
        return

    # Win probability by grid position
    win_prob_data = []
    for grid_pos in range(1, 21):
        grid_data = df[df['GridPosition'] == grid_pos]
        if len(grid_data) > 0:
            win_rate = (grid_data['Position_raw'] == 1).mean() * 100
            podium_rate = (grid_data['Position_raw'] <= 3).mean() * 100
            points_rate = (grid_data['Position_raw'] <= 10).mean() * 100

            win_prob_data.append({
                'Grid Position': grid_pos,
                'Win %': win_rate,
                'Podium %': podium_rate,
                'Points %': points_rate
            })

    prob_df = pd.DataFrame(win_prob_data)

    col1, col2 = st.columns(2)

    with col1:
        # Win probability
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prob_df['Grid Position'],
            y=prob_df['Win %'],
            name='Win',
            marker_color='#FFD700'
        ))
        fig.add_trace(go.Bar(
            x=prob_df['Grid Position'],
            y=prob_df['Podium %'],
            name='Podium',
            marker_color='#C0C0C0'
        ))
        fig.add_trace(go.Bar(
            x=prob_df['Grid Position'],
            y=prob_df['Points %'],
            name='Points',
            marker_color='#CD7F32'
        ))

        fig.update_layout(
            title='Success Probability by Grid Position',
            xaxis_title='Grid Position',
            yaxis_title='Probability (%)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cumulative probability
        fig = go.Figure()
        for col_name, color in [('Win %', '#FFD700'), ('Podium %', '#C0C0C0'), ('Points %', '#CD7F32')]:
            fig.add_trace(go.Scatter(
                x=prob_df['Grid Position'],
                y=prob_df[col_name],
                mode='lines+markers',
                name=col_name.replace(' %', ''),
                line=dict(width=3, color=color),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title='Success Probability Trends',
            xaxis_title='Grid Position',
            yaxis_title='Probability (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Position gain/loss probability
    st.markdown("---")
    st.markdown("### Expected Position Changes")

    if 'position_change' in df.columns:
        change_by_grid = df.groupby('GridPosition')['position_change'].agg(['mean', 'std']).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=change_by_grid['GridPosition'],
            y=change_by_grid['mean'],
            error_y=dict(type='data', array=change_by_grid['std']),
            marker_color=['#4CAF50' if x > 0 else '#F44336' if x < 0 else '#9E9E9E'
                         for x in change_by_grid['mean']]
        ))

        fig.update_layout(
            title='Average Position Change by Grid Position',
            xaxis_title='Grid Position',
            yaxis_title='Avg Position Change',
            showlegend=False
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")

        st.plotly_chart(fig, use_container_width=True)


def model_insights_page():
    """Model interpretation and feature importance analysis"""

    st.markdown("## 🧠 Model Insights & Feature Importance")

    # Load model
    pipeline = load_prediction_pipeline()

    if pipeline is None:
        st.warning("⚠️ Model not available. Feature importance analysis requires the production model.")
        return

    # Create sub-tabs
    insight_tabs = st.tabs([
        "📊 Feature Importance",
        "🔍 Feature Categories",
        "📈 Model Performance",
        "🎯 Prediction Analysis"
    ])

    with insight_tabs[0]:
        feature_importance_viz(pipeline)

    with insight_tabs[1]:
        feature_categories_viz(pipeline)

    with insight_tabs[2]:
        model_performance_viz(pipeline)

    with insight_tabs[3]:
        prediction_analysis_viz(pipeline)


def feature_importance_viz(pipeline):
    """Display feature importance analysis"""

    st.markdown("### Top Feature Importance")
    st.markdown("Features that have the strongest impact on race outcome predictions:")

    # Get feature importances from the model
    if hasattr(pipeline.model, 'feature_importances_'):
        importances = pipeline.model.feature_importances_
        feature_names = pipeline.feature_names

        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Top 20 features
        top_features = importance_df.head(20)

        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 20 Most Important Features',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Most Important Feature",
                top_features.iloc[0]['Feature'],
                f"{top_features.iloc[0]['Importance']:.3f}"
            )

        with col2:
            top_10_sum = top_features.head(10)['Importance'].sum()
            st.metric(
                "Top 10 Features Impact",
                f"{top_10_sum:.1%}",
                "of total importance"
            )

        with col3:
            total_features = len(feature_names)
            st.metric(
                "Total Features",
                total_features,
                "engineered"
            )

        # Show full table in expander
        with st.expander("View All Feature Importances"):
            st.dataframe(
                importance_df,
                use_container_width=True,
                height=400
            )
    else:
        st.info("Feature importance not available for this model type.")


def feature_categories_viz(pipeline):
    """Visualize features grouped by category"""

    st.markdown("### Feature Categories Breakdown")
    st.markdown("Features are organized into strategic categories:")

    # Define feature categories
    categories = {
        'Grid Position': ['GridPosition', 'grid_squared', 'grid_cubed', 'grid_log', 'grid_sqrt',
                         'front_row', 'top_three', 'top_five', 'top_ten', 'back_half',
                         'grid_side', 'grid_side_clean', 'grid_row'],
        'Temporal': ['race_number', 'season_progress', 'races_remaining', 'early_season',
                    'mid_season', 'late_season', 'is_season_opener', 'is_season_finale',
                    'post_2022', 'years_into_regulations'],
        'Circuit': ['pole_win_rate', 'overtaking_difficulty', 'correlation', 'avg_pos_change',
                   'dnf_rate', 'improved_pct', 'track_length', 'num_turns', 'altitude',
                   'longest_straight', 'circuit_type', 'downforce_level', 'is_street'],
        'Team Performance': ['avg_finish_last_5', 'avg_finish_last_3', 'points_last_5',
                           'wins_season', 'podiums_season', 'points_total', 'momentum',
                           'consistency', 'vs_average_grid', 'dnf_rate_last_10',
                           'completion_rate_season'],
        'Driver Stats': ['career_races', 'years_experience', 'is_rookie', 'is_veteran',
                        'races_at_circuit', 'avg_finish_at_circuit', 'is_specialist',
                        'vs_teammate', 'vs_car_potential', 'is_team_leader'],
        'Interactions': ['grid_x_overtaking', 'grid_x_team_delta', 'grid_x_low_df',
                        'momentum_x_variance', 'form_x_contention', 'veteran_new_circuit',
                        'early_x_variance', 'late_contention_pressure']
    }

    # Calculate importance by category
    if hasattr(pipeline.model, 'feature_importances_'):
        importances = pipeline.model.feature_importances_
        feature_names = pipeline.feature_names

        importance_dict = dict(zip(feature_names, importances))

        category_importance = {}
        category_counts = {}

        for category, features in categories.items():
            # Find matching features (partial matches)
            total_importance = 0
            count = 0
            for fname in feature_names:
                for pattern in features:
                    if pattern.lower() in fname.lower():
                        total_importance += importance_dict.get(fname, 0)
                        count += 1
                        break

            category_importance[category] = total_importance
            category_counts[category] = count

        # Create visualization
        cat_df = pd.DataFrame({
            'Category': list(category_importance.keys()),
            'Total Importance': list(category_importance.values()),
            'Feature Count': list(category_counts.values())
        }).sort_values('Total Importance', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                cat_df,
                x='Category',
                y='Total Importance',
                title='Importance by Feature Category',
                color='Total Importance',
                color_continuous_scale='Viridis'
            )
            fig1.update_xaxes(tickangle=45)
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(
                cat_df,
                values='Feature Count',
                names='Category',
                title='Feature Distribution by Category'
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Category details
        st.markdown("### Category Details")
        for _, row in cat_df.iterrows():
            with st.expander(f"{row['Category']} ({row['Feature Count']} features)"):
                cat_features = categories[row['Category']]
                st.markdown("**Example features:**")
                for feat in cat_features[:10]:
                    st.markdown(f"- `{feat}`")


def model_performance_viz(pipeline):
    """Display model performance metrics"""

    st.markdown("### Model Performance Metrics")

    # Display metadata
    metadata = pipeline.metadata

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Performance")
        st.metric("Training MAE", f"{metadata.get('train_mae', 0):.3f} positions")
        st.metric("Training R²", f"{metadata.get('train_r2', 0):.3f}")
        st.metric("Training Samples", metadata.get('train_samples', 'N/A'))

    with col2:
        st.markdown("#### Test Performance")
        st.metric("Test MAE", f"{metadata.get('test_mae', 0):.3f} positions")
        st.metric("Test R²", f"{metadata.get('test_r2', 0):.3f}")
        st.metric("Test Samples", metadata.get('test_samples', 'N/A'))

    # Performance breakdown
    st.markdown("---")
    st.markdown("### Performance Interpretation")

    test_mae = metadata.get('test_mae', 0)
    test_r2 = metadata.get('test_r2', 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mean Absolute Error (MAE)")
        st.markdown(f"""
        The model predicts race finish positions with an average error of **{test_mae:.2f} positions**.

        This means:
        - On average, predictions are within ~{test_mae:.0f} position of the actual result
        - Approximately {(1-test_mae/10)*100:.0f}% accuracy for top-10 predictions
        - High precision for podium and win probability estimates
        """)

    with col2:
        st.markdown("#### R² Score (Coefficient of Determination)")
        st.markdown(f"""
        The model explains **{test_r2*100:.1f}%** of the variance in race outcomes.

        This indicates:
        - Very strong predictive power
        - Captures the relationship between grid position and finish position
        - Accounts for circuit, team, and driver factors effectively
        """)

    # Model specifications
    st.markdown("---")
    st.markdown("### Model Specifications")

    specs_col1, specs_col2 = st.columns(2)

    with specs_col1:
        st.markdown(f"""
        **Algorithm:** {metadata.get('model_type', 'N/A')}
        **Features:** {len(pipeline.feature_names)}
        **Training Period:** {metadata.get('training_date', 'N/A')}
        """)

    with specs_col2:
        st.markdown(f"""
        **Validation MAE:** {metadata.get('val_mae', 0):.3f} positions
        **Validation R²:** {metadata.get('val_r2', 0):.3f}
        **Validation Samples:** {metadata.get('val_samples', 'N/A')}
        """)


def prediction_analysis_viz(pipeline):
    """Analyze prediction patterns and distributions"""

    st.markdown("### Prediction Analysis")
    st.markdown("Understanding how the model makes predictions:")

    # Load training data for analysis
    try:
        train_data = pd.read_csv('data/processed/train.csv')

        # Grid position vs actual finish
        st.markdown("#### Grid Position vs Race Finish Distribution")

        grid_finish_data = train_data.groupby('GridPosition').agg({
            'Position_raw': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)

        grid_finish_data.columns = ['Avg Finish', 'Std Dev', 'Best', 'Worst', 'Count']
        grid_finish_data = grid_finish_data.reset_index()

        fig = go.Figure()

        # Add average line
        fig.add_trace(go.Scatter(
            x=grid_finish_data['GridPosition'],
            y=grid_finish_data['Avg Finish'],
            mode='lines+markers',
            name='Average Finish',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=8)
        ))

        # Add confidence band
        fig.add_trace(go.Scatter(
            x=grid_finish_data['GridPosition'],
            y=grid_finish_data['Avg Finish'] + grid_finish_data['Std Dev'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=grid_finish_data['GridPosition'],
            y=grid_finish_data['Avg Finish'] - grid_finish_data['Std Dev'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(33, 150, 243, 0.2)',
            fill='tonexty',
            showlegend=True
        ))

        # Add diagonal reference
        fig.add_trace(go.Scatter(
            x=[1, 20],
            y=[1, 20],
            mode='lines',
            name='No Change',
            line=dict(dash='dash', color='red', width=2)
        ))

        fig.update_layout(
            title='Grid Position vs Finish Position (Historical Data)',
            xaxis_title='Grid Position',
            yaxis_title='Finish Position',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Position change distribution
        st.markdown("#### Position Change Distribution")

        train_data['position_change'] = train_data['GridPosition'] - train_data['Position_raw']

        fig2 = px.histogram(
            train_data,
            x='position_change',
            nbins=40,
            title='Distribution of Position Changes',
            labels={'position_change': 'Position Change (Grid - Finish)'},
            color_discrete_sequence=['#1976D2']
        )

        fig2.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="No Change"
        )

        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # Key statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_change = train_data['position_change'].mean()
            st.metric("Avg Position Change", f"{avg_change:+.2f}")

        with col2:
            improve_pct = (train_data['position_change'] > 0).mean() * 100
            st.metric("% Improved Position", f"{improve_pct:.1f}%")

        with col3:
            decline_pct = (train_data['position_change'] < 0).mean() * 100
            st.metric("% Lost Position", f"{decline_pct:.1f}%")

        with col4:
            same_pct = (train_data['position_change'] == 0).mean() * 100
            st.metric("% No Change", f"{same_pct:.1f}%")

    except Exception as e:
        st.error(f"Could not load training data: {e}")


def about_page():
    """About page with project information and methodology"""

    st.markdown("## ℹ️ About VANTAGE F1")

    # Create sub-tabs for organized information
    about_tabs = st.tabs([
        "📖 Overview",
        "🔬 Methodology",
        "💾 Data Sources",
        "🛠️ Technical Details"
    ])

    with about_tabs[0]:
        about_overview()

    with about_tabs[1]:
        about_methodology()

    with about_tabs[2]:
        about_data_sources()

    with about_tabs[3]:
        about_technical_details()


def about_overview():
    """Project overview section"""

    st.markdown("### Project Overview")

    st.markdown("""
    **VANTAGE F1** is a comprehensive machine learning system for predicting Formula 1 race outcomes
    based on grid positions, circuit characteristics, team performance, and driver statistics.

    #### What does VANTAGE stand for?

    **V**aluating **A**dvantage **N**umerically **T**hrough **A**nalysis of **G**rid **E**ffects
    """)

    st.markdown("---")

    # Key capabilities
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Prediction Capabilities")
        st.markdown("""
        - **Finish Position Prediction**: Forecast final race position from any grid slot
        - **Win Probability Analysis**: Calculate likelihood of race victory
        - **Podium Probability**: Estimate chances of top-3 finish
        - **Full Grid Simulation**: Predict complete race outcomes
        - **Confidence Intervals**: Quantify prediction uncertainty
        """)

    with col2:
        st.markdown("#### 🏎️ Circuit Intelligence")
        st.markdown("""
        - **18 Historic Circuits**: Analysis of classic F1 tracks
        - **Overtaking Difficulty**: Track-specific passing metrics
        - **Pole Win Rate**: Circuit-specific pole advantage
        - **Circuit Clustering**: Automated track categorization
        - **Track Characteristics**: Physical and statistical attributes
        """)

    st.markdown("---")

    # Model performance summary
    st.markdown("### 📊 Model Performance")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("Test MAE", "0.57 positions", "High Accuracy")

    with perf_col2:
        st.metric("Test R²", "0.971", "Strong Fit")

    with perf_col3:
        st.metric("Features", "136", "Engineered")

    with perf_col4:
        st.metric("Inference", "<50ms", "Per Prediction")

    st.markdown("""
    The model achieves **0.57 position mean absolute error** on unseen test data, meaning predictions
    are typically within ±1 position of actual race results.
    """)


def about_methodology():
    """Detailed methodology section"""

    st.markdown("### 🔬 Methodology")

    st.markdown("#### Machine Learning Approach")

    st.markdown("""
    VANTAGE F1 uses a **Random Forest Regressor** for race outcome prediction. This ensemble method
    combines multiple decision trees to capture complex, non-linear relationships in the data.

    **Why Random Forest?**
    - Handles non-linear relationships between features
    - Robust to outliers and missing data
    - Provides feature importance rankings
    - No assumptions about data distribution
    - Excellent performance on tabular data
    """)

    st.markdown("---")

    st.markdown("#### Feature Engineering Pipeline")

    st.markdown("""
    The model uses **136 engineered features** organized into 7 categories:

    **1. Grid Position Features (13 features)**
    - Raw grid position (1-20)
    - Polynomial transformations (squared, cubed)
    - Logarithmic and square root transformations
    - Position indicators (front row, top 3, top 5, top 10)
    - Grid metadata (side, row number)

    **2. Temporal Features (10 features)**
    - Race number in season
    - Season progress percentage
    - Remaining races in season
    - Season phase indicators (early/mid/late)
    - Special race flags (opener, finale)
    - Regulation era indicators

    **3. Circuit Features (13 features)**
    - Pole win rate (historical)
    - Overtaking difficulty index
    - Grid-finish correlation coefficient
    - Average position change
    - DNF rate and reliability metrics
    - Physical characteristics (length, turns, altitude)
    - Circuit type (street, permanent, hybrid)

    **4. Team Performance Features (11 features)**
    - Rolling performance windows (last 3/5 races)
    - Season totals (wins, podiums, points)
    - Recent form momentum
    - Consistency metrics
    - Reliability statistics
    - Performance vs expectations

    **5. Driver Features (10 features)**
    - Career experience (races, years)
    - Experience level (rookie, veteran)
    - Circuit-specific history
    - Track specialization indicators
    - Recent form metrics
    - Relative performance (vs teammate, vs car potential)

    **6. Interaction Features (8 features)**
    - Grid position × circuit characteristics
    - Form × track difficulty
    - Experience × circuit novelty
    - Strategic phase interactions

    **7. Categorical Encodings (71 features)**
    - Frequency encoding (circuit, team, driver)
    - Target encoding (historical averages)
    - Label encoding (for tree-based models)
    """)

    st.markdown("---")

    st.markdown("#### Model Training Process")

    st.markdown("""
    **Data Preparation**
    1. Historical race data from 2018-2024 seasons
    2. Feature engineering and encoding
    3. Time-based train/validation/test split
    4. Missing value imputation with intelligent defaults

    **Model Configuration**
    - Algorithm: RandomForestRegressor (scikit-learn)
    - Trees: 100 estimators
    - Max depth: 15 levels
    - Min samples split: 5
    - Min samples leaf: 2
    - Random state: 42 (reproducibility)

    **Validation Strategy**
    - Time-based splitting (respects temporal ordering)
    - Training set: 357 races (2018-2022)
    - Validation set: 63 races (2023)
    - Test set: 360 races (2024)

    **Performance Metrics**
    - Mean Absolute Error (MAE): Average position error
    - R² Score: Proportion of variance explained
    - Feature importance rankings
    - Prediction confidence intervals
    """)


def about_data_sources():
    """Data sources and quality section"""

    st.markdown("### 💾 Data Sources")

    st.markdown("""
    #### Historical Race Data

    The model is trained on comprehensive Formula 1 race data spanning **2018-2024 seasons**.

    **Data Coverage:**
    - **780 total races** across 6+ seasons
    - **18 unique circuits** from the modern F1 calendar
    - **10+ teams** including regulation era transitions
    - **40+ drivers** with varying experience levels

    **Data Points per Race:**
    - Grid positions (qualifying results)
    - Final race positions
    - Team assignments
    - Driver identifiers
    - Circuit information
    - Race metadata (date, season, race number)
    """)

    st.markdown("---")

    st.markdown("#### Data Quality & Preprocessing")

    st.markdown("""
    **Quality Assurance:**
    - Data validation for consistency
    - Outlier detection and handling
    - Missing value imputation strategies
    - Encoding validation and testing

    **Preprocessing Steps:**
    1. **Data cleaning**: Remove incomplete or invalid records
    2. **Feature computation**: Calculate all 136 engineered features
    3. **Encoding**: Apply frequency, target, and label encoding
    4. **Normalization**: Scale features appropriately for model training
    5. **Validation**: Ensure feature consistency across train/val/test sets

    **Data Integrity:**
    - Time-based splitting prevents data leakage
    - Validation set from distinct season (2023)
    - Test set from most recent season (2024)
    - No overlap between splits
    """)

    st.markdown("---")

    st.markdown("#### Circuit Database")

    st.markdown("""
    **18 Analyzed Circuits:**

    Monaco, Monza, Spa-Francorchamps, Silverstone, Suzuka, Marina Bay,
    Circuit of the Americas, Interlagos, Yas Marina, Barcelona, Hungaroring,
    Red Bull Ring, Baku, Montreal, Melbourne, Jeddah, Miami, Las Vegas

    **Circuit Statistics Tracked:**
    - Pole win rate (2018-2024)
    - Overtaking difficulty index
    - Grid-finish position correlation
    - Average position changes
    - DNF rates and reliability
    - Track physical characteristics
    - Circuit type classification
    """)


def about_technical_details():
    """Technical implementation details"""

    st.markdown("### 🛠️ Technical Details")

    st.markdown("#### Technology Stack")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        **Machine Learning:**
        - scikit-learn 1.5.0+
        - pandas 2.2.0+
        - numpy 1.26.0+

        **Data Processing:**
        - Feature engineering pipelines
        - Categorical encoding utilities
        - Statistical aggregations
        """)

    with tech_col2:
        st.markdown("""
        **Visualization & UI:**
        - Streamlit 1.37.0+
        - Plotly 5.20.0+
        - Interactive charts and graphs

        **Deployment:**
        - Streamlit Cloud
        - GitHub version control
        - Automated model versioning
        """)

    st.markdown("---")

    st.markdown("#### Model Deployment")

    st.markdown("""
    **Production Pipeline:**
    ```
    models/
    ├── production/
    │   └── simple_predictor_latest/
    │       ├── model.pkl (0.14 MB)
    │       └── metadata.json
    └── preprocessing/
        ├── feature_names.pkl
        ├── circuit_statistics.pkl
        ├── team_baselines.pkl
        ├── driver_statistics.pkl
        └── feature_defaults.pkl
    ```

    **Inference Process:**
    1. Load production model and preprocessing artifacts
    2. Accept raw input (grid position, circuit, team, driver)
    3. Engineer all 136 features using preprocessing pipeline
    4. Generate prediction with confidence interval
    5. Return formatted results with probabilities

    **Performance:**
    - Model size: 0.14 MB (compact)
    - Inference time: <50ms per prediction
    - Batch prediction support
    - Caching for repeated requests
    """)

    st.markdown("---")

    st.markdown("#### API & Interfaces")

    st.markdown("""
    **1. Interactive Dashboard** (this application)
    - Web-based Streamlit interface
    - Single position predictions
    - Full grid simulations
    - Circuit analysis and visualization
    - Model insights and feature importance

    **2. Command Line Interface**
    ```bash
    # Single prediction
    python src/predict_cli.py single \\
        --driver "Max Verstappen" \\
        --team "Red Bull" \\
        --circuit "Monaco" \\
        --grid 1

    # Full grid simulation
    python src/predict_cli.py grid \\
        --grid-file examples/example_grid.json
    ```

    **3. Python API**
    ```python
    from src.prediction_pipeline import F1PredictionPipeline

    pipeline = F1PredictionPipeline()
    result = pipeline.predict(
        grid_position=1,
        circuit_name="Monaco",
        team="Ferrari",
        driver="Charles Leclerc",
        year=2024,
        race_number=7
    )
    ```
    """)

    st.markdown("---")

    st.markdown("#### Project Links")

    st.markdown("""
    - **GitHub Repository**: [github.com/AsteriodBlues/Vantage](https://github.com/AsteriodBlues/Vantage)
    - **Model Performance Documentation**: See repository docs folder
    - **API Specification**: See repository docs folder

    #### Acknowledgments

    Data sourced from historical Formula 1 race results and statistics (2018-2024).
    Project built for educational and analytical purposes.
    """)


if __name__ == "__main__":
    main()
