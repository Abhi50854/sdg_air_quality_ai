"""
Streamlit Web Application for Air Quality Prediction
Premium UI with dark mode, interactive charts, and real-time predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.lstm_predictor import LSTMAQIPredictor
from data.preprocessing import AirQualityPreprocessor
from data.data_loader import AirQualityDataLoader
from api.health_advisor import HealthAdvisor

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor | SDG Climate Action",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark mode design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(123, 44, 191, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: rgba(26, 26, 46, 0.9);
        backdrop-filter: blur(10px);
    }
    
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stSelectbox, .stNumberInput {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'current_aqi' not in st.session_state:
    st.session_state.current_aqi = None

# Initialize components
@st.cache_resource
def load_model_components():
    """Load model and preprocessing components."""
    model = LSTMAQIPredictor(input_shape=(72, 20), forecast_horizon=24)
    preprocessor = AirQualityPreprocessor()
    loader = AirQualityDataLoader()
    advisor = HealthAdvisor()
    
    try:
        model.load_model("models/saved/final_model.keras")
        preprocessor.load_scalers()
    except Exception as e:
        st.warning(f"Model not yet trained. Please run training first.")
    
    return model, preprocessor, loader, advisor

model, preprocessor, loader, advisor = load_model_components()

# Header
st.markdown("<h1 style='text-align: center; font-size: 3.5em; margin-bottom: 10px;'>🌍 Air Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #888; margin-bottom: 30px;'>AI-Powered Climate Action for SDG 3, 11 & 13</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    
    st.markdown("### 📍 Location")
    city = st.text_input("City", value="Los Angeles", help="Enter city name")
    country = st.text_input("Country Code", value="US", help="ISO 3166-1 alpha-2 code")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=34.05, format="%.2f")
    with col2:
        longitude = st.number_input("Longitude", value=-118.24, format="%.2f")
    
    st.markdown("### 👤 Health Profile")
    age = st.slider("Age", 0, 100, 30, help="Your age")
    
    has_respiratory = st.checkbox("Respiratory Condition", help="Asthma, COPD, etc.")
    has_heart = st.checkbox("Heart Condition", help="Cardiovascular disease")
    
    activity_level = st.selectbox(
        "Outdoor Activity Level",
        ["low", "moderate", "high"],
        index=1,
        help="How much time you spend outdoors"
    )
    
    st.markdown("---")
    predict_button = st.button("🚀 Predict Air Quality", use_container_width=True)

# Main content
if predict_button:
    with st.spinner("🔮 Analyzing air quality patterns..."):
        try:
            # Load data
            aqi_data = loader.get_air_quality_data(city=city, country=country, days=7)
            weather_data = loader.get_weather_data(latitude=latitude, longitude=longitude, days=7)
            
            # Merge and preprocess
            merged = loader.merge_datasets(aqi_data, weather_data)
            processed = preprocessor.engineer_features(merged)
            
            # Prepare input sequence
            X_input = processed[preprocessor.feature_columns].iloc[-72:].values
            X_scaled = preprocessor.feature_scaler.transform(X_input)
            X_seq = X_scaled.reshape(1, 72, -1)
            
            # Predict
            y_pred_scaled = model.predict(X_seq)
            predictions = preprocessor.inverse_transform_predictions(y_pred_scaled[0])
            
            # Get current AQI
            current_aqi = merged['pm25'].iloc[-1] if 'pm25' in merged.columns else predictions[0]
            
            # Store in session state
            st.session_state.predictions = predictions
            st.session_state.current_aqi = current_aqi
            st.session_state.timestamps = [
                datetime.now() + timedelta(hours=i+1) for i in range(len(predictions))
            ]
            st.session_state.historical = merged
            
            st.success("✅ Prediction complete!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.predictions is not None:
    predictions = st.session_state.predictions
    current_aqi = st.session_state.current_aqi
    timestamps = st.session_state.timestamps
    
    # AQI Gauge
    st.markdown("## 📊 Current Air Quality Index")
    
    risk_level = advisor.get_risk_level(current_aqi)
    color = advisor.get_color(risk_level)
    
    # Create gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_aqi,
        title={'text': f"Current AQI", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 300], 'tickcolor': 'white'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 228, 0, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [100, 150], 'color': 'rgba(255, 126, 0, 0.2)'},
                {'range': [150, 200], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [200, 300], 'color': 'rgba(143, 63, 151, 0.2)'},  
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': current_aqi
            }
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Risk level banner
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: {color}33; border-radius: 10px; border: 2px solid {color};'>
        <h2 style='margin: 0; color: white;'>🎯 {risk_level.value}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Forecast Chart
    st.markdown("## 🔮 24-Hour Forecast")
    
    df_forecast = pd.DataFrame({
        'Time': timestamps,
        'Predicted AQI': predictions
    })
    
    # Determine color for each prediction
    colors = [advisor.get_color(advisor.get_risk_level(aqi)) for aqi in predictions]
    
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast['Time'],
        y=df_forecast['Predicted AQI'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, color=colors, line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Add horizontal lines for risk levels
    fig_forecast.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig_forecast.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
    fig_forecast.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy (Sensitive)")
    fig_forecast.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
    
    fig_forecast.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        font={'color': 'white', 'family': 'Inter'},
        xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'title': 'AQI'},
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Health Advisory
    st.markdown("## 💊 Personalized Health Advisory")
    
    user_profile = {
        'age': age,
        'has_respiratory_condition': has_respiratory,
        'has_heart_condition': has_heart,
        'outdoor_activity_level': activity_level
    }
    
    advisory = advisor.generate_advisory(current_aqi, user_profile)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <h3>Risk Profile</h3>
            <p style='font-size: 1.5em; margin: 10px 0;'>
                {'⚠️ Sensitive Group' if advisory['is_sensitive_group'] else '✅ General Public'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Health Effects</h4>
            <p>{advisory['health_effects']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Recommendations")
    for i, rec in enumerate(advisory['recommendations'], 1):
        st.markdown(f"**{i}.** {rec}")
    
    st.info(f"💡 {advisory['disclaimer']}")
    
    # Historical Trends
    if hasattr(st.session_state, 'historical'):
        st.markdown("---")
        st.markdown("## 📈 Historical Trends (Last 7 Days)")
        
        historical = st.session_state.historical
        historical['datetime'] = pd.to_datetime(historical['datetime'])
        
        if 'pm25' in historical.columns:
            fig_hist = px.line(
                historical,
                x='datetime',
                y='pm25',
                title='PM2.5 Levels Over Time'
            )
            
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                font={'color': 'white', 'family': 'Inter'},
                xaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'title': 'Date'},
                yaxis={'gridcolor': 'rgba(255,255,255,0.1)', 'title': 'PM2.5 (μg/m³)'},
                height=300
            )
            
            fig_hist.update_traces(line=dict(color='#7b2cbf', width=2))
            
            st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px; background: rgba(255, 255, 255, 0.03); border-radius: 20px; margin: 30px 0;'>
        <h2>👈 Get Started</h2>
        <p style='font-size: 1.2em; color: #888;'>
            Configure your location and health profile in the sidebar,<br>
            then click <strong>"🚀 Predict Air Quality"</strong> to see the forecast.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='text-align: center;'>
            <h3>🤖 AI-Powered</h3>
            <p>LSTM neural networks predict AQI 24 hours ahead with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='text-align: center;'>
            <h3>👤 Personalized</h3>
            <p>Health advisories tailored to your age, conditions, and activity level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card' style='text-align: center;'>
            <h3>🌍 SDG Aligned</h3>
            <p>Supports Climate Action (13), Good Health (3), and Sustainable Cities (11)</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌟 Built with AI for Software Engineering | Supporting UN Sustainable Development Goals</p>
    <p style='font-size: 0.9em;'>Data sources: OpenAQ, Open-Meteo | Model: LSTM Neural Network</p>
</div>
""", unsafe_allow_html=True)
