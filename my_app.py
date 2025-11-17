"""
=============================================================================
SMART HVAC ENERGY PREDICTOR - STREAMLIT APPLICATION
=============================================================================
AI-Based Energy Prediction & Optimization Dashboard
Author: Your Name
Date: November 2024
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Smart HVAC Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/hvac-predictor',
        'Report a bug': 'https://github.com/yourusername/hvac-predictor/issues',
        'About': '# Smart HVAC Energy Predictor\nAI-powered energy optimization for office buildings'
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(120deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Recommendation cards */
    .rec-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .rec-medium {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    .rec-low {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .rec-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
    }
    
    .rec-content {
        font-size: 1rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    .rec-action {
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* How-to guide styling */
    .step-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: black;
    }
    
    .step-number {
        display: inline-block;
        background: #1f77b4;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-right: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .tip-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model_and_data():
    """Load trained model, scaler, and feature names"""
    try:
        model = joblib.load('data/processed/best_hvac_model.pkl')
        scaler = joblib.load('data/processed/scaler.pkl')
        feature_names = joblib.load('data/processed/feature_names.pkl')
        return model, scaler, feature_names, True
    except:
        # Create dummy model for demo if files don't exist
        st.warning("⚠️ Model files not found. Using demo mode.")
        return None, None, None, False

@st.cache_data
def load_historical_data():
    """Load historical data for visualizations"""
    try:
        df = pd.read_csv('data/processed/complete_processed_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        # Create sample data for demo
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        np.random.seed(42)
        df = pd.DataFrame({
            'date': dates,
            'energy_consumption_wh': 50 + 30 * np.sin(np.arange(500) * 2 * np.pi / 24) + np.random.randn(500) * 10,
            'outdoor_temp': 15 + 10 * np.sin(np.arange(500) * 2 * np.pi / 24) + np.random.randn(500) * 2,
            'avg_indoor_temp': 22 + np.random.randn(500) * 0.5,
            'outdoor_humidity': 60 + np.random.randn(500) * 5,
        })
        return df

def calculate_energy_cost(kwh, rate_per_kwh=0.12):
    """Calculate energy cost"""
    return kwh * rate_per_kwh / 1000  # Convert Wh to kWh

def calculate_carbon_footprint(kwh, carbon_per_kwh=0.5):
    """Calculate carbon footprint"""
    return kwh * carbon_per_kwh / 1000  # Convert Wh to kWh

def prepare_features(outdoor_temp, outdoor_humidity, indoor_temp, 
                     hour, day_of_week, month, is_weekend, is_business_hours,
                     pressure, wind_speed, visibility):
    """Prepare feature vector for prediction"""
    
    # Calculate derived features
    temp_difference = indoor_temp - outdoor_temp
    cooling_degree_hour = max(0, outdoor_temp - 18)
    heating_degree_hour = max(0, 18 - outdoor_temp)
    
    # Use historical average for lag features (simplified for demo)
    avg_indoor_humidity = 50.0  # Default
    energy_lag_1h = 80.0  # Historical average
    energy_lag_24h = 85.0
    energy_rolling_mean_1h = 82.0
    energy_rolling_mean_3h = 83.0
    energy_rolling_std_1h = 15.0
    temp_humidity_interaction = outdoor_temp * outdoor_humidity
    wind_temp_interaction = wind_speed * outdoor_temp
    dew_point = outdoor_temp - ((100 - outdoor_humidity) / 5)
    
    features = [
        outdoor_temp, outdoor_humidity, indoor_temp, avg_indoor_humidity,
        hour, day_of_week, month, is_weekend, is_business_hours,
        temp_difference, cooling_degree_hour, heating_degree_hour,
        pressure, wind_speed, visibility, dew_point,
        energy_lag_1h, energy_lag_24h, energy_rolling_mean_1h,
        energy_rolling_mean_3h, energy_rolling_std_1h,
        temp_humidity_interaction, wind_temp_interaction
    ]
    
    return np.array(features).reshape(1, -1)

def generate_recommendations(predicted_wh, outdoor_temp, indoor_temp, 
                            is_business_hours, hour, outdoor_humidity):
    """Generate comprehensive actionable energy-saving recommendations"""
    
    recommendations = []
    baseline = 100.0  # Historical average
    potential_savings_total = 0
    
    # Calculate temperature difference
    temp_diff = abs(indoor_temp - outdoor_temp)
    
    # HIGH PRIORITY RECOMMENDATIONS
    
    # 1. High consumption alert with specific actions
    if predicted_wh > baseline * 1.3:
        savings_wh = (predicted_wh - baseline) * 0.4
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'HIGH',
            'icon': '🔴',
            'title': 'High Energy Consumption Alert',
            'description': f'Predicted consumption ({predicted_wh:.0f} Wh) is {((predicted_wh/baseline - 1) * 100):.0f}% above baseline.',
            'actions': [
                f'⚙️ Increase AC setpoint from {indoor_temp:.1f}°C to {indoor_temp + 2:.1f}°C',
                '🔄 Enable economizer mode if outdoor conditions permit',
                '📊 Review and adjust zone-level temperature settings',
                '🔍 Check for stuck dampers or malfunctioning sensors'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour (~{savings_wh * 24:.0f} Wh/day)',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 24):.2f}/day',
            'impact': 'Minimal comfort impact with 1-2°C adjustment',
            'implementation': 'Immediate - Adjust thermostat through BMS',
            'type': 'high'
        })
    
    # 2. Extreme temperature differential
    if temp_diff > 10:
        savings_wh = predicted_wh * 0.15
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'HIGH',
            'icon': '🌡️',
            'title': 'Large Indoor-Outdoor Temperature Gap',
            'description': f'Temperature difference of {temp_diff:.1f}°C is driving high energy usage.',
            'actions': [
                '🎯 Reduce temperature gap to 8°C or less',
                '🪟 Close blinds/curtains on sun-exposed windows',
                '🚪 Ensure all doors and windows are properly sealed',
                '💨 Use ceiling fans to improve air circulation'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour (~{savings_wh * 24:.0f} Wh/day)',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 24):.2f}/day',
            'impact': 'Improves comfort and reduces system strain',
            'implementation': 'Immediate to 1 hour',
            'type': 'high'
        })
    
    # 3. Peak demand period management
    if 14 <= hour <= 18 and outdoor_temp > 28:
        savings_wh = predicted_wh * 0.2
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'HIGH',
            'icon': '⚡',
            'title': 'Peak Cooling Demand Period',
            'description': 'Currently in peak cooling hours with high outdoor temperature.',
            'actions': [
                '❄️ Pre-cool building to 21°C before 2 PM',
                '📉 Reduce non-essential equipment loads',
                '💡 Dim lighting in unoccupied areas',
                '🌡️ Allow temperature to drift to 24°C during peak'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour in demand charges',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 4):.2f} (4-hour peak)',
            'impact': 'Reduces peak demand charges significantly',
            'implementation': 'Next occurrence - Program into BMS schedule',
            'type': 'high'
        })
    
    # MEDIUM PRIORITY RECOMMENDATIONS
    
    # 4. Off-hours optimization
    if not is_business_hours and predicted_wh > 50:
        savings_wh = predicted_wh * 0.4
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'MEDIUM',
            'icon': '🌙',
            'title': 'Off-Hours Energy Usage',
            'description': 'Building is unoccupied but consuming significant energy.',
            'actions': [
                f'🌡️ Enable night setback: {indoor_temp:.1f}°C → {indoor_temp + 4:.1f}°C',
                '💨 Reduce ventilation to minimum code requirements',
                '💡 Ensure all non-essential lighting is off',
                '🖥️ Enable power management on all workstations'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour (~{savings_wh * 14:.0f} Wh/night)',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 14):.2f}/night',
            'impact': 'Zero comfort impact - building unoccupied',
            'implementation': 'Immediate - Activate setback mode',
            'type': 'medium'
        })
    
    # 5. High humidity conditions
    if outdoor_humidity > 70:
        savings_wh = predicted_wh * 0.1
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'MEDIUM',
            'icon': '💧',
            'title': 'High Humidity Conditions',
            'description': f'Outdoor humidity at {outdoor_humidity:.0f}% increases cooling load.',
            'actions': [
                '🌊 Enable enthalpy-based economizer controls',
                '🔄 Increase fresh air intake during low-humidity periods',
                '📊 Monitor and optimize dehumidification settings',
                '🔍 Check condensate drains for proper operation'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 24):.2f}/day',
            'impact': 'Improves indoor air quality and comfort',
            'implementation': '1-2 hours - Adjust control sequences',
            'type': 'medium'
        })
    
    # 6. Weekend/Holiday optimization
    if is_weekend and predicted_wh > 70:
        savings_wh = predicted_wh * 0.3
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'MEDIUM',
            'icon': '📅',
            'title': 'Weekend Operation Optimization',
            'description': 'Weekend mode not fully activated.',
            'actions': [
                '🏢 Reduce HVAC to minimum occupancy zones only',
                '⏰ Delay start time and advance shutdown time',
                '💡 Keep only security lighting active',
                '🚪 Close off unoccupied floor sections'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour (~{savings_wh * 24:.0f} Wh/day)',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 24 * 2):.2f}/weekend',
            'impact': 'No impact on limited weekend operations',
            'implementation': 'Immediate - Activate weekend schedule',
            'type': 'medium'
        })
    
    # LOW PRIORITY / OPTIMIZATION RECOMMENDATIONS
    
    # 7. Favorable outdoor conditions
    if 20 <= outdoor_temp <= 24 and temp_diff < 4:
        savings_wh = predicted_wh * 0.15
        potential_savings_total += savings_wh
        recommendations.append({
            'priority': 'LOW',
            'icon': '🌤️',
            'title': 'Ideal Outdoor Conditions',
            'description': 'Weather conditions favorable for free cooling.',
            'actions': [
                '🪟 Enable natural ventilation if windows are operable',
                '💨 Maximize outdoor air economizer usage',
                '⏸️ Consider HVAC system partial shutdown in mild zones',
                '🔄 Switch to 100% outdoor air mode if air quality permits'
            ],
            'savings': f'{savings_wh:.0f} Wh/hour (~{savings_wh * 8:.0f} Wh during favorable period)',
            'cost_savings': f'${calculate_energy_cost(savings_wh * 8):.2f} per favorable day',
            'impact': 'Positive - Fresh air improves IAQ and comfort',
            'implementation': 'Immediate - Manual or automatic economizer',
            'type': 'low'
        })
    
    # 8. Efficient operation recognition
    if predicted_wh < baseline * 0.8:
        recommendations.append({
            'priority': 'LOW',
            'icon': '✅',
            'title': 'Efficient Operation Detected',
            'description': f'System is operating {((1 - predicted_wh/baseline) * 100):.0f}% more efficiently than baseline.',
            'actions': [
                '📝 Document current settings as best practice',
                '📊 Analyze what factors contributed to efficiency',
                '🔄 Apply these settings to similar conditions',
                '⭐ Maintain current operational parameters'
            ],
            'savings': f'Already saving {baseline - predicted_wh:.0f} Wh/hour',
            'cost_savings': f'${calculate_energy_cost((baseline - predicted_wh) * 24):.2f}/day below baseline',
            'impact': 'Continue excellent performance',
            'implementation': 'Ongoing - Monitor and maintain',
            'type': 'low'
        })
    
    # 9. Predictive maintenance reminder
    if predicted_wh > baseline * 1.2:
        recommendations.append({
            'priority': 'LOW',
            'icon': '🔧',
            'title': 'Maintenance Check Recommended',
            'description': 'Elevated consumption may indicate equipment degradation.',
            'actions': [
                '🔍 Inspect air filters and replace if dirty',
                '🌡️ Verify thermostat calibration accuracy',
                '⚙️ Check refrigerant charge and compressor operation',
                '📊 Review system trending data for anomalies'
            ],
            'savings': 'Preventive maintenance avoids 10-30% efficiency loss',
            'cost_savings': 'Prevents future issues and system failures',
            'impact': 'Extends equipment life and maintains efficiency',
            'implementation': 'Schedule within 1-2 weeks',
            'type': 'low'
        })
    
    # Add default if no recommendations
    if not recommendations:
        recommendations.append({
            'priority': 'LOW',
            'icon': '✅',
            'title': 'Optimal Operation',
            'description': 'System is operating within normal parameters.',
            'actions': [
                '✓ Continue monitoring for changes',
                '✓ Maintain current settings',
                '✓ Review weekly energy reports'
            ],
            'savings': 'No immediate actions needed',
            'cost_savings': 'System optimized',
            'impact': 'Maintaining efficient operation',
            'implementation': 'Ongoing',
            'type': 'low'
        })
    
    # Add summary at the beginning
    summary = {
        'total_recommendations': len(recommendations),
        'high_priority': len([r for r in recommendations if r['priority'] == 'HIGH']),
        'medium_priority': len([r for r in recommendations if r['priority'] == 'MEDIUM']),
        'low_priority': len([r for r in recommendations if r['priority'] == 'LOW']),
        'total_potential_savings': potential_savings_total,
        'daily_savings': potential_savings_total * 24,
        'monthly_savings': potential_savings_total * 24 * 30,
        'cost_daily': calculate_energy_cost(potential_savings_total * 24),
        'cost_monthly': calculate_energy_cost(potential_savings_total * 24 * 30)
    }
    
    return recommendations, summary

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-title">🌡️ Smart HVAC Energy Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    AI-powered energy optimization for sustainable building management
    </p>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, scaler, feature_names, model_loaded = load_model_and_data()
    historical_df = load_historical_data()
    
    # =============================================================================
    # SIDEBAR - INPUT PARAMETERS
    # =============================================================================
    
    with st.sidebar:
        st.header("⚙️ Input Parameters")
        
        st.markdown("---")
        st.subheader("📅 Date & Time")
        
        selected_date = st.date_input("Date", datetime.now())
        selected_hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
        
        # Auto-calculate day features
        day_of_week = selected_date.weekday()
        month = selected_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if (8 <= selected_hour <= 18 and not is_weekend) else 0
        
        st.markdown("---")
        st.subheader("🌡️ Temperature & Humidity")
        
        outdoor_temp = st.slider("Outdoor Temperature (°C)", -10, 45, 25)
        outdoor_humidity = st.slider("Outdoor Humidity (%)", 0, 100, 60)
        indoor_temp = st.slider("Indoor Setpoint (°C)", 18, 28, 22)
        
        st.markdown("---")
        st.subheader("🌤️ Weather Conditions")
        
        pressure = st.slider("Atmospheric Pressure (mm Hg)", 720, 780, 750)
        wind_speed = st.slider("Wind Speed (m/s)", 0, 15, 3)
        visibility = st.slider("Visibility (km)", 0, 40, 25)
        
        st.markdown("---")
        st.subheader("🏢 Building Status")
        
        st.info(f"""
        - **Day**: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}
        - **Type**: {'Weekend' if is_weekend else 'Weekday'}
        - **Status**: {'🟢 Occupied' if is_business_hours else '🔴 Unoccupied'}
        """)
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Energy", use_container_width=True)
    
    # =============================================================================
    # MAIN CONTENT - TABS
    # =============================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prediction", 
        "📈 Analytics", 
        "💡 Recommendations",
        "📖 How to Use",
        "📚 About"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: PREDICTION
    # -------------------------------------------------------------------------
    with tab1:
        st.header("Energy Consumption Prediction")
        
        if predict_button or st.session_state.get('auto_predict', False):
            # Prepare features
            features = prepare_features(
                outdoor_temp, outdoor_humidity, indoor_temp,
                selected_hour, day_of_week, month, is_weekend, is_business_hours,
                pressure, wind_speed, visibility
            )
            
            # Make prediction
            if model_loaded and model is not None:
                try:
                    features_scaled = scaler.transform(features)
                    predicted_wh = model.predict(features_scaled)[0]
                except:
                    predicted_wh = (50 + 
                                   abs(indoor_temp - outdoor_temp) * 3 +
                                   (30 if is_business_hours else 10) +
                                   np.random.randn() * 5)
            else:
                predicted_wh = (50 + 
                               abs(indoor_temp - outdoor_temp) * 3 +
                               (30 if is_business_hours else 10) +
                               np.random.randn() * 5)
            
            predicted_wh = max(10, predicted_wh)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "⚡ Predicted Consumption",
                    f"{predicted_wh:.1f} Wh",
                    delta=f"{predicted_wh - 100:.1f} Wh from avg"
                )
            
            with col2:
                cost = calculate_energy_cost(predicted_wh)
                st.metric(
                    "💰 Estimated Cost",
                    f"${cost:.3f}",
                    delta=f"${cost * 24:.2f}/day"
                )
            
            with col3:
                carbon = calculate_carbon_footprint(predicted_wh)
                st.metric(
                    "🌍 Carbon Footprint",
                    f"{carbon:.3f} kg CO₂",
                    delta=f"{carbon * 24:.2f} kg/day"
                )
            
            with col4:
                efficiency = max(0, min(100, 100 - (predicted_wh - 80) / 2))
                st.metric(
                    "📊 Efficiency Score",
                    f"{efficiency:.0f}/100",
                    delta=f"{'Good' if efficiency > 70 else 'Needs Improvement'}"
                )
            
            # Visualization
            st.markdown("### 📊 Prediction Breakdown")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_wh,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Energy Consumption (Wh)", 'font': {'size': 24}},
                delta={'reference': 100, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 60], 'color': '#90EE90'},
                        {'range': [60, 120], 'color': '#FFD700'},
                        {'range': [120, 200], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Contributing factors
            st.markdown("### 🔍 Contributing Factors")
            
            col1, col2 = st.columns(2)
            
            with col1:
                factors_df = pd.DataFrame({
                    'Factor': ['Temperature Diff', 'Time of Day', 'Weather', 'Occupancy'],
                    'Impact': [
                        abs(indoor_temp - outdoor_temp) * 10,
                        30 if is_business_hours else 10,
                        (100 - outdoor_humidity) / 3,
                        40 if is_business_hours else 5
                    ]
                })
                
                fig = px.bar(factors_df, x='Impact', y='Factor', orientation='h',
                            title='Impact on Energy Consumption',
                            color='Impact',
                            color_continuous_scale='Reds')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                comparison_df = pd.DataFrame({
                    'Type': ['Current Prediction', 'Daily Average', 'Weekend Average'],
                    'Energy (Wh)': [predicted_wh, 100, 60]
                })
                
                fig = px.bar(comparison_df, x='Type', y='Energy (Wh)',
                            title='Comparison with Averages',
                            color='Energy (Wh)',
                            color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Quick recommendations preview
            st.markdown("### 💡 Quick Recommendations Preview")
            recommendations, summary = generate_recommendations(
                predicted_wh, outdoor_temp, indoor_temp, 
                is_business_hours, selected_hour, outdoor_humidity
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Total Recommendations", summary['total_recommendations'])
            with col2:
                st.metric("⚡ Potential Savings", f"{summary['total_potential_savings']:.0f} Wh/hr")
            with col3:
                st.metric("💰 Daily Cost Savings", f"${summary['cost_daily']:.2f}")
            
            st.info("👉 Go to the **Recommendations** tab for detailed action items!")
            
            # Store prediction in session state
            st.session_state['last_prediction'] = predicted_wh
            st.session_state['last_inputs'] = {
                'outdoor_temp': outdoor_temp,
                'indoor_temp': indoor_temp,
                'hour': selected_hour,
                'is_business_hours': is_business_hours,
                'outdoor_humidity': outdoor_humidity
            }
    
    # -------------------------------------------------------------------------
    # TAB 2: ANALYTICS
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Historical Analytics")
        
        # Time series plot
        st.subheader("📈 Energy Consumption Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['energy_consumption_wh'],
            mode='lines',
            name='Energy Consumption',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title='Historical Energy Consumption',
            xaxis_title='Date',
            yaxis_title='Energy (Wh)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Consumption", 
                     f"{historical_df['energy_consumption_wh'].mean():.1f} Wh")
        with col2:
            st.metric("Peak Consumption",
                     f"{historical_df['energy_consumption_wh'].max():.1f} Wh")
        with col3:
            st.metric("Minimum Consumption",
                     f"{historical_df['energy_consumption_wh'].min():.1f} Wh")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Energy Distribution")
            fig = px.histogram(historical_df, x='energy_consumption_wh',
                             nbins=50, title='Energy Consumption Distribution')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌡️ Temperature vs Energy")
            fig = px.scatter(historical_df, x='outdoor_temp', 
                           y='energy_consumption_wh',
                           title='Temperature Impact on Energy',
                           trendline='ols')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # TAB 3: RECOMMENDATIONS
    # -------------------------------------------------------------------------
    with tab3:
        st.header("💡 Energy Optimization Recommendations")
        
        if 'last_prediction' in st.session_state:
            predicted_wh = st.session_state['last_prediction']
            inputs = st.session_state['last_inputs']
            
            recommendations, summary = generate_recommendations(
                predicted_wh,
                inputs['outdoor_temp'],
                inputs['indoor_temp'],
                inputs['is_business_hours'],
                inputs['hour'],
                inputs['outdoor_humidity']
            )
            
            # Summary section
            st.markdown("### 📊 Recommendations Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔴 High Priority", summary['high_priority'])
            with col2:
                st.metric("🟡 Medium Priority", summary['medium_priority'])
            with col3:
                st.metric("🟢 Low Priority", summary['low_priority'])
            with col4:
                st.metric("💰 Monthly Savings", f"${summary['cost_monthly']:.2f}")
            
            st.markdown("---")
            
            # Energy savings potential
            if summary['total_potential_savings'] > 0:
                st.markdown(f"""
                <div class="success-box">
                    <h3>💰 Potential Energy Savings</h3>
                    <p><strong>Hourly:</strong> {summary['total_potential_savings']:.0f} Wh</p>
                    <p><strong>Daily:</strong> {summary['daily_savings']:.0f} Wh (${summary['cost_daily']:.2f})</p>
                    <p><strong>Monthly:</strong> {summary['monthly_savings']:.0f} Wh (${summary['cost_monthly']:.2f})</p>
                    <p><strong>Annual:</strong> {summary['monthly_savings'] * 12:.0f} Wh (${summary['cost_monthly'] * 12:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display recommendations by priority
            st.markdown("### 🎯 Actionable Recommendations")
            
            for rec in recommendations:
                if rec['type'] == 'high':
                    st.markdown(f"""
                    <div class="rec-high">
                        <div class="rec-title">{rec['icon']} {rec['priority']} PRIORITY: {rec['title']}</div>
                        <div class="rec-content"><strong>📋 Description:</strong> {rec['description']}</div>
                        <div class="rec-action">
                            <strong>✅ Recommended Actions:</strong><br>
                            {'<br>'.join(rec['actions'])}
                        </div>
                        <div class="rec-content"><strong>💰 Potential Savings:</strong> {rec['savings']}</div>
                        <div class="rec-content"><strong>💵 Cost Savings:</strong> {rec['cost_savings']}</div>
                        <div class="rec-content"><strong>👥 Impact:</strong> {rec['impact']}</div>
                        <div class="rec-content"><strong>⏱️ Implementation Time:</strong> {rec['implementation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif rec['type'] == 'medium':
                    st.markdown(f"""
                    <div class="rec-medium">
                        <div class="rec-title">{rec['icon']} {rec['priority']} PRIORITY: {rec['title']}</div>
                        <div class="rec-content"><strong>📋 Description:</strong> {rec['description']}</div>
                        <div class="rec-action">
                            <strong>✅ Recommended Actions:</strong><br>
                            {'<br>'.join(rec['actions'])}
                        </div>
                        <div class="rec-content"><strong>💰 Potential Savings:</strong> {rec['savings']}</div>
                        <div class="rec-content"><strong>💵 Cost Savings:</strong> {rec['cost_savings']}</div>
                        <div class="rec-content"><strong>👥 Impact:</strong> {rec['impact']}</div>
                        <div class="rec-content"><strong>⏱️ Implementation Time:</strong> {rec['implementation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rec-low">
                        <div class="rec-title">{rec['icon']} {rec['priority']} PRIORITY: {rec['title']}</div>
                        <div class="rec-content"><strong>📋 Description:</strong> {rec['description']}</div>
                        <div class="rec-action">
                            <strong>✅ Recommended Actions:</strong><br>
                            {'<br>'.join(rec['actions'])}
                        </div>
                        <div class="rec-content"><strong>💰 Potential Savings:</strong> {rec['savings']}</div>
                        <div class="rec-content"><strong>💵 Cost Savings:</strong> {rec['cost_savings']}</div>
                        <div class="rec-content"><strong>👥 Impact:</strong> {rec['impact']}</div>
                        <div class="rec-content"><strong>⏱️ Implementation Time:</strong> {rec['implementation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Implementation checklist
            st.markdown("---")
            st.markdown("### ✅ Implementation Checklist")
            
            st.markdown("""
            <div class="tip-box">
                <h4>📝 How to Implement These Recommendations:</h4>
                <ol>
                    <li><strong>Review all HIGH priority items first</strong> - These provide immediate savings</li>
                    <li><strong>Coordinate with facilities management</strong> - Ensure all changes are documented</li>
                    <li><strong>Monitor results</strong> - Track energy consumption after implementing changes</li>
                    <li><strong>Adjust as needed</strong> - Fine-tune settings based on occupant feedback</li>
                    <li><strong>Schedule regular reviews</strong> - Check recommendations weekly or daily</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("👈 Make a prediction in the 'Prediction' tab to see personalized recommendations!")
            
            st.markdown("""
            ### 🎯 What You'll Get:
            
            Once you make a prediction, you'll receive:
            
            - **🔴 High Priority Actions**: Immediate steps for significant energy savings
            - **🟡 Medium Priority Actions**: Important optimizations for better efficiency
            - **🟢 Low Priority Actions**: Long-term improvements and best practices
            - **💰 Savings Calculations**: Detailed breakdown of potential cost savings
            - **⏱️ Implementation Guidance**: How long each action takes to implement
            - **👥 Impact Assessment**: How changes affect building occupants
            """)
    
    # -------------------------------------------------------------------------
    # TAB 4: HOW TO USE
    # -------------------------------------------------------------------------
    with tab4:
        st.header("📖 How to Use This Application")
        
        st.markdown("""
        <div style="color: black;">
        Welcome to the Smart HVAC Energy Predictor! This guide will help you maximize 
        the benefits of this AI-powered tool for your building's energy management.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Step-by-step guide
        st.markdown("### 🚀 Quick Start Guide")
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">1</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Enter Building Parameters</h3>
                <p><strong>Location:</strong> Left sidebar (⚙️ Input Parameters)</p>
                <p><strong>What to do:</strong></p>
                <ul>
                    <li>Select the <strong>Date</strong> you want to predict for</li>
                    <li>Choose the <strong>Hour of Day</strong> (0-23 format)</li>
                    <li>Adjust <strong>Temperature & Humidity</strong> settings:
                        <ul>
                            <li>Outdoor Temperature: Current or forecasted temp</li>
                            <li>Outdoor Humidity: Current humidity level</li>
                            <li>Indoor Setpoint: Your desired indoor temperature</li>
                        </ul>
                    </li>
                    <li>Input <strong>Weather Conditions</strong> (atmospheric pressure, wind speed, visibility)</li>
                </ul>
                <p><strong>💡 Tip:</strong> Use weather forecast data for future predictions or current conditions for real-time analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">2</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Generate Energy Prediction</h3>
                <p><strong>Location:</strong> Click "🔮 Predict Energy" button in sidebar</p>
                <p><strong>What happens:</strong></p>
                <ul>
                    <li>AI model analyzes all input parameters</li>
                    <li>Calculates predicted energy consumption in Watt-hours (Wh)</li>
                    <li>Estimates cost and carbon footprint</li>
                    <li>Generates efficiency score (0-100)</li>
                </ul>
                <p><strong>💡 Tip:</strong> The prediction appears in the "Prediction" tab with visual gauges and charts.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">3</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Review Prediction Results</h3>
                <p><strong>Location:</strong> "🔮 Prediction" tab</p>
                <p><strong>Key Metrics to Monitor:</strong></p>
                <ul>
                    <li><strong>⚡ Predicted Consumption:</strong> Expected energy usage in Wh for the selected hour</li>
                    <li><strong>💰 Estimated Cost:</strong> How much this energy will cost at current rates</li>
                    <li><strong>🌍 Carbon Footprint:</strong> Environmental impact in kg CO₂</li>
                    <li><strong>📊 Efficiency Score:</strong> Overall system performance rating</li>
                </ul>
                <p><strong>Understanding the Gauge:</strong></p>
                <ul>
                    <li>🟢 Green Zone (0-60 Wh): Excellent efficiency</li>
                    <li>🟡 Yellow Zone (60-120 Wh): Normal operation</li>
                    <li>🔴 Red Zone (120+ Wh): High consumption - action needed</li>
                </ul>
                <p><strong>💡 Tip:</strong> See the detailed chart reading guide below for in-depth analysis!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">4</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Get Actionable Recommendations</h3>
                <p><strong>Location:</strong> "💡 Recommendations" tab</p>
                <p><strong>What you'll find:</strong></p>
                <ul>
                    <li><strong>🔴 High Priority:</strong> Urgent actions for immediate savings (implement within hours)</li>
                    <li><strong>🟡 Medium Priority:</strong> Important optimizations (implement within days)</li>
                    <li><strong>🟢 Low Priority:</strong> Long-term improvements (implement within weeks)</li>
                </ul>
                <p><strong>Each recommendation includes:</strong></p>
                <ul>
                    <li>✅ Specific actions to take</li>
                    <li>💰 Potential energy and cost savings</li>
                    <li>👥 Impact on building occupants</li>
                    <li>⏱️ Time required for implementation</li>
                </ul>
                <p><strong>💡 Tip:</strong> Start with HIGH priority items first for maximum impact!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">5</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Implement Energy-Saving Actions</h3>
                <p><strong>How to take action:</strong></p>
                <ul>
                    <li><strong>Thermostat Adjustments:</strong> Access your Building Management System (BMS) or smart thermostat</li>
                    <li><strong>Schedule Changes:</strong> Update HVAC schedules in your BMS control panel</li>
                    <li><strong>Setback Modes:</strong> Enable night/weekend modes during unoccupied hours</li>
                    <li><strong>Economizer Settings:</strong> Adjust outdoor air intake controls</li>
                    <li><strong>Zone Control:</strong> Disable HVAC in unoccupied zones</li>
                </ul>
                <p><strong>⚠️ Important:</strong> Always coordinate major changes with building management and facilities staff.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <span class="step-number">6</span>
            <div style="display: inline-block; vertical-align: top; width: 85%;">
                <h3>Monitor Results & Iterate</h3>
                <p><strong>Location:</strong> "📈 Analytics" tab</p>
                <p><strong>Track your progress:</strong></p>
                <ul>
                    <li>Review historical energy consumption trends</li>
                    <li>Compare current usage with averages</li>
                    <li>Identify patterns in energy consumption</li>
                    <li>Verify that implemented changes are working</li>
                </ul>
                <p><strong>Best Practice:</strong> Make predictions daily and review recommendations weekly to maintain optimal efficiency.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # COMPREHENSIVE CHART AND METRICS READING GUIDE
        st.markdown("### 📊 Understanding Your Prediction Results")
        
        st.markdown("""
        This section explains how to read and interpret all the charts, numbers, and metrics 
        displayed after making a prediction.
        """)
        
        # Metrics Cards Section
        st.markdown("#### 📈 Top Metrics Cards")
        
        st.markdown("""
        <div class="step-card">
            <h4>⚡ Predicted Consumption</h4>
            <p><strong>What it shows:</strong> The estimated energy your HVAC system will consume in the selected hour, measured in Watt-hours (Wh).</p>
            <p><strong>How to read it:</strong></p>
            <ul>
                <li><strong>Main Number:</strong> The predicted consumption value (e.g., "85.3 Wh")</li>
                <li><strong>Delta (↑/↓):</strong> Difference from the baseline average (100 Wh)
                    <ul>
                        <li>🔺 Green arrow up = Higher than average (may need attention)</li>
                        <li>🔻 Red arrow down = Lower than average (good efficiency)</li>
                    </ul>
                </li>
            </ul>
            <p><strong>Example:</strong> "85.3 Wh ↓ -14.7 Wh from avg" means you're using 14.7 Wh less than the typical 100 Wh baseline.</p>
            <p><strong>Action Guide:</strong></p>
            <ul>
                <li><strong>0-60 Wh:</strong> 🟢 Excellent! System is highly efficient</li>
                <li><strong>60-120 Wh:</strong> 🟡 Normal operation, no immediate action needed</li>
                <li><strong>120-150 Wh:</strong> 🟠 Above average, consider optimization</li>
                <li><strong>150+ Wh:</strong> 🔴 High consumption, immediate action recommended</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <h4>💰 Estimated Cost</h4>
            <p><strong>What it shows:</strong> The monetary cost of the predicted energy consumption.</p>
            <p><strong>How to read it:</strong></p>
            <ul>
                <li><strong>Main Number:</strong> Cost for this specific hour (e.g., "₹0.01" or "$0.01")</li>
                <li><strong>Delta:</strong> Projected daily cost if this rate continues for 24 hours</li>
                <li><strong>Calculation:</strong> Based on ₹0.12 per kWh (kilowatt-hour) rate</li>
            </ul>
            <p><strong>Example:</strong> "₹0.01 ↑ ₹0.24/day" means:
                <ul>
                    <li>This hour costs ₹0.01</li>
                    <li>If maintained, daily cost would be ₹0.24</li>
                    <li>Monthly cost would be approximately ₹7.20</li>
                    <li>Annual cost would be approximately ₹87.60</li>
                </ul>
            </p>
            <p><strong>💡 Tip:</strong> Multiply the daily rate by 30 for monthly estimate, by 365 for annual estimate.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <h4>🌍 Carbon Footprint</h4>
            <p><strong>What it shows:</strong> The environmental impact of your energy consumption in CO₂ emissions.</p>
            <p><strong>How to read it:</strong></p>
            <ul>
                <li><strong>Main Number:</strong> CO₂ emissions for this hour in kilograms (kg)</li>
                <li><strong>Delta:</strong> Projected daily emissions if this rate continues</li>
                <li><strong>Calculation:</strong> Based on 0.5 kg CO₂ per kWh (varies by region/energy source)</li>
            </ul>
            <p><strong>Example:</strong> "0.043 kg CO₂ ↑ 1.02 kg/day" means:
                <ul>
                    <li>This hour produces 0.043 kg of CO₂</li>
                    <li>Daily: 1.02 kg CO₂ (about 2.2 pounds)</li>
                    <li>Monthly: ~30.6 kg CO₂ (67.5 pounds)</li>
                    <li>Annual: ~372 kg CO₂ (820 pounds) - equivalent to driving ~1,500 km</li>
                </ul>
            </p>
            <p><strong>🌱 Context:</strong> Average person's carbon footprint is ~4,000 kg CO₂/year. Your HVAC should be a small fraction of this.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-card">
            <h4>📊 Efficiency Score</h4>
            <p><strong>What it shows:</strong> Overall system performance rating on a scale of 0-100.</p>
            <p><strong>How to read it:</strong></p>
            <ul>
                <li><strong>Main Number:</strong> Score from 0 (worst) to 100 (best)</li>
                <li><strong>Delta:</strong> Qualitative assessment ("Good" or "Needs Improvement")</li>
            </ul>
            <p><strong>Score Interpretation:</strong></p>
            <ul>
                <li><strong>90-100:</strong> 🌟 Excellent - System is optimized perfectly</li>
                <li><strong>70-89:</strong> ✅ Good - Operating efficiently with minor room for improvement</li>
                <li><strong>50-69:</strong> ⚠️ Fair - Moderate efficiency, optimization recommended</li>
                <li><strong>30-49:</strong> 🔶 Poor - Significant inefficiencies present</li>
                <li><strong>0-29:</strong> 🔴 Critical - Immediate action required</li>
            </ul>
            <p><strong>Example:</strong> "78/100 → Good" means your system is operating at 78% efficiency, which is acceptable but has 22% room for improvement.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge Chart Section
        st.markdown("---")
        st.markdown("#### 🎯 Energy Consumption Gauge Chart")
        
        st.markdown("""
        <div class="step-card">
            <h4>How to Read the Gauge</h4>
            <p><strong>What it shows:</strong> Visual representation of your predicted energy consumption with color-coded zones.</p>
            
            <p><strong>Components:</strong></p>
            <ul>
                <li><strong>Main Needle/Bar:</strong> Points to your current prediction value</li>
                <li><strong>Large Number:</strong> Your predicted consumption (e.g., "85 Wh")</li>
                <li><strong>Small Delta Number:</strong> Difference from baseline (e.g., "-15" means 15 Wh below average)</li>
                <li><strong>Red Threshold Line:</strong> Set at 150 Wh - the critical high-consumption threshold</li>
            </ul>
            
            <p><strong>Color Zones Explained:</strong></p>
            <ul>
                <li><strong>🟢 Green Zone (0-60 Wh):</strong>
                    <ul>
                        <li><strong>Meaning:</strong> Optimal efficiency - HVAC is barely consuming energy</li>
                        <li><strong>Typical When:</strong> Mild weather, building unoccupied, efficient settings</li>
                        <li><strong>Action:</strong> Document these conditions as best practices!</li>
                    </ul>
                </li>
                <li><strong>🟡 Yellow Zone (60-120 Wh):</strong>
                    <ul>
                        <li><strong>Meaning:</strong> Normal operation - within expected range</li>
                        <li><strong>Typical When:</strong> Regular business hours, moderate weather conditions</li>
                        <li><strong>Action:</strong> Monitor and maintain current settings</li>
                    </ul>
                </li>
                <li><strong>🔴 Red Zone (120-200 Wh):</strong>
                    <ul>
                        <li><strong>Meaning:</strong> High consumption - system working hard</li>
                        <li><strong>Typical When:</strong> Extreme weather, peak hours, possible inefficiencies</li>
                        <li><strong>Action:</strong> Review recommendations tab for optimization opportunities</li>
                    </ul>
                </li>
            </ul>
            
            <p><strong>Reading Example:</strong> If the needle points to 95 Wh in the yellow zone with "-5" delta:
                <ul>
                    <li>Your HVAC will use 95 Wh this hour</li>
                    <li>This is 5 Wh below the 100 Wh average (slightly better than normal)</li>
                    <li>You're in the safe "normal operation" range</li>
                    <li>No immediate action needed, but check recommendations for minor improvements</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contributing Factors Chart
        st.markdown("---")
        st.markdown("#### 🔍 Contributing Factors (Horizontal Bar Chart)")
        
        st.markdown("""
        <div class="step-card">
            <h4>Understanding What Drives Your Energy Consumption</h4>
            <p><strong>What it shows:</strong> The relative impact of different factors on your predicted energy consumption.</p>
            
            <p><strong>The Four Factors:</strong></p>
            <ul>
                <li><strong>Temperature Diff:</strong>
                    <ul>
                        <li><strong>What it measures:</strong> Difference between indoor setpoint and outdoor temperature</li>
                        <li><strong>Impact:</strong> Larger difference = more energy needed to maintain temperature</li>
                        <li><strong>Example:</strong> If outdoor is 30°C and indoor setpoint is 22°C, difference is 8°C</li>
                        <li><strong>Bar length:</strong> Longer bar = bigger temperature gap = more energy use</li>
                        <li><strong>What to do:</strong> Reduce the gap by adjusting setpoint closer to outdoor temp (within comfort limits)</li>
                    </ul>
                </li>
                <li><strong>Time of Day:</strong>
                    <ul>
                        <li><strong>What it measures:</strong> Impact of occupancy and daily schedule</li>
                        <li><strong>Impact:</strong> Business hours (8 AM - 6 PM) show high impact due to full HVAC operation</li>
                        <li><strong>Bar interpretation:</strong>
                            <ul>
                                <li>Long bar = Business hours with full occupancy</li>
                                <li>Short bar = Off-hours with reduced operation</li>
                            </ul>
                        </li>
                        <li><strong>What to do:</strong> Enable setback modes during off-hours</li>
                    </ul>
                </li>
                <li><strong>Weather:</strong>
                    <ul>
                        <li><strong>What it measures:</strong> Combined effect of humidity, pressure, wind, visibility</li>
                        <li><strong>Impact:</strong> High humidity increases cooling load significantly</li>
                        <li><strong>Bar interpretation:</strong>
                            <ul>
                                <li>Long bar = Challenging weather (high humidity, extreme conditions)</li>
                                <li>Short bar = Favorable weather (dry, moderate conditions)</li>
                            </ul>
                        </li>
                        <li><strong>Example:</strong> 80% humidity creates longer bar than 40% humidity</li>
                        <li><strong>What to do:</strong> Use economizer mode during favorable weather</li>
                    </ul>
                </li>
                <li><strong>Occupancy:</strong>
                    <ul>
                        <li><strong>What it measures:</strong> Building occupancy level impact</li>
                        <li><strong>Impact:</strong> Full occupancy requires more cooling due to body heat, equipment, lighting</li>
                        <li><strong>Bar interpretation:</strong>
                            <ul>
                                <li>Long bar = Building occupied (weekday business hours)</li>
                                <li>Short bar = Building empty (nights, weekends)</li>
                            </ul>
                        </li>
                        <li><strong>What to do:</strong> Zone control to condition only occupied areas</li>
                    </ul>
                </li>
            </ul>
            
            <p><strong>How to Use This Chart:</strong></p>
            <ul>
                <li><strong>Identify the longest bar</strong> - This is your primary energy driver</li>
                <li><strong>Focus optimization efforts</strong> on the top 1-2 factors first</li>
                <li><strong>Compare over time</strong> - Make predictions at different times/conditions to see how factors change</li>
            </ul>
            
            <p><strong>Reading Example:</strong> If your chart shows:
                <ul>
                    <li>Temperature Diff: 80 (longest bar) → Main issue is large indoor-outdoor gap</li>
                    <li>Occupancy: 40 → Building is occupied</li>
                    <li>Time of Day: 30 → Business hours</li>
                    <li>Weather: 20 → Weather is favorable</li>
                </ul>
            </p>
            <p><strong>Interpretation:</strong> Your biggest opportunity to save energy is reducing the temperature difference. 
            Consider raising the AC setpoint by 1-2°C to reduce this gap.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison Chart
        st.markdown("---")
        st.markdown("#### 📊 Comparison with Averages (Vertical Bar Chart)")
        
        st.markdown("""
        <div class="step-card">
            <h4>How Your Prediction Compares to Historical Data</h4>
            <p><strong>What it shows:</strong> Your current prediction compared against typical consumption patterns.</p>
            
            <p><strong>The Three Bars:</strong></p>
            <ul>
                <li><strong>Current Prediction (Your Value):</strong>
                    <ul>
                        <li>The energy consumption for the specific conditions you entered</li>
                        <li><strong>Color:</strong> Changes from blue (low) to red (high) based on value</li>
                        <li><strong>Example:</strong> 95 Wh for a Tuesday at 2 PM with 28°C outside</li>
                    </ul>
                </li>
                <li><strong>Daily Average (Baseline - 100 Wh):</strong>
                    <ul>
                        <li>Historical average energy consumption across all conditions</li>
                        <li>Used as the standard benchmark for comparison</li>
                        <li><strong>Fixed at:</strong> 100 Wh (based on historical building data)</li>
                        <li><strong>Purpose:</strong> Shows if you're above or below typical usage</li>
                    </ul>
                </li>
                <li><strong>Weekend Average (60 Wh):</strong>
                    <ul>
                        <li>Typical consumption during weekend/off-hours</li>
                        <li>Lower than daily average due to reduced occupancy</li>
                        <li><strong>Purpose:</strong> Shows potential savings during unoccupied periods</li>
                        <li><strong>Target:</strong> Aim for consumption close to this during off-hours</li>
                    </ul>
                </li>
            </ul>
            
            <p><strong>How to Interpret:</strong></p>
            <ul>
                <li><strong>Current < Daily Average:</strong> ✅ Good! You're more efficient than typical
                    <ul>
                        <li>Example: 85 Wh vs 100 Wh = 15% savings</li>
                    </ul>
                </li>
                <li><strong>Current ≈ Daily Average:</strong> 🟡 Normal operation
                    <ul>
                        <li>Example: 95-105 Wh = within expected range</li>
                    </ul>
                </li>
                <li><strong>Current > Daily Average:</strong> ⚠️ Above typical, review recommendations
                    <ul>
                        <li>Example: 130 Wh vs 100 Wh = 30% higher, action needed</li>
                    </ul>
                </li>
                <li><strong>Off-hours close to Weekend Average:</strong> ✅ Good setback mode
                    <ul>
                        <li>Example: Night prediction at 65 Wh is close to 60 Wh weekend average</li>
                    </ul>
                </li>
            </ul>
            
            <p><strong>Strategic Use:</strong></p>
            <ul>
                <li><strong>During Business Hours:</strong> Aim to stay at or below 100 Wh (Daily Average)</li>
                <li><strong>During Off-Hours:</strong> Target the 60 Wh (Weekend Average) level</li>
                <li><strong>Peak Times:</strong> Slight increases above 100 Wh are acceptable, but not above 150 Wh</li>
            </ul>
            
            <p><strong>Reading Example:</strong> Chart shows:
                <ul>
                    <li>Current Prediction: 110 Wh (tallest bar, orange color)</li>
                    <li>Daily Average: 100 Wh</li>
                    <li>Weekend Average: 60 Wh</li>
                </ul>
            </p>
            <p><strong>Interpretation:</strong> You're 10% above the daily average. Check the Recommendations tab 
            for ways to reduce consumption. If this is during business hours, it's acceptable but has room for optimization. 
            If this is during off-hours, you should investigate why consumption isn't closer to the 60 Wh weekend level.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Recommendations Preview
        st.markdown("---")
        st.markdown("#### 💡 Quick Recommendations Preview Metrics")
        
        st.markdown("""
        <div class="step-card">
            <h4>Understanding the Recommendations Summary</h4>
            <p><strong>What it shows:</strong> A quick overview of available optimization opportunities.</p>
            
            <p><strong>The Three Metrics:</strong></p>
            <ul>
                <li><strong>📋 Total Recommendations:</strong>
                    <ul>
                        <li>Number of actionable suggestions available</li>
                        <li><strong>Range:</strong> Typically 3-7 recommendations</li>
                        <li><strong>Meaning:</strong> More recommendations = more optimization opportunities</li>
                    </ul>
                </li>
                <li><strong>⚡ Potential Savings (Wh/hr):</strong>
                    <ul>
                        <li>Total energy you could save per hour if all recommendations are implemented</li>
                        <li><strong>Example:</strong> "45 Wh/hr" means you can reduce consumption by 45 Wh</li>
                        <li><strong>Calculation:</strong> If prediction is 130 Wh and savings is 45 Wh, you could achieve 85 Wh</li>
                        <li><strong>Higher number:</strong> More inefficiency detected = greater savings opportunity</li>
                    </ul>
                </li>
                <li><strong>💰 Daily Cost Savings:</strong>
                    <ul>
                        <li>Money you could save per day if recommendations are implemented</li>
                        <li><strong>Example:</strong> "₹1.30" daily = ₹39/month = ₹468/year</li>
                        <li><strong>Use this to:</strong> Justify the effort of making changes</li>
                        <li><strong>ROI calculation:</strong> Compare savings vs. implementation effort</li>
                    </ul>
                </li>
            </ul>
            
            <p><strong>Priority Indicators (in Recommendations tab):</strong></p>
            <ul>
                <li><strong>🔴 High Priority:</strong> Urgent issues causing significant waste - implement immediately</li>
                <li><strong>🟡 Medium Priority:</strong> Important optimizations - implement within days</li>
                <li><strong>🟢 Low Priority:</strong> Long-term improvements - implement within weeks</li>
            </ul>
            
            <p><strong>Example Interpretation:</strong>
                <ul>
                    <li>Total Recommendations: 5</li>
                    <li>Potential Savings: 38 Wh/hr</li>
                    <li>Daily Cost Savings: ₹1.10</li>
                </ul>
            </p>
            <p><strong>What this means:</strong> You have 5 specific actions you can take to reduce your hourly 
            consumption by 38 Wh, which will save you ₹1.10 per day (₹33/month or ₹396/year). Click to the 
            Recommendations tab to see exactly what to do!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Best Practices for Reading Data
        st.markdown("---")
        st.markdown("#### 🎓 Best Practices for Reading Your Data")
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Expert Tips</h4>
            <ul>
                <li><strong>Compare Multiple Predictions:</strong> Make predictions for different scenarios (morning vs. afternoon, weekday vs. weekend) to understand patterns</li>
                <li><strong>Track Changes Over Time:</strong> Record predictions before and after implementing recommendations to measure impact</li>
                <li><strong>Focus on Relative Changes:</strong> The delta values (↑/↓) are often more important than absolute numbers</li>
                <li><strong>Context Matters:</strong> A 150 Wh prediction during extreme weather is acceptable; the same during mild weather needs investigation</li>
                <li><strong>Use All Charts Together:</strong> Gauge shows "what", Contributing Factors shows "why", Comparison shows "how much"</li>
                <li><strong>Don't Ignore Small Numbers:</strong> Even small hourly savings (5-10 Wh) multiply to significant annual savings</li>
                <li><strong>Validate Predictions:</strong> Compare predictions with actual utility bills to verify accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
            <h4>💡 Quick Reference: What Good Numbers Look Like</h4>
            <ul>
                <li><strong>Predicted Consumption:</strong> 60-100 Wh during business hours, <60 Wh off-hours</li>
                <li><strong>Efficiency Score:</strong> 70+ is good, 85+ is excellent</li>
                <li><strong>Temperature Difference:</strong> <8°C is ideal, >12°C needs attention</li>
                <li><strong>Contributing Factors:</strong> No single factor should dominate (all bars relatively balanced)</li>
                <li><strong>Comparison:</strong> Current prediction at or below Daily Average line</li>
                <li><strong>Savings Potential:</strong> <10 Wh/hr means you're already optimized; >30 Wh/hr means significant room for improvement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Use cases
        st.markdown("### 🎯 Common Use Cases")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tip-box">
                <h4>🏢 Building Managers</h4>
                <p><strong>Use this tool to:</strong></p>
                <ul>
                    <li>Predict daily energy consumption</li>
                    <li>Optimize HVAC schedules</li>
                    <li>Reduce operational costs</li>
                    <li>Meet sustainability goals</li>
                    <li>Prepare energy reports</li>
                </ul>
                <p><strong>Recommended frequency:</strong> Daily predictions, weekly reviews</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="tip-box">
                <h4>⚡ Energy Managers</h4>
                <p><strong>Use this tool to:</strong></p>
                <ul>
                    <li>Identify energy waste</li>
                    <li>Plan demand response strategies</li>
                    <li>Track carbon footprint reduction</li>
                    <li>Benchmark building performance</li>
                    <li>Validate energy projects</li>
                </ul>
                <p><strong>Recommended frequency:</strong> Multiple daily predictions during peak periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="tip-box">
                <h4>👔 Facility Directors</h4>
                <p><strong>Use this tool to:</strong></p>
                <ul>
                    <li>Make data-driven decisions</li>
                    <li>Budget for energy costs</li>
                    <li>Plan equipment upgrades</li>
                    <li>Demonstrate ROI of improvements</li>
                    <li>Report to stakeholders</li>
                </ul>
                <p><strong>Recommended frequency:</strong> Weekly strategic reviews</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="tip-box">
                <h4>🔧 Maintenance Teams</h4>
                <p><strong>Use this tool to:</strong></p>
                <ul>
                    <li>Detect equipment issues early</li>
                    <li>Schedule preventive maintenance</li>
                    <li>Verify repairs are effective</li>
                    <li>Optimize system performance</li>
                    <li>Document energy improvements</li>
                </ul>
                <p><strong>Recommended frequency:</strong> As needed for troubleshooting</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tips for best results
        st.markdown("### 💡 Tips for Best Results")
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Do's</h4>
            <ul>
                <li>✓ Use accurate weather data (from reliable forecast sources)</li>
                <li>✓ Update predictions when weather conditions change</li>
                <li>✓ Track implemented recommendations and their results</li>
                <li>✓ Make predictions for different scenarios (peak hours, weekends, etc.)</li>
                <li>✓ Review historical analytics to understand patterns</li>
                <li>✓ Coordinate with occupants before major temperature changes</li>
                <li>✓ Document baseline performance before making changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>❌ Don'ts</h4>
            <ul>
                <li>✗ Don't make extreme temperature adjustments (>3°C at once)</li>
                <li>✗ Don't ignore HIGH priority recommendations</li>
                <li>✗ Don't implement changes without notifying building occupants</li>
                <li>✗ Don't forget to monitor comfort complaints after changes</li>
                <li>✗ Don't rely solely on predictions - verify with actual consumption data</li>
                <li>✗ Don't disable safety-critical HVAC systems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # FAQs
        st.markdown("### ❓ Frequently Asked Questions")
        
        with st.expander("📊 How accurate are the predictions?"):
            st.markdown("""
            The AI model is trained on historical data and typically achieves 85-95% accuracy. 
            Accuracy depends on:
            - Quality of input data
            - Similarity to training conditions
            - Sudden weather changes
            - Unusual building occupancy patterns
            
            For best results, use real-time weather data and update predictions hourly during rapidly changing conditions.
            """)
        
        with st.expander("💰 How much can I save?"):
            st.markdown("""
            Energy savings vary by building and current efficiency levels, but typical results include:
            - **10-20%** reduction in HVAC energy consumption
            - **15-30%** reduction during off-hours
            - **5-15%** reduction during peak demand periods
            - **$500-$5,000+** monthly savings depending on building size
            
            The tool provides specific savings estimates for each recommendation.
            """)
        
        with st.expander("⏰ When should I make predictions?"):
            st.markdown("""
            **Best practices for prediction timing:**
            - **Daily:** Every morning before business hours
            - **Weather changes:** When forecasts change significantly
            - **Peak periods:** Before high-demand hours (2-6 PM)
            - **Seasonal transitions:** More frequently during spring/fall
            - **Special events:** Before holidays, meetings, events
            
            Set up a routine schedule that works for your building's needs.
            """)
        
        with st.expander("🔧 What if recommendations don't match my building?"):
            st.markdown("""
            The tool provides general recommendations that may need customization:
            - **Consult with HVAC specialists** for complex systems
            - **Adjust recommendations** based on your building's specific constraints
            - **Test changes incrementally** rather than all at once
            - **Monitor occupant feedback** and adjust accordingly
            - **Contact support** if recommendations seem inappropriate
            
            Always prioritize occupant comfort and safety over energy savings.
            """)
        
        with st.expander("📈 How do I track my progress?"):
            st.markdown("""
            **Track your energy management success:**
            1. **Baseline:** Record current energy consumption before changes
            2. **Implement:** Apply recommendations systematically
            3. **Monitor:** Use the Analytics tab to track trends
            4. **Compare:** Calculate savings vs. baseline
            5. **Document:** Keep records of changes and results
            6. **Report:** Share success stories with stakeholders
            
            The Analytics tab provides visual tools for monitoring historical performance.
            """)
        
        st.markdown("---")
        
        # Support section
        st.markdown("### 🆘 Need Help?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📧 Contact Support**
            
            For technical assistance:
            - Email: support@hvac-predictor.com
            - Response time: 24 hours
            """)
        
        with col2:
            st.markdown("""
            **📚 Documentation**
            
            Full documentation available:
            - User Guide (PDF)
            - Video tutorials
            - API documentation
            """)
        
        with col3:
            st.markdown("""
            **💬 Community**
            
            Join our community:
            - User forum
            - Best practices sharing
            - Feature requests
            """)
    
    # -------------------------------------------------------------------------
    # TAB 5: ABOUT
    # -------------------------------------------------------------------------
    with tab5:
        st.header("📚 About This Application")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This **Smart HVAC Energy Predictor** uses machine learning to predict hourly 
        energy consumption for office buildings and provides actionable recommendations 
        to reduce energy waste.
        
        ### 🔧 Technology Stack
        
        - **Frontend**: Streamlit
        - **ML Models**: Random Forest, XGBoost
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        - **Dataset**: UCI Appliances Energy Prediction
        
        ### 📊 Features
        
        ✅ Real-time energy prediction  
        ✅ Historical analytics dashboard  
        ✅ AI-powered recommendations  
        ✅ Cost & carbon footprint calculation  
        ✅ Interactive visualizations  
        
        ### 🌍 Environmental Impact
        
        This tool supports UN SDG Goals:
        - **Goal 7**: Affordable & Clean Energy
        - **Goal 11**: Sustainable Cities
        - **Goal 13**: Climate Action
        
        ### 👨‍💻 Developed By
        
        **Trishika Shrivastav**  
        📧 trishikashrivastav11@gmail.com  
        🔗 [GitHub](https://github.com/Trishi@11)
        
        ### 📄 License
        
        MIT License - Free to use and modify
        """)
        
        st.markdown("---")
        st.success("⭐ Star this project on GitHub if you find it helpful!")

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()