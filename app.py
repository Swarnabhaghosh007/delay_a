app_code = """
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

# Title and Description
st.title("░▒▓ Delivery Delay Predictor ▓▒░")
st.write("Enter the shipment details below to predict the likelihood of a delivery delay.")

# Load the saved model safely
@st.cache_resource
def load_model():
    return joblib.load('logi.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model. Make sure 'logi.sav' exists. Details: {e}")
    st.stop()

# Define feature columns matching the original model training order
features_list = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Interactive User Inputs in columns
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)
    traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
    weather_condition = st.slider("Weather Condition Score (1-5)", min_value=1, max_value=5, value=2)
    delivery_slot = st.slider("Delivery Slot (Morning/Afternoon/Night Shift ID)", min_value=1, max_value=4, value=2)
    driver_experience = st.number_input("Driver Experience (years)", min_value=0, max_value=50, value=5, step=1)
    num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=2, step=1)

with col2:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=3, step=1)
    road_condition = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
    package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=100.0, value=15.0, step=0.1)
    warehouse_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, max_value=500, value=30, step=1)

# Predict Button
if st.button("Predict Delay", type="primary"):
    # Gather inputs into a dataframe
    input_data = pd.DataFrame([[
        delivery_distance, traffic_congestion, weather_condition,
        delivery_slot, driver_experience, num_stops, vehicle_age,
        road_condition, package_weight, fuel_efficiency, warehouse_time
    ]], columns=features_list)
    
    # Predict class and probability
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("--- ")
    if prediction == 1:
        st.error(f"☑️ **Prediction: Delayed** (Confidence: {probabilities[1]:.2%})")
    else:
        st.success(f"✅ **Prediction: On Time** (Confidence: {probabilities[0]:.2%})")
        
    # Display exact probabilities
    st.write(f"*Probability of On Time: {probabilities[0]:.2%}* | *Probability of Delay: {probabilities[1]:.2%}*")
"""

# Write content to app.py
with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app.py successfully written to disk!")
