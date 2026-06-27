import streamlit as st
import numpy as np
import joblib

model = joblib.load('taxi_model.pkl')

st.title("🚕 NYC Taxi Duration Predictor")
st.write("Enter your trip details")

distance = st.slider("Distance (km)", 0.0, 35.0, 5.0)
hour = st.slider("Pickup Hour", 0, 23, 17)

day_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
}
day_name = st.selectbox("Day of Week", list(day_map.keys()))
day = day_map[day_name]

if st.button("Predict Trip Duration"):
    prediction = model.predict([[0, hour, 0, day, distance]])
    actual_minutes = np.expm1(prediction[0])
    st.success(f"🕐 Estimated Trip Duration: {actual_minutes:.1f} minutes")