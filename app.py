# app.py

import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 California House Price Predictor")
st.write("Enter details below to predict the median house value.")

# Input fields for all model features
MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age (years)", min_value=1.0, max_value=100.0, value=20.0)
AveRooms = st.number_input("Average Number of Rooms", min_value=1.0, max_value=15.0, value=5.0)
AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.5, max_value=5.0, value=1.0)
Population = st.number_input("Population in Block", min_value=1.0, max_value=5000.0, value=1000.0)
AveOccup = st.number_input("Average Occupants per Household", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-118.0)

# Predict button
if st.button("Predict"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)[0]
    st.success(f"🏠 Predicted Median House Value: **${prediction * 100000:.2f}**")
