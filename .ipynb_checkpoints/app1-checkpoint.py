import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Load model
model = joblib.load("model/agroman.pkl")

# Title
st.title("🌾 Smart Crop Recommendation System")
st.markdown("### 🌱 Enter soil and environmental conditions")

st.divider()

# Input Layout
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 140, 50)
    P = st.number_input("Phosphorus (P)", 0, 140, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

st.divider()

# Button
predict_btn = st.button("🌱 Predict Best Crop")

# Prediction Logic
if predict_btn:

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    prediction = model.predict(input_data)
    crop = prediction[0]

    # 🌾 Local Image Paths
    crop_images = {
        "rice": "C:/Users/Shree/project/images/rice.jfif",
        "wheat": "C:/Users/Shree/project/images/wheat.jfif",
        "maize": "C:/Users/Shree/project/images/maize.jfif",
        "cotton": "C:/Users/Shree/project/images/cotton.jfif",
        "sugarcane": "C:/Users/Shree/project/images/sugracane.jfif",
        "papaya": "C:/Users/Shree/project/images/papaya.jfif"
    }

    # 🌾 Crop Info
    crop_info = {
        "rice": "🌾 Requires high rainfall and humidity.",
        "wheat": "🌱 Grows in moderate temperature.",
        "maize": "🌽 Needs warm climate and fertile soil.",
        "cotton": "🧵 Requires hot climate and low humidity.",
        "sugarcane": "🍬 Needs high water supply.",
        "papaya": "🥭 Tropical fruit, grows in warm regions."
    }

    # Output
    st.success(f"🌾 Recommended Crop: {crop}")

    # Confidence Score
    try:
        probabilities = model.predict_proba(input_data)
        confidence = np.max(probabilities) * 100
        st.info(f"📊 Confidence: {confidence:.2f}%")
    except:
        st.warning("⚠️ Confidence score not available")

    # Show Image
    if crop.lower() in crop_images:
        try:
            st.image(crop_images[crop.lower()], caption=crop.capitalize(), width=500)
        except:
            st.error("❌ Image not found. Check file path.")

    # Show Info
    if crop.lower() in crop_info:
        st.markdown(f"### ℹ️ Crop Info\n{crop_info[crop.lower()]}")
    else:
        st.markdown("### ℹ️ No additional info available")