import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import requests

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Crop Dashboard", layout="wide")

# -------------------- DARK MODE CSS --------------------
st.markdown("""
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

body {
    background-color: #0e1117;
    color: white;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("model/agroman.pkl")

model = load_model()

# -------------------- HEADER --------------------
st.title("🌾 Smart Crop Recommendation Dashboard")
st.caption("AI-powered crop suggestions with real-time weather integration")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌱 Input Parameters")

# Weather input
city = st.sidebar.text_input("📍 Enter City", "Delhi")
get_weather = st.sidebar.button("🌦️ Get Weather Data")

# Default values
temperature = 25.0
humidity = 60.0

# Weather API
if get_weather:
    API_KEY = "637390b0f846f725606acbb0873f7ef3"  # ← PUT YOUR KEY HERE
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        st.sidebar.success(f"🌡️ Temp: {temperature}°C")
        st.sidebar.success(f"💧 Humidity: {humidity}%")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.write(response.text)

# Sliders
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(temperature))
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(humidity))
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

predict_btn = st.sidebar.button("🌾 Predict")

# -------------------- DATA --------------------
crop_images = {
    "rice": "images/rice.jfif",
    "wheat": "images/wheat.jfif",
    "maize": "images/maize.jfif",
    "cotton": "images/cotton.jfif",
    "sugarcane": "images/sugarcane.jfif",
    "papaya": "images/papaya.jfif"
}

crop_info = {
    "rice": "🌾 Requires high rainfall and humidity.",
    "wheat": "🌱 Grows in moderate temperature.",
    "maize": "🌽 Needs warm climate and fertile soil.",
    "cotton": "🧵 Requires hot climate and low humidity.",
    "sugarcane": "🍬 Needs high water supply.",
    "papaya": "🥭 Tropical fruit, grows in warm regions."
}

# -------------------- PREDICTION --------------------
if predict_btn:

    if ph <= 0 or ph > 14:
        st.error("❌ pH must be between 0 and 14")
        st.stop()

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    with st.spinner("🔍 Analyzing soil conditions..."):
        prediction = model.predict(input_data)
        crop = prediction[0]

    # -------------------- RESULT CARD --------------------
    st.markdown(f"""
    <div class="card">
        <h2>🌾 Recommended Crop</h2>
        <h1 style="color:#4CAF50;">{crop.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- CONFIDENCE + INPUT --------------------
    try:
        probabilities = model.predict_proba(input_data)
        confidence = np.max(probabilities) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Confidence")
            st.progress(int(confidence))
            st.write(f"{confidence:.2f}%")

            if confidence > 80:
                st.success("High confidence prediction ✅")
            elif confidence > 60:
                st.warning("Moderate confidence ⚠️")
            else:
                st.error("Low confidence ❌ Try adjusting inputs")

        with col2:
            st.markdown("### 📋 Input Summary")
            st.write(input_data)

    except:
        st.warning("Confidence not available")

    # -------------------- IMAGE + INFO --------------------
    col1, col2 = st.columns(2)

    with col1:
        if crop.lower() in crop_images:
            st.image(crop_images[crop.lower()], use_container_width=True)

    with col2:
        st.markdown("### ℹ️ Crop Info")
        st.write(crop_info.get(crop.lower(), "No info available"))

    # -------------------- COMPARISON (ANIMATED) --------------------
    avg_values = [50, 50, 50, 25, 60, 6.5, 100]

    compare_df = pd.DataFrame({
        "Feature": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall],
        "Average": avg_values
    })

    fig = px.bar(
        compare_df,
        x="Feature",
        y=["Your Input", "Average"],
        barmode="group",
        title="📊 Input vs Average Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------- FEATURE IMPORTANCE --------------------
    try:
        rf_model = model.named_steps["model"]
        importances = rf_model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="📈 Feature Importance"
        )

        st.plotly_chart(fig2, use_container_width=True)

    except:
        st.warning("Feature importance not available")