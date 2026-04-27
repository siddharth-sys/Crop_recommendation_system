import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import requests
import folium
from streamlit_folium import st_folium

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Smart Crop Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ FORCE SIDEBAR RENDER
st.sidebar.write("👈 Sidebar loaded")

# -------------------- DARK UI --------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Sidebar FIX */
section[data-testid="stSidebar"] {
    background-color: #1c1f26 !important;
    color: white !important;
}

/* Ensure sidebar content visible */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Card UI */
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
st.caption("AI-powered crop suggestions with weather + soil data")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌱 Input Panel")
st.sidebar.write("Adjust parameters below")

# -------------------- LOCATION --------------------
st.sidebar.markdown("### 📍 Location")

latitude = st.sidebar.number_input("Latitude", value=26.9)
longitude = st.sidebar.number_input("Longitude", value=75.8)

# -------------------- WEATHER --------------------
temperature = 25.0
humidity = 60.0

API_KEY = "5a83872e23f74e1181ff839dd521af55"  # 🔴 Replace this

if API_KEY != "5a83872e23f74e1181ff839dd521af55":
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get("cod") == 200:
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]

            st.sidebar.success(f"🌡️ Temp: {temperature}°C")
            st.sidebar.success(f"💧 Humidity: {humidity}%")
        else:
            st.sidebar.warning("Weather API issue")

    except:
        st.sidebar.warning("Weather fetch failed")

# -------------------- INPUT SLIDERS --------------------
st.sidebar.markdown("### 🧪 Soil Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(temperature))
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(humidity))
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

predict_btn = st.sidebar.button("🌾 Predict Crop")

# -------------------- MAP --------------------
st.markdown("### 🗺️ Location Map")

m = folium.Map(location=[latitude, longitude], zoom_start=8)
folium.Marker([latitude, longitude], tooltip="Selected Location").add_to(m)

st_folium(m, width=700)

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
    "papaya": "🥭 Tropical crop for warm regions."
}

# -------------------- PREDICTION --------------------
if predict_btn:

    if ph <= 0 or ph > 14:
        st.error("❌ Invalid pH value")
        st.stop()

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    with st.spinner("🔍 Analyzing..."):
        prediction = model.predict(input_data)
        crop = prediction[0]

    # RESULT
    st.markdown(f"""
    <div class="card">
        <h2>🌾 Recommended Crop</h2>
        <h1 style="color:#4CAF50;">{crop.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    # CONFIDENCE
    try:
        prob = model.predict_proba(input_data)
        confidence = np.max(prob) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Confidence")
            st.progress(int(confidence))
            st.write(f"{confidence:.2f}%")

        with col2:
            st.subheader("📋 Input Summary")
            st.write(input_data)

    except:
        st.warning("No confidence score available")

    # IMAGE + INFO
    col1, col2 = st.columns(2)

    with col1:
        if crop.lower() in crop_images:
            st.image(crop_images[crop.lower()], use_container_width=True)

    with col2:
        st.subheader("ℹ️ Info")
        st.write(crop_info.get(crop.lower(), "No info available"))

    # CHART
    avg_values = [50, 50, 50, 25, 60, 6.5, 100]

    df_compare = pd.DataFrame({
        "Feature": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall],
        "Average": avg_values
    })

    fig = px.bar(
        df_compare,
        x="Feature",
        y=["Your Input", "Average"],
        barmode="group",
        title="📊 Input vs Average"
    )

    st.plotly_chart(fig, use_container_width=True)

    # FEATURE IMPORTANCE
    try:
        rf = model.named_steps["model"]
        importance = rf.feature_importances_

        df_feat = pd.DataFrame({
            "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(df_feat, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    except:
        st.warning("Feature importance unavailable")