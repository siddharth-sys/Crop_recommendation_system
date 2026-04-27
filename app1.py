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

st.sidebar.write("👈 Sidebar loaded")

# -------------------- DARK UI --------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background-color: #0e1117;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #1c1f26 !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
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
st.caption("AI-powered crop suggestions with weather + soil data")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌱 Input Panel")

# -------------------- LOCATION --------------------
st.sidebar.markdown("### 📍 Location")
latitude = st.sidebar.number_input("Latitude", value=26.9)
longitude = st.sidebar.number_input("Longitude", value=75.8)

# -------------------- WEATHER --------------------
temperature = 25.0
humidity = 60.0

API_KEY = st.secrets["API_KEY"]

try:
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if data.get("cod") == 200:
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        st.sidebar.metric("🌡️ Temperature", f"{temperature} °C")
        st.sidebar.metric("💧 Humidity", f"{humidity} %")
except:
    st.sidebar.warning("Weather fetch failed")

# -------------------- INPUTS --------------------
st.sidebar.markdown("### 🧪 Soil Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(temperature))
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(humidity))
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

predict_btn = st.sidebar.button("🌾 Predict Crop")

if st.sidebar.button("🔄 Reset"):
    st.rerun()

# -------------------- MAP --------------------
st.markdown("### 🗺️ Location Map")
m = folium.Map(location=[latitude, longitude], zoom_start=8)
folium.Marker([latitude, longitude]).add_to(m)
st_folium(m, width=700)

# -------------------- FERTILIZER FUNCTION --------------------
def recommend_fertilizer(crop, N, P, K):

    crop_requirements = {
        "rice": {"N": 80, "P": 40, "K": 40},
        "wheat": {"N": 60, "P": 40, "K": 40},
        "maize": {"N": 70, "P": 50, "K": 50},
        "cotton": {"N": 80, "P": 40, "K": 60},
        "sugarcane": {"N": 100, "P": 50, "K": 50},
        "papaya": {"N": 60, "P": 50, "K": 60}
    }

    organic = {
        "N": "🌿 Compost / Vermicompost / Green manure",
        "P": "🌿 Bone meal / Rock phosphate",
        "K": "🌿 Wood ash / Banana peel compost"
    }

    chemical = {
        "N": "⚗️ Urea",
        "P": "⚗️ DAP",
        "K": "⚗️ MOP"
    }

    result = []
    crop = crop.lower()

    if crop in crop_requirements:
        req = crop_requirements[crop]

        if N < req["N"]:
            result.append(("Nitrogen", organic["N"], chemical["N"]))
        if P < req["P"]:
            result.append(("Phosphorus", organic["P"], chemical["P"]))
        if K < req["K"]:
            result.append(("Potassium", organic["K"], chemical["K"]))

        if not result:
            return ["✅ Nutrients sufficient"]

    else:
        return ["No data available"]

    return result

# -------------------- DATA --------------------
crop_images = {
    "rice": "images/rice.jfif",
    "wheat": "images/wheat.jfif",
    "maize": "images/maize.jfif",
    "cotton": "images/cotton.jfif",
    "sugarcane": "images/sugarcane.jfif",
    "papaya": "images/papaya.jfif"
}

# -------------------- PREDICTION --------------------
if predict_btn:

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N","P","K","temperature","humidity","ph","rainfall"]
    )

    prediction = model.predict(input_data)
    crop = prediction[0]

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_
    top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    # RESULT
    st.markdown(f"""
    <div class="card">
        <h2>🌾 Recommended Crop</h2>
        <h1 style="color:#4CAF50;">{crop.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    # TOP 3
    st.subheader("🌾 Top 3 Suggestions")
    for c, p in top3:
        st.write(f"{c} → {p*100:.2f}%")

    # FERTILIZER
    st.subheader("🌱 Fertilizer Plan for " + crop.upper())

    ferts = recommend_fertilizer(crop, N, P, K)

    if isinstance(ferts[0], str):
        st.success(ferts[0])
    else:
        for f in ferts:
            st.markdown(f"### {f[0]} Deficiency")
            st.markdown(f"**Organic:** {f[1]}")
            st.markdown(f"**Chemical:** {f[2]}")

    # CONFIDENCE
    st.subheader("📊 Confidence")
    st.progress(int(np.max(probs)*100))

    # CHART
    df = pd.DataFrame({
        "Feature":["N","P","K","Temp","Humidity","pH","Rainfall"],
        "Value":[N,P,K,temperature,humidity,ph,rainfall]
    })

    fig = px.bar(df, x="Feature", y="Value")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MODEL INFO --------------------
st.markdown("### 🧠 Model Info")
st.write("Random Forest | Soil + Weather based prediction")