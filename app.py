import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("model_rf_selected.pkl")
scaler = joblib.load("scaler_selected.pkl")

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Prediksi Berat Badan", page_icon="🏋️", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("Prediksi Berat Badan dengan Data Mining")
    st.write("""
        Aplikasi ini memprediksi berat badan berdasarkan gaya hidup Anda menggunakan model machine learning.
        Hasil bersifat estimasi dan bukan diagnosis medis.
    """)

    # Input pengguna
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Umur", min_value=1, max_value=120, value=25)

    height_cm = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
    height = height_cm / 100
    st.caption(f"Tinggi dalam meter: {height:.2f} m")

    family_history = st.selectbox("Riwayat Obesitas dalam Keluarga", ["yes", "no"])
    favc = st.selectbox("Sering Makan Berlebih?", ["yes", "no"])
    caec = st.selectbox("Sering Ngemil?", ["no", "Sometimes", "Frequently", "Always"])

    # Dibalik: makin sering makan, makin kurus
    ncp_raw = st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3)
    ncp_transformed = 5 - ncp_raw

    # Aktivitas fisik tetap (tidak dibalik), tapi tetap dibatasi 0.3–2.8
    faf_raw = st.slider("Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0, step=0.1)
    faf_clamped = max(0.3, min(faf_raw, 2.8))
    faf_transformed = 2.9 - faf_clamped  # logika asli: makin aktif → makin kurus


    # Dibalik & dijaga range: makin banyak waktu di layar, makin kurus
    tue_raw = st.slider("Waktu di Depan Layar (0–2 jam)", 0.0, 2.0, 1.0, step=0.1)
    tue_clamped = max(0.3, min(tue_raw, 1.9))
    tue_transformed = 1.9 - tue_clamped

    # Konsumsi alkohol → dibalik
    calc_input = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    calc_mapping = {"no": 0, "Sometimes": 3, "Frequently": 2, "Always": 1}
    calc_transformed = 3 - calc_mapping[calc_input]

    # Mapping input ke format model
    input_dict = {
        'Height': height,
        'family_history_with_overweight': 1 if family_history == "yes" else 0,
        'FAVC': 1 if favc == "yes" else 0,
        'CAEC': {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'NCP': ncp_transformed,
        'FAF': faf_transformed,
        'TUE': tue_transformed,
        'CALC': calc_transformed
    }

    # Ubah ke DataFrame dan scaling
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Prediksi
    if st.button("Prediksi Berat Badan"):
        pred = model.predict(input_scaled)[0]
        st.success(f"💪 Berat badan diprediksi: {pred:.2f} kg")
        st.caption("\u26a0\ufe0f Hasil ini hanya estimasi berdasarkan pola data, bukan diagnosis medis.")
