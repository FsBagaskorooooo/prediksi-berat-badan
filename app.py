import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("model_scaled.pkl")
scaler = joblib.load("scaler.pkl")

# Judul aplikasi
st.title("Prediksi Berat Badan dengan Data Mining")
st.write("Aplikasi ini menggunakan semua fitur dari dataset gaya hidup untuk memprediksi berat badan.")

# Input fitur
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Umur", min_value=1, max_value=120, value=25)

height_cm = st.number_input("Tinggi badan (cm)", min_value=100, max_value=250, value=170)
height = height_cm / 100
st.caption(f"Tinggi dalam meter: {height:.2f} m")

family_history = st.selectbox("Riwayat Obesitas dalam Keluarga", ["yes", "no"])
favc = st.selectbox("Sering Makan Berlebih?", ["yes", "no"])
fcvc = st.slider("Konsumsi Sayur Harian (1-3)", 1, 3, 2)
ncp = st.slider("Jumlah Makan per Hari (1-4)", 1, 4, 3)
caec = st.selectbox("Konsumsi Camilan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok", ["yes", "no"])
ch2o = st.slider("Air Putih per Hari (liter)", 1.0, 3.0, 2.0)
scc = st.selectbox("Punya Masalah Kesehatan?", ["yes", "no"])
faf = st.slider("Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu di Depan Layar (0–2)", 0.0, 2.0, 1.0)
calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi Harian", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Mapping label encoding (harus sesuai training model)
map_yesno = {"yes": 1, "no": 0}
map_gender = {"Male": 1, "Female": 0}
map_caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
map_calc = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
map_mtrans = {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4
}

# Buat array input
# Dictionary sesuai urutan training model
input_dict = {
    'Gender': 1 if gender == "Male" else 0,
    'Age': age,
    'Height': height,
    'family_history_with_overweight': 1 if family_history == "yes" else 0,
    'FAVC': 1 if favc == "yes" else 0,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
    'SMOKE': 1 if smoke == "yes" else 0,
    'CH2O': ch2o,
    'SCC': 1 if scc == "yes" else 0,
    'FAF': faf,
    'TUE': tue,
    'CALC': {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc],
    'MTRANS': {
        "Automobile": 0,
        "Bike": 1,
        "Motorbike": 2,
        "Public_Transportation": 3,
        "Walking": 4
    }[mtrans]
}

# Buat DataFrame 1 baris
input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# Prediksi berat badan
if st.button("Prediksi Berat Badan"):
    pred = model.predict(input_scaled)[0]  # ✅ BENAR: prediksi pakai data yang sudah diskalakan
    st.success(f"Berat badan diprediksi: {pred:.2f} kg")


