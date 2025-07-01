import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Label encode fitur kategorikal
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != "NObeyesdad":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Fitur terpilih
selected_features = [
    "Height",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "Age",
    "Gender",
    "NCP",    # akan dibalik nilainya di app.py
    "FAF",
    "TUE",    # dibalik juga di app.py
    "CALC"    # juga dibalik nilainya di app.py
]

X = df[selected_features]
y = df["Weight"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split dan training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Simpan model dan scaler
joblib.dump(model, "model_rf_selected.pkl")
joblib.dump(scaler, "scaler_selected.pkl")

print("✅ Model dan scaler berhasil dilatih dan disimpan ulang.")
