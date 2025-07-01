import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Encode semua kolom kategorikal (kecuali target klasifikasi)
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'NObeyesdad':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Fitur dan target
X = df.drop(columns=['Weight', 'NObeyesdad'])
y = df['Weight']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model dan tools
joblib.dump(model, "model_rf_scaled.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model Random Forest berhasil dilatih dan disimpan.")
