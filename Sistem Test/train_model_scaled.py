import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Baca dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Label encoding fitur kategorikal
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'NObeyesdad':  # exclude label klasifikasi
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Fitur dan target
X = df.drop(columns=['Weight', 'NObeyesdad'])  # 15 fitur
y = df['Weight']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split & training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan model dan scaler
joblib.dump(model, "model_scaled.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model, scaler, dan encoder berhasil disimpan!")
