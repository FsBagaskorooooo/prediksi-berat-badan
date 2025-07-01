import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Baca file dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")  # ganti sesuai nama file

# Encode semua kolom kategorikal kecuali target klasifikasi
for col in df.select_dtypes(include='object').columns:
    if col != 'NObeyesdad':
        df[col] = LabelEncoder().fit_transform(df[col])

# Buang kolom target klasifikasi
X = df.drop(columns=['Weight', 'NObeyesdad'])
y = df['Weight']

# Split dan latih
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model_weight_15_features.pkl")

print("✅ Model dilatih ulang & disimpan sebagai model_weight_15_features.pkl")
