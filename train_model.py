import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load data
data_2019 = pd.read_csv("data/Jan_2019_ontime.csv").sample(n=50000, random_state=42)
data_2020 = pd.read_csv("data/Jan_2020_ontime.csv").sample(n=50000, random_state=42)
data = pd.concat([data_2019, data_2020], ignore_index=True)

# Drop nulls
data.dropna(subset=["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME", "DEP_DEL15"], inplace=True)

# Target column
data["DELAYED"] = data["DEP_DEL15"]

# Features
features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME"]
X = data[features].copy()
y = data["DELAYED"]

# Encode features
encoders = {}
for col in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col])

# Time formatting
X["DEP_TIME"] = X["DEP_TIME"].apply(lambda x: int(str(x).zfill(4)[:2]))

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, model.predict(X_test)))

# Save model & encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/flight_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
