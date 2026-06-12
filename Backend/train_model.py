import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

import os

# Find the exact folder where train_model.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "mouse_data.csv")

# Read the CSV using the secure path
df = pd.read_csv(csv_path)

X = df[["avg_speed", "max_speed", "total_distance", "points", "avg_time_gap"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

joblib.dump(model, "model.pkl")
print("Saved model.pkl")