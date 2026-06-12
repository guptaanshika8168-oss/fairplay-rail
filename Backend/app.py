from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import math

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

def extract_features(mouse_data):
    if len(mouse_data) < 2:
        return None

    speeds = []
    total_distance = 0
    time_gaps = []

    for i in range(1, len(mouse_data)):
        prev = mouse_data[i - 1]
        curr = mouse_data[i]
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        dt = curr["time"] - prev["time"]
        total_distance += dist
        if dt > 0:
            speeds.append(dist / dt)
            time_gaps.append(dt)

    if not speeds:
        return None

    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    points = len(mouse_data)
    avg_time_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0

    return [avg_speed, max_speed, total_distance, points, avg_time_gap]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mouse_data = data.get("mouseData", [])

    features = extract_features(mouse_data)
    
    # If features is None, it means a touchscreen tap or instant click occurred
    if features is None:
        return jsonify({
            "prediction": "BOT",
            "confidence": 99,
            "average_speed": 0
        })

    # If we do have valid data, let the machine learning model decide
    pred = model.predict([features])[0]
    
    return jsonify({
        "prediction": "BOT" if pred == 1 else "HUMAN",
        "confidence": 95,
        "average_speed": round(features[0], 2)
    })
   

if __name__ == "__main__":
    app.run(port=5000, debug=True)