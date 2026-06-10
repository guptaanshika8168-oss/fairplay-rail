from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mouse_data = data['mouseData']
    
    # Calculate speed
    speeds = []
    for i in range(1, len(mouse_data)):
        prev = mouse_data[i-1]
        current = mouse_data[i]
        distance = ((current['x'] - prev['x'])**2 + (current['y'] - prev['y'])**2)**0.5
        time_diff = current['time'] - prev['time']
        if time_diff > 0:
            speeds.append(distance / time_diff)
    
    average_speed = sum(speeds) / len(speeds)
    
    if average_speed > 400:
        return jsonify({'prediction': 'BOT', 'confidence': 95})
    else:
        return jsonify({'prediction': 'HUMAN', 'confidence': 92})

if __name__ == '__main__':
    print("🚀 Backend running...")
    app.run(port=5000)