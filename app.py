import pandas as pd
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Configuration ---
DATA_FILE = 'thermal_data.csv'
MODEL_FILE = 'thermal_model.joblib'
ZONES = [
    "Main Parking Lot", "Academic Block A", "Academic Block B",
    "Boys Hostel 1", "Boys Hostel 2", "Girls Hostel",
    "Sports Stadium", "Central Library", "Green Quad", "Food Court"
]

# --- Load Model and Data ---
print("Loading model pipeline...")
try:
    pipeline = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    pipeline = None

print("Loading dashboard data...")
try:
    dashboard_df = pd.read_csv(DATA_FILE)
    print("Dashboard data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    dashboard_df = pd.DataFrame()

# --- Status Helper ---
def get_status(temp):
    if temp > 40:
        return "Hotspot"
    elif temp > 36:
        return "Medium"
    else:
        return "Safe"

# --- Create Flask App ---
app = Flask(__name__)
CORS(app)

# --- API: Dashboard Data ---
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    if dashboard_df.empty:
        return jsonify({"error": "Data not loaded on server"}), 500
        
    latest_data = dashboard_df.sort_values('timestamp').groupby('zone').last().reset_index()
    latest_data['id'] = latest_data.index
    latest_data['status'] = latest_data['temp'].apply(get_status)
    latest_data = latest_data[['id', 'zone', 'temp', 'uv', 'status']]
    
    return jsonify(latest_data.to_dict(orient='records'))

# --- API: Forecast ---
@app.route('/api/forecast', methods=['POST'])
def run_forecast():
    if pipeline is None:
        return jsonify({"error": "Model not available on server"}), 500

    data = request.json
    timestamp = pd.to_datetime(f"{data['date']} {data['time']}")
    zone = data['zone']

    # Prepare DataFrame
    X_pred = pd.DataFrame({
        'timestamp': [timestamp],
        'zone': [zone]
    })

    X_pred['hour'] = X_pred['timestamp'].dt.hour
    X_pred['dayofweek'] = X_pred['timestamp'].dt.dayofweek
    X_pred['month'] = X_pred['timestamp'].dt.month

    for z in ZONES:
        X_pred[f'zone_{z}'] = (zone == z)

    features = ['hour', 'dayofweek', 'month'] + [f'zone_{z}' for z in ZONES]
    X_pred = X_pred[features]

    prediction = pipeline.predict(X_pred)
    return jsonify({
        'temp': float(prediction[0][0]),
        'uv': float(prediction[0][1])
    })

# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
