from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow all origins (restrict in production if needed)

# Load the trained scikit-learn models and scaler
model_path = os.path.join(os.path.dirname(__file__), 'pond_model.pkl')
try:
    rf_model, gb_model, scaler = joblib.load(model_path)  # Load the tuple (rf_model, gb_model, scaler)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define the endpoint to receive weather parameters and predict DO level
@app.route('/api/weather', methods=['POST'])
def weather_data():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract parameters with defaults (aligned with training features)
        temperature = data.get('Temperature', '25')
        pressure = data.get('Pressure', '1013')
        humidity = data.get('Humidity', '60')
        delta_temp = data.get('DeltaTemp', '15')
        days_passed = data.get('DaysPassed', '0')
        wind_speed = data.get('WindSpeed', '15')
        cloud_coverage = data.get('CloudCoverage', '12')
        dew_point = data.get('DewPoint', '54')
        wind_gust = data.get('WindGust', '2')
        moonshine_hours = data.get('MoonshineHours', '0.6')

        # Prepare input data for the model (match training feature order)
        features = [
            float(temperature) if temperature else 0,
            float(pressure) if pressure else 0,
            float(humidity) if humidity else 0,
            float(delta_temp) if delta_temp else 0,
            float(days_passed) if days_passed else 0,
            float(wind_speed) if wind_speed else 0,
            float(cloud_coverage) if cloud_coverage else 0,
            float(dew_point) if dew_point else 0,
            float(wind_gust) if wind_gust else 0,
            float(moonshine_hours) if moonshine_hours else 0
        ]

        # Convert to numpy array and scale
        input_data = np.array([features])
        input_data_scaled = scaler.transform(input_data)

        # Make predictions using both models
        pred_rf = rf_model.predict(input_data_scaled)
        pred_gb = gb_model.predict(input_data_scaled)
        final_pred = (pred_rf + pred_gb) / 2  # Ensemble prediction

        # Response
        response = {
            "status": "success",
            "received_data": {
                "Temperature": temperature,
                "Pressure": pressure,
                "Humidity": humidity,
                "DeltaTemp": delta_temp,
                "DaysPassed": days_passed,
                "WindSpeed": wind_speed,
                "CloudCoverage": cloud_coverage,
                "DewPoint": dew_point,
                "WindGust": wind_gust,
                "MoonshineHours": moonshine_hours
            },
            "prediction": float(final_pred[0]),
            "message": "Dissolved Oxygen (DO) level prediction generated successfully"
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Optional: Serve a simple homepage
@app.route('/')
def home():
    return "Welcome to the AKVA."

if __name__ == '__main__':
    # Use host 0.0.0.0 and port from environment variable for Render
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
