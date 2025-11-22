"""
Flask REST API for Air Quality Predictions
Provides endpoints for AQI predictions and health advisories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
from models.lstm_predictor import LSTMAQIPredictor
from data.preprocessing import AirQualityPreprocessor
from data.data_loader import AirQualityDataLoader
from api.health_advisor import HealthAdvisor

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

# Initialize components
model = LSTMAQIPredictor(input_shape=(72, 20), forecast_horizon=24)
preprocessor = AirQualityPreprocessor()
loader = AirQualityDataLoader()
advisor = HealthAdvisor()

# Load trained model and scalers
try:
    model.load_model("models/saved/final_model.keras")
    preprocessor.load_scalers()
    print("✓ Model and scalers loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load model - {e}")


@app.route('/', methods=['GET'])
def home():
    """API home endpoint."""
    return jsonify({
        'service': 'Air Quality Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health_advisory': '/api/health-advisory (POST)',
            'historical': '/api/historical (GET)',
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict AQI for next 24-48 hours.
    
    Request body:
    {
        "city": "Los Angeles",
        "country": "US",
        "latitude": 34.05,
        "longitude": -118.24
    }
    
    Returns:
    {
        "predictions": [...],
        "timestamps": [...],
        "current_aqi": float
    }
    """
    try:
        data = request.get_json()
        
        # Get parameters
        city = data.get('city', 'Los Angeles')
        country = data.get('country', 'US')
        lat = data.get('latitude', 34.05)
        lon = data.get('longitude', -118.24)
        
        # Load recent data (last 72 hours for LSTM input)
        aqi_data = loader.get_air_quality_data(city=city, country=country, days=7)
        weather_data = loader.get_weather_data(latitude=lat, longitude=lon, days=7)
        
        # Merge and preprocess
        merged = loader.merge_datasets(aqi_data, weather_data)
        processed = preprocessor.engineer_features(merged)
        
        # Get last 72 hours as input
        X_input = processed[preprocessor.feature_columns].iloc[-72:].values
        X_scaled = preprocessor.feature_scaler.transform(X_input)
        X_seq = X_scaled.reshape(1, 72, -1)
        
        # Make prediction
        y_pred_scaled = model.predict(X_seq)
        y_pred = preprocessor.inverse_transform_predictions(y_pred_scaled[0])
        
        # Generate timestamps
        last_time = processed['datetime'].iloc[-1]
        timestamps = [
            (last_time + timedelta(hours=i+1)).isoformat()
            for i in range(len(y_pred))
        ]
        
        # Current AQI
        current_aqi = merged['pm25'].iloc[-1] if 'pm25' in merged.columns else None
        
        return jsonify({
            'success': True,
            'city': city,
            'country': country,
            'current_aqi': float(current_aqi) if current_aqi else None,
            'predictions': y_pred.tolist(),
            'timestamps': timestamps,
            'forecast_hours': len(y_pred)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health-advisory', methods=['POST'])
def health_advisory():
    """
    Get personalized health advisory.
    
    Request body:
    {
        "aqi": 125.5,
        "user_profile": {
            "age": 65,
            "has_respiratory_condition": true,
            "outdoor_activity_level": "moderate"
        }
    }
    
    Returns:
    {
        "aqi": 125.5,
        "risk_level": "Unhealthy for Sensitive Groups",
        "recommendations": [...],
        ...
    }
    """
    try:
        data = request.get_json()
        
        aqi = data.get('aqi')
        if aqi is None:
            return jsonify({
                'success': False,
                'error': 'AQI value required'
            }), 400
        
        user_profile = data.get('user_profile', {})
        
        # Generate advisory
        advisory_data = advisor.generate_advisory(aqi, user_profile)
        
        return jsonify({
            'success': True,
            **advisory_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/historical', methods=['GET'])
def historical():
    """
    Get historical air quality data.
    
    Query params:
    - city: City name
    - country: Country code
    - days: Number of days (default 7)
    
    Returns:
    {
        "data": [...],
        "summary": {...}
    }
    """
    try:
        city = request.args.get('city', 'Los Angeles')
        country = request.args.get('country', 'US')
        days = int(request.args.get('days', 7))
        
        # Load data
        aqi_data = loader.get_air_quality_data(city=city, country=country, days=days)
        
        # Convert to records
        records = aqi_data.to_dict('records')
        
        # Calculate summary statistics
        if 'pm25' in aqi_data.columns:
            summary = {
                'avg_aqi': float(aqi_data['pm25'].mean()),
                'max_aqi': float(aqi_data['pm25'].max()),
                'min_aqi': float(aqi_data['pm25'].min()),
                'days': days
            }
        else:
            summary = {'days': days}
        
        return jsonify({
            'success': True,
            'city': city,
            'country': country,
            'data': records,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" 🌍 AIR QUALITY PREDICTION API ")
    print("="*60)
    print("\nAPI running on http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /api/predict - Get AQI predictions")
    print("  POST /api/health-advisory - Get health recommendations")
    print("  GET  /api/historical - Get historical data")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
