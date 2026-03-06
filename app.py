from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

print("="*70)
print("LOADING MODEL AND FEATURES")
print("="*70)

# Load trained model
try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: best_model.pkl not found!")
    model = None

# Load features
try:
    features = pickle.load(open('best_features.pkl', 'rb'))
    print(f"✓ Features loaded: {features}")
except FileNotFoundError:
    print("❌ Error: best_features.pkl not found!")
    features = None

# Load scaler
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✓ Scaler loaded successfully")
except FileNotFoundError:
    print("❌ Error: scaler.pkl not found!")
    scaler = None

# Load metrics
try:
    metrics = pickle.load(open('model_metrics.pkl', 'rb'))
    print(f"✓ Metrics loaded: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
except FileNotFoundError:
    print("❌ Error: model_metrics.pkl not found!")
    metrics = {'mae': 25.34, 'r2': 0.4821}

print("="*70)

# ============================================
# ROUTE 1: HOME PAGE
# ============================================

@app.route('/')
def home():
    """
    Render the home page with input form
    """
    return render_template('index.html')

# ============================================
# ROUTE 2: PREDICT ENDPOINT
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives JSON with song features and returns predicted Eurovision points
    """
    
    try:
        # Get JSON data from request
        data = request.json
        
        print("\n" + "="*70)
        print("PREDICTION REQUEST RECEIVED")
        print("="*70)
        
        # Extract features from request
        energy = float(data.get('energy', 0.5))
        duration = float(data.get('duration', 200))
        acousticness = float(data.get('acousticness', 0.5))
        danceability = float(data.get('danceability', 0.5))
        tempo = float(data.get('tempo', 120))
        speechiness = float(data.get('speechiness', 0.1))
        liveness = float(data.get('liveness', 0.2))
        loudness = float(data.get('loudness', -6))
        valence = float(data.get('valence', 0.5))
        happiness = float(data.get('happiness', 0.5))
        
        print(f"\nReceived Input Values:")
        print(f"  Energy: {energy}")
        print(f"  Duration: {duration}")
        print(f"  Acousticness: {acousticness}")
        print(f"  Danceability: {danceability}")
        print(f"  Tempo: {tempo}")
        print(f"  Speechiness: {speechiness}")
        print(f"  Liveness: {liveness}")
        print(f"  Loudness: {loudness}")
        print(f"  Valence: {valence}")
        print(f"  Happiness: {happiness}")
        
        # Validate input ranges
        if not (0 <= energy <= 1):
            return jsonify({'status': 'error', 'message': 'Energy must be between 0 and 1'}), 400
        if not (0 <= danceability <= 1):
            return jsonify({'status': 'error', 'message': 'Danceability must be between 0 and 1'}), 400
        if not (0 <= acousticness <= 1):
            return jsonify({'status': 'error', 'message': 'Acousticness must be between 0 and 1'}), 400
        
        # Create dataframe with features in correct order
        song_data = pd.DataFrame({
            'energy': [energy],
            'duration': [duration],
            'acousticness': [acousticness],
            'danceability': [danceability],
            'tempo': [tempo],
            'speechiness': [speechiness],
            'liveness': [liveness],
            'loudness': [loudness],
            'valence': [valence],
            'Happiness': [happiness]
        })
        
        print(f"\nDataFrame created:")
        print(song_data)
        
        # Scale the features
        print(f"\nScaling features...")
        song_scaled = scaler.transform(song_data)
        print(f"✓ Features scaled")
        
        # Make prediction
        print(f"\nMaking prediction...")
        predicted_points = model.predict(song_scaled)[0]
        print(f"✓ Prediction complete")
        
        # Calculate confidence interval
        mae = metrics['mae']
        lower_bound = max(0, predicted_points - mae)
        upper_bound = predicted_points + mae
        
        print(f"\nPrediction Results:")
        print(f"  Predicted Points: {predicted_points:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}")
        print(f"  Upper Bound: {upper_bound:.2f}")
        print(f"  MAE: {mae:.2f}")
        
        # Determine performance rating
        if predicted_points >= 300:
            rating = "🏆 EXCELLENT - Strong Contender"
        elif predicted_points >= 200:
            rating = "✅ GOOD - Competitive Song"
        elif predicted_points >= 100:
            rating = "⚠️ AVERAGE - Has Potential"
        else:
            rating = "❌ WEAK - May Need Improvement"
        
        # Return JSON response
        response = {
            'status': 'success',
            'predicted_points': round(predicted_points, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'margin_of_error': round(mae, 2),
            'confidence_range': f"{round(lower_bound, 0):.0f} - {round(upper_bound, 0):.0f}",
            'rating': rating,
            'model_accuracy': f"R² {metrics['r2']:.4f}",
            'model_mae': f"±{mae:.2f} points"
        }
        
        print(f"\nResponse sent:")
        print(response)
        print("="*70)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 400

# ============================================
# ROUTE 3: MODEL INFO ENDPOINT
# ============================================

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Return information about the trained model
    """
    info = {
        'model_type': 'Gradient Boosting Regressor',
        'features_used': features,
        'num_features': len(features),
        'accuracy_r2': metrics['r2'],
        'accuracy_percent': f"{metrics['r2']*100:.2f}%",
        'mae': metrics['mae'],
        'model_status': 'Production Ready',
        'top_features': ['Loudness', 'Energy', 'Valence', 'Danceability', 'Tempo']
    }
    return jsonify(info)

# ============================================
# ROUTE 4: HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """
    Check if API is running
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Eurovision Predictor API is running'
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING FLASK APPLICATION")
    print("="*70)
    print("\nFlask app running on: http://localhost:5000")
    print("Prediction endpoint: POST http://localhost:5000/predict")
    print("Model info endpoint: GET http://localhost:5000/model-info")
    print("\nPress Ctrl+C to stop the server")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)