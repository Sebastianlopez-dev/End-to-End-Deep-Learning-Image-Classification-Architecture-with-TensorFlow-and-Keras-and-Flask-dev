"""
Flask Web Application for CIFAR-10 Image Classification.

This app allows users to upload one or multiple images and receive
predictions with class probabilities from the best trained model.

Usage:
    conda run -n ironhack.nn python app/app.py

Then navigate to http://localhost:5001 in your browser.
(Port 5001 is used because macOS Monterey+ reserves 5000 for AirPlay.)
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import CLASS_NAMES

# TensorFlow / Keras imports
from keras.api.models import load_model
from keras.api.utils import img_to_array

# ── Configuration ───────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

# Try to load the best model (transfer learning first, then custom CNN)
TRANSFER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mobilenetv2_tl.keras')
CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'custom_cnn.keras')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Global models dictionary
models = {}
default_model_name = None


def load_all_models():
    """Load both Custom CNN and MobileNetV2 models."""
    global models, default_model_name

    if os.path.exists(TRANSFER_MODEL_PATH):
        models['transfer'] = {
            'model': load_model(TRANSFER_MODEL_PATH),
            'name': 'MobileNetV2 Transfer Learning'
        }
        print(f"✅ Loaded model: {models['transfer']['name']}")
        default_model_name = 'transfer'

    if os.path.exists(CUSTOM_MODEL_PATH):
        models['custom'] = {
            'model': load_model(CUSTOM_MODEL_PATH),
            'name': 'Custom CNN'
        }
        print(f"✅ Loaded model: {models['custom']['name']}")
        default_model_name = 'custom'

    if not models:
        print("⚠️  No trained models found! Please run training scripts first.")
        default_model_name = None


def preprocess_image(image_bytes, target_size=(32, 32)):
    """
    Preprocess a single uploaded image for prediction.

    Steps:
    1. Open and convert to RGB
    2. Resize to target_size (32x32 for CNN or 96x96 for MobileNetV2)
    3. Normalize pixel values to [0, 1]
    4. Add batch dimension

    Args:
        image_bytes: Raw image bytes from upload.
        target_size: The expected resolution tuple (width, height).

    Returns:
        numpy array of shape (1, target_size[0], target_size[1], 3), float32, normalized.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Automatically resize the image to the required target size
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_predictions(img_array, model_choice, top_n=5):
    """
    Get top-N class predictions with probabilities.

    Args:
        img_array: Preprocessed image array (1, 32, 32, 3).
        model_choice: String key selecting which model to use ('custom' or 'transfer').
        top_n: Number of top predictions to return.

    Returns:
        list of dicts: [{'class': str, 'probability': float}, ...]
    """
    if not models or model_choice not in models:
        return [{'class': 'No model loaded', 'probability': 0.0}]

    selected_model = models[model_choice]['model']
    predictions = selected_model.predict(img_array, verbose=0)[0]
    top_indices = predictions.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'class': CLASS_NAMES[idx],
            'probability': float(predictions[idx])
        })
    return results


# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render the main upload page."""
    current_model = models.get(default_model_name, {}).get('name', 'Loading...') if default_model_name else 'No model loaded'
    return render_template('index.html', model_name=current_model)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload(s) and return predictions.

    Accepts single or multiple files via the 'files' form field.
    Returns JSON with predictions for each uploaded image.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    model_choice = request.form.get('model_choice', default_model_name)
    all_results = []

    for file in files:
        if file.filename == '':
            continue

        safe_name = secure_filename(file.filename) or 'unknown'

        try:
            image_bytes = file.read()
            target_size = (96, 96) if model_choice == 'transfer' else (32, 32)
            img_array = preprocess_image(image_bytes, target_size=target_size)
            predictions = get_predictions(img_array, model_choice, top_n=10)

            all_results.append({
                'filename': safe_name,
                'predictions': predictions
            })
        except Exception as e:
            all_results.append({
                'filename': safe_name,
                'error': str(e)
            })

    current_model_name = models.get(model_choice, {}).get('name', 'Unknown Model')

    return render_template('index.html',
                           model_name=current_model_name,
                           selected_model=model_choice,
                           results=all_results)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint: returns JSON predictions (for programmatic access).
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    model_choice = request.form.get('model_choice', default_model_name)
    current_model_name = models.get(model_choice, {}).get('name', 'Unknown Model')
    
    all_results = []

    for file in files:
        if file.filename == '':
            continue

        safe_name = secure_filename(file.filename) or 'unknown'

        try:
            image_bytes = file.read()
            target_size = (96, 96) if model_choice == 'transfer' else (32, 32)
            img_array = preprocess_image(image_bytes, target_size=target_size)
            predictions = get_predictions(img_array, model_choice, top_n=10)
            all_results.append({
                'filename': safe_name,
                'predictions': predictions
            })
        except Exception as e:
            all_results.append({
                'filename': safe_name,
                'error': str(e)
            })

    return jsonify({'model': current_model_name, 'results': all_results})


# ── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_all_models()
    port = int(os.environ.get('PORT', 5001))
    print(f"\n 🚀 Starting Flask app at http://localhost:{port}")
    print(f"   Available Models: {', '.join([m['name'] for m in models.values()])}")
    print(f"   Classes: {', '.join(CLASS_NAMES)}")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
