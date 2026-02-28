#!/usr/bin/env python3
"""
Flask Application for Cryptographic Algorithm Identification

Provides:
- /predict endpoint for API access (JSON)
- / endpoint for web UI
- /health endpoint for status checks

Inference latency target: <50ms per request
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template
from inference import CryptoPredictor, ALGORITHM_DESCRIPTIONS


# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Get or initialize predictor instance."""
    global predictor
    if predictor is None:
        models_dir = os.environ.get('MODELS_DIR', 'models')
        predictor = CryptoPredictor(models_dir=models_dir)
    return predictor


# ============================================================
# API Endpoints
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict cryptographic algorithm from ciphertext.
    
    Request JSON:
    {
        "ciphertext": "<hex string or base64>"
    }
    
    Response JSON:
    {
        "algorithm": "AES",
        "confidence": 95.5,
        "description": "...",
        "model_predictions": {...},
        "inference_time_ms": 12.5
    }
    """
    try:
        # Get ciphertext from request
        data = request.get_json(silent=True) or {}
        ciphertext = data.get('ciphertext', '')
        
        # Also check form data and query params
        if not ciphertext:
            ciphertext = request.form.get('ciphertext', '')
        if not ciphertext:
            ciphertext = request.args.get('ciphertext', '')
        
        if not ciphertext:
            return jsonify({
                'error': 'Missing required field: ciphertext',
                'usage': 'POST /predict with JSON body: {"ciphertext": "hex_string"}'
            }), 400
        
        # Get prediction
        pred = get_predictor()
        result = pred.predict(ciphertext)
        
        return jsonify(result)
    
    except FileNotFoundError as e:
        return jsonify({
            'error': 'Models not found. Please train models first.',
            'details': str(e)
        }), 503
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        models = pred.available_models
        algorithms = pred.supported_algorithms
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models,
            'supported_algorithms': algorithms
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@app.route('/algorithms', methods=['GET'])
def list_algorithms():
    """List supported algorithms with descriptions."""
    return jsonify({
        'algorithms': ALGORITHM_DESCRIPTIONS
    })


# ============================================================
# Web UI Endpoints
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """Web UI for predictions."""
    result = None
    ciphertext = ''
    error = None
    
    if request.method == 'POST':
        ciphertext = request.form.get('ciphertext', '').strip()
        
        if not ciphertext:
            error = 'Please enter a ciphertext to analyze.'
        else:
            try:
                pred = get_predictor()
                result = pred.predict(ciphertext)
            except FileNotFoundError:
                error = 'Models not loaded. Please train models first by running: python training/train_models.py'
            except Exception as e:
                error = f'Prediction failed: {str(e)}'
    
    return render_template(
        'index.html',
        result=result,
        ciphertext=ciphertext,
        error=error,
        algorithms=ALGORITHM_DESCRIPTIONS
    )


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith('/api') or request.is_json:
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    if request.path.startswith('/api') or request.is_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('error.html', error='Internal server error'), 500


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   Cryptographic Algorithm Identification System              ║
╠══════════════════════════════════════════════════════════════╣
║   Web UI:     http://localhost:{port}/                          ║
║   API:        http://localhost:{port}/predict                   ║
║   Health:     http://localhost:{port}/health                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
