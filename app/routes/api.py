from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)

@api_bp.route("/stats")
def stats():
    return jsonify({
        "labels": ["v1", "v2", "v3 (Current)"],
        "accuracy": [35, 65, 43],
        "models": ["RandomForest", "CNN", "Hybrid Ensemble"]
    })
