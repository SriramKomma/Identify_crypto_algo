from flask import Blueprint, jsonify, request
from app.services.pipeline import pipeline

api_bp = Blueprint('api', __name__)

@api_bp.route("/stats")
def stats():
    return jsonify({
        "labels": ["v1", "v2", "v3", "v4 (Metadata)"],
        "accuracy": [35, 65, 43, 100],
        "models": ["RandomForest", "CNN", "Hybrid Ensemble", "Hybrid + Metadata"]
    })

@api_bp.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    ciphertext = data.get("ciphertext")
    if not ciphertext:
        return jsonify({"error": "ciphertext is required"}), 400

    def _parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    meta_inputs = {
        "PlaintextLen": _parse_int(data.get("plaintext_len")),
        "KeyLen": _parse_int(data.get("key_len")),
        "BlockSize": _parse_int(data.get("block_size")),
        "IVLen": _parse_int(data.get("iv_len")),
        "Mode": data.get("mode") or None,
    }

    missing = [k for k, v in meta_inputs.items() if v is None]
    ciphertext_only = bool(data.get("ciphertext_only"))
    warning = None
    if missing and ciphertext_only:
        warning = "Ciphertext-only mode enabled; accuracy may be lower (<95%)."
    elif missing:
        warning = "Metadata incomplete; accuracy may be lower. Provide all fields or set ciphertext_only=true."

    # Intercept via new 4-stage pipeline
    pipeline_result = pipeline.identify(ciphertext, meta_inputs)
    
    if "error" in pipeline_result:
        return jsonify(pipeline_result), 500

    if warning and pipeline_result.get("stage") == "ML-ensemble":
        pipeline_result["warning"] = warning
        
    return jsonify(pipeline_result)

