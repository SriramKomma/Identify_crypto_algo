from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.services.predictor import predictor
from app.models.history import PredictionHistory
from app import db

web_bp = Blueprint('web', __name__)

@web_bp.route("/", methods=["GET"])
def index():
    # Fetch recent history from DB
    try:
        history = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).limit(10).all()
        history_list = [h.to_dict() for h in history]
    except Exception:
        # Fallback if DB is not ready
        history_list = []
        
    return render_template("index.html", history=history_list, meta_inputs={}, ciphertext_only=False)

@web_bp.route("/predict", methods=["POST"])
def predict():
    ciphertext = request.form.get("ciphertext")
    use_cnn = request.form.get("use_cnn") == "on"
    ciphertext_only = request.form.get("ciphertext_only") == "on"
    
    if not ciphertext:
        flash("Please enter a ciphertext.")
        return redirect(url_for("web.index"))
    def _parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    meta_inputs = {
        "PlaintextLen": _parse_int(request.form.get("plaintext_len")),
        "KeyLen": _parse_int(request.form.get("key_len")),
        "BlockSize": _parse_int(request.form.get("block_size")),
        "IVLen": _parse_int(request.form.get("iv_len")),
        "Mode": request.form.get("mode") or None,
    }

    missing = [k for k, v in meta_inputs.items() if v is None]
    if missing and not ciphertext_only:
        flash("For 95%+ accuracy, provide all metadata fields or enable ciphertext-only mode (lower accuracy).")
        return redirect(url_for("web.index"))
    if missing and ciphertext_only:
        flash("Ciphertext-only mode enabled; accuracy may be lower (<95%).")

    hybrid_results, meta = predictor.predict_hybrid(ciphertext, meta_inputs)
    hybrid_results, meta = predictor.predict_hybrid(ciphertext)
    cnn_result = predictor.predict_cnn(ciphertext) if use_cnn else None

    predicted_algorithm = max(hybrid_results, key=hybrid_results.get) if hybrid_results else "Unknown"
    description = predictor.get_description(predicted_algorithm)
    
    # Save to DB
    try:
        new_entry = PredictionHistory(
            ciphertext_snippet=ciphertext[:20] + "...",
            predicted_algorithm=predicted_algorithm,
            confidence=hybrid_results[predicted_algorithm] if hybrid_results else 0.0
        )
        db.session.add(new_entry)
        db.session.commit()
        
        history = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).limit(10).all()
        history_list = [h.to_dict() for h in history]
    except Exception:
        db.session.rollback()
        history_list = []

    return render_template(
        "index.html",
        results=hybrid_results,
        ciphertext=ciphertext,
        predicted_algorithm=predicted_algorithm,
        algorithm_description=description,
        cnn_result=cnn_result,
        meta=meta,
        history=history_list,
        meta_inputs=meta_inputs,
        ciphertext_only=ciphertext_only
    )
