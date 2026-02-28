import joblib
import os
import math
import re
import numpy as np
import pandas as pd

# Defer TensorFlow import to avoid deadlock on startup
tf = None

def _lazy_import_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf

class PredictorService:
    def __init__(self):
        # Assuming run from root, so models are in ./models
        self.model_dir = os.path.join(os.getcwd(), 'models') 
        self._hybrid_model = None
        self._hybrid_le = None
        self._cnn_model = None
        self._cnn_le = None
        self._models_loaded = False
        self.hex_re = re.compile(r'^[0-9a-fA-F]+$')
        self.algorithm_descriptions = {
            'AES': "Advanced Encryption Standard (AES) - Symmetric block cipher (128-bit block).",
            'DES': "Data Encryption Standard (DES) - Older symmetric cipher (64-bit block).",
            '3DES': "Triple DES - Applies DES three times for stronger security.",
            'Blowfish': "Blowfish - Fast symmetric block cipher (64-bit block).",
            'RSA': "RSA - Asymmetric algorithm used for secure data transmission.",
            'ECC': "Elliptic Curve Cryptography - Asymmetric, high security with smaller keys.",
            'Diffie-Hellman': "Diffie-Hellman - Key exchange protocol."
        }
        # Models are loaded lazily on first use

    @property
    def hybrid_model(self):
        self._ensure_models_loaded()
        return self._hybrid_model

    @property
    def hybrid_le(self):
        self._ensure_models_loaded()
        return self._hybrid_le

    @property
    def cnn_model(self):
        self._ensure_models_loaded()
        return self._cnn_model

    @property
    def cnn_le(self):
        self._ensure_models_loaded()
        return self._cnn_le

    def _ensure_models_loaded(self):
        if self._models_loaded:
            return
        self._load_models()
        self._models_loaded = True

    def _load_models(self):
        hybrid_path = os.path.join(self.model_dir, "hybrid_ensemble.pkl")
        hybrid_le_path = os.path.join(self.model_dir, "hybrid_label_encoder.pkl")
        cnn_path = os.path.join(self.model_dir, "cnn_cipher_model.h5")
        cnn_le_path = os.path.join(self.model_dir, "cnn_label_encoder.pkl")

        if os.path.exists(hybrid_path):
            self._hybrid_model = joblib.load(hybrid_path)
        if os.path.exists(hybrid_le_path):
            self._hybrid_le = joblib.load(hybrid_le_path)
        if os.path.exists(cnn_path) and os.path.exists(cnn_le_path):
            _lazy_import_tf()
            self._cnn_model = tf.keras.models.load_model(cnn_path)
            self._cnn_le = joblib.load(cnn_le_path)

    def shannon_entropy(self, s):
        if not s:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in probs)

    def hex_ratio(self, s):
        return sum(1 for c in s if c in "0123456789abcdefABCDEF") / len(s) if s else 0.0

    def byte_stats(self, s):
        try:
            if self.hex_re.match(s) and len(s) % 2 == 0:
                arr = np.frombuffer(bytes.fromhex(s), dtype=np.uint8).astype(float)
            else:
                arr = np.array([ord(c) for c in s], dtype=float)
            return float(arr.mean()), float(arr.std())
        except Exception:
            return 0.0, 0.0

    def infer_features(self, ciphertext, meta_inputs=None):
        ct = str(ciphertext).strip()
        ent = self.shannon_entropy(ct)
        hr = self.hex_ratio(ct)
        meanb, stdb = self.byte_stats(ct)
        length = len(ct)
        meta_inputs = meta_inputs or {}

        features = pd.DataFrame([{
            "Ciphertext": ct,
            "PlaintextLen": meta_inputs.get("PlaintextLen"),
            "KeyLen": meta_inputs.get("KeyLen"),
            "BlockSize": meta_inputs.get("BlockSize"),
            "IVLen": meta_inputs.get("IVLen"),
            "Mode": meta_inputs.get("Mode"),
            "Entropy": ent,
            "HexRatio": hr,
            "ByteMean": meanb,
            "ByteStd": stdb,
            "CipherLen": length
        }])

        meta_display = {
            "Ciphertext": ct[:50] + "..." if len(ct) > 50 else ct,
            "Length": length,
            "Entropy": round(ent, 4),
            "HexRatio": round(hr, 2),
            "ByteMean": round(meanb, 2),
            "ByteStd": round(stdb, 2),
            "PlaintextLen": meta_inputs.get("PlaintextLen"),
            "KeyLen": meta_inputs.get("KeyLen"),
            "BlockSize": meta_inputs.get("BlockSize"),
            "IVLen": meta_inputs.get("IVLen"),
            "Mode": meta_inputs.get("Mode")
        }
        return features, meta_display

    def predict_hybrid(self, ciphertext, meta_inputs=None):
        if not self.hybrid_model or not self.hybrid_le:
            return None, None
            
        features_df, meta_display = self.infer_features(ciphertext, meta_inputs)
        
        input_df = features_df.copy()
        
        probs = self.hybrid_model.predict_proba(input_df)
        labels = self.hybrid_le.inverse_transform(range(len(probs[0])))
        
        results = {labels[i]: round(probs[0][i] * 100, 2) for i in range(len(labels))}
        return results, meta_display

    def predict_cnn(self, ciphertext):
        if not self.cnn_model or not self.cnn_le:
            return None
            
        try:
            if self.hex_re.match(ciphertext) and len(ciphertext) % 2 == 0:
                raw = bytes.fromhex(ciphertext[:512])
            else:
                raw = bytes([ord(c) for c in ciphertext[:256]])
        except:
            raw = bytes([ord(c) for c in ciphertext[:256]])

        seq = list(raw)[:256]
        seq += [0] * (256 - len(seq))
        arr = np.array([seq])

        preds = self.cnn_model.predict(arr)[0]
        labels = self.cnn_le.inverse_transform(range(len(preds)))
        
        results = {labels[i]: round(float(preds[i]) * 100, 2) for i in range(len(labels))}
        return results

    def get_description(self, algo_name):
        return self.algorithm_descriptions.get(algo_name, "Unknown Algorithm")

predictor = PredictorService()
