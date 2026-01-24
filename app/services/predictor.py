import joblib
import os
import math
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredictorService:
    def __init__(self):
        # Assuming run from root, so models are in ./models
        self.model_dir = os.path.join(os.getcwd(), 'models') 
        self.hybrid_model = None
        self.hybrid_le = None
        self.cnn_model = None
        self.cnn_le = None
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
        self._load_models()

    def _load_models(self):
        hybrid_path = os.path.join(self.model_dir, "hybrid_ensemble.pkl")
        hybrid_le_path = os.path.join(self.model_dir, "hybrid_label_encoder.pkl")
        cnn_path = os.path.join(self.model_dir, "cnn_cipher_model.h5")
        cnn_le_path = os.path.join(self.model_dir, "cnn_label_encoder.pkl")

        if os.path.exists(hybrid_path):
            self.hybrid_model = joblib.load(hybrid_path)
        if os.path.exists(hybrid_le_path):
            self.hybrid_le = joblib.load(hybrid_le_path)
        if os.path.exists(cnn_path) and os.path.exists(cnn_le_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            self.cnn_le = joblib.load(cnn_le_path)

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

    def infer_features(self, ciphertext):
        ct = str(ciphertext).strip()
        ent = self.shannon_entropy(ct)
        hr = self.hex_ratio(ct)
        meanb, stdb = self.byte_stats(ct)
        length = len(ct)
        
        features = pd.DataFrame([[ent, hr, meanb, stdb, length]], 
                                columns=["Entropy", "HexRatio", "ByteMean", "ByteStd", "CipherLen"])
        
        meta_display = {
            "Ciphertext": ct[:50] + "..." if len(ct) > 50 else ct,
            "Length": length,
            "Entropy": round(ent, 4),
            "HexRatio": round(hr, 2),
            "ByteMean": round(meanb, 2),
            "ByteStd": round(stdb, 2)
        }
        return features, meta_display

    def predict_hybrid(self, ciphertext):
        if not self.hybrid_model or not self.hybrid_le:
            return None, None
            
        features_df, meta_display = self.infer_features(ciphertext)
        
        input_df = features_df.copy()
        input_df["Ciphertext"] = ciphertext
        
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

        preds = self.cnn_model.predict(arr)
        idx = np.argmax(preds)
        label = self.cnn_le.inverse_transform([idx])[0]
        return {"label": label, "prob": round(float(preds[0][idx]) * 100, 2)}

    def get_description(self, algo_name):
        return self.algorithm_descriptions.get(algo_name, "Unknown Algorithm")

predictor = PredictorService()
