"""
Optimized Inference Module for Cryptographic Algorithm Identification

Provides ensemble prediction using:
- Random Forest
- XGBoost
- LightGBM
- Logistic Regression

Uses probability averaging for final prediction.
"""

import os
import sys
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.enhanced import extract_features


ALGORITHM_DESCRIPTIONS = {
    'AES': "AES - Advanced Encryption Standard (128-bit block cipher)",
    'DES': "DES - Data Encryption Standard (legacy 56-bit block cipher)",
    '3DES': "Triple DES - Legacy block cipher",
    'Blowfish': "Blowfish - Fast block cipher with variable key length",
    'RSA': "RSA - Asymmetric encryption based on prime factorization",
    'MD5': "MD5 - 128-bit hash function (deprecated for security)",
    'SHA1': "SHA-1 - 160-bit hash function (deprecated)",
    'SHA256': "SHA-256 - Part of SHA-2 family, 256-bit hash",
    'SHA512': "SHA-512 - Part of SHA-2 family, 512-bit hash",
    'Base64': "Base64 - Binary-to-text encoding scheme",
    'Hex': "Hexadecimal - Base-16 encoding",
    'LegacyBlock': "Legacy Block Cipher (DES/Blowfish) - 8-byte block ciphers"
}


class CryptoPredictor:
    """Ensemble predictor for cryptographic algorithm identification."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self._rf_model = None
        self._xgb_model = None
        self._lgb_model = None
        self._lr_model = None
        self._scaler = None
        self._lr_scaler = None
        self._label_encoder = None
        self._models_loaded = False
    
    def _ensure_models_loaded(self):
        """Lazy load models on first prediction."""
        if self._models_loaded:
            return
        
        le_path = self.models_dir / 'label_encoder.pkl'
        if le_path.exists():
            self._label_encoder = joblib.load(le_path)
        else:
            raise FileNotFoundError(f"Label encoder not found: {le_path}")
        
        rf_path = self.models_dir / 'rf_model.pkl'
        if rf_path.exists():
            self._rf_model = joblib.load(rf_path)
            print(f"✓ Loaded Random Forest model")
        
        xgb_path = self.models_dir / 'xgb_model.pkl'
        if xgb_path.exists():
            self._xgb_model = joblib.load(xgb_path)
            print(f"✓ Loaded XGBoost model")
        
        lgb_path = self.models_dir / 'lgb_model.pkl'
        if lgb_path.exists():
            self._lgb_model = joblib.load(lgb_path)
            print(f"✓ Loaded LightGBM model")
        
        lr_path = self.models_dir / 'lr_model.pkl'
        lr_scaler_path = self.models_dir / 'lr_scaler.pkl'
        if lr_path.exists() and lr_scaler_path.exists():
            self._lr_model = joblib.load(lr_path)
            self._lr_scaler = joblib.load(lr_scaler_path)
            print(f"✓ Loaded Logistic Regression model")
        
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
        
        self._models_loaded = True
    
    def _get_proba(self, model, features, use_scaler=False, scaler=None):
        """Get prediction probabilities from a model."""
        if model is None:
            return None
        
        try:
            if use_scaler and scaler is not None:
                features = scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(features)[0]
            elif hasattr(model, 'predict'):
                pred = model.predict(features)
                proba = np.zeros(len(self._label_encoder.classes_))
                idx = np.where(self._label_encoder.classes_ == pred[0])[0]
                if len(idx) > 0:
                    proba[idx[0]] = 1.0
                return proba
        except Exception as e:
            print(f"Error getting probabilities: {e}")
        
        return None
    
    def predict(self, ciphertext: Union[str, bytes], use_ensemble: bool = True) -> Dict:
        """Predict cryptographic algorithm from ciphertext."""
        start_time = time.time()
        
        self._ensure_models_loaded()
        
        features = extract_features(ciphertext)
        
        predictions = {}
        probas = {}
        
        rf_proba = self._get_proba(self._rf_model, features)
        if rf_proba is not None:
            predictions['random_forest'] = self._label_encoder.inverse_transform([np.argmax(rf_proba)])[0]
            probas['random_forest'] = {
                self._label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(rf_proba)
            }
        
        xgb_proba = self._get_proba(self._xgb_model, features)
        if xgb_proba is not None:
            predictions['xgboost'] = self._label_encoder.inverse_transform([np.argmax(xgb_proba)])[0]
            probas['xgboost'] = {
                self._label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(xgb_proba)
            }
        
        lgb_proba = self._get_proba(self._lgb_model, features)
        if lgb_proba is not None:
            predictions['lightgbm'] = self._label_encoder.inverse_transform([np.argmax(lgb_proba)])[0]
            probas['lightgbm'] = {
                self._label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(lgb_proba)
            }
        
        lr_proba = self._get_proba(self._lr_model, features, use_scaler=True, scaler=self._lr_scaler)
        if lr_proba is not None:
            predictions['logistic_regression'] = self._label_encoder.inverse_transform([np.argmax(lr_proba)])[0]
            probas['logistic_regression'] = {
                self._label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(lr_proba)
            }
        
        if use_ensemble and len(probas) > 0:
            avg_proba = {}
            all_algos = set()
            for p in probas.values():
                all_algos.update(p.keys())
            
            for algo in all_algos:
                algo_probas = [p.get(algo, 0.0) for p in probas.values()]
                avg_proba[algo] = np.mean(algo_probas)
            
            final_pred = max(avg_proba.keys(), key=lambda k: avg_proba[k])
            final_conf = avg_proba[final_pred]
        elif predictions:
            final_pred = list(predictions.values())[0]
            final_conf = 1.0 / len(self._label_encoder.classes_)
        else:
            final_pred = 'Unknown'
            final_conf = 0.0
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'algorithm': final_pred,
            'confidence': round(final_conf * 100, 2),
            'description': ALGORITHM_DESCRIPTIONS.get(final_pred, "Unknown algorithm"),
            'model_predictions': {
                model: {
                    'prediction': pred,
                    'confidence': round(max(probas[model].values()) * 100, 2)
                }
                for model, pred in predictions.items()
            },
            'inference_time_ms': round(inference_time, 2)
        }
    
    @property
    def available_models(self) -> List[str]:
        """List of successfully loaded models."""
        self._ensure_models_loaded()
        models = []
        if self._rf_model is not None:
            models.append('random_forest')
        if self._xgb_model is not None:
            models.append('xgboost')
        if self._lgb_model is not None:
            models.append('lightgbm')
        if self._lr_model is not None:
            models.append('logistic_regression')
        return models
    
    @property
    def supported_algorithms(self) -> List[str]:
        """List of algorithms the model can identify."""
        self._ensure_models_loaded()
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        return list(ALGORITHM_DESCRIPTIONS.keys())


_predictor_instance = None

def get_predictor(models_dir: str = 'models') -> CryptoPredictor:
    """Get or create singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CryptoPredictor(models_dir)
    return _predictor_instance

def predict(ciphertext: Union[str, bytes], use_ensemble: bool = True) -> Dict:
    """Convenience function for prediction."""
    predictor = get_predictor()
    return predictor.predict(ciphertext, use_ensemble)
