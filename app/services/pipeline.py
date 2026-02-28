import re
import base64
from app.services.predictor import predictor

class CryptoPipeline:
    def __init__(self):
        self.hex_re = re.compile(r'^[0-9A-Fa-f]+$')
        self.b64_re = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

    def _is_hex(self, s):
        return bool(self.hex_re.match(s))

    def detect_base64(self, ciphertext):
        # Stage 1: Encoding Detection
        if len(ciphertext) % 4 != 0:
            return False
            
        if not self.b64_re.match(ciphertext):
            return False
            
        # If it's pure hex, it's highly ambiguous but likely a hash or raw ciphertext.
        # True random base64 almost always contains non-hex characters.
        if self._is_hex(ciphertext):
            return False
            
        try:
            base64.b64decode(ciphertext, validate=True)
            return True
        except Exception:
            return False

    def detect_hash(self, ciphertext):
        # Stage 2: Hash Detection
        if not self._is_hex(ciphertext):
            return None
            
        length = len(ciphertext)
        if length == 32:
            return "MD5"
        elif length == 64:
            return "SHA256"
            
        return None

    def ensemble_refinement(self, hybrid_probs, cnn_probs):
        # Stage 4: Combine ML + CNN
        if not hybrid_probs:
            return None, 0.0
            
        final_probs = {}
        
        if not cnn_probs:
            # Fall back to just hybrid model if CNN isn't loaded
            for algo, prob in hybrid_probs.items():
                final_probs[algo] = prob
        else:
            # Weighted average: 60% Hybrid, 40% CNN
            # (Assuming both models output probabilities for the same labels)
            all_labels = set(hybrid_probs.keys()).union(set(cnn_probs.keys()))
            for algo in all_labels:
                h_prob = hybrid_probs.get(algo, 0.0)
                c_prob = cnn_probs.get(algo, 0.0)
                final_probs[algo] = (h_prob * 0.6) + (c_prob * 0.4)
                
        predicted_algo = max(final_probs, key=final_probs.get)
        confidence = final_probs[predicted_algo]
        
        # Round confidence to 4 decimal places as float (e.g. 0.9821)
        return predicted_algo, round(confidence / 100.0, 4)

    def identify(self, ciphertext, meta_inputs=None):
        # Format the ciphertext to strip whitespace
        ciphertext = str(ciphertext).strip()
        
        # Stage 1: Encoding Detection
        if self.detect_base64(ciphertext):
            return {
                "algorithm": "Base64",
                "confidence": 1.0,
                "stage": "Encoding"
            }
            
        # Stage 2: Hash Detection
        hash_type = self.detect_hash(ciphertext)
        if hash_type:
            return {
                "algorithm": hash_type,
                "confidence": 1.0,
                "stage": "Hash"
            }
            
        # Stage 3: Encryption Type Classification
        hybrid_probs, meta_display = predictor.predict_hybrid(ciphertext, meta_inputs)
        
        if not hybrid_probs:
            # Provide a fallback if models aren't loaded 
            return {
                "error": "ML models not available for Stage 3.",
                "stage": "ML-ensemble"
            }
            
        cnn_probs = predictor.predict_cnn(ciphertext)
        
        # Stage 4: Ensemble Refinement
        predicted_algo, confidence = self.ensemble_refinement(hybrid_probs, cnn_probs)
        
        return {
            "algorithm": predicted_algo,
            "confidence": confidence,
            "stage": "ML-ensemble",
            "meta": meta_display,
            "hybrid_probs": hybrid_probs,
            "cnn_probs": cnn_probs
        }

pipeline = CryptoPipeline()
