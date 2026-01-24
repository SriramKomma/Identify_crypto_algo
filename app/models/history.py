from app import db
from datetime import datetime

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ciphertext_snippet = db.Column(db.String(100), nullable=False)
    predicted_algorithm = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def to_dict(self):
        return {
            'ciphertext': self.ciphertext_snippet,
            'prediction': self.predicted_algorithm,
            'confidence': self.confidence,
            'timestamp': self.timestamp.strftime("%H:%M:%S")
        }
