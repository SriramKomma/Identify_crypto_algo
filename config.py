import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'supersecret'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Models are in the project root/models by default, but we can configure this if needed.
    # For now, the predictor service will look for them.
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
