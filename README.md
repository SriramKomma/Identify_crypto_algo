# Cryptographic Algorithm Identifier

A machine learning-based system to identify cryptographic algorithms (AES, DES, RSA, etc.) from ciphertext.

## Features

- **Hybrid Ensemble Learning**: Combines Random Forest and XGBoost for robust classification.
- **Deep Learning**: 1D-CNN for raw byte pattern analysis.
- **Web Dashboard**: Interactive UI for real-time analysis and visualization.
- **API**: REST API for statistical metrics.
- **Dockerized**: Easy deployment with Docker.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd crypt-copy
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python run.py
    ```
    Access the app at `http://127.0.0.1:5000`.

## Docker Deployment

1.  **Build the image**:
    ```bash
    docker build -t crypto-id .
    ```

2.  **Run container**:
    ```bash
    docker run -p 5000:5000 crypto-id
    ```

## Testing

Run the test suite:
```bash
python -m pytest
```

## Architecture

- **`app/`**: Flask application package.
    - **`services/`**: Core logic (ML inference).
    - **`routes/`**: Web and API endpoints.
    - **`models/`**: Database models (SQLAlchemy).
- **`models/`**: Trained ML models (pkl/h5).
- **`tests/`**: Pytest suite.
