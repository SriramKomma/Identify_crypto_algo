# 🔐 Automated Identification of Cryptographic Algorithms Using AI & ML

An end-to-end machine learning system that identifies cryptographic algorithms used to generate ciphertext, **without any encryption metadata or plaintext**.

## 🎯 Supported Algorithms

| Algorithm | Type | Key Size | Block Size |
|-----------|------|----------|------------|
| **AES** | Symmetric | 128/192/256-bit | 128-bit |
| **DES** | Symmetric | 56-bit | 64-bit |
| **3DES** | Symmetric | 168-bit | 64-bit |
| **Blowfish** | Symmetric | 32-448-bit | 64-bit |
| **RSA** | Asymmetric | 2048/4096-bit | Variable |
| **ECC** | Asymmetric | 256-bit (P-256) | Variable |

## 📁 Project Structure

```
├── app.py                    # Flask web application
├── feature_extraction/       # Feature engineering module
│   └── __init__.py          # Shannon entropy, byte stats, n-grams
├── data/                     # Dataset directory
│   └── generate_dataset.py  # Ciphertext generator
├── training/                 # Model training scripts
│   └── train_models.py      # Train RF, LR, CNN models
├── inference/               # Prediction service
│   └── __init__.py          # Ensemble predictor
├── models/                   # Saved trained models
├── templates/               # HTML templates
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python data/generate_dataset.py -n 2000 -o data/crypto_dataset.csv
```

This generates 2000 samples per algorithm (12,000 total).

### 3. Train Models

```bash
# Train Random Forest + Logistic Regression (fast)
python training/train_models.py --skip-cnn

# Or train all models including CNN (requires TensorFlow)
python training/train_models.py
```

### 4. Run the Application

```bash
python app.py
```

Access at: http://localhost:5000

## 🔌 API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ciphertext": "3C670A3FE286635D64B954F17CA29B424BF5B6F2D21649090DCEEE06F82DFBA0"}'
```

### Response

```json
{
  "algorithm": "AES",
  "confidence": 95.5,
  "description": "Advanced Encryption Standard (AES) - A symmetric block cipher...",
  "model_predictions": {
    "random_forest": {"prediction": "AES", "confidence": 96.2},
    "logistic_regression": {"prediction": "AES", "confidence": 94.8}
  },
  "inference_time_ms": 12.5
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Web UI |
| `/predict` | POST | API prediction |
| `/health` | GET | Health check |
| `/algorithms` | GET | List supported algorithms |

## 🧠 System Architecture

### 1. Feature Extraction Layer

Extracts **285 features** from raw ciphertext:

- **Shannon Entropy** (1 feature) - Measures randomness
- **Hex Character Ratio** (1 feature) - Encoding detection
- **Byte Statistics** (7 features) - Mean, std, variance, skewness, kurtosis
- **Byte Frequency Distribution** (256 features) - Full histogram
- **N-gram Statistics** (10 features) - Bigram patterns
- **Block Pattern Features** (8 features) - Block size detection
- **Length Features** (2 features) - Raw and log-scaled

### 2. ML/DL Inference Layer

Three models trained in parallel:

| Model | Input | Description |
|-------|-------|-------------|
| **Random Forest** | Engineered features | 300 trees, balanced classes |
| **Logistic Regression** | Scaled features | L2 regularization, multinomial |
| **CNN** (optional) | Raw byte sequences | 3-layer conv + dense |

### 3. Ensemble Strategy

1. Run all available models in parallel
2. **Maximum voting** to determine winner
3. On tie → use **average probability** across models
4. Compute final confidence score

## 📊 Evaluation Metrics

After training, the system reports:

- **Accuracy** - Overall correctness
- **Precision** - Per-algorithm precision
- **Recall** - Per-algorithm recall
- **F1-Score** - Harmonic mean of precision/recall

Target performance: **>90% accuracy** on ciphertext-only classification.

## ⚙️ Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `MODELS_DIR` | models | Models directory |
| `LOAD_CNN` | false | Load CNN model |
| `DEBUG` | false | Flask debug mode |

## 🔬 Data Generation

The dataset generator creates authentic ciphertexts using:

```python
# AES - CBC mode with random IV
cipher = AES.new(key, AES.MODE_CBC, iv)

# RSA - OAEP padding
cipher = PKCS1_OAEP.new(key.publickey())

# ECC - ECDSA signatures + key exchange simulation
signer = DSS.new(key, 'fips-186-3')
```

**No metadata leakage** - Only ciphertext is stored and used for training.

## 🛠️ Development

### Run Tests

```bash
pytest tests/ -v
```

### Train with Custom Dataset

```bash
python training/train_models.py -d your_dataset.csv -o models
```

### Generate More Data

```bash
python data/generate_dataset.py -n 5000 --min-length 32 --max-length 512
```

## 📈 Performance

- **Inference latency**: <50ms per request
- **Model size**: ~50MB (RF + LR)
- **Memory usage**: ~200MB at runtime

## 🔒 Security Notes

- This tool is for **educational and research purposes**
- Real-world ciphertext identification is significantly harder
- Performance degrades with:
  - Very short ciphertexts (<16 bytes)
  - Non-standard encryption modes
  - Hybrid/composite encryption schemes

---

Built with ❤️ using Python, scikit-learn, TensorFlow, and Flask
