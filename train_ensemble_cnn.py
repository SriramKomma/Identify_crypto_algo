import pandas as pd
import numpy as np
import joblib, os, re, tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Feature Extraction Helpers (Consistent with App)
# ---------------------------------------------------------
HEX_RE = re.compile(r'^[0-9a-fA-F]+$')

def shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def hex_ratio(s):
    return sum(1 for c in s if c in "0123456789abcdefABCDEF") / len(s) if s else 0.0

def byte_stats(s):
    try:
        # Check if it looks like hex and has even length
        if HEX_RE.match(s) and len(s) % 2 == 0:
            arr = np.frombuffer(bytes.fromhex(s), dtype=np.uint8).astype(float)
        else:
            # Treat as raw string if not valid hex
            arr = np.array([ord(c) for c in s], dtype=float)
        return float(arr.mean()), float(arr.std())
    except Exception:
        return 0.0, 0.0

def extract_features(df):
    """
    Derive features from Ciphertext column.
    Returns a DataFrame with numerical features.
    """
    data = []
    for ct in df["Ciphertext"].astype(str):
        ent = shannon_entropy(ct)
        hr = hex_ratio(ct)
        meanb, stdb = byte_stats(ct)
        length = len(ct)
        data.append([ent, hr, meanb, stdb, length])
    
    return pd.DataFrame(data, columns=["Entropy", "HexRatio", "ByteMean", "ByteStd", "CipherLen"])

def load_data(path="datasets/dataset_v3.csv"):
    df = pd.read_csv(path)
    # We only need Algorithm and Ciphertext. We will re-compute features to be safe/honest.
    X_raw = df[["Ciphertext"]]
    y = df["Algorithm"]
    return X_raw, y

def train_cnn_on_bytes(X_raw, y):
    # Convert hex ciphertext to byte sequences
    seqs = []
    for s in X_raw["Ciphertext"].astype(str):
        try:
            # Try to decode as hex, if fails treat as raw bytes
            if HEX_RE.match(s) and len(s) % 2 == 0:
                 b = bytes.fromhex(s[:512]) # limit length
            else:
                 b = bytes([ord(c) for c in s[:256]])
        except:
            b = bytes([ord(c) for c in s[:256]])
            
        seqs.append([x for x in b])
        
    maxlen = 256 # Fixed length for consistency
    X_pad = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating='post')
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    model = Sequential([
        Embedding(256, 64, input_length=maxlen),
        Conv1D(128, 5, activation="relu"),
        MaxPooling1D(2),
        Conv1D(64, 5, activation="relu"), # Added another layer
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(le.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y_cat, test_size=0.1, random_state=42)
    
    print("🚀 Training CNN Model...")
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1) # Increased epochs
    
    model.save("models/cnn_cipher_model.h5")
    joblib.dump(le, "models/cnn_label_encoder.pkl")
    return model, le

def train_hybrid():
    X_raw, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 1. Extract manual features
    print("Extracting features...")
    X_features = extract_features(X_raw)
    
    # 2. Combine Text (TF-IDF) + Numerical Features
    # We need to pass the Ciphertext column to TF-IDF, and the extracted features to StandardScaler
    
    # Create a combined DataFrame for the pipeline
    X_combined = pd.concat([X_raw.reset_index(drop=True), X_features.reset_index(drop=True)], axis=1)
    
    num_cols = ["Entropy", "HexRatio", "ByteMean", "ByteStd", "CipherLen"]
    text_col = "Ciphertext"

    preproc = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char", ngram_range=(2,4), max_features=10000), text_col),
        ("num", StandardScaler(), num_cols)
    ])

    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, n_jobs=4, eval_metric="mlogloss")
    
    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba"
    )

    pipe = Pipeline([
        ("pre", preproc),
        ("clf", stack)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_enc, stratify=y_enc, test_size=0.2, random_state=42)
    
    print("🚀 Training Stacking Ensemble (Honest Mode - No Leakage)...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Ensemble Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/hybrid_ensemble.pkl")
    joblib.dump(le, "models/hybrid_label_encoder.pkl")
    print("✅ Models saved in models/")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_hybrid()
    
    print("\n📡 Training CNN for byte pattern detection...")
    X_raw, y = load_data()
    train_cnn_on_bytes(X_raw, y)
    print("\n🔥 All models trained successfully.")
