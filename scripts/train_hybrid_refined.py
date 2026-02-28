#!/usr/bin/env python3
"""
Refined Hybrid Model Training Script for Crypto Algorithm Classification.
Trains a Statistical Random Forest and a 1D CNN, combining them in a weighted ensemble. 
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import tensorflow as tf
os.environ["OMP_NUM_THREADS"] = "1"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import math
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

TARGET_ALGOS = ["AES", "DES", "RSA", "SHA256", "MD5", "Base64"]
MAX_SEQ_LEN = 2048

# ---------------------------------------------------------
# Feature Extraction (Statistical Model)
# ---------------------------------------------------------
def shannon_entropy(byte_arr):
    if len(byte_arr) == 0: return 0.0
    _, counts = np.unique(byte_arr, return_counts=True)
    probs = counts / len(byte_arr)
    return -np.sum(probs * np.log2(probs))

def byte_histogram(byte_arr):
    hist = np.zeros(256, dtype=int)
    unique, counts = np.unique(byte_arr, return_counts=True)
    hist[unique] = counts
    return hist

def chi_square_score(byte_arr):
    if len(byte_arr) == 0: return 0.0
    hist = byte_histogram(byte_arr)
    expected = len(byte_arr) / 256.0
    expected_arr = np.full(256, expected)
    # Manual chi-square to avoid scipy/OpenBLAS dependency on macOS
    chi_sq = np.sum(np.square(hist - expected_arr) / (expected_arr + 1e-5))
    return float(chi_sq)

def block_repetition_score(byte_arr, block_size=16):
    """Measures how many blocks of `block_size` are identical."""
    if len(byte_arr) < block_size * 2:
        return 0.0
    blocks = [tuple(byte_arr[i:i+block_size]) for i in range(0, len(byte_arr) - len(byte_arr)%block_size, block_size)]
    if len(blocks) == 0: return 0.0
    unique_blocks = set(blocks)
    # Score = 1.0 means all blocks are identical, 0.0 means all are unique
    return 1.0 - (len(unique_blocks) / len(blocks))

def extract_statistical_features(hex_str: str):
    """Extracts all required statistical features from raw hex ciphertext"""
    try:
        # Check if valid hex, otherwise fallback to rough utf-8 bytes (Base64 case)
        if len(hex_str) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in hex_str):
            byte_arr = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
        else:
            byte_arr = np.frombuffer(hex_str.encode('utf-8'), dtype=np.uint8)
    except:
        byte_arr = np.frombuffer(hex_str.encode('utf-8', errors='ignore'), dtype=np.uint8)

    length = len(byte_arr)
    entropy = shannon_entropy(byte_arr)
    chi2 = chi_square_score(byte_arr)
    hist = byte_histogram(byte_arr)
    len_mod_8 = length % 8
    len_mod_16 = length % 16
    brs_8 = block_repetition_score(byte_arr, 8)
    brs_16 = block_repetition_score(byte_arr, 16)
    
    # Feature vector: [entropy, chi2, length, len_mod_8, len_mod_16, brs_8, brs_16] + [256 hist bins]
    features = [entropy, chi2, length, len_mod_8, len_mod_16, brs_8, brs_16]
    features.extend(hist)
    return features

# ---------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------
def prepare_data(df):
    """Processes DataFrame to extract statistical features and padded byte arrays."""
    print("  -> Extracting statistical features...")
    X_stat = np.array([extract_statistical_features(ct) for ct in df["Ciphertext"]])
    
    print("  -> Extracting raw byte sequences for CNN...")
    X_seq = []
    for ct in df["Ciphertext"]:
        try:
            if len(ct) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in ct):
                b = bytes.fromhex(ct[:MAX_SEQ_LEN*2])
            else:
                b = ct.encode('utf-8')[:MAX_SEQ_LEN]
        except:
            b = ct.encode('utf-8', errors='ignore')[:MAX_SEQ_LEN]
            
        seq = [b_val for b_val in b]
        if len(seq) < MAX_SEQ_LEN:
            seq += [0] * (MAX_SEQ_LEN - len(seq)) # Pad with zeros
        else:
            seq = seq[:MAX_SEQ_LEN] # Truncate exactly
        X_seq.append(seq)
        
    X_seq = np.array(X_seq)
    y = np.array(df["Label"])
    return X_stat, X_seq, y

def load_all_datasets():
    print("Loading datasets...")
    train_df = pd.read_csv("datasets/train_15k.csv")
    val_df = pd.read_csv("datasets/val_15k.csv")
    test_df = pd.read_csv("datasets/test_15k.csv")
    
    print("Preparing Training Data:")
    X_stat_tr, X_seq_tr, y_tr = prepare_data(train_df)
    print("Preparing Validation Data:")
    X_stat_va, X_seq_va, y_va = prepare_data(val_df)
    print("Preparing Test Data:")
    X_stat_te, X_seq_te, y_te = prepare_data(test_df)
    
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)
    y_te_enc = le.transform(y_te)
    
    return (X_stat_tr, X_seq_tr, y_tr_enc), (X_stat_va, X_seq_va, y_va_enc), (X_stat_te, X_seq_te, y_te_enc), le

# ---------------------------------------------------------
# Model Training
# ---------------------------------------------------------
def train_random_forest(X_train, y_train):
    print("\n🌲 Training Statistical Random Forest Model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Base model
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Hyperparameter space
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # We use a randomized search to save time while still tuning
    print("  -> Tuning hyperparameters...")
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=2, cv=3, 
                                scoring='accuracy', random_state=42, verbose=1)
    search.fit(X_train_scaled, y_train)
    
    best_rf = search.best_estimator_
    print(f"  -> Best RF Params: {search.best_params_}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/rf_scaler.pkl")
    joblib.dump(best_rf, "models/rf_model.pkl")
    return best_rf, scaler

def train_cnn(X_train, y_train, X_val, y_val, num_classes):
    print("\n🧠 Training 1D CNN Model...")
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    model = Sequential([
        Embedding(input_dim=256, output_dim=64, input_length=MAX_SEQ_LEN),
        
        Conv1D(filters=128, kernel_size=7, padding='same'),
        BatchNormAndRelu(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=128, kernel_size=5, padding='same'),
        BatchNormAndRelu(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=3, padding='same'),
        BatchNormAndRelu(),
        GlobalAveragePooling1D(),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    
    # To avoid exhausting memory or stalling, keep batch_size reasonable
    model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
              epochs=15, batch_size=128, callbacks=[early_stop], verbose=1)
              
    model.save("models/cnn_model.h5")
    return model
    
def BatchNormAndRelu():
    from tensorflow.keras.layers import BatchNormalization, Activation
    return Sequential([BatchNormalization(), Activation('relu')])

# ---------------------------------------------------------
# Evaluation & Ensemble
# ---------------------------------------------------------
def evaluate_ensemble(rf_model, scaler, cnn_model, X_stat_te, X_seq_te, y_te, le):
    print("\n⚖️ Evaluating Hybrid Ensemble (0.4 RF + 0.6 CNN)...")
    
    # RF Predictions
    X_stat_te_scaled = scaler.transform(X_stat_te)
    rf_probs = rf_model.predict_proba(X_stat_te_scaled)
    
    # CNN Predictions
    cnn_probs = cnn_model.predict(X_seq_te, batch_size=256, verbose=0)
    
    # Weighted Average
    ensemble_probs = (0.4 * rf_probs) + (0.6 * cnn_probs)
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='macro')
    
    print(f"\n--- ENSEMBLE TEST PERFORMANCE ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_te, y_pred)
    df_cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(df_cm)
    
    if acc >= 0.98:
        print("\n🏆 TARGET ACHIEVED! Test accuracy is >= 98%.")
    else:
        print("\n⚠️ Target missed. Test accuracy is < 98%.")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    tf.keras.backend.clear_session()
    
    # 1. Load Data
    train_data, val_data, test_data, le = load_all_datasets()
    X_stat_tr, X_seq_tr, y_tr_enc = train_data
    X_stat_va, X_seq_va, y_va_enc = val_data
    X_stat_te, X_seq_te, y_te_enc = test_data
    
    joblib.dump(le, "models/label_encoder.pkl")
    
    # 2. Train RF
    rf_model, scaler = train_random_forest(X_stat_tr, y_tr_enc)
    
    # Validate RF separately for reference
    rf_val_acc = accuracy_score(y_va_enc, rf_model.predict(scaler.transform(X_stat_va)))
    print(f"  -> RF Validation Accuracy: {rf_val_acc * 100:.2f}%")
    
    # 3. Train CNN
    cnn_model = train_cnn(X_seq_tr, y_tr_enc, X_seq_va, y_va_enc, num_classes=len(le.classes_))
    
    # 4. Evaluate Ensemble
    evaluate_ensemble(rf_model, scaler, cnn_model, X_stat_te, X_seq_te, y_te_enc, le)

if __name__ == "__main__":
    main()
