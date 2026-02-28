#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

TARGET_ALGOS = ["AES", "DES", "RSA", "SHA256", "MD5", "Base64"]

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
    chi_sq = np.sum(np.square(hist - expected_arr) / (expected_arr + 1e-5))
    return float(chi_sq)

def block_repetition_score(byte_arr, block_size=16):
    if len(byte_arr) < block_size * 2: return 0.0
    blocks = [tuple(byte_arr[i:i+block_size]) for i in range(0, len(byte_arr) - len(byte_arr)%block_size, block_size)]
    if len(blocks) == 0: return 0.0
    unique_blocks = set(blocks)
    return 1.0 - (len(unique_blocks) / len(blocks))

def extract_statistical_features(hex_str: str):
    try:
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
    
    features = [entropy, chi2, length, len_mod_8, len_mod_16, brs_8, brs_16]
    features.extend(hist)
    return features

def filter_data(df):
    """Remove samples smaller than 32 bytes based on physical hex length (64 chars) or base64 length."""
    return df[df["Length"] >= 32].reset_index(drop=True)

def prepare_data(df):
    df = filter_data(df)
    X_stat = np.array([extract_statistical_features(ct) for ct in df["Ciphertext"]])
    y = np.array(df["Label"])
    return X_stat, y

def main():
    print("Loading datasets for RF...")
    train_df = pd.read_csv("datasets/train_15k.csv")
    val_df = pd.read_csv("datasets/val_15k.csv")
    test_df = pd.read_csv("datasets/test_15k.csv")
    
    # We can combine train and val for GridSearchCV's native cross-validation logic
    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    X_stat_tr, y_tr = prepare_data(combined_train_df)
    X_stat_te, y_te = prepare_data(test_df)
    
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")
    
    print(f"\n🌲 Training Statistical Random Forest Model on {len(X_stat_tr)} valid samples (>32 bytes)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_stat_tr)
    X_test_scaled = scaler.transform(X_stat_te)
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # 5-Fold Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 25, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    from sklearn.model_selection import GridSearchCV
    print("  -> Running 5-Fold GridSearchCV...")
    search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    search.fit(X_train_scaled, y_tr_enc)
    
    best_rf = search.best_estimator_
    print(f"  -> Best RF Params: {search.best_params_}")
    print(f"  -> Best CV Accuracy: {search.best_score_ * 100:.2f}%")
    
    # Feature Importance
    print("\n📊 Feature Importance Analysis:")
    feature_names = ["Entropy", "Chi-Square", "Length", "Len%8", "Len%16", "Rep8", "Rep16"] + [f"Byte_{i}" for i in range(256)]
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for idx in indices[:10]:
        print(f"  - {feature_names[idx]}: {importances[idx]:.4f}")
        
    print("\n🎯 Calibrating Probabilities (Isotonic)...")
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_rf = CalibratedClassifierCV(best_rf, method='isotonic', cv='prefit')
    calibrated_rf.fit(X_train_scaled, y_tr_enc) # Prefit uses the training data just to scale the bounds
    
    from sklearn.metrics import brier_score_loss
    # Calculate Brier score for calibration metric (needs binary format, so we do it per class)
    calibrated_probs = calibrated_rf.predict_proba(X_test_scaled)
    y_te_bin = np.eye(len(le.classes_))[y_te_enc]
    brier_sum = 0
    for i in range(len(le.classes_)):
        brier_sum += brier_score_loss(y_te_bin[:, i], calibrated_probs[:, i])
    avg_brier = brier_sum / len(le.classes_)
    print(f"  -> Average Brier Score (Calibration): {avg_brier:.4f}")
    
    joblib.dump(scaler, "models/rf_scaler.pkl")
    joblib.dump(calibrated_rf, "models/rf_model.pkl")
    
    # Save test probabilities for ensemble later
    np.save("models/rf_test_probs.npy", calibrated_probs)
    # Also save the filtered ground truth to align the CNN test results
    np.save("models/rf_test_labels.npy", y_te_enc)
    print("✅ RF Training complete. Calibrated test probabilities saved.")

if __name__ == "__main__":
    main()
