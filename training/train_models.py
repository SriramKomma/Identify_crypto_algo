#!/usr/bin/env python3
"""
Model Training Script for Cryptographic Algorithm Identification

Trains three models:
1. Random Forest Classifier (on engineered features)
2. Logistic Regression (on engineered features with regularization)
3. CNN (on raw byte sequences)

Evaluates each model and saves to /models directory.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from feature_extraction import extract_features, extract_raw_bytes, FEATURE_SIZE, RAW_SEQUENCE_LENGTH


# ============================================================
# Data Loading & Feature Extraction
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples")
    return df


def extract_all_features(df: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract engineered features from all ciphertexts.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels
    """
    if verbose:
        print("Extracting engineered features...")
    
    X = []
    for i, row in df.iterrows():
        features = extract_features(row['Ciphertext'])
        X.append(features)
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(df)} samples")
    
    X = np.array(X)
    y = df['Algorithm'].values
    
    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
    
    return X, y


def extract_all_raw_bytes(df: pd.DataFrame, max_length: int = RAW_SEQUENCE_LENGTH, 
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract raw byte sequences for CNN input.
    
    Returns:
        X: Raw byte sequences (n_samples, max_length)
        y: Labels
    """
    if verbose:
        print("Extracting raw byte sequences for CNN...")
    
    X = []
    for i, row in df.iterrows():
        raw = extract_raw_bytes(row['Ciphertext'], max_length)
        X.append(raw)
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(df)} samples")
    
    X = np.array(X)
    y = df['Algorithm'].values
    
    if verbose:
        print(f"  Raw sequence shape: {X.shape}")
    
    return X, y


# ============================================================
# Model Training Functions
# ============================================================

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train Random Forest classifier.
    """
    if verbose:
        print("\n" + "="*60)
        print("Training Random Forest Classifier")
        print("="*60)
    
    # Initialize model with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    if verbose:
        print("Training model...")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest", verbose)
    
    return model, metrics


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Train Logistic Regression with regularization.
    """
    if verbose:
        print("\n" + "="*60)
        print("Training Logistic Regression Classifier")
        print("="*60)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model with L2 regularization
    model = LogisticRegression(
        C=1.0,  # Regularization strength
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    if verbose:
        print("Training model...")
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred, "Logistic Regression", verbose)
    
    return model, scaler, metrics


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    epochs: int = 20,
    batch_size: int = 64,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train CNN on raw byte sequences.
    
    Uses lazy import to avoid TensorFlow deadlock on startup.
    """
    if verbose:
        print("\n" + "="*60)
        print("Training CNN Classifier")
        print("="*60)
        print("Loading TensorFlow...")
    
    # Lazy import TensorFlow to avoid startup deadlock
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
        Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    
    # Suppress TF logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Encode labels
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)
    
    # Build CNN model
    if verbose:
        print("Building CNN architecture...")
    
    model = Sequential([
        # Embedding layer: map byte values (0-255) to dense vectors
        Embedding(input_dim=256, output_dim=64, input_length=RAW_SEQUENCE_LENGTH),
        
        # First conv block
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # Second conv block
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # Third conv block
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        
        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if verbose:
        model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    if verbose:
        print("\nTraining CNN...")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_enc = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    
    metrics = evaluate_model(y_test, y_pred, "CNN", verbose)
    
    return model, metrics


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute and display evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if verbose:
        print(f"\n{model_name} Results:")
        print("-" * 40)
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    
    return metrics


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train models for cryptographic algorithm identification'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='data/crypto_dataset.csv',
        help='Path to dataset CSV'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--skip-cnn',
        action='store_true',
        help='Skip CNN training (useful if TensorFlow has issues)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Algorithm'])
    
    # Save label encoder
    joblib.dump(label_encoder, output_dir / 'label_encoder.pkl')
    if verbose:
        print(f"Label encoder saved. Classes: {label_encoder.classes_}")
    
    # --------------------------------------------------------
    # Train Random Forest & Logistic Regression (engineered features)
    # --------------------------------------------------------
    X_features, y = extract_all_features(df, verbose)
    
    X_train_f, X_test_f, y_train, y_test = train_test_split(
        X_features, y, test_size=args.test_size, stratify=y, random_state=42
    )
    
    if verbose:
        print(f"\nTrain/Test split: {len(X_train_f)} train, {len(X_test_f)} test")
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train_f, y_train, X_test_f, y_test, verbose)
    joblib.dump(rf_model, output_dir / 'random_forest.pkl')
    if verbose:
        print(f"✓ Random Forest saved to {output_dir / 'random_forest.pkl'}")
    
    # Train Logistic Regression
    lr_model, lr_scaler, lr_metrics = train_logistic_regression(
        X_train_f, y_train, X_test_f, y_test, verbose
    )
    joblib.dump(lr_model, output_dir / 'logistic_regression.pkl')
    joblib.dump(lr_scaler, output_dir / 'lr_scaler.pkl')
    if verbose:
        print(f"✓ Logistic Regression saved to {output_dir / 'logistic_regression.pkl'}")
    
    # --------------------------------------------------------
    # Train CNN (raw bytes)
    # --------------------------------------------------------
    cnn_metrics = None
    if not args.skip_cnn:
        X_raw, _ = extract_all_raw_bytes(df, verbose=verbose)
        
        X_train_r, X_test_r, _, _ = train_test_split(
            X_raw, y, test_size=args.test_size, stratify=y, random_state=42
        )
        
        try:
            cnn_model, cnn_metrics = train_cnn(
                X_train_r, y_train, X_test_r, y_test, label_encoder, verbose=verbose
            )
            cnn_model.save(output_dir / 'cnn_model.h5')
            if verbose:
                print(f"✓ CNN saved to {output_dir / 'cnn_model.h5'}")
        except Exception as e:
            print(f"⚠ CNN training failed: {e}")
            print("  System will continue with Random Forest and Logistic Regression only.")
    else:
        if verbose:
            print("\n⚠ CNN training skipped (--skip-cnn flag)")
    
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"\nRandom Forest:       {rf_metrics['accuracy']*100:.2f}% accuracy")
        print(f"Logistic Regression: {lr_metrics['accuracy']*100:.2f}% accuracy")
        if cnn_metrics:
            print(f"CNN:                 {cnn_metrics['accuracy']*100:.2f}% accuracy")
        
        print(f"\nModels saved to: {output_dir.absolute()}")
        print("\nTraining complete! ✓")


if __name__ == '__main__':
    main()
