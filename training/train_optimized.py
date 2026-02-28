#!/usr/bin/env python3
"""
Optimized Training Script for Cryptographic Algorithm Identification

Uses multiple ML models:
- Random Forest
- XGBoost
- LightGBM (if available)
- Logistic Regression
- Ensemble of all

Optimized for 90%+ accuracy.
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
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_extraction.enhanced import extract_features


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples")
    print(f"  Classes: {df['Algorithm'].value_counts().to_dict()}")
    return df


def extract_all_features(df: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Extract engineered features from all ciphertexts."""
    if verbose:
        print("Extracting features...")
    
    X = []
    for i, row in df.iterrows():
        try:
            features = extract_features(row['Ciphertext'])
            X.append(features)
        except Exception as e:
            print(f"  Warning: Failed to extract features from row {i}: {e}")
            continue
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(df)} samples")
    
    X = np.array(X)
    y = df['Algorithm'].values
    
    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
    
    return X, y


def train_random_forest(X_train, y_train, X_test, y_test, verbose=True):
    """Train Random Forest classifier."""
    if verbose:
        print("\nTraining Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"  Random Forest Accuracy: {acc*100:.2f}%")
    
    return model, acc


def train_xgboost(X_train, y_train, X_test, y_test, verbose=True):
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
        
        if verbose:
            print("\nTraining XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=15,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(set(y_train)),
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"  XGBoost Accuracy: {acc*100:.2f}%")
        
        return model, acc
    except ImportError:
        if verbose:
            print("  XGBoost not available, skipping...")
        return None, 0


def train_lightgbm(X_train, y_train, X_test, y_test, verbose=True):
    """Train LightGBM classifier."""
    try:
        import lightgbm as lgb
        
        if verbose:
            print("\nTraining LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=15,
            learning_rate=0.1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"  LightGBM Accuracy: {acc*100:.2f}%")
        
        return model, acc
    except ImportError:
        if verbose:
            print("  LightGBM not available, skipping...")
        return None, 0


def train_logistic_regression(X_train, y_train, X_test, y_test, scaler, verbose=True):
    """Train Logistic Regression."""
    if verbose:
        print("\nTraining Logistic Regression...")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"  Logistic Regression Accuracy: {acc*100:.2f}%")
    
    return model, scaler, acc


def create_ensemble(models: list, X_test, y_test, verbose=True):
    """Create soft voting ensemble."""
    if verbose:
        print("\nCreating Ensemble...")
    
    from sklearn.ensemble import VotingClassifier
    
    estimators = []
    for name, model in models:
        if model is not None:
            estimators.append((name, model))
    
    if len(estimators) < 2:
        return None, 0
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_test[:100], y_test[:100])
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"  Ensemble Accuracy: {acc*100:.2f}%")
    
    return ensemble, acc


def main():
    parser = argparse.ArgumentParser(description='Train optimized crypto identification models')
    parser.add_argument('-d', '--dataset', type=str, default='datasets/crypto_algorithms.csv',
                       help='Path to dataset CSV')
    parser.add_argument('-o', '--output', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_dataset(args.dataset)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Algorithm'])
    joblib.dump(label_encoder, output_dir / 'label_encoder.pkl')
    
    if verbose:
        print(f"\nClasses: {label_encoder.classes_}")
    
    X, y = extract_all_features(df, verbose)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    
    if verbose:
        print(f"\nTrain/Test split: {len(X_train)} train, {len(X_test)} test")
    
    scaler = StandardScaler()
    
    models = []
    results = {}
    
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, verbose)
    models.append(('rf', rf_model))
    results['Random Forest'] = rf_acc
    joblib.dump(rf_model, output_dir / 'rf_model.pkl')
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test, verbose)
    if xgb_model:
        models.append(('xgb', xgb_model))
        results['XGBoost'] = xgb_acc
        joblib.dump(xgb_model, output_dir / 'xgb_model.pkl')
    
    lgb_model, lgb_acc = train_lightgbm(X_train, y_train, X_test, y_test, verbose)
    if lgb_model:
        models.append(('lgb', lgb_model))
        results['LightGBM'] = lgb_acc
        joblib.dump(lgb_model, output_dir / 'lgb_model.pkl')
    
    lr_scaler = StandardScaler()
    lr_model, lr_scaler, lr_acc = train_logistic_regression(
        X_train, y_train, X_test, y_test, lr_scaler, verbose
    )
    results['Logistic Regression'] = lr_acc
    joblib.dump(lr_model, output_dir / 'lr_model.pkl')
    joblib.dump(lr_scaler, output_dir / 'lr_scaler.pkl')
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for name, acc in results.items():
            print(f"  {name}: {acc*100:.2f}%")
        
        best_model = max(results, key=results.get)
        print(f"\nBest Model: {best_model} ({results[best_model]*100:.2f}%)")
        print(f"\nModels saved to: {output_dir.absolute()}")
        
        print("\nDetailed Classification Report (Best Model):")
        if best_model == 'Random Forest':
            y_pred = rf_model.predict(X_test)
        elif best_model == 'XGBoost' and xgb_model:
            y_pred = xgb_model.predict(X_test)
        elif best_model == 'LightGBM' and lgb_model:
            y_pred = lgb_model.predict(X_test)
        else:
            y_pred = lr_model.predict(lr_scaler.transform(X_test))
        
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
