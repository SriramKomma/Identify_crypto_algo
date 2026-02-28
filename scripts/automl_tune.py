import argparse
import math
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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
        if HEX_RE.match(s) and len(s) % 2 == 0:
            arr = np.frombuffer(bytes.fromhex(s), dtype=np.uint8).astype(float)
        else:
            arr = np.array([ord(c) for c in s], dtype=float)
        return float(arr.mean()), float(arr.std())
    except Exception:
        return 0.0, 0.0

def extract_cipher_features(ct_series):
    rows = []
    for ct in ct_series.astype(str):
        ent = shannon_entropy(ct)
        hr = hex_ratio(ct)
        meanb, stdb = byte_stats(ct)
        length = len(ct)
        rows.append([ent, hr, meanb, stdb, length])
    return pd.DataFrame(rows, columns=["Entropy", "HexRatio", "ByteMean", "ByteStd", "CipherLen"])

def _safe_col(df, name, default=None):
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df))

def build_feature_frame(df):
    ct_series = df["Ciphertext"].astype(str)
    cipher_feats = extract_cipher_features(ct_series)
    base = pd.DataFrame({
        "Ciphertext": ct_series,
        "PlaintextLen": _safe_col(df, "PlaintextLen"),
        "KeyLen": _safe_col(df, "KeyLen"),
        "BlockSize": _safe_col(df, "BlockSize"),
        "IVLen": _safe_col(df, "IVLen"),
        "Mode": _safe_col(df, "Mode"),
    })
    return pd.concat([base.reset_index(drop=True), cipher_feats.reset_index(drop=True)], axis=1)

def build_pipeline(model_name="xgb"):
    num_cols = ["PlaintextLen", "KeyLen", "BlockSize", "IVLen", "Entropy", "HexRatio", "ByteMean", "ByteStd", "CipherLen"]
    cat_cols = ["Mode"]
    text_col = "Ciphertext"

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=10000), text_col),
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    if model_name == "rf":
        clf = RandomForestClassifier(random_state=42)
        param_distributions = {
            "clf__n_estimators": [200, 400, 600, 800],
            "clf__max_depth": [None, 10, 20, 30, 40],
            "clf__min_samples_split": [2, 4, 6, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None]
        }
    else:
        clf = XGBClassifier(
            random_state=42,
            n_jobs=4,
            eval_metric="mlogloss",
            objective="multi:softprob"
        )
        param_distributions = {
            "clf__n_estimators": [200, 400, 600, 800],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 0.1, 0.2]
        }

    pipe = Pipeline([
        ("pre", preproc),
        ("clf", clf)
    ])
    return pipe, param_distributions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/dataset_v3.csv", help="Path to dataset CSV.")
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb", help="Model to tune.")
    parser.add_argument("--n-iter", type=int, default=25, help="Random search iterations.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument("--output-prefix", default="tuned", help="Output prefix for model artifacts.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = build_feature_frame(df)
    y = df["Algorithm"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    pipe, param_dist = build_pipeline(args.model)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=args.cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_
    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Best params: {search.best_params_}")
    print(f"Holdout accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=le.classes_))

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{args.output_prefix}_ensemble.pkl")
    le_path = os.path.join("models", f"{args.output_prefix}_label_encoder.pkl")
    joblib.dump(best, model_path)
    joblib.dump(le, le_path)
    print(f"Saved model to {model_path}")
    print(f"Saved label encoder to {le_path}")

if __name__ == "__main__":
    main()
