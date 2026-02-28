#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

TARGET_ALGOS = ["AES", "DES", "RSA", "SHA256", "MD5", "Base64"]
MAX_SEQ_LEN = 2048
BATCH_SIZE = 128
EPOCHS = 15

# Determine device (macOS MPS if available, else CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using PyTorch device: {device}")

class TemperatureScaler(nn.Module):
    """
    A thin decorator to apply Temperature Scaling on logits.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def filter_data(df):
    """Remove samples smaller than 32 bytes."""
    return df[df["Length"] >= 32].reset_index(drop=True)

def apply_adversarial_attack(X_seq, attack_type):
    """Applies an adversarial perturbation to the raw sequence data."""
    X_adv = X_seq.copy()
    if attack_type == "truncate":
        for i in range(len(X_adv)):
            # Find actual length (before 0 padding)
            non_zero = np.where(X_adv[i] != 0)[0]
            if len(non_zero) > 0:
                actual_len = non_zero[-1] + 1
                cut_len = int(actual_len * 0.8) # Keep 80%
                X_adv[i][cut_len:] = 0
    elif attack_type == "flip_last":
        for i in range(len(X_adv)):
            non_zero = np.where(X_adv[i] != 0)[0]
            if len(non_zero) > 0:
                last_idx = non_zero[-1]
                X_adv[i][last_idx] = (X_adv[i][last_idx] + 1) % 256
    return torch.tensor(X_adv).to(device)

def prepare_data(df):
    df = filter_data(df)
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
            seq += [0] * (MAX_SEQ_LEN - len(seq))
        else:
            seq = seq[:MAX_SEQ_LEN]
        X_seq.append(seq)
        
    X_seq = np.array(X_seq, dtype=np.int64)
    y = np.array(df["Label"])
    return X_seq, y

class CryptoCNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CryptoCNN1D, self).__init__()
        self.embedding = nn.Embedding(256, 64)
        
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print("Loading datasets for CNN...")
    train_df = pd.read_csv("datasets/train_15k.csv")
    val_df = pd.read_csv("datasets/val_15k.csv")
    test_df = pd.read_csv("datasets/test_15k.csv")
    
    X_seq_tr, y_tr = prepare_data(train_df)
    X_seq_va, y_va = prepare_data(val_df)
    X_seq_te, y_te = prepare_data(test_df)
    
    classes_list = sorted(list(set(y_tr)))
    class_map = {c: i for i, c in enumerate(classes_list)}
    
    y_tr_enc = np.array([class_map[c] for c in y_tr], dtype=np.int64)
    y_va_enc = np.array([class_map[c] for c in y_va], dtype=np.int64)
    y_te_enc = np.array([class_map[c] for c in y_te], dtype=np.int64)
    num_classes = len(classes_list)
    
    train_dataset = TensorDataset(torch.tensor(X_seq_tr), torch.tensor(y_tr_enc))
    val_dataset = TensorDataset(torch.tensor(X_seq_va), torch.tensor(y_va_enc))
    test_dataset = TensorDataset(torch.tensor(X_seq_te), torch.tensor(y_te_enc))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n🧠 Training 1D CNN Model on {len(X_seq_tr)} valid samples...")
    model = CryptoCNN1D(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    patience = 3
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/cnn_model_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
                
    model.load_state_dict(torch.load("models/cnn_model_best.pt"))
    model.eval()
    
    print("\n🌡️ Calibrating CNN with Temperature Scaling...")
    scaler = TemperatureScaler().to(device)
    
    optimizer_cal = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    with torch.no_grad():
        logits_list = []
        labels_list = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits_list.append(model(batch_x))
            labels_list.append(batch_y)
        val_logits = torch.cat(logits_list)
        val_labels = torch.cat(labels_list).to(device)
        
    def eval_calibration():
        optimizer_cal.zero_grad()
        loss = criterion(scaler(val_logits), val_labels)
        loss.backward()
        return loss
    
    optimizer_cal.step(eval_calibration)
    print(f"  -> Optimal Temperature: {scaler.temperature.item():.4f}")
    
    print("\n⚔️ Running Adversarial Tests...")
    with torch.no_grad():
        X_adv_trunc = apply_adversarial_attack(X_seq_te, "truncate")
        X_adv_flip = apply_adversarial_attack(X_seq_te, "flip_last")
        trunc_probs = torch.softmax(scaler(model(X_adv_trunc)), dim=1).cpu().numpy()
        flip_probs = torch.softmax(scaler(model(X_adv_flip)), dim=1).cpu().numpy()
        
        trunc_preds = np.argmax(trunc_probs, axis=1)
        flip_preds = np.argmax(flip_probs, axis=1)
        print(f"  -> Truncated Sequence Accuracy: {np.mean(trunc_preds == y_te_enc)*100:.2f}%")
        print(f"  -> Flipped Last Byte Accuracy: {np.mean(flip_preds == y_te_enc)*100:.2f}%")
    
    print("\n⚖️ Evaluating Hybrid Ensemble (0.4 RF + 0.6 CNN)...")
    all_cnn_probs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            # Apply Temperature Scaling
            outputs = scaler(model(batch_x))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_cnn_probs.append(probs)
    
    cnn_probs = np.vstack(all_cnn_probs)
    # Using the filtered test labels saved by RF script (avoids misalignment)
    if os.path.exists("models/rf_test_probs.npy"):
        try:
            rf_probs = np.load("models/rf_test_probs.npy")
            y_te_enc_rf = np.load("models/rf_test_labels.npy")
        except:
            rf_probs = np.random.uniform(size=cnn_probs.shape)
            y_te_enc_rf = y_te_enc
            print("  ⚠️ RF probs loaded failure. Check scripts.")
    else:
        print("  ⚠️ RF model test probs not found. Proceeding with CNN only for eval...")
        rf_probs = np.zeros_like(cnn_probs)
        y_te_enc_rf = y_te_enc
        
    # We must match the lengths. They should be identical due to the same filter_data function
    ensemble_probs = (0.4 * rf_probs) + (0.6 * cnn_probs)
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    acc = np.mean(y_te_enc_rf == y_pred)
    
    print(f"\n--- ENSEMBLE TEST PERFORMANCE ---")
    print(f"Final Test Accuracy:  {acc * 100:.2f}%")
    
    print("\nPer-Class Accuracy:")
    for i, c in enumerate(classes_list):
        mask = (y_te_enc_rf == i)
        class_acc = np.mean(y_pred[mask] == i) if np.sum(mask) > 0 else 0
        print(f"  - {c}: {class_acc * 100:.2f}%")
        
    # Standard confusion matrix via Pandas
    print("\nConfusion Matrix:")
    cm = pd.crosstab(pd.Series(y_te_enc_rf, name='Actual'), pd.Series(y_pred, name='Predicted'))
    cm.index = [classes_list[i] for i in cm.index]
    cm.columns = [classes_list[i] for i in cm.columns]
    print(cm)
    
    if acc >= 0.98:
        print("\n🏆 TARGET ACHIEVED! Test accuracy is >= 98%.")
    else:
        print("\n⚠️ Target missed. Test accuracy is < 98%.")

if __name__ == "__main__":
    main()
