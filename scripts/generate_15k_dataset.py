#!/usr/bin/env python3
"""
High-Performance Dataset Generator for Crypto Algorithm Classification.
Generates 15,000 samples per algorithm: AES, DES, RSA, SHA256, MD5, Base64.
"""

import os
import random
import multiprocessing
from functools import partial
import pandas as pd
from tqdm import tqdm
from base64 import b64encode
from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256, MD5
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from sklearn.model_selection import train_test_split

SAMPLES_PER_ALGO = 15000
TARGET_ALGOS = ["AES", "DES", "RSA", "SHA256", "MD5", "Base64"]

# Shared RSA Keys to speed up generation (generating 2048-bit keys 15,000 times takes forever)
# We pre-generate a pool of 10 keys for each size to simulate different keys without the overhead
print("Pre-generating RSA keys...")
RSA_1024_POOL = [RSA.generate(1024) for _ in range(5)]
RSA_2048_POOL = [RSA.generate(2048) for _ in range(5)]

def get_random_plaintext():
    length = random.randint(16, 2048)
    return get_random_bytes(length)

def gen_aes(_):
    while True:
        try:
            pt = get_random_plaintext()
            key = get_random_bytes(16) # 128-bit key
            mode = random.choice(["CBC", "ECB"])
            
            if mode == "CBC":
                iv = get_random_bytes(16)
                cipher = AES.new(key, AES.MODE_CBC, iv)
                ct_bytes = cipher.encrypt(pad(pt, AES.block_size))
                ct_hex = (iv + ct_bytes).hex().upper()
            else:
                cipher = AES.new(key, AES.MODE_ECB)
                ct_bytes = cipher.encrypt(pad(pt, AES.block_size))
                ct_hex = ct_bytes.hex().upper()
                
            return "AES", ct_hex, len(ct_hex)
        except Exception:
            continue

def gen_des(_):
    while True:
        try:
            pt = get_random_plaintext()
            key = get_random_bytes(8) # 56-bit key (8 bytes with parity)
            iv = get_random_bytes(8)
            cipher = DES.new(key, DES.MODE_CBC, iv)
            ct_bytes = cipher.encrypt(pad(pt, DES.block_size))
            ct_hex = (iv + ct_bytes).hex().upper()
            return "DES", ct_hex, len(ct_hex)
        except Exception:
            continue

def gen_rsa(_):
    while True:
        try:
            pt = get_random_plaintext()
            key_size = random.choice([1024, 2048])
            key = random.choice(RSA_1024_POOL if key_size == 1024 else RSA_2048_POOL)
            
            cipher = PKCS1_OAEP.new(key.publickey())
            
            # Max plaintext length for RSA-OAEP with SHA-1 is key_size/8 - 2*hashLen - 2
            # For 1024-bit (128 bytes): 128 - 2*20 - 2 = 86 bytes
            # For 2048-bit (256 bytes): 256 - 2*20 - 2 = 214 bytes
            max_chunk = (key_size // 8) - 42 
            
            ct_bytes = b""
            for i in range(0, len(pt), max_chunk):
                chunk = pt[i:i+max_chunk]
                ct_bytes += cipher.encrypt(chunk)
                
            ct_hex = ct_bytes.hex().upper()
            return "RSA", ct_hex, len(ct_hex)
        except Exception:
            continue

def gen_sha256(_):
    pt = get_random_plaintext()
    h = SHA256.new(pt)
    ct_hex = h.hexdigest().upper()
    return "SHA256", ct_hex, len(ct_hex)

def gen_md5(_):
    pt = get_random_plaintext()
    h = MD5.new(pt)
    ct_hex = h.hexdigest().upper()
    return "MD5", ct_hex, len(ct_hex)

def gen_base64(_):
    pt = get_random_plaintext()
    ct_b64 = b64encode(pt).decode('utf-8')
    return "Base64", ct_b64, len(ct_b64)

GENERATORS = {
    "AES": gen_aes,
    "DES": gen_des,
    "RSA": gen_rsa,
    "SHA256": gen_sha256,
    "MD5": gen_md5,
    "Base64": gen_base64
}

def worker(algo, chunk_size):
    """Worker function for multiprocessing pool"""
    func = GENERATORS[algo]
    results = []
    for _ in range(chunk_size):
        results.append(func(None))
    return results

def main():
    os.makedirs("datasets", exist_ok=True)
    
    # We want 15,000 unique samples. We'll generate 16,000 to account for potential duplicates (especially in short plaintexts, though rare here)
    TARGET_PER_ALGO = 16000 
    
    all_data = []
    
    print("\n🚀 Beginning massively parallel generation...")
    # Use pool to maximize CPU usage
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    for algo in TARGET_ALGOS:
        print(f"Generating {TARGET_PER_ALGO} raw samples for {algo}...")
        
        chunk_size = 1000
        num_chunks = TARGET_PER_ALGO // chunk_size
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Map the worker across chunks
            chunk_results = list(tqdm(
                pool.imap(partial(worker, algo), [chunk_size] * num_chunks), 
                total=num_chunks, 
                desc=algo
            ))
            
        # Flatten the list of lists
        algo_data = [item for sublist in chunk_results for item in sublist]
        all_data.extend(algo_data)
        
    print("\n✅ Generation Complete. Deduplicating and processing...")
    
    df = pd.DataFrame(all_data, columns=["Label", "Ciphertext", "Length"])
    
    # Drop exact duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=["Ciphertext"], inplace=True)
    print(f"Dropped {initial_len - len(df)} duplicate payloads.")
    
    # Downsample to exactly 15,000 per class to guarantee perfect balance
    balanced_dfs = []
    for algo in TARGET_ALGOS:
        algo_df = df[df["Label"] == algo]
        if len(algo_df) < SAMPLES_PER_ALGO:
            print(f"⚠️ WARNING: {algo} only has {len(algo_df)} unique samples (Target: {SAMPLES_PER_ALGO})")
            balanced_dfs.append(algo_df)
        else:
            balanced_dfs.append(algo_df.sample(SAMPLES_PER_ALGO, random_state=42))
            
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Final Dataset Distribution:")
    print(final_df["Label"].value_counts())
    
    print("\n✂️ Splitting into 80/10/10 (Train/Val/Test)...")
    # First split: 80% Train, 20% Temp
    train_df, temp_df = train_test_split(final_df, test_size=0.20, random_state=42, stratify=final_df["Label"])
    
    # Second split: 50% Val, 50% Test from the Temp (which is 20% of total) -> 10% each
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["Label"])
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    # Save to disk
    train_file = "datasets/train_15k.csv"
    val_file = "datasets/val_15k.csv"
    test_file = "datasets/test_15k.csv"
    
    print("\n💾 Saving CSV files...")
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    print("Done!")

if __name__ == "__main__":
    # Required for multiprocessing on macOS
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    main()
