#!/usr/bin/env python3
"""
Optimized Dataset Generator for Cryptographic Algorithm Identification

Generates balanced ciphertext samples for:
- Symmetric: AES, DES, Blowfish
- As 3DES,ymmetric: RSA
- Hashing: MD5, SHA1, SHA256
- Encoding: Base64, Hex

Each sample contains ONLY the ciphertext - no metadata leakage.
"""

import os
import sys
import random
import string
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

ALGORITHMS = ['AES', 'DES', '3DES', 'Blowfish', 'RSA', 'MD5', 'SHA1', 'SHA256', 'Base64', 'Hex']
DEFAULT_SAMPLES_PER_ALGO = 5000


def generate_random_plaintext(min_len: int = 16, max_len: int = 256) -> bytes:
    """Generate random plaintext of variable length."""
    length = random.randint(min_len, max_len)
    return os.urandom(length)


def generate_text_plaintext(min_len: int = 16, max_len: int = 256) -> bytes:
    """Generate text-like plaintext."""
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(chars) for _ in range(length)).encode('utf-8')


def encrypt_aes(plaintext: bytes) -> bytes:
    """AES-CBC encryption."""
    key_size = random.choice([16, 24, 32])
    key = os.urandom(key_size)
    iv = os.urandom(16)
    
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(plaintext, AES.block_size)
    return iv + cipher.encrypt(padded)


def encrypt_des(plaintext: bytes) -> bytes:
    """DES-CBC encryption."""
    key = os.urandom(8)
    iv = os.urandom(8)
    
    from Crypto.Cipher import DES
    from Crypto.Util.Padding import pad
    
    cipher = DES.new(key, DES.MODE_CBC, iv)
    padded = pad(plaintext, DES.block_size)
    return iv + cipher.encrypt(padded)


def encrypt_3des(plaintext: bytes) -> bytes:
    """Triple DES encryption."""
    while True:
        key = os.urandom(24)
        try:
            from Crypto.Cipher import DES3
            DES3.adjust_key_parity(key)
            break
        except ValueError:
            continue
    
    iv = os.urandom(8)
    cipher = DES3.new(key, DES3.MODE_CBC, iv)
    from Crypto.Util.Padding import pad
    padded = pad(plaintext, DES3.block_size)
    return iv + cipher.encrypt(padded)


def encrypt_blowfish(plaintext: bytes) -> bytes:
    """Blowfish-CBC encryption."""
    key_size = random.randint(16, 56)
    key = os.urandom(key_size)
    iv = os.urandom(8)
    
    from Crypto.Cipher import Blowfish
    from Crypto.Util.Padding import pad
    
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    padded = pad(plaintext, Blowfish.block_size)
    return iv + cipher.encrypt(padded)


def encrypt_rsa(plaintext: bytes) -> bytes:
    """RSA-OAEP encryption."""
    key_size = random.choice([2048, 4096])
    
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_OAEP
    
    key = RSA.generate(key_size)
    cipher = PKCS1_OAEP.new(key.publickey())
    
    max_chunk = key_size // 8 - 66
    ciphertext = b''
    for i in range(0, len(plaintext), max_chunk):
        chunk = plaintext[i:i + max_chunk]
        ciphertext += cipher.encrypt(chunk)
    
    return ciphertext


def hash_md5(plaintext: bytes) -> bytes:
    """MD5 hash."""
    from Crypto.Hash import MD5
    return MD5.new(plaintext).digest()


def hash_sha1(plaintext: bytes) -> bytes:
    """SHA1 hash."""
    from Crypto.Hash import SHA1
    return SHA1.new(plaintext).digest()


def hash_sha256(plaintext: bytes) -> bytes:
    """SHA256 hash."""
    from Crypto.Hash import SHA256
    return SHA256.new(plaintext).digest()


def encode_base64(plaintext: bytes) -> bytes:
    """Base64 encoding."""
    import base64
    return base64.b64encode(plaintext)


def encode_hex(plaintext: bytes) -> bytes:
    """Hex encoding."""
    return plaintext.hex().encode('utf-8')


ENCRYPT_FUNCTIONS = {
    'AES': encrypt_aes,
    'DES': encrypt_des,
    '3DES': encrypt_3des,
    'Blowfish': encrypt_blowfish,
    'RSA': encrypt_rsa,
    'MD5': hash_md5,
    'SHA1': hash_sha1,
    'SHA256': hash_sha256,
    'Base64': encode_base64,
    'Hex': encode_hex
}


def generate_sample(algorithm: str) -> Tuple[str, str]:
    """Generate a single ciphertext sample. Returns (algorithm, hex_ciphertext)."""
    # Mix random binary and text plaintexts
    if random.random() < 0.5:
        plaintext = generate_random_plaintext(16, 256)
    else:
        plaintext = generate_text_plaintext(16, 256)
    
    output = ENCRYPT_FUNCTIONS[algorithm](plaintext)
    return algorithm, output.hex().upper()


def generate_dataset_worker(args: Tuple) -> dict:
    """Worker function for parallel generation."""
    algo, idx = args
    try:
        algo, ciphertext = generate_sample(algo)
        return {'Algorithm': algo, 'Ciphertext': ciphertext}
    except Exception as e:
        return None


def generate_dataset(
    samples_per_algo: int = DEFAULT_SAMPLES_PER_ALGO,
    output_path: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate complete dataset with balanced samples for each algorithm."""
    
    tasks = []
    for algo in ALGORITHMS:
        for _ in range(samples_per_algo):
            tasks.append((algo, _))
    
    random.shuffle(tasks)
    
    data = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(generate_dataset_worker, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)
            
            completed += 1
            if verbose and completed % 1000 == 0:
                print(f"  Generated {completed}/{len(tasks)} samples...")
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nDataset saved to: {output_path}")
    
    if verbose:
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        for algo in ALGORITHMS:
            count = len(df[df['Algorithm'] == algo])
            print(f"    {algo}: {count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate cryptographic dataset')
    parser.add_argument('-n', '--samples', type=int, default=DEFAULT_SAMPLES_PER_ALGO,
                       help=f'Samples per algorithm (default: {DEFAULT_SAMPLES_PER_ALGO})')
    parser.add_argument('-o', '--output', type=str, default='datasets/crypto_algorithms.csv',
                       help='Output CSV path')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_dataset(
        samples_per_algo=args.samples,
        output_path=str(output_path),
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
