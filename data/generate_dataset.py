#!/usr/bin/env python3
"""
Dataset Generator for Cryptographic Algorithm Identification

Generates balanced ciphertext samples for:
- AES (128/192/256-bit keys)
- DES (56-bit key)
- 3DES (168-bit key)
- Blowfish (variable key)
- RSA (2048/4096-bit keys)
- ECC (secp256r1)

Each sample contains ONLY the ciphertext - no metadata leakage.
"""

import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Crypto.Cipher import AES, DES, DES3, Blowfish
from Crypto.PublicKey import RSA, ECC
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Hash import SHA256
from Crypto.Signature import DSS

# Constants
ALGORITHMS = ['AES', 'DES', '3DES', 'Blowfish', 'RSA', 'ECC']
DEFAULT_SAMPLES_PER_ALGO = 2000


def generate_random_plaintext(min_len: int = 16, max_len: int = 256) -> bytes:
    """Generate random plaintext of variable length."""
    length = random.randint(min_len, max_len)
    return get_random_bytes(length)


def encrypt_aes(plaintext: bytes) -> bytes:
    """
    Encrypt using AES in CBC mode.
    Key sizes: 16 (128-bit), 24 (192-bit), or 32 (256-bit) bytes.
    """
    key_size = random.choice([16, 24, 32])
    key = get_random_bytes(key_size)
    iv = get_random_bytes(16)  # AES block size is 16 bytes
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(plaintext, AES.block_size)
    ciphertext = cipher.encrypt(padded)
    
    # Return IV + ciphertext (as would be transmitted)
    return iv + ciphertext


def encrypt_des(plaintext: bytes) -> bytes:
    """
    Encrypt using DES in CBC mode.
    Key size: 8 bytes (56-bit effective).
    """
    key = get_random_bytes(8)
    iv = get_random_bytes(8)  # DES block size is 8 bytes
    
    cipher = DES.new(key, DES.MODE_CBC, iv)
    padded = pad(plaintext, DES.block_size)
    ciphertext = cipher.encrypt(padded)
    
    return iv + ciphertext


def encrypt_3des(plaintext: bytes) -> bytes:
    """
    Encrypt using Triple DES in CBC mode.
    Key size: 24 bytes (168-bit effective).
    """
    # Ensure proper 3DES key (avoid weak keys)
    while True:
        key = get_random_bytes(24)
        try:
            DES3.adjust_key_parity(key)
            break
        except ValueError:
            continue
    
    iv = get_random_bytes(8)  # 3DES block size is 8 bytes
    
    cipher = DES3.new(key, DES3.MODE_CBC, iv)
    padded = pad(plaintext, DES3.block_size)
    ciphertext = cipher.encrypt(padded)
    
    return iv + ciphertext


def encrypt_blowfish(plaintext: bytes) -> bytes:
    """
    Encrypt using Blowfish in CBC mode.
    Key size: 4-56 bytes (variable).
    """
    key_size = random.randint(16, 56)  # Use reasonable key sizes
    key = get_random_bytes(key_size)
    iv = get_random_bytes(8)  # Blowfish block size is 8 bytes
    
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    padded = pad(plaintext, Blowfish.block_size)
    ciphertext = cipher.encrypt(padded)
    
    return iv + ciphertext


def encrypt_rsa(plaintext: bytes) -> bytes:
    """
    Encrypt using RSA-OAEP.
    Key sizes: 2048 or 4096 bits.
    
    Note: RSA can only encrypt small amounts of data directly.
    For longer plaintexts, we encrypt in chunks or use hybrid encryption.
    """
    key_size = random.choice([2048, 4096])
    key = RSA.generate(key_size)
    
    cipher = PKCS1_OAEP.new(key.publickey())
    
    # RSA-OAEP max plaintext size = key_size/8 - 2*hash_size - 2
    # For SHA-256: max = key_size/8 - 66
    max_chunk = key_size // 8 - 66
    
    # Encrypt in chunks if needed
    ciphertext = b''
    for i in range(0, len(plaintext), max_chunk):
        chunk = plaintext[i:i + max_chunk]
        ciphertext += cipher.encrypt(chunk)
    
    return ciphertext


def encrypt_ecc(plaintext: bytes) -> bytes:
    """
    Generate ECC-based output.
    
    ECC is typically used for key exchange/signatures, not direct encryption.
    We simulate ECC output by:
    1. Generating an ECDSA signature
    2. Combining with a random shared secret simulation
    
    This produces output characteristic of ECC operations.
    """
    # Generate ECC key pair
    key = ECC.generate(curve='P-256')
    
    # Create ECDSA signature of the plaintext hash
    h = SHA256.new(plaintext)
    signer = DSS.new(key, 'fips-186-3')
    signature = signer.sign(h)
    
    # Simulate ECIES-like output: ephemeral public key point + encrypted data
    # Generate another ephemeral key to simulate key exchange
    ephemeral = ECC.generate(curve='P-256')
    
    # Export public key point (this is what would be transmitted)
    pub_point = ephemeral.public_key().export_key(format='DER')
    
    # Combine signature + public key point (characteristic ECC output)
    return signature + pub_point


# Encryption function dispatch
ENCRYPT_FUNCTIONS = {
    'AES': encrypt_aes,
    'DES': encrypt_des,
    '3DES': encrypt_3des,
    'Blowfish': encrypt_blowfish,
    'RSA': encrypt_rsa,
    'ECC': encrypt_ecc
}


def generate_sample(algorithm: str, min_pt_len: int = 16, max_pt_len: int = 256) -> str:
    """
    Generate a single ciphertext sample.
    
    Returns:
        Hex-encoded ciphertext string
    """
    plaintext = generate_random_plaintext(min_pt_len, max_pt_len)
    encrypt_func = ENCRYPT_FUNCTIONS[algorithm]
    ciphertext = encrypt_func(plaintext)
    return ciphertext.hex().upper()


def generate_dataset(
    samples_per_algo: int = DEFAULT_SAMPLES_PER_ALGO,
    output_path: str = None,
    min_pt_len: int = 16,
    max_pt_len: int = 256,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate complete dataset with balanced samples for each algorithm.
    
    Args:
        samples_per_algo: Number of samples per algorithm
        output_path: Path to save CSV (optional)
        min_pt_len: Minimum plaintext length
        max_pt_len: Maximum plaintext length
        verbose: Print progress
        
    Returns:
        DataFrame with columns ['Algorithm', 'Ciphertext']
    """
    data = []
    
    for algo in ALGORITHMS:
        if verbose:
            print(f"Generating {samples_per_algo} samples for {algo}...")
        
        for i in range(samples_per_algo):
            try:
                ciphertext = generate_sample(algo, min_pt_len, max_pt_len)
                data.append({
                    'Algorithm': algo,
                    'Ciphertext': ciphertext
                })
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to generate sample {i} for {algo}: {e}")
                continue
            
            if verbose and (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{samples_per_algo} samples")
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nDataset saved to: {output_path}")
    
    if verbose:
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Samples per algorithm:")
        for algo in ALGORITHMS:
            count = len(df[df['Algorithm'] == algo])
            print(f"    {algo}: {count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate ciphertext dataset for cryptographic algorithm identification'
    )
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=DEFAULT_SAMPLES_PER_ALGO,
        help=f'Number of samples per algorithm (default: {DEFAULT_SAMPLES_PER_ALGO})'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/crypto_dataset.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=16,
        help='Minimum plaintext length (default: 16)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum plaintext length (default: 256)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generate_dataset(
        samples_per_algo=args.samples,
        output_path=str(output_path),
        min_pt_len=args.min_length,
        max_pt_len=args.max_length,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
