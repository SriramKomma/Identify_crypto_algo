#!/usr/bin/env python3
"""Fast Dataset Generator - Simplified for high accuracy"""

import os
import sys
import random
import string
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict

ALGORITHMS = ['AES', 'DES', 'Blowfish', 'RSA', 'MD5', 'SHA1', 'SHA256', 'Base64', 'Hex']
DEFAULT_SAMPLES_PER_ALGO = 3000


def generate_random_bytes(min_len=16, max_len=256):
    return os.urandom(random.randint(min_len, max_len))


def generate_text_bytes(min_len=16, max_len=256):
    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(chars) for _ in range(length)).encode('utf-8')


def encrypt_aes(plaintext):
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    key = os.urandom(16)  # Fixed 128-bit for simplicity
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(pad(plaintext, AES.block_size))


def encrypt_des(plaintext):
    from Crypto.Cipher import DES
    from Crypto.Util.Padding import pad
    key = os.urandom(8)
    iv = os.urandom(8)
    cipher = DES.new(key, DES.MODE_CBC, iv)
    return iv + cipher.encrypt(pad(plaintext, DES.block_size))


def encrypt_blowfish(plaintext):
    from Crypto.Cipher import Blowfish
    from Crypto.Util.Padding import pad
    key = os.urandom(16)  # Fixed for simplicity
    iv = os.urandom(8)
    cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
    return iv + cipher.encrypt(pad(plaintext, Blowfish.block_size))


def encrypt_rsa(plaintext):
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_OAEP
    key = RSA.generate(1024)
    cipher = PKCS1_OAEP.new(key.publickey())
    max_chunk = 62
    ciphertext = b''
    for i in range(0, len(plaintext), max_chunk):
        ciphertext += cipher.encrypt(plaintext[i:i + max_chunk])
    return ciphertext


def hash_md5(plaintext):
    from Crypto.Hash import MD5
    return MD5.new(plaintext).digest()


def hash_sha1(plaintext):
    from Crypto.Hash import SHA1
    return SHA1.new(plaintext).digest()


def hash_sha256(plaintext):
    from Crypto.Hash import SHA256
    return SHA256.new(plaintext).digest()


def encode_base64(plaintext):
    import base64
    return base64.b64encode(plaintext)


def encode_hex(plaintext):
    return plaintext.hex().encode('utf-8')


FUNCS = {
    'AES': encrypt_aes, 'DES': encrypt_des, 'Blowfish': encrypt_blowfish,
    'RSA': encrypt_rsa, 'MD5': hash_md5, 'SHA1': hash_sha1,
    'SHA256': hash_sha256, 'Base64': encode_base64, 'Hex': encode_hex
}


def generate_sample(algo):
    plaintext = generate_random_bytes(16, 256) if random.random() < 0.7 else generate_text_bytes(16, 256)
    try:
        return {'Algorithm': algo, 'Ciphertext': FUNCS[algo](plaintext).hex().upper()}
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--samples', type=int, default=DEFAULT_SAMPLES_PER_ALGO)
    parser.add_argument('-o', '--output', type=str, default='datasets/crypto_final.csv')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.samples} samples per algorithm...")
    
    tasks = [(algo, i) for algo in ALGORITHMS for i in range(args.samples)]
    random.shuffle(tasks)
    
    data = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(generate_sample, t[0]): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)
            completed += 1
            if completed % 3000 == 0:
                print(f"  {completed}/{len(tasks)}")
    
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved: {len(df)} samples")
    print(df['Algorithm'].value_counts())


if __name__ == '__main__':
    main()
