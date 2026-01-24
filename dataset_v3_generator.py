from Crypto.Cipher import AES, DES3, Blowfish, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA, ECC
from Crypto.Signature import DSS
from Crypto.Random import get_random_bytes
from Crypto.Util import Padding
from Crypto.Hash import SHA256
from base64 import b16encode
from cryptography.hazmat.primitives.asymmetric import dh
import csv, math, os, time
from tqdm import tqdm
import numpy as np

# ---------- Helper Functions ----------
def shannon_entropy(s):
    """Compute Shannon entropy of string s"""
    if not s:
        return 0.0
    probs = [s.count(c)/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def hex_ratio(s):
    """Calculate ratio of hex characters in ciphertext"""
    return sum(1 for c in s if c in "0123456789abcdefABCDEF") / len(s)

def byte_stats(s):
    """Compute mean and std of ciphertext bytes"""
    try:
        b = bytes.fromhex(s)
        arr = np.frombuffer(b, dtype=np.uint8)
        return arr.mean(), arr.std()
    except Exception:
        return 0.0, 0.0

# ---------- Crypto functions ----------
def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, AES.block_size))
    return b16encode(cipher.iv + ct_bytes).decode(), len(key), AES.block_size, len(cipher.iv), "CBC"

def encrypt_des(plaintext, key):
    cipher = DES.new(key, DES.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, DES.block_size))
    return b16encode(cipher.iv + ct_bytes).decode(), len(key), DES.block_size, len(cipher.iv), "CBC"

def encrypt_3des(plaintext, key):
    key = DES3.adjust_key_parity(key)
    cipher = DES3.new(key, DES3.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, DES3.block_size))
    return b16encode(cipher.iv + ct_bytes).decode(), len(key), DES3.block_size, len(cipher.iv), "CBC"

def encrypt_blowfish(plaintext, key):
    cipher = Blowfish.new(key, Blowfish.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, Blowfish.block_size))
    return b16encode(cipher.iv + ct_bytes).decode(), len(key), Blowfish.block_size, len(cipher.iv), "CBC"

def encrypt_rsa(plaintext, pubkey):
    cipher = PKCS1_OAEP.new(pubkey)
    ciphertext = cipher.encrypt(plaintext)
    return b16encode(ciphertext).decode(), 2048, 0, 0, "Asymmetric"

def sign_ecc(plaintext, privkey):
    h = SHA256.new(plaintext)
    signer = DSS.new(privkey, 'fips-186-3')
    sig = signer.sign(h)
    return b16encode(sig).decode(), 256, 0, 0, "Asymmetric"

def generate_dh_shared_key(params):
    priv = params.generate_private_key()
    peer = params.generate_private_key()
    shared = priv.exchange(peer.public_key())
    return b16encode(shared).decode(), 2048, 0, 0, "KeyExchange"

# ---------- Dataset Generation ----------
def generate_dataset_v3(samples_per_algo=400):
    dataset = []

    rsa_key = RSA.generate(2048)
    rsa_pub = rsa_key.publickey()
    ecc_priv = ECC.generate(curve='P-256')
    dh_params = dh.generate_parameters(generator=2, key_size=2048)

    total_samples = samples_per_algo * 7
    print(f"\n🚀 Generating {samples_per_algo} samples per algorithm (Total ≈ {total_samples})")
    start_time = time.time()

    for i in tqdm(range(samples_per_algo), desc="Generating dataset", ncols=90):
        pt = get_random_bytes(32)

        # Symmetric algorithms (correct key sizes)
        for name, func, keysize in [
            ("AES", encrypt_aes, 16),
            ("DES", encrypt_des, 8),
            ("3DES", encrypt_3des, 24),
            ("Blowfish", encrypt_blowfish, 16),
        ]:
            key = get_random_bytes(keysize)
            ct, keylen, block, ivlen, mode = func(pt, key)
            ent = shannon_entropy(ct)
            hr = hex_ratio(ct)
            bm, bs = byte_stats(ct)
            dataset.append([name, ct, len(pt), keylen, block, ivlen, mode, ent, hr, bm, bs])

        # Asymmetric / Key Exchange algorithms
        for name, func in [
            ("RSA", lambda pt: encrypt_rsa(pt, rsa_pub)),
            ("ECC", lambda pt: sign_ecc(pt, ecc_priv)),
            ("Diffie-Hellman", lambda pt: generate_dh_shared_key(dh_params)),
        ]:
            ct, keylen, block, ivlen, mode = func(pt)
            ent = shannon_entropy(ct)
            hr = hex_ratio(ct)
            bm, bs = byte_stats(ct)
            dataset.append([name, ct, len(pt), keylen, block, ivlen, mode, ent, hr, bm, bs])

    # Save dataset
    os.makedirs("datasets", exist_ok=True)
    output_file = "datasets/dataset_v3.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Ciphertext", "PlaintextLen", "KeyLen", "BlockSize", "IVLen", 
                         "Mode", "Entropy", "HexRatio", "ByteMean", "ByteStd"])
        writer.writerows(dataset)

    elapsed = time.time() - start_time
    print(f"\n✅ Dataset generated successfully: {output_file}")
    print(f"📦 Total Samples: {len(dataset)}")
    print(f"⏱️ Time taken: {elapsed:.2f} seconds (~{elapsed/60:.1f} min)")

if __name__ == "__main__":
    generate_dataset_v3(400)
