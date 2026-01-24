from Crypto.Cipher import AES, DES3, Blowfish, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA, ECC
from Crypto.Signature import DSS
from Crypto.Random import get_random_bytes
from Crypto.Util import Padding
from Crypto.Hash import SHA256
from base64 import b16encode
from cryptography.hazmat.primitives.asymmetric import dh
from tqdm import tqdm
import csv
import os


# -------------------------------
# Encryption / Signing Functions
# -------------------------------

def encrypt_aes(plaintext):
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, AES.block_size))
    return b16encode(cipher.iv + ct_bytes).decode()


def encrypt_des(plaintext):
    key = get_random_bytes(8)
    cipher = DES.new(key, DES.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, DES.block_size))
    return b16encode(cipher.iv + ct_bytes).decode()


def encrypt_3des(plaintext):
    key = DES3.adjust_key_parity(get_random_bytes(24))
    cipher = DES3.new(key, DES3.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, DES3.block_size))
    return b16encode(cipher.iv + ct_bytes).decode()


def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)
    cipher = Blowfish.new(key, Blowfish.MODE_CBC)
    ct_bytes = cipher.encrypt(Padding.pad(plaintext, Blowfish.block_size))
    return b16encode(cipher.iv + ct_bytes).decode()


def encrypt_rsa(plaintext, rsa_pub_key):
    cipher = PKCS1_OAEP.new(rsa_pub_key)
    ciphertext = cipher.encrypt(plaintext)
    return b16encode(ciphertext).decode()


def sign_ecc(plaintext, ecc_priv_key):
    h = SHA256.new(plaintext)
    signer = DSS.new(ecc_priv_key, 'fips-186-3')
    signature = signer.sign(h)
    return b16encode(signature).decode()


def generate_dh_shared_key(parameters):
    private_key = parameters.generate_private_key()
    peer_private_key = parameters.generate_private_key()
    shared_key = private_key.exchange(peer_private_key.public_key())
    return b16encode(shared_key).decode()


# -------------------------------
# Dataset Generation
# -------------------------------

def generate_dataset(num_samples_per_algo=1000, output_file='dataset2.csv'):
    dataset = []

    # Pre-generate expensive keys only once
    rsa_key = RSA.generate(2048)
    rsa_pub_key = rsa_key.publickey()
    ecc_priv_key = ECC.generate(curve='P-256')
    parameters = dh.generate_parameters(generator=2, key_size=2048)

    algorithms = [
        ("AES", encrypt_aes),
        ("DES", encrypt_des),
        ("3DES", encrypt_3des),
        ("Blowfish", encrypt_blowfish),
    ]

    # Use tqdm for progress bar
    print(f"\n🚀 Generating dataset with {num_samples_per_algo} samples per algorithm...\n")

    for i in tqdm(range(num_samples_per_algo), desc="Generating Samples", ncols=80):
        pt = get_random_bytes(32)  # random plaintext (32 bytes)

        # Symmetric Algorithms
        for name, func in algorithms:
            dataset.append([name, func(pt)])

        # Asymmetric / Key Exchange
        dataset.append(['RSA', encrypt_rsa(pt, rsa_pub_key)])
        dataset.append(['ECC', sign_ecc(pt, ecc_priv_key)])
        dataset.append(['Diffie-Hellman', generate_dh_shared_key(parameters)])

    # Write dataset to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Ciphertext'])
        writer.writerows(dataset)

    print(f"\n✅ Dataset generated successfully: {output_file}")
    print(f"📦 Total Samples: {len(dataset)}")


# -------------------------------
# Main Entry
# -------------------------------

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    output_path = os.path.join("datasets", "dataset2.csv")
    generate_dataset(num_samples_per_algo=1000, output_file=output_path)
