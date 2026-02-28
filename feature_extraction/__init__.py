"""
Feature Extraction Module for Cryptographic Algorithm Identification

Extracts statistical features from ciphertext without using any metadata.
All features are derived purely from the ciphertext bytes/hex string.
"""

import math
import re
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Union

# Regex to detect valid hex strings
HEX_PATTERN = re.compile(r'^[0-9a-fA-F]+$')


def is_hex_string(s: str) -> bool:
    """Check if string is valid hexadecimal."""
    return bool(HEX_PATTERN.match(s)) and len(s) % 2 == 0


def to_bytes(ciphertext: Union[str, bytes]) -> bytes:
    """
    Convert ciphertext to bytes.
    Handles hex strings, raw strings, and bytes.
    """
    if isinstance(ciphertext, bytes):
        return ciphertext
    
    s = str(ciphertext).strip()
    
    # Try hex decoding first
    if is_hex_string(s):
        try:
            return bytes.fromhex(s)
        except ValueError:
            pass
    
    # Fall back to UTF-8 encoding
    return s.encode('utf-8', errors='replace')


def shannon_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of byte data.
    
    Higher entropy (~8.0) indicates more randomness (typical of strong encryption).
    Lower entropy indicates patterns or structure.
    
    Returns:
        Entropy value between 0.0 and 8.0
    """
    if not data:
        return 0.0
    
    freq = Counter(data)
    length = len(data)
    
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    
    return entropy


def byte_frequency_distribution(data: bytes) -> np.ndarray:
    """
    Calculate normalized frequency distribution over all 256 byte values.
    
    Returns:
        256-dimensional numpy array of frequencies (sums to 1.0)
    """
    freq = np.zeros(256, dtype=np.float32)
    
    if not data:
        return freq
    
    for byte in data:
        freq[byte] += 1
    
    return freq / len(data)


def byte_statistics(data: bytes) -> Dict[str, float]:
    """
    Calculate statistical measures of byte values.
    
    Returns:
        Dictionary with mean, std, variance, skewness, kurtosis, min, max
    """
    if not data:
        return {
            'mean': 0.0, 'std': 0.0, 'variance': 0.0,
            'skewness': 0.0, 'kurtosis': 0.0,
            'min': 0.0, 'max': 0.0
        }
    
    arr = np.array(list(data), dtype=np.float64)
    
    mean = np.mean(arr)
    std = np.std(arr)
    variance = np.var(arr)
    
    # Skewness: measure of asymmetry
    if std > 0:
        skewness = np.mean(((arr - mean) / std) ** 3)
    else:
        skewness = 0.0
    
    # Kurtosis: measure of tailedness (excess kurtosis, normal = 0)
    if std > 0:
        kurtosis = np.mean(((arr - mean) / std) ** 4) - 3.0
    else:
        kurtosis = 0.0
    
    return {
        'mean': float(mean),
        'std': float(std),
        'variance': float(variance),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }


def hex_char_ratio(ciphertext: str) -> float:
    """
    Calculate ratio of hexadecimal characters in the string.
    Useful for detecting encoding format.
    """
    if not ciphertext:
        return 0.0
    
    hex_chars = set('0123456789abcdefABCDEF')
    hex_count = sum(1 for c in ciphertext if c in hex_chars)
    
    return hex_count / len(ciphertext)


def ngram_frequencies(data: bytes, n: int = 2) -> np.ndarray:
    """
    Calculate n-gram byte frequencies.
    
    For n=2 (bigrams), returns 65536-dimensional sparse representation.
    We use a reduced representation: top-k most common + statistical summary.
    
    Returns:
        Array of n-gram statistics
    """
    if len(data) < n:
        return np.zeros(10, dtype=np.float32)
    
    ngrams = []
    for i in range(len(data) - n + 1):
        ngram = tuple(data[i:i+n])
        ngrams.append(ngram)
    
    freq = Counter(ngrams)
    total = len(ngrams)
    
    # Calculate statistics over n-gram frequencies
    freqs = np.array(list(freq.values()), dtype=np.float64) / total
    
    return np.array([
        len(freq),                    # Number of unique n-grams
        np.mean(freqs),               # Mean frequency
        np.std(freqs),                # Std of frequencies
        np.max(freqs),                # Max frequency (most common)
        np.min(freqs),                # Min frequency
        np.median(freqs),             # Median frequency
        np.percentile(freqs, 25),     # 25th percentile
        np.percentile(freqs, 75),     # 75th percentile
        shannon_entropy(data) / 8.0,  # Normalized entropy
        len(data) / 1024.0            # Normalized length (KB)
    ], dtype=np.float32)


def block_pattern_features(data: bytes) -> np.ndarray:
    """
    Extract features related to block cipher patterns.
    
    Different algorithms have different block sizes:
    - DES/3DES/Blowfish: 8 bytes (64 bits)
    - AES: 16 bytes (128 bits)
    - RSA/ECC: Variable (typically 128-512 bytes)
    """
    if len(data) < 8:
        return np.zeros(8, dtype=np.float32)
    
    features = []
    
    # Check for repeating blocks at different sizes
    for block_size in [8, 16, 32, 64]:
        if len(data) >= block_size * 2:
            blocks = [data[i:i+block_size] for i in range(0, len(data) - block_size + 1, block_size)]
            unique_blocks = len(set(blocks))
            total_blocks = len(blocks)
            features.append(unique_blocks / total_blocks if total_blocks > 0 else 1.0)
        else:
            features.append(1.0)
    
    # Check if length is multiple of common block sizes
    features.append(1.0 if len(data) % 8 == 0 else 0.0)
    features.append(1.0 if len(data) % 16 == 0 else 0.0)
    features.append(1.0 if len(data) % 64 == 0 else 0.0)
    features.append(1.0 if len(data) % 128 == 0 else 0.0)
    
    return np.array(features, dtype=np.float32)


def extract_features(ciphertext: Union[str, bytes]) -> np.ndarray:
    """
    Extract all features from ciphertext for ML model input.
    
    This is the main feature extraction function that combines all
    individual feature extractors into a single feature vector.
    
    Args:
        ciphertext: Raw bytes or hex string of ciphertext
        
    Returns:
        Feature vector as numpy array (total: 285 features)
    """
    # Convert to bytes
    data = to_bytes(ciphertext)
    original_str = str(ciphertext) if isinstance(ciphertext, str) else ciphertext.hex()
    
    features = []
    
    # 1. Shannon entropy (1 feature)
    features.append(shannon_entropy(data))
    
    # 2. Hex character ratio (1 feature)
    features.append(hex_char_ratio(original_str))
    
    # 3. Byte statistics (7 features)
    stats = byte_statistics(data)
    features.extend([
        stats['mean'],
        stats['std'],
        stats['variance'],
        stats['skewness'],
        stats['kurtosis'],
        stats['min'],
        stats['max']
    ])
    
    # 4. Byte frequency distribution (256 features)
    freq_dist = byte_frequency_distribution(data)
    features.extend(freq_dist.tolist())
    
    # 5. N-gram features (10 features)
    ngram_feats = ngram_frequencies(data, n=2)
    features.extend(ngram_feats.tolist())
    
    # 6. Block pattern features (8 features)
    block_feats = block_pattern_features(data)
    features.extend(block_feats.tolist())
    
    # 7. Length-based features (2 features)
    features.append(len(data))  # Raw length
    features.append(math.log1p(len(data)))  # Log-scaled length
    
    return np.array(features, dtype=np.float32)


def extract_raw_bytes(ciphertext: Union[str, bytes], max_length: int = 512) -> np.ndarray:
    """
    Extract raw byte sequence for CNN input.
    
    Pads or truncates to fixed length for neural network input.
    
    Args:
        ciphertext: Raw bytes or hex string
        max_length: Fixed output length (default 512)
        
    Returns:
        Numpy array of shape (max_length,) with byte values 0-255
    """
    data = to_bytes(ciphertext)
    
    # Truncate if too long
    if len(data) > max_length:
        data = data[:max_length]
    
    # Create padded array
    result = np.zeros(max_length, dtype=np.uint8)
    result[:len(data)] = list(data)
    
    return result


def get_feature_names() -> List[str]:
    """
    Get names of all features for interpretability.
    
    Returns:
        List of feature names matching extract_features output
    """
    names = ['entropy', 'hex_ratio']
    names.extend(['byte_mean', 'byte_std', 'byte_var', 'byte_skew', 'byte_kurt', 'byte_min', 'byte_max'])
    names.extend([f'byte_freq_{i}' for i in range(256)])
    names.extend(['ngram_unique', 'ngram_mean', 'ngram_std', 'ngram_max', 'ngram_min',
                  'ngram_median', 'ngram_p25', 'ngram_p75', 'norm_entropy', 'norm_length'])
    names.extend(['block8_unique', 'block16_unique', 'block32_unique', 'block64_unique',
                  'mod8', 'mod16', 'mod64', 'mod128'])
    names.extend(['raw_length', 'log_length'])
    
    return names


# Feature vector size for reference
FEATURE_SIZE = 285
RAW_SEQUENCE_LENGTH = 512
