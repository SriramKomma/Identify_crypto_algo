"""
Enhanced Feature Extraction Module for Cryptographic Algorithm Identification

Extracts statistical features from ciphertext without using any metadata.
Enhanced version with more discriminative features for block cipher identification.
"""

import math
import re
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Union
import zlib

HEX_PATTERN = re.compile(r'^[0-9a-fA-F]+$')


def is_hex_string(s: str) -> bool:
    return bool(HEX_PATTERN.match(s)) and len(s) % 2 == 0


def to_bytes(ciphertext: Union[str, bytes]) -> bytes:
    if isinstance(ciphertext, bytes):
        return ciphertext
    
    s = str(ciphertext).strip()
    
    if is_hex_string(s):
        try:
            return bytes.fromhex(s)
        except ValueError:
            pass
    
    return s.encode('utf-8', errors='replace')


def shannon_entropy(data: bytes) -> float:
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
    freq = np.zeros(256, dtype=np.float32)
    
    if not data:
        return freq
    
    for byte in data:
        freq[byte] += 1
    
    return freq / len(data)


def byte_statistics(data: bytes) -> Dict[str, float]:
    if not data:
        return {'mean': 0.0, 'std': 0.0, 'variance': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 'min': 0.0, 'max': 0.0}
    
    arr = np.array(list(data), dtype=np.float64)
    
    mean = np.mean(arr)
    std = np.std(arr)
    variance = np.var(arr)
    
    skewness = np.mean(((arr - mean) / std) ** 3) if std > 0 else 0.0
    kurtosis = (np.mean(((arr - mean) / std) ** 4) - 3.0) if std > 0 else 0.0
    
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
    if not ciphertext:
        return 0.0
    
    hex_chars = set('0123456789abcdefABCDEF')
    hex_count = sum(1 for c in ciphertext if c in hex_chars)
    
    return hex_count / len(ciphertext)


def ngram_frequencies(data: bytes, n: int = 2) -> np.ndarray:
    if len(data) < n:
        return np.zeros(10, dtype=np.float32)
    
    ngrams = [tuple(data[i:i+n]) for i in range(len(data) - n + 1)]
    freq = Counter(ngrams)
    total = len(ngrams)
    
    freqs = np.array(list(freq.values()), dtype=np.float64) / total
    
    return np.array([
        len(freq),
        np.mean(freqs),
        np.std(freqs),
        np.max(freqs),
        np.min(freqs),
        np.median(freqs),
        np.percentile(freqs, 25),
        np.percentile(freqs, 75),
        shannon_entropy(data) / 8.0,
        len(data) / 1024.0
    ], dtype=np.float32)


def block_pattern_features(data: bytes) -> np.ndarray:
    if len(data) < 8:
        return np.zeros(8, dtype=np.float32)
    
    features = []
    
    for block_size in [8, 16, 32, 64]:
        if len(data) >= block_size * 2:
            blocks = [data[i:i+block_size] for i in range(0, len(data) - block_size + 1, block_size)]
            unique_blocks = len(set(blocks))
            total_blocks = len(blocks)
            features.append(unique_blocks / total_blocks if total_blocks > 0 else 1.0)
        else:
            features.append(1.0)
    
    features.append(1.0 if len(data) % 8 == 0 else 0.0)
    features.append(1.0 if len(data) % 16 == 0 else 0.0)
    features.append(1.0 if len(data) % 64 == 0 else 0.0)
    features.append(1.0 if len(data) % 128 == 0 else 0.0)
    
    return np.array(features, dtype=np.float32)


def autocorrelation_features(data: bytes) -> np.ndarray:
    """Detect repeating patterns via autocorrelation."""
    if len(data) < 16:
        return np.zeros(5, dtype=np.float32)
    
    arr = np.array(list(data), dtype=np.float32)
    arr = arr - np.mean(arr)
    
    autocorr = np.correlate(arr, arr, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    
    return np.array([
        np.mean(autocorr[1:11]),
        np.max(autocorr[1:]),
        np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0,
        np.std(autocorr[:min(32, len(autocorr))]),
        float(any(autocorr[1:32] > 0.3))
    ], dtype=np.float32)


def byte_pair_features(data: bytes) -> np.ndarray:
    """Features based on byte pair patterns."""
    if len(data) < 2:
        return np.zeros(6, dtype=np.float32)
    
    pairs = Counter(zip(data[:-1], data[1:]))
    total = len(data) - 1
    
    pair_freqs = np.array(list(pairs.values()), dtype=np.float32) / total
    
    top_5 = sorted(pair_freqs, reverse=True)[:5]
    while len(top_5) < 5:
        top_5.append(0.0)
    
    return np.array([
        len(pairs) / 256.0,
        np.mean(pair_freqs),
        np.std(pair_freqs),
        np.max(pair_freqs),
        top_5[0],
        top_5[1] if len(top_5) > 1 else 0.0
    ], dtype=np.float32)


def compression_test(data: bytes) -> np.ndarray:
    """Test compressibility - encrypted data should not compress well."""
    if len(data) < 16:
        return np.zeros(3, dtype=np.float32)
    
    try:
        original_size = len(data)
        compressed = zlib.compress(data, level=9)
        compressed_size = len(compressed)
        
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        savings = 1.0 - ratio
        
        return np.array([
            ratio,
            savings,
            1.0 if savings < 0.1 else 0.0
        ], dtype=np.float32)
    except:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def block_size_indicators(data: bytes) -> np.ndarray:
    """Specific features for detecting block size patterns."""
    if len(data) < 32:
        return np.zeros(10, dtype=np.float32)
    
    features = []
    
    for bs in [8, 16, 24, 32, 48, 64, 128, 256, 512, 1024]:
        if len(data) >= bs * 3:
            n_blocks = len(data) // bs
            if n_blocks >= 2:
                blocks = [data[i*bs:(i+1)*bs] for i in range(min(n_blocks, 100))]
                unique_ratio = len(set(blocks)) / len(blocks)
                features.append(unique_ratio)
            else:
                features.append(1.0)
        else:
            features.append(1.0)
    
    return np.array(features, dtype=np.float32)


def chi_square_uniformity(data: bytes) -> float:
    """Chi-square test for uniformity of byte distribution."""
    if len(data) < 16:
        return 0.0
    
    observed = byte_frequency_distribution(data) * len(data)
    expected = len(data) / 256.0
    
    chi_sq = np.sum((observed - expected) ** 2 / expected)
    return chi_sq / 256.0


def byte_transition_features(data: bytes) -> np.ndarray:
    """Features about byte value transitions."""
    if len(data) < 2:
        return np.zeros(4, dtype=np.float32)
    
    transitions = np.abs(np.diff(np.array(list(data), dtype=np.float32)))
    
    return np.array([
        np.mean(transitions),
        np.std(transitions),
        np.mean(transitions > 100),
        np.mean(transitions > 200)
    ], dtype=np.float32)


def run_length_features(data: bytes) -> np.ndarray:
    """Run-length encoding features."""
    if len(data) < 2:
        return np.zeros(4, dtype=np.float32)
    
    runs = []
    current_byte = data[0]
    current_run = 1
    
    for byte in data[1:]:
        if byte == current_byte:
            current_run += 1
        else:
            runs.append(current_run)
            current_byte = byte
            current_run = 1
    runs.append(current_run)
    
    runs_arr = np.array(runs, dtype=np.float32)
    
    return np.array([
        len(runs) / len(data),
        np.mean(runs_arr),
        np.max(runs_arr),
        np.std(runs_arr)
    ], dtype=np.float32)


def extract_features(ciphertext: Union[str, bytes]) -> np.ndarray:
    """Extract all features from ciphertext for ML model input."""
    data = to_bytes(ciphertext)
    original_str = str(ciphertext) if isinstance(ciphertext, str) else ciphertext.hex()
    
    features = []
    
    features.append(shannon_entropy(data))
    features.append(hex_char_ratio(original_str))
    
    stats = byte_statistics(data)
    features.extend([stats['mean'], stats['std'], stats['variance'], stats['skewness'], stats['kurtosis'], stats['min'], stats['max']])
    
    freq_dist = byte_frequency_distribution(data)
    features.extend(freq_dist.tolist())
    
    ngram_feats = ngram_frequencies(data, n=2)
    features.extend(ngram_feats.tolist())
    
    block_feats = block_pattern_features(data)
    features.extend(block_feats.tolist())
    
    features.append(len(data))
    features.append(math.log1p(len(data)))
    
    autocorr_feats = autocorrelation_features(data)
    features.extend(autocorr_feats.tolist())
    
    pair_feats = byte_pair_features(data)
    features.extend(pair_feats.tolist())
    
    comp_feats = compression_test(data)
    features.extend(comp_feats.tolist())
    
    bs_feats = block_size_indicators(data)
    features.extend(bs_feats.tolist())
    
    features.append(chi_square_uniformity(data))
    
    trans_feats = byte_transition_features(data)
    features.extend(trans_feats.tolist())
    
    rl_feats = run_length_features(data)
    features.extend(rl_feats.tolist())
    
    return np.array(features, dtype=np.float32)


def extract_raw_bytes(ciphertext: Union[str, bytes], max_length: int = 512) -> np.ndarray:
    data = to_bytes(ciphertext)
    
    if len(data) > max_length:
        data = data[:max_length]
    
    result = np.zeros(max_length, dtype=np.uint8)
    result[:len(data)] = list(data)
    
    return result


def get_feature_names() -> List[str]:
    names = ['entropy', 'hex_ratio']
    names.extend(['byte_mean', 'byte_std', 'byte_var', 'byte_skew', 'byte_kurt', 'byte_min', 'byte_max'])
    names.extend([f'byte_freq_{i}' for i in range(256)])
    names.extend(['ngram_unique', 'ngram_mean', 'ngram_std', 'ngram_max', 'ngram_min', 'ngram_median', 'ngram_p25', 'ngram_p75', 'norm_entropy', 'norm_length'])
    names.extend(['block8_unique', 'block16_unique', 'block32_unique', 'block64_unique', 'mod8', 'mod16', 'mod64', 'mod128'])
    names.extend(['raw_length', 'log_length'])
    names.extend(['autocorr_mean', 'autocorr_max', 'autocorr_lag', 'autocorr_std', 'autocorr_high'])
    names.extend(['pair_unique', 'pair_mean', 'pair_std', 'pair_max', 'top_pair_1', 'top_pair_2'])
    names.extend(['compress_ratio', 'compress_savings', 'is_compressible'])
    names.extend([f'block_ind_{bs}' for bs in [8, 16, 24, 32, 48, 64, 128, 256, 512, 1024]])
    names.append('chi_square')
    names.extend(['trans_mean', 'trans_std', 'trans_high', 'trans_very_high'])
    names.extend(['run_density', 'run_mean', 'run_max', 'run_std'])
    return names


FEATURE_SIZE = 336
RAW_SEQUENCE_LENGTH = 512
