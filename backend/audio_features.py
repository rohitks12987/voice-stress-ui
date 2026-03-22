from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size), mode="constant")
    num_frames = 1 + max(0, (y.size - frame_length) // hop_length)
    if num_frames <= 0:
        num_frames = 1
    shape = (num_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        if right <= left:
            continue
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        for k in range(left, center):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, right):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (right - k) / float(right - center)
    return fbank


def _estimate_pitch_autocorr(frames: np.ndarray, sr: int) -> np.ndarray:
    # Stress speech mostly lies in the 50-500 Hz band.
    min_lag = max(1, int(sr / 500))
    max_lag = max(min_lag + 2, int(sr / 50))

    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    if frame_rms.size == 0:
        return np.array([], dtype=np.float32)

    threshold = np.percentile(frame_rms, 60.0)
    voiced_indices = np.where(frame_rms >= threshold)[0]
    if voiced_indices.size == 0:
        return np.array([], dtype=np.float32)

    # Keep runtime bounded on long clips.
    if voiced_indices.size > 120:
        stride = int(np.ceil(voiced_indices.size / 120.0))
        voiced_indices = voiced_indices[::stride]

    pitches = []
    for idx in voiced_indices:
        frame = frames[idx] - float(np.mean(frames[idx]))
        if np.allclose(frame, 0.0):
            continue

        corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
        if corr.size <= min_lag:
            continue

        upper = min(max_lag, corr.size)
        segment = corr[min_lag:upper]
        if segment.size == 0:
            continue

        best_lag = int(np.argmax(segment)) + min_lag
        if best_lag <= 0:
            continue

        freq = float(sr / best_lag)
        if 50.0 <= freq <= 500.0:
            pitches.append(freq)

    return np.asarray(pitches, dtype=np.float32)


def extract_audio_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict | None:
    """Extracts stable, fast audio features from a mono waveform."""
    try:
        if y is None:
            return None

        y = np.asarray(y, dtype=np.float32).flatten()
        if y.size < 1024:
            return None

        frame_length = max(256, int(0.025 * sr))
        hop_length = max(128, int(0.010 * sr))
        n_fft = 1
        while n_fft < frame_length:
            n_fft *= 2

        frames = _frame_signal(y, frame_length, hop_length)
        window = np.hamming(frame_length).astype(np.float32)
        windowed = frames * window

        spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
        power_spec = (np.abs(spectrum) ** 2) / float(n_fft)
        eps = 1e-10

        # MFCCs
        fbank = _mel_filterbank(sr, n_fft, n_mels=40)
        mel_energies = np.maximum(power_spec @ fbank.T, eps)
        log_mel = np.log(mel_energies)
        mfcc_matrix = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
        if mfcc_matrix.shape[0] > 1:
            mfcc_delta = np.diff(mfcc_matrix, axis=0)
        else:
            mfcc_delta = np.zeros_like(mfcc_matrix)

        # Core spectral and temporal features
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        signs = np.sign(frames)
        signs[signs == 0] = 1
        zcr = np.mean(np.sum(np.diff(signs, axis=1) != 0, axis=1) / float(frame_length))

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        power_sum = np.sum(power_spec, axis=1) + eps
        centroid_frames = np.sum(power_spec * freqs[None, :], axis=1) / power_sum
        bandwidth_frames = np.sqrt(
            np.sum(((freqs[None, :] - centroid_frames[:, None]) ** 2) * power_spec, axis=1) / power_sum
        )

        # Spectral rolloff (85% cumulative energy).
        cumulative_energy = np.cumsum(power_spec, axis=1)
        rolloff_target = 0.85 * power_sum[:, None]
        rolloff_idx = np.argmax(cumulative_energy >= rolloff_target, axis=1)
        rolloff_hz = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]

        # Spectral flatness and flux.
        geometric_mean = np.exp(np.mean(np.log(power_spec + eps), axis=1))
        arithmetic_mean = np.mean(power_spec + eps, axis=1)
        flatness = geometric_mean / arithmetic_mean

        norm_spec = power_spec / (power_sum[:, None] + eps)
        if norm_spec.shape[0] > 1:
            flux = np.sqrt(np.sum(np.diff(norm_spec, axis=0) ** 2, axis=1))
        else:
            flux = np.array([0.0], dtype=np.float32)

        # Low/high frequency energy ratio.
        split_bin = int(np.searchsorted(freqs, 1000.0))
        split_bin = min(max(split_bin, 1), power_spec.shape[1] - 1)
        low_energy = np.sum(power_spec[:, :split_bin], axis=1)
        high_energy = np.sum(power_spec[:, split_bin:], axis=1) + eps
        low_high_ratio = low_energy / high_energy

        # Compact chroma scalar from pitch-class energy distribution.
        valid_bins = np.where(freqs > 0)[0]
        chroma_dist = np.zeros(12, dtype=np.float32)
        if valid_bins.size > 0:
            midi = np.round(69 + 12 * np.log2(freqs[valid_bins] / 440.0)).astype(int)
            pitch_classes = np.mod(midi, 12)
            bin_energy = np.mean(power_spec[:, valid_bins], axis=0)
            for pc in range(12):
                chroma_dist[pc] = float(np.sum(bin_energy[pitch_classes == pc]))
        chroma_total = float(np.sum(chroma_dist))
        chroma_scalar = float(np.max(chroma_dist) / (chroma_total + eps)) if chroma_total > 0 else 0.0

        pitch_vals = _estimate_pitch_autocorr(frames, sr)

        features = {}
        for i in range(n_mfcc):
            features[f"mfcc_{i}"] = float(np.mean(mfcc_matrix[:, i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfcc_matrix[:, i]))
            features[f"mfcc_delta_{i}"] = float(np.mean(np.abs(mfcc_delta[:, i])))

        features.update(
            {
                "pitch_mean": float(np.mean(pitch_vals)) if pitch_vals.size else 0.0,
                "pitch_std": float(np.std(pitch_vals)) if pitch_vals.size else 0.0,
                "pitch_p10": float(np.percentile(pitch_vals, 10)) if pitch_vals.size else 0.0,
                "pitch_p90": float(np.percentile(pitch_vals, 90)) if pitch_vals.size else 0.0,
                "rms": float(np.mean(frame_rms)),
                "rms_std": float(np.std(frame_rms)),
                "rms_p90": float(np.percentile(frame_rms, 90)),
                "spectral_centroid": float(np.mean(centroid_frames)),
                "spectral_bandwidth": float(np.mean(bandwidth_frames)),
                "spectral_rolloff": float(np.mean(rolloff_hz)),
                "spectral_flatness": float(np.mean(flatness)),
                "spectral_flux": float(np.mean(flux)),
                "low_high_energy_ratio": float(np.mean(low_high_ratio)),
                "chroma": chroma_scalar,
                "zero_crossing_rate": float(zcr),
            }
        )
        return features
    except Exception:
        return None
