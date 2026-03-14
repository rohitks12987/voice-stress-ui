import librosa
import numpy as np


def extract_audio_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict | None:
    """Extracts a dictionary of audio features from a numpy array."""
    try:
        # Prevent processing of too-short audio clips which can cause errors
        if len(y) < 1024:  # A small threshold, e.g., one frame
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # Improved Pitch Detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        indices = magnitudes.argmax(axis=0)
        pitch_vals = pitches[indices, np.arange(pitches.shape[1])]
        pitch_vals = pitch_vals[pitch_vals > 0]  # Filter out silence

        features = {f"mfcc_{i}": float(np.mean(mfccs[i])) for i in range(n_mfcc)}
        features.update(
            {
                "pitch_mean": float(np.mean(pitch_vals)) if pitch_vals.size else 0.0,
                "pitch_std": float(np.std(pitch_vals)) if pitch_vals.size else 0.0,
                "rms": float(np.mean(rms)),
                "spectral_centroid": float(np.mean(centroid)),
                "spectral_bandwidth": float(np.mean(bandwidth)),
                "chroma": float(np.mean(chroma)),
                "zero_crossing_rate": float(np.mean(zcr)),
            }
        )
        return features
    except Exception:
        # Broad exception to avoid crashing the whole process on a single bad file
        return None