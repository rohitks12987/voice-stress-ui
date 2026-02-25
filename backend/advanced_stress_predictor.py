from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union
import joblib
import librosa
import numpy as np
import pandas as pd

class AdvancedStressPredictor:
    LABEL_LOW = "LOW"
    LABEL_MODERATE = "MODERATE"
    LABEL_HIGH = "HIGH"

    def __init__(self, model_path: Union[str, Path], window_size: int = 5):
        self.model_path = str(model_path)
        self.window_size = window_size
        self.sample_rate = 22050
        self.clip_duration = 3.0  # Training segment length
        self.n_mfcc = 13
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self._load_model_bundle()

    def _load_model_bundle(self) -> None:
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict):
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler")
            self.label_encoder = bundle.get("label_encoder")
            self.feature_names = bundle.get("feature_names") or []
        else:
            self.model = bundle

    def extract_features_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        if y.ndim > 1: y = librosa.to_mono(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]

        features = {f"mfcc_{i}": float(np.mean(mfccs[i])) for i in range(self.n_mfcc)}
        features.update({
            "pitch_mean": float(np.mean(pitch_vals)) if pitch_vals.size else 0.0,
            "pitch_std": float(np.std(pitch_vals)) if pitch_vals.size else 0.0,
            "rms": float(np.mean(rms)),
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "chroma": float(np.mean(chroma)),
            "zero_crossing_rate": float(np.mean(zcr))
        })
        return features

    def _predict_raw_probabilities(self, features: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        df = pd.DataFrame([features])
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns: df[col] = 0.0
            df = df[self.feature_names]
        
        x_arr = self.scaler.transform(df) if self.scaler else df.values
        probs = self.model.predict_proba(x_arr)[0]
        classes = [str(c).upper() for c in self.label_encoder.classes_] if self.label_encoder else ["LOW", "MODERATE", "HIGH"]
        return probs, classes

    def predict_long_audio(self, audio_path: str) -> Dict:
        """Processes audio of any length by segmenting it into 3s chunks."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        seg_samples = int(self.clip_duration * sr)
        
        if len(y) < seg_samples: # Fallback for short audio
            feat = self.extract_features_from_audio(y, sr)
            probs, classes = self._predict_raw_probabilities(feat)
            all_probs = [probs]
        else:
            all_probs = []
            step = int(sr * 1.0) # 1-second step for overlap
            for start in range(0, len(y) - seg_samples, step):
                chunk = y[start : start + seg_samples]
                feat = self.extract_features_from_audio(chunk, sr)
                probs, classes = self._predict_raw_probabilities(feat)
                all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        label = classes[int(np.argmax(avg_probs))]
        return {
            "stress_level": label,
            "confidence": float(np.max(avg_probs)),
            "probabilities": {classes[i]: float(avg_probs[i]) for i in range(len(classes))},
            "duration": duration
        }