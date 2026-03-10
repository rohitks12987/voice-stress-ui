from __future__ import annotations
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Union
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
        # Analyze in short windows to reduce silence bias from real recordings.
        self.clip_duration = 2.5
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
        
        # Improved Pitch Detection: Pick the dominant frequency per frame
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        indices = magnitudes.argmax(axis=0)
        pitch_vals = pitches[indices, np.arange(pitches.shape[1])]
        pitch_vals = pitch_vals[pitch_vals > 0] # Filter out silence

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

    def _load_audio_fallback_wav(self, audio_path: str) -> Tuple[np.ndarray, int]:
        with wave.open(audio_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            raw = (raw - 128.0) / 128.0
        elif sample_width == 2:
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            raw = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")

        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)

        if sr != self.sample_rate:
            raw = librosa.resample(raw, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        return raw.astype(np.float32), sr

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            return librosa.load(audio_path, sr=self.sample_rate)
        except Exception:
            return self._load_audio_fallback_wav(audio_path)

    def _prepare_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Remove leading/trailing silence; browser recordings often include long quiet sections.
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if y_trimmed.size > 0:
            y = y_trimmed

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = y / peak

        return y.astype(np.float32)

    def predict_long_audio(self, audio_path: str) -> Dict:
        """Processes audio of any length; defaults to full-track analysis."""
        y, sr = self._load_audio(audio_path)
        raw_duration = librosa.get_duration(y=y, sr=sr)
        y = self._prepare_audio(y, sr)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 0.4:
            raise ValueError("Speech is too short after silence removal. Please record a longer, clearer sample.")
        
        # Full-track mode (undefined clip duration).
        if not self.clip_duration or self.clip_duration <= 0:
            feat = self.extract_features_from_audio(y, sr)
            probs, classes = self._predict_raw_probabilities(feat)
            all_probs = [probs]
        else:
            seg_samples = int(self.clip_duration * sr)
        
            if len(y) < seg_samples: # Fallback for short audio
                feat = self.extract_features_from_audio(y, sr)
                probs, classes = self._predict_raw_probabilities(feat)
                all_probs = [probs]
            else:
                all_probs = []
                step = int(sr * 1.5) # 1.5-second step for less overlap and faster processing
                for start in range(0, len(y) - seg_samples + 1, step):
                    chunk = y[start : start + seg_samples]
                    feat = self.extract_features_from_audio(chunk, sr)
                    probs, classes = self._predict_raw_probabilities(feat)
                    all_probs.append(probs)
                # Ensure at least one segment is processed for edge-length clips.
                if not all_probs:
                    chunk = y[-seg_samples:] if len(y) >= seg_samples else y
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
