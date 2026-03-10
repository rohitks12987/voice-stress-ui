import os
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


class VoiceStressTrainer:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        """Extract features using the same schema expected by AdvancedStressPredictor."""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            if len(y) < int(sr * 0.5):
                return None

            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if y_trimmed.size > 0:
                y = y_trimmed

            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0:
                y = y / peak

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
            pitch_vals = pitch_vals[pitch_vals > 0]  # Filter out silence

            features = {f"mfcc_{i}": float(np.mean(mfccs[i])) for i in range(self.n_mfcc)}
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
            return None

    def emotion_to_stress(self, emotion):
        emo = (emotion or "").strip().lower()
        if emo in {"angry", "anger", "fearful", "fear", "disgust"}:
            return "HIGH"
        if emo in {"sad", "surprised", "surprise"}:
            return "MODERATE"
        if emo in {"neutral", "calm", "happy"}:
            return "LOW"
        return None

    def _load_ravdess(self, root: Path):
        data = []
        map_code = {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised",
        }
        base = root / "Ravdess" / "audio_speech_actors_01-24"
        if not base.exists():
            return data

        for audio_file in base.glob("Actor_*/*.wav"):
            parts = audio_file.stem.split("-")
            if len(parts) < 3:
                continue
            try:
                emo = map_code.get(int(parts[2]))
            except ValueError:
                continue
            stress = self.emotion_to_stress(emo)
            if not stress:
                continue
            feat = self.extract_features(audio_file)
            if feat:
                feat["stress_level"] = stress
                data.append(feat)
        return data

    def _load_crema(self, root: Path):
        data = []
        emo_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }
        base = root / "Crema"
        if not base.exists():
            return data

        for audio_file in base.glob("*.wav"):
            parts = audio_file.stem.split("_")
            if len(parts) < 3:
                continue
            emo = emo_map.get(parts[2].upper())
            stress = self.emotion_to_stress(emo)
            if not stress:
                continue
            feat = self.extract_features(audio_file)
            if feat:
                feat["stress_level"] = stress
                data.append(feat)
        return data

    def _load_tess(self, root: Path):
        data = []
        base = root / "Tess"
        if not base.exists():
            return data

        for audio_file in base.glob("*/*.wav"):
            parts = audio_file.stem.split("_")
            if len(parts) < 3:
                continue
            emo = parts[-1].lower()
            if emo == "ps":
                emo = "surprised"
            stress = self.emotion_to_stress(emo)
            if not stress:
                continue
            feat = self.extract_features(audio_file)
            if feat:
                feat["stress_level"] = stress
                data.append(feat)
        return data

    def _balance_dataset(self, df):
        counts = df["stress_level"].value_counts()
        target = counts.max()
        chunks = []
        for label, group in df.groupby("stress_level"):
            if len(group) < target:
                group = resample(group, replace=True, n_samples=target, random_state=42)
            chunks.append(group)
        out = pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        return out

    def train(self, archive_root="archive"):
        root = Path(archive_root)
        print(f"Loading datasets from: {root.absolute()}")

        ravdess = self._load_ravdess(root)
        crema = self._load_crema(root)
        tess = self._load_tess(root)
        data = ravdess + crema + tess

        print(f"Loaded samples -> RAVDESS: {len(ravdess)}, CREMA: {len(crema)}, TESS: {len(tess)}")
        if not data:
            print("Error: no training data found.")
            return

        df = pd.DataFrame(data)
        print("Raw label distribution:", df["stress_level"].value_counts().to_dict())
        df = self._balance_dataset(df)
        print("Balanced label distribution:", df["stress_level"].value_counts().to_dict())

        X = df.drop(columns=["stress_level"]).fillna(0.0)
        y = self.label_encoder.fit_transform(df["stress_level"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = ExtraTreesClassifier(
            n_estimators=700,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        print(f"Training on {len(X_train)} samples...")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        output_dir = Path(__file__).parent / "models"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "voice_stress_model.pkl"
        
        bundle = {
            "model": model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": list(X.columns),
        }
        joblib.dump(bundle, output_path)
        print(f"Success: Model saved to {output_path}")


if __name__ == "__main__":
    # Automatically locate the archive folder in the project root
    base_dir = Path(__file__).resolve().parent
    archive_dir = base_dir.parent / "archive"
    VoiceStressTrainer().train(str(archive_dir))
