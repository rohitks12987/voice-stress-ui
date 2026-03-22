import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

from audio_features import extract_audio_features


class VoiceStressTrainer:
    def __init__(
        self,
        sample_rate=22050,
        n_mfcc=13,
        min_audio_sec=0.5,
        trim_top_db=25.0,
        random_state=42,
        n_jobs=None,
        balance_train=False,
        fit_on_all_data=False,
    ):
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.min_audio_sec = float(min_audio_sec)
        self.trim_top_db = float(trim_top_db)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs) if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
        self.balance_train = bool(balance_train)
        self.fit_on_all_data = bool(fit_on_all_data)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def _load_audio(self, file_path):
        y, sr = sf.read(str(file_path), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != self.sample_rate:
            y = signal.resample_poly(y, self.sample_rate, sr).astype(np.float32)
            sr = self.sample_rate
        return y.astype(np.float32), int(sr)

    def _trim_silence(self, y):
        if y.size == 0:
            return y
        peak = float(np.max(np.abs(y)))
        if peak <= 0:
            return y

        threshold = peak * (10 ** (-self.trim_top_db / 20.0))
        indices = np.where(np.abs(y) >= threshold)[0]
        if indices.size == 0:
            return y
        return y[int(indices[0]) : int(indices[-1]) + 1]

    def extract_features(self, file_path):
        """Extract features using the same schema expected by AdvancedStressPredictor."""
        try:
            y, sr = self._load_audio(file_path)
            if len(y) < int(sr * self.min_audio_sec):
                return None

            y_trimmed = self._trim_silence(y)
            if y_trimmed.size > 0:
                y = y_trimmed

            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0:
                y = y / peak

            return extract_audio_features(y, sr, n_mfcc=self.n_mfcc)
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
            data.append({"path": str(audio_file), "stress_level": stress})
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
            data.append({"path": str(audio_file), "stress_level": stress})
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
            data.append({"path": str(audio_file), "stress_level": stress})
        return data

    def _extract_row(self, sample):
        feat = self.extract_features(sample["path"])
        if not feat:
            return None
        feat["stress_level"] = sample["stress_level"]
        return feat

    def _extract_all_features(self, samples):
        total = len(samples)
        if total == 0:
            return []

        def _serial_extract():
            rows_local = []
            for idx, sample in enumerate(samples, start=1):
                row = self._extract_row(sample)
                if row:
                    rows_local.append(row)
                if idx % 500 == 0 or idx == total:
                    print(f"  Extracted {idx}/{total} files...")
            return rows_local

        if self.n_jobs <= 1:
            return _serial_extract()

        try:
            rows = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=5)(
                joblib.delayed(self._extract_row)(sample) for sample in samples
            )
            return [row for row in rows if row]
        except Exception as exc:
            print(f"Parallel extraction unavailable ({exc}). Falling back to single-worker mode.")
            return _serial_extract()

    def _balance_dataset(self, X_df, y_arr):
        df = X_df.copy()
        df["target"] = y_arr

        counts = df["target"].value_counts()
        target = counts.max()
        chunks = []
        for _, group in df.groupby("target"):
            if len(group) < target:
                group = resample(group, replace=True, n_samples=target, random_state=self.random_state)
            chunks.append(group)

        out = (
            pd.concat(chunks, ignore_index=True)
            .sample(frac=1.0, random_state=self.random_state)
            .reset_index(drop=True)
        )
        y_balanced = out["target"].to_numpy()
        X_balanced = out.drop(columns=["target"])
        return X_balanced, y_balanced

    def _candidate_models(self):
        return [
            (
                "extra_trees_regularized",
                ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=30,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                False,
            ),
            (
                "extra_trees_generalized",
                ExtraTreesClassifier(
                    n_estimators=900,
                    max_depth=22,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features=0.6,
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                False,
            ),
            (
                "random_forest_generalized",
                RandomForestClassifier(
                    n_estimators=900,
                    max_depth=24,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features=0.6,
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                False,
            ),
            (
                "extra_trees_large_reference",
                ExtraTreesClassifier(
                    n_estimators=800,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                False,
            ),
        ]

    def _fit_model(self, model, X_data, y_data, sample_weight=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        try:
            model.fit(X_data, y_data, **fit_kwargs)
            return model
        except PermissionError:
            if hasattr(model, "set_params") and hasattr(model, "n_jobs"):
                print("Model parallelism blocked by environment. Retrying with n_jobs=1.")
                model = model.set_params(n_jobs=1)
                model.fit(X_data, y_data, **fit_kwargs)
                return model
            raise

    def _compute_sample_weights(self, y_data):
        counts = pd.Series(y_data).value_counts()
        total = float(len(y_data))
        n_classes = float(len(counts))
        class_weights = {label: total / (n_classes * count) for label, count in counts.items()}
        return np.asarray([class_weights[label] for label in y_data], dtype=np.float32)

    def _fit_best_model(self, X_train, y_train):
        X_search, X_val, y_search, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=self.random_state,
        )

        scaler = StandardScaler()
        X_search_scaled = scaler.fit_transform(X_search)
        X_val_scaled = scaler.transform(X_val)

        best_name = None
        best_model = None
        best_score = -1.0
        best_uses_weights = False
        sample_weights = self._compute_sample_weights(y_search)

        for name, model, uses_weights in self._candidate_models():
            print(f"Training candidate model: {name}")
            model = self._fit_model(
                model,
                X_search_scaled,
                y_search,
                sample_weight=sample_weights if uses_weights else None,
            )

            val_pred = model.predict(X_val_scaled)
            train_pred = model.predict(X_search_scaled)
            macro_f1 = f1_score(y_val, val_pred, average="macro")
            train_macro_f1 = f1_score(y_search, train_pred, average="macro")
            val_acc = accuracy_score(y_val, val_pred)
            overfit_gap = max(0.0, train_macro_f1 - macro_f1)
            generalization_score = macro_f1 - 0.20 * overfit_gap
            print(
                "  Validation -> macro_f1={:.4f}, accuracy={:.4f}, train_macro_f1={:.4f}, gen_score={:.4f}".format(
                    macro_f1, val_acc, train_macro_f1, generalization_score
                )
            )
            if generalization_score > best_score:
                best_score = generalization_score
                best_name = name
                best_model = model
                best_uses_weights = uses_weights

        # Refit scaler + selected model on full training split.
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        final_weights = self._compute_sample_weights(y_train) if best_uses_weights else None
        best_model = self._fit_model(best_model, X_train_scaled, y_train, sample_weight=final_weights)
        return best_name, best_model, best_uses_weights

    def _calc_metrics(self, y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }

    def train(self, archive_root="archive", max_files_per_dataset=None):
        root = Path(archive_root)
        print(f"Loading datasets from: {root.absolute()}")

        ravdess = self._load_ravdess(root)
        crema = self._load_crema(root)
        tess = self._load_tess(root)

        if max_files_per_dataset:
            cap = int(max_files_per_dataset)
            ravdess = ravdess[:cap]
            crema = crema[:cap]
            tess = tess[:cap]

        data = ravdess + crema + tess
        print(f"Loaded samples -> RAVDESS: {len(ravdess)}, CREMA: {len(crema)}, TESS: {len(tess)}")
        if not data:
            print("Error: no training data found.")
            return

        print(f"Extracting features with {self.n_jobs} worker(s)...")
        rows = self._extract_all_features(data)
        if not rows:
            print("Error: feature extraction failed for all files.")
            return

        df = pd.DataFrame(rows)
        print("Raw label distribution:", df["stress_level"].value_counts().to_dict())

        X = df.drop(columns=["stress_level"]).fillna(0.0)
        y = self.label_encoder.fit_transform(df["stress_level"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        if self.balance_train:
            X_train, y_train = self._balance_dataset(X_train, y_train)
            class_map = {
                self.label_encoder.inverse_transform([int(label)])[0]: int(count)
                for label, count in pd.Series(y_train).value_counts().to_dict().items()
            }
            print("Balanced train distribution:", class_map)

        print(f"Training on {len(X_train)} samples...")
        best_model_name, model, best_model_uses_weights = self._fit_best_model(X_train, y_train)
        print(f"Selected model: {best_model_name}")

        X_train_scaled = self.scaler.transform(X_train)
        y_train_pred = model.predict(X_train_scaled)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        train_metrics = self._calc_metrics(y_train, y_train_pred)
        test_metrics = self._calc_metrics(y_test, y_pred)
        print(f"Train Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
        print(f"Train Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
        print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        final_fit_samples = int(len(X_train))
        if self.fit_on_all_data:
            print("Refitting final model on combined train+test data for maximum learnable coverage.")
            final_model = clone(model)
            self.scaler = StandardScaler()
            X_full_scaled = self.scaler.fit_transform(X)
            full_weights = self._compute_sample_weights(y) if best_model_uses_weights else None
            final_model = self._fit_model(final_model, X_full_scaled, y, sample_weight=full_weights)
            model = final_model
            final_fit_samples = int(len(X))

        output_dir = Path(__file__).parent / "models"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "voice_stress_model.pkl"

        bundle = {
            "model": model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": list(X.columns),
            "n_mfcc": self.n_mfcc,
            "sample_rate": self.sample_rate,
            "trim_top_db": self.trim_top_db,
            "model_name": best_model_name,
            "metrics": {
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                # Backward-compatible aliases (historically pointed to test metrics).
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            },
            "test_size": 0.2,
            "fit_on_all_data": self.fit_on_all_data,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "final_fit_samples": final_fit_samples,
        }
        joblib.dump(bundle, output_path)
        print(f"Success: Model saved to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    archive_dir = base_dir.parent / "archive"

    parser = argparse.ArgumentParser(description="Train the voice stress classifier.")
    parser.add_argument("--archive-root", type=str, default=str(archive_dir))
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Upsample minority classes on the training split (can improve minority recall).",
    )
    parser.add_argument(
        "--fit-on-all-data",
        action="store_true",
        help="After test evaluation, retrain selected model on train+test combined data for final deployment.",
    )
    args = parser.parse_args()

    trainer = VoiceStressTrainer(
        n_mfcc=args.n_mfcc,
        n_jobs=args.workers,
        balance_train=args.balance_train,
        fit_on_all_data=args.fit_on_all_data,
    )
    trainer.train(archive_root=args.archive_root, max_files_per_dataset=args.max_files_per_dataset)
