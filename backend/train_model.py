import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class VoiceStressTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        """Extract audio features for stress detection"""
        try:
            # Load audio file with error handling
            y, sr = librosa.load(file_path, duration=3.0)  # 3 second clips

            # Skip if audio is too short
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                return None

            features = {}

            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = np.mean(mfccs[i])

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma'] = np.mean(chroma)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = np.mean(contrast)

            # Zero crossing rate
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))

            # RMS energy
            features['rms'] = np.mean(librosa.feature.rms(y=y))

            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0

            # Spectral centroid
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            # Spectral bandwidth
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            return features

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def load_ravdess_dataset(self, ravdess_path):
        """Load Ravdess dataset"""
        data = []

        if not ravdess_path.exists():
            print(f"Ravdess path not found: {ravdess_path}")
            return data

        print("Loading Ravdess dataset...")
        for actor_dir in ravdess_path.glob("Actor_*"):
            if actor_dir.is_dir():
                for audio_file in actor_dir.glob("*.wav"):
                    # Parse filename: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
                    # e.g., "03-01-01-01-01-01-01.wav"
                    parts = audio_file.stem.split('-')
                    if len(parts) >= 3:
                        try:
                            emotion_code = int(parts[2])  # Third element is emotion

                            # Ravdess emotion mapping
                            emotion_mapping = {
                                1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                                5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                            }

                            emotion = emotion_mapping.get(emotion_code, 'neutral')
                            stress_level = self.emotion_to_stress(emotion)

                            features = self.extract_features(str(audio_file))
                            if features:
                                features['stress_level'] = stress_level
                                features['emotion'] = emotion
                                features['dataset'] = 'ravdess'
                                data.append(features)
                                print(f"Processed: {audio_file.name} -> {emotion} ({stress_level})")

                        except (ValueError, IndexError) as e:
                            print(f"Error parsing filename {audio_file.name}: {e}")
                            continue

        return data

    def load_crema_dataset(self, crema_path):
        """Load Crema dataset"""
        data = []

        if not crema_path.exists():
            print(f"Crema path not found: {crema_path}")
            return data

        print("Loading Crema dataset...")
        for audio_file in crema_path.glob("*.wav"):
            # Parse filename: e.g., "1001_DFA_ANG_XX.wav"
            parts = audio_file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]  # Third element is emotion

                # Crema emotion mapping
                emotion_mapping = {
                    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
                    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
                }

                emotion = emotion_mapping.get(emotion_code, 'neutral')
                stress_level = self.emotion_to_stress(emotion)

                features = self.extract_features(str(audio_file))
                if features:
                    features['stress_level'] = stress_level
                    features['emotion'] = emotion
                    features['dataset'] = 'crema'
                    data.append(features)
                    print(f"Processed: {audio_file.name} -> {emotion} ({stress_level})")

        return data

    def load_tess_dataset(self, tess_path):
        """Load Tess dataset"""
        data = []

        if not tess_path.exists():
            print(f"Tess path not found: {tess_path}")
            return data

        print("Loading Tess dataset...")
        emotion_folders = ['OAF_angry', 'OAF_disgust', 'OAF_Fear', 'OAF_happy',
                          'OAF_neutral', 'OAF_Pleasant_surprise', 'OAF_Sad',
                          'YAF_angry', 'YAF_disgust', 'YAF_fear', 'YAF_happy',
                          'YAF_neutral', 'YAF_pleasant_surprised', 'YAF_sad']

        for folder in emotion_folders:
            folder_path = tess_path / folder
            if folder_path.exists():
                # Extract emotion from folder name
                emotion_part = folder.split('_')[1].lower()
                if emotion_part == 'pleasant':
                    emotion = 'surprised'
                elif emotion_part == 'fear':
                    emotion = 'fearful'
                else:
                    emotion = emotion_part

                stress_level = self.emotion_to_stress(emotion)

                for audio_file in folder_path.glob("*.wav"):
                    features = self.extract_features(str(audio_file))
                    if features:
                        features['stress_level'] = stress_level
                        features['emotion'] = emotion
                        features['dataset'] = 'tess'
                        data.append(features)
                        print(f"Processed: {audio_file.name} -> {emotion} ({stress_level})")

        return data

    def load_savee_dataset(self, savee_path):
        """Load Savee dataset"""
        data = []

        if not savee_path.exists():
            print(f"Savee path not found: {savee_path}")
            return data

        print("Loading Savee dataset...")
        # Savee emotion mapping from filenames
        emotion_mapping = {
            'a': 'anger', 'd': 'disgust', 'f': 'fear', 'h': 'happiness',
            'n': 'neutral', 'sa': 'sadness', 'su': 'surprise'
        }

        for audio_file in savee_path.glob("*.wav"):
            # Parse filename: e.g., "DC_a01.wav" or "JE_sa01.wav"
            name = audio_file.stem
            if len(name) >= 4 and name[2] == '_':
                emotion_code = name[3:-2] if len(name) > 5 else name[3]

                emotion = emotion_mapping.get(emotion_code, 'neutral')
                stress_level = self.emotion_to_stress(emotion)

                features = self.extract_features(str(audio_file))
                if features:
                    features['stress_level'] = stress_level
                    features['emotion'] = emotion
                    features['dataset'] = 'savee'
                    data.append(features)
                    print(f"Processed: {audio_file.name} -> {emotion} ({stress_level})")

        return data

    def emotion_to_stress(self, emotion):
        """Convert emotion to stress level"""
        stress_map = {
            'neutral': 'low', 'calm': 'low', 'happy': 'low', 'happiness': 'low',
            'sad': 'moderate', 'surprised': 'moderate', 'surprise': 'moderate',
            'angry': 'high', 'anger': 'high', 'fearful': 'high', 'fear': 'high',
            'disgust': 'high', 'sadness': 'high'
        }
        return stress_map.get(emotion.lower(), 'moderate')

    def load_dataset(self, data_path):
        """Load and process all voice datasets"""
        data_path = Path(data_path)

        all_data = []

        # Load each dataset
        datasets = {
            'ravdess': data_path / "Ravdess" / "audio_speech_actors_01-24",
            'crema': data_path / "Crema",
            'tess': data_path / "Tess",
            'savee': data_path / "Savee"
        }

        for name, path in datasets.items():
            try:
                print(f"Checking {name} at: {path}")
                print(f"Path exists: {path.exists()}")

                if name == 'ravdess':
                    data = self.load_ravdess_dataset(path)
                elif name == 'crema':
                    data = self.load_crema_dataset(path)
                elif name == 'tess':
                    data = self.load_tess_dataset(path)
                elif name == 'savee':
                    data = self.load_savee_dataset(path)

                all_data.extend(data)
                print(f"Loaded {len(data)} samples from {name}")

            except Exception as e:
                print(f"Error loading {name} dataset: {e}")

        if not all_data:
            raise ValueError("No data loaded from any dataset")

        df = pd.DataFrame(all_data)
        print(f"\nTotal samples loaded: {len(df)}")

        # Show distribution
        print("\nStress level distribution:")
        print(df['stress_level'].value_counts())
        print("\nEmotion distribution:")
        print(df['emotion'].value_counts())

        return df

    def train(self, data_path):
        """Train the stress detection model"""
        print("Loading dataset...")
        df = self.load_dataset(data_path)

        # Prepare features and labels
        feature_cols = [col for col in df.columns if col not in ['stress_level', 'emotion', 'dataset']]
        X = df[feature_cols]
        y = self.label_encoder.fit_transform(df['stress_level'])

        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        # Handle missing values
        X = X.fillna(X.mean())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_scaled)

        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                  target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png')
        plt.close()

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()

        # Save model
        self.save_model()

        return self.model

    def save_model(self, model_path='models/voice_stress_model.pkl'):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': None
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")

    def load_model(self, model_path='models/voice_stress_model.pkl'):
        """Load trained model"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            print(f"Model loaded from {model_path}")
            return True
        return False

    def predict_stress(self, audio_file_path):
        """Predict stress level from audio file"""
        if not self.model:
            if not self.load_model():
                return None

        features = self.extract_features(audio_file_path)
        if not features:
            return None

        # Prepare feature vector
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.fillna(feature_df.mean())
        features_scaled = self.scaler.transform(feature_df)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        stress_level = self.label_encoder.inverse_transform([prediction])[0]

        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        return {
            'stress_level': stress_level,
            'confidence': float(confidence),
            'probabilities': dict(zip(self.label_encoder.classes_, probabilities))
        }

def main():
    """Main training function"""
    print("Voice Stress Detection Model Training")
    print("=" * 50)

    trainer = VoiceStressTrainer()

    try:
        # Train the model - use parent directory for archive
        archive_path = Path(__file__).parent.parent / "archive"
        print(f"Using archive path: {archive_path}")
        trainer.train(archive_path)

        print("\nTraining completed successfully!")
        print("Model saved to: models/voice_stress_model.pkl")
        print("Confusion matrix saved to: models/confusion_matrix.png")
        print("Feature importance plot saved to: models/feature_importance.png")

    except Exception as e:
        print(f"Training failed: {e}")
        return False

    return True

if __name__ == "__main__":
    main()