import os
import joblib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class VoiceStressTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        """Extracts features from the full duration of the audio file."""
        try:
            # Load audio (Entire duration)
            y, sr = librosa.load(file_path, sr=22050)
            if len(y) < 11025: return None # Skip files shorter than 0.5s
            
            # MFCCs (The core 'shape' of the voice)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = {f'mfcc_{i}': np.mean(mfccs[i]) for i in range(13)}
            
            # Vocal Vitals (Pitch, Energy, and Texture)
            features['chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            features['rms'] = np.mean(librosa.feature.rms(y=y))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Pitch tracking (Crucial for stress detection)
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def emotion_to_stress(self, emotion):
        """Maps standard dataset emotions to your 3 stress levels."""
        low_stress = ['neutral', 'calm', 'happy']
        high_stress = ['angry', 'fearful', 'disgust', 'sad']
        
        emo = emotion.lower()
        if emo in high_stress: return 'HIGH'
        if emo in low_stress: return 'LOW'
        return 'MODERATE'

    def train(self, data_path):
        """Walks through the dataset folder and trains the Random Forest model."""
        data = []
        path = Path(data_path)
        
        print(f"Starting feature extraction from: {path.absolute()}")
        
        # Logic for RAVDESS Dataset Structure
        for actor_dir in path.glob("Actor_*"):
            for audio_file in actor_dir.glob("*.wav"):
                # RAVDESS filename logic: 03-01-XX... (XX is emotion code)
                emo_code = int(audio_file.stem.split('-')[2])
                mapping = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'}
                
                stress_label = self.emotion_to_stress(mapping.get(emo_code, 'neutral'))
                features = self.extract_features(audio_file)
                
                if features:
                    features['stress_level'] = stress_label
                    data.append(features)

        if not data:
            print("Error: No data found! Check if your 'archive' folder contains 'Actor_XX' subfolders.")
            return

        # Prepare Dataframe
        df = pd.DataFrame(data)
        X = df.drop(columns=['stress_level'])
        y = self.label_encoder.fit_transform(df['stress_level'])

        # Split and Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Classifier
        print(f"Training on {len(X_train)} samples...")
        model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        # Save the Bundle
        os.makedirs('models', exist_ok=True)
        bundle = {
            'model': model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': list(X.columns)
        }
        joblib.dump(bundle, 'models/voice_stress_model.pkl')
        print("Success: Model saved to models/voice_stress_model.pkl")

if __name__ == "__main__":
    # Ensure your RAVDESS 'archive' folder is in the same directory as this script
    # or update this path accordingly.
    trainer = VoiceStressTrainer()
    trainer.train("archive")