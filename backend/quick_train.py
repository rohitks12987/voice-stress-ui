import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_features(file_path):
    """Extract basic audio features"""
    try:
        y, sr = librosa.load(file_path, duration=2.0)  # 2 second clips

        if len(y) < sr * 0.5:  # Skip very short files
            return None

        features = {}

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)  # Reduced to 5
        for i in range(5):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])

        # Basic features
        features['rms'] = np.mean(librosa.feature.rms(y=y))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_sample_data(archive_path):
    """Load a small sample of data for testing"""
    data = []

    # Load just a few Ravdess files
    ravdess_path = archive_path / "Ravdess" / "audio_speech_actors_01-24" / "Actor_01"
    if ravdess_path.exists():
        count = 0
        for audio_file in ravdess_path.glob("*.wav"):
            if count >= 10:  # Just 10 files for testing
                break

            features = extract_features(str(audio_file))
            if features:
                # Parse emotion from filename
                parts = audio_file.stem.split('-')
                emotion_code = int(parts[2]) if len(parts) > 2 else 1

                emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
                emotion = emotion_mapping.get(emotion_code, 'neutral')

                stress_map = {'neutral': 'low', 'calm': 'low', 'happy': 'low', 'sad': 'moderate', 'surprised': 'moderate', 'angry': 'high', 'fearful': 'high', 'disgust': 'high'}
                stress_level = stress_map.get(emotion, 'moderate')

                features['stress_level'] = stress_level
                features['emotion'] = emotion
                data.append(features)
                count += 1
                print(f"Processed: {audio_file.name} -> {emotion} ({stress_level})")

    return data

def main():
    print("Quick Model Training Test")
    print("=" * 30)

    archive_path = Path(__file__).parent.parent / "archive"
    print(f"Archive path: {archive_path}")

    # Load sample data
    data = load_sample_data(archive_path)
    if not data:
        print("No data loaded!")
        return

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")
    print("Stress distribution:", df['stress_level'].value_counts().to_dict())

    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['stress_level', 'emotion']]
    X = df[feature_cols]
    y = LabelEncoder().fit_transform(df['stress_level'])

    # Handle missing values
    X = X.fillna(X.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Test
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(".3f")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': LabelEncoder().fit(df['stress_level']),
        'feature_names': feature_cols
    }
    joblib.dump(model_data, 'models/voice_stress_model.pkl')
    print("Model saved to models/voice_stress_model.pkl")

if __name__ == "__main__":
    main()