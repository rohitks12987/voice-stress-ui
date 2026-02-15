import requests
import os
from pathlib import Path

def test_model_prediction():
    """Test if the ML model is being used for predictions"""

    # Find a test audio file
    archive_path = Path("../archive")
    test_file = None

    # Look for a wav file in the datasets
    for root, dirs, files in os.walk(archive_path):
        for file in files:
            if file.endswith('.wav'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break

    if not test_file:
        print("No test audio file found")
        return

    print(f"Testing with file: {test_file}")

    # Test multiple times to see if results are consistent (ML should be consistent, random would vary)
    results = []

    for i in range(3):
        with open(test_file, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            response = requests.post('http://127.0.0.1:8000/api/upload', files=files)

            if response.status_code == 200:
                data = response.json()
                stress_level = data.get('stress_level')
                score = data.get('score')
                results.append((stress_level, score))
                print(f"Test {i+1}: {stress_level}, Score: {score}")
            else:
                print(f"Test {i+1} failed: {response.status_code}")

    # Check consistency
    if len(results) >= 2:
        stress_levels = [r[0] for r in results]
        scores = [r[1] for r in results]

        if len(set(stress_levels)) == 1:
            print("✓ Stress levels are consistent - ML model is working!")
        else:
            print("✗ Stress levels vary - might still be using random scoring")

        score_range = max(scores) - min(scores)
        if score_range < 5:  # Small variation is expected
            print("✓ Scores are relatively consistent")
        else:
            print(f"✗ Scores vary significantly (range: {score_range})")

if __name__ == "__main__":
    test_model_prediction()