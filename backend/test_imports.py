import os
import sys
sys.path.append('.')

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import librosa
    print("✓ librosa imported")
except ImportError as e:
    print(f"✗ librosa import failed: {e}")

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✓ sklearn imported")
except ImportError as e:
    print(f"✗ sklearn import failed: {e}")

try:
    import joblib
    print("✓ joblib imported")
except ImportError as e:
    print(f"✗ joblib import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

print("All imports tested.")