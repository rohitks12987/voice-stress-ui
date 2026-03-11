import requests
import json
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000/api"
TEST_EMAIL = "sos_test_user@vocalvibe.pro"
TEST_PASS = "password123"

def run_sos_test():
    print("🚨 Starting Emergency SOS System Test...")
    print("=" * 40)
    
    session = requests.Session()

    # 1. Register/Login User
    print("\n[1] Registering Test User...")
    try:
        res = session.post(f"{BASE_URL}/patient/register", json={
            "name": "SOS Tester", "email": TEST_EMAIL, "password": TEST_PASS
        })
        if res.status_code == 409:
            print("    ℹ️ User already exists (Skipping registration)")
        elif res.status_code == 200:
            print("    ✅ User registered successfully")
    except Exception as e:
        print(f"    ❌ Connection failed: {e}")
        return

    # 2. Add Emergency Contact
    print("\n[2] Adding Emergency Contact...")
    contact_payload = {
        "user_email": TEST_EMAIL,
        "name": "Dr. Strange",
        "phone": "+15550199999", # Replace with real number to test SMS
        "relationship": "Doctor"
    }
    
    res = session.post(f"{BASE_URL}/user/contacts", json=contact_payload)
    print(f"    Response: {res.text}")
    
    # 3. Trigger SOS Alert
    print("\n[3] 🔴 TRIGGERING SOS ALERT...")
    sos_payload = {
        "user_email": TEST_EMAIL,
        "location": "VocalVibe HQ - Test Lab 1"
    }
    
    res = session.post(f"{BASE_URL}/user/sos", json=sos_payload)
    if res.status_code == 200:
        print(f"    ✅ SOS SUCCESS: {res.json()['message']}")
    else:
        print(f"    ❌ SOS FAILED: {res.text}")

if __name__ == "__main__":
    run_sos_test()
