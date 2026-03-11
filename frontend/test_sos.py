import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"

def test_sos_system():
    print("\n--- 🚨 TESTING SOS EMERGENCY SYSTEM ---")
    
    # 1. Setup a Test User
    user_email = "emergency_test@vocalvibe.pro"
    
    # Ensure user exists (Register/Login simulation)
    print(f"[1] Setting up test user: {user_email}")
    requests.post(f"{BASE_URL}/patient/register", json={
        "name": "Test Patient",
        "email": user_email,
        "password": "securepass123"
    })

    # 2. Add an Emergency Contact
    print("[2] Adding emergency contact...")
    contact_payload = {
        "user_email": user_email,
        "name": "Dr. Smith (Test)",
        "phone": "+15550109999", # Replace with your real number to test SMS
        "relationship": "Doctor"
    }
    resp = requests.post(f"{BASE_URL}/user/contacts", json=contact_payload)
    if resp.status_code == 200:
        print("   ✅ Contact added successfully.")
    else:
        print(f"   ⚠️ Contact add failed: {resp.text}")

    # 3. Trigger SOS
    print("[3] Triggering SOS Alert...")
    sos_payload = {
        "user_email": user_email,
        "location": "Latitude: 28.6139, Longitude: 77.2090 (Test Location)"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/user/sos", json=sos_payload)
        print(f"   Response Code: {resp.status_code}")
        print(f"   Response Body: {json.dumps(resp.json(), indent=2)}")
        
        if resp.status_code == 200:
            print("\n✅ TEST PASSED: SOS Alert processed by server.")
        else:
            print("\n❌ TEST FAILED: Server returned error.")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_sos_system()