import requests
import json
import os
import sys

# Add backend directory to path to import from create_test_user
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_test_user import create_user as ensure_test_user_exists

# --- CONFIGURATION ---
BASE_URL = "http://127.0.0.1:8000/api"

# This is the user who will trigger the SOS alert.
# The script will ensure this user exists.
TEST_SENDER_EMAIL = "testuser@vocalvibe.pro"

# !!! IMPORTANT !!!
# Change this to an email address you can access to verify the alert.
TEST_RECIPIENT_EMAIL = "pottamrohitks12@gmail.com"

def run_email_test():
    """
    A focused test to verify that SOS email alerts are sent correctly.
    """
    print("🚨 Starting SOS Email Alert System Test...")
    print("=" * 50)

    # 0. Pre-flight checks
    if TEST_RECIPIENT_EMAIL == "pottamrohitks12@gmail.com":
        print("❌ CONFIGURATION ERROR: Please edit this script and change")
        print("   `TEST_RECIPIENT_EMAIL` to an email address you can check.")
        print("=" * 50)
        return

    print(f"ℹ️  Test User: {TEST_SENDER_EMAIL}")
    print(f"📬 Recipient: {TEST_RECIPIENT_EMAIL} (check this inbox)")
    print("-" * 50)

    # 1. Ensure the test user exists in the database.
    print("[1] Ensuring test user exists...")
    try:
        # This function is imported from create_test_user.py
        # It will create the user if they don't exist, and print a message.
        ensure_test_user_exists()
    except Exception as e:
        print(f"    ❌ Failed to ensure user exists. Error: {e}")
        print("       Please make sure your database is running and configured correctly.")
        return

    # 2. Add a new emergency contact (the recipient) for the test user.
    print("\n[2] Adding emergency contact via API...")
    contact_payload = {
        "user_email": TEST_SENDER_EMAIL,
        "name": "Test Recipient",
        "email": TEST_RECIPIENT_EMAIL, # This is the crucial field for email alerts
        "relationship": "Test"
    }
    try:
        res = requests.post(f"{BASE_URL}/user/contacts", json=contact_payload)
        if res.status_code == 200:
            print(f"    ✅ Contact '{contact_payload['name']}' added for {TEST_SENDER_EMAIL}.")
        elif res.status_code == 400 and "Max 5 contacts" in res.text:
            print(f"    ⚠️  Could not add contact: User already has maximum (5) contacts.")
            print(f"        Continuing test with existing contacts.")
        else:
            res.raise_for_status() # Raise an exception for other errors
    except requests.exceptions.RequestException as e:
        print(f"    ❌ FAILED to add contact. Error: {e}")
        print("       Is the backend server running? `python backend/app.py`")
        return

    # 3. Trigger the SOS alert from the test user.
    print("\n[3] 🔴 TRIGGERING SOS ALERT via API...")
    sos_payload = {
        "user_email": TEST_SENDER_EMAIL,
        "location": "Test Location: SOS Email Verification"
    }
    try:
        res = requests.post(f"{BASE_URL}/user/sos", json=sos_payload)
        print(f"    Server Response ({res.status_code}): {res.text}")

        if res.status_code == 200:
            print("\n[4] ✅ SOS alert processed by server.")
            print("\n" + "="*50)
            print("👇 FINAL STEP 👇")
            print(f"Please check the inbox for '{TEST_RECIPIENT_EMAIL}' for an SOS email.")
            print("Also, check the backend server's console output for any SMTP errors.")
            print("="*50)
        else:
            print("\n[4] ❌ SOS alert failed on the server.")
            print("    Please check the backend server logs for the full error.")

    except requests.exceptions.RequestException as e:
        print(f"    ❌ FAILED to trigger SOS. Error: {e}")
        print("       Is the backend server running? `python backend/app.py`")
        return

if __name__ == "__main__":
    run_email_test()