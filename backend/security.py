import functools
import json
import base64
import hmac
import time
import os
from hashlib import sha256

from flask import request, jsonify

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "clinical-secret-2026")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@stress-tone.local")


def generate_token(email: str, expires_in: int = 28800) -> str:
    """Generates a secure session token."""
    payload = {"email": email, "exp": int(time.time()) + expires_in}
    p_part = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = hmac.new(ADMIN_SECRET.encode(), p_part.encode(), sha256).hexdigest()
    return f"{p_part}.{sig}"


def verify_token(token: str) -> dict | None:
    """Verifies a session token and returns its payload if valid."""
    if not token:
        return None
    try:
        p_part, sig = token.split('.', 1)
        expected_sig = hmac.new(ADMIN_SECRET.encode(), p_part.encode(), sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None

        # Add padding and decode
        padding = "=" * (-len(p_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(p_part + padding))

        # Check Expiry
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except (ValueError, IndexError, json.JSONDecodeError, base64.binascii.Error):
        return None


def staff_required(f):
    """Decorator that protects routes by requiring a valid staff token."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "")

        staff_data = verify_token(token)
        # For staff, we also verify the email matches the admin email
        if not staff_data or staff_data.get('email') != ADMIN_EMAIL:
            return jsonify({"status": "error", "message": "Secure Clinical Access Required"}), 401

        return f(*args, **kwargs)
    return decorated_function