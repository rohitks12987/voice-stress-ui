import functools
from flask import request, jsonify
import json
import base64
import hmac
from hashlib import sha256
import time
import os

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "clinical-secret-2026")

def verify_staff_token(token):
    """Token verify karne ka logic"""
    if not token: return None
    try:
        p_part, sig = token.split('.')
        expected = hmac.new(ADMIN_SECRET.encode(), p_part.encode(), sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        
        # Check Expiry
        payload = json.loads(base64.urlsafe_b64decode(p_part + "=="))
        if payload['exp'] < time.time(): return None
        return payload
    except:
        return None

def staff_required(f):
    """Decorator jo Admin routes ko protect karega"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "")
        
        staff_data = verify_staff_token(token)
        if not staff_data:
            return jsonify({"status": "error", "message": "Clinical Authorization Failed"}), 401
            
        return f(*args, **kwargs)
    return decorated_function