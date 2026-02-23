import os
import json
import time
import random
import hmac
import base64
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# --- PATH CONFIGURATION ---
# This locates your folders correctly even if you run from /backend
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# --- CONFIGURATION ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""), # Default XAMPP/WAMP is empty
    "database": os.getenv("DB_NAME", "vocalvibe_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@vocalvibe.pro")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "clinical-secret-2026")

# --- DATABASE HELPERS ---
def get_db():
    return pymysql.connect(**DB_CONFIG)

def init_db():
    # Ensure database exists
    conn = pymysql.connect(host=DB_CONFIG["host"], user=DB_CONFIG["user"], password=DB_CONFIG["password"])
    conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    conn.close()

    # Sync Tables
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(255),
                email VARCHAR(255) UNIQUE NOT NULL,
                age INT, phone VARCHAR(50), address TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255),
                date VARCHAR(50), time VARCHAR(50),
                stress_level VARCHAR(50), emotion VARCHAR(50),
                score DECIMAL(5,2), audio_file VARCHAR(255),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    db.commit()
    db.close()

try:
    init_db()
    print("✓ Clinical Database Synchronized")
except Exception as e:
    print(f"✗ Database Error: {e}")

# --- AUTH HELPERS ---
def gen_token(email):
    payload = {"email": email, "exp": int(time.time()) + 28800}
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().replace("=", "")
    sig = hmac.new(ADMIN_SECRET.encode(), p.encode(), sha256).hexdigest()
    return f"{p}.{sig}"

def verify_token(token):
    try:
        p_part, sig = token.split('.')
        payload = json.loads(base64.urlsafe_b64decode(p_part + "=="))
        return payload if payload['exp'] > time.time() else None
    except: return None

# --- API ROUTES ---

@app.route("/api/upload", methods=["POST"])
def upload_audio():
    file = request.files.get('audio')
    user_email = request.form.get('user_email', 'guest@vocalvibe.pro')
    if not file: return jsonify({"status": "error", "message": "No audio"}), 400

    # Analysis Simulation
    score = random.uniform(20, 90)
    stress = "Low" if score < 45 else "Moderate" if score < 75 else "High"
    emo = random.choice(["Calm", "Focused"]) if score < 45 else random.choice(["Tense", "Anxious"])
    
    filename = f"{int(time.time())}_{user_email.replace('@','_')}.wav"
    file.save(str(UPLOAD_DIR / filename))

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO analysis_history (user_email, date, time, stress_level, emotion, score, audio_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_email, datetime.now().strftime("%b %d, %Y"), datetime.now().strftime("%I:%M %p"), stress, emo, score, filename))
    db.commit()
    db.close()
    return jsonify({"status": "success", "stress_level": stress, "score": round(score, 1), "emotion": emo})

@app.route("/api/dashboard-summary", methods=["GET"])
def get_summary():
    email = request.args.get('user_email')
    if not email: return jsonify({"total": 0, "distribution": {"low":0,"moderate":0,"high":0}})
    
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) as total, AVG(score) as avg FROM analysis_history WHERE user_email=%s", (email,))
        stats = cur.fetchone()
        cur.execute("SELECT stress_level, COUNT(*) as count FROM analysis_history WHERE user_email=%s GROUP BY stress_level", (email,))
        dist_rows = cur.fetchall()
    db.close()
    
    dist = {"low": 0, "moderate": 0, "high": 0}
    for r in dist_rows:
        lvl = r['stress_level'].lower()
        if lvl in dist: dist[lvl] = r['count']
        
    return jsonify({
        "total": stats['total'], 
        "avg_score": round(stats['avg'] or 0, 1), 
        "distribution": dist
    })

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.json
    if data.get("email") == ADMIN_EMAIL and data.get("password") == ADMIN_PASS:
        return jsonify({"status": "success", "token": gen_token(ADMIN_EMAIL)})
    return jsonify({"status": "error", "message": "Access Denied"}), 401

@app.route("/api/history", methods=["GET"])
def get_history():
    email = request.args.get('user_email')
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT * FROM analysis_history WHERE user_email=%s ORDER BY id DESC", (email,))
        rows = cur.fetchall()
    db.close()
    return jsonify(rows)

# --- STATIC FILE SERVING (FIXES 404 & JSON ERRORS) ---

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    # If path starts with api/, but wasn't caught by routes above, it's a 404
    if path.startswith("api/"):
        return jsonify({"error": "Not Found"}), 404
    
    # Check if the file exists in the frontend folder
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    
    # Otherwise, default to index (for SPA-style routing)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)