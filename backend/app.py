import os
import json
import time
import random
import hmac
import base64
import functools # Needed for the security guard
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# --- PATH CONFIGURATION ---
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
    "password": os.getenv("DB_PASSWORD", ""), 
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
    conn = pymysql.connect(host=DB_CONFIG["host"], user=DB_CONFIG["user"], password=DB_CONFIG["password"])
    conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    conn.close()

    db = get_db()
    with db.cursor() as cur:
        # 1. Patients/Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(255),
                email VARCHAR(255) UNIQUE NOT NULL,
                age INT, phone VARCHAR(50), address TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 2. Vocal Scan History
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
        # 3. CLINICAL ACCESS LOGS (For Biometric Gateway Audit)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                staff_email VARCHAR(255),
                action VARCHAR(255),
                ip_address VARCHAR(50),
                access_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    db.commit()
    db.close()

try:
    init_db()
    print("[OK] Clinical Database Synchronized & Secured")
except Exception as e:
    print(f"[ERROR] Database Error: {e}")

# --- SECURITY & AUTH HELPERS ---
def gen_token(email):
    # Generates a secure session token valid for 8 hours
    payload = {"email": email, "exp": int(time.time()) + 28800}
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().replace("=", "")
    sig = hmac.new(ADMIN_SECRET.encode(), p.encode(), sha256).hexdigest()
    return f"{p}.{sig}"

def verify_token(token):
    if not token: return None
    try:
        p_part, sig = token.split('.', 1)
        expected = hmac.new(ADMIN_SECRET.encode(), p_part.encode(), sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        padding = "=" * (-len(p_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(p_part + padding))
        return payload if payload['exp'] > time.time() else None
    except: return None

# --- THE CLINICAL GATEWAY GUARD (Decorator) ---
def staff_required(f):
    """Protects routes by requiring a valid staff token."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "")
        
        staff_data = verify_token(token)
        if not staff_data or staff_data['email'] != ADMIN_EMAIL:
            return jsonify({"status": "error", "message": "Secure Clinical Access Required"}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- API ROUTES ---

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    if data.get("email") == ADMIN_EMAIL and data.get("password") == ADMIN_PASS:
        # LOG THE SUCCESSFUL ACCESS
        db = get_db()
        with db.cursor() as cur:
            cur.execute("INSERT INTO access_logs (staff_email, action, ip_address) VALUES (%s, %s, %s)", 
                       (ADMIN_EMAIL, "Authorized Gateway Access", request.remote_addr))
        db.commit()
        db.close()
        return jsonify({"status": "success", "token": gen_token(ADMIN_EMAIL)})
    return jsonify({"status": "error", "message": "Biometric Credentials Mismatch"}), 401

@app.route("/api/admin/overview", methods=["GET"])
@staff_required
def admin_overview():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT user_email) as total_users FROM analysis_history")
        u_count = cur.fetchone()
        cur.execute("SELECT COUNT(*) as total_analyses, AVG(score) as avg_score FROM analysis_history")
        a_count = cur.fetchone()
        cur.execute("SELECT COUNT(*) as high_stress FROM analysis_history WHERE stress_level = 'High'")
        h_count = cur.fetchone()
    db.close()

    return jsonify({
        "status": "success",
        "counts": {
            "total_users": u_count['total_users'],
            "total_analyses": a_count['total_analyses'],
            "average_score": round(a_count['avg_score'] or 0, 1),
            "high_stress_count": h_count['high_stress']
        }
    })

@app.route("/api/admin/users", methods=["GET"])
@staff_required
def admin_list_users():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT u.*, COUNT(a.id) as analyses_count 
            FROM users u 
            LEFT JOIN analysis_history a ON u.email = a.user_email 
            GROUP BY u.email
        """)
        users = cur.fetchall()
    db.close()
    return jsonify({"status": "success", "users": users})

@app.route("/api/upload", methods=["POST"])
def upload_audio():
    file = request.files.get('audio')
    user_email = request.form.get('user_email', 'guest@vocalvibe.pro')
    if not file: return jsonify({"status": "error", "message": "No audio"}), 400

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
        
    return jsonify({"total": stats['total'], "avg_score": round(stats['avg'] or 0, 1), "distribution": dist})

@app.route("/api/history", methods=["GET"])
def get_history():
    email = request.args.get('user_email')
    if not email:
        return jsonify({"status": "error", "message": "user_email is required"}), 400
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT * FROM analysis_history WHERE user_email=%s ORDER BY id DESC", (email,))
        rows = cur.fetchall()
    db.close()
    return jsonify(rows)

# --- STATIC FILE SERVING ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/admin_dashboard.html")
def admin_dashboard_alias():
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route("/<path:path>")
def serve_static(path):
    if path.startswith("api/"):
        return jsonify({"error": "Resource Not Found"}), 404
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
