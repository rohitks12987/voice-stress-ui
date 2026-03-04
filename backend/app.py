import os
import json
import time
import hmac
import base64
import traceback
import functools # Needed for the security guard
import mimetypes
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pymysql
import pymysql.cursors
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from advanced_stress_predictor import AdvancedStressPredictor

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

MODEL_PATH_CANDIDATES = [
    PROJECT_ROOT / "models" / "voice_stress_model.pkl",
    BASE_DIR / "models" / "voice_stress_model.pkl",
]
STRESS_BASE_SCORES = {"low": 30.0, "moderate": 60.0, "high": 85.0}
STRESS_TO_EMOTION = {"low": "Calm", "moderate": "Tense", "high": "Anxious"}
_predictor = None
_active_model_path = None
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
REMOTE_AI_ENABLED = os.getenv("REMOTE_AI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
REMOTE_AI_PROVIDER = os.getenv("REMOTE_AI_PROVIDER", "huggingface").strip().lower()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "superb/wav2vec2-base-superb-er").strip()
REMOTE_AI_TIMEOUT_SEC = float(os.getenv("REMOTE_AI_TIMEOUT_SEC", "45"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
CHAT_FALLBACK_ENABLED = os.getenv("CHAT_FALLBACK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "gemini").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def _load_trainer():
    global _predictor, _active_model_path
    if _predictor is not None:
        return True
    available = [p for p in MODEL_PATH_CANDIDATES if p.exists()]
    if not available:
        return False

    # Prefer the most recently trained model if multiple files exist.
    available.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for model_path in available:
        try:
            _predictor = AdvancedStressPredictor(model_path=str(model_path), window_size=5)
            _active_model_path = str(model_path)
            app.logger.info("Loaded stress model from %s", _active_model_path)
            return True
        except Exception:
            continue
    return False


def _build_stress_score(stress_level, confidence):
    base = STRESS_BASE_SCORES.get(stress_level.lower(), 60.0)
    adjusted = base + (float(confidence) - 0.5) * 20.0
    return max(0.0, min(100.0, adjusted))

def _calibrate_stress_label(prediction):
    raw_label = str(prediction.get("stress_level", "MODERATE")).upper()
    confidence = float(prediction.get("confidence", 0.0))
    probs = prediction.get("probabilities") or {}

    high = float(probs.get("HIGH", 0.0))
    low = float(probs.get("LOW", 0.0))
    moderate = float(probs.get("MODERATE", 0.0))

    # Reduce false-positive HIGH on uncertain real-world recordings.
    if raw_label == "HIGH":
        if high < 0.60:
            if low >= high - 0.06:
                return "LOW", confidence
            if moderate >= 0.20:
                return "MODERATE", confidence
        if high < 0.68 and moderate >= 0.24:
            return "MODERATE", confidence

    # Reduce unstable LOW when probabilities are close.
    if raw_label == "LOW" and low < 0.58 and high >= low - 0.04:
        return "MODERATE", confidence

    return raw_label, confidence


def _resolve_upload_extension(file_obj):
    raw_name = secure_filename(file_obj.filename or "")
    ext = Path(raw_name).suffix.lower()
    if ext in ALLOWED_UPLOAD_EXTENSIONS:
        return ext
    return ".wav"

def _emotion_to_stress_level(emotion_label):
    emo = (emotion_label or "").strip().lower()
    if emo in {"angry", "anger", "fear", "fearful", "disgust", "disgusted"}:
        return "HIGH"
    if emo in {"sad", "surprised", "surprise"}:
        return "MODERATE"
    if emo in {"neutral", "calm", "happy"}:
        return "LOW"
    return "MODERATE"

def _predict_with_remote_ai(audio_path):
    if REMOTE_AI_PROVIDER != "huggingface":
        raise RuntimeError(f"Unsupported REMOTE_AI_PROVIDER: {REMOTE_AI_PROVIDER}")
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is missing")

    endpoint = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    content_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": content_type,
    }
    with open(audio_path, "rb") as f:
        payload = f.read()

    resp = requests.post(endpoint, headers=headers, data=payload, timeout=REMOTE_AI_TIMEOUT_SEC)
    if resp.status_code == 503:
        raise RuntimeError("Remote AI model is loading. Try again in a few seconds.")
    if not resp.ok:
        body = resp.text[:500]
        raise RuntimeError(f"Remote AI request failed ({resp.status_code}): {body}")

    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    if not isinstance(data, list) or not data:
        raise RuntimeError("Remote AI returned no prediction scores")

    scores = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        score = float(item.get("score", 0.0))
        if not label:
            continue
        scores.append((label, score))
    if not scores:
        raise RuntimeError("Remote AI output format is invalid")

    scores.sort(key=lambda x: x[1], reverse=True)
    top_label, top_score = scores[0]
    stress_level = _emotion_to_stress_level(top_label)

    stress_probs = {"HIGH": 0.0, "LOW": 0.0, "MODERATE": 0.0}
    for label, score in scores:
        stress_probs[_emotion_to_stress_level(label)] += float(score)

    duration = 0.0
    try:
        duration = round(float(_predictor.predict_long_audio(str(audio_path)).get("duration", 0.0)), 2) if _predictor else 0.0
    except Exception:
        duration = 0.0

    return {
        "stress_level": stress_level,
        "confidence": float(top_score),
        "probabilities": stress_probs,
        "duration": duration,
        "source": f"remote:{REMOTE_AI_PROVIDER}:{HF_MODEL_ID}",
    }

def _chat_fallback_reply(message):
    text = (message or "").lower()
    if any(k in text for k in ["stress", "anxiety", "tension"]):
        return "Stress can raise heart rate, muscle tension, and fatigue. Try 4-7-8 breathing for 2 minutes and re-check your voice scan."
    if any(k in text for k in ["sleep", "insomnia", "tired"]):
        return "For better sleep, keep a fixed bedtime, avoid caffeine after 2 PM, and reduce screen use 60 minutes before bed."
    if any(k in text for k in ["panic", "emergency", "suicide", "harm"]):
        return "If you are in immediate danger, call emergency services now. You can also use the SOS button in this app."
    if any(k in text for k in ["hello", "hi", "hey"]):
        return "Hello. I can help with stress, sleep, breathing routines, and understanding your scan trends."
    return "I can help with stress management, breathing routines, sleep hygiene, and how to interpret your scan history."

def _chat_with_gemini(message):
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in {"your_key_here", "your_real_gemini_key"}:
        raise RuntimeError("GEMINI_API_KEY is missing")

    configured = [m.strip() for m in GEMINI_MODEL.split(",") if m.strip()]
    model_candidates = []
    for model in configured:
        if model not in model_candidates:
            model_candidates.append(model)
    if not model_candidates:
        model_candidates = ["gemini-2.0-flash"]

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are a concise mental wellness assistant for a stress voice app. "
                            "Give safe, practical, non-diagnostic guidance in 2-4 short sentences. "
                            "If there is self-harm risk, advise immediate emergency help.\n\n"
                            f"User: {message}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 180},
    }

    errors = []
    for model in model_candidates:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(endpoint, json=payload, timeout=20)
        if resp.status_code == 429:
            raise RuntimeError(
                "Gemini quota exceeded (429). Enable billing or wait for quota reset: "
                "https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        if not resp.ok:
            errors.append(f"{model} -> {resp.status_code}: {resp.text[:220]}")
            continue

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            errors.append(f"{model} -> empty candidates")
            continue

        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        reply = " ".join([t.strip() for t in text_parts if t and t.strip()]).strip()
        if reply:
            return reply
        errors.append(f"{model} -> empty text")

    if not errors:
        errors = ["no model candidates were attempted"]
    raise RuntimeError(f"Gemini request failed: {' | '.join(errors)}")

def _chat_with_openai(message):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    if OPENAI_API_KEY.startswith("AIza"):
        raise RuntimeError("OPENAI_API_KEY is invalid (looks like a Gemini key). Use an OpenAI key starting with 'sk-'.")

    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.5,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise mental wellness assistant for a stress voice app. "
                    "Provide safe, practical, non-diagnostic guidance in 2-4 short sentences. "
                    "If self-harm risk appears, advise immediate emergency help."
                ),
            },
            {"role": "user", "content": message},
        ],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
    if not resp.ok:
        raise RuntimeError(f"OpenAI request failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices")
    content = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned empty text")
    return content

def _chat_with_provider(message):
    if CHAT_PROVIDER == "gemini":
        return _chat_with_gemini(message), "gemini"
    if CHAT_PROVIDER == "openai":
        return _chat_with_openai(message), "openai"
    raise RuntimeError(f"Unsupported CHAT_PROVIDER: {CHAT_PROVIDER}")

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
                password_hash VARCHAR(255),
                age INT, phone VARCHAR(50), address TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            SELECT COUNT(*) AS col_count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'users'
              AND COLUMN_NAME = 'password_hash'
        """, (DB_CONFIG["database"],))
        has_password_col = cur.fetchone()["col_count"] > 0
        if not has_password_col:
            cur.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)")
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

@app.route("/api/patient/register", methods=["POST"])
def patient_register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not full_name or not email or not password:
        return jsonify({"status": "error", "message": "Name, email, and password are required"}), 400
    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters"}), 400

    password_hash = generate_password_hash(password)
    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            existing = cur.fetchone()
            if existing:
                return jsonify({"status": "error", "message": "Email already registered"}), 409
            cur.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                (full_name, email, password_hash),
            )
        db.commit()
    finally:
        db.close()

    return jsonify({"status": "success", "message": "Registration completed", "user_email": email})

@app.route("/api/patient/login", methods=["POST"])
def patient_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute("SELECT full_name, email, password_hash FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
    finally:
        db.close()

    if not user or not user.get("password_hash"):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401
    if not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "user": {"name": user.get("full_name") or "", "email": user["email"]},
    })

@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"status": "error", "message": "Message is required"}), 400

    try:
        reply, source = _chat_with_provider(message)
        return jsonify({"status": "success", "reply": reply, "source": source})
    except Exception as e:
        if not CHAT_FALLBACK_ENABLED:
            return jsonify({
                "status": "error",
                "message": f"Real AI unavailable: {str(e)}",
            }), 503
        fallback = _chat_fallback_reply(message)
        return jsonify({
            "status": "success",
            "reply": fallback,
            "source": "fallback",
            "note": str(e),
        })
# ... (Keep existing imports/config from your original app.py) ...

@app.route("/api/upload", methods=["POST"])
def upload_audio():
    file = request.files.get('audio')
    user_email = request.form.get('user_email', 'guest@vocalvibe.pro')
    if not file: return jsonify({"status": "error", "message": "No audio"}), 400

    file_ext = _resolve_upload_extension(file)
    filename = f"{int(time.time())}_{secure_filename(user_email.replace('@','_'))}{file_ext}"
    saved_path = UPLOAD_DIR / filename
    file.save(str(saved_path))

    prediction = None
    analysis_source = "local"
    remote_error = None

    # Optional remote AI path (API gateway style). Falls back to local model.
    if REMOTE_AI_ENABLED:
        try:
            prediction = _predict_with_remote_ai(str(saved_path))
            analysis_source = prediction.get("source", "remote")
        except Exception as e:
            remote_error = str(e)
            app.logger.warning("Remote AI analysis failed, falling back to local model: %s", remote_error)

    if prediction is None:
        if not _load_trainer():
            return jsonify({"status": "error", "message": "Model not found"}), 503
        try:
            prediction = _predictor.predict_long_audio(str(saved_path))
            analysis_source = f"local:{Path(_active_model_path).name if _active_model_path else 'unknown'}"
        except Exception as e:
            app.logger.error("Audio analysis failed for %s: %s\n%s", str(saved_path), str(e), traceback.format_exc())
            err_msg = str(e).strip() or f"{type(e).__name__} with empty message"
            return jsonify({"status": "error", "message": f"Audio analysis failed: {err_msg}"}), 400

    stress_label, confidence = _calibrate_stress_label(prediction)
    stress = stress_label.capitalize()
    score = _build_stress_score(stress_label.lower(), confidence)
    emo = STRESS_TO_EMOTION.get(stress_label.lower(), "Focused")

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO analysis_history (user_email, date, time, stress_level, emotion, score, audio_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_email, datetime.now().strftime("%b %d, %Y"), datetime.now().strftime("%I:%M %p"), stress, emo, score, filename))
    db.commit()
    db.close()

    return jsonify({
        "status": "success",
        "stress_level": stress,
        "score": round(score, 1),
        "emotion": emo,
        "duration": round(prediction["duration"], 2),
        "model": Path(_active_model_path).name if _active_model_path else "unknown",
        "analysis_source": analysis_source,
        "confidence": round(float(prediction.get("confidence", 0.0)), 4),
        "probabilities": prediction.get("probabilities", {}),
        "remote_error": remote_error,
    })

# ... (Keep all other routes) ...

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
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 500))

    db = get_db()
    with db.cursor() as cur:
        if email:
            cur.execute(
                "SELECT * FROM analysis_history WHERE user_email=%s ORDER BY id DESC LIMIT %s",
                (email, limit),
            )
            rows = cur.fetchall()
        else:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.replace("Bearer ", "")
            staff_data = verify_token(token)
            if not staff_data or staff_data["email"] != ADMIN_EMAIL:
                db.close()
                return jsonify({"status": "error", "message": "user_email is required"}), 400
            cur.execute("SELECT * FROM analysis_history ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
    db.close()
    return jsonify(rows)

# --- STATIC FILE SERVING ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/admin_dashboard.html")
def admin_dashboard_alias():
    return send_from_directory(app.static_folder, "admin_dashboard.html")

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
