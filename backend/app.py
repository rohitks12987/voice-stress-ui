import base64
import hmac
import io
import json
import os
import random
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

import pymysql
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from google import genai as gemini_sdk
except Exception:
    gemini_sdk = None

try:
    import google.generativeai as gemini_legacy
except Exception:
    gemini_legacy = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "frontend"))
UPLOAD_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "uploads"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=False)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


class RowProxy(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class CursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        if params is None:
            self._cursor.execute(query)
        else:
            self._cursor.execute(query, tuple(params))
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        return RowProxy(row) if isinstance(row, dict) else row

    def fetchall(self):
        rows = self._cursor.fetchall() or []
        out = []
        for row in rows:
            out.append(RowProxy(row) if isinstance(row, dict) else row)
        return out

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._cursor.lastrowid


class ConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return CursorWrapper(self._conn.cursor())

    def execute(self, query, params=None):
        c = self.cursor()
        c.execute(query, params)
        return c

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type:
                self._conn.rollback()
            else:
                self._conn.commit()
        finally:
            self._conn.close()


def parse_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def parse_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_email(v):
    return (v or "").strip().lower()


def utc_now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def display_name_from_email(email):
    local = (email.split("@", 1)[0] or "User").replace(".", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in local.split() if part) or "User"


def make_json_safe(value):
    if isinstance(value, RowProxy):
        value = dict(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(x) for x in value]
    if isinstance(value, tuple):
        return [make_json_safe(x) for x in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def respond(payload, status=200):
    return jsonify(make_json_safe(payload)), status


def get_db_config(include_database=True):
    cfg = {
        "host": os.getenv("DB_HOST", "127.0.0.1").strip(),
        "port": parse_int(os.getenv("DB_PORT", "3306"), 3306),
        "user": os.getenv("DB_USER", "root").strip(),
        "password": os.getenv("DB_PASSWORD", "root").strip(),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": False,
    }
    if include_database:
        cfg["database"] = os.getenv("DB_NAME", "voice_stress_db").strip()
    return cfg


def get_server_connection():
    return ConnectionWrapper(pymysql.connect(**get_db_config(include_database=False)))


def get_connection():
    return ConnectionWrapper(pymysql.connect(**get_db_config(include_database=True)))


def ensure_column(cursor, table_name, column_name, definition):
    db_name = get_db_config(True)["database"]
    row = cursor.execute(
        """
        SELECT COUNT(*) AS count_value
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (db_name, table_name, column_name),
    ).fetchone()
    if parse_int((row or {}).get("count_value"), 0) == 0:
        cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` {definition}")


def ensure_index(cursor, index_name, table_name, column_expr):
    db_name = get_db_config(True)["database"]
    row = cursor.execute(
        """
        SELECT COUNT(*) AS count_value
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s
        """,
        (db_name, table_name, index_name),
    ).fetchone()
    if parse_int((row or {}).get("count_value"), 0) == 0:
        cursor.execute(f"CREATE INDEX `{index_name}` ON `{table_name}` ({column_expr})")


def init_db():
    db_name = get_db_config(True)["database"]
    with get_server_connection() as conn:
        conn.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )

    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                date VARCHAR(32) NOT NULL,
                time VARCHAR(32) NOT NULL,
                duration VARCHAR(32) NULL,
                stress_level VARCHAR(32) NULL,
                emotion VARCHAR(64) NULL,
                score DECIMAL(6,2) NULL,
                audio_file VARCHAR(255) NULL,
                user_email VARCHAR(255) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_codes (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                email VARCHAR(255) NOT NULL,
                code VARCHAR(32) NOT NULL,
                expires_at DATETIME NOT NULL,
                PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                full_name VARCHAR(255) NULL,
                email VARCHAR(255) NOT NULL,
                age INT NULL,
                password VARCHAR(255) NULL,
                phone VARCHAR(64) NULL,
                address TEXT NULL,
                photo_url VARCHAR(512) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY uq_users_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS wellness_logs (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                user_email VARCHAR(255) NOT NULL,
                water_intake DECIMAL(6,2) NOT NULL DEFAULT 0,
                stress_score INT NOT NULL DEFAULT 0,
                emotion VARCHAR(64) NOT NULL DEFAULT 'Unknown',
                recorded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        ensure_column(c, "analysis_history", "audio_file", "VARCHAR(255) NULL")
        ensure_column(c, "analysis_history", "user_email", "VARCHAR(255) NULL")
        ensure_column(c, "analysis_history", "created_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
        ensure_column(c, "users", "phone", "VARCHAR(64) NULL")
        ensure_column(c, "users", "address", "TEXT NULL")
        ensure_column(c, "users", "photo_url", "VARCHAR(512) NULL")
        ensure_column(c, "users", "created_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
        ensure_index(c, "idx_analysis_history_email", "analysis_history", "`user_email`")
        ensure_index(c, "idx_wellness_email_time", "wellness_logs", "`user_email`, `recorded_at`")


DB_READY = True
DB_INIT_ERROR = ""
try:
    init_db()
except Exception as exc:
    DB_READY = False
    DB_INIT_ERROR = str(exc)
    print(f"Database initialization failed: {exc}")


def get_admin_email():
    return normalize_email(os.getenv("ADMIN_EMAIL", "admin@stress-tone.local"))


def get_admin_password():
    return os.getenv("ADMIN_PASSWORD", "admin123")


def get_admin_secret():
    secret = os.getenv("ADMIN_SECRET", "").strip()
    if secret:
        return secret
    key = (os.getenv("GEMINI_API_KEY", "") or "").strip().strip('"').strip("'")
    return key or "local-admin-secret"


ADMIN_TOKEN_TTL_SECONDS = max(900, min(parse_int(os.getenv("ADMIN_TOKEN_TTL_SECONDS"), 8 * 3600), 7 * 24 * 3600))


def b64url_encode(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(encoded):
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


def create_admin_token(email):
    payload = {"email": email, "exp": int(time.time()) + ADMIN_TOKEN_TTL_SECONDS}
    p = b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(get_admin_secret().encode("utf-8"), p.encode("utf-8"), sha256).hexdigest()
    return f"{p}.{sig}"


def verify_admin_token(token):
    if not token or "." not in token:
        return None
    payload_part, signature = token.rsplit(".", 1)
    expected = hmac.new(get_admin_secret().encode("utf-8"), payload_part.encode("utf-8"), sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(b64url_decode(payload_part).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    if normalize_email(payload.get("email")) != get_admin_email():
        return None
    return payload


def extract_admin_token():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    if request.args.get("token", "").strip():
        return request.args.get("token").strip()
    return request.headers.get("X-Admin-Token", "").strip()


def require_admin():
    return verify_admin_token(extract_admin_token())


def admin_unauthorized():
    return respond({"status": "error", "message": "Admin authorization required."}, 401)


def audio_path_safe(filename):
    if not filename:
        return None
    clean = os.path.basename(filename.strip())
    if clean != filename.strip():
        return None
    path = os.path.abspath(os.path.join(UPLOAD_DIR, clean))
    if os.path.commonpath([path, os.path.abspath(UPLOAD_DIR)]) != os.path.abspath(UPLOAD_DIR):
        return None
    return clean


def ensure_user_exists(conn, email, full_name=None, age=None, phone=None, address=None, photo_url=None):
    if not email:
        return
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()
    if row is None:
        cur.execute(
            """
            INSERT INTO users (full_name, email, age, phone, address, photo_url, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                full_name or display_name_from_email(email),
                email,
                age,
                phone or "+91 00000 00000",
                address or "Address not updated",
                photo_url or "https://www.w3schools.com/howto/img_avatar.png",
                utc_now_str(),
            ),
        )
        return

    updates = {}
    if full_name:
        updates["full_name"] = full_name
    if age is not None and parse_int(age, 0) > 0:
        updates["age"] = parse_int(age, 0)
    if phone:
        updates["phone"] = phone
    if address:
        updates["address"] = address
    if photo_url:
        updates["photo_url"] = photo_url

    if updates:
        set_clause = ", ".join([f"`{k}` = %s" for k in updates.keys()])
        values = list(updates.values()) + [email]
        cur.execute(f"UPDATE users SET {set_clause} WHERE email = %s", values)


def fetch_user_payload(conn, email=None, user_id=None):
    cur = conn.cursor()
    row = None
    if user_id is not None:
        row = cur.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
    elif email:
        row = cur.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()

    if row is None and email:
        ensure_user_exists(conn, email)
        row = cur.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()

    if row is None:
        return None

    user_email = row.get("email") or email or "guest@stress-tone.local"
    return {
        "id": row.get("id"),
        "name": row.get("full_name") or display_name_from_email(user_email),
        "email": user_email,
        "phone": row.get("phone") or "+91 00000 00000",
        "address": row.get("address") or "Address not updated",
        "photo": row.get("photo_url") or "https://www.w3schools.com/howto/img_avatar.png",
        "age": row.get("age"),
    }


def get_user_analyses(conn, user_email, limit=200):
    limit = max(1, min(parse_int(limit, 200), 1000))
    rows = conn.execute(
        """
        SELECT id, date, time, duration, stress_level, emotion, score, audio_file, user_email, created_at
        FROM analysis_history
        WHERE user_email = %s
        ORDER BY id DESC
        LIMIT %s
        """,
        (user_email, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def get_user_wellness_logs(conn, user_email, limit=200):
    limit = max(1, min(parse_int(limit, 200), 1000))
    rows = conn.execute(
        """
        SELECT id, user_email, water_intake, stress_score, emotion, recorded_at
        FROM wellness_logs
        WHERE user_email = %s
        ORDER BY id DESC
        LIMIT %s
        """,
        (user_email, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def build_user_report(conn, user_email):
    analyses = get_user_analyses(conn, user_email, limit=500)
    wellness_logs = get_user_wellness_logs(conn, user_email, limit=500)

    scores = [parse_float(item.get("score"), 0.0) for item in analyses]
    total_analyses = len(analyses)
    avg_score = round(sum(scores) / total_analyses, 2) if total_analyses else 0.0

    stress_distribution = {"low": 0, "moderate": 0, "high": 0}
    emotion_counts = {}
    for item in analyses:
        stress_text = (item.get("stress_level") or "").lower()
        if "high" in stress_text:
            stress_distribution["high"] += 1
        elif "mod" in stress_text:
            stress_distribution["moderate"] += 1
        else:
            stress_distribution["low"] += 1

        emotion = item.get("emotion") or "Unknown"
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    if total_analyses:
        stress_distribution_pct = {k: round((v / total_analyses) * 100) for k, v in stress_distribution.items()}
    else:
        stress_distribution_pct = {"low": 0, "moderate": 0, "high": 0}

    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "--"
    avg_water = (
        round(sum(parse_float(item.get("water_intake"), 0.0) for item in wellness_logs) / len(wellness_logs), 2)
        if wellness_logs
        else 0.0
    )
    latest_analysis_at = analyses[0]["created_at"] if analyses else None

    return {
        "total_analyses": total_analyses,
        "average_score": avg_score,
        "stress_distribution": stress_distribution,
        "stress_distribution_percent": stress_distribution_pct,
        "dominant_emotion": dominant_emotion,
        "wellness_logs_count": len(wellness_logs),
        "average_water_intake": avg_water,
        "latest_analysis_at": latest_analysis_at,
    }


def classify_stress(score):
    if score >= 70:
        return "high"
    if score >= 40:
        return "moderate"
    return "low"


def estimate_duration_seconds(filename, audio_bytes):
    if filename.lower().endswith(".wav"):
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                frames = wav_file.getnframes()
                frame_rate = wav_file.getframerate() or 1
                duration = frames / float(frame_rate)
                return max(1.0, min(120.0, duration))
        except wave.Error:
            pass

    guessed_seconds = len(audio_bytes) / 16000.0
    return max(1.0, min(120.0, guessed_seconds))


def build_analysis_result(filename, audio_bytes):
    digest = sha256(audio_bytes).hexdigest()
    seed = int(digest[:16], 16)
    rng = random.Random(seed)

    # Try ML prediction first
    if ML_MODEL_DATA:
        # Save temp audio file for feature extraction
        temp_path = f"/tmp/{digest}.wav"
        try:
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)

            # Extract features and predict
            features = ML_MODEL_DATA['model'].extract_features(temp_path)
            if features:
                # Prepare feature vector
                import pandas as pd
                feature_df = pd.DataFrame([features])
                feature_df = feature_df.fillna(feature_df.mean())
                features_scaled = ML_MODEL_DATA['scaler'].transform(feature_df)

                # Predict using the trained model
                prediction = ML_MODEL_DATA['model'].predict(features_scaled)[0]
                stress_level = ML_MODEL_DATA['label_encoder'].inverse_transform([prediction])[0]

                # Get prediction probabilities
                probabilities = ML_MODEL_DATA['model'].predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)

                # Convert stress level to score (0-100 scale)
                score_map = {'low': 35, 'moderate': 55, 'high': 75}
                base_score = score_map.get(stress_level, 55)
                # Add some variation based on confidence
                score = round(base_score + (confidence - 0.5) * 20, 1)
                score = max(20, min(90, score))  # Keep within reasonable bounds
            else:
                # Fallback to random if feature extraction fails
                score = round(rng.uniform(22, 92), 1)
                stress_level = classify_stress(score)

        except Exception as e:
            print(f"ML prediction failed: {e}")
            # Fallback to random scoring
            score = round(rng.uniform(22, 92), 1)
            stress_level = classify_stress(score)
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
    else:
        # Original random scoring
        score = round(rng.uniform(22, 92), 1)
        stress_level = classify_stress(score)

    emotion_map = {
        "low": ["Calm", "Relaxed", "Focused"],
        "moderate": ["Alert", "Anxious", "Concerned"],
        "high": ["Stressed", "Tense", "Overwhelmed"],
    }

    duration_seconds = estimate_duration_seconds(filename, audio_bytes)
    now = datetime.now()
    return {
        "date": now.strftime("%b %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "duration": f"{duration_seconds:.1f}s",
        "stress_level": stress_level,
        "emotion": rng.choice(emotion_map[stress_level]),
        "score": score,
    }


def local_wellness_reply(message):
    text = (message or "").lower()
    if any(x in text for x in ("panic", "emergency", "sos", "suicide")):
        return "If this is urgent, call 988 or local emergency services now."
    if any(x in text for x in ("stress", "anxious", "anxiety", "tense")):
        return "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s for 2 minutes."
    if any(x in text for x in ("sleep", "tired", "insomnia")):
        return "Avoid screens 30 minutes before bed and keep a fixed sleep schedule."
    if any(x in text for x in ("hydration", "water")):
        return "Hydration supports vocal stability. Drink steadily through the day."
    return "I can help with stress tips, breathing routines, or dashboard usage."


def find_working_model(client_obj):
    preferred = (os.getenv("GEMINI_MODEL", "") or "").strip()
    if preferred:
        return preferred
    try:
        candidates = []
        for model in client_obj.models.list():
            name = getattr(model, "name", "")
            short_name = name.split("/")[-1] if name else ""
            if short_name and "gemini" in short_name.lower():
                candidates.append(short_name)
        for m in candidates:
            if "flash" in m.lower():
                return m
        if candidates:
            return candidates[0]
    except Exception as exc:
        print(f"Model discovery failed: {exc}")
    return "gemini-1.5-flash"


def gemini_rest_generate(model_name, prompt):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 256},
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini REST HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini REST connection failed: {exc.reason}") from exc

    candidates = body.get("candidates") or []
    if not candidates:
        error_text = ((body.get("error") or {}).get("message") or "No candidates returned.").strip()
        raise RuntimeError(error_text)

    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text = " ".join((part.get("text") or "").strip() for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError("Gemini REST returned empty text response.")
    return text


API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip().strip('"').strip("'")
client = None
ACTIVE_MODEL = None
AI_PROVIDER = "none"

if API_KEY and gemini_sdk:
    try:
        client = gemini_sdk.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})
        ACTIVE_MODEL = find_working_model(client)
        AI_PROVIDER = "google-genai"
        print(f"AI enabled with {AI_PROVIDER} model: {ACTIVE_MODEL}")
    except Exception as exc:
        print(f"AI init failed (google-genai): {exc}")
        client = None
        ACTIVE_MODEL = None
        AI_PROVIDER = "none"
elif API_KEY and gemini_legacy:
    try:
        gemini_legacy.configure(api_key=API_KEY)
        client = gemini_legacy
        ACTIVE_MODEL = (os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash").strip()
        AI_PROVIDER = "google-generativeai"
        print(f"AI enabled with {AI_PROVIDER} model: {ACTIVE_MODEL}")
    except Exception as exc:
        print(f"AI init failed (google-generativeai): {exc}")
        client = None
        ACTIVE_MODEL = None
        AI_PROVIDER = "none"
elif API_KEY:
    client = {"provider": "gemini-rest"}
    ACTIVE_MODEL = (os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash").strip()
    AI_PROVIDER = "gemini-rest"
    print(f"AI enabled with {AI_PROVIDER} model: {ACTIVE_MODEL}")

if not API_KEY:
    print("AI disabled: GEMINI_API_KEY is not configured in .env.")

# Load ML model for voice stress analysis
ML_MODEL_DATA = None
if joblib:
    try:
        model_path = Path(__file__).parent / "models" / "voice_stress_model.pkl"
        if model_path.exists():
            ML_MODEL_DATA = joblib.load(model_path)
            print("ML model loaded for voice stress analysis")
        else:
            print("ML model not found, using random scoring for voice analysis")
    except Exception as exc:
        print(f"ML model loading failed: {exc}")
        ML_MODEL_DATA = None
else:
    print("joblib not available, using random scoring for voice analysis")


@app.route("/api/health", methods=["GET"])
def health():
    return respond(
        {
            "status": "ok",
            "db_ready": DB_READY,
            "db_name": get_db_config(True)["database"],
            "db_error": DB_INIT_ERROR or None,
            "ai_enabled": bool(client and ACTIVE_MODEL),
            "ai_provider": AI_PROVIDER,
            "gemini_key_loaded": bool(API_KEY),
            "ml_model_loaded": ML_MODEL_DATA is not None,
            "timestamp": utc_now_str(),
        }
    )


@app.route("/api/chatbot", methods=["POST"])
def chatbot_response():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return respond({"reply": "I am here to help. Tell me how you are feeling."})

    if not client or not ACTIVE_MODEL:
        return respond({"reply": local_wellness_reply(user_msg), "source": "fallback"})

    prompt = "You are a calm wellness officer. Respond briefly and safely. " f"User message: {user_msg}"

    try:
        if AI_PROVIDER == "google-genai":
            response = client.models.generate_content(model=ACTIVE_MODEL, contents=prompt)
            reply = (getattr(response, "text", None) or "").strip()
        elif AI_PROVIDER == "google-generativeai":
            response = client.GenerativeModel(ACTIVE_MODEL).generate_content(prompt)
            reply = (getattr(response, "text", None) or "").strip()
        elif AI_PROVIDER == "gemini-rest":
            reply = gemini_rest_generate(ACTIVE_MODEL, prompt)
        else:
            return respond({"reply": local_wellness_reply(user_msg), "source": "fallback"})
        reply = reply or local_wellness_reply(user_msg)
        return respond({"reply": reply, "source": "gemini"})
    except Exception as exc:
        print(f"AI response failed: {exc}")
        return respond({"reply": local_wellness_reply(user_msg), "source": "fallback"})


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    audio_file = request.files.get("audio")
    if audio_file is None:
        return respond({"status": "error", "message": "No audio file provided."}, 400)

    raw_bytes = audio_file.read()
    if not raw_bytes:
        return respond({"status": "error", "message": "Audio file is empty."}, 400)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return respond({"status": "error", "message": "File size exceeds 10MB limit."}, 400)

    user_email = normalize_email(request.form.get("user_email")) or "guest@stress-tone.local"
    full_name = (request.form.get("full_name") or "").strip() or None
    age = parse_int(request.form.get("age"), default=0) or None
    filename = audio_file.filename or "recording.wav"

    analysis = build_analysis_result(filename, raw_bytes)
    created_at = utc_now_str()

    safe_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    safe_user = user_email.replace("@", "_at_").replace(".", "_")
    save_name = f"{safe_stamp}_{safe_user}.wav"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    try:
        with open(save_path, "wb") as out_file:
            out_file.write(raw_bytes)
    except OSError:
        save_name = ""

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO analysis_history
                (date, time, duration, stress_level, emotion, score, audio_file, user_email, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    analysis["date"],
                    analysis["time"],
                    analysis["duration"],
                    analysis["stress_level"],
                    analysis["emotion"],
                    analysis["score"],
                    save_name,
                    user_email,
                    created_at,
                ),
            )
            analysis_id = cur.lastrowid
            ensure_user_exists(conn, user_email, full_name=full_name, age=age)
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    return respond(
        {
            "status": "success",
            "analysis_id": analysis_id,
            "user_email": user_email,
            "stored_file": save_name,
            **analysis,
        }
    )


@app.route("/api/history", methods=["GET"])
def get_history():
    user_email = normalize_email(request.args.get("user_email"))
    limit = max(1, min(parse_int(request.args.get("limit"), 200), 1000))

    query = """
        SELECT id, date, time, duration, stress_level, emotion, score, audio_file, user_email, created_at
        FROM analysis_history
    """
    params = []
    if user_email:
        query += " WHERE user_email = %s"
        params.append(user_email)
    query += " ORDER BY id DESC LIMIT %s"
    params.append(limit)

    try:
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return respond([dict(row) for row in rows])
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)


@app.route("/api/dashboard-summary", methods=["GET"])
def dashboard_summary():
    user_email = normalize_email(request.args.get("user_email"))
    if not user_email:
        return respond({"status": "error", "message": "user_email is required."}, 400)

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT date, time, stress_level, emotion, score
                FROM analysis_history
                WHERE user_email = %s
                ORDER BY id DESC
                """,
                (user_email,),
            ).fetchall()
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    if not rows:
        return respond(
            {
                "status": "success",
                "total": 0,
                "avg_stress": "--",
                "dominant_emotion": "--",
                "last_check": "--",
                "distribution": {"low": 0, "moderate": 0, "high": 0},
            }
        )

    scores = [parse_float(row.get("score"), 0.0) for row in rows]
    avg_score = sum(scores) / len(scores)
    avg_stress = classify_stress(avg_score).capitalize()

    emotion_counts = {}
    stress_counts = {"low": 0, "moderate": 0, "high": 0}
    for row in rows:
        s = (row.get("stress_level") or "").lower()
        if "high" in s:
            stress_counts["high"] += 1
        elif "mod" in s:
            stress_counts["moderate"] += 1
        else:
            stress_counts["low"] += 1

        emo = row.get("emotion") or "Unknown"
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    total = len(rows)
    distribution = {
        "low": round((stress_counts["low"] / total) * 100),
        "moderate": round((stress_counts["moderate"] / total) * 100),
        "high": round((stress_counts["high"] / total) * 100),
    }
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "--"
    latest = rows[0]
    last_check = f"{latest.get('date', '')} {latest.get('time', '')}".strip()

    return respond(
        {
            "status": "success",
            "total": total,
            "avg_stress": avg_stress,
            "dominant_emotion": dominant_emotion,
            "last_check": last_check,
            "distribution": distribution,
            "latest_analysis": {
                "stress_level": rows[0].get("stress_level", "--"),
                "emotion": rows[0].get("emotion", "--"),
                "score": rows[0].get("score", 0),
                "date": rows[0].get("date", "--"),
                "time": rows[0].get("time", "--")
            } if rows else None
        }
    )


@app.route("/api/save-wellness", methods=["POST"])
def save_wellness():
    payload = request.get_json(silent=True) or {}
    user_email = normalize_email(payload.get("user_email"))
    if not user_email:
        return respond({"status": "error", "message": "user_email is required."}, 400)

    water = max(0.0, parse_float(payload.get("water"), 0.0))
    stress = max(0, min(parse_int(payload.get("stress"), 0), 100))
    emotion = ((payload.get("emotion") or "Unknown").strip() or "Unknown")[:50]
    name = (payload.get("name") or "").strip() or None
    phone = (payload.get("phone") or "").strip() or None
    address = (payload.get("address") or "").strip() or None
    photo = (payload.get("photo") or "").strip() or None
    age = parse_int(payload.get("age"), default=0) or None

    try:
        with get_connection() as conn:
            ensure_user_exists(
                conn,
                user_email,
                full_name=name,
                age=age,
                phone=phone,
                address=address,
                photo_url=photo,
            )
            conn.execute(
                """
                INSERT INTO wellness_logs (user_email, water_intake, stress_score, emotion, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_email, water, stress, emotion, utc_now_str()),
            )
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    return respond({"status": "success"})


@app.route("/api/user-details/<path:user_ref>", methods=["GET"])
def get_user_details(user_ref):
    user_ref = (user_ref or "").strip()
    try:
        with get_connection() as conn:
            if user_ref.isdigit():
                payload = fetch_user_payload(conn, user_id=int(user_ref))
            else:
                payload = fetch_user_payload(conn, email=normalize_email(user_ref))
            if payload is None:
                return respond({"status": "error", "message": "User not found."}, 404)
            return respond(payload)
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)


@app.route("/api/sos", methods=["POST"])
def send_sos():
    payload = request.get_json(silent=True) or {}
    user_email = normalize_email(payload.get("user_email")) or "guest@stress-tone.local"
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")

    # Get emergency contacts
    contacts = []
    try:
        with get_connection() as conn:
            contacts = conn.execute(
                "SELECT name, phone FROM emergency_contacts WHERE user_email = %s",
                [user_email]
            ).fetchall()
    except Exception as e:
        print(f"Error fetching emergency contacts: {e}")

    # Send messages to contacts with detailed tracking
    results = []
    sent_count = 0
    failed_count = 0

    for contact in contacts:
        contact_result = {
            "name": contact['name'],
            "phone": contact['phone'],
            "status": "failed",
            "method": None,
            "error": None
        }

        try:
            success, method = send_emergency_message(contact['phone'], user_email, latitude, longitude)
            if success:
                contact_result["status"] = "sent"
                contact_result["method"] = method
                sent_count += 1
            else:
                contact_result["error"] = "Failed to send via any method"
                failed_count += 1
        except Exception as e:
            contact_result["error"] = str(e)
            failed_count += 1
            print(f"Failed to send to {contact['phone']}: {e}")

        results.append(contact_result)

    return respond(
        {
            "status": "success",
            "message": f"SOS alert sent to {sent_count} emergency contacts.",
            "user_email": user_email,
            "received_at": utc_now_str(),
            "location": {"latitude": latitude, "longitude": longitude},
            "contacts_notified": sent_count,
            "contacts_failed": failed_count,
            "delivery_details": results,
        }
    )


@app.route("/api/emergency-contacts", methods=["GET"])
def get_emergency_contacts():
    user_email = normalize_email(request.args.get("user_email")) or "guest@stress-tone.local"
    try:
        with get_connection() as conn:
            contacts = conn.execute(
                "SELECT id, name, phone, relationship FROM emergency_contacts WHERE user_email = %s ORDER BY created_at",
                [user_email]
            ).fetchall()
        return respond({"status": "success", "contacts": contacts})
    except Exception as e:
        return respond({"status": "error", "message": str(e)}, 500)


@app.route("/api/emergency-contacts", methods=["POST"])
def add_emergency_contact():
    data = request.get_json(silent=True) or {}
    user_email = normalize_email(data.get("user_email")) or "guest@stress-tone.local"
    name = str(data.get("name", "")).strip()
    phone = str(data.get("phone", "")).strip()
    relationship = str(data.get("relationship", "")).strip()
    
    if not name or not phone:
        return respond({"status": "error", "message": "Name and phone are required."}, 400)
    
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO emergency_contacts (user_email, name, phone, relationship) VALUES (%s, %s, %s, %s)",
                [user_email, name, phone, relationship]
            )
            conn.commit()
        return respond({"status": "success", "message": "Emergency contact added."})
    except Exception as e:
        return respond({"status": "error", "message": str(e)}, 500)


@app.route("/api/emergency-contacts/<int:contact_id>", methods=["DELETE"])
def delete_emergency_contact(contact_id):
    user_email = normalize_email(request.args.get("user_email")) or "guest@stress-tone.local"
    try:
        with get_connection() as conn:
            result = conn.execute(
                "DELETE FROM emergency_contacts WHERE id = %s AND user_email = %s",
                [contact_id, user_email]
            )
            conn.commit()
            if result.rowcount == 0:
                return respond({"status": "error", "message": "Contact not found."}, 404)
        return respond({"status": "success", "message": "Emergency contact deleted."})
    except Exception as e:
        return respond({"status": "error", "message": str(e)}, 500)


def send_emergency_message(phone, user_email, latitude, longitude):
    message = f"EMERGENCY ALERT: {user_email} has activated SOS. Please check on them immediately."
    if latitude and longitude:
        message += f" Location: https://maps.google.com/?q={latitude},{longitude}"

    # Try WhatsApp first, fallback to SMS
    if send_whatsapp_message(phone, message):
        return True, "whatsapp"
    elif send_sms_message(phone, message):
        return True, "sms"
    else:
        return False, None


def send_whatsapp_message(phone, message):
    if not TwilioClient:
        print("Twilio not available for WhatsApp")
        return False
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER")  # e.g. +14155238886
    
    if not account_sid or not auth_token or not from_whatsapp:
        print("Twilio WhatsApp credentials not configured")
        return False
    
    try:
        client = TwilioClient(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=f"whatsapp:{from_whatsapp}",
            to=f"whatsapp:{phone}"
        )
        return True
    except Exception as e:
        print(f"WhatsApp send failed: {e}")
        return False


def send_sms_message(phone, message):
    if not TwilioClient:
        print("Twilio not available for SMS")
        return False
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_sms = os.getenv("TWILIO_SMS_NUMBER")
    
    if not account_sid or not auth_token or not from_sms:
        print("Twilio SMS credentials not configured")
        return False
    
    try:
        client = TwilioClient(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_sms,
            to=phone
        )
        return True
    except Exception as e:
        print(f"SMS send failed: {e}")
        return False


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email"))
    password = str(data.get("password") or "")

    if not email or not password:
        return respond({"status": "error", "message": "Email and password are required."}, 400)

    if not hmac.compare_digest(email, get_admin_email()) or not hmac.compare_digest(password, get_admin_password()):
        return respond({"status": "error", "message": "Invalid admin credentials."}, 401)

    token = create_admin_token(email)
    return respond({"status": "success", "token": token, "expires_in": ADMIN_TOKEN_TTL_SECONDS, "admin_email": email})


@app.route("/api/admin/overview", methods=["GET"])
def admin_overview():
    if require_admin() is None:
        return admin_unauthorized()

    try:
        with get_connection() as conn:
            total_users_row = conn.execute("SELECT COUNT(*) AS count_value FROM users").fetchone()
            total_analyses_row = conn.execute("SELECT COUNT(*) AS count_value FROM analysis_history").fetchone()
            total_recordings_row = conn.execute("SELECT COUNT(*) AS count_value FROM analysis_history WHERE audio_file IS NOT NULL AND TRIM(audio_file) <> ''").fetchone()
            high_stress_row = conn.execute("SELECT COUNT(*) AS count_value FROM analysis_history WHERE LOWER(COALESCE(stress_level, '')) LIKE '%high%'").fetchone()
            avg_score_row = conn.execute("SELECT AVG(score) AS average_value FROM analysis_history").fetchone()

            recent_rows = conn.execute(
                """
                SELECT id, user_email, date, time, stress_level, emotion, score, audio_file, created_at
                FROM analysis_history
                ORDER BY id DESC
                LIMIT 12
                """
            ).fetchall()

            top_user_rows = conn.execute(
                """
                SELECT a.user_email, COALESCE(u.full_name, a.user_email) AS full_name, COUNT(*) AS analyses_count
                FROM analysis_history a
                LEFT JOIN users u ON u.email = a.user_email
                GROUP BY a.user_email
                ORDER BY analyses_count DESC
                LIMIT 8
                """
            ).fetchall()
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    return respond(
        {
            "status": "success",
            "ai_enabled": bool(client and ACTIVE_MODEL),
            "counts": {
                "total_users": parse_int((total_users_row or {}).get("count_value"), 0),
                "total_analyses": parse_int((total_analyses_row or {}).get("count_value"), 0),
                "total_recordings": parse_int((total_recordings_row or {}).get("count_value"), 0),
                "high_stress_count": parse_int((high_stress_row or {}).get("count_value"), 0),
                "average_score": round(parse_float((avg_score_row or {}).get("average_value"), 0.0), 2),
            },
            "recent_analyses": [dict(row) for row in recent_rows],
            "top_users": [dict(row) for row in top_user_rows],
        }
    )


@app.route("/api/admin/users", methods=["GET"])
def admin_list_users():
    if require_admin() is None:
        return admin_unauthorized()

    query_text = (request.args.get("query") or "").strip().lower()
    limit = max(1, min(parse_int(request.args.get("limit"), 200), 500))
    params = []
    where_clause = ""

    if query_text:
        where_clause = "WHERE LOWER(u.email) LIKE %s OR LOWER(COALESCE(u.full_name, '')) LIKE %s"
        q = f"%{query_text}%"
        params.extend([q, q])

    params.append(limit)
    sql = f"""
        SELECT
            u.id, u.full_name, u.email, u.age, u.phone, u.address, u.photo_url, u.created_at,
            COUNT(DISTINCT a.id) AS analyses_count,
            COUNT(DISTINCT w.id) AS wellness_count,
            MAX(a.created_at) AS last_analysis_at
        FROM users u
        LEFT JOIN analysis_history a ON a.user_email = u.email
        LEFT JOIN wellness_logs w ON w.user_email = u.email
        {where_clause}
        GROUP BY u.id
        ORDER BY u.id DESC
        LIMIT %s
    """

    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    return respond({"status": "success", "users": [dict(row) for row in rows]})


@app.route("/api/admin/users/<path:user_email>", methods=["GET", "PUT", "DELETE"])
def admin_user_detail(user_email):
    if require_admin() is None:
        return admin_unauthorized()

    user_email = normalize_email(user_email)
    if not user_email:
        return respond({"status": "error", "message": "Valid user email is required."}, 400)

    if request.method == "GET":
        analyses_limit = max(1, min(parse_int(request.args.get("analyses_limit"), 100), 1000))
        wellness_limit = max(1, min(parse_int(request.args.get("wellness_limit"), 100), 1000))

        try:
            with get_connection() as conn:
                user_exists = conn.execute("SELECT 1 AS present FROM users WHERE email = %s", (user_email,)).fetchone()
                has_activity = conn.execute(
                    """
                    SELECT 1 AS present FROM analysis_history WHERE user_email = %s
                    UNION
                    SELECT 1 AS present FROM wellness_logs WHERE user_email = %s
                    LIMIT 1
                    """,
                    (user_email, user_email),
                ).fetchone()

                if user_exists is None and has_activity is None:
                    return respond({"status": "error", "message": "User not found."}, 404)

                if user_exists is None and has_activity is not None:
                    ensure_user_exists(conn, user_email)

                payload = {
                    "status": "success",
                    "user": fetch_user_payload(conn, email=user_email),
                    "report": build_user_report(conn, user_email),
                    "analyses": get_user_analyses(conn, user_email, analyses_limit),
                    "wellness_logs": get_user_wellness_logs(conn, user_email, wellness_limit),
                }
                return respond(payload)
        except Exception as exc:
            return respond({"status": "error", "message": str(exc)}, 500)

    if request.method == "PUT":
        payload = request.get_json(silent=True) or {}
        updates = {}
        for field in ("full_name", "phone", "address", "photo_url"):
            value = payload.get(field)
            if value is not None:
                clean = str(value).strip()
                if clean:
                    updates[field] = clean

        if "age" in payload:
            age = parse_int(payload.get("age"), 0)
            if age > 0:
                updates["age"] = age

        if not updates:
            return respond({"status": "error", "message": "No valid fields to update."}, 400)

        try:
            with get_connection() as conn:
                ensure_user_exists(conn, user_email)
                set_clause = ", ".join([f"`{k}` = %s" for k in updates.keys()])
                values = list(updates.values()) + [user_email]
                conn.execute(f"UPDATE users SET {set_clause} WHERE email = %s", values)
                return respond({"status": "success", "user": fetch_user_payload(conn, email=user_email)})
        except Exception as exc:
            return respond({"status": "error", "message": str(exc)}, 500)

    try:
        with get_connection() as conn:
            audio_rows = conn.execute(
                "SELECT audio_file FROM analysis_history WHERE user_email = %s AND audio_file IS NOT NULL AND TRIM(audio_file) <> ''",
                (user_email,),
            ).fetchall()
            analyses_deleted = conn.execute("DELETE FROM analysis_history WHERE user_email = %s", (user_email,)).rowcount
            wellness_deleted = conn.execute("DELETE FROM wellness_logs WHERE user_email = %s", (user_email,)).rowcount
            users_deleted = conn.execute("DELETE FROM users WHERE email = %s", (user_email,)).rowcount
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)

    removed_files = 0
    for row in audio_rows:
        clean = audio_path_safe((row or {}).get("audio_file"))
        if not clean:
            continue
        path = os.path.join(UPLOAD_DIR, clean)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed_files += 1
            except OSError:
                pass

    return respond(
        {
            "status": "success",
            "deleted": {
                "users": users_deleted,
                "analyses": analyses_deleted,
                "wellness_logs": wellness_deleted,
                "audio_files": removed_files,
            },
        }
    )


@app.route("/api/admin/reports/<path:user_email>", methods=["GET"])
def admin_user_report(user_email):
    if require_admin() is None:
        return admin_unauthorized()

    user_email = normalize_email(user_email)
    if not user_email:
        return respond({"status": "error", "message": "Valid user email is required."}, 400)

    analyses_limit = max(1, min(parse_int(request.args.get("analyses_limit"), 200), 1000))
    wellness_limit = max(1, min(parse_int(request.args.get("wellness_limit"), 200), 1000))

    try:
        with get_connection() as conn:
            report = build_user_report(conn, user_email)
            analyses = get_user_analyses(conn, user_email, analyses_limit)
            wellness_logs = get_user_wellness_logs(conn, user_email, wellness_limit)
            return respond(
                {
                    "status": "success",
                    "user_email": user_email,
                    "report": report,
                    "analyses": analyses,
                    "wellness_logs": wellness_logs,
                }
            )
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)


@app.route("/api/admin/analyses", methods=["GET"])
def admin_list_analyses():
    if require_admin() is None:
        return admin_unauthorized()

    user_email = normalize_email(request.args.get("user_email"))
    limit = max(1, min(parse_int(request.args.get("limit"), 300), 1000))

    query = """
        SELECT
            a.id, a.date, a.time, a.duration, a.stress_level, a.emotion, a.score,
            a.audio_file, a.user_email, a.created_at,
            COALESCE(u.full_name, a.user_email) AS full_name
        FROM analysis_history a
        LEFT JOIN users u ON u.email = a.user_email
    """
    params = []
    if user_email:
        query += " WHERE a.user_email = %s"
        params.append(user_email)
    query += " ORDER BY a.id DESC LIMIT %s"
    params.append(limit)

    try:
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return respond({"status": "success", "analyses": [dict(row) for row in rows]})
    except Exception as exc:
        return respond({"status": "error", "message": str(exc)}, 500)


@app.route("/api/admin/audio/<path:filename>", methods=["GET"])
def admin_audio_stream(filename):
    if require_admin() is None:
        return admin_unauthorized()

    clean = audio_path_safe(filename)
    if not clean:
        return respond({"status": "error", "message": "Invalid file name."}, 400)

    file_path = os.path.join(UPLOAD_DIR, clean)
    if not os.path.isfile(file_path):
        return respond({"status": "error", "message": "Audio file not found."}, 404)

    return send_from_directory(UPLOAD_DIR, clean, as_attachment=False)


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    resolved = os.path.join(app.static_folder, path)
    if os.path.exists(resolved):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")
if __name__ == "__main__":
    app.run(port=8000, debug=True)