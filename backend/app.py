import os
import traceback
import mimetypes
import time
from datetime import datetime
from pathlib import Path
import urllib.parse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pymysql
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client
except ImportError:
    TwilioRestException = Exception
    Client = None

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)

from security import staff_required, generate_token, verify_token
from advanced_stress_predictor import AdvancedStressPredictor

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
_active_model_metrics = {}
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
REMOTE_AI_ENABLED = os.getenv("REMOTE_AI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
REMOTE_AI_PROVIDER = os.getenv("REMOTE_AI_PROVIDER", "huggingface").strip().lower()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "superb/wav2vec2-base-superb-er").strip()
REMOTE_AI_TIMEOUT_SEC = float(os.getenv("REMOTE_AI_TIMEOUT_SEC", "15"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
CHAT_FALLBACK_ENABLED = os.getenv("CHAT_FALLBACK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "gemini").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# --- ADMIN/STAFF CONFIG ---
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@stress-tone.local")

# --- TWILIO CONFIG (FOR SOS) ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# --- SMTP CONFIG (FOR STAFF ALERTS) ---
SMTP_SERVER = os.getenv("SMTP_SERVER")
try:
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
except ValueError:
    SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL")

def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _load_trainer():
    global _predictor, _active_model_path, _active_model_metrics
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
            if hasattr(_predictor, "get_bundle_metrics"):
                _active_model_metrics = _predictor.get_bundle_metrics()
            else:
                _active_model_metrics = {}
            app.logger.info("Loaded stress model from %s", _active_model_path)
            return True
        except Exception:
            continue
    return False


def _get_model_accuracy_percent():
    """Return model accuracy percentage from bundle metrics if available."""
    metrics = _active_model_metrics if isinstance(_active_model_metrics, dict) else {}
    candidates = (
        metrics.get("test_accuracy"),
        metrics.get("accuracy"),
    )
    for value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 1.0:
            return round(numeric * 100.0, 1), "model_test_accuracy"
        if numeric <= 100.0:
            return round(numeric, 1), "model_test_accuracy"
    return None, "avg_analysis_score_fallback"


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
        
    if file_obj.mimetype:
        if "webm" in file_obj.mimetype: return ".webm"
        if "ogg" in file_obj.mimetype: return ".ogg"
        if "mp4" in file_obj.mimetype or "m4a" in file_obj.mimetype: return ".m4a"
        if "mpeg" in file_obj.mimetype or "mp3" in file_obj.mimetype: return ".mp3"
        if "wav" in file_obj.mimetype: return ".wav"
        
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
    """Provides clinical-style fallback responses when AI is offline."""
    # Add spaces to ensure we match whole words (avoids matching 'hi' in 'this')
    text = f" {(message or '').lower()} "
    
    # Helper to check for whole words
    def has_word(keywords):
        return any(f" {k} " in text for k in keywords)
    
    # Emergency
    if has_word(["suicide", "kill", "die", "hurt", "harm", "emergency", "sos"]):
        return "⚠️ CRITICAL: If you are in immediate danger, please call emergency services (911/112) or use the red SOS button immediately. You are not alone."

    # Greeting & General
    if has_word(["hello", "hi", "hey", "greetings"]):
        return "Hello. I am your Clinical Wellness Assistant. I can help you interpret your stress levels, suggest coping mechanisms, or guide you through breathing exercises."
    
    if has_word(["who", "made", "created", "developer", "about"]):
        return "I am the VocalVibe Clinical Assistant, developed to support mental wellness tracking via vocal biomarkers."

    if has_word(["yes", "sure", "ok", "okay", "yeah"]):
        return "Great. You can proceed by clicking the microphone icon to start a new analysis, or ask me for a specific relaxation tip."

    # Specific Clinical Topics
    if has_word(["stress", "anxious", "anxiety", "tension", "worried", "panic"]):
        if has_word(["how", "check", "measure", "test"]):
             return "To check your stress, go to the 'Voice Scan' tab, click the microphone, and speak for 10-15 seconds."
        return "High stress levels often affect vocal pitch and jitter. I recommend the 'Box Breathing' technique: Inhale 4s, Hold 4s, Exhale 4s, Hold 4s. Would you like to try a voice scan now?"
    if has_word(["sleep", "insomnia", "tired", "awake", "rest"]):
        return "Sleep hygiene is vital for mental health. Try to maintain a consistent sleep schedule and avoid screens 1 hour before bed. High vocal jitter often correlates with fatigue."
    if has_word(["breath", "relax", "calm", "meditation"]):
        return "Relaxation starts with the breath. Try the 4-7-8 technique: Inhale quietly for 4 seconds, hold for 7 seconds, and exhale forcefully for 8 seconds."
    if has_word(["sad", "depressed", "unhappy", "cry", "lonely"]):
        return "It is okay to feel down sometimes. If these feelings persist, consider speaking to a professional. Deep breathing can help regulate immediate emotional responses."
    if has_word(["history", "report", "scan", "result", "record"]):
        return "You can view your detailed analysis trends in the 'History' tab. This tracks your vocal biomarkers over time to help identify stress triggers."

    # Default fallback
    return "I am currently operating in offline mode. I can assist with stress management tips, breathing exercises, and explaining your voice analysis results. Please verify your internet connection for full AI capabilities."

def _chat_with_gemini(message):
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in {"your_key_here", "your_real_gemini_key"}:
        raise RuntimeError("GEMINI_API_KEY is missing")

    configured = [m.strip() for m in GEMINI_MODEL.split(",") if m.strip()]
    model_candidates = []
    for model in configured:
        if model not in model_candidates:
            model_candidates.append(model)
    if not model_candidates:
        model_candidates = ["gemini-1.5-flash"]

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
        resp = requests.post(endpoint, json=payload, timeout=10)
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
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
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

def _send_email_alert_to_staff(user_name, user_email, location):
    """Sends an email alert to the clinical staff."""
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER_EMAIL, ADMIN_EMAIL]):
        print("⚠️ [SOS WARNING] SMTP settings not configured. Skipping email to clinical staff. Please set SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER_EMAIL in .env")
        return False

    subject = f"CRITICAL SOS ALERT: Patient {user_name}"
    body = f"""
    <p><b>A critical SOS alert has been triggered by a user on the VocalVibe platform.</b></p>
    <p>Please take immediate action.</p>
    <ul style="list-style-type: none; padding: 0;">
        <li style="padding-bottom: 5px;"><b>User Name:</b> {user_name}</li>
        <li style="padding-bottom: 5px;"><b>User Email:</b> {user_email}</li>
        <li style="padding-bottom: 5px;"><b>Last Known Location:</b> {location or 'Not Provided'}</li>
        <li style="padding-bottom: 5px;"><b>Time of Alert:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
    </ul>
    <p>This is an automated notification. Please check the system dashboard for more details and attempt to contact the user or their emergency contacts.</p>
    """

    msg = MIMEMultipart()
    msg['From'] = f"VocalVibe Alert System <{SMTP_SENDER_EMAIL}>"
    msg['To'] = ADMIN_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        # Using 'with' ensures the connection is closed
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
            print(f"✅ [SOS SUCCESS] Email alert sent to clinical staff at {ADMIN_EMAIL}")
            return True
    except Exception as e:
        print(f"❌ [SOS FAILED] Could not send email to clinical staff. Error: {e}")
        return False

def _send_sos_notification(contact, user_name, location):
    """
    Sends a real SOS SMS/Alert using Twilio.
    Falls back to console logging if Twilio is not configured.
    """
    message_body = (
        f"URGENT! {user_name} has triggered an SOS alert from the VocalVibe app. "
        f"Last known location: {location or 'Not available'}. Please contact them immediately."
    )
    
    # Log to console regardless
    print(f"🚨 [SOS ALERT] Preparing to notify {contact['name']} ({contact['phone']}): {message_body}")

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("⚠️ [SOS WARNING] Twilio credentials not set. Skipping real SMS. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER in your .env file.")
        return True # Simulate success for logging
        
    if Client is None:
        print("⚠️ [SOS WARNING] Twilio module not installed. Run 'pip install twilio'. Skipping SMS...")
        return True

    try:
        # Ensure the contact phone number is in E.164 format (e.g., +919876543210)
        to_phone = contact['phone']
        if not to_phone.startswith('+'):
            print(f"⚠️ [SOS WARNING] Phone number for {contact['name']} ({to_phone}) may not be in E.164 format. SMS might fail.")

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        print(f"✅ [SOS SUCCESS] SMS sent to {contact['phone']} (SID: {message.sid})")
        return True
    except TwilioRestException as e:
        print(f"❌ [SOS FAILED] Could not send SMS to {contact['phone']}. Error: {e}")
        return False

# --- DATABASE & SECURITY CONFIG ---
from init_db import db_config, create_db as initialize_database_schema

DB_CFG = db_config()
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin123")
ALERT_ACTIONS = {
    "acknowledge": {"status": "ACKNOWLEDGED", "timestamp_column": "acknowledged_at", "label": "Acknowledged"},
    "call_done": {"status": "CALLED", "timestamp_column": "call_done_at", "label": "Call Marked Done"},
    "follow_up": {"status": "FOLLOW_UP", "timestamp_column": "follow_up_at", "label": "Follow-up Scheduled"},
    "resolved": {"status": "RESOLVED", "timestamp_column": "resolved_at", "label": "Resolved"},
}

# --- DATABASE HELPERS ---
def get_db():
    """Returns a new database connection."""
    return pymysql.connect(**DB_CFG)


def _get_staff_email_from_request():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    payload = verify_token(token)
    if payload and payload.get("email"):
        return payload["email"]
    return ADMIN_EMAIL


def _sync_high_risk_alerts(cur):
    """
    Ensures every HIGH stress analysis has a corresponding triage alert record.
    Uses INSERT IGNORE with a unique key on analysis_id to keep this idempotent.
    """
    cur.execute(
        """
        INSERT IGNORE INTO clinical_alerts (analysis_id, user_email, stress_level, score, status)
        SELECT
            a.id,
            NULLIF(TRIM(a.user_email), ''),
            COALESCE(a.stress_level, 'High'),
            a.score,
            'NEW'
        FROM analysis_history a
        WHERE LOWER(COALESCE(a.stress_level, '')) = 'high'
        """
    )

try:
    # Ensure the database and tables exist on startup
    initialize_database_schema()
    print("[OK] Clinical Database Synchronized & Secured")
except Exception as e:
    print(f"[ERROR] Database Error: {e}")

# --- API ROUTES ---

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if email == ADMIN_EMAIL.lower() and data.get("password") == ADMIN_PASS:
        # LOG THE SUCCESSFUL ACCESS
        db = get_db()
        try:
            with db.cursor() as cur:
                cur.execute("INSERT INTO access_logs (staff_email, action, ip_address) VALUES (%s, %s, %s)", 
                           (ADMIN_EMAIL, "Authorized Gateway Access", request.remote_addr))
            db.commit()
        finally:
            db.close()
        token = generate_token(ADMIN_EMAIL)
        return jsonify({"status": "success", "token": token})
    return jsonify({"status": "error", "message": "Biometric Credentials Mismatch"}), 401

@app.route("/api/admin/overview", methods=["GET"])
@staff_required
def admin_overview():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(id) as total_users FROM users")
        u_count = cur.fetchone()
        cur.execute("SELECT COUNT(*) as total_analyses, AVG(score) as avg_score FROM analysis_history")
        a_count = cur.fetchone()
        cur.execute("SELECT stress_level, COUNT(*) as count FROM analysis_history GROUP BY stress_level")
        dist_rows = cur.fetchall()

        # Backward-compatible fallback:
        # Some existing datasets contain analysis rows even when the users table is empty.
        if (u_count.get("total_users") or 0) == 0 and (a_count.get("total_analyses") or 0) > 0:
            cur.execute(
                """
                SELECT COUNT(DISTINCT LOWER(TRIM(user_email))) as total_users
                FROM analysis_history
                WHERE user_email IS NOT NULL AND TRIM(user_email) <> ''
                """
            )
            u_count = cur.fetchone()
    db.close()

    total_analyses = a_count['total_analyses'] or 0
    dist = {"low": 0, "moderate": 0, "high": 0}
    for r in dist_rows:
        if r.get('stress_level'):
            lvl = r['stress_level'].lower()
            if lvl in dist:
                dist[lvl] = r['count']

    percentages = {
        "low": round((dist["low"] / total_analyses) * 100) if total_analyses > 0 else 0,
        "moderate": round((dist["moderate"] / total_analyses) * 100) if total_analyses > 0 else 0,
        "high": round((dist["high"] / total_analyses) * 100) if total_analyses > 0 else 0,
    }

    model_accuracy = None
    accuracy_source = "avg_analysis_score_fallback"
    if _load_trainer():
        model_accuracy, accuracy_source = _get_model_accuracy_percent()
    if model_accuracy is None:
        model_accuracy = round(a_count['avg_score'] or 0, 1)
    avg_analysis_score = round(a_count['avg_score'] or 0, 1)

    return jsonify({
        "status": "success",
        "counts": {
            "total_users": u_count['total_users'],
            "total_analyses": total_analyses,
            "average_score": model_accuracy,
            "average_score_source": accuracy_source,
            "avg_analysis_score": avg_analysis_score,
            "high_stress_count": dist["high"]
        },
        "distribution": dist,
        "percentages": percentages
    })


@app.route("/api/admin/high-risk-queue", methods=["GET"])
@staff_required
def admin_high_risk_queue():
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    db = get_db()
    try:
        with db.cursor() as cur:
            _sync_high_risk_alerts(cur)

            cur.execute("SELECT COUNT(*) as total_count FROM clinical_alerts")
            total_count = (cur.fetchone() or {}).get("total_count") or 0

            cur.execute("SELECT COUNT(*) as open_count FROM clinical_alerts WHERE status <> 'RESOLVED'")
            open_count = (cur.fetchone() or {}).get("open_count") or 0

            cur.execute(
                """
                SELECT
                    ca.analysis_id,
                    ca.user_email,
                    ca.stress_level,
                    ca.score,
                    ca.status,
                    ca.notes,
                    ca.assigned_to,
                    ca.last_action_by,
                    ca.acknowledged_at,
                    ca.call_done_at,
                    ca.follow_up_at,
                    ca.resolved_at,
                    ca.updated_at,
                    a.date,
                    a.time,
                    a.audio_file,
                    a.created_at as analysis_created_at
                FROM clinical_alerts ca
                LEFT JOIN analysis_history a ON a.id = ca.analysis_id
                ORDER BY
                    CASE WHEN ca.status = 'RESOLVED' THEN 1 ELSE 0 END,
                    CASE ca.status
                        WHEN 'NEW' THEN 0
                        WHEN 'ACKNOWLEDGED' THEN 1
                        WHEN 'CALLED' THEN 2
                        WHEN 'FOLLOW_UP' THEN 3
                        ELSE 4
                    END,
                    COALESCE(ca.updated_at, a.created_at) DESC,
                    ca.analysis_id DESC
                LIMIT %s
                """,
                (limit,),
            )
            queue = cur.fetchall()

        db.commit()
    finally:
        db.close()

    base_url = request.host_url.rstrip('/')
    for row in queue:
        if row.get("audio_file"):
            filename = os.path.basename(row["audio_file"])
            row["audio_file"] = f"{base_url}/uploads/{urllib.parse.quote(filename)}"

    return jsonify(
        {
            "status": "success",
            "summary": {
                "total_count": total_count,
                "open_count": open_count,
                "resolved_count": max(total_count - open_count, 0),
            },
            "queue": queue,
        }
    )


@app.route("/api/admin/high-risk-queue/<int:analysis_id>/action", methods=["POST"])
@staff_required
def admin_high_risk_action(analysis_id):
    data = request.get_json(silent=True) or {}
    action = str(data.get("action") or "").strip().lower()
    notes = data.get("notes")
    notes = notes.strip() if isinstance(notes, str) else None

    config = ALERT_ACTIONS.get(action)
    if not config:
        return jsonify(
            {
                "status": "error",
                "message": "Invalid action. Use one of: acknowledge, call_done, follow_up, resolved",
            }
        ), 400

    staff_email = _get_staff_email_from_request()
    ts_column = config["timestamp_column"]
    status_value = config["status"]

    db = get_db()
    try:
        with db.cursor() as cur:
            _sync_high_risk_alerts(cur)

            cur.execute("SELECT analysis_id FROM clinical_alerts WHERE analysis_id=%s", (analysis_id,))
            existing = cur.fetchone()
            if not existing:
                return jsonify({"status": "error", "message": "High-risk alert not found"}), 404

            if notes:
                cur.execute(
                    f"""
                    UPDATE clinical_alerts
                    SET status=%s, assigned_to=%s, last_action_by=%s, notes=%s, {ts_column}=NOW(), updated_at=NOW()
                    WHERE analysis_id=%s
                    """,
                    (status_value, staff_email, staff_email, notes, analysis_id),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE clinical_alerts
                    SET status=%s, assigned_to=%s, last_action_by=%s, {ts_column}=NOW(), updated_at=NOW()
                    WHERE analysis_id=%s
                    """,
                    (status_value, staff_email, staff_email, analysis_id),
                )

        db.commit()
    finally:
        db.close()

    return jsonify(
        {
            "status": "success",
            "message": f"Alert #{analysis_id} updated: {config['label']}",
            "analysis_id": analysis_id,
            "new_status": status_value,
            "updated_by": staff_email,
        }
    )

@app.route("/api/admin/users", methods=["GET"])
@staff_required
def admin_list_users():
    search_query = request.args.get("search", "").strip()
    db = get_db()
    with db.cursor() as cur:
        if search_query:
            sql = """
                SELECT u.*, COUNT(a.id) as analyses_count
                FROM users u
                LEFT JOIN analysis_history a ON u.email = a.user_email
                WHERE u.full_name LIKE %s OR u.email LIKE %s
                GROUP BY u.email
            """
            like_query = f"%{search_query}%"
            cur.execute(sql, (like_query, like_query))
        else:
            cur.execute("""
                SELECT u.*, COUNT(a.id) as analyses_count
                FROM users u
                LEFT JOIN analysis_history a ON u.email = a.user_email
                GROUP BY u.email
            """)
        users = cur.fetchall()
    db.close()
    return jsonify({"status": "success", "users": users})

@app.route("/api/admin/user/<int:user_id>", methods=["GET", "PUT", "DELETE"])
@staff_required
def admin_manage_user(user_id):
    db = get_db()
    try:
        with db.cursor() as cur:
            if request.method == "GET":
                cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
                user = cur.fetchone()
                if not user:
                    return jsonify({"status": "error", "message": "User not found"}), 404
                
                cur.execute("SELECT * FROM analysis_history WHERE user_email=%s ORDER BY id DESC", (user['email'],))
                history = cur.fetchall()
                
                base_url = request.host_url.rstrip('/')
                for row in history:
                    if row.get("audio_file"):
                        filename = os.path.basename(row['audio_file'])
                        row["audio_file"] = f"{base_url}/uploads/{urllib.parse.quote(filename)}"

                return jsonify({"status": "success", "user": user, "history": history})

            if request.method == "PUT":
                data = request.get_json(silent=True) or {}
                full_name = data.get("full_name")
                email = data.get("email")
                age = data.get("age")
                phone = data.get("phone")
                address = data.get("address")

                if not full_name or not email:
                    return jsonify({"status": "error", "message": "Full name and email are required"}), 400

                cur.execute(
                    "UPDATE users SET full_name=%s, email=%s, age=%s, phone=%s, address=%s WHERE id=%s",
                    (full_name, email, age, phone, address, user_id)
                )
                if cur.rowcount == 0:
                    return jsonify({"status": "error", "message": "User not found or no changes made"}), 404
                db.commit()
                return jsonify({"status": "success", "message": "User updated successfully"})

            if request.method == "DELETE":
                cur.execute("SELECT email FROM users WHERE id=%s", (user_id,))
                user = cur.fetchone()
                if not user:
                    return jsonify({"status": "error", "message": "User not found"}), 404
                
                user_email = user['email']
                
                cur.execute("DELETE FROM analysis_history WHERE user_email=%s", (user_email,))
                cur.execute("DELETE FROM emergency_contacts WHERE user_email=%s", (user_email,))
                cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
                db.commit()
                return jsonify({"status": "success", "message": f"User '{user_email}' and all associated data have been deleted."})
    finally:
        db.close()

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
        print(f"[-] AI Chat Error: {str(e)}") # Log error to terminal for debugging
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

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "voice-stress-ui",
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

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
    timestamp = datetime.now()

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO analysis_history (user_email, date, time, stress_level, emotion, score, audio_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_email, timestamp.strftime("%b %d, %Y"), timestamp.strftime("%I:%M %p"), stress, emo, score, filename))
    db.commit()
    db.close()

    return jsonify({
        "status": "success",
        "stress_level": stress,
        "score": round(score, 1),
        "emotion": emo,
        "duration": round(float(prediction.get("duration", 0.0)), 2),
        "model": Path(_active_model_path).name if _active_model_path else "unknown",
        "analysis_source": analysis_source,
        "confidence": round(float(prediction.get("confidence", 0.0)), 4),
        "probabilities": prediction.get("probabilities", {}),
        "remote_error": remote_error,
    })

@app.route("/api/dashboard-summary", methods=["GET"])
def get_summary():
    email = request.args.get('user_email')
    if not email:
        return jsonify({
            "total": 0,
            "avg_score": 0,
            "distribution": {"low": 0, "moderate": 0, "high": 0},
            "percentages": {"low": 0, "moderate": 0, "high": 0}
        })

    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) as total, AVG(score) as avg FROM analysis_history WHERE user_email=%s", (email,))
        stats = cur.fetchone()
        cur.execute("SELECT stress_level, COUNT(*) as count FROM analysis_history WHERE user_email=%s GROUP BY stress_level", (email,))
        dist_rows = cur.fetchall()
    db.close()

    total_analyses = stats.get('total') or 0
    dist = {"low": 0, "moderate": 0, "high": 0}
    for r in dist_rows:
        if r.get('stress_level'):
            lvl = r['stress_level'].lower()
            if lvl in dist:
                dist[lvl] = r.get('count') or 0

    percentages = {
        "low": round((dist["low"] / total_analyses) * 100) if total_analyses > 0 else 0,
        "moderate": round((dist["moderate"] / total_analyses) * 100) if total_analyses > 0 else 0,
        "high": round((dist["high"] / total_analyses) * 100) if total_analyses > 0 else 0,
    }

    return jsonify({
        "total": total_analyses,
        "avg_score": round(stats.get('avg') or 0, 1),
        "distribution": dist,
        "percentages": percentages
    })

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
                return jsonify({"status": "error", "message": "Secure Clinical Access Required"}), 401
            cur.execute("SELECT * FROM analysis_history ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
    db.close()

    # Fix audio paths to be absolute URLs so they work even if frontend is on a different port
    base_url = request.host_url.rstrip('/')
    for row in rows:
        if row.get("audio_file"):
            filename = os.path.basename(row['audio_file'])
            row["audio_file"] = f"{base_url}/uploads/{urllib.parse.quote(filename)}"
        
        # Add clinical recommendation for the 'Action' column
        s_level = (row.get("stress_level") or "").upper()
        if s_level == "HIGH":
            row["recommendation"] = "Contact Doctor"
            row["action_class"] = "high-action"
        elif s_level == "MODERATE":
            row["recommendation"] = "Breathing Exercise"
            row["action_class"] = "mod-action"
        else:
            row["recommendation"] = "View Report"
            row["action_class"] = "low-action"

    return jsonify(rows)

@app.route("/api/user/contacts", methods=["GET", "POST", "DELETE"])
def manage_contacts():
    db = get_db()
    if request.method == "GET":
        email = request.args.get("user_email")
        if not email:
            return jsonify({"status": "error", "message": "User email required"}), 400
        
        with db.cursor() as cur:
            cur.execute("SELECT * FROM emergency_contacts WHERE user_email=%s", (email,))
            contacts = cur.fetchall()
        db.close()
        return jsonify({"status": "success", "contacts": contacts})

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        email = data.get("user_email")
        name = data.get("name")
        phone = data.get("phone")
        relation = data.get("relationship", "Other")
        
        if not email or not name or not phone:
            return jsonify({"status": "error", "message": "Missing fields"}), 400
        
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM emergency_contacts WHERE user_email=%s", (email,))
            if cur.fetchone()['count'] >= 5:
                db.close()
                return jsonify({"status": "error", "message": "Max 5 contacts allowed"}), 400
            
            cur.execute(
                "INSERT INTO emergency_contacts (user_email, name, phone, relationship) VALUES (%s, %s, %s, %s)",
                (email, name, phone, relation)
            )
        db.commit()
        db.close()
        return jsonify({"status": "success", "message": "Contact added"})

    if request.method == "DELETE":
        data = request.get_json(silent=True) or {}
        contact_id = data.get("id")
        email = data.get("user_email") # Security check to ensure ownership
        
        with db.cursor() as cur:
            cur.execute("DELETE FROM emergency_contacts WHERE id=%s AND user_email=%s", (contact_id, email))
        db.commit()
        db.close()
        return jsonify({"status": "success", "message": "Contact deleted"})

@app.route("/api/user/sos", methods=["POST"])
def trigger_sos():
    data = request.get_json(silent=True) or {}
    email = data.get("user_email")
    location = data.get("location", "Unknown Location")
    
    if not email:
        return jsonify({"status": "error", "message": "User email required"}), 400
        
    db = get_db()
    with db.cursor() as cur:
        # Get User Name for the message
        cur.execute("SELECT full_name FROM users WHERE email=%s", (email,))
        user_row = cur.fetchone()
        user_name = user_row['full_name'] if user_row else "User"
        
        # Get Contacts
        cur.execute("SELECT * FROM emergency_contacts WHERE user_email=%s", (email,))
        contacts = cur.fetchall()
    db.close()
    
    if not contacts:
        return jsonify({"status": "error", "message": "No emergency contacts found. Please add contacts in Settings."}), 404
        
    # Notify clinical staff via email
    staff_notified = _send_email_alert_to_staff(user_name, email, location)
        
    sent_count = 0
    for contact in contacts:
        if _send_sos_notification(contact, user_name, location):
            sent_count += 1
            
    message = f"SOS Alert sent to {sent_count} contacts."
    if staff_notified:
        message = f"SOS Alert sent to {sent_count} contacts and the clinical team."
            
    return jsonify({
        "status": "success", 
        "message": message,
        "contacts_notified": sent_count
    })

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    safe_name = os.path.basename(filename)
    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"status": "error", "message": "Audio file not found"}), 404

    mimetype, _ = mimetypes.guess_type(safe_name)
    if safe_name.lower().endswith('.webm'):
        mimetype = 'audio/webm'
    elif safe_name.lower().endswith('.m4a'):
        mimetype = 'audio/mp4'

    return send_from_directory(str(UPLOAD_DIR), safe_name, mimetype=mimetype, as_attachment=False)

# --- STATIC FILE SERVING ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    if path.startswith("api/"):
        return jsonify({"error": "Resource Not Found"}), 404
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    host = os.getenv("HOST", os.getenv("FLASK_HOST", "0.0.0.0")).strip() or "0.0.0.0"
    port = _env_int("PORT", _env_int("FLASK_PORT", 8000))
    debug = _env_flag("FLASK_DEBUG", True)
    use_reloader = _env_flag("FLASK_USE_RELOADER", False)

    print(
        f"[STARTUP] VocalVibe backend on http://127.0.0.1:{port} "
        f"(bind={host}, debug={debug}, reloader={use_reloader})"
    )
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
    
