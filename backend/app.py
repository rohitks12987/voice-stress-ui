import os
import sqlite3
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai  # Modern library

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) 

DATABASE = 'database.db'
API_KEY = "AIzaSyA2Qr_DaKDB4cFrHxjheKfZSOCJutgg4Eg"

# ---------------------------------------------------------
# 🤖 SELF-HEALING AI INTEGRATION
# ---------------------------------------------------------
# We use v1beta as it often provides better model discovery
try:
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1beta'})
    
    def find_working_model():
        """Automatically finds the correct model name for your API key"""
        try:
            for m in client.models.list():
                # We look for a model that supports generating content
                if 'generateContent' in m.supported_actions:
                    # Return the clean name (e.g., 'gemini-1.5-flash')
                    return m.name.split('/')[-1] 
            return "gemini-1.5-flash" # Fallback if list fails
        except Exception as e:
            print(f"⚠️ Discovery Warning: {e}")
            return "gemini-1.5-flash"

    ACTIVE_MODEL = find_working_model()
    print(f"✅ AI Connected: Using detected model '{ACTIVE_MODEL}'")

except Exception as e:
    print(f"❌ AI Init Failed: {e}")
    client = None
    ACTIVE_MODEL = None

# ---------------------------------------------------------
# 🗄️ DATABASE INITIALIZATION
# ---------------------------------------------------------
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS analysis_history 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, time TEXT, 
             duration TEXT, stress_level TEXT, emotion TEXT, score REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS verification_codes 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, 
             code TEXT NOT NULL, expires_at TEXT NOT NULL)''')
        conn.commit()
    print("✅ Database Initialized!")

# ---------------------------------------------------------
# 💬 CHATBOT API ROUTE
# ---------------------------------------------------------
@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json(silent=True) or {}
    user_msg = data.get('message')
    
    if not user_msg:
        return jsonify({"reply": "I'm here. How can I help?"})

    if not client or not ACTIVE_MODEL:
        return jsonify({"reply": "AI module is currently offline."}), 503

    try:
        # Uses the model detected during startup
        response = client.models.generate_content(
            model=ACTIVE_MODEL, 
            contents=f"You are a Wellness Officer. Be very brief. User: {user_msg}"
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"🔴 AI SERVER ERROR: {e}")
        # Return a polite message so the user isn't stuck
        return jsonify({"reply": "I'm having a brief connection issue. Try a deep breath while I reconnect!"}), 200

# ---------------------------------------------------------
# 🎤 AUDIO UPLOAD & HISTORY
# ---------------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file"}), 400
    
    now = datetime.now()
    res = {
        "date": now.strftime("%b %d, %Y"), 
        "time": now.strftime("%I:%M %p"), 
        "duration": "12.5s", 
        "stress_level": "moderate", 
        "emotion": "Anxious", 
        "score": 73
    }
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO analysis_history 
                (date, time, duration, stress_level, emotion, score) VALUES (?,?,?,?,?,?)''',
                (res['date'], res['time'], res['duration'], res['stress_level'], res['emotion'], res['score']))
            conn.commit()
        return jsonify({"status": "success", **res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_history ORDER BY id DESC')
            return jsonify([dict(row) for row in cursor.fetchall()])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------
# 🌐 STATIC FILE SERVING
# ---------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    init_db()
    app.run(port=8000, debug=True)