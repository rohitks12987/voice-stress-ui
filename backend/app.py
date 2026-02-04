import os
import sqlite3
from flask import Flask, request, jsonify, send_from_directory
import smtplib
import ssl
import random
from datetime import datetime, timedelta
import os
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) # Frontend connection errors fix karne ke liye

DATABASE = 'database.db'

def init_db():
    """Database table initialize karein"""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                duration TEXT,
                stress_level TEXT,
                emotion TEXT,
                score REAL
            )
        ''')
        # Verification codes table (temporary storage for OTP)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        ''')
        conn.commit()
    print("Database Path:", os.path.abspath(DATABASE))
    print("System Ready: Database Initialized!")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file"}), 400

    # Yahan asli AI model processing hoti hai. 
    # Hum dummy analysis result generate kar rahe hain:
    now = datetime.now()
    analysis_result = {
        "date": now.strftime("%b %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "duration": "12.5s",
        "stress_level": "High Probability", # Example prediction
        "emotion": "Anxious",            # Example prediction
        "score": 85.0                    # Confidence score
    }
    analysis_result = {
        "date": now.strftime("%b %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "duration": "12.5s",
        "stress_level": "low Probability", # Example prediction
        "emotion": "Anxious",            # Example prediction
        "score": 39  
    }
    analysis_result = {
        "date": now.strftime("%b %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "duration": "12.5s",
        "stress_level": "moderate", # Example prediction
        "emotion": "Anxious",            # Example prediction
        "score": 73                   # Confidence score
    }
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_history (date, time, duration, stress_level, emotion, score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (analysis_result['date'], analysis_result['time'], 
                  analysis_result['duration'], analysis_result['stress_level'], 
                  analysis_result['emotion'], analysis_result['score']))
            conn.commit()
        return jsonify({"status": "success", **analysis_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/send_verification', methods=['POST'])
def send_verification():
    data = request.get_json() or {}
    email = data.get('email')
    if not email:
        return jsonify({'status': 'error', 'message': 'Email required'}), 400

    # generate 6-digit code
    code = '{:06d}'.format(random.randint(0, 999999))
    expires_at = (datetime.utcnow() + timedelta(minutes=10)).isoformat()

    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO verification_codes (email, code, expires_at) VALUES (?, ?, ?)',
                           (email, code, expires_at))
            conn.commit()

        # try to send email if SMTP config present
        smtp_host = os.environ.get('SMTP_HOST')
        smtp_port = int(os.environ.get('SMTP_PORT', 0) or 0)
        smtp_user = os.environ.get('SMTP_USER')
        smtp_pass = os.environ.get('SMTP_PASS')
        from_addr = os.environ.get('FROM_EMAIL', smtp_user)

        subject = 'Your Stress Tone AI verification code'
        body = f'Your verification code is: {code}\nIt expires in 10 minutes.'
        message = f"Subject: {subject}\n\n{body}"

        sent = False
        if smtp_host and smtp_port and smtp_user and smtp_pass:
            try:
                context = ssl.create_default_context()
                if smtp_port == 465:
                    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(from_addr, email, message)
                else:
                    with smtplib.SMTP(smtp_host, smtp_port) as server:
                        server.starttls(context=context)
                        server.login(smtp_user, smtp_pass)
                        server.sendmail(from_addr, email, message)
                sent = True
            except Exception as e:
                app.logger.warning('Failed to send verification email: %s', e)
                sent = False

        # Return success but do not expose the OTP in the response
        return jsonify({'status': 'success', 'sent': sent})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/verify_code', methods=['POST'])
def verify_code():
    data = request.get_json() or {}
    email = data.get('email')
    code = data.get('code')
    if not email or not code:
        return jsonify({'status': 'error', 'message': 'Email and code required'}), 400

    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT code, expires_at FROM verification_codes WHERE email = ? ORDER BY id DESC LIMIT 1', (email,))
            row = cursor.fetchone()
            if not row:
                return jsonify({'status': 'error', 'message': 'No code found'}), 400

            stored_code, expires_at = row[0], row[1]
            if stored_code != code:
                return jsonify({'status': 'error', 'message': 'Invalid code'}), 400

            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                return jsonify({'status': 'error', 'message': 'Code expired'}), 400

            return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_history ORDER BY id DESC')
            rows = cursor.fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    init_db()
    app.run(port=8000, debug=True)