# Database and Gemini Setup

## 1) MySQL configuration

This project now uses MySQL (not SQLite) from `.env`.
Legacy `database.db` file (if present) is no longer used by backend routes.

Current `.env` values:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=chin1987
DB_NAME=voice_stress_db
```

Use the same values in MySQL Workbench if you want to inspect tables directly.

## 2) Install dependencies

```bash
pip install -r requirement.txt
```

Required DB/AI packages are included in `requirement.txt`:
- `pymysql`
- `python-dotenv`
- `google-genai`
- `google-generativeai` (legacy fallback support)

Even if Gemini SDK packages are missing, backend now has a direct REST fallback (`ai_provider = gemini-rest`) that uses only your `.env` API key.

## 3) Initialize database

```bash
python backend/init_db.py
```

This creates database `voice_stress_db` and tables:
- `analysis_history`
- `users`
- `wellness_logs`
- `verification_codes`

Table columns:
- `analysis_history`: `id`, `date`, `time`, `duration`, `stress_level`, `emotion`, `score`, `audio_file`, `user_email`, `created_at`
- `users`: `id`, `full_name`, `email`, `age`, `password`, `phone`, `address`, `photo_url`, `created_at`
- `wellness_logs`: `id`, `user_email`, `water_intake`, `stress_score`, `emotion`, `recorded_at`
- `verification_codes`: `id`, `email`, `code`, `expires_at`

## 4) Run backend

```bash
python backend/app.py
```

Health check:

```text
GET /api/health
```

Important fields in response:
- `db_ready`: MySQL connection + schema status
- `db_name`: active database name
- `ai_enabled`: Gemini availability
- `ai_provider`: `google-genai`, `google-generativeai`, `gemini-rest`, or `none`
- `gemini_key_loaded`: whether `GEMINI_API_KEY` was loaded from `.env`

## 5) Gemini API key

Gemini key is read from:

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

Backend supports both SDKs:
- modern: `google-genai`
- fallback: `google-generativeai`
- no-sdk fallback: `gemini-rest` (direct HTTP API call)

If AI is still disabled:
1. Ensure `.env` is in project root (`e:\voice-stress-ui\.env`)
2. Reinstall dependencies from `requirement.txt`
3. Restart backend process
4. Recheck `GET /api/health`

## 6) Admin panel

Frontend page:

```text
/admin.html
```

Login credentials are from `.env`:

```env
ADMIN_EMAIL=admin@stress-tone.local
ADMIN_PASSWORD=admin123
```

Admin features:
- user list and search
- view/edit/delete user data
- view user reports and history
- stream uploaded audio recordings
