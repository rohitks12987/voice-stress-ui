import os

import pymysql
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=False)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)


def parse_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def db_config(include_database=True):
    cfg = {
        "host": os.getenv("DB_HOST", "127.0.0.1").strip(),
        "port": parse_int(os.getenv("DB_PORT", "3306"), 3306),
        "user": os.getenv("DB_USER", "root").strip(),
        "password": os.getenv("DB_PASSWORD", "root"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": False,
    }
    if include_database:
        cfg["database"] = os.getenv("DB_NAME", "voice_stress_db").strip()
    return cfg


def create_db():
    db_name = db_config(True)["database"]

    server_conn = pymysql.connect(**db_config(False))
    try:
        with server_conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        server_conn.commit()
    finally:
        server_conn.close()

    conn = pymysql.connect(**db_config(True))
    try:
        with conn.cursor() as cur:
            cur.execute(
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
                    PRIMARY KEY (id),
                    INDEX idx_analysis_history_email (user_email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            cur.execute(
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

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS wellness_logs (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    user_email VARCHAR(255) NOT NULL,
                    water_intake DECIMAL(6,2) NOT NULL DEFAULT 0,
                    stress_score INT NOT NULL DEFAULT 0,
                    emotion VARCHAR(64) NOT NULL DEFAULT 'Unknown',
                    recorded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id),
                    INDEX idx_wellness_email_time (user_email, recorded_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            cur.execute(
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

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS emergency_contacts (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    user_email VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    phone VARCHAR(64) NOT NULL,
                    relationship VARCHAR(64) NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id),
                    INDEX idx_emergency_user (user_email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

        conn.commit()
        print(f"MySQL database '{db_name}' and tables are ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    create_db()