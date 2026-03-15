import os
import sys

# Add the current directory to the Python path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymysql
from werkzeug.security import generate_password_hash
from init_db import db_config

def create_user():
    email = "testuser@vocalvibe.pro"
    password = "password123"
    name = "Test Patient"
    
    hashed_pw = generate_password_hash(password)
    
    db = pymysql.connect(**db_config(True))
    try:
        with db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            if cur.fetchone():
                print(f"User {email} already exists!")
                return
            
            cur.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                (name, email, hashed_pw)
            )
        db.commit()
        print(f"✅ Successfully created user: {email} with password: {password}")
    except Exception as e:
        print(f"❌ Error inserting user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_user()