import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymysql
from init_db import db_config

def view_users():
    # Connect to the active MySQL database
    db = pymysql.connect(**db_config(True))
    try:
        with db.cursor() as cur:
            cur.execute("SELECT id, full_name, email, created_at FROM users")
            users = cur.fetchall()
            
            print("\n" + "="*60)
            print(" 👥 USERS CURRENTLY IN MYSQL DATABASE (`voice_stress_db`)")
            print("="*60)
            if not users:
                print(" The users table is currently empty.")
            else:
                for u in users:
                    print(f" ID: {u['id']} | Name: {u['full_name']} | Email: {u['email']}")
            print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    view_users()