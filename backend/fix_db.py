import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymysql
from init_db import db_config

def fix_database():
    db = pymysql.connect(**db_config(True))
    try:
        with db.cursor() as cur:
            # 0. Add the missing password column if it does not already exist
            try:
                cur.execute("ALTER TABLE users ADD COLUMN password VARCHAR(255) NULL AFTER email")
                print("✅ Added 'password' column to users table.")
            except Exception as e:
                print(f"⚠️ Column 'password' add skip (might already exist): {e}")

            # 1. Jo passwords 'scrypt' se start hote hain, unhe plain text 'password' se replace kar do
            cur.execute("UPDATE users SET password = 'password' WHERE password LIKE 'scrypt:%'")
            print(f"✅ Reset {cur.rowcount} hashed passwords to plain text 'password'.")
            
            # 2. Database table se age, phone, aur address columns delete kar do
            try:
                cur.execute("ALTER TABLE users DROP COLUMN age, DROP COLUMN phone, DROP COLUMN address")
                print("✅ Dropped age, phone, and address columns from users table.")
            except Exception as e:
                print(f"⚠️ Columns drop skip hue (shayad pehle hi delete ho chuke hain): {e}")
                
            # 3. Fix emergency_contacts table to support email and optional phone
            try:
                cur.execute("ALTER TABLE emergency_contacts ADD COLUMN email VARCHAR(255) NULL AFTER phone")
                print("✅ Added 'email' column to emergency_contacts table.")
            except Exception as e:
                print(f"⚠️ Column 'email' add skip (might already exist): {e}")

            try:
                cur.execute("ALTER TABLE emergency_contacts MODIFY COLUMN phone VARCHAR(64) NULL")
                print("✅ Modified 'phone' column in emergency_contacts to be nullable.")
            except Exception as e:
                print(f"⚠️ Column 'phone' modify skip: {e}")

        db.commit()
        print("🎉 Database successfully fixed!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    fix_database()