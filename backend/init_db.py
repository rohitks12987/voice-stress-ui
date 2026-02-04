import sqlite3

def create_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Table structure updated to match your history page needs
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
    conn.commit()
    conn.close()
    print("Database and Table created successfully!")

# Is line ko init_db.py ke create_db() function mein add karein
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        email TEXT UNIQUE,
        age INTEGER,
        password TEXT
    )
''')

if __name__ == '__main__':
    create_db()