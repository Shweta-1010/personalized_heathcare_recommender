import sqlite3

conn = sqlite3.connect("healthcare_system.db")
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    symptoms TEXT,
    medicines TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Demo users
users = [
    ("testuser", "1234"),
    ("admin", "admin123"),
    ("user1", "userpass"),
    ("doctor", "med123"),
    ("patient", "care456")
]
c.executemany("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", users)

# sample recs
sample_recs = [
    ("testuser", "fever and cough", "Paracetamol, Dolo-650"),
    ("user1", "stomach pain and acidity", "Pantoprazole, Digene"),
]
c.executemany("INSERT INTO recommendations (username, symptoms, medicines) VALUES (?, ?, ?)", sample_recs)

conn.commit()
conn.close()
print("Database created/updated: healthcare_system.db")
