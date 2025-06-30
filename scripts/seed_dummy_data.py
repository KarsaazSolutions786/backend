import os
import uuid
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_batch

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "eindr_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "eindr_pass")

# Map of database -> list[ (SQL, params_list) ]
DATA: dict[str, list[tuple[str, list[tuple]]]] = {}

now = datetime.utcnow()

# ----------------------- auth_db -----------------------
user_rows = [
    (str(uuid.uuid4()), f"user{i}@example.com", f"hashed_pw_{i}", True, True, now)
    for i in range(1, 6)
]
DATA["auth_db"] = [
    (
        "INSERT INTO users (id, email, password_hash, is_active, is_verified, created_at) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        user_rows,
    )
]

# ----------------------- user_db -----------------------
profile_rows = [
    (row[0], f"User {i}", "en", "UTC") for i, row in enumerate(user_rows, start=1)
]
DATA["user_db"] = [
    (
        "INSERT INTO profiles (user_id, full_name, language, timezone) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        profile_rows,
    )
]

# ----------------------- reminder_db -------------------
reminder_rows = []
for idx, (uid, *_rest) in enumerate(user_rows, start=1):
    reminder_rows.append(
        (str(uuid.uuid4()), uid, f"Doctor appointment #{idx}", "Annual check-up", now + timedelta(days=idx))
    )
DATA["reminder_db"] = [
    (
        "INSERT INTO reminders (id, user_id, title, description, time) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        reminder_rows,
    )
]

# ----------------------- note_db -----------------------
note_rows = [
    (str(uuid.uuid4()), uid, f"Sample note for {email}", now)
    for uid, email, *_ in [(u[0], u[1]) for u in user_rows]
]
DATA["note_db"] = [
    (
        "INSERT INTO notes (id, user_id, content, created_at) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        note_rows,
    )
]

# ----------------------- ledger_db ---------------------
ledger_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Alice", 25.0, "owe"),
    (str(uuid.uuid4()), user_rows[1][0], "Bob", 12.5, "owed"),
]
DATA["ledger_db"] = [
    (
        "INSERT INTO ledger_entries (id, user_id, contact_name, amount, direction) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        ledger_rows,
    )
]

# ----------------------- friend_db ---------------------
friend_rows = [
    (str(uuid.uuid4()), user_rows[0][0], user_rows[1][0], "accepted"),
    (str(uuid.uuid4()), user_rows[2][0], user_rows[3][0], "pending"),
]
DATA["friend_db"] = [
    (
        "INSERT INTO friendships (id, user_id, friend_id, status) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        friend_rows,
    )
]

# ----------------------- history_db --------------------
history_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Created reminder", "reminder", now),
    (str(uuid.uuid4()), user_rows[1][0], "Added note", "note", now),
]
DATA["history_db"] = [
    (
        "INSERT INTO history_logs (id, user_id, content, interaction_type, created_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        history_rows,
    )
]

# ----------------------- chat_db -----------------------
chat_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Hello, Eindr!", now, "assistant", "Hi! How can I help?"),
]
DATA["chat_db"] = [
    (
        "INSERT INTO chat_history (id, user_id, query, created_at, responder, response) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        chat_rows,
    )
]

def seed_database(db_name: str, statements):
    dsn = f"dbname={db_name} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    for sql, rows in statements:
        execute_batch(cur, sql, rows)
    cur.close()
    conn.close()
    print(f"Seeded {db_name} ✔️")

if __name__ == "__main__":
    for db, stmts in DATA.items():
        try:
            seed_database(db, stmts)
        except Exception as e:
            print(f"⚠️  Skipped {db}: {e}") 
import uuid
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_batch

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "eindr_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "eindr_pass")

# Map of database -> list[ (SQL, params_list) ]
DATA: dict[str, list[tuple[str, list[tuple]]]] = {}

now = datetime.utcnow()

# ----------------------- auth_db -----------------------
user_rows = [
    (str(uuid.uuid4()), f"user{i}@example.com", f"hashed_pw_{i}", True, True, now)
    for i in range(1, 6)
]
DATA["auth_db"] = [
    (
        "INSERT INTO users (id, email, password_hash, is_active, is_verified, created_at) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        user_rows,
    )
]

# ----------------------- user_db -----------------------
profile_rows = [
    (row[0], f"User {i}", "en", "UTC") for i, row in enumerate(user_rows, start=1)
]
DATA["user_db"] = [
    (
        "INSERT INTO profiles (user_id, full_name, language, timezone) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        profile_rows,
    )
]

# ----------------------- reminder_db -------------------
reminder_rows = []
for idx, (uid, *_rest) in enumerate(user_rows, start=1):
    reminder_rows.append(
        (str(uuid.uuid4()), uid, f"Doctor appointment #{idx}", "Annual check-up", now + timedelta(days=idx))
    )
DATA["reminder_db"] = [
    (
        "INSERT INTO reminders (id, user_id, title, description, time) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        reminder_rows,
    )
]

# ----------------------- note_db -----------------------
note_rows = [
    (str(uuid.uuid4()), uid, f"Sample note for {email}", now)
    for uid, email, *_ in [(u[0], u[1]) for u in user_rows]
]
DATA["note_db"] = [
    (
        "INSERT INTO notes (id, user_id, content, created_at) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        note_rows,
    )
]

# ----------------------- ledger_db ---------------------
ledger_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Alice", 25.0, "owe"),
    (str(uuid.uuid4()), user_rows[1][0], "Bob", 12.5, "owed"),
]
DATA["ledger_db"] = [
    (
        "INSERT INTO ledger_entries (id, user_id, contact_name, amount, direction) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        ledger_rows,
    )
]

# ----------------------- friend_db ---------------------
friend_rows = [
    (str(uuid.uuid4()), user_rows[0][0], user_rows[1][0], "accepted"),
    (str(uuid.uuid4()), user_rows[2][0], user_rows[3][0], "pending"),
]
DATA["friend_db"] = [
    (
        "INSERT INTO friendships (id, user_id, friend_id, status) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        friend_rows,
    )
]

# ----------------------- history_db --------------------
history_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Created reminder", "reminder", now),
    (str(uuid.uuid4()), user_rows[1][0], "Added note", "note", now),
]
DATA["history_db"] = [
    (
        "INSERT INTO history_logs (id, user_id, content, interaction_type, created_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        history_rows,
    )
]

# ----------------------- chat_db -----------------------
chat_rows = [
    (str(uuid.uuid4()), user_rows[0][0], "Hello, Eindr!", now, "assistant", "Hi! How can I help?"),
]
DATA["chat_db"] = [
    (
        "INSERT INTO chat_history (id, user_id, query, created_at, responder, response) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        chat_rows,
    )
]

def seed_database(db_name: str, statements):
    dsn = f"dbname={db_name} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    for sql, rows in statements:
        execute_batch(cur, sql, rows)
    cur.close()
    conn.close()
    print(f"Seeded {db_name} ✔️")

if __name__ == "__main__":
    for db, stmts in DATA.items():
        try:
            seed_database(db, stmts)
        except Exception as e:
            print(f"⚠️  Skipped {db}: {e}") 
 
 