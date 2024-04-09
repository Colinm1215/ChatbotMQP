import signal
import sqlite3
import sys

conn = sqlite3.connect('User_Database.db')


def signal_handler(sig, frame):
    close_database()
    sys.exit(0)


def create_database():
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mytable
                 (id INTEGER PRIMARY KEY, name TEXT)''')
    conn.commit()


def close_database():
    if conn:
        conn.close()


def insert_user(id, name):
    c = conn.cursor()
    c.execute("INSERT INTO mytable (id, name) VALUES (?, ?)", (id, name))
    conn.commit()


def search_id_by_name(name):
    c = conn.cursor()
    c.execute("SELECT * FROM mytable WHERE name=?", (name,))
    result = c.fetchone()
    return result


def search_name_by_id(id):
    c = conn.cursor()
    c.execute("SELECT * FROM mytable WHERE id=?", (id,))
    result = c.fetchone()
    return result


def generate_new_id():
    c = conn.cursor()
    c.execute("SELECT MAX(id) FROM mytable")
    max_id = c.fetchone()[0]
    new_id = max_id + 1 if max_id is not None else 1
    return new_id


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
create_database()
