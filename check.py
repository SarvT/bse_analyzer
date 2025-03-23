import sqlite3
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()

# Connect to the database
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),  # Default to localhost if not set
    database=os.getenv("DB_NAME", "bse_updates"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "")
)
# conn = sqlite3.connect("chroma_db/chroma.sqlite3")
cursor = conn.cursor()

# Execute the query
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# cursor.execute("CREATE TABLE stock_updates (id INTEGER PRIMARY KEY AUTOINCREMENT,stock_code TEXT,submitted_date TIMESTAMP,announcement TEXT);")
# cursor.execute("SELECT * FROM stock_updates ORDER BY submitted_date DESC LIMIT 5;")
# cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' AND submitted_date <= '2025-02-20' ORDER BY submitted_date DESC LIMIT 5")
# cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' AND submitted_date <= '2025-02-20 23:26:47.861055' ORDER BY submitted_date DESC LIMIT 5")
# cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' ORDER BY submitted_date DESC LIMIT 5;")
# cursor.execute("SELECT * FROM stock_updates;")

# Fetch results
results = cursor.fetchall()

# Print results
for row in results:
    print(row)

# Close connection
conn.close()
