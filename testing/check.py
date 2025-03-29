# import sqlite3
# from dotenv import load_dotenv
# import psycopg2
# import os

# load_dotenv()

# # Connect to the database
# # conn = psycopg2.connect(
# #     host=os.getenv("DB_HOST", "localhost"),  # Default to localhost if not set
# #     database=os.getenv("DB_NAME", "bse_updates"),
# #     user=os.getenv("DB_USER", "postgres"),
# #     password=os.getenv("DB_PASSWORD", "")
# # )
# conn = sqlite3.connect("chroma_db/chroma.sqlite3")
# cursor = conn.cursor()

# # Execute the query
# # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# # cursor.execute("CREATE TABLE stock_updates (id INTEGER PRIMARY KEY AUTOINCREMENT,stock_code TEXT,submitted_date TIMESTAMP,announcement TEXT);")
# # cursor.execute("SELECT * FROM stock_updates ORDER BY submitted_date DESC LIMIT 5;")
# # cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' AND submitted_date <= '2025-02-20' ORDER BY submitted_date DESC LIMIT 5")
# # cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' AND submitted_date <= '2025-02-20 23:26:47.861055' ORDER BY submitted_date DESC LIMIT 5")
# # cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '501242' ORDER BY submitted_date DESC LIMIT 5;")
# # cursor.execute("SELECT * FROM stock_updates;")
# cursor.execute("SELECT * FROM stock_updates WHERE stock_code = '500164' ORDER BY submitted_date DESC LIMIT 5;")


# # Fetch results
# results = cursor.fetchall()

# # Print results
# for row in results:
#     print(row)

# # Close connection
# conn.close()



# # import requests
# # API_BASE_URL = "http://localhost:8000"
# # response = requests.get(f"{API_BASE_URL}/updates/500112?limit=10")
# # print(response.json())  # Check if API returns expected data
from langchain_chroma import Chroma

# Load the Chroma vector database
vector_db = Chroma(persist_directory="./chroma_db")

# Retrieve all stored documents
docs = vector_db.get(include=["metadatas", "documents"])

# Print the stored documents
for i, doc in enumerate(docs["documents"]):
    print(f"\n🔹 **Document {i+1}:**\n{doc}")
    print(f"📌 Metadata: {docs['metadatas'][i]}")

from langchain_chroma import Chroma

# Load the Chroma vector database
vector_db = Chroma(persist_directory="./chroma_db")

# Check the number of stored records
print(f"Total records in ChromaDB: {vector_db._collection.count()}")
