import requests
import datetime
import logging
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bse_scraper")

class BSEScraper:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.base_url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

        # Connect to the database
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "bse_updates"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )

    def fetch_stock_updates(self):
        """Fetch stock updates from the BSE API."""
        today = datetime.datetime.now().strftime("%Y%m%d")
        last_month = (datetime.datetime.now().replace(day=1)).strftime("%Y%m%d")

        params = {
            "pageno": "1",
            "strCat": "-1",  # Default category
            "strPrevDate": last_month,  # Start date (YYYYMMDD)
            "strScrip": self.stock_code,  # Stock code
            "strSearch": "P",  # Search filter (Unknown usage, keep it)
            "strToDate": today,  # End date (YYYYMMDD)
            "strType": "C",  # Announcement Type
            "subcategory": "-1"  # Default subcategory
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bseindia.com/corporates/ann.html"
        }

        response = requests.get(self.base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            print(data)
            logger.info(f"Fetched {len(data.get('Table', []))} updates for stock {self.stock_code}")
            return data.get("Table", [])
        else:
            logger.error(f"Failed to fetch updates: {response.status_code}")
            return []

    def save_updates(self, updates):
        """Save updates to the database."""
        with self.conn.cursor() as cur:
            for update in updates:
                try:
                    cur.execute(
                        "INSERT INTO stock_updates (stock_code, update_type, title, summary, submitted_date, raw_content) VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            self.stock_code,
                            update.get("NEWSSUB")[:100],  # Announcement type
                            update.get("HEADLINE"),  # Title
                            update.get("HEADLINE"),  # Use headline as summary (or extract from content)
                            update.get("NEWS_DT"),  # Date
                            Json(update)  # Store raw JSON data
                        )
                    )
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error inserting update: {e}")
                    self.conn.rollback()

    def run(self):
        """Run the scraper."""
        updates = self.fetch_stock_updates()
        if updates:
            self.save_updates(updates)
        else:
            logger.info("No updates found.")

if __name__ == "__main__":
    code = input("the stock code?")
    scraper = BSEScraper(code)  # Example: Reliance Industries
    scraper.run()
