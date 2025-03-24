import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import os
from datetime import datetime
import re
import PyPDF2
import io
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from urllib.parse import urljoin
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bse_scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger("bse_scraper")

load_dotenv()
BASE_PDF_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachHis/"  # Correct path for attachments

class BSEScraper:
    """
    Scraper for BSE corporate filings for a specific stock
    """
    
    def __init__(self, stock_code):
        """
        Initialize the scraper with a specific stock code
        
        Args:
            stock_code (str): The BSE security code or ticker
        """
        self.conn = None
        self.connect_to_db()
        self.stock_code = stock_code
        self.base_url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        self.api_url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

        # self.filing_url = "https://www.bseindia.com/corporates/ann.html"
        # self.headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        #     "Accept-Language": "en-US,en;q=0.5",
        #     "Referer": "https://www.bseindia.com/",
        #     "Connection": "keep-alive",
        #     "Upgrade-Insecure-Requests": "1",
        # }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bseindia.com/corporates/ann.html"
        }

        
    
    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "bse_updates"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "")
            )
            logger.info("Connected to database successfully")
            
            self._create_tables()
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def fetch_stock_updates(self):
        """Fetch stock updates from the BSE API."""
        today = datetime.now().strftime("%Y%m%d")
        one_year_ago = (datetime.now() - timedelta(days=(365*2))).strftime("%Y%m%d")
        # print(today)
        # print(one_year_ago)

        params = {
            "pageno": "1",
            "strCat": "-1",
            "strPrevDate": one_year_ago, 
            "strScrip": self.stock_code,
            "strSearch": "P",  
            "strToDate": today,
            "strType": "C", 
            "subcategory": "-1"
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bseindia.com/corporates/ann.html"
        }

        response = requests.get(self.base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            # print("fetched data: ", data)
            logger.info(f"Fetched {len(data.get('Table', []))} updates for stock {self.stock_code}")
            return data.get("Table", [])
        else:
            logger.error(f"Failed to fetch updates: {response.status_code}")
            return []
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with self.conn.cursor() as cur:
            # Create stock_updates table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS stock_updates (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(20) NOT NULL,
                    update_type VARCHAR(244),
                    title TEXT,
                    summary TEXT,
                    file_url TEXT,
                    file_type VARCHAR(10),
                    submitted_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_content JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_updates(stock_code);
                CREATE INDEX IF NOT EXISTS idx_update_type ON stock_updates(update_type);
                CREATE INDEX IF NOT EXISTS idx_submitted_date ON stock_updates(submitted_date);
            ''')
            self.conn.commit()
    
    def _get_session(self):
        """Get a session with cookies and headers set"""
        session = requests.Session()
        session.headers.update(self.headers)
        
        try:
            session.get(self.base_url)
            return session
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            raise
    
    def search_stock_updates(self):
        """
        Search for updates related to the specified stock code
        """
        session = self._get_session()
        
        form_data = {
            "scripcode": self.stock_code,
            "fromdt": "", 
            "todt": "",    
            "symbol": "",
            "segmentid": "0",
            "categoryid": "0"
        }
        try:
            response = session.get(self.filing_url)

            if response.status_code != 200:
                logger.error(f"Failed to access BSE filing page: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for hidden_input in soup.find_all("input", type="hidden"):
                if hidden_input.get("name") and hidden_input.get("value"):
                    form_data[hidden_input["name"]] = hidden_input["value"]
            
            search_response = session.post(self.filing_url, data=form_data)
            if search_response.status_code != 200:
                logger.error(f"Search request failed: {search_response.status_code}")
                return []
            
            results_soup = BeautifulSoup(search_response.content, 'html.parser')
            updates = self._parse_results(results_soup)
            
            return updates
        except Exception as e:
            logger.error(f"Error searching for stock updates: {e}")
            return []
    def _parse_results(self, soup):
        """
        Parse the search results page and extract filing information
        
        Args:
            soup (BeautifulSoup): The parsed HTML of the results page
            
        Returns:
            list: List of dictionaries containing filing information
        """
        updates = []
        
        results_table = soup.find("table", {"id": "announcementTable"})
        if not results_table:
            logger.warning("Could not find results table")
            return updates
        
        for row in results_table.find_all("tr")[1:]:
            try:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                
                date_str = cols[0].text.strip()
                time_str = cols[1].text.strip()
                subject = cols[2].text.strip()
                
                file_link = None
                file_type = None
                link_elem = cols[3].find("a")
                if link_elem and "href" in link_elem.attrs:
                    file_link = urljoin(self.base_url, link_elem["href"])
                    if file_link.lower().endswith(".pdf"):
                        file_type = "pdf"
                    elif file_link.lower().endswith(".xbrl"):
                        file_type = "xbrl"
                    else:
                        file_type = "unknown"
                
                update_type = "General"
                update_types = ["Financial Results", "Board Meeting", "Dividend", 
                               "AGM", "EGM", "Investor Presentation", "Earnings Call",
                               "Credit Rating", "Acquisition", "Merger", "Demerger",
                               "Rights Issue", "Bonus Issue", "Stock Split"]
                
                for type_name in update_types:
                    if type_name.lower() in subject.lower():
                        update_type = type_name
                        break
                
                date_time_str = f"{date_str} {time_str}"
                submitted_date = datetime.strptime(date_time_str, "%d-%m-%Y %H:%M:%S")
                
                update = {
                    "stock_code": self.stock_code,
                    "update_type": update_type,
                    "title": subject,
                    "file_url": file_link,
                    "file_type": file_type,
                    "submitted_date": submitted_date,
                    "summary": None,
                    "raw_content": {
                        "date": date_str,
                        "time": time_str,
                        "subject": subject
                    }
                }
                
                updates.append(update)
                
            except Exception as e:
                logger.error(f"Error parsing row: {e}")
                continue
        # print("updates: ", updates)
        return updates
    
    def _extract_text_from_pdf(self, pdf_url):
        """
        Download PDF and extract text content
        
        Args:
            pdf_url (str): URL to the PDF file
            
        Returns:
            str: Extracted text or None if extraction fails
        """
        try:
            response = requests.get(pdf_url, headers=self.headers, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download PDF: {response.status_code}")
                return None
            
            with io.BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                
                max_pages = min(5, len(reader.pages))
                for page_num in range(max_pages):
                    text += reader.pages[page_num].extract_text() + "\n"
                
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def _generate_summary(self, update):
        """
        Generate a summary for the update based on its content
        
        Args:
            update (dict): The update information
            
        Returns:
            str: A generated summary
        """
        text_content = None
        if update["file_url"] and update["file_type"] == "pdf":
            text_content = self._extract_text_from_pdf(update["file_url"])
        
        if text_content:
            summary = text_content[:500].strip()
            summary = re.sub(r'\s+', ' ', summary) 
            if len(text_content) > 500:
                summary += "..."
        else:
            summary = f"Update regarding {update['update_type']}: {update['title']}"
        
        return summary
    
    # def save_updates(self, updates):
    #     """Save updates to PostgreSQL and index in ChromaDB."""
    #     with self.db_conn.cursor() as cur:
    #         for update in updates:
    #             try:
    #                 # Check if update exists
    #                 cur.execute("SELECT id FROM stock_updates WHERE stock_code = %s AND title = %s",
    #                             (self.stock_code, update.get("HEADLINE")))
    #                 existing = cur.fetchone()
    #                 if existing:
    #                     continue  # Skip duplicate

    #                 # Insert into PostgreSQL
    #                 cur.execute("""
    #                     INSERT INTO stock_updates (stock_code, update_type, title, summary, submitted_date, file_url)
    #                     VALUES (%s, %s, %s, %s, %s, %s)
    #                 """, (
    #                     self.stock_code,
    #                     update.get("NEWSSUB")[:100],
    #                     update.get("HEADLINE"),
    #                     update.get("HEADLINE"),
    #                     update.get("NEWS_DT"),
    #                     update.get("PDF_LINK")
    #                 ))
    #                 self.db_conn.commit()

    #                 # Add to ChromaDB (Vector Store)
    #                 doc_text = f"""
    #                 STOCK: {self.stock_code}
    #                 UPDATE TYPE: {update.get('NEWSSUB')}
    #                 TITLE: {update.get('HEADLINE')}
    #                 DATE: {update.get('NEWS_DT')}

    #                 SUMMARY:
    #                 {update.get('HEADLINE')}

    #                 SOURCE URL: {update.get('PDF_LINK')}
    #                 """
    #                 self.vector_store.add_documents([Document(page_content=doc_text)])

    #             except Exception as e:
    #                 logger.error(f"Error inserting update: {e}")
    #                 self.db_conn.rollback()

    def save_updates(self, updates):
        """Fetch real-time stock updates, format them correctly, and store in PostgreSQL."""
        with self.conn.cursor() as cur:
            for update in updates:
                try:
                    # Extract attachment filename and create full PDF link
                    attachment_name = update.get("ATTACHMENTNAME", "").strip()
                    full_pdf_link = BASE_PDF_URL + attachment_name if attachment_name else "Not available"
                    
                    # Format announcement details
                    headline = update.get("HEADLINE", "")[:250]  # Safe truncation
                    
                    # Check if update already exists - using the same data format for check and insert
                    cur.execute("""
                        SELECT id FROM stock_updates 
                        WHERE stock_code = %s AND title = %s AND submitted_date = %s
                    """, (
                        self.stock_code, 
                        headline,
                        update.get("NEWS_DT")
                    ))
                    
                    existing = cur.fetchone()
                    if existing:
                        logger.info(f"Skipping duplicate entry: {headline}")
                        continue  # Skip duplicate

                    # Insert into PostgreSQL
                    cur.execute("""
                        INSERT INTO stock_updates (stock_code, update_type, title, summary, submitted_date, file_url, raw_content)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self.stock_code,
                        update.get("NEWSSUB", "")[:100],  # Truncate type
                        headline,
                        headline,  # Using headline as summary too
                        update.get("NEWS_DT"),
                        full_pdf_link,
                        Json(update)  # Store the complete raw data
                    ))
                    self.conn.commit()
                    logger.info(f"Inserted new update: {headline}")

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
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BSE Corporate Filings Scraper")
    parser.add_argument("stock_code", help="BSE Stock Code or Ticker")
    args = parser.parse_args()
    
    try:
        scraper = BSEScraper(args.stock_code)
        scraper.run()
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    finally:
        if 'scraper' in locals():
            scraper.close()

