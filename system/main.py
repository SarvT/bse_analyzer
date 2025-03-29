import os
import argparse
import logging
import threading
import time
from dotenv import load_dotenv
import uvicorn
from api_service import app as api_app
from scraper_mod import BSEScraper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler()]
)
logger = logging.getLogger("main")

class RestartHandler(FileSystemEventHandler):
    """Watches for file changes and restarts the script"""
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            logger.info(f"Detected change in {event.src_path}. Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)


def run_scraper(stock_code, interval):
    """Run the BSE scraper in a separate thread"""
    try:
        scraper = BSEScraper(stock_code)
        scraper.run()
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    finally:
        if 'scraper' in locals():
            scraper.close()

def run_api_server(host, port):
    """Run the FastAPI server"""
    try:
        uvicorn.run(api_app, host=host, port=port)
    except Exception as e:
        logger.error(f"API server error: {e}")

def run_gradio_app():
    """Run the Gradio app in a subprocess"""
    try:
        os.system("python interface.py")
    except Exception as e:
        logger.error(f"Gradio app error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSE Stock Updates System")
    parser.add_argument("stock_code", help="BSE Stock Code or Ticker")
    parser.add_argument("--interval", type=int, default=3600, 
                        help="Interval between checks in seconds (default: 3600)")
    parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()

    # Watchdog for automatic restart
    event_handler = RestartHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()

    scraper_thread = threading.Thread(
        target=run_scraper,
        args=(args.stock_code, args.interval),
        daemon=True
    )
    scraper_thread.start()
    logger.info(f"Started scraper for stock code: {args.stock_code}")
    
    api_thread = threading.Thread(
        target=run_api_server,
        args=(args.api_host, args.api_port),
        daemon=True
    )
    api_thread.start()
    logger.info(f"Started API server at {args.api_host}:{args.api_port}")
    
    time.sleep(10)
    
    logger.info("Starting Gradio app...")
    run_gradio_app()

    observer.join()