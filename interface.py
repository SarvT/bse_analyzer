import gradio as gr
import os
import logging
import requests
import json
from dotenv import load_dotenv
from rag_system import BSEUpdatesRAG
from datetime import date

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("bse_app")

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize RAG system
rag_system = BSEUpdatesRAG()

def get_available_stocks():
    """Get available stocks from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stocks")
        response.raise_for_status()
        stocks = response.json()
        return [s["stock_code"] for s in stocks]
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        return rag_system.get_available_stocks() 

def get_stock_updates(stock_code, limit=10):
    """Get recent updates for a stock"""
    try:
        # results = self.vector_store.similarity_search(f"Updates for {stock_code}", search_kwargs={"k": 5})
        # return results if results else ["No updates found."]
        # print(results)
        response = requests.get(f"{API_BASE_URL}/updates/{stock_code}?limit={limit}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching updates for {stock_code}: {e}")
        return []

def format_updates(updates):
    """Format updates for better UI display with duplicate removal."""
    if not updates:
        return "No recent updates found."

    seen_titles = set()
    formatted = ""
    for i, update in enumerate(updates):
        if update["title"] in seen_titles:
            continue  # Skip duplicates
        seen_titles.add(update["title"])

        # Convert timestamp to readable format
        try:
            date_obj = date.fromisoformat(update["submitted_date"])
            formatted_date = date_obj.strftime("%b %d, %Y %I:%M %p")
        except ValueError:
            formatted_date = update["submitted_date"]

        formatted += f"### {i+1}. {update['title']}\n\n"
        formatted += f"** Type:** `{update['update_type']}`\n"
        formatted += f"** Date:** `{formatted_date}`\n\n"
        formatted += f"** Summary:** {update['summary'][:250]}...\n\n"
        if update.get("file_url"):
            formatted += f" [View Document]({update['file_url']})\n\n"
        formatted += "---\n\n"  

    return formatted


def ask_question(stock_code, question):
    """Ask a question about the stock updates"""
    if not question.strip():
        return "Please enter a question."
    
    try:
        answer = rag_system.query(question, stock_code)
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return f"Sorry, an error occurred while processing your question: {str(e)}"

def update_stock_info(stock_code):
    """Update information when stock is selected"""
    if not stock_code:
        return "", "No stock selected. Please select a stock to view updates and ask questions."
    
    updates = get_stock_updates(stock_code)
    formatted_updates = format_updates(updates)
    return formatted_updates, f"Selected stock: {stock_code}. You can now ask questions about this stock's updates."

with gr.Blocks(theme=gr.themes.Soft(), title="BSE Stock Updates") as app:
    gr.Markdown("#  BSE Stock Updates Analyzer", elem_id="header")
    gr.Markdown(" Get real-time corporate filings from BSE with AI-powered insights.")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            stock_selector = gr.Dropdown(
                label=" Select Stock Code",
                choices=get_available_stocks(),
                interactive=True
            )
            refresh_button = gr.Button("Refresh Stocks")

            gr.Markdown("### About this App", elem_id="about-section")
            gr.Markdown("""
            -  **Real-time tracking of corporate filings**
            -  **AI-powered Q&A system**
            -  **Links to original documents**
            """, elem_id="about-text")
        
        with gr.Column(scale=2, min_width=500):
            status_text = gr.Markdown(" **Please select a stock to view updates**", elem_id="status")
            updates_display = gr.Markdown(" **No updates to display yet**", elem_id="updates")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Ask Questions About Stock Updates", elem_id="qa-section")
            question_input = gr.Textbox(
                label=" Your Question",
                placeholder="Example: What are the recent financial results?",
                lines=2
            )
            ask_button = gr.Button(" Ask AI")
            answer_output = gr.Markdown(" **Answers will appear here**", elem_id="answer-box")

    # Ensure layout spacing and styles
    app.css = """
    #header { text-align: center; font-size: 24px; font-weight: bold; }
    #about-section { font-size: 18px; font-weight: bold; }
    #about-text { font-size: 14px; color: gray; }
    #status { font-size: 16px; font-weight: bold; }
    #updates { font-size: 14px; line-height: 1.6; }
    #qa-section { font-size: 18px; font-weight: bold; margin-top: 20px; }
    #answer-box { font-size: 14px; line-height: 1.5; color: #4CAF50; }
    """
    
    # Set up interactions
    stock_selector.change(update_stock_info, inputs=stock_selector, outputs=[updates_display, status_text])
    refresh_button.click(lambda: gr.update(choices=get_available_stocks()), outputs=stock_selector)
    ask_button.click(ask_question, inputs=[stock_selector, question_input], outputs=answer_output)

if __name__ == "__main__":
    app.launch(share=False)