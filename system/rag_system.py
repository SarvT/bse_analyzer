import os
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
import threading
import time
from datetime import datetime, timedelta
# import asyncpg
import json


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_system.log", encoding="utf-8"), 
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("rag_system")




class BSEUpdatesRAG:
    """
    RAG system for BSE stock updates
    """
    
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize the RAG system
        
        Args:
            persist_directory (str): Directory to persist vector database
        """
        # self.vector_store.delete_collection()
        # self._populate_vector_store()



        self.persist_directory = persist_directory
        self.db_conn = None
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.chain = None
        self.currently_tracked_stocks = set()
        self.cache = {}
        
        # Initialize components
        self._connect_to_db()
        self._initialize_embedding_model()
        self._initialize_vector_store()
        self._initialize_llm()
        self._setup_rag_chain()
        
        # Start background update thread
        self.should_run = True
        self.update_thread = threading.Thread(target=self._background_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()



    def retrieve_stock_updates(self, stock_code, top_k=3):
        """Retrieve stock updates efficiently from PostgreSQL, including source document details."""
        try:
            # Ensure stock_code is a string and stripped
            stock_code = str(stock_code).strip()
            
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT title, update_type, summary, submitted_date, file_url
                    FROM stock_updates
                    WHERE stock_code = %s
                    ORDER BY submitted_date DESC
                    LIMIT %s;
                """, (stock_code, top_k))
                updates = cur.fetchall()

            logger.info(f"📊 Retrieved Updates for {stock_code}: {updates}")

            if updates:
                result = [
                    f"- **{row['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}**\n"
                    f"- **Type:** {row['update_type']}\n"
                    f"- **Summary:** {row['summary']}\n"
                    f"- **Source Document:** [{row['title']}]({row['file_url']})\n"
                    for row in updates
                ]
                return result

            return ["No relevant updates found."]

        except Exception as e:
            logger.error(f"Error retrieving stock updates: {e}")
            return ["No relevant updates found."]


    
    def _connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "bse_updates"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to database successfully")
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM stock_updates;")
                result =  cur.fetchone()
                print(f"Total records in stock_updates: {result['count']}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            # self.embedding_model = HuggingFaceEmbeddings(
            #     model_name="sentence-transformers/all-MiniLM-L6-v2",
            #     # model_kwargs={"device": "cuda"}
            # )
            self.embedding_model =  HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Embedding model initialization error: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self.vector_store =  Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_metadata={"hnsw:space": "cosine", "hnsw:M": 16}
                )
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
            else:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                logger.info(f"Created new vector store at {self.persist_directory}")
                self._populate_vector_store()
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            os.environ["OLLAMA_CUDA"] = "1"
            # self.llm = Ollama(model="mistral", streaming=True)
            self.llm =  Ollama(model="llama3.1")
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")

            self.llm = None
            logger.warning("Falling back to template-based answers")
    
    def _setup_rag_chain(self):
        """Set up the RAG chain using LangChain"""
        try:
            template = """
                You are an AI financial analyst specializing in BSE stock updates.  
                **Use ONLY the provided context to answer the question.**  

                **Context:**  
                {context}

                **User Question:**  
                {question}

                **Instructions:**  
                - If the context **is empty**, reply: "No relevant updates found."  
                - **Always mention the source document title and link** used in the answer.  
                - **Do NOT use outside knowledge** or guess stock performance.  
                - Only summarize the provided stock updates.

                **Final Answer (Include Source):**
                """

            prompt =  ChatPromptTemplate.from_template(template)

            # self.chain = (
            #     {
            #         "context": lambda query: "\n\n".join(self.retrieve_stock_updates(stock_code=query, top_k=10)),
            #         "question": RunnablePassthrough()
            #     }
            #     | prompt
            #     | self.llm
            #     | StrOutputParser()
            # )
            self.chain = (
                {
                    "context": lambda query: "\n\n".join(
                        self.retrieve_stock_updates_from_json(stock_code=query, query=query, top_k=10)
                    ),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            
            logger.info("RAG chain initialized successfully")
        except Exception as e:
            logger.error(f"RAG chain initialization error: {e}")
            self.chain = None
    
    @property
    def _retriever(self):
        """Get the retriever from the vector store"""
        return self.vector_store.as_retrivever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
    
    def _populate_vector_store(self):
        """Populate vector store with existing stock updates"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT DISTINCT stock_code FROM stock_updates")
                stocks = [row["stock_code"] for row in cur.fetchall()]
            
            logger.info(f"Found {len(stocks)} stocks to index")
            
            for stock_code in stocks:
                self._add_stock_updates_to_vector_store(stock_code, days_limit=100)
                self.currently_tracked_stocks.add(stock_code)
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info("Vector store populated and persisted")
        except Exception as e:
            logger.error(f"Vector store population error: {e}")

    def _save_to_json(self, stock_code, updates):
        file_path = f"./json_data/{stock_code}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(updates, f, indent=4, default=str)
        logger.info(f"✅ Saved {len(updates)} updates for {stock_code} to JSON")

    def retrieve_stock_updates_from_json(self, stock_code, query, top_k=3):
        file_path = f"./json_data/{stock_code}.json"
        if not os.path.exists(file_path):
            return ["No relevant updates found."]

        with open(file_path, "r", encoding="utf-8") as f:
            updates = json.load(f)

        # Simple keyword-based filtering
        filtered_updates = [
            update for update in updates if query.lower() in update["summary"].lower()
        ][:top_k]

        if filtered_updates:
            return [
                f"- **{update['submitted_date']}**\n"
                f"- **Type:** {update['update_type']}\n"
                f"- **Summary:** {update['summary']}\n"
                f"- **Source Document:** [{update['title']}]({update['file_url']})\n"
                for update in filtered_updates
            ]
        
        return ["No relevant updates found."]


    def _add_stock_updates_to_json(self, stock_code, days_limit=365):
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, stock_code, update_type, title, summary, file_url, submitted_date
                    FROM stock_updates
                    WHERE stock_code = %s
                    AND submitted_date >= %s
                    ORDER BY submitted_date DESC;
                """, (stock_code, datetime.now() - timedelta(days=days_limit)))

                updates = cur.fetchall()

            if not updates:
                logger.info(f"⚠️ No updates found for stock {stock_code}")
                return

            self._save_to_json(stock_code, updates)

        except Exception as e:
            logger.error(f"❌ Error saving updates for stock {stock_code}: {e}")

    def _add_stock_updates_to_vector_store(self, stock_code, days_limit=365):
        """
        Add updates for a specific stock to the vector store while avoiding duplicates.
        """
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, stock_code, update_type, title, summary, file_url, submitted_date
                    FROM stock_updates
                    WHERE stock_code = %s
                    AND submitted_date >= %s
                    ORDER BY submitted_date DESC;
                """, (stock_code, datetime.now() - timedelta(days=days_limit)))

                updates = cur.fetchall()

            if not updates:
                logger.info(f"⚠️ No updates found for stock {stock_code}")
                return

            logger.info(f"✅ Adding {len(updates)} updates for stock {stock_code} to ChromaDB")

            documents = []
            for update in updates:
                summary_text = update["summary"].strip() if update["summary"] else "No summary available."
                
                # ✅ Log before adding
                logger.info(f"📌 Storing: {update['title']} | {summary_text[:100]}...")

                doc = Document(
                    page_content=summary_text,
                    metadata={
                        "id": update["id"],
                        "stock_code": update["stock_code"],
                        "update_type": update["update_type"],
                        "title": update["title"],
                        "submitted_date": update["submitted_date"].isoformat(),
                        "source": update["file_url"] if update["file_url"] else "Not available"
                    }
                )
                documents.append(doc)

            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"✅ Successfully added {len(documents)} documents to ChromaDB")

        except Exception as e:
            logger.error(f"❌ Error adding updates for stock {stock_code} to vector store: {e}")


    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache = {}

    
    def _background_update_loop(self):
        """Background thread to periodically check for new updates"""
        logger.info("Starting background update thread")
        
        while self.should_run:
            try:
                # Check for new stock codes
                with self.db_conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT stock_code FROM stock_updates")
                    current_stocks = {row["stock_code"] for row in cur.fetchall()}
                
                # Check for new stocks that aren't being tracked
                new_stocks = current_stocks - self.currently_tracked_stocks
                if new_stocks:
                    logger.info(f"Found {len(new_stocks)} new stocks to track: {new_stocks}")
                    for stock_code in new_stocks:
                        self._add_stock_updates_to_vector_store(stock_code, days_limit=100)
                        self.currently_tracked_stocks.add(stock_code)
                    self.vector_store.persist()
                
                for stock_code in self.currently_tracked_stocks:
                    latest_date = self._get_latest_update_date(stock_code)
                    
                    if latest_date:
                        self._add_new_updates(stock_code, latest_date)
                    else:
                        self._add_stock_updates_to_vector_store(stock_code, days_limit=100)
                
                time.sleep(3600)  
            except Exception as e:
                logger.error(f"Error in background update thread: {e}")
                time.sleep(300) 

    def _get_latest_update_date(self, stock_code):
        """
        Get the latest update date for a stock in the vector store
        
        Args:
            stock_code (str): The stock code
            
        Returns:
            datetime or None: The latest update date or None if no updates
        """
        try:
            # Query the vector store for metadata
            results = self.vector_store.get(
                where={"stock_code": stock_code},
                limit=10,
                include=["metadatas"]
            )
            
            if results and results["metadatas"]:
                # Get the submitted_date from metadata
                date_str = results["metadatas"][0].get("submitted_date")
                if date_str:
                    return datetime.fromisoformat(date_str)
            
            return None
        except Exception as e:
            logger.error(f"Error getting latest update date: {e}")
            return None
    
    def _add_new_updates(self, stock_code, latest_date):
        """
        Add new updates that are more recent than the latest date
        
        Args:
            stock_code (str): The stock code
            latest_date (datetime): The latest update date in the vector store
        """
        try:
            # Get updates newer than latest_date
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, stock_code, update_type, title, summary, file_url, submitted_date
                    FROM stock_updates
                    WHERE stock_code = %s
                    AND submitted_date > %s
                    ORDER BY submitted_date DESC
                """, (stock_code, latest_date))
                
                updates = cur.fetchall()
            
            if not updates:
                return
            
            logger.info(f"Adding {len(updates)} new updates for stock {stock_code}")
            
            documents = []
            for update in updates:
                update_text = f"""
                STOCK: {update['stock_code']}
                UPDATE TYPE: {update['update_type']}
                TITLE: {update['title']}
                DATE: {update['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}
                
                SUMMARY:
                {update['summary']}
                
                SOURCE URL: {update['file_url'] if update['file_url'] else 'Not available'}
                """
                
                documents.append(
                    Document(
                        page_content=update_text,
                        metadata={
                            "id": update["id"],
                            "stock_code": update["stock_code"],
                            "update_type": update["update_type"],
                            "title": update["title"],
                            "submitted_date": update["submitted_date"].isoformat(),
                            "source": update["file_url"] if update["file_url"] else "Not available"
                        }
                    )
                )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_documents = text_splitter.split_documents(documents)
            
            self.vector_store.add_documents(split_documents)
            self.vector_store.persist()
            
        except Exception as e:
            logger.error(f"Error adding new updates for stock {stock_code}: {e}")
    
    def query(self, question, stock_code=None):
        """
        Query the RAG system
        
        Args:
            question (str): The user's question
            stock_code (str, optional): Specific stock code to filter by
            
        Returns:
            str: The answer to the question
        """
        try:
            if not self.chain:
                return "Sorry, the RAG system is not fully initialized. Please try again later."
            
            if stock_code is None:
                import re
                match = re.search(r'\b(\d{6})\b', question)
                if match:
                    stock_code = match.group(1)
            
            if stock_code:
                logger.info(f"📍 Extracted Stock Code: {stock_code}")
                
                logger.info(f"🧠 Final Context Sent to LLM for {stock_code}: {self.retrieve_stock_updates(stock_code, top_k=10)}")
                
                response = self.chain.invoke(f"For stock {stock_code}: {question}")
            else:
                response = self.chain.invoke(question)
            
            return response
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return f"Sorry, an error occurred: {str(e)}"
    
    def get_available_stocks(self):
        """
        Get all available stocks in the database
        
        Returns:
            list: List of stock codes
        """
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT DISTINCT stock_code FROM stock_updates ORDER BY stock_code")
                return [row["stock_code"] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting available stocks: {e}")
            return []
    
    def close(self):
        """Clean up resources"""
        self.should_run = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")
        
        if self.vector_store:
            self.vector_store.persist()
            logger.info("Vector store persisted")