import os
import logging
from dotenv import load_dotenv
import psycopg2
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

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()]
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

    # def retrieve_stock_updates(self, stock_code, top_k=5):
    #     """
    #     Retrieve latest stock updates with unique and diverse results.
    #     """
    #     if not self.vector_store:
    #         logger.error("Vector store is not initialized.")
    #         return []

    #     docs = self.vector_store.max_marginal_relevance_search(
    #         query=f"Updates for stock {stock_code}",
    #         k=top_k
    #     )

    #     # Remove duplicates based on title
    #     seen_titles = set()
    #     unique_docs = []
    #     for doc in docs:
    #         if doc.metadata["title"] not in seen_titles:
    #             seen_titles.add(doc.metadata["title"])
    #             unique_docs.append(doc)

    #     return unique_docs
    # def retrieve_stock_updates(self, stock_code, top_k=5):
    #     """
    #     Retrieve latest stock updates from PostgreSQL, fallback to ChromaDB if needed.
    #     """
    #     try:
    #         with self.db_conn.cursor() as cur:
    #             cur.execute("""
    #                 SELECT DISTINCT ON (title) title, update_type, summary, submitted_date, file_url
    #                 FROM stock_updates
    #                 WHERE stock_code = %s
    #                 ORDER BY title, submitted_date DESC
    #                 LIMIT %s;
    #             """, (stock_code, top_k))

    #             updates = cur.fetchall()

    #         if updates:
    #             return [
    #                 {
    #                     "title": row["title"],
    #                     "update_type": row["update_type"],
    #                     "summary": row["summary"],
    #                     "submitted_date": row["submitted_date"].isoformat(),
    #                     "file_url": row["file_url"] if row["file_url"] else "Not available"
    #                 }
    #                 for row in updates
    #             ]

    #         # If no updates in PostgreSQL, fallback to ChromaDB
    #         return self.vector_store.similarity_search(f"Updates for {stock_code}", k=top_k)

    #     except Exception as e:
    #         logger.error(f"Error retrieving stock updates: {e}")
    #         return []
    # def retrieve_stock_updates(self, stock_code, top_k=5):
    #     """
    #     Retrieve latest stock updates from PostgreSQL, formatted for structured output.
    #     """
    #     try:
    #         with self.db_conn.cursor() as cur:
    #             cur.execute("""
    #                 SELECT DISTINCT ON (title) title, update_type, summary, submitted_date, file_url
    #                 FROM stock_updates
    #                 WHERE stock_code = %s
    #                 ORDER BY title, submitted_date DESC
    #                 LIMIT %s;
    #             """, (stock_code, top_k))

    #             updates = cur.fetchall()

    #         if updates:
    #             return [
    #                 f"- **{row['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}**\n"
    #                 f"- **Type:** {row['update_type']}\n"
    #                 f"- **Summary:** {row['summary'][:250]}...\n"
    #                 f"- [View Announcement]({row['file_url']})\n"
    #                 for row in updates
    #             ]

    #         # If no updates in PostgreSQL, fallback to ChromaDB for semantic match
    #         chroma_results = self.vector_store.similarity_search(f"Updates for {stock_code}", k=top_k)
    #         return [doc.page_content for doc in chroma_results]

    #     except Exception as e:
    #         logger.error(f"Error retrieving stock updates: {e}")
    #         return ["No relevant updates found."]

    # def retrieve_stock_updates(self, stock_code, top_k=3):
    #     """Retrieve stock updates efficiently using PostgreSQL first, then ChromaDB."""
    #     try:
    #         with self.db_conn.cursor() as cur:
    #             cur.execute("""
    #                 SELECT title, update_type, summary, submitted_date, file_url
    #                 FROM stock_updates
    #                 WHERE stock_code = %s
    #                 ORDER BY submitted_date DESC
    #                 LIMIT %s;
    #             """, (stock_code, top_k))
    #             updates = cur.fetchall()

    #         if updates:
    #             return [
    #                 f"- **{row['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}**\n"
    #                 f"- **Type:** {row['update_type']}\n"
    #                 f"- **Summary:** {row['summary'][:250]}...\n"
    #                 f"- [View Announcement]({row['file_url']})\n"
    #                 for row in updates
    #             ]

    #         # Use MMR search in ChromaDB for diversity
    #         return self.vector_store.max_marginal_relevance_search(
    #             f"Updates for {stock_code}",
    #             k=top_k
    #         )

    #     except Exception as e:
    #         logger.error(f"Error retrieving stock updates: {e}")
    #         return ["No relevant updates found."]

    def retrieve_stock_updates(self, stock_code, top_k=3):
        """Retrieve stock updates efficiently using PostgreSQL first, then ChromaDB."""
        try:
            # Use a cached response if available
            cache_key = f"{stock_code}_{top_k}"
            if hasattr(self, 'query_cache') and cache_key in self.query_cache:
                return self.query_cache[cache_key]
                
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT title, update_type, summary, submitted_date, file_url
                    FROM stock_updates
                    WHERE stock_code = %s
                    ORDER BY submitted_date DESC
                    LIMIT %s;
                """, (stock_code, top_k))
                updates = cur.fetchall()

            if updates:
                result = [
                    f"- **{row['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}**\n"
                    f"- **Type:** {row['update_type']}\n"
                    f"- **Summary:** {row['summary'][:250]}...\n"
                    f"- [View Announcement]({row['file_url']})\n"
                    for row in updates
                ]
                # Cache the result
                if hasattr(self, 'query_cache'):
                    self.query_cache[cache_key] = result
                return result

            # Use faster similarity search
            docs = self.vector_store.similarity_search(
                f"Updates for {stock_code}",
                k=top_k * 2  # Get more results initially
            )
            
            # Filter for diversity in memory
            seen_titles = set()
            filtered_docs = []
            for doc in docs:
                title = doc.metadata.get("title", "")
                if title not in seen_titles and len(filtered_docs) < top_k:
                    seen_titles.add(title)
                    filtered_docs.append(doc)
            
            # Return results
            result = [doc.page_content for doc in filtered_docs]
            if hasattr(self, 'query_cache'):
                self.query_cache[cache_key] = result
            return result

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
                result = cur.fetchone()
                print(f"Total records in stock_updates: {result['count']}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            # Using HuggingFace embeddings (all-MiniLM-L6-v2)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                # model_kwargs={"device": "cuda"}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Embedding model initialization error: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        try:
            # Check if persist directory exists
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                # Load existing vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
            else:
                # Create new vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                logger.info(f"Created new vector store at {self.persist_directory}")
                # Initial population of vector store
                self._populate_vector_store()
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            os.environ["OLLAMA_CUDA"] = "1"
            # self.llm = Ollama(model="mistral", streaming=True)
            # self.llm = Ollama(model="llama3.1")
            self.llm = Ollama(model="mistral")
            # self.llm = Ollama(model="llama3.1", streaming=True)
            # self.llm = Ollama(model="llama3.1", 
                            #   model_kwargs={"device": "cuda"}
                            #   )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")

            self.llm = None
            logger.warning("Falling back to template-based answers")
    
    # def _setup_rag_chain(self):
    #     """Set up the RAG chain using LangChain"""
    #     try:
    #         # template = """
    #         # You are an expert financial analyst specialized in analyzing BSE (Bombay Stock Exchange) corporate filings.
    #         # Answer the user's question about stock information based on the provided context.
            
    #         # Context: {context}
            
    #         # Question: {question}
            
    #         # Answer the question with detail and precision. If the information is not available in the context, 
    #         # say "I don't have enough information to answer this question from the available filings."
    #         # Focus only on the facts presented in the context and avoid speculation.
    #         # """
    #         template = """
    #                 You are an expert financial analyst specializing in BSE (Bombay Stock Exchange) corporate filings.
    #                 Your goal is to provide clear, concise answers based on the provided updates.

    #                 **Context:**
    #                 {context}

    #                 **User Question:**
    #                 {question}

    #                 **Guidelines:**
    #                 - Summarize relevant updates concisely.
    #                 - Remove duplicate entries.
    #                 - If multiple updates exist for the same event, only mention it once.
    #                 - If no relevant information is available, respond with: "No relevant updates found."

    #                 **Answer:**
    #                 """

    #         prompt = ChatPromptTemplate.from_template(template)
            
    #         self.chain = (
    #             {"context": lambda query: self.retrieve_stock_updates(stock_code=query), "question": RunnablePassthrough()}
    #             | prompt
    #             | self.llm
    #             | StrOutputParser()
    #         )
    #         logger.info("RAG chain initialized successfully")
    #     except Exception as e:
    #         logger.error(f"RAG chain initialization error: {e}")
    #         self.chain = None
    def _setup_rag_chain(self):
        """Set up the RAG chain using LangChain"""
        try:
            # template = """
            # You are an expert financial analyst specialized in analyzing BSE (Bombay Stock Exchange) corporate filings.
            # Answer the user's question about stock information based on the provided context.
            
            # Context: {context}
            
            # Question: {question}
            
            # Answer the question with detail and precision. If the information is not available in the context, 
            # say "I don't have enough information to answer this question from the available filings."
            # Focus only on the facts presented in the context and avoid speculation.
            # """
            template = """
                    You are an expert financial analyst specializing in BSE (Bombay Stock Exchange) corporate filings.
                    Your goal is to provide clear, concise answers based on the provided updates.

                    **Context:**
                    {context}

                    **User Question:**
                    {question}

                    **Guidelines:**
                    - Summarize relevant updates concisely.
                    - Remove duplicate entries.
                    - If multiple updates exist for the same event, only mention it once.
                    - If no relevant information is available, respond with: "No relevant updates found."

                    **Answer:**
                    """

            prompt = ChatPromptTemplate.from_template("""
            You are an AI financial analyst specializing in BSE stock updates.

            **Instructions:**
            - Provide a **concise summary** of stock updates.
            - **Do NOT list identical updates multiple times**.
            - If no relevant updates exist, respond with: "No relevant updates found."

            **Context:**
            {context}

            **User Question:**
            {question}

            **Structured Answer:**
            """)

            self.chain = (
                {
                    "context": lambda query: self.retrieve_stock_updates(stock_code=query),
                    "question": RunnablePassthrough()
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
        # retrieved_docs = self.vector_store.similarity_search(query="latest updates", k=5)
        # print(retrieved_docs)

        """Get the retriever from the vector store"""
        return self.vector_store.as_retriever(
            # search_type="similarity",
            # search_kwargs={"k": 5}
            search_type="mmr",
            search_kwargs={"k": 3}
        )
    
    def _populate_vector_store(self):
        """Populate vector store with existing stock updates"""
        try:
            # Get all stock codes to track
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT DISTINCT stock_code FROM stock_updates")
                stocks = [row["stock_code"] for row in cur.fetchall()]
            
            logger.info(f"Found {len(stocks)} stocks to index")
            
            for stock_code in stocks:
                self._add_stock_updates_to_vector_store(stock_code, days_limit=730)
                self.currently_tracked_stocks.add(stock_code)
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info("Vector store populated and persisted")
        except Exception as e:
            logger.error(f"Vector store population error: {e}")
    
    def _add_stock_updates_to_vector_store(self, stock_code, days_limit=365):
        """
        Add updates for a specific stock to the vector store while avoiding duplicates.
        
        Args:
            stock_code (str): The stock code to add updates for.
            days_limit (int): Limit to recent updates within this number of days.
        """
        try:
            # Get updates for the stock
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, stock_code, update_type, title, summary, file_url, submitted_date
                    FROM stock_updates
                    WHERE stock_code = %s
                    AND submitted_date >= %s
                    ORDER BY submitted_date DESC
                """, (stock_code, datetime.now() - timedelta(days=days_limit)))

                updates = cur.fetchall()

            if not updates:
                logger.info(f"No updates found for stock {stock_code}")
                return

            logger.info(f"Adding {len(updates)} updates for stock {stock_code} to vector store")

            # Track unique updates to avoid duplicates
            existing_titles = set()
            documents = []

            for update in updates:
                # Ensure summary is meaningful
                summary_text = update["summary"].strip() if update["summary"] else "Summary not available."

                # Check for duplicate titles
                if update["title"] in existing_titles:
                    continue  # Skip duplicates
                existing_titles.add(update["title"])

                # Format the text properly
                update_text = f"""
                STOCK: {update['stock_code']}
                UPDATE TYPE: {update['update_type']}
                TITLE: {update['title']}
                DATE: {update['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}
                
                SUMMARY:
                {summary_text}
                
                SOURCE URL: {update['file_url'] if update['file_url'] else 'Not available'}
                """

                # Create LangChain document
                doc = Document(
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
                documents.append(doc)

            # Split documents if they're too long (helps with retrieval)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_documents = text_splitter.split_documents(documents)

            # Add to vector store
            self.vector_store.add_documents(split_documents)
            self.vector_store.persist()  # Ensure updates are saved

            logger.info(f"Successfully added {len(split_documents)} unique updates for stock {stock_code}")

        except Exception as e:
            logger.error(f"Error adding updates for stock {stock_code} to vector store: {e}")

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
                        self._add_stock_updates_to_vector_store(stock_code)
                        self.currently_tracked_stocks.add(stock_code)
                    self.vector_store.persist()
                
                # Check for new updates to existing stocks
                for stock_code in self.currently_tracked_stocks:
                    # Get the latest update date in the vector store
                    latest_date = self._get_latest_update_date(stock_code)
                    
                    if latest_date:
                        # Get newer updates from database
                        self._add_new_updates(stock_code, latest_date)
                    else:
                        # No existing updates, add all
                        self._add_stock_updates_to_vector_store(stock_code)
                
                # Sleep for a while
                time.sleep(3600)  # Check once per hour
                
            except Exception as e:
                logger.error(f"Error in background update thread: {e}")
                time.sleep(300)  # Sleep shorter time on error
    
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
                limit=5,
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
            
            # Prepare documents
            documents = []
            for update in updates:
                # Create a rich text representation of the update
                update_text = f"""
                STOCK: {update['stock_code']}
                UPDATE TYPE: {update['update_type']}
                TITLE: {update['title']}
                DATE: {update['submitted_date'].strftime('%Y-%m-%d %H:%M:%S')}
                
                SUMMARY:
                {update['summary']}
                
                SOURCE URL: {update['file_url'] if update['file_url'] else 'Not available'}
                """
                
                # Create LangChain document
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
            
            # Split documents if they're too long
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_documents = text_splitter.split_documents(documents)
            
            # Add to vector store
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
            
            # Add stock code context if provided
            if stock_code:
                question = f"For stock {stock_code}: {question}"
            
            # Run the chain
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return f"Sorry, an error occurred: {str(e)}"
    
    # def query(self, question, stock_code=None, stream=True):
    #     """
    #     Query the RAG system with streaming support
        
    #     Args:
    #         question (str): The user's question
    #         stock_code (str, optional): Specific stock code to filter by
    #         stream (bool): Whether to stream the response
            
    #     Returns:
    #         str: The response text
    #     """
    #     try:
    #         if not self.chain:
    #             return "Sorry, the RAG system is not fully initialized. Please try again later."

    #         # Add stock code context if provided
    #         if stock_code:
    #             question = f"For stock {stock_code}: {question}"
            
    #         if stream:
    #             response = ""
    #             for chunk in self.chain.stream(question):
    #                 response += chunk
    #             return response
    #         else:
    #             # Return the complete response
    #             return self.chain.invoke(question)
                
    #     except Exception as e:
    #         logger.error(f"Error querying RAG system: {e}")
    #         return f"Sorry, an error occurred: {str(e)}"
    
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
        
        # Persist vector store one last time
        if self.vector_store:
            self.vector_store.persist()
            logger.info("Vector store persisted")