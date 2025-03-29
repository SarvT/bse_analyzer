import os
from typing import List, Dict, Any
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

class PostgresConnector:
    """Connects to PostgreSQL and retrieves data."""
    
    def __init__(self, connection_string: str):
        """Initialize with PostgreSQL connection string."""
        self.connection_string = connection_string
    
    def connect(self):
        """Establish connection to PostgreSQL."""
        return psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries."""
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
    
    def get_tables(self) -> List[str]:
        """Get a list of all tables in the database."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        results = self.execute_query(query)
        return [row['table_name'] for row in results]
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table."""
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = %s
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_data(self, table_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get data from a specific table with optional limit."""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)


class EmbeddingService:
    """Handles text embeddings using Google's text embeddings."""
    
    def __init__(self, api_key: str):
        """Initialize with Google API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Use text-embedding model
        self.model = "models/embedding-001"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query" if len(texts) == 1 else "retrieval_document"
            )
            embeddings.append(result["embedding"])
            
        return embeddings


class ChromaVectorStore:
    """Manages the Chroma vector database for document storage and retrieval."""
    
    def __init__(self, 
                 google_api_key: str, 
                 collection_name: str = "rag_documents", 
                 persist_directory: str = "./chroma_db"):
        """Initialize with collection name and persistence directory."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService(api_key=google_api_key)
        
        # Define custom embedding function class
        class GoogleEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, embedding_service):
                self.embedding_service = embedding_service
                
            def __call__(self, texts):
                return self.embedding_service.get_embeddings(texts)
        
        # Create instance of custom embedding function
        self.embedding_function = GoogleEmbeddingFunction(self.embedding_service)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection - improved error handling
        try:
            # First try to get the existing collection
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: {collection_name}")
        except Exception as e:
            if "does not exist" in str(e).lower():
                # Collection doesn't exist, so create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"Created new collection: {collection_name}")
            else:
                # Re-raise if it's a different exception
                print(f"Error accessing collection: {e}")
                # Try to create anyway as fallback
                try:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"Created new collection as fallback: {collection_name}")
                except Exception as e2:
                    print(f"Failed to create collection: {e2}")
                    raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        if not documents:
            return
            
        # Generate IDs
        import hashlib
        ids = []
        for i, doc in enumerate(documents):
            # Create a more reliable hash-based ID
            doc_hash = hashlib.md5(doc.encode()).hexdigest()[:8]
            ids.append(f"doc_{i}_{doc_hash}")
        
        # Add documents to collection
        if metadatas:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.collection.add(
                documents=documents,
                ids=ids
            )
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store and return the most relevant documents."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def delete_collection(self):
        """Delete the collection to start fresh."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")

class RAGSystem:
    """Main RAG system that combines PostgreSQL data, vector storage, and generation."""
    
    def __init__(
        self, 
        postgres_connection_string: str,
        google_api_key: str,
        model_name: str = "gemini-pro",
        collection_name: str = "rag_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the RAG system with database connections and LLM."""
        # Set API key
        self.google_api_key = google_api_key
        genai.configure(api_key=google_api_key)
        
        # Initialize components
        self.postgres = PostgresConnector(postgres_connection_string)
        self.vector_store = ChromaVectorStore(
            google_api_key=google_api_key,
            collection_name=collection_name
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize LLM with LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # Define RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based only on the provided context. 
            If you don't know the answer based on the context, say "I don't have enough information to answer that question." 
            Do not make up information or hallucinate. Always cite your sources from the context.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        # Build the RAG chain
        self.rag_chain = (
            self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def index_table(self, table_name: str, text_columns: List[str] = None):
        """Index a specific table from PostgreSQL into the vector store."""
        # Get table data
        data = self.postgres.get_table_data(table_name)
        if not data:
            print(f"No data found in table {table_name}")
            return
        
        # Get schema to identify text columns if not specified
        if not text_columns:
            schema = self.postgres.get_table_schema(table_name)
            text_columns = [
                col['column_name'] for col in schema 
                if col['data_type'] in ('text', 'varchar', 'char', 'character varying')
            ]
        
        if not text_columns:
            print(f"No text columns found in table {table_name}")
            return
        
        # Process each row
        documents = []
        metadatas = []
        
        for row in data:
            # Create a document from the text columns
            document_parts = []
            for col in text_columns:
                if col in row and row[col]:
                    document_parts.append(f"{col}: {row[col]}")
            
            if document_parts:
                document = "\n".join(document_parts)
                
                # Create metadata from non-text fields for retrieval context
                metadata = {
                    "table": table_name,
                    "id": str(row.get("id", "")),
                }
                
                # Add metadata fields for filtering
                for key, value in row.items():
                    if key not in text_columns and value is not None:
                        metadata[key] = str(value)
                
                # Split document if it's too large
                chunks = self.text_splitter.split_text(document)
                
                # Add each chunk with the same metadata
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk"] = i
                    metadatas.append(chunk_metadata)
        
        # Add to vector store
        print(f"Indexing {len(documents)} chunks from table {table_name}")
        self.vector_store.add_documents(documents, metadatas)
        
    def index_all_tables(self):
        """Index all tables in the PostgreSQL database."""
        tables = self.postgres.get_tables()
        for table in tables:
            self.index_table(table)
    
    def direct_gemini_query(self, question: str, context: str) -> str:
        """Use Gemini API directly instead of through LangChain."""
        model = genai.GenerativeModel(model_name="gemini-pro")
        
        system_prompt = """You are a helpful assistant that answers questions based only on the provided context. 
        If you don't know the answer based on the context, say "I don't have enough information to answer that question." 
        Do not make up information or hallucinate. Always cite your sources from the context."""
        
        response = model.generate_content(
            [
                {"role": "user", "parts": [system_prompt]},
                {"role": "user", "parts": [f"Context:\n{context}\n\nQuestion: {question}"]}
            ],
            generation_config={"temperature": 0}
        )
        
        return response.text
    
    def query(self, question: str, n_results: int = 5, use_direct_api: bool = False) -> str:
        """Query the RAG system with a natural language question."""
        # Retrieve relevant documents
        results = self.vector_store.query(question, n_results=n_results)
        
        # Extract the documents and prepare context
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant information found in the database to answer this question."
        
        # Prepare context with documents and their metadata
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source_info = f"[Source: Table '{meta.get('table', 'unknown')}'"
            if "id" in meta:
                source_info += f", ID: {meta.get('id')}"
            source_info += "]"
            
            context_parts.append(f"Document {i+1} {source_info}:\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer using either LangChain or direct API
        if use_direct_api:
            answer = self.direct_gemini_query(question, context)
        else:
            answer = self.rag_chain.invoke({
                "context": context,
                "question": question
            })
        
        return answer

    def get_db_schema_overview(self) -> str:
        """Get an overview of the database schema to understand available data."""
        tables = self.postgres.get_tables()
        
        schema_overview = []
        for table in tables:
            columns = self.postgres.get_table_schema(table)
            column_info = [f"{col['column_name']} ({col['data_type']})" for col in columns]
            schema_overview.append(f"Table: {table}\nColumns: {', '.join(column_info)}\n")
        
        return "\n".join(schema_overview)


# Example usage
def main():
    # Environment variables for credentials
    postgres_conn_string = "postgresql://postgres:master@localhost:5432/bse_updates"
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Initialize the RAG system
    rag_system = RAGSystem(
        postgres_connection_string=postgres_conn_string,
        google_api_key=google_api_key,
        model_name="gemini", 
        collection_name="postgres_data"
    )
    
    # Print database schema overview
    print("Database Schema Overview:")
    print(rag_system.get_db_schema_overview())
    
    # Index all tables
    print("Indexing all tables...")
    rag_system.index_all_tables()
    
    # Example query
    question = "What are the top selling products?"
    print(f"\nQuestion: {question}")
    
    # Option to use LangChain or direct Gemini API
    answer = rag_system.query(question, use_direct_api=True)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()