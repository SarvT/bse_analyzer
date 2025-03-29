from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="BSE Stock Updates API",
    description="API for BSE corporate filing updates",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST", "localhost"),
    database=os.getenv("DB_NAME", "bse_updates"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
    cursor_factory=RealDictCursor
)

# Models
class StockUpdate(BaseModel):
    id: int
    stock_code: str
    update_type: str
    title: str
    summary: str
    file_url: Optional[str]
    file_type: Optional[str]
    submitted_date: datetime
    created_at: datetime

    class Config:
        orm_mode = True

class StockCodeInfo(BaseModel):
    stock_code: str
    total_updates: int
    latest_update: datetime

# Database dependency
def get_db():
    """Get a database connection from the pool"""
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BSE Updates API"}

@app.get("/stocks", response_model=List[StockCodeInfo], tags=["Stock Information"])
async def get_stocks(db: psycopg2.extensions.connection = Depends(get_db)):
    """
    Get all stock codes and their update counts
    """
    try:
        with db.cursor() as cur:
            cur.execute("""
                SELECT 
                    stock_code,
                    COUNT(*) as total_updates,
                    MAX(submitted_date) as latest_update
                FROM stock_updates
                GROUP BY stock_code
                ORDER BY stock_code
            """)
            results = cur.fetchall()
            return list(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/updates/{stock_code}", response_model=List[StockUpdate], tags=["Updates"])
async def get_stock_updates(
    stock_code: str,
    update_type: Optional[str] = None,
    days: int = Query(60, ge=1, le=762, description="Number of days of history to return"),
    limit: int = Query(300, ge=1, le=500, description="Maximum number of records to return"),
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get updates for a specific stock code with optional filtering
    """
    try:
        with db.cursor() as cur:
            query = """
                SELECT *
                FROM stock_updates
                WHERE stock_code = %s
                AND submitted_date >= %s
            """
            params = [stock_code, datetime.now() - timedelta(days=days)]
            
            if update_type:
                query += " AND update_type = %s"
                params.append(update_type)
            
            query += " ORDER BY submitted_date DESC LIMIT %s"
            params.append(limit)
            query_params = (stock_code, datetime.now() - timedelta(days=days), limit)
            # print("Executing query:", query % query_params)
            cur.execute(query, params)
            results = cur.fetchall()
            # print("API Query Results:", results)  # Debugging print

            return list(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/updates/{stock_code}/{update_id}", response_model=StockUpdate, tags=["Updates"])
async def get_update_by_id(
    stock_code: str,
    update_id: int,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get a specific update by ID
    """
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT * FROM stock_updates WHERE id = %s AND stock_code = %s",
                (update_id, stock_code)
            )
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Update not found")
            
            return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/update-types", tags=["Updates"])
async def get_update_types(
    stock_code: Optional[str] = None,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get all update types available in the database
    """
    try:
        with db.cursor() as cur:
            query = """
                SELECT DISTINCT update_type
                FROM stock_updates
            """
            params = []
            
            if stock_code:
                query += " WHERE stock_code = %s"
                params.append(stock_code)
            
            query += " ORDER BY update_type"
            
            cur.execute(query, params)
            results = cur.fetchall()
            return [r["update_type"] for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    """Close the connection pool on shutdown"""
    db_pool.closeall()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)