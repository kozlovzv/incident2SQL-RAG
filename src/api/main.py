from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import os
from pathlib import Path
from src.rag.retriever import RAGRetriever
from src.models.text2sql import Text2SQLGenerator
from src.models.database import init_db
from src.models.init_data import init_sample_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Query(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Show me all high severity network incidents from last week"}
            ]
        }
    }


class SQLResponse(BaseModel):
    sql_query: str
    context: Optional[List[Dict[str, Any]]] = None
    results: Optional[List[Dict[str, Any]]] = None


app = FastAPI(
    title="Text2SQL+RAG API",
    description="API for converting natural language queries to SQL using RAG",
    version="1.0.0",
)

# Global components
BASE_DIR = Path(__file__).resolve().parent.parent.parent
db_url = f"sqlite:///{BASE_DIR}/data/incidents.db"
engine = None
SessionLocal = None
retriever = None
text2sql = None


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global engine, SessionLocal, retriever, text2sql
    try:
        # Initialize database
        logger.info("Initializing database...")
        engine = init_db(db_url)
        SessionLocal = sessionmaker(bind=engine)

        # Initialize Text2SQL
        logger.info("Initializing Text2SQL model...")
        text2sql = Text2SQLGenerator()

        # Initialize RAG with incident descriptions
        logger.info("Initializing RAG retriever...")
        retriever = RAGRetriever()

        # Mark components as ready
        app.state.components_ready = True

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        app.state.components_ready = False
        raise e


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post(
    "/query",
    response_model=SQLResponse,
    description="Convert natural language query to SQL and execute it",
)
async def process_query(query: Query, db: Session = Depends(get_db)):
    """
    Convert a natural language query to SQL, retrieve relevant context, and execute the query.
    The system will:
    1. Find relevant incident descriptions to provide context
    2. Generate an SQL query using the context and schema information
    3. Execute the query and return results
    """
    try:
        logger.info(f"Processing query: {query.text}")

        # Validate components are initialized
        if not hasattr(app.state, "components_ready") or not app.state.components_ready:
            raise HTTPException(
                status_code=503, detail="Service components not fully initialized"
            )

        if not text2sql or not retriever:
            raise HTTPException(
                status_code=503, detail="Required components not initialized"
            )

        # Get relevant context
        context = retriever.retrieve(query.text) if retriever else []
        logger.info(f"Retrieved {len(context)} context items")

        # Generate SQL with error handling
        try:
            sql = text2sql.generate_sql(query.text, context)
            logger.info(f"Generated SQL: {sql}")
        except ValueError as e:
            logger.error(f"SQL Generation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        # Execute query with error handling
        try:
            result = db.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            logger.info(f"Query returned {len(rows)} results")
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Database query execution failed"
            )

        return SQLResponse(sql_query=sql, context=context, results=rows)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Check if the service is healthy and all components are initialized"""
    if not hasattr(app.state, "components_ready"):
        app.state.components_ready = False

    return {
        "status": "healthy" if app.state.components_ready else "degraded",
        "components": {
            "database": "ready" if engine else "not ready",
            "text2sql": "ready" if text2sql else "not ready",
            "retriever": "ready" if retriever else "not ready",
        },
    }
