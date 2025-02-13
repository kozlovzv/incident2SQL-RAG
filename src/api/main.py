from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from pathlib import Path
from src.rag.retriever import RAGRetriever
from src.models.text2sql import Text2SQLGenerator
from src.models.database import init_db
import logging
import time

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
    """Response model with improved readability"""

    query: str
    sql: str
    results: List[Dict[str, Any]]
    execution_time: float
    total_results: int
    context: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


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


@app.post("/query")
async def process_query(query: Query, db: Session = Depends(get_db)):
    start_time = time.time()

    try:
        logger.info(f"Processing query: {query.text}")

        if not hasattr(app.state, "components_ready") or not app.state.components_ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Service components not fully initialized",
                    "query": query.text,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )

        if not text2sql or not retriever:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Required components not initialized",
                    "query": query.text,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )

        # Get relevant context
        context = retriever.retrieve(query.text) if retriever else []
        logger.info(f"Retrieved {len(context)} context items")

        # Generate SQL with error handling
        try:
            sql = text2sql.generate_sql(query.text, context)
            logger.info(f"Generated SQL: {sql}")
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": str(e),
                    "query": query.text,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )

        # Execute query with error handling
        try:
            result = db.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            logger.info(f"Query returned {len(rows)} results")

            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "query": query.text,
                        "sql": sql,
                        "results": rows,
                        "context": context,
                    },
                    "metadata": {
                        "total_results": len(rows),
                        "execution_time": round(time.time() - start_time, 3),
                    },
                }
            )

        except SQLAlchemyError as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Database query execution failed",
                    "query": query.text,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "query": query.text,
                "execution_time": round(time.time() - start_time, 3),
            },
        )


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
