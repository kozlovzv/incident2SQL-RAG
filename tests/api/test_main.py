import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.api.main import app
from src.models.database import Base
from src.models.init_data import init_sample_data
from src.models.text2sql import Text2SQLGenerator
from src.rag.retriever import RAGRetriever

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


# Test fixtures
@pytest.fixture(scope="function")
def engine():
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(engine):
    # Initialize test database with sample data
    init_sample_data(engine.url)

    # Initialize components
    app.state.components_ready = True
    app.state.text2sql = Text2SQLGenerator()
    app.state.retriever = RAGRetriever()

    # Ensure we have test data and index it
    with Session(engine) as session:
        incidents = session.execute(
            text("SELECT description FROM incidents")
        ).fetchall()
        documents = [inc[0] for inc in incidents]
        if not documents:
            # Insert at least one test document with required fields
            session.execute(
                text(
                    "INSERT INTO incidents (title, description) VALUES ('Test Incident', 'Test network incident')"
                )
            )
            session.commit()
            documents = ["Test network incident"]

        app.state.retriever.index_documents(documents)

    with TestClient(app) as client:
        yield client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_endpoint(client):
    response = client.post("/query", json={"text": "Show all incidents"})
    assert response.status_code == 200
    assert "results" in response.json()
