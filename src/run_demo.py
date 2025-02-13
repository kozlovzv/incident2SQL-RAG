from src.models.init_data import init_sample_data
from src.rag.retriever import RAGRetriever
from src.models.text2sql import Text2SQLGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logging.basicConfig(level=logging.INFO)


def run_demo():
    # Initialize database with sample data
    db_url = "sqlite:///data/incidents.db"
    init_sample_data(db_url)

    # Initialize components
    retriever = RAGRetriever()
    text2sql = Text2SQLGenerator()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    # Index sample documents
    with Session() as session:
        incidents = session.execute(
            text("SELECT description FROM incidents")
        ).fetchall()
        documents = [inc[0] for inc in incidents]
        retriever.index_documents(documents)

    # Test queries
    test_queries = [
        "Show all high severity incidents",
        "Find network related incidents from last week",
        "What are the unresolved security incidents?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        # Get relevant context
        context = retriever.retrieve(query)
        print(f"Retrieved Context: {context[0]['content'] if context else 'None'}")

        # Generate SQL
        sql = text2sql.generate_sql(query, context)
        print(f"Generated SQL: {sql}")

        # Execute query
        with Session() as session:
            try:
                results = session.execute(text(sql)).fetchall()
                print(f"Results: {results}")
            except Exception as e:
                print(f"Error executing SQL: {e}")


if __name__ == "__main__":
    run_demo()
