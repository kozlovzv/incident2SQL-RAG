import pytest
import numpy as np
from src.rag.retriever import RAGRetriever


@pytest.fixture
def sample_documents():
    return [
        "High severity network outage incident affecting trading operations",
        "Customer data leak due to misconfigured access controls",
        "Failed software deployment causing system downtime",
    ]


def test_rag_retriever_indexing(sample_documents):
    retriever = RAGRetriever()
    retriever.index_documents(sample_documents)
    assert retriever.index is not None
    assert len(retriever.documents) == len(sample_documents)


def test_rag_retrieval(sample_documents):
    retriever = RAGRetriever()
    retriever.index_documents(sample_documents)

    query = "network incident"
    results = retriever.retrieve(query, k=2)

    assert len(results) <= 2
    assert isinstance(results[0], dict)
    assert "content" in results[0]
    assert "score" in results[0]
    # First result should be the network outage incident
    assert "network" in results[0]["content"].lower()
