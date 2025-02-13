import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from src.rag.retriever import RAGRetriever
import faiss


@pytest.fixture
def sample_documents():
    return [
        {
            "content": "Network outage in data center",
            "metadata": {"category": "network"},
        },
        {
            "content": "Server crash during peak hours",
            "metadata": {"category": "server"},
        },
        {
            "content": "Database connectivity issues",
            "metadata": {"category": "database"},
        },
    ]


def test_basic_retrieval(sample_documents):
    retriever = RAGRetriever(similarity_threshold=0.3)  # Используем более низкий порог
    retriever.index_documents(sample_documents)

    results = retriever.retrieve("network problems", k=2)
    assert len(results) > 0, "Should return at least one result"
    assert all("score" in doc for doc in results), "Each result should have a score"
    assert all("metadata" in doc for doc in results), "Each result should have metadata"


def test_persistence(sample_documents):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # Create and save index
        retriever1 = RAGRetriever(index_path=str(index_path))
        retriever1.index_documents(sample_documents)
        assert retriever1.save_index(), "Failed to save index"

        # Verify files exist
        assert (index_path.with_suffix(".faiss")).exists(), (
            "FAISS index file should exist"
        )
        assert (index_path.with_suffix(".meta")).exists(), "Metadata file should exist"

        # Load index in new instance
        retriever2 = RAGRetriever(index_path=str(index_path))
        results = retriever2.retrieve("network", k=1)
        assert len(results) > 0, "Should return results after loading"
        assert results[0]["metadata"]["category"] == "network", (
            "Should preserve metadata"
        )


def test_monitoring():
    retriever = RAGRetriever()
    docs = [{"content": f"Document {i}", "metadata": {"id": i}} for i in range(3)]
    retriever.index_documents(docs)

    # Make some queries
    retriever.retrieve("document 1")
    retriever.retrieve("document 2")

    stats = retriever.get_performance_stats()
    assert "avg_latency" in stats, "Should track latency"
    assert "total_queries" in stats, "Should track query count"
    assert stats["total_queries"] == 2, "Should count correct number of queries"


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


def test_document_preprocessing(sample_documents):
    retriever = RAGRetriever(similarity_threshold=0.3)
    retriever.index_documents(sample_documents)

    # Test case insensitive matching
    results1 = retriever.retrieve("NETWORK", k=1)
    results2 = retriever.retrieve("network", k=1)
    assert len(results1) == len(results2), "Case should not affect results"

    # Test whitespace normalization
    results3 = retriever.retrieve("network    outage", k=1)
    assert len(results3) > 0, "Should handle extra whitespace"


def test_long_document_chunking():
    long_text = " ".join(["test content"] * 300)  # Create long document
    retriever = RAGRetriever(chunk_size=50)
    retriever.index_documents([{"content": long_text}])

    results = retriever.retrieve("test", k=2)
    assert len(results) > 0


def test_empty_documents():
    retriever = RAGRetriever()

    # Test with empty list
    retriever.index_documents([])
    results = retriever.retrieve("test")
    assert len(results) == 0, "Empty index should return no results"

    # Test with invalid documents
    retriever.index_documents([{"content": ""}, {"content": None}])
    results = retriever.retrieve("test")
    assert len(results) == 0, "Invalid documents should be filtered out"


def test_index_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"
        test_docs = [
            {"content": "Test document 1", "metadata": {"id": 1}},
            {"content": "Test document 2", "metadata": {"id": 2}},
        ]

        # Create and verify first instance
        retriever1 = RAGRetriever(index_path=str(index_path))
        retriever1.index_documents(test_docs)
        results1 = retriever1.retrieve("test document", k=1)
        assert len(results1) > 0, "Should return results before saving"

        success = retriever1.save_index()
        assert success, "Should successfully save index"

        # Verify files exist
        assert (index_path.with_suffix(".faiss")).exists(), (
            "FAISS index file should exist"
        )
        assert (index_path.with_suffix(".meta")).exists(), "Metadata file should exist"

        # Create and verify second instance
        retriever2 = RAGRetriever(index_path=str(index_path))
        results2 = retriever2.retrieve("test document", k=1)
        assert len(results2) > 0, "Should return results after loading"
        assert results2[0]["content"] == results1[0]["content"], "Results should match"


def test_ivf_search_quality():
    # Test that IVF search maintains quality
    docs = [
        {"content": f"Test document {i}", "metadata": {"id": i}} for i in range(100)
    ]

    retriever = RAGRetriever(similarity_threshold=0.5)
    retriever.index_documents(docs)

    results = retriever.retrieve("test document 1", k=5)
    assert len(results) > 0
    assert all(doc["score"] >= 0.5 for doc in results)


def test_ivf_index_creation(sample_documents):
    # Используем force_ivf=True для принудительного создания IVF индекса
    retriever = RAGRetriever(
        n_clusters=2, force_ivf=True
    )  # Small number of clusters for test
    retriever.index_documents(sample_documents)

    assert isinstance(retriever.index, faiss.IndexIVFFlat)
    # Check if index is properly initialized
    assert retriever.index.is_trained
    assert retriever.index.ntotal == len(sample_documents)


def test_embedding_cache():
    retriever = RAGRetriever(cache_size=2)
    test_texts = ["test1", "test2", "test3"]

    # First call - no cache hits
    emb1 = retriever._get_embeddings(test_texts[:2])
    # Second call - should hit cache
    emb2 = retriever._get_embeddings(test_texts[:2])

    assert len(retriever.cache) == 2
    assert np.array_equal(emb1, emb2)


def test_dynamic_similarity_threshold(sample_documents):
    retriever = RAGRetriever(similarity_threshold=0.3)
    retriever.index_documents(sample_documents)

    # Test with very similar query
    results1 = retriever.retrieve("network outage", k=2)
    # Test with less similar query
    results2 = retriever.retrieve("random query", k=2)

    assert len(results1) > 0
    assert results1[0]["score"] > retriever.similarity_threshold
    # Less similar query might return fewer or no results
    assert len(results2) <= len(results1)


def test_document_chunking():
    long_doc = {
        "content": " ".join(["test content"] * 100),
        "metadata": {"source": "test"},
    }

    retriever = RAGRetriever(chunk_size=50)
    retriever.index_documents([long_doc])

    # Check if document was split into chunks
    chunk_sizes = [len(doc.content.split()) for doc in retriever.documents]
    assert all(size <= retriever.chunk_size for size in chunk_sizes)
    # Check if metadata was preserved
    assert all(doc.metadata["source"] == "test" for doc in retriever.documents)


def test_incremental_updates():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"
        retriever = RAGRetriever(index_path=str(index_path))

        # Initial documents
        initial_docs = [{"content": f"doc{i}"} for i in range(3)]
        assert retriever.index_documents(initial_docs)

        # Add new documents
        new_docs = [{"content": f"new_doc{i}"} for i in range(2)]
        assert retriever.update_index(new_docs)

        # Verify total document count
        assert len(retriever.documents) == 5

        # Try adding duplicate document
        assert retriever.update_index([{"content": "doc0"}])
        assert len(retriever.documents) == 5  # Should not increase


def test_monitoring_metrics():
    retriever = RAGRetriever()
    docs = [{"content": f"Document {i}"} for i in range(3)]
    retriever.index_documents(docs)

    # Make multiple queries
    for _ in range(3):
        retriever.retrieve("test query")

    stats = retriever.monitor.get_stats()

    assert "avg_latency" in stats
    assert "latency_details" in stats
    assert "cache_performance" in stats
    assert "error_rate" in stats
    assert stats["total_queries"] == 3
    assert "hourly_trends" in stats

    # Validate latency details structure
    latency_details = stats["latency_details"]
    assert "avg_ms" in latency_details
    assert "median_ms" in latency_details
    assert "std_dev_ms" in latency_details
    assert "percentiles" in latency_details


def test_version_compatibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"
        retriever = RAGRetriever(index_path=str(index_path))

        # Save index with current version
        docs = [{"content": "test"}]
        retriever.index_documents(docs)

        # Modify version in metadata to simulate old version
        meta_path = index_path.with_suffix(".meta")
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
        meta_data["version"] = "1.0"
        with open(meta_path, "wb") as f:
            pickle.dump(meta_data, f)

        # Load should work with warning
        new_retriever = RAGRetriever(index_path=str(index_path))
        assert new_retriever.load_index()
        assert len(new_retriever.documents) == 1


def test_index_optimization():
    # Test with small dataset
    small_docs = [{"content": f"Short document {i}"} for i in range(10)]
    retriever_small = RAGRetriever()
    retriever_small.index_documents(small_docs, optimize=True)

    # Verify parameters for small dataset
    assert retriever_small.n_clusters <= 4  # Should be around sqrt(10)
    assert retriever_small.chunk_size < 512  # Should adjust for short documents

    # Test with larger dataset
    large_docs = [
        {"content": " ".join(["Long document"] * 100) + str(i)} for i in range(1000)
    ]
    retriever_large = RAGRetriever()
    retriever_large.index_documents(large_docs, optimize=True)

    # Verify parameters for large dataset
    assert retriever_large.n_clusters > 100  # Should be around 4*sqrt(1000)
    assert retriever_large.chunk_size > 64  # Should adjust for longer documents
    assert isinstance(retriever_large.index, faiss.IndexIVFFlat)
    assert retriever_large.index.nprobe <= 256  # Should be optimized for speed
