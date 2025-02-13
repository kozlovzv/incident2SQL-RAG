from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dataclasses import dataclass, field
import re
import pickle
from pathlib import Path
import time
from datetime import datetime
from .monitoring import RAGMonitor


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self):
        return bool(self.content)


class RAGRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        chunk_size: int = 512,
        index_path: str = None,
        cache_size: int = 1000,  # Size of LRU cache
        n_clusters: int = 100,  # Number of clusters for IVF index
        force_ivf: bool = False,  # Force using IVF index even for small datasets
    ):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.index_path = Path(index_path) if index_path else None
        self.index = None
        self.monitor = RAGMonitor()
        self.cache = {}  # Simple LRU cache for embeddings
        self.cache_size = cache_size
        self.n_clusters = n_clusters
        self.force_ivf = force_ivf
        if self.index_path:
            # Check if index files exist
            faiss_path, meta_path = self._get_index_paths()
            if faiss_path.exists() and meta_path.exists():
                self.load_index()  # Try to load existing index

    def _get_index_paths(self) -> tuple[Path, Path]:
        """Get paths for index files"""
        if not self.index_path:
            raise ValueError("No index path provided")
        base_path = self.index_path
        return base_path.with_suffix(".faiss"), base_path.with_suffix(".meta")

    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings with caching"""
        try:
            embeddings = []
            cache_hits = 0
            to_encode = []
            text_mapping = {}

            # Check cache first
            for i, text in enumerate(texts):
                text_hash = hash(text)
                if text_hash in self.cache:
                    embeddings.append(self.cache[text_hash])
                    cache_hits += 1
                else:
                    to_encode.append(text)
                    text_mapping[len(to_encode) - 1] = i

            # Encode new texts
            if to_encode:
                new_embeddings = self.model.encode(to_encode)

                # Update cache
                for i, emb in enumerate(new_embeddings):
                    text_hash = hash(to_encode[i])
                    self.cache[text_hash] = emb

                    # Simple cache size control
                    if len(self.cache) > self.cache_size:
                        # Remove oldest entry
                        self.cache.pop(next(iter(self.cache)))

                # Merge cached and new embeddings
                final_embeddings = [None] * len(texts)
                for i, emb in enumerate(embeddings):
                    final_embeddings[i] = emb
                for i, emb in enumerate(new_embeddings):
                    final_embeddings[text_mapping[i]] = emb

            print(f"Cache hits: {cache_hits}/{len(texts)}")
            return np.array(final_embeddings if to_encode else embeddings).astype(
                "float32"
            )
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def _create_index(self, documents: List[Document]) -> bool:
        """Создание индекса FAISS с IVF"""
        if not documents:
            print("No documents provided for indexing")
            return False
        try:
            print(f"Creating index for {len(documents)} documents")
            embeddings = self._get_embeddings([doc.content for doc in documents])
            if embeddings is None or embeddings.size == 0:
                return False

            dimension = embeddings.shape[1]

            # Используем IVF индекс если принудительно включен или датасет большой
            if self.force_ivf or len(documents) >= 50:
                # Create IVF index with clustering
                quantizer = faiss.IndexFlatL2(dimension)
                n_clusters = min(
                    self.n_clusters, len(documents)
                )  # Adjust clusters based on data size
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, n_clusters, faiss.METRIC_L2
                )
                # Train the index
                print("Training IVF index...")
                self.index.train(embeddings)
                # Add vectors to the index
                self.index.add(embeddings)
                # Set number of probes for better recall
                self.index.nprobe = min(20, n_clusters)
                print(f"Successfully created IVF index with {n_clusters} clusters")
            else:
                # Для маленьких наборов данных используем простой IndexFlatL2
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings)
                print("Created simple FlatL2 index for small dataset")

            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def _validate_index(self) -> bool:
        """Validate index state"""
        try:
            if not isinstance(self.index, (faiss.IndexFlatL2, faiss.IndexIVFFlat)):
                print("Not a valid FAISS index")
                return False

            if len(self.documents) == 0:
                print("No documents loaded")
                return False

            if self.index.ntotal != len(self.documents):
                print(
                    f"Index size mismatch: expected {len(self.documents)}, got {self.index.ntotal}"
                )
                return False

            # Проверяем работоспособность индекса
            test_vector = np.zeros((1, self.index.d), dtype="float32")
            try:
                self.index.search(test_vector, 1)
            except Exception as e:
                print(f"Index search test failed: {e}")
                return False

            return True
        except Exception as e:
            print(f"Error validating index: {e}")
            return False

    def optimize_index_params(self) -> None:
        """Optimize index parameters based on dataset characteristics"""
        if not self.documents:
            return

        n_vectors = len(self.documents)

        # Optimize number of clusters (rule of thumb: sqrt(N) for small datasets,
        # 4*sqrt(N) for larger ones)
        if n_vectors < 1000:
            self.n_clusters = max(2, int(np.sqrt(n_vectors)))
        else:
            self.n_clusters = max(2, int(4 * np.sqrt(n_vectors)))

        # Optimize nprobe (number of clusters to search)
        # Balance between speed and recall
        if self.index and isinstance(self.index, faiss.IndexIVFFlat):
            if n_vectors < 1000:
                self.index.nprobe = min(
                    self.n_clusters, 20
                )  # Higher recall for small datasets
            else:
                self.index.nprobe = min(
                    self.n_clusters // 4, 256
                )  # Balance for larger datasets

        # Adjust chunk size based on average document length
        avg_doc_length = np.mean([len(doc.content.split()) for doc in self.documents])
        if avg_doc_length > self.chunk_size * 2:
            # For very long documents, use larger chunks
            self.chunk_size = min(512, int(avg_doc_length / 3))
        elif avg_doc_length < self.chunk_size / 2:
            # For short documents, use smaller chunks
            self.chunk_size = max(64, int(avg_doc_length * 1.5))

    def index_documents(
        self, documents: List[Dict[str, Any]], save: bool = True, optimize: bool = True
    ) -> bool:
        """Index documents with optional parameter optimization"""
        if not documents:
            print("No documents to index")
            return False

        processed_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            # Preprocess content
            content = self._preprocess_text(content)
            if not content:
                continue

            # Split into chunks with overlap
            chunks = self._chunk_text(content)

            # Create document objects with metadata
            for i, chunk in enumerate(chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_length": len(doc.get("content", "")),
                        "chunk_length": len(chunk),
                        "processing_date": datetime.utcnow().isoformat(),
                    }
                )

                doc_obj = Document(content=chunk, metadata=metadata)
                processed_docs.append(doc_obj)

        if processed_docs:
            print(f"Indexing {len(processed_docs)} processed documents")
            self.documents = processed_docs

            # Optimize parameters if requested
            if optimize:
                self.optimize_index_params()

            if self._create_index(processed_docs):
                if save and self.index_path:
                    return self.save_index()
                return True

        return False

    def save_index(self) -> bool:
        """Enhanced index saving with validation"""
        if not self.index_path or not self._validate_index():
            print("Cannot save index: missing required data or invalid index")
            return False
        try:
            print("Saving index to disk...")
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss_path, meta_path = self._get_index_paths()
            temp_faiss = faiss_path.with_suffix(".faiss.tmp")
            temp_meta = meta_path.with_suffix(".meta.tmp")

            # Save FAISS index
            print("Saving FAISS index to temporary file")
            faiss.write_index(self.index, str(temp_faiss))

            # Save metadata with index type information
            print("Saving metadata with versioning")
            meta_data = {
                "version": "2.0",
                "last_updated": datetime.utcnow().isoformat(),
                "index_type": "flat_l2"
                if isinstance(self.index, faiss.IndexFlatL2)
                else "ivf",
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "hash": hash(doc.content),
                    }
                    for doc in self.documents
                ],
                "config": {
                    "similarity_threshold": self.similarity_threshold,
                    "chunk_size": self.chunk_size,
                    "n_clusters": self.n_clusters,
                    "force_ivf": self.force_ivf,
                    "model_name": self.model.__class__.__name__,
                },
                "stats": {
                    "total_documents": len(self.documents),
                    "index_size": temp_faiss.stat().st_size,
                    "dimension": self.index.d,
                    "is_trained": getattr(
                        self.index, "is_trained", True
                    ),  # True for FlatL2
                    "ntotal": self.index.ntotal,
                },
            }

            with open(temp_meta, "wb") as f:
                pickle.dump(meta_data, f)

            # Atomic rename
            temp_faiss.rename(faiss_path)
            temp_meta.rename(meta_path)
            print("Successfully saved index and metadata")
            return True

        except Exception as e:
            print(f"Error saving index: {e}")
            # Cleanup temporary files
            for temp_file in [temp_faiss, temp_meta]:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as cleanup_error:
                    print(f"Error cleaning up temporary file: {cleanup_error}")
            return False

    def load_index(self) -> bool:
        """Enhanced index loading with better validation and fallback"""
        if not self.index_path:
            print("No index path provided")
            return False
        try:
            faiss_path, meta_path = self._get_index_paths()
            if not faiss_path.exists() or not meta_path.exists():
                print("Missing required index files")
                return False

            # Load metadata first
            print(f"Loading metadata from {meta_path}")
            with open(meta_path, "rb") as f:
                meta_data = pickle.load(f)

            # Version compatibility check with warning
            version = meta_data.get("version", "1.0")
            if version != "2.0":
                print(f"Warning: Loading index with version {version}")

            # Load configuration
            config = meta_data.get("config", {})
            self.similarity_threshold = config.get(
                "similarity_threshold", self.similarity_threshold
            )
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.n_clusters = config.get("n_clusters", self.n_clusters)
            self.force_ivf = config.get("force_ivf", self.force_ivf)

            # Load documents
            docs_data = meta_data.get("documents", [])
            if not docs_data:
                print("No documents found in metadata")
                return False

            # Convert document data to Document objects
            self.documents = [
                Document(content=doc["content"], metadata=doc.get("metadata", {}))
                for doc in docs_data
            ]

            # Try to load existing index
            try:
                print(f"Loading FAISS index from {faiss_path}")
                self.index = faiss.read_index(str(faiss_path))

                # Validate loaded index
                if self._validate_index():
                    print(
                        f"Successfully loaded index with {len(self.documents)} documents"
                    )
                    return True
                else:
                    print("Loaded index validation failed, recreating index...")
            except Exception as e:
                print(f"Error loading index, recreating: {e}")

            # Fallback: recreate index from documents
            print("Recreating index from saved documents")
            if self._create_index(self.documents):
                print("Successfully recreated index")
                return True

            print("Failed to recreate index")
            self.index = None
            self.documents = []
            return False

        except Exception as e:
            print(f"Error loading index: {e}")
            self.documents = []
            self.index = None
            return False

    def update_index(self, new_documents: List[Dict[str, Any]]) -> bool:
        """Incrementally update the index with new documents"""
        if not new_documents:
            return True  # No updates needed

        try:
            # Process new documents
            print(f"Processing {len(new_documents)} new documents")
            new_processed_docs = []

            for doc in new_documents:
                content = doc.get("content", "")
                if not content:
                    continue

                # Calculate document hash
                doc_hash = hash(content)

                # Skip if document already exists
                if any(
                    hash(existing.content) == doc_hash for existing in self.documents
                ):
                    continue

                # Process new document
                processed_content = self._preprocess_text(content)
                chunks = self._chunk_text(processed_content)

                for chunk in chunks:
                    metadata = doc.get("metadata", {}).copy()
                    metadata["added_date"] = datetime.utcnow().isoformat()
                    new_processed_docs.append(
                        Document(content=chunk, metadata=metadata)
                    )

            if not new_processed_docs:
                print("No new unique documents to add")
                return True

            # Get embeddings for new documents
            print(f"Getting embeddings for {len(new_processed_docs)} new documents")
            new_embeddings = self._get_embeddings(
                [doc.content for doc in new_processed_docs]
            )

            if new_embeddings is None:
                return False

            # Add to index
            print("Adding new vectors to index")
            self.index.add(new_embeddings)

            # Update documents list
            self.documents.extend(new_processed_docs)

            # Save updated index
            print("Saving updated index")
            return self.save_index()

        except Exception as e:
            print(f"Error updating index: {e}")
            return False

    def retrieve(
        self, query: str, k: int = 3, return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with dynamic k based on similarity"""
        start_time = time.time()

        if not self._validate_index() or not self.documents:
            print("No valid index available")
            return []

        query = self._preprocess_text(query)
        if not query:
            print("Empty query")
            return []

        try:
            print(f"Processing query: {query}")
            query_embedding = self._get_embeddings([query])
            if query_embedding is None:
                return []

            # Search with larger k initially
            initial_k = min(k * 3, len(self.documents))
            distances, indices = self.index.search(query_embedding, initial_k)

            results = []
            best_similarity = None
            dynamic_threshold = self.similarity_threshold

            for score, idx in zip(distances[0], indices[0]):
                if idx >= len(self.documents) or idx < 0:
                    continue

                similarity = 1 / (1 + score)  # Convert distance to similarity score

                # Update dynamic threshold based on best match
                if best_similarity is None:
                    best_similarity = similarity
                    dynamic_threshold = max(
                        self.similarity_threshold,
                        similarity
                        * 0.6,  # More strict threshold relative to best match
                    )

                if similarity >= dynamic_threshold:
                    doc = self.documents[idx]
                    result = {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": float(similarity) if return_scores else None,
                    }
                    results.append(result)
                elif results:  # Stop if we've found some results but similarity dropped
                    break

            final_results = results[:k]
            duration = time.time() - start_time

            if final_results:
                self.monitor.log_query(
                    query=query,
                    results=final_results,
                    duration=duration,
                    cache_hits=query_embedding is not None,
                    total_candidates=initial_k,
                )

            return final_results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text:
            return ""

        # Basic cleaning
        text = text.strip().lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep important punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        # Normalize dates (convert various formats to YYYY-MM-DD)
        date_patterns = [
            (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", r"\3-\2-\1"),  # DD/MM/YYYY
            (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", r"\1-\2-\3"),  # YYYY/MM/DD
        ]
        for pattern, replacement in date_patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Enhanced text chunking with overlap and semantic boundaries"""
        if not text:
            return []

        # Configure chunking parameters
        overlap = min(100, self.chunk_size // 4)  # 25% overlap

        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)

            # If single sentence is longer than chunk_size, split it
            if sentence_length > self.chunk_size:
                for i in range(0, sentence_length, self.chunk_size - overlap):
                    chunk = " ".join(sentence_words[i : i + self.chunk_size])
                    if chunk:
                        chunks.append(chunk)
                continue

            # Check if adding the sentence would exceed chunk_size
            if current_length + sentence_length > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap from previous chunk if possible
                overlap_words = current_chunk[-overlap:] if current_chunk else []
                current_chunk = overlap_words + sentence_words
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk.extend(sentence_words)
                current_length += sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return self.monitor.get_stats()
