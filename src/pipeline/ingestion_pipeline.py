import logging
from pathlib import Path
from typing import Iterable, List

from src.retrieval.document import Document
from src.retrieval.chunking import SectionAwareChunker
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.vector_store import PineconeVectorStore
from src.data.document_loader import load_documents  # Phase A parser


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IngestionPipeline:
    """
    Orchestrates Nexus-RAG ingestion:
    Document → Chunk → Embed → Vector Store
    """

    def __init__(
        self,
        data_dir: Path,
        index_name: str,
        namespace: str,
    ):
        self.data_dir = data_dir

        self.chunker = SectionAwareChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            namespace=namespace,
        )

    def ingest(self):
        return self.ingest_documents()

    def ingest_documents(self, doc_ids: Iterable[str] | None = None):
        logger.info("Starting ingestion pipeline")

        # -----------------------------
        # Phase A: Load documents
        # -----------------------------
        documents: List[Document] = load_documents(self.data_dir)
        requested_doc_ids = set(doc_ids or [])
        if requested_doc_ids:
            documents = [
                doc for doc in documents
                if doc.doc_id in requested_doc_ids or f"{doc.doc_id}.txt" in requested_doc_ids
            ]
            missing = requested_doc_ids - {doc.doc_id for doc in documents} - {
                f"{doc.doc_id}.txt" for doc in documents
            }
            if missing:
                raise ValueError(f"Requested documents not found: {sorted(missing)}")

        logger.info(f"Loaded {len(documents)} documents")

        total_chunks = 0

        for doc in documents:
            logger.info(f"Ingesting document: {doc.doc_id}")

            # Safe re-ingestion
            self.vector_store.delete_by_doc_id(doc.doc_id)

            # -----------------------------
            # Phase B: Chunking
            # -----------------------------
            chunks = self.chunker.chunk_document(doc)
            logger.info(f"  Produced {len(chunks)} chunks")
            total_chunks += len(chunks)

            # -----------------------------
            # Phase C: Embedding
            # -----------------------------
            embeddings = self.embedder.embed_chunks(chunks)

            # -----------------------------
            # Phase D: Vector Store
            # -----------------------------
            self.vector_store.upsert(embeddings)

        logger.info("Ingestion completed successfully")
        logger.info(f"Total chunks ingested: {total_chunks}")
