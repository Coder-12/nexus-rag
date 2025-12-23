import logging
from pathlib import Path
from typing import List

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
        logger.info("Starting ingestion pipeline")

        # -----------------------------
        # Phase A: Load documents
        # -----------------------------
        documents: List[Document] = load_documents(self.data_dir)
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