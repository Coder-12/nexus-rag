"""
Build BM25 Index - Nexus RAG
Builds BM25 index from cached chunks and saves for eager loading.

Usage:
    python scripts/build_bm25_index.py
    
This should be run:
1. After initial ingestion (when chunks are created)
2. Whenever corpus is updated (re-ingestion)
3. To rebuild cache if needed
"""

import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.document_loader import load_documents
from src.retrieval.chunking import SectionAwareChunker
from src.retrieval.bm25_retriever import initialize_bm25_system

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Build BM25 index from documents and cache to disk.
    """
    logger.info("=" * 80)
    logger.info("Nexus RAG - BM25 Index Builder")
    logger.info("=" * 80)
    
    # Configuration
    data_dir = PROJECT_ROOT / "data" / "raw"
    cache_dir = PROJECT_ROOT / "cache"
    cache_path = cache_dir / "bm25_index.pkl"
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Cache path: {cache_path}")
    
    # -----------------------------
    # Step 1: Load documents
    # -----------------------------
    logger.info("\n[1/3] Loading documents...")
    documents = load_documents(data_dir)
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    # -----------------------------
    # Step 2: Chunk documents
    # -----------------------------
    logger.info("\n[2/3] Chunking documents...")
    chunker = SectionAwareChunker()
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        logger.info(f"  {doc.doc_id}: {len(chunks)} chunks")
    
    logger.info(f"✅ Total chunks: {len(all_chunks)}")
    
    # -----------------------------
    # Step 3: Build and cache BM25 index
    # -----------------------------
    logger.info("\n[3/3] Building BM25 index...")
    
    # Force rebuild to create new cache
    bm25_retriever = initialize_bm25_system(
        cache_path=cache_path,
        chunks=all_chunks,
        force_rebuild=True
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BM25 INDEX BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Chunks indexed: {len(all_chunks)}")
    logger.info(f"Cache saved to: {cache_path}")
    logger.info(f"Cache size: {cache_path.stat().st_size / 1024:.2f} KB")
    logger.info("\nNext: Run tests with `python tests/test_bm25_retrieval.py`")


if __name__ == "__main__":
    main()