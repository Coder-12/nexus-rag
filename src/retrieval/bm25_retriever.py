"""
BM25 Sparse Retrieval for Nexus-RAG
Keyword-based retrieval to complement vector search.

Features:
- Custom technical tokenizer (handles BERT, GPT, RLHF, etc.)
- Disk caching for fast startup (0.5s load time)
- Eager initialization during system startup
- In-memory index for fast queries (<50ms)
"""

from typing import List, Tuple, Dict
import re
import pickle
from pathlib import Path
import logging

from rank_bm25 import BM25Okapi
import numpy as np

from src.retrieval.chunk import Chunk

logger = logging.getLogger(__name__)


class TechnicalTokenizer:
    """
    Custom tokenizer for ML/AI technical corpus.
    Preserves acronyms, hyphenated terms, and technical terminology.
    """
    
    # Technical terms that should stay intact
    PRESERVE_TERMS = {
        # Architecture
        "encoder-decoder", "self-attention", "cross-attention",
        "multi-head", "feed-forward", "layer-norm",
        
        # Training
        "fine-tuning", "pre-training", "meta-learning",
        "in-context", "zero-shot", "few-shot", "one-shot",
        
        # Acronyms (detected automatically if all caps)
        # BERT, GPT, RLHF, etc.
    }
    
    # Patterns for technical term detection
    ACRONYM_PATTERN = re.compile(r'\b[A-Z]{2,}\b')  # 2+ uppercase letters
    HYPHENATED_PATTERN = re.compile(r'\b\w+(?:-\w+)+\b')  # word-word-word
    ALPHANUM_PATTERN = re.compile(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text while preserving technical terms.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens (lowercased, technical terms preserved)
        """
        # Lowercase entire text first
        text = text.lower()
        
        tokens = []
        
        # Find all alphanumeric sequences (including hyphens)
        matches = self.ALPHANUM_PATTERN.finditer(text)
        
        for match in matches:
            token = match.group(0)
            
            # Skip very short tokens (noise)
            if len(token) < 2:
                continue
                
            # Preserve hyphenated terms
            if '-' in token and token in self.PRESERVE_TERMS:
                tokens.append(token)
            
            # Preserve common hyphenated patterns
            elif '-' in token and len(token.split('-')) <= 3:
                # Keep short hyphenated terms (encoder-decoder, pre-training)
                tokens.append(token)
            
            # Regular token
            else:
                # Remove remaining hyphens and split
                parts = token.replace('-', ' ').split()
                tokens.extend(parts)
        
        return tokens
    
    def __call__(self, text: str) -> List[str]:
        """Allow tokenizer to be called as a function"""
        return self.tokenize(text)


class BM25Index:
    """
    In-memory BM25 index for Nexus-RAG corpus.
    Built once, cached to disk, reused for all queries.
    """
    
    def __init__(self, chunks: List[Chunk], tokenizer: TechnicalTokenizer = None):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: All chunks from ingestion pipeline
            tokenizer: Custom tokenizer (optional, uses default if None)
        """
        self.chunks = chunks
        self.chunk_ids = [
            c.chunk_id
            for c in chunks
        ]
        print("BM25 ID SAMPLE:", self.chunk_ids[:5])
        
        self.tokenizer = tokenizer or TechnicalTokenizer()
        
        # Tokenize entire corpus
        logger.info(f"Tokenizing {len(chunks)} chunks for BM25 index...")
        self.tokenized_corpus = [
            self.tokenizer.tokenize(c.text) for c in chunks
        ]
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"✅ BM25 index built: {len(chunks)} chunks")
    
    def save(self, cache_path: Path):
        """
        Save BM25 index to disk for fast loading.
        
        Args:
            cache_path: Path to save cache file (.pkl)
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "chunk_ids": self.chunk_ids,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25": self.bm25,
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"✅ BM25 index cached to {cache_path}")
    
    @classmethod
    def load(cls, cache_path: Path, tokenizer: TechnicalTokenizer = None):
        """
        Load BM25 index from disk cache.
        
        Args:
            cache_path: Path to cache file (.pkl)
            tokenizer: Custom tokenizer (optional)
            
        Returns:
            BM25Index instance loaded from cache
        """
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        
        # Create instance without rebuilding
        instance = cls.__new__(cls)
        instance.chunk_ids = cache_data["chunk_ids"]
        instance.tokenized_corpus = cache_data["tokenized_corpus"]
        instance.bm25 = cache_data["bm25"]
        instance.tokenizer = tokenizer or TechnicalTokenizer()
        instance.chunks = None  # Not stored in cache (IDs only)
        
        logger.info(f"✅ BM25 index loaded from cache: {len(instance.chunk_ids)} chunks")
        
        return instance


class BM25Retriever:
    """
    BM25 retrieval interface for Nexus-RAG.
    Returns chunk_ids + scores (metadata fetched separately from Pinecone).
    """
    
    def __init__(self, bm25_index: BM25Index):
        """
        Initialize retriever with BM25 index.
        
        Args:
            bm25_index: Pre-built BM25 index
        """
        self.index = bm25_index
    
    def search(
        self, 
        query: str, 
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        # Tokenize query
        query_tokens = self.index.tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning(f"Empty query after tokenization: {query}")
            return []
        
        # BM25 scoring
        scores = self.index.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = [
            (self.index.chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # Filter out zero scores
        ]
        
        return results
    
    def search_with_threshold(
        self,
        query: str,
        top_k: int = 100,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search with minimum score threshold.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of (chunk_id, bm25_score) tuples above threshold
        """
        results = self.search(query, top_k)
        return [(cid, score) for cid, score in results if score >= min_score]


def build_bm25_index_from_chunks(chunks: List[Chunk]) -> BM25Index:
    """
    Convenience function to build BM25 index from chunks.
    
    Args:
        chunks: List of Chunk objects from ingestion
        
    Returns:
        Built BM25Index
    """
    tokenizer = TechnicalTokenizer()
    return BM25Index(chunks, tokenizer)


def initialize_bm25_system(
    cache_path: Path,
    chunks: List[Chunk] = None,
    force_rebuild: bool = False
) -> BM25Retriever:
    """
    Initialize BM25 system with eager loading and caching.
    
    This is the main entry point for BM25 retrieval.
    
    Args:
        cache_path: Path to BM25 cache file (.pkl)
        chunks: Chunks to build index from (if rebuilding)
        force_rebuild: Force rebuild even if cache exists
        
    Returns:
        Ready-to-use BM25Retriever
        
    Usage:
        # During system startup
        bm25_retriever = initialize_bm25_system(
            cache_path=Path("cache/bm25_index.pkl"),
            chunks=all_chunks  # from ingestion
        )
        
        # Query
        results = bm25_retriever.search("BERT GPT difference", top_k=100)
    """
    # Check if cache exists and we're not forcing rebuild
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading BM25 index from cache...")
        bm25_index = BM25Index.load(cache_path)
    else:
        # Build new index
        if chunks is None:
            raise ValueError(
                "chunks required to build BM25 index (cache not found or force_rebuild=True)"
            )
        
        logger.info("Building new BM25 index...")
        bm25_index = build_bm25_index_from_chunks(chunks)
        
        # Cache for future use
        bm25_index.save(cache_path)
    
    # Create retriever
    retriever = BM25Retriever(bm25_index)
    
    logger.info("✅ BM25 system initialized and ready")
    
    return retriever