# src/app_factory.py

import os
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.intelligent_router import initialize_routing_system
from src.generation.answer_synthesizer import AnswerSynthesizer
from src.generation.trust_formatter import TrustFormatter
from src.pipeline.rag_pipeline import RAGPipeline


def build_pipeline() -> RAGPipeline:
    """
    Build the public Nexus RAG pipeline.

    This keeps the public app using the same production RAG path:
    router -> retrieval/rerank -> synthesizer -> trust formatter.
    """
    load_dotenv()

    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    if not pinecone_index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME")

    pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "tier1_v1")
    cohere_api_key = os.getenv("COHERE_API_KEY", "")
    bm25_cache_path = Path(os.getenv("BM25_CACHE_PATH", "cache/bm25_index.pkl"))

    router = initialize_routing_system(
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        cohere_api_key=cohere_api_key,
        bm25_cache_path=bm25_cache_path,
    )

    synthesizer = AnswerSynthesizer(
        model=os.getenv("NEXUS_LLM_MODEL", "gpt-4o-mini"),
        verifier_model=os.getenv("NEXUS_VERIFIER_MODEL", "gpt-4o-mini"),
        max_context_tokens=int(os.getenv("NEXUS_MAX_CONTEXT_TOKENS", "3000")),
        max_context_chunks=int(os.getenv("NEXUS_MAX_CONTEXT_CHUNKS", "6")),
        max_chunks_per_doc=int(os.getenv("NEXUS_MAX_CHUNKS_PER_DOC", "2")),
        enable_reflexion=os.getenv("NEXUS_ENABLE_REFLEXION", "true").lower() == "true",
    )

    trust_formatter = TrustFormatter()

    return RAGPipeline(
        router=router,
        synthesizer=synthesizer,
        trust_formatter=trust_formatter,
    )
