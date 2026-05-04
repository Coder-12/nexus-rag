import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.logging_config import setup_logging
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.intelligent_router import initialize_routing_system
from src.generation.answer_synthesizer import AnswerSynthesizer
from src.generation.trust_formatter import TrustFormatter

from dotenv import load_dotenv

load_dotenv()

def build_pipeline():
    router = initialize_routing_system(
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_namespace="tier1_v1",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )
    return RAGPipeline(
        router=router,
        synthesizer=AnswerSynthesizer(),
        trust_formatter=TrustFormatter(),
    )


def main(query: str):
    setup_logging()

    pipeline = build_pipeline()
    print(pipeline.run(query))


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is RAG?"
    # print(f"query = {query}")
    main(query)
