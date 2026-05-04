from pathlib import Path
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.pipeline.ingestion_pipeline import IngestionPipeline


if __name__ == "__main__":
    pipeline = IngestionPipeline(
        data_dir=Path("data/raw"),
        index_name=os.environ["PINECONE_INDEX_NAME"],
        namespace="tier1_v1",
    )

    doc_ids = [
        doc_id.strip()
        for doc_id in os.getenv("INGEST_DOC_IDS", "").split(",")
        if doc_id.strip()
    ]

    if doc_ids:
        pipeline.ingest_documents(doc_ids=doc_ids)
    else:
        pipeline.ingest()
