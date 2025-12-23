from typing import List, Dict, Optional
import os
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PineconeVectorStore:
    """
    Thin Pinecone adapter for Nexus-RAG.
    No retrieval logic. No business logic.
    """

    def __init__(
        self,
        index_name: str,
        namespace: Optional[str] = None,
    ):
        self.client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
        )

        self.index = self.client.Index(index_name)
        self.namespace = namespace

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def upsert(self, vectors: List[Dict]):
        """
        vectors: List[{
            "chunk_id": str,
            "vector": List[float],
            "metadata": dict
        }]
        """
        payload = [
            (
                v["chunk_id"],
                v["vector"],
                v["metadata"],
            )
            for v in vectors
        ]

        self.index.upsert(
            vectors=payload,
            namespace=self.namespace,
        )

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters,
            namespace=self.namespace,
        )

    def stats(self):
        return self.index.describe_index_stats()

    def delete_by_doc_id(self, doc_id: str):
        """
        Safe deletion for re-ingestion / debugging.
        No-op if namespace does not exist.
        """
        try:
            self.index.delete(
                filter={"doc_id": {"$eq": doc_id}},
                namespace=self.namespace,
            )
        except NotFoundException:
            # Namespace does not exist yet — safe to ignore
            pass
