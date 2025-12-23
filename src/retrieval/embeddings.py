from typing import List
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from src.retrieval.chunk import Chunk


class EmbeddingGenerator:
    """
    Deterministic embedding layer for Nexus-RAG.
    Converts chunks into dense vectors.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
    ):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed_chunks(self, chunks: List[Chunk]) -> List[dict]:
        """
        Returns list of dicts:
        {
            "chunk_id": str,
            "vector": List[float],
            "metadata": dict
        }
        """
        embeddings = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            texts = [c.text for c in batch]

            vectors = self._embed_batch(texts)

            for chunk, vector in zip(batch, vectors):
                embeddings.append({
                    "chunk_id": chunk.chunk_id,
                    "vector": vector,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "section_id": chunk.section_id,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "section_path": " > ".join(chunk.section_path),
                        **chunk.metadata,
                    }
                })

        return embeddings

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
