from typing import List
import tiktoken

from src.retrieval.document import Document, Section
from src.retrieval.chunk import Chunk


class SectionAwareChunker:
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        target_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []

        for section in document.sections:
            section_chunks = self._chunk_section(
                document.doc_id, document.cluster, section
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_section(
        self,
        doc_id: str,
        doc_cluster: str,
        section: Section
    ) -> List[Chunk]:

        tokens = self.encoder.encode(section.text)
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + self.target_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunk_id = f"{doc_id}::{section.section_id}::chunk::{chunk_index}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_id=section.section_id,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    chunk_index=chunk_index,
                    total_chunks=-1,  # filled later
                    section_path=section.path,
                    metadata={
                        "tier": "Tier-1",
                        "cluster": doc_cluster,
                    },
                )
            )

            chunk_index += 1
            start += self.target_tokens - self.overlap_tokens

        # fill total_chunks
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total

        return chunks
