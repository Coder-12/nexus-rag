from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Chunk:
    """
    Atomic retrieval unit for Nexus-RAG.
    """
    chunk_id: str                  # deterministic
    doc_id: str
    section_id: str

    text: str                      # chunk content
    token_count: int

    chunk_index: int               # index within section
    total_chunks: int              # chunks in this section

    section_path: List[str]        # inherited from Section
    metadata: Dict[str, str]       # lightweight, non-semantic