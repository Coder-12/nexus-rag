from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class Section:
    """
    Logical section within a document.
    This is the smallest semantic unit before chunking.
    """
    section_id: str                 # deterministic, stable
    title: str                      # section heading
    text: str                       # raw section text
    path: List[str]                 # hierarchical path (e.g. ["Architecture", "Encoder-Decoder"])
    order: int                      # section order in document


@dataclass
class Document:
    """
    Canonical document abstraction for Nexus-RAG.
    Represents a Tier-1 knowledge artifact.
    """
    doc_id: str                     # stable, deterministic (filename-based)
    title: str                      # document title
    source: str                     # e.g. "Wikipedia"
    domain: str                     # e.g. "AI / ML"
    cluster: str                    # e.g. "Core Architecture"
    tier: str                       # e.g. "Tier-1"
    
    raw_text: str                   # full document text (unchunked)
    sections: List[Section]         # parsed semantic sections
    
    metadata: Dict[str, str]        # extensible, non-semantic signals