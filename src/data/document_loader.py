from pathlib import Path
from typing import List
import re

from src.retrieval.document import Document, Section


SECTION_PATTERN = re.compile(r"=+\n(.+?)\n=+")


def load_documents(data_dir: Path) -> List[Document]:
    """
    Load and parse all .txt documents from data/raw into Document objects.
    """
    documents = []

    for file_path in sorted(data_dir.glob("*.txt")):
        documents.append(_load_single_document(file_path))

    return documents


def _load_single_document(file_path: Path) -> Document:
    raw_text = file_path.read_text(encoding="utf-8")

    lines = raw_text.splitlines()

    # -----------------------------
    # Title (first non-empty line)
    # -----------------------------
    title = next(line for line in lines if line.strip())

    doc_id = file_path.stem

    # -----------------------------
    # Extract metadata block
    # -----------------------------
    metadata = {}
    source = "unknown"
    domain = "unknown"
    cluster = "unknown"
    tier = "unknown"

    if "[METADATA — FOR SYSTEM USE ONLY]" in raw_text:
        meta_block = raw_text.split("[METADATA — FOR SYSTEM USE ONLY]")[1]
        meta_lines = meta_block.splitlines()

        for line in meta_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "source":
                    source = value
                elif key == "domain":
                    domain = value
                elif key == "cluster":
                    cluster = value
                elif key == "tier":
                    tier = value
                else:
                    metadata[key] = value

    # -----------------------------
    # Section parsing
    # -----------------------------
    sections = []
    matches = list(SECTION_PATTERN.finditer(raw_text))

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

        section_text = raw_text[start:end].strip()

        section_id = f"{doc_id}::section::{i}"

        sections.append(
            Section(
                section_id=section_id,
                title=section_title,
                text=section_text,
                path=[section_title],
                order=i,
            )
        )

    return Document(
        doc_id=doc_id,
        title=title,
        source=source,
        domain=domain,
        cluster=cluster,
        tier=tier,
        raw_text=raw_text,
        sections=sections,
        metadata=metadata,
    )