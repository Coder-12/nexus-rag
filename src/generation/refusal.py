from enum import Enum


class RefusalReason(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
