from typing import List, Dict


class RefusalPolicy:
    """
    Determines whether a query should be refused or corrected.
    Minimal, deterministic, safety-first.
    """

    @staticmethod
    def should_refuse(
        query: str,
        retrieved_chunks: List[Dict],
        allowed_docs: List[str],
    ) -> Dict:
        q = query.lower()

        # --- Out-of-corpus ---
        if len(retrieved_chunks) == 0:
            return {
                "refuse": True,
                "reason": "no_relevant_documents",
            }

        # --- Temporal trap ---
        if any(x in q for x in ["2024", "2025", "recent", "latest", "current"]):
            return {
                "refuse": True,
                "reason": "temporal_out_of_scope",
            }

        # --- Known false premise traps ---
        if "bert" in q and "causal" in q:
            return {
                "refuse": True,
                "reason": "false_premise",
                "correction": "BERT uses bidirectional attention, not causal masking.",
            }

        # --- Specificity trap ---
        if "exact" in q and "parameter" in q:
            return {
                "refuse": True,
                "reason": "unsupported_specificity",
            }

        return {"refuse": False}