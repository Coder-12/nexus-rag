"""
Query Analyzer - Nexus RAG
LLM-based query analysis for intelligent routing decisions.

Responsibilities:
- Analyze user query intent and complexity
- Extract routing-relevant features
- Be fast, cheap, observable, and safe
- NEVER perform retrieval or routing
"""

from typing import Dict, List
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes queries to extract structured features for routing decisions.
    Uses a lightweight LLM for semantic intent + deterministic post-processing
    for production stability.
    """

    ANALYSIS_PROMPT = """Analyze the following search query and return a structured JSON classification.

Query:
"{query}"

Return ONLY valid JSON with the following fields:

{{
    "intent": "factual|comparative|analytical|relationship|procedural",
    "complexity": 1-10,
    "confidence": 0.0-1.0,
    "entities": ["important entities or concepts"],
    "requires_multiple_docs": true|false,
    "keywords": ["important", "terms", "from", "query"]
}}

Intent definitions:
- factual: definition or direct lookup
- comparative: compare / difference / vs
- analytical: explanation of how or why
- relationship: how concepts connect
- procedural: steps, how-to, instructions

Rules:
- Return ONLY JSON
- No markdown
- No explanations
"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Args:
            model: OpenAI model used for query analysis
                   (default: gpt-4o-mini for speed + cost)
        """
        self.client = OpenAI()
        self.model = model

    def analyze(self, query: str) -> Dict:
        """
        Analyze a query and return routing-relevant features.

        Args:
            query: User query string

        Returns:
            Dict with normalized analysis fields
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query analysis engine. Output ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": self.ANALYSIS_PROMPT.format(query=query)
                    },
                ],
                temperature=0.2,
                max_tokens=300,
            )

            raw_content = response.choices[0].message.content.strip()

            # Defensive cleanup (in case model leaks formatting)
            if raw_content.startswith("```"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()

            analysis = json.loads(raw_content)
            analysis = self._postprocess_analysis(analysis, query)

            logger.info(
                "Query analyzed | intent=%s complexity=%s confidence=%.2f",
                analysis["intent"],
                analysis["complexity"],
                analysis["confidence"],
            )

            return analysis

        except Exception as e:
            logger.warning(
                "Query analysis failed (%s). Falling back to rule-based analysis.",
                str(e),
            )
            return self._fallback_analysis(query)

    # ------------------------------------------------------------------
    # Post-processing & Normalization
    # ------------------------------------------------------------------

    def _postprocess_analysis(self, analysis: Dict, query: str) -> Dict:
        """
        Normalize and enrich LLM output with deterministic features.
        This guarantees schema stability for routing logic.
        """
        # Required fields with defaults
        intent = analysis.get("intent", "factual")
        complexity = int(analysis.get("complexity", 5))
        confidence = float(analysis.get("confidence", 0.6))
        keywords = analysis.get("keywords", [])
        entities = analysis.get("entities", [])
        requires_multiple_docs = bool(
            analysis.get("requires_multiple_docs", intent in {"comparative", "relationship"})
        )

        # Normalize keywords
        normalized_keywords = [
            k.lower() for k in keywords if isinstance(k, str) and len(k) > 1
        ]

        # Normalize entities
        normalized_entities = sorted({
            e.lower().strip()
            for e in entities
            if isinstance(e, str) and len(e.strip()) > 2
        })

        # Deterministic keyword density
        query_tokens = query.split()
        keyword_density = (
            len(normalized_keywords) / max(len(query_tokens), 1)
        )

        return {
            "intent": intent,
            "complexity": max(1, min(10, complexity)),
            "confidence": max(0.0, min(1.0, confidence)),
            "entities": normalized_entities,
            "keywords": normalized_keywords,
            "keyword_density": round(keyword_density, 3),
            "requires_multiple_docs": requires_multiple_docs,
        }

    # ------------------------------------------------------------------
    # Fallback (Rule-Based, Deterministic)
    # ------------------------------------------------------------------

    def _fallback_analysis(self, query: str) -> Dict:
        """
        Safe rule-based fallback when LLM is unavailable.
        Ensures routing NEVER breaks.
        """
        q = query.lower()

        if any(w in q for w in ["difference", "compare", "versus", "vs"]):
            intent = "comparative"
            complexity = 6
        elif any(w in q for w in ["how does", "explain", "why"]):
            intent = "analytical"
            complexity = 5
        elif any(w in q for w in ["how to", "steps", "procedure"]):
            intent = "procedural"
            complexity = 5
        elif any(w in q for w in ["related", "relationship", "connection"]):
            intent = "relationship"
            complexity = 7
        elif any(w in q for w in ["what is", "define", "meaning"]):
            intent = "factual"
            complexity = 3
        else:
            intent = "factual"
            complexity = 4

        tokens = query.split()
        keywords = [t.lower() for t in tokens[:6] if len(t) > 1]

        return {
            "intent": intent,
            "complexity": complexity,
            "confidence": 0.5,
            "entities": [],
            "keywords": keywords,
            "keyword_density": round(len(keywords) / max(len(tokens), 1), 3),
            "requires_multiple_docs": intent in {"comparative", "relationship"},
        }