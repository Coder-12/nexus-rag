import json
from typing import Any, Dict, Iterable

from openai import OpenAI


FAITHFULNESS_MODEL = "gpt-4.1-mini"
FAITHFULNESS_TEMPERATURE = 0.0


def build_context(evidence_context: Iterable[Dict[str, Any]]) -> str:
    blocks = []
    for idx, item in enumerate(evidence_context, 1):
        doc_id = item.get("doc_id", "unknown")
        section = item.get("section", "unknown")
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        blocks.append(f"[{idx}] Source: {doc_id} | {section}\n{text}")
    return "\n\n".join(blocks)


def build_faithfulness_prompt(question: str, answer: str, context: str) -> str:
    return f"""
You are evaluating answer faithfulness for a RAG system.

Judge ONLY whether the answer is supported by the retrieved context.
Do NOT judge whether the answer fully answers the question.
Do NOT use outside knowledge.
A claim is supported if it is explicitly stated in the context OR reasonably
inferable by combining retrieved sources. Do not require exact wording. Mark a
claim unsupported only when the retrieved context lacks the factual basis or
contradicts the claim.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Return JSON ONLY:
{{
  "faithful": true|false,
  "faithfulness_score": 0.0-1.0,
  "unsupported_claims": [<claims in the answer not supported by context>],
  "citation_supported": true|false,
  "justification": "<brief explanation>"
}}
""".strip()


class FaithfulnessVerifier:
    def __init__(self, model: str = FAITHFULNESS_MODEL):
        self.client = OpenAI()
        self.model = model

    def verify(self, question: str, answer: str, evidence_context: list[dict]) -> dict:
        context = build_context(evidence_context)
        if not answer.strip():
            return {
                "faithful": False,
                "faithfulness_score": 0.0,
                "unsupported_claims": ["Empty answer"],
                "citation_supported": False,
                "justification": "Empty answer cannot be supported by context.",
            }

        if not context.strip():
            refusal = "don't have enough information" in answer.lower() or "insufficient" in answer.lower()
            return {
                "faithful": refusal,
                "faithfulness_score": 1.0 if refusal else 0.0,
                "unsupported_claims": [] if refusal else ["No retrieved context available"],
                "citation_supported": refusal,
                "justification": "No retrieved context was available.",
            }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=FAITHFULNESS_TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are a strict RAG faithfulness evaluator."},
                {
                    "role": "user",
                    "content": build_faithfulness_prompt(
                        question=question,
                        answer=answer,
                        context=context,
                    ),
                },
            ],
        )

        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {
                "faithful": False,
                "faithfulness_score": 0.0,
                "unsupported_claims": ["Malformed faithfulness verifier output"],
                "citation_supported": False,
                "justification": "Faithfulness verifier returned malformed JSON.",
            }

        return {
            "faithful": bool(parsed.get("faithful", False)),
            "faithfulness_score": float(parsed.get("faithfulness_score", 0.0)),
            "unsupported_claims": list(parsed.get("unsupported_claims", [])),
            "citation_supported": bool(parsed.get("citation_supported", False)),
            "justification": str(parsed.get("justification", "")),
        }
