import json
from typing import Dict, Any
from openai import OpenAI

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_TEMPERATURE = 0.0


# ---------------------------------------------------------------------
# Judge Prompt (DO NOT MODIFY)
# ---------------------------------------------------------------------

def build_judge_prompt(question: Dict[str, Any], model_answer: str) -> str:
    return f"""
You are an impartial, highly critical evaluation judge for a production-grade
Retrieval-Augmented Generation (RAG) system.

You MUST judge strictly against the provided answer key.
You MUST penalize hallucinations and unsupported claims.
You MUST NOT introduce external knowledge.

----------------------------------------
QUESTION METADATA
----------------------------------------
Question ID: {question['id']}
Section Type: {question['section']}
Difficulty: {question['difficulty']}
Cluster: {question['cluster']}

Question:
{question['question']}

Expected Answer Key:
{question['answer_key']}

----------------------------------------
MODEL ANSWER
----------------------------------------
{model_answer}

----------------------------------------
SCORING RUBRIC
----------------------------------------
5 - EXCELLENT
4 - GOOD
3 - ACCEPTABLE
2 - POOR
1 - FAILING
0 - CRITICAL FAILURE

----------------------------------------
OUTPUT FORMAT (JSON ONLY)
----------------------------------------
{{
  "score": <integer 0-5>,
  "verdict": "<EXCELLENT|GOOD|ACCEPTABLE|POOR|FAILING|CRITICAL FAILURE>",
  "justification": "<concise explanation>",
  "hallucination": <true|false>,
  "missing_elements": [<list>],
  "extra_unsupported_claims": [<list>]
}}
""".strip()


# ---------------------------------------------------------------------
# Judge Runner
# ---------------------------------------------------------------------

class LLMJudge:
    """
    Stateless LLM Judge.
    """

    def __init__(self):
        self.client = OpenAI()

    def judge(self, question: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
        prompt = build_judge_prompt(question, model_answer)

        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "score": 0,
                "verdict": "CRITICAL FAILURE",
                "justification": "Judge output malformed",
                "hallucination": True,
                "missing_elements": question.get("answer_key", []),
                "extra_unsupported_claims": ["Malformed judge output"],
            }
