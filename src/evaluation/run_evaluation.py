import json
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime, UTC
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


load_dotenv()
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EVAL_QUESTIONS_PATH = Path(os.getenv("EVAL_QUESTIONS_PATH", "data/evaluation/eval_questions.jsonl"))
OUTPUT_PATH = Path(os.getenv("EVAL_OUTPUT_PATH", "outputs/evaluation_results.jsonl"))

JUDGE_MODEL = "gpt-4.1-mini"  # conservative + cheap + deterministic
JUDGE_TEMPERATURE = 0.0
ENABLE_FAITHFULNESS = os.getenv("EVAL_ENABLE_FAITHFULNESS", "1").lower() not in {
    "0",
    "false",
    "no",
}

logger = logging.getLogger(__name__)


def to_eval_answer(answer_obj: dict) -> str:
    if not isinstance(answer_obj, dict):
        return "I don't have enough information to answer this question."

    if answer_obj.get("refused"):
        return "I don't have enough information to answer this question."

    answer = answer_obj.get("answer")

    # 🔥 HANDLE TRUST FORMATTER OUTPUT
    if isinstance(answer, dict):
        text = answer.get("text", "")
    else:
        text = answer or ""

    if not isinstance(text, str):
        return "I don't have enough information to answer this question."

    if not text.strip():
        return "I don't have enough information to answer this question."

    # normalize
    text = (
        text.replace("Comparison:", "")
            .replace("Steps:", "")
            .replace("Explanation:", "")
            .replace("\n", " ")
            .replace("-", " ")
    )

    return " ".join(text.split())


def load_eval_questions(path: Path) -> list[dict]:
    """
    Load either strict JSONL questions or the annotated generalization eval
    format used for audit-reviewed question sets.
    """
    text = path.read_text()
    first_content = next((line.strip() for line in text.splitlines() if line.strip()), "")

    if first_content.startswith("{"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    return parse_annotated_eval(text)


def parse_annotated_eval(text: str) -> list[dict]:
    questions = []
    current_section = "GENERALIZATION"
    current_cluster = "Generalization"
    current_id = None
    current_question = None
    current_docs = []
    current_answer_key = []
    collecting_answer_key = False

    def flush():
        nonlocal current_id, current_question, current_docs, current_answer_key
        if not current_id or not current_question:
            return
        questions.append({
            "id": current_id,
            "section": current_section,
            "question": current_question,
            "cluster": current_cluster,
            "docs": current_docs,
            "sections": [],
            "difficulty": "Generalization",
            "answer_key": current_answer_key,
        })
        current_id = None
        current_question = None
        current_docs = []
        current_answer_key = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if collecting_answer_key:
            if line.startswith("[") or re.match(r"^([A-Z]{2}-\d{2})$", line):
                collecting_answer_key = False
            elif line.startswith(("HOP ", "IMPLICIT:")):
                current_answer_key.append(line)
                continue

        section_match = re.match(r"^[A-G]\.\s+(.+?)\s+\(\d+\)$", line)
        if section_match:
            flush()
            current_cluster = section_match.group(1).strip()
            current_section = current_cluster.upper().replace(" ", "_").replace("-", "_")
            continue

        id_match = re.match(r"^([A-Z]{2}-\d{2})$", line)
        if id_match:
            flush()
            current_id = id_match.group(1)
            continue

        question_match = re.match(r'^Question:\s+"(.+)"$', line)
        if question_match:
            current_question = question_match.group(1).strip()
            continue

        docs_match = re.match(r"^\[DOCS\]:\s+(.+)$", line)
        if docs_match:
            current_docs = [
                normalize_eval_doc_ref(doc)
                for doc in re.split(r",|→", docs_match.group(1))
                if normalize_eval_doc_ref(doc)
            ]
            continue

        answer_match = re.match(r"^\[ANSWER_KEY\]:\s+(.+)$", line)
        if answer_match:
            current_answer_key = [
                item.strip()
                for item in re.split(r";\s*|\s+\|\s+", answer_match.group(1))
                if item.strip()
            ]
            collecting_answer_key = not current_answer_key
            continue

        if line == "[ANSWER_KEY]:":
            current_answer_key = []
            collecting_answer_key = True

    flush()

    if not questions:
        raise ValueError(f"No evaluation questions could be parsed from {EVAL_QUESTIONS_PATH}")

    return questions


def normalize_eval_doc_ref(doc_ref: str) -> str:
    doc_ref = doc_ref.strip()
    if not doc_ref or doc_ref.startswith("NONE"):
        return ""

    doc_ref = doc_ref.split()[0].strip()
    if doc_ref.endswith(".txt"):
        doc_ref = doc_ref[:-4]

    aliases = {
        "ai_alignment_tier1_core": "ai_alignment",
        "attention_mechanism_tier1_core": "attention_mechanism",
        "bert_language_model_tier1_core": "bert_architecture",
        "embeddings_machine_learning_tier1_core": "embeddings",
        "fine_tuning_deep_learning_tier1_core": "fine_tuning",
        "gpt_generative_pretrained_transformer_tier1_core": "gpt_architecture",
        "in_context_learning_tier1_core": "in_context_learning",
        "large_language_model_llm_tier1_core": "large_language_models",
        "prompt_engineering_tier1_core": "prompt_engineering",
        "retrieval_augmented_generation_rag_tier1_core": "retrieval_augmented_generation",
        "rlhf_reinforcement_learning_human_feedback_tier1_core": "reinforcement_learning_with_human_feedback",
        "semantic_search_tier1_core": "semantic_search",
        "seq2seq_sequence_to_sequence_tier1_core": "encoder_decoder_models",
        "transfer_learning_tier1_core": "transfer_learning",
        "transformer_architecture_tier1_core": "transformer_architecture",
        "vector_database_tier1_core": "vector_database",
    }
    return aliases.get(doc_ref, doc_ref)


# ---------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------
def run_evaluation():
    # Local imports after sys.path modification to avoid module-level imports after executable code
    from src.logging_config import setup_logging
    from src.evaluation.run_llm_judge import LLMJudge
    from src.evaluation.faithfulness_verifier import FaithfulnessVerifier
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.retrieval.hybrid_retrieval import initialize_hybrid_system
    from src.retrieval.intelligent_router import initialize_routing_system
    from src.generation.answer_synthesizer import AnswerSynthesizer
    from src.generation.trust_formatter import TrustFormatter

    setup_logging()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    router = initialize_routing_system(
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_namespace="tier1_v1",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        bm25_cache_path=Path("cache/bm25_index.pkl"),
    )
    pipeline = RAGPipeline(
        router=router,
        synthesizer=AnswerSynthesizer(),
        trust_formatter=TrustFormatter(),
    )

    judge = LLMJudge()
    faithfulness_verifier = FaithfulnessVerifier() if ENABLE_FAITHFULNESS else None

    questions = load_eval_questions(EVAL_QUESTIONS_PATH)

    eval_ids = {
        item.strip()
        for item in os.getenv("EVAL_IDS", "").split(",")
        if item.strip()
    }
    if eval_ids:
        questions = [q for q in questions if q["id"] in eval_ids]
        missing_ids = eval_ids - {q["id"] for q in questions}
        if missing_ids:
            raise ValueError(f"Unknown EVAL_IDS: {sorted(missing_ids)}")

    print(f"🧪 Running evaluation on {len(questions)} questions\n")

    with open(OUTPUT_PATH, "w") as out:
        for idx, q in enumerate(questions, 1):
            print(f"[{idx}/{len(questions)}] {q['id']}")

            # ---------------------------------------------------------
            # Run RAG system
            # ---------------------------------------------------------
            try:
                pipeline_output = pipeline.run(q["question"])
            except Exception as e:
                pipeline_output = f"[PIPELINE ERROR] {str(e)}"
            
            if isinstance(pipeline_output, dict):
                model_answer_obj = pipeline_output
                meta = pipeline_output.get("meta", {})
            else:
                model_answer_obj = {
                    "answer": str(pipeline_output),
                    "refused": False
                }
                meta = {}

            model_answer = to_eval_answer(model_answer_obj)
            
            print("EVAL ANSWER:", model_answer)
            judge_json = judge.judge(q, model_answer)
            faithfulness_json = None
            if faithfulness_verifier is not None:
                faithfulness_json = faithfulness_verifier.verify(
                    question=q["question"],
                    answer=model_answer,
                    evidence_context=meta.get("evidence_context", []),
                )
            logger.info(
                "EVAL_QUERY_TRACE %s",
                json.dumps(
                    {
                        "question_id": q["id"],
                        "judge_score": judge_json.get("score"),
                        "hallucination": judge_json.get("hallucination"),
                        "faithful": (
                            faithfulness_json.get("faithful")
                            if faithfulness_json is not None
                            else None
                        ),
                        "faithfulness_score": (
                            faithfulness_json.get("faithfulness_score")
                            if faithfulness_json is not None
                            else None
                        ),
                    }
                ),
            )
            # ---------------------------------------------------------
            # Save result
            # ---------------------------------------------------------
            record = {
                "question_id": q["id"],
                "section": q["section"],
                "difficulty": q["difficulty"],
                "cluster": q["cluster"],
                "question": q["question"],
                "expected_docs": q.get("docs", []),
                "answer_key": q.get("answer_key", []),
                "model_answer": model_answer,
                "citations": (
                    model_answer_obj.get("citations", [])
                    if isinstance(model_answer_obj, dict)
                    else []
                ),
                "meta": meta,
                "judge": judge_json,
                "faithfulness": faithfulness_json,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            out.write(json.dumps(record) + "\n")

    print("\n✅ Evaluation complete.")
    print(f"📄 Results saved to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
