import json
import os
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

EVAL_RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", "outputs/evaluation_results.jsonl"))


# --------------------------------------------------
# Aggregation Logic
# --------------------------------------------------

def is_refusal(answer_text: str) -> bool:
    answer = (answer_text or "").strip().lower()
    if not answer:
        return True

    refusal_prefixes = (
        "i don't have enough information",
        "i do not have enough information",
        "i cannot answer",
        "cannot answer",
        "insufficient information",
    )
    return any(answer.startswith(prefix) for prefix in refusal_prefixes)


def aggregate_metrics():
    assert EVAL_RESULTS_PATH.exists(), f"Missing {EVAL_RESULTS_PATH}"

    total = 0
    score_sum = 0
    zero_scores = 0
    hallucinations = 0
    refusals = 0

    # Per-section stats
    section_stats = defaultdict(lambda: {
        "count": 0,
        "score_sum": 0,
        "hallucinations": 0,
        "zero_scores": 0,
        "refusals": 0,
    })

    with open(EVAL_RESULTS_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            total += 1
            section = record["section"]

            judge = record["judge"]
            score = judge.get("score", 0)
            hallucinated = judge.get("hallucination", False)

            score_sum += score
            section_stats[section]["count"] += 1
            section_stats[section]["score_sum"] += score

            if score == 0:
                zero_scores += 1
                section_stats[section]["zero_scores"] += 1

            if hallucinated:
                hallucinations += 1
                section_stats[section]["hallucinations"] += 1

            if is_refusal(record.get("model_answer", "")):
                refusals += 1
                section_stats[section]["refusals"] += 1

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("NEXUS-RAG — PHASE-0 EVALUATION METRICS")
    print("=" * 80)

    print(f"\nTotal Questions Evaluated: {total}")
    print(f"Overall Average Score: {score_sum / total:.2f}")
    print(f"Zero-Score Rate: {zero_scores / total:.2%}")
    print(f"Hallucination Rate: {hallucinations / total:.2%}")
    print(f"Refusal Rate: {refusals / total:.2%}")

    print("\n" + "-" * 80)
    print("SECTION-WISE BREAKDOWN")
    print("-" * 80)

    for section, stats in sorted(section_stats.items()):
        avg = stats["score_sum"] / stats["count"]
        zero_rate = stats["zero_scores"] / stats["count"]
        halluc_rate = stats["hallucinations"] / stats["count"]
        refusal_rate = stats["refusals"] / stats["count"]

        print(f"\n[{section}]")
        print(f"  Count            : {stats['count']}")
        print(f"  Avg Score        : {avg:.2f}")
        print(f"  Zero-Score Rate  : {zero_rate:.2%}")
        print(f"  Hallucination Rt : {halluc_rate:.2%}")
        print(f"  Refusal Rate     : {refusal_rate:.2%}")

    print("\n" + "=" * 80)
    print("END OF METRICS (Baseline v0)")
    print("=" * 80)


# --------------------------------------------------
if __name__ == "__main__":
    aggregate_metrics()
