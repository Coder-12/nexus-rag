import json
import re
from pathlib import Path

INPUT_FILE = Path("data/evaluation/eval_corpus.txt")
OUTPUT_FILE = Path("data/evaluation/eval_questions.jsonl")


def normalize_answer_key(text):
    # split on ; or →
    parts = re.split(r";|→", text)
    return [p.strip() for p in parts if p.strip()]

def main():
    raw = INPUT_FILE.read_text()
    lines = raw.splitlines()

    questions = []
    current = {}

    for line in lines:
        line = line.strip()

        # New question
        q_match = re.match(r"(Q\d+\.\d+)\s+\[(.+?)\]", line)
        if q_match:
            if current:
                questions.append(current)
                current = {}

            current["id"] = q_match.group(1)
            current["section"] = q_match.group(2)
            continue

        if line.startswith("Question:"):
            current["question"] = line.replace("Question:", "").strip().strip('"')
            continue

        if line.startswith("[CLUSTER]:"):
            current["cluster"] = line.replace("[CLUSTER]:", "").strip()
            continue

        if line.startswith("[DOCS]:"):
            docs = []
            if ".txt" in line:
                docs = [d.strip() for d in re.findall(r"\w+\.txt", line)]
            current["docs"] = docs
            continue

        if line.startswith("- ") and ".txt" in line:
            if "docs" not in current:
                current["docs"] = []
            current["docs"].append(line.replace("-", "").strip().split()[0])
            continue

        if line.startswith("[SECTION]:"):
            sec = line.replace("[SECTION]:", "").strip()
            current["sections"] = [sec]
            continue

        if line.startswith("[DIFFICULTY]:"):
            current["difficulty"] = line.replace("[DIFFICULTY]:", "").strip()
            continue

        if line.startswith("[ANSWER_KEY]:"):
            ak = line.replace("[ANSWER_KEY]:", "").strip()
            current["answer_key"] = normalize_answer_key(ak)
            continue

    if current:
        questions.append(current)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with OUTPUT_FILE.open("w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(questions)} evaluation questions")
    print(f"📄 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()