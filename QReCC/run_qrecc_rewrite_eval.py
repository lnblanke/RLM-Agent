# Usage:
# python run_qrecc_rewrite_eval.py --data_path qrecc_data/qrecc_test.json --output_path qrecc_rewrite_results.json

import json
import re
import argparse
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_counts:
        common += min(pred_counts[t], gold_counts.get(t, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def lcs_length(x_tokens, y_tokens) -> int:
    m, n = len(x_tokens), len(y_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if x_tokens[i] == y_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n]


def rouge_l(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    lcs = lcs_length(pred_tokens, gold_tokens)

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def load_qrecc(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        examples.append({
            "id": i,
            "conversation_no": item.get("Conversation_no"),
            "turn_no": item.get("Turn_no"),
            "context": item.get("Context", []),
            "question": item.get("Question", ""),
            "gold_rewrite": item.get("Rewrite", ""),
            "gold_answer": item.get("Answer", ""),
        })

    return examples


class RLMClient:
    def __init__(self, model):
        self.model = model

    def rewrite(self, context: List[str], question: str) -> str:
        # TODO: Remove it, only for testing
        if self.model is None:
            return question

        prompt = build_rewrite_prompt(context, question)

        response = self.model.generate(
            prompt=prompt,
            max_depth=3,
        )

        return response.strip()

def build_rewrite_prompt(context: List[str], question: str) -> str:
    context_text = "\n".join(context)

    return f"""
    You are doing conversational question rewriting.

    Conversation context:
    {context_text}

    Current question:
    {question}

    Rewrite the current question into a standalone question.
    Only output the rewritten question.
    """


def evaluate_rewrite_only(
    data_path: str,
    output_path: str,
    max_examples: int = None,
):
    examples = load_qrecc(data_path)

    if max_examples is not None:
        examples = examples[:max_examples]

    rlm = RLMClient(model=None)  # TODO: Replace it

    results = []
    em_scores = []
    f1_scores = []
    rouge_scores = []

    for ex in examples:
        pred_rewrite = rlm.rewrite(
            context=ex["context"],
            question=ex["question"],
        )

        em = exact_match(pred_rewrite, ex["gold_rewrite"])
        f1 = token_f1(pred_rewrite, ex["gold_rewrite"])
        rouge = rouge_l(pred_rewrite, ex["gold_rewrite"])

        em_scores.append(em)
        f1_scores.append(f1)
        rouge_scores.append(rouge)

        results.append({
            "id": ex["id"],
            "conversation_no": ex["conversation_no"],
            "turn_no": ex["turn_no"],
            "context": ex["context"],
            "question": ex["question"],
            "gold_rewrite": ex["gold_rewrite"],
            "pred_rewrite": pred_rewrite,
            "rewrite_em": em,
            "rewrite_f1": f1,
            "rewrite_rouge_l": rouge,
        })

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    print("Rewrite-only evaluation")
    print(f"Examples: {len(examples)}")
    print(f"Exact Match: {avg_em:.4f}")
    print(f"Token F1: {avg_f1:.4f}")
    print(f"ROUGE-L: {avg_rouge:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "rewrite_exact_match": avg_em,
                "rewrite_token_f1": avg_f1,
                "rewrite_rouge_l": avg_rouge,
                "num_examples": len(examples),
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="qrecc_rewrite_results.json")
    parser.add_argument("--max_examples", type=int, default=None)

    args = parser.parse_args()

    evaluate_rewrite_only(
        data_path=args.data_path,
        output_path=args.output_path,
        max_examples=args.max_examples,
    )