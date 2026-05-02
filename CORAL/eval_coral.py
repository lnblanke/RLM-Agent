#!/usr/bin/env python3
"""Evaluation utilities for the CORAL conversational RAG benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_pid(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("pid", "docid", "doc_id", "id"):
            if key in value:
                return str(value[key])
    return str(value)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list or JSONL records")
        return data
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
            records.append(record)
    return records


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = load_json_or_jsonl(path)
    for index, conv in enumerate(conversations):
        if "conv_id" not in conv or "turns" not in conv:
            raise ValueError(f"{path}: conversation #{index} lacks conv_id or turns")
        if not isinstance(conv["turns"], list):
            raise ValueError(f"{path}: {conv['conv_id']} turns must be a list")
    return conversations


def sample_id(conv_id: str, turn_id: Any) -> str:
    return f"{conv_id}_{turn_id}"


def iter_turns(conversations: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for conv in conversations:
        history: list[dict[str, Any]] = []
        for turn in conv["turns"]:
            turn_id = turn.get("turn_id")
            record = {
                "sample_id": sample_id(conv["conv_id"], turn_id),
                "conv_id": conv["conv_id"],
                "turn_id": turn_id,
                "question": turn.get("question", ""),
                "reference_response": turn.get("response", ""),
                "gold_pids": [normalize_pid(pid) for pid in turn.get("golden_docs_pids", [])],
                "history": list(history),
            }
            yield record
            history.append(
                {
                    "turn_id": turn_id,
                    "question": turn.get("question", ""),
                    "response": turn.get("response", ""),
                }
            )


def load_qrels(path: Path | None) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if path is None:
        return {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}:{line_no}: expected TREC qrel: qid iter docid rel")
        qid, _, docid, rel = parts[:4]
        qrels[qid][normalize_pid(docid)] = int(float(rel))
    return dict(qrels)


def qrels_from_conversations(conversations: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for turn in iter_turns(conversations):
        if turn["gold_pids"]:
            qrels[turn["sample_id"]] = {pid: 1 for pid in turn["gold_pids"]}
    return qrels


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() in {".trec", ".run", ".txt"}:
        return load_trec_run(path)
    predictions = {}
    for record in load_json_or_jsonl(path):
        qid = record.get("sample_id") or record.get("query_id") or record.get("qid") or record.get("id")
        if qid is None:
            raise ValueError(f"{path}: prediction record lacks sample_id/query_id/qid/id")
        predictions[str(qid)] = record
    return predictions


def load_trec_run(path: Path) -> dict[str, dict[str, Any]]:
    ranked: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"{path}:{line_no}: expected TREC run: qid Q0 docid rank score tag")
        qid, _, docid, rank, score, _ = parts[:6]
        ranked[qid].append((int(rank), float(score), normalize_pid(docid)))
    predictions = {}
    for qid, rows in ranked.items():
        ordered = sorted(rows, key=lambda item: (item[0], -item[1]))
        predictions[qid] = {"sample_id": qid, "retrieved_pids": [docid for _, _, docid in ordered]}
    return predictions


def extract_ranked_pids(record: dict[str, Any]) -> list[str]:
    for key in ("retrieved_pids", "retrieved_docs", "documents", "doc_ids", "pids"):
        if key in record and record[key] is not None:
            return [normalize_pid(item) for item in record[key]]
    return []


def extract_response(record: dict[str, Any]) -> str | None:
    for key in ("response", "generated_response", "answer", "prediction", "output"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def extract_citations(record: dict[str, Any]) -> list[str]:
    for key in ("citations", "predicted_citations", "cited_pids", "citation_pids"):
        if key in record and record[key] is not None:
            return [normalize_pid(item) for item in record[key]]
    return []


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    predictions: dict[str, dict[str, Any]],
    cutoffs: list[int],
) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for qid, rels in qrels.items():
        if not rels:
            continue
        ranked = extract_ranked_pids(predictions.get(qid, {}))
        relevant = {pid for pid, rel in rels.items() if rel > 0}
        for cutoff in cutoffs:
            top = ranked[:cutoff]
            hits = [1 if pid in relevant else 0 for pid in top]
            hit_count = sum(hits)
            totals[f"retrieval_hit@{cutoff}"].append(1.0 if hit_count else 0.0)
            totals[f"retrieval_recall@{cutoff}"].append(hit_count / len(relevant))
            totals[f"retrieval_precision@{cutoff}"].append(hit_count / cutoff)
            reciprocal = 0.0
            for rank, is_hit in enumerate(hits, 1):
                if is_hit:
                    reciprocal = 1.0 / rank
                    break
            totals[f"retrieval_mrr@{cutoff}"].append(reciprocal)
            ideal = dcg([1] * min(len(relevant), cutoff))
            totals[f"retrieval_ndcg@{cutoff}"].append(dcg(hits) / ideal if ideal else 0.0)
    return {name: mean(values) for name, values in totals.items()}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    overlap = 0
    for token in ref_tokens:
        if pred_counts[token] > 0:
            overlap += 1
            pred_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for j, ref_token in enumerate(ref_tokens, 1):
            if pred_token == ref_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def generation_metrics(
    turns: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, float]:
    f1s = []
    rouges = []
    exacts = []
    lengths = []
    for turn in turns:
        qid = turn["sample_id"]
        prediction = extract_response(predictions.get(qid, {}))
        if prediction is None:
            continue
        reference = turn["reference_response"]
        f1s.append(token_f1(prediction, reference))
        rouges.append(rouge_l(prediction, reference))
        exacts.append(1.0 if " ".join(tokenize(prediction)) == " ".join(tokenize(reference)) else 0.0)
        lengths.append(len(tokenize(prediction)))
    if not f1s:
        return {}
    return {
        "generation_token_f1": mean(f1s),
        "generation_rouge_l": mean(rouges),
        "generation_exact_match": mean(exacts),
        "generation_avg_tokens": mean(lengths),
    }


def citation_metrics(
    qrels: dict[str, dict[str, int]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    for qid, rels in qrels.items():
        gold = {pid for pid, rel in rels.items() if rel > 0}
        predicted = set(extract_citations(predictions.get(qid, {})))
        if not gold and not predicted:
            continue
        overlap = len(gold & predicted)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    if not f1s:
        return {}
    return {
        "citation_precision": mean(precisions),
        "citation_recall": mean(recalls),
        "citation_f1": mean(f1s),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(args: argparse.Namespace) -> None:
    conversations = load_conversations(args.conversations)
    turns = list(iter_turns(conversations))
    docs_per_turn = [len(turn["gold_pids"]) for turn in turns]
    summary = {
        "conversation_count": len(conversations),
        "turn_count": len(turns),
        "avg_turns_per_conversation": len(turns) / len(conversations) if conversations else 0.0,
        "turns_with_gold_pids": sum(1 for count in docs_per_turn if count),
        "avg_gold_pids_per_turn": mean(docs_per_turn),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def export_prompts(args: argparse.Namespace) -> None:
    conversations = load_conversations(args.conversations)
    with args.output.open("w", encoding="utf-8") as handle:
        for turn in iter_turns(conversations):
            prompt = {
                "sample_id": turn["sample_id"],
                "conv_id": turn["conv_id"],
                "turn_id": turn["turn_id"],
                "history": turn["history"],
                "question": turn["question"],
                "gold_pids": turn["gold_pids"],
            }
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")


def evaluate(args: argparse.Namespace) -> None:
    conversations = load_conversations(args.conversations)
    turns = list(iter_turns(conversations))
    qrels = load_qrels(args.qrels) if args.qrels else qrels_from_conversations(conversations)
    predictions = load_predictions(args.predictions)
    metrics: dict[str, Any] = {
        "conversation_count": len(conversations),
        "turn_count": len(turns),
        "evaluated_qrels": len(qrels),
        "prediction_count": len(predictions),
    }
    metrics.update(retrieval_metrics(qrels, predictions, args.cutoffs))
    metrics.update(generation_metrics(turns, predictions))
    metrics.update(citation_metrics(qrels, predictions))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def oracle_predictions(args: argparse.Namespace) -> None:
    conversations = load_conversations(args.conversations)
    with args.output.open("w", encoding="utf-8") as handle:
        for turn in iter_turns(conversations):
            record = {
                "sample_id": turn["sample_id"],
                "retrieved_pids": turn["gold_pids"],
                "response": turn["reference_response"],
                "citations": turn["gold_pids"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a CORAL conversation file")
    summarize_parser.add_argument("--conversations", type=Path, required=True)
    summarize_parser.set_defaults(func=summarize)

    prompt_parser = subparsers.add_parser("export-prompts", help="Flatten conversations into turn-level prompts")
    prompt_parser.add_argument("--conversations", type=Path, required=True)
    prompt_parser.add_argument("--output", type=Path, required=True)
    prompt_parser.set_defaults(func=export_prompts)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval, generation, and citation predictions")
    eval_parser.add_argument("--conversations", type=Path, required=True)
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--qrels", type=Path)
    eval_parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    eval_parser.set_defaults(func=evaluate)

    oracle_parser = subparsers.add_parser("oracle-predictions", help="Create reference predictions for smoke tests")
    oracle_parser.add_argument("--conversations", type=Path, required=True)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.set_defaults(func=oracle_predictions)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
