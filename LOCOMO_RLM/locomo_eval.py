import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(s: Any) -> str:
    if s is None:
        return ""

    if isinstance(s, list):
        s = " ".join(str(x) for x in s)

    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: Any) -> float:
    gold_answers = gold if isinstance(gold, list) else [gold]
    pred_norm = normalize_text(pred)
    return float(any(pred_norm == normalize_text(g) for g in gold_answers))


def token_f1_single(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: Any) -> float:
    gold_answers = gold if isinstance(gold, list) else [gold]
    return max(token_f1_single(pred, str(g)) for g in gold_answers)


def rouge_l_single(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gold_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: Any) -> float:
    gold_answers = gold if isinstance(gold, list) else [gold]
    return max(rouge_l_single(pred, str(g)) for g in gold_answers)


def get_question(qa: Dict[str, Any]) -> str:
    if "question" in qa and qa["question"] is not None:
        return str(qa["question"])
    return ""


def get_answer(qa: Dict[str, Any]) -> Any:
    if "answer" in qa and qa["answer"] is not None:
        return str(qa["answer"])
    return ""


def get_category(qa: Dict[str, Any]) -> str:
    return str(qa.get("category", "unknown"))

def format_history(history: List[Dict[str, str]], max_chars: Optional[int] = None) -> str:
    chunks = []

    for i, msg in enumerate(history):
        role = msg.get("type", "unknown")
        content = msg.get("content", "")
        chunks.append(f"Turn {i + 1} | {role}:\n{content}")

    text = "\n\n".join(chunks)

    if max_chars is not None and len(text) > max_chars:
        text = text[-max_chars:]

    return text


def build_full_context_prompt(
    history: List[Dict[str, str]],
    question: str,
    max_context_chars: Optional[int] = None,
) -> str:
    conversation_text = format_history(history, max_chars=max_context_chars)

    return f"""
You are answering questions about a long multi-session conversation.

Use ONLY the conversation history below.
Do not use outside knowledge.
If the answer is not supported by the conversation, answer "I don't know."

Conversation history:
{conversation_text}

Question:
{question}

Answer with a short, direct answer only.
""".strip()


class BaseEvalAgent:
    def answer(self, sample: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Return:
            {
                "prediction": str,
                "log": Any
            }
        """
        raise NotImplementedError


class OpenAIFullContextBaseline(BaseEvalAgent):
    """
    LLM full-context baseline.

    It directly gives the whole conversation history to OpenAI.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        max_context_chars: Optional[int] = None,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_context_chars = max_context_chars

    def answer(self, sample: Dict[str, Any], question: str) -> Dict[str, Any]:
        history = sample.get("history", [])

        prompt = build_full_context_prompt(
            history=history,
            question=question,
            max_context_chars=self.max_context_chars,
        )

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful evaluator for long-context conversational memory. "
                                "Answer strictly based on the provided conversation."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                prediction = response.choices[0].message.content.strip()

                return {
                    "prediction": prediction,
                    "log": {
                        "agent": "openai_full_context",
                        "model": self.model,
                    },
                }

            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[OpenAI error] attempt={attempt + 1}, error={e}")
                time.sleep(wait_time)

        return {
            "prediction": f"[ERROR] {last_error}",
            "log": {
                "agent": "openai_full_context",
                "error": str(last_error),
            },
        }


class RLMAgentEvalWrapper(BaseEvalAgent):
    """
    Wrapper for teammate's RLMAgent interface.

    Given interface:

        from src import RLMAgent

        agent = RLMAgent(args.model_name, top_k=args.top_k)

        response, log = agent.forward(message)

    Important:
    - Their current example does not pass dataset into RLMAgent.
    - Therefore this wrapper puts LoCoMo history/documents into the message.
    - If later RLMAgent supports explicit memory loading, update TODO section below.

    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 5,
        rebuild_per_sample: bool = True,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.rebuild_per_sample = rebuild_per_sample
        self._build_agent()
        self.current_sample_id = None

    def _build_agent(self) -> None:
        """
        Build RLMAgent.

        """

        try:
            from src import RLMAgent
        except ImportError as e:
            raise ImportError(
                "Cannot import RLMAgent."
            ) from e

        self.agent = RLMAgent(
            self.model_name,
            top_k=self.top_k,
        )
    
    def load_memory(self, sample:  Dict[str, Any]):
        sample_id = sample.get("sample_id", "unknown_sample")
        # if (
        #     self.agent is None
        #     or self.rebuild_per_sample
        #     or self.current_sample_id != sample_id
        # ):
        #     self._build_agent(sample)  
        #     self.current_sample_id = sample_id
        history = sample.get("conversation", {})
        conversations = []

        for k, v in history.items():
            if k.startswith("session") and isinstance(v, list):
                for msg in v:
                    conversations.append({"type": msg["speaker"], "content": msg["text"]})

        self.agent.load_conversation(conversations)


    def answer(self, sample: Dict[str, Any], question: str) -> Dict[str, Any]:
        sample_id = sample.get("sample_id", "unknown_sample")

        self.load_memory(sample)

        prompt = """
Answer a question based on past conversation. Do not use outside knowledge.
If the answer is not supported by the conversation, answer "I don't know."

Question:
{question}

Answer with a short, direct answer only.
"""

        response, log = self.agent.forward(prompt.format(question=question))

        return {
            "prediction": str(response).strip(),
            "log": log,
        }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_results(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = {
        "em": mean([x["metrics"]["em"] for x in items]),
        "f1": mean([x["metrics"]["f1"] for x in items]),
        "rouge_l": mean([x["metrics"]["rouge_l"] for x in items]),
        "count": len(items),
    }

    by_category = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)

    category_summary = {}

    for category, group in by_category.items():
        category_summary[category] = {
            "em": mean([x["metrics"]["em"] for x in group]),
            "f1": mean([x["metrics"]["f1"] for x in group]),
            "rouge_l": mean([x["metrics"]["rouge_l"] for x in group]),
            "count": len(group),
        }

    return {
        "overall": overall,
        "by_category": category_summary,
    }


def build_eval_agent(args) -> BaseEvalAgent:
    model_name = args.model_name

    if args.agent == "openai":
        return OpenAIFullContextBaseline(
            model=model_name,
            temperature=0.0,
            max_context_chars=args.max_context_chars,
        )

    if args.agent == "rlm":
        return RLMAgentEvalWrapper(
            model_name=model_name,
            top_k=args.top_k,
            rebuild_per_sample=True,
        )

    raise ValueError(f"Unknown agent type: {args.agent}")


def evaluate(
    data_path: str,
    out_path: str,
    eval_agent: BaseEvalAgent,
    args,
):
    data = load_json(data_path)

    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []

    for sample_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        history = sample.get("conversation", [])
        qa_items = sample.get("qa", [])

        if args.max_questions_per_sample is not None:
            qa_items = qa_items[:args.max_questions_per_sample]

        print(f"\nEvaluating sample {sample_idx + 1}/{len(data)}: {sample_id}")
        print(f"History turns Num: {len(history)}, QA Num: {len(qa_items)}")

        for qa_idx, qa in enumerate(qa_items):
            question = get_question(qa)
            gold_answer = get_answer(qa)
            category = get_category(qa)

            if not question:
                continue

            print(f"QA {qa_idx + 1}/{len(qa_items)} | category={category}")

            agent_output = eval_agent.answer(
                sample=sample,
                question=question,
            )

            prediction = agent_output["prediction"]
            agent_log = agent_output.get("log")

            metrics = {
                "em": exact_match(prediction, gold_answer),
                "f1": token_f1(prediction, gold_answer),
                "rouge_l": rouge_l(prediction, gold_answer),
            }

            item = {
                "sample_id": sample_id,
                "sample_idx": sample_idx,
                "qa_idx": qa_idx,
                "category": category,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "metrics": metrics,
                "agent_log": agent_log,
            }

            results.append(item)

            print(f"Prediction: {prediction}")
            print(f"Gold: {gold_answer}")
            print(f"Metrics: {metrics}")

            # Save partial results after every question.
            partial_output = {
                "config": vars(args),
                "summary": aggregate_results(results),
                "items": results,
            }
            save_json(partial_output, out_path)

    final_output = {
        "config": vars(args),
        "summary": aggregate_results(results),
        "items": results,
    }

    save_json(final_output, out_path)

    print("\n========== Final Summary ==========")
    print(json.dumps(final_output["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to locomo_rlm_format.json",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/locomo_eval.json",
        help="Path to save evaluation results",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="openai",
        choices=["openai", "rlm"],
        help="Evaluation agent type: openai or rlm",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name. Same meaning as --model. Added for RLMAgent compatibility.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k setting passed to RLMAgent",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N samples for debugging",
    )

    parser.add_argument(
        "--max-questions-per-sample",
        type=int,
        default=None,
        help="Evaluate only first N QA items per sample for debugging",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help=(
            "Only used by --agent openai. "
            "If set, keep only the latest N characters of conversation history."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    eval_agent = build_eval_agent(args)

    evaluate(
        data_path=args.data,
        out_path=args.out,
        eval_agent=eval_agent,
        args=args,
    )
