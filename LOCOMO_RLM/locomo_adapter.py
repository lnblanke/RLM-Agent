import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_locomo(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _session_num(key: str) -> int:
    m = re.search(r"session_(\d+)$", key)
    return int(m.group(1)) if m else 10**9


def get_ordered_session_keys(conversation: Dict[str, Any]) -> List[str]:
    return sorted(
        [k for k in conversation.keys() if re.fullmatch(r"session_\d+", k)],
        key=_session_num,
    )


def locomo_to_history(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    conversation = sample["conversation"]

    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")

    history = []

    for session_key in get_ordered_session_keys(conversation):
        session_date = conversation.get(f"{session_key}_date_time", "")

        for turn in conversation.get(session_key, []):
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")

            if not text:
                continue

            # RLM Agent 只认识 user / agent
            # 这里默认 speaker_a -> user, speaker_b -> agent
            msg_type = "user" if speaker == speaker_a else "agent"

            content = (
                f"[session={session_key} | date={session_date} | "
                f"dia_id={dia_id} | speaker={speaker}]\n{text}"
            )

            history.append({
                "type": msg_type,
                "content": content,
            })

    return history


def locomo_to_documents(
    sample: Dict[str, Any],
    use_turns: bool = True,
    use_observations: bool = True,
    use_session_summaries: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:

    conversation = sample["conversation"]
    documents = []
    metadata = []

    for session_key in get_ordered_session_keys(conversation):
        session_date = conversation.get(f"{session_key}_date_time", "")

        if use_turns:
            for turn in conversation.get(session_key, []):
                dia_id = turn.get("dia_id", "")
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")

                if not text:
                    continue

                doc = (
                    f"[type=turn][session={session_key}][date={session_date}]"
                    f"[dia_id={dia_id}][speaker={speaker}]\n{text}"
                )

                documents.append(doc)
                metadata.append({
                    "type": "turn",
                    "session": session_key,
                    "date": session_date,
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "text": text,
                })

        if use_observations:
            obs_key = f"{session_key}_observation"
            obs = sample.get("observation", {}).get(obs_key)

            if obs:
                doc = (
                    f"[type=observation][session={session_key}][date={session_date}]\n"
                    f"{obs}"
                )

                documents.append(doc)
                metadata.append({
                    "type": "observation",
                    "session": session_key,
                    "date": session_date,
                    "text": obs,
                })

        if use_session_summaries:
            summary_key = f"{session_key}_summary"
            summary = sample.get("session_summary", {}).get(summary_key)

            if summary:
                doc = (
                    f"[type=session_summary][session={session_key}][date={session_date}]\n"
                    f"{summary}"
                )

                documents.append(doc)
                metadata.append({
                    "type": "session_summary",
                    "session": session_key,
                    "date": session_date,
                    "text": summary,
                })

    return documents, metadata


def get_qa_items(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sample.get("qa", [])


def convert_sample(sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    documents, metadata = locomo_to_documents(sample)
    history = locomo_to_history(sample)
    qa = get_qa_items(sample)

    return {
        "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
        "history": history,
        "documents": documents,
        "metadata": metadata,
        "qa": qa,
        "stats": {
            "num_history_messages": len(history),
            "num_documents": len(documents),
            "num_qa": len(qa),
        },
    }


def convert_locomo_file(input_path: str, output_path: str) -> None:
    raw_data = load_locomo(input_path)

    converted = [
        convert_sample(sample, sample_idx=i)
        for i, sample in enumerate(raw_data)
    ]

    save_json(converted, output_path)

    print(f"Converted {len(converted)} samples")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/locomo10.json",
        help="Path to original LoCoMo json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/locomo_rlm_format.json",
        help="Path to save converted RLM-format json file",
    )

    args = parser.parse_args()
    convert_locomo_file(args.input, args.output)


if __name__ == "__main__":
    main()