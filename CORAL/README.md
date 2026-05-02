# CORAL Evaluation Harness

This folder contains a standalone evaluation harness for **CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmented Generation**, a Findings of NAACL 2025 benchmark for multi-turn conversational RAG.

The user-provided `https://github.com/lnblanke/RLM-Agent/tree/main` URL was not publicly cloneable from this environment: GitHub returned a credential prompt/404. I kept the public benchmark source in `upstream/` and implemented this harness in `rlm_proj/CORAL`.

## Dataset

CORAL evaluates three related tasks:

1. Conversational passage retrieval.
2. Response generation.
3. Citation labeling.

The current public dataset is hosted on Hugging Face:

```bash
git lfs clone https://huggingface.co/datasets/ariya2357/CORAL
```

Expected files include:

```text
passage_corpus.json
test/new_test_conversation.json
test/new_test_qrel.trec
test/test_rewrite_new.jsonl
train/new_train_conversation.json
train/new_train_qrel.trec
train/train_rewrite_new.jsonl
```

Conversation files are JSON arrays. Each conversation has `conv_id` and a `turns` list. Each turn has `turn_id`, `question`, `response`, and `golden_docs_pids`.

## Harness Commands

All commands below run from this folder:

```bash
cd /home/mintong/DT_Agent/DecodingTrust-Agent/rlm_proj/CORAL
```

Summarize a conversation split:

```bash
python eval_coral.py summarize \
  --conversations ../CORAL_DATA/test/new_test_conversation.json
```

Flatten conversations into turn-level prompts for a model runner:

```bash
python eval_coral.py export-prompts \
  --conversations ../CORAL_DATA/test/new_test_conversation.json \
  --output prompts.test.jsonl
```

Create oracle predictions for smoke testing:

```bash
python eval_coral.py oracle-predictions \
  --conversations ../CORAL_DATA/test/new_test_conversation.json \
  --output oracle.test.jsonl
```

Evaluate JSONL predictions:

```bash
python eval_coral.py evaluate \
  --conversations ../CORAL_DATA/test/new_test_conversation.json \
  --qrels ../CORAL_DATA/test/new_test_qrel.trec \
  --predictions predictions.test.jsonl
```

Evaluate a TREC retrieval run:

```bash
python eval_coral.py evaluate \
  --conversations ../CORAL_DATA/test/new_test_conversation.json \
  --qrels ../CORAL_DATA/test/new_test_qrel.trec \
  --predictions retrieval.run
```

## Prediction Formats

JSONL predictions should contain one record per turn. The harness accepts these field aliases:

```json
{
  "sample_id": "Test_a_2_3",
  "retrieved_pids": ["6", "8"],
  "response": "Generated answer text.",
  "citations": ["6", "8"]
}
```

Accepted ID fields: `sample_id`, `query_id`, `qid`, `id`.

Accepted retrieval fields: `retrieved_pids`, `retrieved_docs`, `documents`, `doc_ids`, `pids`.

Accepted generation fields: `response`, `generated_response`, `answer`, `prediction`, `output`.

Accepted citation fields: `citations`, `predicted_citations`, `cited_pids`, `citation_pids`.

TREC retrieval runs should use:

```text
qid Q0 docid rank score tag
```

For example:

```text
Test_a_2_3 Q0 6 1 12.4 bm25
Test_a_2_3 Q0 8 2 11.9 bm25
```

## Metrics

Retrieval metrics:

- `retrieval_hit@k`
- `retrieval_recall@k`
- `retrieval_precision@k`
- `retrieval_mrr@k`
- `retrieval_ndcg@k`

Generation metrics:

- `generation_token_f1`
- `generation_rouge_l`
- `generation_exact_match`
- `generation_avg_tokens`

Citation metrics:

- `citation_precision`
- `citation_recall`
- `citation_f1`

Generation metrics are lightweight lexical checks for harness-level regression testing. For paper-level reproduction, pair this output with the official CORAL protocol and model-based/free-form answer evaluation choices used in the paper.

## Notes

- If `--qrels` is omitted, the harness falls back to `golden_docs_pids` from the conversation file.
- `sample_id` is constructed as `{conv_id}_{turn_id}`, matching the qrel/rewrite IDs such as `Test_a_2_3`.
- The harness has no third-party Python dependencies.
