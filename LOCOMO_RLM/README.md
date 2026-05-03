LoCoMo Evaluation
This script evaluates an OpenAI LLM baseline on the converted LoCoMo dataset.
---
1. Install dependencies
pip install openai bm25s
---
2. Set OpenAI API key
Linux / macOS:
export OPENAI_API_KEY="your_api_key_here"

---

3. Full run

OpenAI Baseline:

python3 locomo_eval.py  \
  --agent openai \
  --data data/locomo_rlm_format.json \
  --out results/locomo_openai_eval.json \
  --model-name gpt-4o-mini \
  --max-samples 1 \
  --max-questions-per-sample 3

RLM:

python3 locomo_eval.py \
  --agent rlm \
  --data data/locomo_rlm_format.json \
  --out results/locomo_rlm_eval.json \
  --model-name gpt-4o-mini \
  --top-k 5 \
  --max-samples 1 \
  --max-questions-per-sample 3

---
4. Metrics (EM / F1 / ROUGE-L)
per-question predictions
category breakdown
---
5. Arguments
--data: input dataset  
--out: output file  
--model-name: OpenAI model  
--max-samples: debug with first N samples  
--max-questions-per-sample: debug with first N questions  
--max-context-chars: truncate long history