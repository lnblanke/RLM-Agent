# RLM-Agent
Repository for Conversational agent with Recursive Language Models

# Getting Started

For CORAL dataset, the passage corpus need to be manually downloaded:
```bash
cd data && wget https://huggingface.co/datasets/ariya2357/CORAL/resolve/main/passage_corpus.json
```

To evaluate RLM-agent and baselines, you can simply run the following command:

```bash
python main.py --model-name=[model name] --dataset=[dataset] --agent=[agent] --output-file=[output-file] [--max-samples=[max samples]] [--max_questions_per_sample=[max question per sample]]
```

- `[model name]` is either a HuggingFace model path or a model supported by OpenAI API. For OpenAI models, a text file `api_key.txt` that contains an API key is required in the root directory. 
- `[dataset]` can be one of "QReCC" or "CORAL". The LoCoMo dataset is also available (with "LOCOMO"), but we were unable to get meaningful results from the dataset with our current experiment settings. 
- `[agent]` can be one of "rlm", "full-context", or "rag".
- `[output-file]` is the path to the output file.
- `[max samples]` (optional) specifies the subset size for evaluation.
- `[max question per sample]` (optional) can be specified for datasets containing multiple questions per sample.