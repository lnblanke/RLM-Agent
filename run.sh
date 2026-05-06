#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/scratch/jiahaof4/huggingface_cache"

for dataset in "QReCC" "LOCOMO" "CORAL"; do
    for agent in "full-context" "rag" "rlm"; do
        python main.py --model-name=gpt-5-mini --dataset=$dataset --max-samples=100 --max_questions_per_sample=1 --agent=$agent --output-file=${dataset}-${agent}.json
    done
done