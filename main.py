import argparse
from pathlib import Path
from QReCC.run_qrecc_rewrite_eval import evaluate_rewrite_only
from LOCOMO_RLM.locomo_eval import build_eval_agent, evaluate as locomo_evaluate
from CORAL.eval_coral import predict, evaluate as coral_evaluate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="rlm", choices=["rlm", "rag", "full-context"])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max_questions_per_sample", type=int)
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_known_args()[0]

def main(args):
    if args.dataset == "QReCC":
        evaluate_rewrite_only(args.agent, args.model_name, data_path="data/qrecc-test.json", output_path=args.output_file, max_examples=args.max_samples)
    elif args.dataset == "LOCOMO":
        agent = build_eval_agent(args)
        locomo_evaluate("data/locomo10.json", args.output_file, agent, args)
    elif args.dataset == "CORAL":
        args.corpus = Path("data/passage_corpus.json")
        args.conversations = Path("data/new_test_conversation.json")
        args.output_file = args.output_file
        args.qrels = Path("data/new_test_qrel.trec")
        args.cutoffs = [1, 3, 5, 10, 20]
        predict(args)
        args.predictions = Path(args.output_file)
        coral_evaluate(args)
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")

if __name__ == "__main__":
    args = get_args()
    main(args)