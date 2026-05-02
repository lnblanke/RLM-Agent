import argparse
from src import RLMAgent
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-file", type=str)
    return parser.parse_known_args()[0]

def main(args):
    agent = RLMAgent(args.model_name, top_k=args.top_k)
    traces = []

    while True:
        message = input("User: ")

        if message == "Exit":
            break

        traces.append({"type": "user", "message": message})

        response, log = agent.forward(message)
        print("Assistant:", response)

        traces.append({"type": "assistant", "message": response, "log": log})

    if args.log_file is not None:
        with open(args.log_file, "w") as f:
            json.dump(traces, f)

if __name__ == "__main__":
    args = get_args()
    main(args)