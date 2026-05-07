from vllm import LLM, SamplingParams
import bm25s
import re
import sys
import numpy as np
from io import StringIO
import contextlib
import logging
from openai import OpenAI
import openai
import time
import tiktoken
from transformers import AutoTokenizer
from src.prompts import *

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

sampling_params = SamplingParams(temperature=0, top_p=0.95, top_k=50, seed=0, max_tokens=16384)

logger = logging.getLogger(__name__)

class GPTTokenizer:
    def __init__(self, name):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4-mini")
    
    def apply_chat_template(self, messages, *args, **kwargs):
        return self.tokenizer.encode(' '.join([msg["content"] for msg in messages]), *args, **kwargs)

class OpenAIModel:
    def __init__(self, model):
        with open("api_key.txt", 'r') as f:
            self.client = OpenAI(api_key=f.readline())
        self.model = model

    def chat(self, message, **kwargs):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                ).choices[0].message.content

                break
            except openai.RateLimitError as e:
                print(f'query_gpt_model: RateLimitError {e.message}: {e}')
                time.sleep(30)
            except openai.APIError as e:
                print(f'query_gpt_model: APIError {e.message}: {e}')
                print(f'query_gpt_model: Retrying after 5 seconds...')
                time.sleep(5)

        return response
class RLMAgent:
    def __init__(self, model_name, documents=[], top_k=5):
        if model_name.startswith("gpt"):
            self.model = OpenAIModel(model_name)
            self.tokenizer = GPTTokenizer(model_name)
        else:
            self.model = LLM(model_name, gpu_memory_utilization=0.8, trust_remote_code=True, max_model_len=16384)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.history = []
        self.top_k = min(top_k, len(documents))
        self.index(documents)

    def index(self, documents):
        if len(documents) > 0:
            self.searcher = bm25s.BM25(corpus=documents)
            self.searcher.index(bm25s.tokenize(documents))

    def load_conversation(self, conversation):
        if len(self.history) > 0:
            logger.warning(f"Overwritting conversation history. Original: {len(self.history)} New: {len(conversation)}")

        self.history = []

        last_msg = "agent"

        for msg in conversation:
            if type(msg) == str:
                if last_msg == "agent":
                    last_msg = "user"
                else:
                    last_msg = "agent"

                msg = {"type": last_msg, "content": msg}
            else:
                assert type(msg) == dict
                last_msg = msg["type"]

            self.history.append(msg)

    def model_call(self, prompt):
        self.token_count += len(self.tokenizer.apply_chat_template(prompt))
        output = self.model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)
        
        if not isinstance(output, str): 
            output = output[0].outputs[0].text

        if output.startswith("<think>"):
            try:
                match = re.match("<think>(.*)</think>(.*)", output, flags=re.DOTALL)
                output = (match.group(1).strip('\n'), match.group(2).strip('\n'))
            except Exception as e:
                print(e)

        return output

    def exec_lookup(self, record, message=None):
        msg_list = []
        user_count, agent_count = 0, 0

        for msg in self.history:
            if msg["type"] == "user":
                user_count += 1
            else:
                agent_count += 1

        if message is None:
            msg = lookup_prompt.format(conv_count=user_count + agent_count)
            msg_list.append({"role": "system", "content": msg})

        while True:
            if message is None:
                output = self.model_call(record + msg_list)
            else:
                output = message
                message = None

            if type(output) == str:
                msg_list.append({"role": "assistant", "content": output})
            else:
                msg_list.append({"role": "assistant", "thinking": output[0], "content": output[1]})
                output = output[1]

            match = re.match(r"Lookup (.*)", output)

            try:
                indices = list(map(int, match.group(1).split(',')))
            except:
                logger.error("exec_lookup: cannot parse " + output)
                msg_list.append({"role": "system", "content": "Cannot parse input."})
                return msg_list

            if any([idx <= 0 or idx > len(self.history) for idx in indices]):
                logger.error(f"exec_lookup: index out of range")
                msg_list.append({"role": "system", "content": "Some indices are out of range. Please try again using valid indices."})
                continue

            if len(indices) > 10:
                logger.error(f"exec_lookup: too many indices")
                msg_list.append({"role": "system", "content": "You are looking up too many indices"})
                continue
            
            msg = lookup_followup_prompt([(idx, self.history[idx - 1]["type"], self.history[idx - 1]["content"]) for idx in indices])
            msg_list.append({"role": "system", "content": msg})
            return msg_list

    def exec_search(self, query):
        return self.searcher.retrieve(bm25s.tokenize(query), k=self.top_k, show_progress=False, return_as="documents")[0]

    def exec_tool(self, record, message=None):
        if message is None:
            msg = tool_prompt
            msg_list = [{"role": "system", "content": msg}]
        else:
            msg_list = []

        while True:
            if message is None:
                output = self.model_call(record + msg_list)
            else:
                output = message
                message = None
            if type(output) == str:
                msg_list.append({"role": "assistant", "content": output})
            else:
                msg_list.append({"role": "assistant", "thinking": output[0], "content": output[1]})
                output = output[1]

            match = re.search(r"Tool ([1-2])(\n|: )(.*)", output, flags=re.DOTALL)

            try:
                opt = int(match.group(1))
            except:
                logger.error("exec_tool: cannot parse " + output)
                msg_list.append({"role": "system", "content": "Cannot parse input."})
                return None, msg_list

            if opt == 1:
                try:
                    match = re.search("```python\n(.*?)\n```", match.group(3).strip('\n'), flags=re.DOTALL)
                    code = match.group(1)
                except:
                    logger.error("exec_tool_python: cannot parse " + output)
                    msg_list.append({"role": "system", "content": "Cannot parse input. Please try again using the specified format."})
                    continue

                if "os" in code or "subprocess" in code or "shutil" in code:
                    logger.error("exec_tool_python: usage of forbidden libraries\n" + output)
                    msg_list.append({"role": "system", "content": "The program contains usage of forbidden libraries"})
                    continue

                tool = {"name": "python", "code": code}

                try:
                    with stdoutIO() as s:
                        exec(code)
                except Exception as e:
                    logger.error(f"exec_tool_python: execution error ({e})\n" + output)
                    msg_list.append({"role": "system", "content": python_failed_prompt.format(error=e)})
                    continue

                msg_list.append({"role": "system", "content": python_prompt.format(result=s.getvalue())})
            else:
                query = match.group(3).strip(' ')
                tool = {"name": "search", "query": query, "docs": []}
                if self.top_k > 0:
                    docs = self.exec_search(query=query)
                    tool["docs"] = docs.tolist()
                    msg_list.append({"role": "system", "content": search_prompt(docs)})
                else:
                    msg_list.append({"role": "system", "content": "The search agent is currently unavailabe because no documents are loaded into the database."})

            break

        return tool, msg_list

    def exec_subtask(self, task, init=False):
        msg_list = []
        log = []
        error = False

        while True:
            if error:
                error = False
            else:
                if init:
                    if len(msg_list) == 0:
                        msg = init_prompt.format(message=task)
                    else:
                        msg = followup_prompt.format(message=task)
                else:
                    if len(msg_list) == 0:
                        msg = task_prompt.format(task=task)
                    else:
                        msg = task_followup_prompt.format(task=task)

                msg_list.append({"role": "system", "content": msg})

            output = self.model_call(msg_list)
            if type(output) == str:
                msg_list.append({"role": "assistant", "content": output})
                thinking = None
            else:
                msg_list.append({"role": "assistant", "content": output[1]})
                thinking = output[0]
                output = output[1]

            if output.startswith("Lookup"):
                lookup_msgs = self.exec_lookup(record=msg_list, message=output)
                msg_list += lookup_msgs

                log.append({
                    "type": "lookup",
                    "messages": lookup_msgs,
                    "thinking": thinking,
                })
                continue
            if output.startswith("Tool"):
                tool, tool_msgs = self.exec_tool(record=msg_list, message=output)
                msg_list += tool_msgs

                log.append({
                    "type": "tool calling",
                    "tool": tool,
                    "messages": tool_msgs,
                    "thinking": thinking,
                })
                continue

            match = re.search(r"Action ([0-3])(.*)", output)

            try:
                opt = int(match.group(1))
            except:
                error = True
                logger.error("exec_subtask: cannot parse " + output)
                msg_list.append({"role": "system", "content": "Cannot parse input. Please try again using the specified format."})
                log.append({
                    "type": "error",
                    "message": "exec_subtask: cannot parse " + output
                })
                return output, log

            if opt == 1:
                lookup_msgs = self.exec_lookup(record=msg_list)
                msg_list += lookup_msgs

                log.append({
                    "type": "lookup",
                    "messages": lookup_msgs,
                    "thinking": thinking,
                })
            elif opt == 2:
                tool, tool_msgs = self.exec_tool(record=msg_list)
                msg_list += tool_msgs

                log.append({
                    "type": "tool calling",
                    "tool": tool,
                    "messages": tool_msgs,
                    "thinking": thinking,
                })
            elif opt == 3:
                tasks = match.group(2)[2:].split(',')

                task_responses = []

                log.append({
                    "type": "task",
                    "tasks": [],
                    "thinking": thinking,
                })

                for task in tasks:
                    task_output, task_log = self.exec_subtask(task)
                    task_responses.append((task, task_output))
                    log[-1]["tasks"].append({"task": task, "output": task_output, "log": task_log})

                msg_list.append({"role": "system", "content": task_complete_prompt(task_responses)}) 
            else:
                response = match.group(2)[2:].strip(' ')

                log.append({
                    "type": "respond",
                    "content": response,
                    "thinking": thinking,
                })

                return response, log

    def forward(self, message):
        self.token_count = 0
        response, log = self.exec_subtask(message, init=True)    
        self.history += [{"type": "user", "content": message}, {"type": "agent", "content": response}]

        log.append({"token_count": self.token_count})

        return response, log

full_context_prompt = """You are an AI agent engaging in a conversation with the user and now you need to reply to a new message from the user.

Message: {message}

Past conversations:

{conversations}

"""

full_context_prompt_with_retrieval = """You are an AI agent engaging in a conversation with the user and now you need to reply to a new message from the user.

Message: {message}

Top {top_k} relevant documents: 
{documents}

Past conversations:

{conversations}

"""

class FullContextAgent(RLMAgent):
    def forward(self, message):
        self.token_count = 0
        past_conversation = '\n'.join(["{speaker}: {content}".format(speaker=turn["type"], content=turn["content"]) for turn in self.history])

        if self.top_k == 0:
            prompt = full_context_prompt.format(message=message, conversations=past_conversation)
            docs = np.array([])
        else:
            docs = self.exec_search(message)
            prompt = full_context_prompt_with_retrieval.format(message=message, top_k=self.top_k, documents='\n'.join(docs), conversations=past_conversation)

        response = self.model_call([{"role": "system", "content": prompt}])
        self.history += [{"type": "user", "content": message}, {"type": "agent", "content": response}]
        return response, [{
            "type": "tool calling",
            "tool": {
                "name": "search",
                "docs": docs.tolist()
            }
        }, {"token_count": self.token_count}]
    
rag_prompt = """You are an AI agent engaging in a conversation with the user and now you need to reply to a new message from the user.

Message: {message}

Top {k} most relevant messages from past conversations:

{conversations}

"""

class RAGAgent(RLMAgent):
    def forward(self, message):
        self.token_count = 0
        if len(self.history) > 0:
            corpus = ["{speaker}: {content}".format(speaker=turn["type"], content=turn["content"]) for turn in self.history]
            searcher = bm25s.BM25(corpus=corpus)
            searcher.index(bm25s.tokenize(corpus))

            top_k = min(10, len(self.history))
            
            docs = searcher.retrieve(bm25s.tokenize(message), k=top_k, show_progress=False, return_as="documents")[0]
        else:
            top_k = 0
            docs = []

        prompt = rag_prompt.format(message=message, conversations='\n'.join(docs), k=top_k)
        response = self.model_call([{"role": "system", "content": prompt}])
        self.history += [{"type": "user", "content": message}, {"type": "agent", "content": response}]
        return response, [{"token_count": self.token_count}]