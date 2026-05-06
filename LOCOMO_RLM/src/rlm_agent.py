# from vllm import LLM, SamplingParams
from openai import OpenAI
import bm25s
import re
import sys
from io import StringIO
import contextlib
import logging
from src.prompts import *

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

# sampling_params = SamplingParams(temperature=0, top_p=0.95, top_k=50, seed=0, max_tokens=16384)

logger = logging.getLogger(__name__)

class RLMAgent:
    def __init__(self, model_name, documents=[], top_k=5):
        # self.model = LLM(model_name, gpu_memory_utilization=0.8, trust_remote_code=True, max_model_len=16384)
        self.client = OpenAI()
        self.model_name = model_name
        self.history = []
        self.top_k = min(top_k, len(documents))
        self.index(documents)

    def index(self, documents):
        if len(documents) > 0:
            self.searcher = bm25s.BM25(corpus=documents)
            self.searcher.index(bm25s.tokenize(documents))

    def model_call(self, prompt, max_retries=10):
        # output = self.model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text

        # if output.startswith("<think>"):
        #     try:
        #         match = re.match("<think>(.*)</think>(.*)", output, flags=re.DOTALL)
        #         output = (match.group(1).strip('\n'), match.group(2).strip('\n'))
        #     except Exception as e:
        #         print(e)
        
        messages = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ["system", "user", "assistant"]:
                role = "user"

            messages.append({
                "role": role,
                "content": content
            })
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    top_p=0.95,
                    max_tokens=512,
                )
                output = response.choices[0].message.content or ""

                if output.startswith("<think>"):
                    try:
                        match = re.match(r"<think>(.*)</think>(.*)", output, flags=re.DOTALL)
                        output = (
                            match.group(1).strip("\n"),
                            match.group(2).strip("\n")
                        )
                    except Exception as e:
                        print(e)

                return output
            
            except RateLimitError as e:
                wait = min(60, 1 + attempt * 2 + random.uniform(0, 1))
                print(f"[RateLimit] 429 at attempt {attempt + 1}/{max_retries}. Sleep {wait:.1f}s and retry...")
                time.sleep(wait)

        raise RuntimeError("Rate limit still exceeded after retries.")

        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     temperature=0,
        #     top_p=0.95,
        #     max_tokens=512,
        # )

        # output = response.choices[0].message.content or ""

        # if output.startswith("<think>"):
        #     try:
        #         match = re.match(r"<think>(.*)</think>(.*)", output, flags=re.DOTALL)
        #         output = (
        #             match.group(1).strip("\n"),
        #             match.group(2).strip("\n")
        #         )
        #     except Exception as e:
        #         print(e)

        # return output

    def exec_lookup(self, record):
        msg_list = []
        user_count, agent_count = 0, 0

        for msg in self.history:
            if msg["type"] == "user":
                user_count += 1
            else:
                agent_count += 1

        idx = None

        while True:
            if idx == None:
                msg = lookup_prompt.format(conv_count=user_count + agent_count, user_conv_count=user_count, agent_conv_count=agent_count)
            elif idx != -1:
                msg = lookup_followup_prompt.format(idx=idx + 1, sender=self.history[idx]["type"], message=self.history[idx]["content"])

            msg_list.append({"role": "system", "content": msg})

            output = self.model_call(record + msg_list)

            if type(output) == str:
                msg_list.append({"role": "assistant", "content": output})
            else:
                msg_list.append({"role": "assistant", "thinking": output[0], "content": output[1]})
                output = output[1]

            if output.startswith("Lookup"):
                match = re.match(r"Lookup ([0-9]*)", output)

                try:
                    idx = int(match.group(1)) - 1
                except:
                    logger.error("exec_lookup: cannot parse " + output)
                    idx = -1
                    msg_list.append({"role": "system", "content": "Cannot parse input. Please try again using the specified format."})

                if idx < 0 or idx >= len(self.history):
                    logger.error(f"exec_lookup: index {idx + 1} out of range")
                    idx = -1
                    msg_list.append({"role": "system", "content": "Index is out of range. Please try again using a valid index."})
            else:
                return msg_list

    def exec_search(self, query):
        return self.searcher.retrieve(bm25s.tokenize(query), k=self.top_k, show_progress=False, return_as="documents")[0]

    def exec_tool(self, record):
        msg = tool_prompt
        msg_list = [{"role": "system", "content": msg}]

        while True:
            output = self.model_call(record + msg_list)
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
                msg_list.append({"role": "system", "content": "Cannot parse input. Please try again using the specified format."})
                continue

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
                tool = {"name": "search", "query": query}
                docs = self.exec_search(query=query)
                msg_list.append({"role": "system", "content": search_prompt(docs)})

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
                        msg = followup_prompt
                else:
                    if len(msg_list) == 0:
                        msg = task_prompt.format(task=task)
                    else:
                        msg = task_followup_prompt

                msg_list.append({"role": "system", "content": msg})

            output = self.model_call(msg_list)
            if type(output) == str:
                msg_list.append({"role": "assistant", "content": output})
                thinking = None
            else:
                msg_list.append({"role": "assistant", "content": output[1]})
                thinking = output[0]
                output = output[1]

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
                # continue
                return "failed", log

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
        response, log = self.exec_subtask(message, init=True)    
        self.history += [{"type": "user", "content": message}, {"type": "agent", "content": response}]
        return response, log