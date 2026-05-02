init_prompt = """You are an AI agent engaging in a conversation with the user and now you need to reply to a new message from the user. Due to length limit, the past conversations are not shown but you can request to look up (as specified later). To improve the quality of the reply, you can take the following actions several times before replying to the message:

1. Lookup a specified section of the past conversation
2. Call the external tools available (search engine, Python interpreter, etc.)
3. Create a list of subtasks that will be dispatched to other agents to complete

User message: {message}

If you want to take action 1 or 2, respond with "Action 1" or "Action 2", and we will provide further instructions.
If you want to take action 3, respond with "Action 3: [task 1],[task 2],...", where [task i] is the description of i-th task to be performed.
If you want to directly respond to the message, respond with "Action 0: [message to reply]".
"""

followup_prompt = """You can now continue to take actions to construct a high-quality response to user message. As the reminder, you can take the following actions:

1. Lookup a specified section of the past conversation
2. Call the external tools available (search engine, Python interpreter, etc.)
3. Create a list of subtasks that will be dispatched to other agents to complete

If you want to take action 1 or 2, respond with "Action 1" or "Action 2", and we will provide further instructions.
If you want to take action 3, respond with "Action 3: [task 1],[task 2],...", where [task i] is the description of i-th task to be performed.
If you want to respond to the message, respond with "Action 0: [message to reply]".
"""

lookup_prompt = """You just asked to lookup the past conversation. According to the record, there are {conv_count} total messages from past conversation (excluding the one user just sent), including {user_conv_count} user messages and {agent_conv_count} agent replies. 

If you want to read the x-th message, respond with "Lookup [x]". The messages are ordered by the time sent, where the first message is the earliest one in the conversation and {conv_count} is the latest one. x should be in the range [1, {conv_count}].
"""

lookup_followup_prompt = """Message {idx} is: 

Sender: {sender}
Content: {message}

If you want to read other messages, respond with "Lookup [x]" to read the x-th message. Otherwise, respond with "Quit"
"""

tool_prompt = """You just asked to call the external tools available. Currently the system supports the following external tools to use:
1. A Python interpreter that can run Python scripts
2. A search agent that can collect relevant information for a given query from an external database.

If you want to use tool 1, respond with the following format:

Tool 1
```python
[Python script to execute]
```

You should not use any system-related modules (e.g. os, shutil, subprocess, etc.) in the code to execute. Please provide full Python scripts (using print() to get the final output), or the interpreter will output None.

If you want to use tool 2, respond with "Tool 2: [search query]", where [search query] is the query provided to the search agent.
"""

python_prompt = """The output of your program is 

{result}
"""

python_failed_prompt = """The execution of your program failed due to the following error:

{error}
"""

search_prompt = lambda docs: f"The top {len(docs)} most relevant documents for your query are listed below.\n\n" + "\n\n".join(f"Document {i}: {doc}" for i, doc in enumerate(docs)) + '\n'

task_prompt = """You are an AI agent engaging in a conversation with a user and you are asked to complete a given task. Due to length limit, the past conversations are not shown but you can request to look up (as specified later). To improve the quality of the reply, you can take the following actions several times to complete the task:

1. Lookup a specified section of the past conversation
2. Call the external tools available (search engine, Python interpreter, etc.)
3. Create a list of subtasks that will be dispatched to other agents to complete

Task: {task}

If you want to take action 1 or 2, respond with "Action 1" or "Action 2", and we will provide further instructions.
If you want to take action 3, respond with "Action 3: [subtask 1],[subtask 2],...", where [subtask i] is the description of i-th subtask to be performed.
If you think the task can be directly completed, respond with "Action 0: [output of the task]".
"""

task_followup_prompt = """You can now continue to take actions to complete the task. As the reminder, you can take the following actions:

1. Lookup a specified section of the past conversation
2. Call the external tools available (search engine, Python interpreter, etc.)
3. Create a list of subtasks that will be dispatched to other agents to complete

If you want to take action 1 or 2, respond with "Action 1" or "Action 2", and we will provide further instructions.
If you want to take action 3, respond with "Action 3: [subtask 1],[subtask 2],...", where [subtask i] is the description of i-th subtask to be performed.
If you think the task can be completed, respond with "Action 0: [output of the task]".
"""

task_complete_prompt = lambda tasks: "The subtasks you requested have been completed. Here are the output of each subtask:\n\n" + '\n\n'.join([f"Subtask: {task}\nOutput: {output}" for (task, output) in tasks]) + '\n'