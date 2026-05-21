# s01: The Agent Loop

> *"One loop & Bash is all you need"* -- 一个 tool + 一个 loop = 一个 agent。
>
> **Harness layer**: Loop -- model 与真实世界的第一次连接。



## Problem

语言模型可以推理代码，但它无法*触及*真实世界 -- 无法读取文件、运行测试或检查错误。没有 loop，每次 tool call 都需要你手动复制粘贴结果。你自己变成了那个 loop。



## Solution

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until stop_reason != "tool_use")
```

一个退出条件控制整个流程。Loop 持续运行，直到 model 停止调用 tool。



## How It Works

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

1. 用户 prompt 成为第一条 message。
2. 将 messages + tool definitions 发送给 LLM。
3. 追加 assistant response。检查 `stop_reason` -- 如果 model 没有调用 tool，流程结束。
4. 执行每个 tool call，收集结果，作为 user message 追加。回到步骤 2。



## What Changed

| Component     | Before     | After                          |
|---------------|------------|--------------------------------|
| Agent loop    | (none)     | `while True` + stop_reason     |
| Tools         | (none)     | `bash` (one tool)              |
| Messages      | (none)     | Accumulating list              |
| Control flow  | (none)     | `stop_reason != "tool_use"`    |



## Try It

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`






# s02: Tool Use

> *"Adding a tool means adding one handler"* -- loop 保持不变；新 tool 注册到 dispatch map 中。
>
> **Harness layer**: Tool dispatch -- 扩展 model 的能力范围。



## Problem

只有 `bash` 时，agent 所有操作都通过 shell 执行。`cat` 的截断行为不可预测，`sed` 在遇到特殊字符时会失败，而且每次 bash 调用都是一个不受约束的安全面。专用的 tool（如 `read_file` 和 `write_file`）可以在 tool 层面实施路径沙箱。

关键洞察：添加 tool 不需要修改 loop。



## Solution

```
+--------+      +-------+      +------------------+
|  User  | ---> |  LLM  | ---> | Tool Dispatch    |
| prompt |      |       |      | {                |
+--------+      +---+---+      |   bash: run_bash |
                    ^           |   read: run_read |
                    |           |   write: run_wr  |
                    +-----------+   edit: run_edit |
                    tool_result | }                |
                                +------------------+

The dispatch map is a dict: {tool_name: handler_function}.
One lookup replaces any if/elif chain.
```



## How It Works

1. 每个 tool 有一个 handler 函数。路径沙箱防止逃出工作区。

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]
```

2. Dispatch map 将 tool name 映射到 handler。

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}
```

3. 在 loop 中，通过名称查找 handler。Loop 本体与 s01 完全相同。

```python
for block in response.content:
    if block.type == "tool_use":
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**block.input) if handler \
            else f"Unknown tool: {block.name}"
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

添加一个 tool = 添加一个 handler + 添加一个 schema entry。Loop 永远不需要改变。



## What Changed From s01

| Component      | Before (s01)       | After (s02)                |
|----------------|--------------------|----------------------------|
| Tools          | 1 (bash only)      | 4 (bash, read, write, edit)|
| Dispatch       | Hardcoded bash call | `TOOL_HANDLERS` dict       |
| Path safety    | None               | `safe_path()` sandbox      |
| Agent loop     | Unchanged          | Unchanged                  |



## Try It

```sh
cd learn-claude-code
python agents/s02_tool_use.py
```

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`






# s03: TodoWrite

> *"An agent without a plan drifts"* -- 先列出步骤，再执行。
>
> **Harness layer**: Planning -- 让 model 保持方向，但不硬编码路线。



## Problem

在多步骤任务中，model 会失去方向。它会重复工作、跳过步骤或偏离目标。长对话使问题更严重 -- 随着 tool result 填满 context，system prompt 的影响力逐渐减弱。一个 10 步的重构任务可能完成了步骤 1-3，然后 model 开始即兴发挥，因为它忘记了步骤 4-10。



## Solution

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if rounds_since_todo >= 3:
                inject <reminder> into tool_result
```



## How It Works

1. TodoManager 存储带有状态的条目。同一时间只有一个条目可以是 `in_progress`。

```python
class TodoManager:
    def update(self, items: list) -> str:
        validated, in_progress_count = [], 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item["id"], "text": item["text"],
                              "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        return self.render()
```

2. `todo` tool 像其他 tool 一样加入 dispatch map。

```python
TOOL_HANDLERS = {
    # ...base tools...
    "todo": lambda **kw: TODO.update(kw["items"]),
}
```

3. Nag reminder 在 model 连续 3 轮以上未调用 `todo` 时注入提醒。

```python
if rounds_since_todo >= 3 and messages:
    last = messages[-1]
    if last["role"] == "user" and isinstance(last.get("content"), list):
        last["content"].insert(0, {
            "type": "text",
            "text": "<reminder>Update your todos.</reminder>",
        })
```

"同一时间只有一个 in_progress" 的约束强制顺序执行。Nag reminder 创造了问责机制。



## What Changed From s02

| Component      | Before (s02)     | After (s03)                |
|----------------|------------------|----------------------------|
| Tools          | 4                | 5 (+todo)                  |
| Planning       | None             | TodoManager with statuses  |
| Nag injection  | None             | `<reminder>` after 3 rounds|
| Agent loop     | Simple dispatch  | + rounds_since_todo counter|



## Try It

```sh
cd learn-claude-code
python agents/s03_todo_write.py
```

1. `Refactor the file hello.py: add type hints, docstrings, and a main guard`
2. `Create a Python package with __init__.py, utils.py, and tests/test_utils.py`
3. `Review all Python files and fix any style issues`






# s04: Subagents


> *"Break big tasks down; each subtask gets a clean context"* -- subagent 使用独立的 messages[]，保持主对话的整洁。
>
> **Harness layer**: Context isolation -- 保护 model 的思维清晰度。



## Problem

随着 agent 工作，它的 messages 数组不断增长。每次文件读取、每次 bash 输出都永久留在 context 中。"这个项目使用什么测试框架？"可能需要读取 5 个文件，但 parent 只需要答案："pytest。"



## Solution

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ----------> | while tool_use:  |
|   prompt="..."   |             |   call tools     |
|                  |  summary    |   append results |
|   result = "..." | <---------- | return last text |
+------------------+             +------------------+

Parent context stays clean. Subagent context is discarded.
```



## How It Works

1. Parent 获得一个 `task` tool。Child 获得所有 base tool，但没有 `task`（防止递归生成）。

```python
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task",
     "description": "Spawn a subagent with fresh context.",
     "input_schema": {
         "type": "object",
         "properties": {"prompt": {"type": "string"}},
         "required": ["prompt"],
     }},
]
```

2. Subagent 以 `messages=[]` 开始，运行自己的 loop。只有最终的文本返回给 parent。

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant",
                             "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input)
                results.append({"type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

Child 的整个 message 历史（可能 30+ 次 tool call）被丢弃。Parent 收到一段简短的摘要作为普通的 `tool_result`。



## What Changed From s03

| Component      | Before (s03)     | After (s04)               |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (base) + task (parent)  |
| Context        | Single shared    | Parent + child isolation  |
| Subagent       | None             | `run_subagent()` function |
| Return value   | N/A              | Summary text only         |



## Try It

```sh
cd learn-claude-code
python agents/s04_subagent.py
```

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`






# s05: Skills


> *"Load knowledge when you need it, not upfront"* -- 通过 tool_result 注入，而非放在 system prompt 中。
>
> **Harness layer**: On-demand knowledge -- 在 model 需要时加载领域知识。



## Problem

你希望 agent 遵循特定领域的工作流：git 规范、测试模式、代码审查清单。将所有内容放在 system prompt 中会在未使用的 skill 上浪费 token。10 个 skill，每个 2000 token = 20,000 token，其中大部分与当前任务无关。



## Solution

```
System prompt (Layer 1 -- always present):
+--------------------------------------+
| You are a coding agent.              |
| Skills available:                    |
|   - git: Git workflow helpers        |  ~100 tokens/skill
|   - test: Testing best practices     |
+--------------------------------------+

When model calls load_skill("git"):
+--------------------------------------+
| tool_result (Layer 2 -- on demand):  |
| <skill name="git">                   |
|   Full git workflow instructions...  |  ~2000 tokens
|   Step 1: ...                        |
| </skill>                             |
+--------------------------------------+
```

Layer 1: system prompt 中的 skill *名称*（开销低）。Layer 2: 通过 tool_result 加载完整 *内容*（按需加载）。



## How It Works

1. 每个 skill 是一个包含 `SKILL.md` 文件的目录，带有 YAML frontmatter。

```
skills/
  pdf/
    SKILL.md       # ---\n name: pdf\n description: Process PDF files\n ---\n ...
  code-review/
    SKILL.md       # ---\n name: code-review\n description: Review code\n ---\n ...
```

2. SkillLoader 扫描 `SKILL.md` 文件，使用目录名作为 skill 标识符。

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        for f in sorted(skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body}

    def get_descriptions(self) -> str:
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'."
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
```

3. Layer 1 写入 system prompt。Layer 2 只是另一个 tool handler。

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Skills available:
{SKILL_LOADER.get_descriptions()}"""

TOOL_HANDLERS = {
    # ...base tools...
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}
```

Model 了解有哪些 skill（开销低），并在相关时加载它们（开销高）。



## What Changed From s04

| Component      | Before (s04)     | After (s05)                |
|----------------|------------------|----------------------------|
| Tools          | 5 (base + task)  | 5 (base + load_skill)      |
| System prompt  | Static string    | + skill descriptions       |
| Knowledge      | None             | skills/\*/SKILL.md files   |
| Injection      | None             | Two-layer (system + result)|



## Try It

```sh
cd learn-claude-code
python agents/s05_skill_loading.py
```

1. `What skills are available?`
2. `Load the agent-builder skill and follow its instructions`
3. `I need to do a code review -- load the relevant skill first`
4. `Build an MCP server using the mcp-builder skill`





# s06: Context Compact

> *"Context will fill up; you need a way to make room"* -- 三层压缩策略，支持无限会话。
>
> **Harness layer**: Compression -- 为无限会话提供干净的记忆。



## Problem

Context window 是有限的。一次 `read_file` 读取 1000 行文件大约消耗 ~4000 token。读取 30 个文件并运行 20 条 bash 命令后，你会达到 100,000+ token。Agent 如果没有压缩机制，就无法处理大型代码库。



## Solution

三个层级，压缩力度递增：

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```



## How It Works

1. **Layer 1 -- micro_compact**: 每次 LLM 调用前，将旧的 tool result 替换为占位符。

```python
def micro_compact(messages: list) -> list:
    tool_results = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for j, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((i, j, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    for _, _, part in tool_results[:-KEEP_RECENT]:
        if len(part.get("content", "")) > 100:
            part["content"] = f"[Previous: used {tool_name}]"
    return messages
```

2. **Layer 2 -- auto_compact**: 当 token 超过阈值时，将完整 transcript 保存到磁盘，然后让 LLM 进行总结。

```python
def auto_compact(messages: list) -> list:
    # Save transcript for recovery
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    # LLM summarizes
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity..."
            + json.dumps(messages, default=str)[:80000]}],
        max_tokens=2000,
    )
    return [
        {"role": "user", "content": f"[Compressed]\n\n{response.content[0].text}"},
        {"role": "assistant", "content": "Understood. Continuing."},
    ]
```

3. **Layer 3 -- manual compact**: `compact` tool 按需触发相同的总结流程。

4. Loop 整合了所有三个层级：

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)                        # Layer 1
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)       # Layer 2
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)       # Layer 3
```

Transcript 在磁盘上保留了完整历史。没有任何内容真正丢失 -- 只是从活跃 context 中移出了。



## What Changed From s05

| Component      | Before (s05)     | After (s06)                |
|----------------|------------------|----------------------------|
| Tools          | 5                | 5 (base + compact)         |
| Context mgmt   | None             | Three-layer compression    |
| Micro-compact  | None             | Old results -> placeholders|
| Auto-compact   | None             | Token threshold trigger    |
| Transcripts    | None             | Saved to .transcripts/     |



## Try It

```sh
cd learn-claude-code
python agents/s06_context_compact.py
```

1. `Read every Python file in the agents/ directory one by one`（观察 micro-compact 替换旧结果）
2. `Keep reading files until compression triggers automatically`
3. `Use the compact tool to manually compress the conversation`



