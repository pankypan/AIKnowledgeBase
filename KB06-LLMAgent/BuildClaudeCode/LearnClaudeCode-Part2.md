# s07: Task System

> *"Todos live in memory; tasks live on disk"* -- JSON 文件取代内存状态，压缩无法抹去它们。
>
> **Harness layer**: Persistence -- 让任务在 context 压缩和进程重启之间存活。



## Problem

s03 的 TodoWrite 将计划存在内存中。Context 压缩（s06）一触发，todo 列表就消失了。多步骤的大任务，特别是那些跨越多轮对话的任务，需要一种不会随 messages 数组一起被清除的持久化机制。



## Solution

```
.tasks/
  1.json   { id:1, subject:"...", status:"pending",   blockedBy:[] }
  2.json   { id:2, subject:"...", status:"in_progress", blockedBy:[1] }
  3.json   { id:3, subject:"...", status:"completed",  blockedBy:[] }

task_list output:
  [ ] 1: Write tests
  [>] 2: Implement feature  (blocked by: 1)
  [x] 3: Update docs
```

每个任务是磁盘上的一个 JSON 文件。Context 清空了，任务还在。



## How It Works

1. 任务 schema：id、subject、description、status 和 blockedBy 依赖列表。

```python
# task JSON schema
{
    "id":          int,
    "subject":     str,
    "description": str,
    "status":      "pending" | "in_progress" | "completed",
    "blockedBy":   [int, ...],   # task IDs this task is waiting on
    "owner":       str,
}
```

2. TaskManager 提供 CRUD，并在完成时自动清理依赖。

```python
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._counter = self._init_counter()

    def create(self, subject: str, description: str) -> str:
        task_id = self._counter
        self._counter += 1
        task = {"id": task_id, "subject": subject,
                "description": description, "status": "pending",
                "blockedBy": [], "owner": ""}
        self._save(task)
        return f"Created task {task_id}: {subject}"

    def update(self, task_id: int, status: str,
               add_blocked_by: list = None,
               remove_blocked_by: list = None) -> str:
        task = self._load(task_id)
        task["status"] = status
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"]) | set(add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [x for x in task["blockedBy"]
                                  if x not in remove_blocked_by]
        self._save(task)
        if status == "completed":
            self._clear_dependency(task_id)   # unblock downstream tasks
        return f"Updated task {task_id} -> {status}"

    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        for f in self.dir.glob("*.json"):
            t = json.loads(f.read_text())
            if completed_id in t.get("blockedBy", []):
                t["blockedBy"].remove(completed_id)
                f.write_text(json.dumps(t, indent=2))
```

3. 四个 task tool 加入 dispatch map。

```python
TOOL_HANDLERS = {
    # base tools unchanged from s02
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # task tools
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw["description"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw["status"],
                                              kw.get("add_blocked_by"),
                                              kw.get("remove_blocked_by")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}
```

任务文件永远留在磁盘上。Context 压缩、进程重启，model 仍可通过 `task_list` 重新定向。



## What Changed From s06

| Component      | Before (s06)        | After (s07)                  |
|----------------|---------------------|------------------------------|
| Tools          | 5 (base + compact)  | 8 (base + 4 task ops)        |
| Task state     | In-memory (s03 todo)| JSON files in `.tasks/`      |
| Dependencies   | None                | blockedBy list + auto-clear  |
| Persistence    | Context only        | Survives compression/restart |



## Try It

```sh
cd learn-claude-code
python agents/s07_task_system.py
```

1. `Create tasks to refactor hello.py: 1) add type hints, 2) add docstrings (blocked by 1), 3) write tests (blocked by 2)`
2. `List all tasks`
3. `Mark task 1 as completed and show what unblocked`
4. `What tasks are still pending?`（kill 进程重启后再试，任务仍然存在）




# s08: Background Tasks

> *"Fire and forget -- the agent doesn't block while the command runs"* -- daemon thread 执行命令，通知队列在下次 LLM 调用前注入结果。
>
> **Harness layer**: Async execution -- 让 model 在等待慢命令时保持思考。



## Problem

测试套件、构建、`npm install`、依赖扫描 -- 这些命令运行需要数十秒甚至数分钟。用同步 `bash` 调用，agent 在 subprocess 结束前什么都做不了。整个会话卡住等待一个 I/O 操作。



## Solution

```
                  background_run("pytest tests/")
                  |
agent_loop        |   BackgroundManager
+-----------+     v   +--------------------+
|           | ------> | task_id = uuid()   |
| call LLM  |         | start daemon thread|
|           | <------ | return task_id     |
|  continue |         +--------------------+
|  thinking |                 |
|           |         thread executes...
| next turn |                 |
|  [drain]  | <-- notification pushed to queue
| <bg:abc>  |
| completed |
+-----------+
```

Model 立即收到 `task_id`，继续工作；结果在后台准备好后注入。



## How It Works

1. BackgroundManager 用 daemon thread + UUID 管理任务。

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running",
                               "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR,
                               capture_output=True, text=True, timeout=300)
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output, status = "Error: Timeout (300s)", "timeout"
        except Exception as e:
            output, status = f"Error: {e}", "error"

        self.tasks[task_id].update({"status": status, "result": output})
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id, "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def drain_notifications(self) -> list:
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs
```

2. Agent loop 每次调用 LLM 前先 drain 通知，作为 `<background-results>` 注入。

```python
def agent_loop(messages: list):
    while True:
        notifs = BG.drain_notifications()         # check completed tasks
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}"
                for n in notifs
            )
            messages.append({
                "role": "user",
                "content": f"<background-results>\n{notif_text}\n</background-results>",
            })

        response = client.messages.create(...)    # LLM now sees results
        # ... tool execution ...
```

3. 两个新 tool。

```python
TOOL_HANDLERS = {
    # base tools unchanged
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}
```



## What Changed From s07

| Component        | Before (s07)        | After (s08)                     |
|------------------|---------------------|---------------------------------|
| Tools            | 8                   | 6 (base + 2 background tools)   |
| Execution        | Blocking subprocess | Daemon thread + UUID task_id    |
| Result delivery  | Immediate           | Notification queue → injection  |
| Parallelism      | None                | Multiple simultaneous commands  |



## Try It

```sh
cd learn-claude-code
python agents/s08_background_tasks.py
```

1. `Run pytest in the background and continue working while it runs`
2. `Start a background build, then read a file while waiting`
3. `Check the status of all background tasks`
4. `Start three background tasks simultaneously`




# s09: Agent Teams

> *"Teammates that can talk to each other"* -- 持久化 worker + 文件邮箱，让多个 agent 真正协作。
>
> **Harness layer**: Team mailboxes -- 多个 model，通过文件协调。



## Problem

s04 的 subagent 是一次性的：父 agent 发出任务，子 agent 执行完就消失，上下文全部丢弃。它们无法并行工作，无法向彼此发消息，也无法持续等待下一个任务。真正的 team 需要持久化的 worker。



## Solution

```
Lead agent (main thread)
  spawn_teammate("alice", "tester", "Run all tests")
  spawn_teammate("bob",   "writer", "Update docs")
  send_message("alice", "Focus on auth module")
         |
         v
.team/
  config.json       {"members": [{"name":"alice","status":"working"}, ...]}
  inbox/
    alice.jsonl     {"type":"message","from":"lead","content":"..."}
    bob.jsonl       {...}
    lead.jsonl      {"type":"message","from":"alice","content":"done"}

Each teammate runs _teammate_loop() in its own thread.
Communication is async: write to JSONL, read on next iteration.
```



## How It Works

1. MessageBus 将每条消息追加到接收方的 JSONL 文件中；`read_inbox` 读取后清空。

```python
class MessageBus:
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        msg = {"type": msg_type, "from": sender,
               "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = [json.loads(line) for line in
                    inbox_path.read_text().strip().splitlines() if line]
        inbox_path.write_text("")    # drain
        return messages
```

2. TeammateManager 在独立线程中运行每个 teammate 的 agent loop，并将状态持久化到 `config.json`。

```python
class TeammateManager:
    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = {"name": name, "role": role, "status": "working"}
        self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt), daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user",
                                 "content": json.dumps(msg)})
            response = client.messages.create(
                model=MODEL, system=sys_prompt,
                messages=messages, tools=tools, max_tokens=8000,
            )
            # ... tool dispatch, append results ...
            if response.stop_reason != "tool_use":
                break
```

3. Lead loop 每轮先读取自己的 inbox，再调用 LLM。

```python
def agent_loop(messages: list):
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
        response = client.messages.create(...)
```



## What Changed From s08

| Component      | Before (s08)        | After (s09)                         |
|----------------|---------------------|-------------------------------------|
| Tools (lead)   | 6                   | 9 (+ spawn_teammate, list, send, read_inbox, broadcast) |
| Tools (worker) | N/A                 | 6 (base + send_message + read_inbox)|
| Concurrency    | background threads  | Multi-agent, each with own loop     |
| Messaging      | None                | JSONL append-only inboxes           |
| Persistence    | None                | config.json + inbox files           |



## Try It

```sh
cd learn-claude-code
python agents/s09_agent_teams.py
```

1. `Spawn a teammate named alice to find all TODO comments in the codebase`
2. `List teammates and their status`
3. `/inbox` （查看 lead 收到的消息）
4. `Spawn two teammates: one to read files, one to run tests -- coordinate them`




# s10: Team Protocols

> *"Same request_id correlation pattern, two domains"* -- Shutdown FSM 和 Plan Approval FSM，靠 request_id 关联握手。
>
> **Harness layer**: Protocols -- model 之间的结构化握手。



## Problem

没有协议的 team 会产生协调失败：teammate 在 lead 没批准的情况下就开始大规模改动；或者 lead 想让 teammate 停下来，但 teammate 还在跑着，浪费算力。Team 需要两种基础协议：优雅关闭和计划审批。



## Solution

两个 FSM，共用同一个 `request_id` 关联模式：

```
Shutdown Protocol (pending → approved | rejected)
Lead                              Teammate
  shutdown_request  ---------->
  {request_id: abc}               receives request
                                  decides: approve?
  shutdown_response <----------
  {request_id: abc,               shutdown_response
   approve: true}                 {request_id: abc, approve: true}
                                  |
                                  v
                              status -> "shutdown", thread stops

Plan Approval Protocol (pending → approved | rejected)
Teammate                          Lead
  plan_approval     ---------->
  {plan: "..."}                   reviews plan text
                                  approve/reject?
  plan_approval_resp <---------
  {request_id: xyz,               plan_approval
   approve: true}                 {req_id: xyz, approve: true}
```



## How It Works

1. 两个 tracker dict 按 `request_id` 跟踪待处理的握手请求，由 `_tracker_lock` 保护。

```python
shutdown_requests = {}   # {req_id: {"target": name, "status": "pending|approved|rejected"}}
plan_requests    = {}    # {req_id: {"from": name, "plan": text, "status": "pending|..."}}
_tracker_lock    = threading.Lock()
```

2. Lead 发起 shutdown -- 生成 request_id，发消息，等 teammate 回应。

```python
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"
```

3. Teammate 收到 `shutdown_request`，调用 `shutdown_response` tool，线程退出。

```python
# Inside teammate's _exec():
if tool_name == "shutdown_response":
    req_id = args["request_id"]
    approve = args["approve"]
    with _tracker_lock:
        if req_id in shutdown_requests:
            shutdown_requests[req_id]["status"] = \
                "approved" if approve else "rejected"
    BUS.send(sender, "lead", args.get("reason", ""),
             "shutdown_response", {"request_id": req_id, "approve": approve})
    return f"Shutdown {'approved' if approve else 'rejected'}"
# In _teammate_loop: if shutdown_response tool called with approve=True, set should_exit=True
```

4. Plan Approval：teammate 提交 plan，lead 审批。

```python
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"
```



## What Changed From s09

| Component         | Before (s09)  | After (s10)                       |
|-------------------|---------------|-----------------------------------|
| Tools (lead)      | 9             | 12 (+ shutdown_request, shutdown_response, plan_approval) |
| Message types     | message, broadcast | + shutdown_request/response, plan_approval_response |
| Shutdown          | Kill thread   | Graceful FSM with approval        |
| Plan governance   | None          | Submit → review → approve/reject  |
| Correlation       | None          | request_id tracker dicts          |



## Try It

```sh
cd learn-claude-code
python agents/s10_team_protocols.py
```

1. `Spawn alice and ask her to shut down gracefully when done`
2. `Spawn bob and have him submit a plan before doing any file edits`
3. `/team` （查看成员状态）
4. `Request alice to shut down, then check shutdown status`




# s11: Autonomous Agents

> *"The agent finds work itself"* -- idle cycle 扫描任务板，自动认领未分配任务。
>
> **Harness layer**: Autonomy -- 让 model 无需显式指令就能发现并承担工作。



## Problem

s09/s10 的 teammate 需要 lead 显式分配任务。如果 lead 正在忙、或者 context 被压缩了，任务就堆积起来。真正自主的 team 应该让空闲的 teammate 自己去找工作，而不是等待指令。



## Solution

```
Teammate lifecycle:

  [spawn]
     |
     v
+-----------+   tool_use?  +-----------+
|   WORK    | -----------> |  continue |
|  phase    | <----------- |    loop   |
+-----------+  tool result +-----------+
     |
     | stop_reason != tool_use  OR  idle tool called
     v
+------------------+   timeout / claimed task
|   IDLE phase     | --------------------------> [WORK again]
|  poll every 5s:  |
|  1. read inbox   |   no work found, inbox empty
|  2. scan .tasks/ | --------------------------> [shutdown]
|  3. claim task   |
+------------------+
```



## How It Works

1. Identity re-injection 防止压缩后 teammate 忘记自己是谁。

```python
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": (f"<identity>You are '{name}', role: {role}, "
                    f"team: {team_name}. Continue your work.</identity>"),
    }
```

2. `_loop` 方法实现 WORK → IDLE → WORK 切换。

```python
def _loop(self, name: str, role: str, prompt: str):
    messages = [make_identity_block(name, role, team_name),
                {"role": "user", "content": prompt}]
    tools = self._teammate_tools()

    for _ in range(50):            # WORK phase
        inbox = BUS.read_inbox(name)
        for msg in inbox:
            messages.append({"role": "user", "content": json.dumps(msg)})

        response = client.messages.create(
            model=MODEL, system=sys_prompt,
            messages=messages, tools=tools, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # IDLE phase: poll for new work
            for _ in range(12):          # up to 60s (12 × 5s)
                time.sleep(5)
                inbox = BUS.read_inbox(name)
                unclaimed = self._find_unclaimed_task()
                if inbox or unclaimed:
                    if unclaimed:
                        self._claim_task(name, unclaimed)
                    messages.append(make_identity_block(name, role, team_name))
                    break               # return to WORK phase
            else:
                break                   # no work found, shutdown
            continue

        # ... tool dispatch as before ...
```

3. `_find_unclaimed_task` 扫描 `.tasks/` 找 pending 且 blockedBy 为空的任务。

```python
def _find_unclaimed_task(self) -> dict | None:
    for f in sorted(TASKS_DIR.glob("*.json")):
        task = json.loads(f.read_text())
        if task["status"] == "pending" and not task.get("blockedBy"):
            return task
    return None
```

4. Lead 拥有 14 个 tool，teammate 拥有 9 个（含 `idle` tool 主动触发 IDLE 切换）。

```python
# Teammate-only tool: explicitly enter idle phase
{"name": "idle",
 "description": "Enter idle mode to wait for new tasks.",
 "input_schema": {"type": "object", "properties": {}}}
```



## What Changed From s10

| Component        | Before (s10)       | After (s11)                        |
|------------------|--------------------|------------------------------------|
| Tools (lead)     | 12                 | 14 (+ task management)             |
| Tools (teammate) | 8                  | 9 (+ idle)                         |
| Agent lifecycle  | Start → work → end | WORK ↔ IDLE cycle                  |
| Work discovery   | Lead assigns       | Teammate self-claims from .tasks/  |
| Identity         | Initial prompt     | Re-injected after compression      |



## Try It

```sh
cd learn-claude-code
python agents/s11_autonomous_agents.py
```

1. `Create 5 pending tasks in .tasks/ then spawn 3 teammates -- watch them self-assign`
2. `Spawn a teammate with no specific task and observe idle polling`
3. `Add a new task to .tasks/ while a teammate is in idle mode`
4. `/team` （观察 teammate 状态自动从 idle 变为 working）




# s12: Worktree + Task Isolation

> *"Isolate by directory, coordinate by task ID"* -- Git worktree 提供目录级隔离；EventBus 追踪生命周期。
>
> **Harness layer**: Isolation -- 并行执行，无文件冲突。



## Problem

并行 teammate 在同一个目录里工作时会产生冲突：同时写同一个文件、并发运行 git 命令、测试输出互相覆盖。即使逻辑上任务不相关，文件系统层面也会互相干扰。



## Solution

```
主工作目录 (WORKDIR)
  ├── .tasks/           任务板：JSON 文件，记录 task_id → worktree 绑定
  ├── .events/          EventBus：append-only JSONL 生命周期日志
  └── .worktrees/
        ├── task-001/   Git worktree，绑定 task 1
        │     └── (独立文件系统，独立 git HEAD)
        └── task-002/   Git worktree，绑定 task 2
              └── (独立文件系统，独立 git HEAD)

agent_loop
  worktree_create(name="task-001", task_id=1)
    → git worktree add .worktrees/task-001
    → bind task 1 → task-001
  worktree_run(name="task-001", command="pytest")
    → subprocess.run(cwd=.worktrees/task-001)
  worktree_remove(name="task-001")
    → git worktree remove .worktrees/task-001
    → emit event "removed"
```



## How It Works

1. EventBus 记录所有 worktree 生命周期事件，便于审计和调试。

```python
class EventBus:
    def __init__(self, events_dir: Path):
        self.log = events_dir / "events.jsonl"
        events_dir.mkdir(exist_ok=True)

    def emit(self, event_type: str, name: str, metadata: dict = None):
        event = {
            "type": event_type, "name": name,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        with open(self.log, "a") as f:
            f.write(json.dumps(event) + "\n")

    def list_recent(self, n: int = 20) -> list:
        if not self.log.exists():
            return []
        lines = self.log.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l]
```

2. WorktreeManager 封装 `git worktree add/remove`，并维护 task 绑定。

```python
class WorktreeManager:
    def create(self, name: str, task_id: int = None) -> str:
        wt_path = self.dir / name
        result = self._run_git(f"worktree add {wt_path}")
        if "Error" in result:
            return result
        self.worktrees[name] = {"path": str(wt_path), "task_id": task_id}
        if task_id:
            TASKS.bind_worktree(task_id, name)
        EVENTS.emit("created", name, {"task_id": task_id})
        return f"Created worktree '{name}' at {wt_path}"

    def run(self, name: str, command: str) -> str:
        wt = self.worktrees.get(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        r = subprocess.run(command, shell=True,
                           cwd=wt["path"],
                           capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:50000]

    def remove(self, name: str) -> str:
        wt = self.worktrees.pop(name, None)
        if not wt:
            return f"Error: '{name}' not found"
        self._run_git(f"worktree remove {wt['path']} --force")
        EVENTS.emit("removed", name)
        return f"Removed worktree '{name}'"
```

3. TaskManager 新增 `bind_worktree` / `unbind_worktree`。

```python
class TaskManager:
    def bind_worktree(self, task_id: int, worktree_name: str) -> str:
        task = self._load(task_id)
        task["worktree"] = worktree_name
        self._save(task)
        return f"Bound task {task_id} → worktree '{worktree_name}'"

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task.pop("worktree", None)
        self._save(task)
        return f"Unbound worktree from task {task_id}"
```

4. 共 18 个 tool：4 base + 5 task ops + 7 worktree ops + worktree_events。

```python
TOOL_HANDLERS = {
    # base (4)
    "bash": ..., "read_file": ..., "write_file": ..., "edit_file": ...,
    # task ops (5)
    "task_create": ..., "task_list": ..., "task_get": ...,
    "task_update": ..., "task_bind_worktree": ...,
    # worktree ops (7)
    "worktree_create": ..., "worktree_list": ..., "worktree_status": ...,
    "worktree_run": ..., "worktree_remove": ..., "worktree_keep": ...,
    # observability (1)
    "worktree_events": ...,
}
```



## What Changed From s11

| Component        | Before (s11)        | After (s12)                        |
|------------------|---------------------|------------------------------------|
| Tools            | 14                  | 18 (+ 5 worktree ops + events)     |
| Isolation        | None (shared dir)   | Git worktree per task              |
| Task schema      | id, status, blockedBy | + worktree binding              |
| Observability    | None                | EventBus (append-only JSONL)       |
| File conflicts   | Possible            | Eliminated by directory isolation  |



## Try It

```sh
cd learn-claude-code
python agents/s12_worktree_task_isolation.py
```

1. `Create two tasks and a worktree for each, then run tests in both simultaneously`
2. `Create a worktree bound to task 1, run the tests, then remove it`
3. `Show worktree events to see the lifecycle log`
4. `List all worktrees and their associated task IDs`




# 全局总结：Harness 层演进路线

```
s01  Agent Loop         while True + stop_reason         基础闭环
s02  Tool Use           dispatch map + safe_path          工具扩展
s03  TodoWrite          TodoManager + nag reminder        规划约束
s04  Subagents          独立 messages[] + 摘要返回         上下文隔离
s05  Skills             两层注入 (system + tool_result)   按需知识
s06  Context Compact    三层压缩 + transcript 存档         无限会话
─────────────────────────────────────────────────────────
s07  Task System        JSON 文件持久化 + blockedBy        跨压缩存活
s08  Background Tasks   daemon thread + 通知队列           异步执行
s09  Agent Teams        JSONL 邮箱 + 线程 teammate         多 agent 协作
s10  Team Protocols     request_id FSM (shutdown + plan)  结构化握手
s11  Autonomous Agents  WORK↔IDLE + 自动认领               自主工作发现
s12  Worktree Isolation git worktree + EventBus            目录级隔离
```

每一层都在前一层之上添加一个精确的机制，loop 的核心结构从未改变。
