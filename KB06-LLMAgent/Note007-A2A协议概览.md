## A2A 简要介绍

A2A 协议是 Google 在今年 4 月 9 日对外发布的一个协议，全称是 Agent to Agent 协议，这个协议规定了 Agent 与 Agent 之间的沟通规范

<div align="center">
    <img src="https://i-blog.csdnimg.cn/direct/c8096462188141d5bfff4a58b3c6cad7.png#pic_center" style="zoom:60%;" />
</div>




## A2A 协议使用场景

假设你住在西雅图，你想在未来三天里面挑一天飞往纽约，不仅如此，你还有一个额外的要求，因为你喜欢好天气，所以呢你希望出发的那天西雅图阳光明媚

如果你生活在一个没有大模型，没有 Agent 的时代，你需要自己来做以下三件事情：

1. 查找西雅图未来三天的天气预报
2. 根据天气预报，从这三天里面选择西雅图天气最好的一天
3. 查询那一天从西雅图到纽约的机票信息


不过好在你生活在一个有着大模型的时代，你发现有个系统正好可以帮你做这几件事情：

<div align="center">
    <img src="https://i-blog.csdnimg.cn/direct/dfc2ea2b3ae048318a2604804d94fd25.png#pic_center" style="zoom:60%;" />
</div>

这个系统内部部署了三个 Agent，分别是调度 Agent、天气 Agent 和机票 Agent，这个系统可以一次性帮你完成之前的三个任务

----


那现在有个问题，A2A 协议作用在这个链路中的哪一部分呢？🤔

<div align="center">
    <img src="https://i-blog.csdnimg.cn/direct/93449cc000e443c48a735b53a7a0269e.png#pic_center" style="zoom:60%;" />
</div>

没错，就是 Agent 与 Agent 之间的交互部分，简单来说，只要两个 Agent 之间需要沟通，它们就需要使用 A2A 协议，这个就是 A2A 协议的使用场景了

----


天气查询 Agent 内部肯定有一个大模型，但是除了大模型之外，它可能还部署了多个 MCP 工具，有些负责天气预告，有些负责历史气象分析

<div align="center">
    <img src="https://i-blog.csdnimg.cn/direct/bc836166c20349ac8fc88b068c95dd2c.png#pic_center" style="zoom:60%;" />
</div>

所以总结一下，**A2A 协议是作用在 Agent 与 Agent 之间的，MCP 协议是作用在 Agent 内部的，它们的作用域不同**，这个呢就是这两个协议的最大区别了  



## A2A 协议开发案例

下面用一个简单的例子来演示 A2A 协议在实际开发中是怎么运作的。我们沿用前面的场景：**调度 Agent 向天气 Agent 查询天气信息**

整个流程分为三步：
1. 天气 Agent 发布自己的 **Agent Card**（相当于名片），告诉别人"我能做什么"
2. 调度 Agent **发现**天气 Agent（通过读取 Agent Card）
3. 调度 Agent 通过 A2A 协议向天气 Agent **发送任务**，天气 Agent 返回结果

---

### 第一步：天气 Agent 发布 Agent Card

每个 A2A Agent 都需要在 `/.well-known/agent.json` 路径下发布一个 JSON 格式的 Agent Card，描述自己的能力：

```json
{
  "name": "天气查询 Agent",
  "description": "查询指定城市未来几天的天气预报",
  "version": "1.0.0",
  "url": "http://localhost:5001/a2a",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "capabilities": {
    "streaming": false
  },
  "skills": [
    {
      "id": "weather_forecast",
      "name": "天气预报查询",
      "description": "输入城市名称，返回未来三天的天气预报",
      "tags": ["天气", "预报"],
      "examples": ["查询西雅图未来三天的天气", "北京明天天气怎么样"]
    }
  ]
}
```

这个 Agent Card 就像一张名片，其他 Agent 通过它就能知道：这个 Agent 叫什么、能做什么、怎么联系它

---

### 第二步：天气 Agent 服务端实现

天气 Agent 内部有一个 LLM 负责理解用户意图，还有一个天气查询工具（对应前面提到的 MCP 工具）负责获取真实数据。LLM 决定"要不要调用工具、怎么调用"，工具负责"拿数据"，最后 LLM 再把结果组织成自然语言返回

下面是用 Python + Flask + OpenAI 实现的版本：

```python
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量

# ========== 工具定义（相当于 Agent 内部的 MCP 工具）==========

WEATHER_DATA = {
    "西雅图": [
        {"date": "2026-04-12", "weather": "多云", "temp": "12°C"},
        {"date": "2026-04-13", "weather": "晴天", "temp": "18°C"},
        {"date": "2026-04-14", "weather": "小雨", "temp": "10°C"},
    ]
}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "查询指定城市未来三天的天气预报",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
}]

def get_weather_forecast(city: str) -> str:
    """工具函数：查询天气数据"""
    if city in WEATHER_DATA:
        return json.dumps(WEATHER_DATA[city], ensure_ascii=False)
    return json.dumps({"error": f"暂不支持 {city} 的天气查询"}, ensure_ascii=False)

# ========== Agent 核心逻辑：LLM + 工具调用 ==========

def run_agent(user_text: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个天气查询助手，使用工具查询天气后用简洁友好的方式回复。"},
        {"role": "user", "content": user_text}
    ]

    # 第一次调用 LLM：让它决定是否需要调用工具
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )
    msg = response.choices[0].message

    # 如果 LLM 决定调用工具
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = get_weather_forecast(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 第二次调用 LLM：让它根据工具返回的数据生成最终回复
        final = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return final.choices[0].message.content

    return msg.content

# ========== A2A 协议层 ==========

@app.route("/.well-known/agent.json")
def agent_card():
    return jsonify({
        "name": "天气查询 Agent",
        "description": "基于 LLM 的智能天气查询助手",
        "version": "1.0.0",
        "url": "http://localhost:5001/a2a",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "capabilities": {"streaming": False},
        "skills": [{
            "id": "weather_forecast",
            "name": "天气预报查询",
            "description": "输入城市名称，返回未来三天的天气预报",
            "tags": ["天气", "预报"],
            "examples": ["查询西雅图未来三天的天气"]
        }]
    })

@app.route("/a2a", methods=["POST"])
def a2a_endpoint():
    req = request.json

    if req.get("method") == "message/send":
        user_text = req["params"]["message"]["parts"][0]["text"]
        result_text = run_agent(user_text)  # 调用 LLM Agent

        return jsonify({
            "jsonrpc": "2.0",
            "id": req.get("id"),
            "result": {
                "kind": "task",
                "id": "task-001",
                "status": {"state": "completed"},
                "artifacts": [{
                    "artifactId": "weather-result",
                    "parts": [{"kind": "text", "text": result_text}]
                }]
            }
        })

    return jsonify({"jsonrpc": "2.0", "id": req.get("id"),
                     "error": {"code": -32601, "message": "Method not found"}})

if __name__ == "__main__":
    app.run(port=5001)
```

可以看到，天气 Agent 内部的处理流程是这样的：

```
A2A 请求进来
    → LLM 第一次调用：理解用户意图，决定调用 get_weather_forecast 工具
    → 工具执行：查询天气数据
    → LLM 第二次调用：把原始数据组织成自然语言回复
    → 通过 A2A 协议返回结果
```

这正好对应了前面说的：**A2A 协议管 Agent 之间的通信，MCP/工具管 Agent 内部的能力调用**

---

### 第三步：调度 Agent 发现并调用天气 Agent

调度 Agent 作为 A2A 客户端，先读取 Agent Card 了解天气 Agent 的能力，然后发送任务：

```python
import requests

# 1. 发现：获取天气 Agent 的 Agent Card
card = requests.get("http://localhost:5001/.well-known/agent.json").json()
print(f"发现 Agent: {card['name']}")
print(f"技能: {card['skills'][0]['name']}")
a2a_url = card["url"]  # 拿到 A2A 端点地址

# 2. 调用：通过 A2A 协议发送 JSON-RPC 请求
response = requests.post(a2a_url, json={
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "查询西雅图未来三天的天气"}],
            "messageId": "msg-001"
        }
    }
})

# 3. 解析结果
result = response.json()["result"]
weather_text = result["artifacts"][0]["parts"][0]["text"]
print(f"\n天气 Agent 返回:\n{weather_text}")
```

运行后输出：

```
发现 Agent: 天气查询 Agent
技能: 天气预报查询

天气 Agent 返回:
西雅图未来三天天气：
  2026-04-12: 多云 12°C
  2026-04-13: 晴天 ☀️ 18°C
  2026-04-14: 小雨 10°C
```

---

### 补充：多轮对话与上下文管理

前面的例子是"一问一答"的单轮交互，但真实场景中 Agent 之间经常需要多轮沟通。比如调度 Agent 先问"西雅图天气怎么样"，天气 Agent 回答后，调度 Agent 接着问"那哪天最适合出行"——这时天气 Agent 需要记住之前聊过什么

A2A 协议通过两个 ID 来管理上下文：

- **`contextId`**：代表一次完整的对话会话，多轮交互共享同一个 contextId
- **`taskId`**：代表一个具体的任务，一个 context 下可以有多个 task

此外，A2A 还定义了一个关键的任务状态 **`input-required`**，表示"我还需要更多信息才能完成任务"，这就是多轮对话的驱动机制

改造天气 Agent 来支持多轮对话，关键实现点如下：

**服务端（天气 Agent）：**

1. 在内存里维护一个 `context_store` 字典，以 `contextId` 为 key、LLM 对话历史（`messages` 列表）为 value，每个会话独立存储
2. 处理 `message/send` 请求时，从 `params.message.contextId` 取出会话 ID；如果客户端没传，就生成一个新的 UUID（表示首轮对话）
3. 根据 `contextId` 拿到对应的历史 `messages`，把本轮用户输入追加进去，再丢给 LLM（带上 tools），让模型在完整上下文里推理
4. LLM 的最终回复也追加到 `messages` 里，下一轮自然就能看到之前聊过什么
5. 响应里**把 `contextId` 原样返回**，告诉客户端"下次想继续聊就带上这个 ID"

**客户端（调度 Agent）：**

1. 首轮请求的 `message` 里**不带** `contextId`，服务端会生成一个并在响应里返回
2. 从 `resp["result"]["contextId"]` 取出会话 ID 保存起来
3. 后续每一轮的 `message` 里都把这个 `contextId` 带上，服务端就会复用同一份对话历史

运行效果：

```
西雅图未来三天天气：
  2026-04-12: 多云 12°C
  2026-04-13: 晴天 18°C
  2026-04-14: 小雨 10°C

根据天气情况，4月13日最适合出行，当天是晴天，气温18°C，非常舒适！
```

第二轮提问"这三天里哪天最适合出行"并没有提到"西雅图"，但天气 Agent 依然能正确回答——因为 `contextId` 把两轮对话关联在了一起，LLM 能看到完整的对话历史

总结一下 A2A 多轮对话的机制：

```
第一轮请求（无 contextId）
    → 天气 Agent 生成新的 contextId，创建对话历史
    → 返回结果 + contextId

第二轮请求（带上 contextId）
    → 天气 Agent 根据 contextId 找到之前的对话历史
    → LLM 看到完整上下文，理解"这三天"指的是西雅图的三天
    → 返回结果
```

---

### 小结

通过这个例子可以看到 A2A 协议的核心交互模式：

| 步骤 | 做了什么 | 对应 A2A 概念 |
|------|---------|-------------|
| 发布名片 | 天气 Agent 在 `/.well-known/agent.json` 暴露自身能力 | **Agent Card** |
| 发现对方 | 调度 Agent 读取 Agent Card，知道对方能做什么 | **Agent 发现** |
| 发送任务 | 调度 Agent 向天气 Agent 发送 JSON-RPC 请求 | **message/send** |
| 返回结果 | 天气 Agent 返回包含天气信息的 artifact | **Task + Artifact** |
| 多轮对话 | 通过 contextId 关联多次请求，Agent 保持对话上下文 | **contextId + 对话历史** |

这就是 A2A 协议的核心工作流程。在真实场景中，还可以使用**流式响应**（SSE）、**input-required 状态**（Agent 主动要求补充信息）、**推送通知**（Webhook）等高级特性，但核心思路都是一样的：**Agent Card 发现 → JSON-RPC 通信 → Task 管理 → contextId 维持上下文**



## A2A 相关的框架和 SDK

前面的开发案例是用 Flask 手写的，目的是让你看清 A2A 协议底层在做什么。但在实际项目中，不需要自己处理 JSON-RPC 解析、Task 状态管理这些细节，已经有成熟的 SDK 和框架可以用了

### 1. Google 官方 SDK：a2a-python

这是 Google 官方维护的 Python SDK，也是目前最核心的选择

- **仓库**：[google/a2a-python](https://github.com/google/a2a-python)
- **安装**：`pip install a2a-sdk`（需要 Python 3.10+）
- **当前版本**：v1.0.0-alpha（2026 年 3 月发布，支持协议 v1.0 和 v0.3）
- **支持的传输协议**：JSON-RPC、HTTP+JSON/REST、gRPC
- **可选集成**：FastAPI、Starlette、PostgreSQL、SQLite、OpenTelemetry 等

用这个 SDK，前面手写的天气 Agent 可以大幅简化，Agent Card 生成、JSON-RPC 路由、Task 生命周期管理都由 SDK 处理

### 2. Google ADK（Agent Development Kit）

ADK 是 Google 的 Agent 开发框架，内置了 A2A 支持，一行代码就能把 Agent 暴露为 A2A 服务

- **仓库**：[google/adk-python](https://github.com/google/adk-python)
- **核心能力**：用 `agent.to_a2a()` 一行代码把任意 ADK Agent 包装成 A2A 服务端
- **CLI 支持**：`adk api_server --a2a` 命令直接启动 A2A 服务
- **自动生成**：Agent Card、JSON-RPC 端点、SSE 流式响应都自动处理
- **多语言**：Python、Java、Go 都已支持

如果你是从零开始构建 Agent，ADK 是最省事的选择

### 3. 主流 Agent 框架的 A2A 支持

A2A 是一个通信协议，不是框架。主流的 Agent 编排框架也在逐步集成 A2A 支持：

| 框架 | A2A 支持 | 集成方式 | 适合场景 |
|------|---------|---------|---------|
| **LangGraph** | 原生支持 | 自动为每个 assistant 生成 `/a2a/` 端点 | 复杂的有状态工作流，需要持久化和容错 |
| **CrewAI** | 原生支持 | 提供 `A2AServerConfig` / `A2AClientConfig` | 基于角色的团队协作，上手简单 |
| **AutoGen** | 暂不支持 | 正在向 Microsoft Agent Framework 过渡 | 微软生态 |

### 怎么选？

- **想理解原理** → 像前面的例子一样手写，用 Flask + OpenAI 直接对接 A2A 协议
- **想快速开发** → 用 Google ADK，`to_a2a()` 一行搞定
- **需要精细控制** → 用 `a2a-python` SDK，灵活但比 ADK 底层一些
- **已有 LangGraph/CrewAI 项目** → 直接用框架自带的 A2A 集成

















