# LangChain 源码阅读笔记

Step 2 写 LangGraph 版时，对照 Step 1 的原生实现，读源码、跑小实验得到的发现。

> 基于的版本：`langchain-core 1.7.7`、`langchain-openai 1.7.4`、`langgraph 1.2.20`、`openai 3.46.0`、`pydantic 2.14.3`。
> 升级依赖后，文中提到的内部函数名和行为可能会变，以当时的源码为准。

## 1. `ChatOpenAI`：适配器，不只是封装

```python
llm = ChatOpenAI(model=..., api_key=..., base_url=..., reasoning_effort="low")
llm_with_tools = llm.bind_tools(TOOLS)      # 返回新对象，原 llm 不变
r = llm_with_tools.invoke(messages)         # 返回 AIMessage
```

- 继承链：`ChatOpenAI → BaseChatOpenAI → BaseChatModel → BaseLanguageModel`。
- `BaseChatModel` 是 LangChain 给所有聊天模型定的统一接口（类似 JDBC 的 `Connection`）：输入消息列表，输出 `AIMessage`，工具调用在 `.tool_calls`，token 在 `.usage_metadata`。换模型厂商只改创建 `llm` 的那一行。
- 内部持有一个 `openai.OpenAI` 客户端：`llm.root_client`。

| | Step 1（openai SDK） | Step 2（ChatOpenAI） |
| --- | --- | --- |
| 工具描述 | 手写 `TOOLS` JSON Schema | `bind_tools([...])` 自动生成 |
| 调用 | `client.chat.completions.create(model=..., messages=..., tools=...)` | `llm_with_tools.invoke(messages)` |
| 返回 | `resp.choices[0].message` | `AIMessage` |
| 工具参数 | `call.function.arguments` 是 JSON 字符串，要自己 `json.loads` | `r.tool_calls[i]["args"]` 已经是 dict |
| token | `resp.usage.total_tokens` | `r.usage_metadata["total_tokens"]` |

### 它替你做了、但看不到的事

- **丢掉了 GLM 的 `reasoning_content`**：`AIMessage.additional_kwargs` 里只有 `refusal`。按 OpenAI 标准格式解析，不认识的厂商扩展字段被忽略。
- **重试和超时**：`max_retries`、`request_timeout` 默认是 `None`，沿用 openai SDK 的默认值，失败时可能悄悄重试。Step 5 要显式设置。
- **解析失败的工具调用**放进 `AIMessage.invalid_tool_calls`，不会出现在 `tool_calls` 里。只看 `tool_calls` 会漏掉。
- **双向转换**：收到响应时把 `arguments` 字符串 `json.loads` 成 `args` dict；发请求时再 `json.dumps` 回去。

### 怎么看盒子里面

- `r.response_metadata`：原始响应的一部分（`finish_reason`、`model_name`、`token_usage`）。
- `from langchain_core.globals import set_debug; set_debug(True)`：打印每次调用的输入输出。
- `llm.root_client`：框架不支持的功能，直接用底层客户端。
- PyCharm 里 Ctrl+左键跳进源码。

## 2. 消息类：`BaseMessage` 继承体系

| 类 | 对应的 `role` | 比 `BaseMessage` 多的字段 |
| --- | --- | --- |
| `SystemMessage` | `system` | 无 |
| `HumanMessage` | `user` | 无 |
| `AIMessage` | `assistant` | `tool_calls`、`invalid_tool_calls`、`usage_metadata` |
| `ToolMessage` | `tool` | `tool_call_id`、`status`、`artifact` |

`BaseMessage` 的公共字段：`content`、`type`、`id`、`name`、`additional_kwargs`、`response_metadata`。

- **role 由类表示**，不是字典里的字段。`ChatOpenAI` 发请求时才转换成 `{"role": ..., "content": ...}`，和 Step 1 手写的格式完全一样（转换函数：`langchain_openai.chat_models.base._convert_message_to_dict`）。
- **公共字段放基类，特有字段放子类**：`HumanMessage` 上根本设不了 `tool_calls`。Step 1 的字典没有这种约束。
- **`type` 字段（`"ai"`、`"tool"` 等）用于序列化**：存进数据库后类信息丢失，读回时靠 `type` 恢复成对应的类（可辨识联合）。Step 5 的 SQLite checkpointer 会用到。
- **多态**：State 里声明 `messages: Annotated[list[BaseMessage], add_messages]`，同一个列表放四种消息。
- **流式输出**：`AIMessageChunk` 继承 `AIMessage` 和 `BaseMessageChunk`，多个 chunk 用 `+` 拼成完整消息。

### 打印消息

`print(r)` 输出的是 pydantic 的 `__str__`：`字段=值` 用空格分隔，值是 Python 字典格式（单引号、`None`），不是 JSON。

| 方法 | 用途 |
| --- | --- |
| `r.pretty_print()` | 给人看，显示内容和工具调用，不显示 token |
| `r.model_dump_json(indent=2, include={"content", "tool_calls", "usage_metadata"})` | 真正的 JSON，可选字段 |
| `r.content`、`r.tool_calls`、`r.usage_metadata` | 代码里直接取 |
| `repr(r)` / REPL 里直接输入 `r` | 带类名：`AIMessage(content=...)` |

`response_metadata["token_usage"]` 是 OpenAI 原始格式，`usage_metadata` 是 LangChain 统一格式，用后者。

## 3. `@tool`：函数变成 `StructuredTool` 对象

```python
@tool(parse_docstring=True)
def list_assignments(course: str | None = None) -> list[dict]:
    """列出未提交的作业，可按课程过滤

    Args:
        course: 课程名，可省略
    """
    return tools.list_assignments(course)
```

- 装饰后 `list_assignments` 是 `StructuredTool` 对象，**不能直接 `()` 调用**（`TypeError: 'StructuredTool' object is not callable`），要用 `.invoke({...})`。
- 用 `.name`、`.description`、`.args` 查看框架生成的 schema。框架替你生成了东西，打印出来确认，不要假设。
- 函数名 → 工具名；docstring → 描述；类型标注 → 参数 schema（`str | None` 生成 `anyOf: [string, null]`）；没有默认值的参数自动 `required`。

### docstring 的坑

- **默认不解析 docstring**：整段原样作为工具描述，参数描述为空。
- **`parse_docstring=True` 只认 Google 风格**（`Args:`）。PyCharm 默认的 reST 风格（`:param x:`）会报 `ValueError: Found invalid Google-Style docstring.`。可以在 Settings → Tools → Python Integrated Tools → Docstring format 改成 Google。
- 格式要求：摘要后空一行，再写 `Args:`，参数缩进。
- **`Returns:` 不会进入工具描述**，模型看不到。想让模型知道返回值格式，写在 `Args:` 之前的描述段落里。
- 替代写法：`course: Annotated[str | None, "课程名，可省略"] = None`，不依赖 docstring 格式。
- docstring 现在是**发给模型的内容**：写得和实际行为不一致（比如说"按课程分组"但实际是平铺列表），会误导模型。

### 类型标注变成了运行时校验

`.invoke()` 在调用函数**之前**用 pydantic 按类型标注校验参数。`deadline: str` 时传 `None` 会直接抛 `ValidationError`，函数里处理 `None` 的代码根本执行不到。类型标注要和实际行为一致（这里应为 `str | None`）。Step 1 里类型标注只是提示，这里是契约。

## 4. `tool.invoke()`：一个函数，按输入形状分支

Python 没有方法重载（同名 `def` 会覆盖前一个）。`BaseTool.invoke` 只有一个实现，参数类型是 `str | dict[str, Any] | ToolCall`，在运行时判断：

```python
# langchain_core/tools/base.py
def _is_tool_call(x: Any) -> bool:
    return isinstance(x, dict) and x.get("type") == "tool_call"
```

| 传入 | 返回 |
| --- | --- |
| `call["args"]`（只有参数） | 函数的原始返回值（dict / list / str） |
| `call`（整个 tool_call，带 `id` 和 `type: "tool_call"`） | `ToolMessage`：结果已序列化成字符串放进 `content`，`tool_call_id` 已配对 |

传整个 `call` 等于把 Step 1 的这几步交给了框架：

```python
result = fn(**json.loads(call.function.arguments))
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(result, ensure_ascii=False)})
```

### tools 节点的写法

```python
TOOLS = [list_assignments, get_assignment, days_until]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}   # 和 bind_tools(TOOLS) 用同一个列表，不会写错名字

for call in ai_message.tool_calls:
    results.append(TOOLS_BY_NAME[call["name"]].invoke(call))
```

仍然需要错误处理：参数不合法抛 `ValidationError`，工具名不存在抛 `KeyError`。出错时自己构造 `ToolMessage(content=错误信息, tool_call_id=call["id"], status="error")` 返回给模型。

## 5. `@tool` 和 `@mcp.tool()` 的区别

| | `@tool`（LangChain） | `@mcp.tool()`（MCP） |
| --- | --- | --- |
| 产物 | `StructuredTool` 对象 | 把函数注册到 `mcp` 这个 server 上 |
| 谁能用 | 同一进程里的 LangChain / LangGraph agent | 任何 MCP 客户端（自己的 agent、Claude Code、其他语言） |
| 调用方式 | 进程内调用 | 通过 MCP 协议（stdio / HTTP）跨进程 |
| 对应 Step | 2、3 | 4 |

两者读取的信息相同（函数名、docstring、类型标注）。Step 4 里 `langchain-mcp-adapters` 会把 MCP 工具再转换成 LangChain 的 `Tool`。`tools.py` 保持纯函数、不依赖框架，就是为了能被不同的框架包装。

## 6. `Annotated[...]`：类型标注 + 额外信息

```python
from typing import Annotated
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```
`Annotated[类型, 额外信息]` 的第一个参数是类型，后面的参数是附加信息。**Python 本身完全忽略这些附加信息**，由读取它的框架决定含义：

- **LangGraph：声明 reducer。** 建图时，LangGraph 读出 `add_messages`，把它当作这个字段的合并函数。
- **LangChain `@tool`：给参数加描述。** 例如 `course: Annotated[str | None, "课程名，可省略"] = None`，这种写法不依赖 docstring。

### `add_messages` 做了什么

节点返回更新后，LangGraph 执行 `新列表 = add_messages(旧列表, 返回的列表)`。它做的不只是追加：

| 行为 | 实验 |
| --- | --- |
| 把元组或字典转成消息对象 | `add_messages([], [("user", "hi")])` 得到 `HumanMessage` |
| 自动补上 `id` | 上面得到的 `HumanMessage.id` 不为空 |
| `id` 相同时替换，而不是追加 | 先加 `AIMessage("v1", id="x")`，再加 `AIMessage("v2", id="x")`，列表里只剩 `v2` |

所以调用图时可以直接传 `{"messages": [("user", question)]}`。

如果字段没有 reducer，比如 `count: int`，节点返回的新值会直接**覆盖**旧值。

## 7. LangGraph 节点：返回部分更新

```python
def chat(state: State) -> dict:
    r = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [r]}
```

- **节点不直接修改 state，而是返回"要更新哪些字段"**，由图通过 reducer 合并。`return {"messages": [r]}` 相当于 Step 1 里的 `messages.append(assistant)`。
- **`[r]` 要用列表包起来**：reducer 合并的是消息列表。tools 节点一次可以返回多条 `ToolMessage`。
- **忘了 `return` 不会报错**：函数返回 `None`，LangGraph 把它当作"不更新任何字段"，图照常运行结束，只是这一步的结果丢了。实验结果：
  - 有 `return`：`[SystemMessage, HumanMessage, AIMessage]`
  - 没有 `return`：`[SystemMessage, HumanMessage]`
- **返回类型标 `dict`，不标 `State`**：节点返回的只是部分字段，不是完整的 State。
- **system prompt 在节点里拼接，不存进 state**：调用方只需要传用户问题。以后把 state 存进数据库（checkpointer）时，不会每轮都存一份重复的 system prompt。

### 为什么不能直接修改 state

```python
def mutate(state):
    state["messages"].append(AIMessage("mutated"))   # 修改对象
    state["count"] = 99                               # 重新赋值
```

实验结果：

| 现象 | 原因 |
| --- | --- |
| 下一个节点**看到了** `mutated` | 节点拿到的 list 和图内部存的是同一个对象（引用），`append` 直接改到了图的内部数据 |
| `count` 仍然是 0 | 节点拿到的 state 字典只是一份副本，给它的 key 重新赋值不影响图 |
| `stream_mode="updates"` 输出 `{'a': None}` | 图不知道这一步发生了变化 |
| 新消息的 `id` 是 `None` | 没有经过 `add_messages`，也就没有转换和补 id |

能不能生效，取决于你碰巧用的是"修改对象"还是"重新赋值"（和 Java 里修改参数对象的字段、给参数重新赋值的区别一样）。即使碰巧生效，也绕过了 reducer、stream 和 checkpointer。

**并行写入**：同一步里两个节点都写 `messages`，reducer 会把两份都合并进去（`['hi', 'from p1', 'from p2']`）。都写没有 reducer 的 `count`，会报 `InvalidUpdateError: Can receive only one value per step`。如果改成大家直接修改同一个对象，就成了多线程共享可变状态，结果取决于执行顺序。

**类比 React / Redux**：不能写 `state.x = 1`，要用 `setState({x: 1})`；reducer 是 `(旧值, 更新) => 新值`；直接修改 state 不会触发刷新，DevTools 也回放不了。用 Java 的话说接近事件溯源：节点只提交"发生了什么"，由框架负责应用。

**结论：节点把 state 当作只读，返回值是节点和运行时之间唯一的接口。**

## 8. 运行时：节点的返回值给谁用

`graph_builder.compile()` 生成 `CompiledStateGraph`，底层叫 Pregel。节点的返回值交给它处理。

### channel：State 每个字段对应的存储格子

| State 字段 | channel 类型 | 收到新值时 |
| --- | --- | --- |
| `messages: Annotated[list, add_messages]` | `BinaryOperatorAggregate` | `self.value = self.operator(self.value, value)`，其中 `operator` 就是 `add_messages` |
| `count: int`（没有 reducer） | `LastValue` | 同一步只能收到一个值，否则抛 `InvalidUpdateError`；然后用新值覆盖旧值 |

相关源码在 `.venv/lib/python3.12/site-packages/langgraph/` 下：

- `pregel/_algo.py` 的 `apply_writes`：把本步所有节点的返回值按 key 分组，调用 `channels[chan].update(vals)`。
- `channels/binop.py`、`channels/last_value.py`：上表中两种 channel 的 `update` 实现。

### 一个 super-step 的流程

```
invoke(input)
  │
  ▼  ┌──────────────────── 一个 super-step ────────────────────────┐
     │ 1. 读 channel → 组装 state → 传给本步要运行的节点          │
     │ 2. 节点运行，返回 {"messages": [...]}  ← 我们写的代码只管这里 │
     │ 3. apply_writes：按 key 调用 channel.update（reducer）    │
     │ 4. 发出 stream 事件、保存 checkpoint                        │
     │ 5. 执行条件边 / 普通边，决定下一步运行哪些节点               │
     └────────────────────────────────────────────────────────────┘
  │  重复，直到走到 END，或者超过 recursion_limit
  ▼
返回最终 state
```

返回值有三个使用方：

1. **运行时本身**（第 3 步）：按字段写进 channel，经过 reducer 合并。
2. **下一步的节点和条件边**（第 1、5 步）：读到的是 channel 合并后的最新值。`chat` 和 `use_tool` 之间不直接传值，而是通过 channel 交换数据。哪些节点被触发，由"这一步哪些 channel 被更新了"决定。
3. **外部观察者**（第 4 步）：`stream_mode="updates"` 把节点的原始返回值按 `{节点名: 返回值}` 发给调用方；checkpointer（Step 5）把每步的 channel 值和版本号存起来，之后可以恢复或回放。

### 对照 Step 1

| Step 1 手写 | LangGraph |
| --- | --- |
| `for step in range(max_steps)` 循环 | 运行时循环执行 super-step，上限是 `recursion_limit` |
| `messages.append(...)` | 第 3 步：`add_messages` reducer |
| `if not msg.tool_calls: return output` | 第 5 步：条件边 `should_continue` 返回 `END` |
| 调用模型、执行工具 | 第 2 步：节点，这是我们唯一需要写的部分 |
## 9. 预制组件：`ToolNode` 和 `tools_condition`

```python
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))  # 替代手写的 use_tool
graph_builder.add_conditional_edges("agent", tools_condition)               # 替代手写的 should_continue
```

- `tools_condition` 的返回类型是 `Literal["tools", "__end__"]`，所以**工具节点必须命名为 `"tools"`**。`"__end__"` 就是 `END` 常量的值。
- `ToolNode` **不能脱离图单独 `.invoke`**，会报 `Missing required config key 'N/A' for 'tools'`。它依赖图运行时注入的 config。要单独测试，就搭一个 `START → tools → END` 的小图。

### 坑：默认的错误处理只处理参数校验错误

同样 4 种情况，和手写的 `use_tool` 对比：

| 情况 | 手写 `use_tool` | `ToolNode` 默认 | `handle_tool_errors=True` |
| --- | --- | --- | --- |
| 正常调用 | success | success | success |
| 工具名不存在 | error | error（会列出可用工具） | error |
| 漏传参数（`ValidationError`） | error | error | error |
| **工具内部抛异常**（`days_until("明天")`） | error | **抛出异常，整张图崩溃** | error |

原因（`langgraph/prebuilt/tool_node.py`）：

```python
def _default_handle_tool_errors(e: Exception) -> str:
    if isinstance(e, ToolInvocationError):   # 只处理参数校验错误
        return e.message
    raise e                                  # 其他异常全部抛出
```

只写 `ToolNode(TOOLS)` 的话，"工具出错时不让图崩溃"这个保证会**悄悄失效**。正常流程的 eval 也发现不了，因为 eval 不会触发工具内部异常。

`handle_tool_errors` 还可以传字符串（固定的错误提示）、函数（自定义错误格式），或者异常类型/元组（只处理这几种异常）。

### 用测试锁住这个配置

`tests/test_graph.py` 从编译好的图里取出**真正注册的节点**来测，而不是在测试里重新创建一个 `ToolNode`：

```python
graph.graph.builder.nodes["tools"].runnable   # graph.py 里实际配置的 ToolNode
```

验证过：把 `handle_tool_errors=True` 删掉后，`test_tool_error_becomes_error_message[configured-days_until-...]` 会失败。

### 比手写版多做的事

- **并行执行多个工具调用**：`executor.map(self._run_one, tool_calls, ...)` 用了线程池；手写版的 for 循环是串行执行的。
- **错误信息带引导**：会列出可用的工具名，结尾加上 "Please fix your mistakes."，引导模型修正后重试。
- **扩展点**：`wrap_tool_call` 可以在工具调用前后插入逻辑（日志、重试、权限检查），相当于 hook。Step 5 可能用得上。

### 对比结果

切换到预制组件后，eval 仍然是 3/3，token 和步数与手写版基本一致（1940 vs 1974、1895 vs 1889）。Step 3 的 `create_agent` 内部应该也是用 `ToolNode` 执行工具（目前还没装 `langchain`，到时候读源码确认），届时同样要检查错误处理的默认行为。
