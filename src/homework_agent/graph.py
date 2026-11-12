import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from homework_agent import tools
from homework_agent.prompts import SYSTEM_PROMPT
from homework_agent.run_result import RunResult
from langchain_core.messages import (BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.errors import GraphRecursionError
from langgraph.graph import add_messages, StateGraph

# 加载环境变量
load_dotenv()


@tool(parse_docstring=True)
def list_assignments(course: str | None = None) -> list[dict]:
    """
    列出未提交作业的摘要（id、课程、标题、截止时间），可按课程过滤。要看作业要求和附件，再用 get_assignment 查询。

    Args:
        course: 课程名，可省略
    """
    return tools.list_assignments(course)


@tool(parse_docstring=True)
def get_assignment(assignment_id: str) -> dict | None:
    """
    按作业 ID 获取作业完整信息，包括说明、要求和附件名；找不到时返回 None。

    Args:
        assignment_id: 作业 id
    """
    return tools.get_assignment(assignment_id)


@tool(parse_docstring=True)
def days_until(deadline: str | None) -> str:
    """
    按 ISO 格式的截止时间计算当前距离截止的剩余时间

    返回"X天Y小时"；不足一天返回"X小时Y分钟"；已过期返回"已过期"；无截止时间返回"无截止时间"。

    Args:
        deadline: ISO 格式的截止时间字符串
    """
    return tools.days_until(deadline)


TOOLS = [list_assignments, get_assignment, days_until]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# 创建客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    reasoning_effort="low",
)

llm_with_tools = llm.bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: State) -> dict:
    r = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [r]}


def use_tool(state: State) -> dict:
    tool_messages = []
    for call in state["messages"][-1].tool_calls:
        t = TOOLS_BY_NAME.get(call["name"])
        if t is None:
            content, status = f"工具 {call['name']} 不存在", "error"
        else:
            try:
                tool_messages.append(t.invoke(call))
                continue
            except Exception as e:
                content, status = f"{type(e).__name__}: {e}", "error"
        tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"], status=status))
    return {"messages": tool_messages}


def should_continue(state: State) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


graph_builder = StateGraph(State)
graph_builder.add_node("agent", chat)
graph_builder.add_node("tools", use_tool)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph = graph_builder.compile()


def run(question: str, max_steps: int = 8, verbose: bool = True) -> RunResult:
    output = RunResult(answer=None)
    try:
        for chunk in graph.stream(
                input={"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": 2 * max_steps},
                stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                for msg in update["messages"]:
                    if verbose:
                        print(f"── {node} ──")
                        msg.pretty_print()
                    if isinstance(msg, AIMessage):
                        output.steps += 1
                        output.tool_calls += [c["name"] for c in msg.tool_calls]
                        output.answer = msg.content
                        if msg.usage_metadata:
                            output.tokens += msg.usage_metadata["total_tokens"]
    except GraphRecursionError:
        output.answer = "超过最大步数，未能完成"
    return output


if __name__ == "__main__":
    run("软件工程有哪些作业？")
    print(graph.get_graph().draw_mermaid())
