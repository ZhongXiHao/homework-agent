from typing import Annotated, TypedDict

from homework_agent import tools
from langchain_core.messages import (BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

SYSTEM_PROMPT = """
你是一个作业助手，帮助学生查询作业信息。
你可以根据用户的提问，选择调用合适的工具，并将结果返回给用户。请注意，所有的时间都是以北京时间为准。
**作业信息只能来自工具的返回结果，不要编造。工具出错或没有数据时，如实告诉用户。**
"""

# 加载环境变量
load_dotenv()


@tool(parse_docstring=True)
def list_assignments(course: str | None = None) -> list[dict]:
    """
    列出未提交的作业，可按课程过滤

    返回 list[dict]: 作业列表，每个作业是一个字典，包含 id, 课程名称, 作业标题, 截止时间, 描述, 附件名称

    Args:
        course: 课程名，可省略
    """
    return tools.list_assignments(course)


@tool(parse_docstring=True)
def get_assignment(assignment_id: str) -> dict | None:
    """
    按作业 ID 获取作业说明、要求、附件名

    返回一个字典，包含作业的详细信息，包括 id, 课程名称, 作业标题, 截止时间, 描述, 附件名称；或 None，如果找不到该作业

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

if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage(content="软件工程有哪些作业")]})
    print(graph.get_graph().draw_mermaid())
    for m in result["messages"]:
        m.pretty_print()
