"""测试 tools 节点的错误处理：只跑 START → tools → END 的小图，不调用模型。"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from homework_agent import graph, tools


def build_tools_only(node):
    g = StateGraph(graph.State)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


# graph.py 里实际注册的 tools 节点（USE_PREBUILT 时是 ToolNode），以及手写的 use_tool
NODES = {
    "configured": graph.graph.builder.nodes["tools"].runnable,
    "use_tool": graph.use_tool,
}


@pytest.fixture(params=NODES.keys())
def app(request, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ASSIGNMENT_DIR", tmp_path)  # 空目录，不依赖真实 data/
    return build_tools_only(NODES[request.param])


def run_call(app, name, args):
    call = {"name": name, "args": args, "id": "c1", "type": "tool_call"}
    result = app.invoke({"messages": [AIMessage("", tool_calls=[call])]})
    return result["messages"][-1]


def test_success(app):
    msg = run_call(app, "list_assignments", {})
    assert msg.status == "success"
    assert msg.tool_call_id == "c1"


@pytest.mark.parametrize(
    "name, args",
    [
        ("nope", {}),                           # 工具名不存在
        ("get_assignment", {}),                 # 缺参数：ValidationError
        ("days_until", {"deadline": "明天"}),   # 工具内部抛 ValueError
    ],
)
def test_tool_error_becomes_error_message(app, name, args):
    msg = run_call(app, name, args)  # 不应抛异常
    assert msg.status == "error"
    assert msg.tool_call_id == "c1"  # 仍与 tool_call 配对，否则模型那边协议会断
