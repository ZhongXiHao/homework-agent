from langchain_core.utils.function_calling import convert_to_openai_tool
from homework_agent import graph, native


def test_prompt_shared():
    assert native.SYSTEM_PROMPT is graph.SYSTEM_PROMPT


def test_tool_schemas_aligned():
    assert native.TOOLS == [convert_to_openai_tool(t) for t in graph.TOOLS]
