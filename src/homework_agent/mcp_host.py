import asyncio
import os

from dotenv import load_dotenv
from homework_agent.prompts import SYSTEM_PROMPT
from homework_agent.run_result import RunResult
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

load_dotenv()

mcp_client = MultiServerMCPClient(
    {"homework": {"command": "uv", "args": ["run", "python", "-m", "homework_agent.mcp_server"], "transport": "stdio"}})

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    reasoning_effort="low",
)

_runner = asyncio.Runner()

def run(question, max_steps=8, verbose=True) -> RunResult:
    return _runner.run(_arun(question, max_steps, verbose))


async def _arun(question, max_steps, verbose) -> RunResult:
    # get_tools、create_agent、async for chunk in agent.astream(...)
    mcp_tools = await mcp_client.get_tools()
    agent = create_agent(model=llm, system_prompt=SYSTEM_PROMPT, tools=mcp_tools)
    output = RunResult(answer=None)
    try:
        async for chunk in agent.astream(
                {"messages": [HumanMessage(question)]},
                config=RunnableConfig(recursion_limit=max_steps * 2),
                stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                if verbose:
                    print(f"── {node} ──")
                for msg in update["messages"]:
                    if verbose:
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
    run("请列出所有未提交的作业", verbose=True)
