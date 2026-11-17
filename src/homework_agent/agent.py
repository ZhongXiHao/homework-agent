import asyncio
import logging
import os
import sys
import uuid

from dotenv import load_dotenv
from homework_agent.checkpointer import init_checkpointer, close_checkpointer
from homework_agent.prompts import SYSTEM_PROMPT
from homework_agent.run_result import RunResult
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.exceptions import ModelAuthenticationError, ModelTimeoutError, ModelRateLimitError, ModelError
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

load_dotenv()

logger = logging.getLogger(__name__)

RUN_TIMEOUT = 180

mcp_client = MultiServerMCPClient(
    {"homework": {"command": "uv", "args": ["run", "python", "-m", "homework_agent.mcp_server"], "transport": "stdio"}})

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    reasoning_effort="low",
    timeout=60,
    max_retries=2
)

human_in_the_loop_middleware = HumanInTheLoopMiddleware(
    # interrupt_on=dict[str, bool | InterruptOnConfig],
    interrupt_on={
        "add_to_calendar": True
    }
)

_runner = asyncio.Runner()


async def _build_agent(checkpointer):
    mcp_tools = await mcp_client.get_tools()
    return create_agent(model=llm, system_prompt=SYSTEM_PROMPT, tools=mcp_tools, checkpointer=checkpointer,
                        middleware=[human_in_the_loop_middleware])


def run(question, max_steps=8, verbose=True, my_checkpointer=None, thread_id: str | None = None) -> RunResult:
    return _runner.run(_arun({"messages": [HumanMessage(question)]}, max_steps, verbose, my_checkpointer, thread_id))


async def _arun(graph_input, max_steps, verbose, my_checkpointer, thread_id) -> RunResult:
    # get_tools、create_agent、async for chunk in agent.astream(...)
    agent = await _build_agent(my_checkpointer)
    output = RunResult(answer=None)
    config = RunnableConfig(recursion_limit=max_steps * 2)
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    try:
        async with asyncio.timeout(RUN_TIMEOUT):  # 给整轮对话设一个总时限
            async for chunk in agent.astream(
                    input=graph_input,
                    config=config,
                    stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        output.interrupt = update[0].value  # 记下来交给调用方，不在这里 input()
                        continue
                    if not update:
                        continue
                    if verbose:
                        print(f"── {node} ──")
                    for msg in update["messages"]:
                        if verbose:
                            msg.pretty_print()
                        # 只统计 model 节点产生的 AIMessage：HITL 恢复后，它的 after_model 节点会把同一条
                        # AIMessage（同 id，由 add_messages 替换）再输出一次，不过滤就会重复计数
                        if isinstance(msg, AIMessage) and node == "model":
                            output.steps += 1
                            output.tool_calls += [c["name"] for c in msg.tool_calls]
                            output.answer = msg.content
                            if msg.usage_metadata:
                                output.tokens += msg.usage_metadata["total_tokens"]
    except GraphRecursionError:
        output.answer = "超过最大步数，未能完成"
    except ModelAuthenticationError:
        raise  # 配置错误，不吞
    except ModelTimeoutError:
        logger.warning("模型请求超时（已重试 %d 次）", llm.max_retries)
        output.answer = "模型响应超时，请稍后再试"
    except TimeoutError:
        logger.warning("整轮对话超过 %d 秒", RUN_TIMEOUT)
        output.answer = "处理时间过长，请稍后再试"
    except ModelRateLimitError:
        output.answer = "请求太频繁，请稍后再试"
    except ModelError:
        logger.exception("模型调用失败")
        output.answer = "模型服务出错，请稍后再试"
    return output


def resume(decisions: list[dict], thread_id: str | None = None, max_steps=8, verbose=True, my_checkpointer=None):
    return _runner.run(
        _arun(Command(resume={"decisions": decisions}), max_steps, verbose, my_checkpointer=my_checkpointer,
              thread_id=thread_id))


async def pending_interrupt(checkpointer, thread_id: str) -> dict | None:
    """这个 thread 是否停在确认环节（上次中断后程序退出、页面关闭等情况）。"""
    agent = await _build_agent(checkpointer)
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    return state.interrupts[0].value if state.interrupts else None


async def _ask_decisions(interrupt: dict) -> list[dict]:
    """在终端逐个询问用户是否批准。decision 数量必须和 action_requests 一致。"""
    decisions = []
    for req in interrupt["action_requests"]:
        answer = await asyncio.to_thread(input, f"确认执行 {req['name']}({req['args']})？[y/N] ")
        if answer.strip().lower() == "y":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "用户拒绝了这次操作，没有执行"})
    return decisions


async def _resolve_interrupts(r: RunResult, ck, thread_id: str) -> RunResult:
    # 用 while：恢复后模型可能再次调用写入工具，再次中断
    while r.interrupt:
        decisions = await _ask_decisions(r.interrupt)
        r = await _arun(Command(resume={"decisions": decisions}), max_steps=8, verbose=False,
                        my_checkpointer=ck, thread_id=thread_id)
    return r


async def chat(thread_id: str | None):
    ck = await init_checkpointer()
    try:
        thread_id = thread_id or str(uuid.uuid4())
        print(f"thread_id = {thread_id}")  # 记下来，下次用

        # 上次停在确认环节：先处理完，否则历史里会留下没有 ToolMessage 的悬空调用，模型容易编造结果
        if pending := await pending_interrupt(ck, thread_id):
            print("上次还有一个操作等待确认：")
            r = await _resolve_interrupts(RunResult(answer=None, interrupt=pending), ck, thread_id)
            print("助手：", r.answer)

        while (q := (await asyncio.to_thread(input, "你：")).strip()) not in ("", "exit"):
            r = await _arun({"messages": [HumanMessage(q)]}, max_steps=8, verbose=False, my_checkpointer=ck,
                            thread_id=thread_id)
            r = await _resolve_interrupts(r, ck, thread_id)
            print("助手：", r.answer)
    finally:
        await close_checkpointer()


if __name__ == "__main__":
    _runner.run(chat(sys.argv[1] if len(sys.argv) > 1 else None))
