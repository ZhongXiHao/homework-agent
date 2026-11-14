import logging
import os
from typing import Callable

from dotenv import load_dotenv
from homework_agent.graph import TOOLS
from homework_agent.prompts import SYSTEM_PROMPT
from homework_agent.run_result import RunResult
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, wrap_tool_call
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from typing_extensions import override

load_dotenv()

logger = logging.getLogger(__name__)

@wrap_tool_call
def tool_error_message(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    name = request.tool_call["name"]
    logger.info("调用工具 %s，参数 %s", name, request.tool_call["args"])
    try:
        tool_response = handler(request)
        logger.info("工具 %s 调用成功", name)
    except Exception as e:
        tool_response = ToolMessage(
            content=f"{type(e).__name__}: {e}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
        # 工具出错是预期内的情况（模型会看到错误并重试），用 warning，不打印堆栈
        logger.warning("工具 %s 调用失败: %s: %s", name, type(e).__name__, e)
    return tool_response


class LoggingMiddleware(AgentMiddleware):
    @override
    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        try:
            logger.info("准备调用模型，已有 %d 条消息", len(request.messages))
            model_response = handler(request)
        except Exception:
            # 模型调用失败是非预期错误：记录堆栈后继续抛出，不吞掉
            logger.exception("模型调用失败")
            raise
        # 日志代码本身不能让 run 崩溃：有的厂商不返回 usage
        usage = model_response.result[-1].usage_metadata
        logger.info("本次 token: %s", usage["total_tokens"] if usage else "未知")
        return model_response


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    reasoning_effort="low",
)

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    middleware=[LoggingMiddleware(), tool_error_message],
)


def run(question: str, max_steps: int = 8, verbose: bool = True) -> RunResult:
    output = RunResult(answer=None)
    try:
        for chunk in agent.stream(
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
    # 日志配置属于程序入口，run() 和 middleware 只负责产出日志
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run("这周最要紧的作业是什么？"))
