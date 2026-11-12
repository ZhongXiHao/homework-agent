import json
import os
from homework_agent import tools
from dotenv import load_dotenv
from openai import OpenAI
from homework_agent.prompts import SYSTEM_PROMPT
from homework_agent.run_result import RunResult

# 与 graph.py 中 @tool 生成的 schema 逐字一致，保证两个版本发给模型的内容相同、评测可比
TOOLS = [
    # ---------- list_assignments ----------
    {"type": "function", "function": {
        "name": "list_assignments",
        "description": "列出未提交的作业，可按课程过滤 返回 list[dict]: 作业列表，每个作业是一个字典，包含 id, 课程名称, 作业标题, 截止时间, 描述, 附件名称",
        "parameters": {"type": "object",
                       "properties": {
                           "course": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                      "default": None, "description": "课程名，可省略"}}}}},
    # --------- get_assignment ----------
    {"type": "function", "function": {
        "name": "get_assignment",
        "description": "按作业 ID 获取作业说明、要求、附件名 返回一个字典，包含作业的详细信息，包括 id, 课程名称, 作业标题, 截止时间, 描述, 附件名称；或 None，如果找不到该作业",
        "parameters": {"type": "object",
                       "properties": {
                           "assignment_id": {"type": "string", "description": "作业 id"}},
                       "required": ["assignment_id"]}}},
    # --------- days_until ----------
    {"type": "function", "function": {
        "name": "days_until",
        "description": "按 ISO 格式的截止时间计算当前距离截止的剩余时间 返回\"X天Y小时\"；不足一天返回\"X小时Y分钟\"；已过期返回\"已过期\"；无截止时间返回\"无截止时间\"。",
        "parameters": {"type": "object",
                       "properties": {
                           "deadline": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                        "description": "ISO 格式的截止时间字符串"}},
                       "required": ["deadline"]}}},
]
FUNCS = {"list_assignments": tools.list_assignments,
         "get_assignment": tools.get_assignment,
         "days_until": tools.days_until}

load_dotenv()

client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
MODEL = os.getenv("LLM_MODEL_ID")


def run(question: str, max_steps: int = 8, verbose: bool = True) -> RunResult:
    output = RunResult(answer=None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, reasoning_effort="low")
        msg = resp.choices[0].message
        if verbose:
            print(
                f"Step {step + 1}: Tokens used: {resp.usage.total_tokens}, Finish reason: {resp.choices[0].finish_reason}")
        output.tokens = output.tokens + resp.usage.total_tokens

        assistant = {"role": "assistant", "content": msg.content or ""}

        if assistant["content"]:
            if verbose:
                print(f"Step {step + 1}: Assistant said:\n{assistant['content']}... end\n")

        if msg.tool_calls:
            assistant["tool_calls"] = [c.model_dump() for c in msg.tool_calls]
            if verbose:
                print(f"Step {step + 1}: Assistant called tools:\n{assistant['tool_calls']}... end\n")

        messages.append(assistant)

        if not msg.tool_calls:  # 没有工具调用 = 最终回答
            output.answer = msg.content
            output.steps = step + 1
            return output

        for call in msg.tool_calls:  # 一次响应里可能有多个工具调用
            try:
                fn = FUNCS[call.function.name]
                result = fn(**json.loads(call.function.arguments))
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"Step {step + 1}: Tool {call.function.name} returned:\n{result}... end\n")
            output.tool_calls.append(call.function.name)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    output.answer = "超过最大步数，未能完成"
    output.steps = max_steps
    return output


def main() -> None:
    """命令行入口：带参数时回答一次，不带参数时进入交互模式。"""
    import sys

    if len(sys.argv) > 1:
        r = run(" ".join(sys.argv[1:]))
        print_result(r)
        return

    print("作业助手已启动，输入 exit 或按 Ctrl+D 退出。")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question in ("exit", "quit"):
            break
        if question:
            r = run(question)
            print_result(r)


def print_result(r: RunResult) -> None:
    print(f"Final answer: {r.answer}")
    print(f"Tokens used: {r.tokens}, Steps: {r.steps}, Tools: {r.tool_calls}\n")


if __name__ == '__main__':
    main()
