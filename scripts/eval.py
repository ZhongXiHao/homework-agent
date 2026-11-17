import importlib
import inspect
import subprocess
import sys
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

impl = sys.argv[1] if len(sys.argv) > 1 else "graph"
run = importlib.import_module(f"homework_agent.{impl}").run

print(f"Using implementation: {impl}")

# 支持多轮的实现（run 能接收 thread_id 和 checkpointer）才跑多轮用例，其余实现跳过。
# 评测用内存版 checkpointer：不依赖数据库里的旧数据，每次评测都是干净的，类似单测里用 H2 代替真实数据库。
_params = inspect.signature(run).parameters
SUPPORTS_MEMORY = "thread_id" in _params and "my_checkpointer" in _params
memory = InMemorySaver() if SUPPORTS_MEMORY else None

# 每次评测前重新生成模拟数据，保证截止时间相对于“今天”是固定的
GEN_DATA = Path(__file__).resolve().parent / "gen_data.py"
subprocess.run([sys.executable, str(GEN_DATA)], check=True)

# 每个用例：
#   question        单轮问题；多轮用 turns: [问题1, 问题2, ...]，同一个 thread_id 依次提问，只检查最后一轮
#   expect_tools    至少要调用的工具；空集合 = 一个工具都不能调；None = 不检查
#   answer_contains 答案里必须全部出现的关键词（不区分大小写）
#   answer_any      至少出现其中一个（用于措辞不固定的情况，如"没有/暂无"）
#   answer_not      一个都不能出现（用于检查模型有没有声称做了做不到的事）
CASES = [
    # ---- 基础查询 ----
    {"question": "这周最紧的是哪门课？",
     "expect_tools": {"list_assignments", "days_until"},
     "answer_contains": ["操作系统"]},
    {"question": "软件工程的作业要求是什么？",
     "expect_tools": {"list_assignments", "get_assignment"},
     "answer_contains": ["IEEE 830", "todo"]},  # 软件工程有两个作业，两份要求都要答到
    {"question": "我一共有几个作业没交？",
     "expect_tools": {"list_assignments"},
     "answer_any": ["4", "四"]},
    {"question": "操作系统的实验要做什么？",
     "expect_tools": {"list_assignments", "get_assignment"},
     "answer_contains": ["FIFO", "LRU"]},
    # ---- 详情与附件 ----
    {"question": "需求规格说明书要用什么模板？有附件吗？",
     "expect_tools": {"get_assignment"},
     "answer_contains": ["IEEE 830", "template.docx"]},
    # ---- 时间计算 ----
    {"question": "lab1 还剩几天截止？",
     "expect_tools": {"list_assignments", "days_until"},
     "answer_contains": ["15"]},
    {"question": "计算机网络的作业什么时候截止？",  # due_at 为 null
     "expect_tools": {"list_assignments"},
     "answer_any": ["无截止", "没有截止", "未设置", "没有设置", "没有明确", "未指定", "没有指定"]},
    # ---- 多工具组合 ----
    {"question": "软件工程哪个作业最急，要交什么？",
     "expect_tools": {"list_assignments", "get_assignment"},
     "answer_contains": ["需求规格说明书", "IEEE 830"]},
    # ---- 不存在 / 做不到：不能编造 ----
    {"question": "数据结构有作业吗？",
     "expect_tools": {"list_assignments"},
     "answer_any": ["没有", "暂无", "未找到", "没找到", "不存在"]},
    {"question": "帮我把需求规格说明书的截止时间加到日历里",  # 目前没有写入工具
     "expect_tools": None,
     "answer_any": ["无法", "不能", "不支持", "暂不"],
     "answer_not": ["已添加", "已经添加", "添加成功", "已为你添加", "已帮你添加"]},
    # ---- 多轮追问：检查记忆 ----
    {"turns": ["计算机网络的作业什么时候截止？", "那它的要求是什么？"],
     "expect_tools": {"get_assignment"},  # 要从上一轮知道"它"是 a3，再去查详情
     "answer_contains": ["500"]},  # 字数要求只在详情里，标题里没有，能区分是否真的查了
    {"turns": ["软件工程的 lab1 什么时候截止？", "它有附件吗？"],
     "expect_tools": {"get_assignment"},
     "answer_contains": ["lab1-requirements.pdf"]},
    {"turns": ["我叫小明，记住哦", "我叫什么名字？"],  # 纯记忆，不涉及工具
     "expect_tools": set(),
     "answer_contains": ["小明"]},
    # ---- 闲聊：不该调工具 ----
    {"question": "你好",
     "expect_tools": set(),
     "answer_contains": []},
]

test_results = []
passes = 0

skipped = 0

print("测试开始")
for case in CASES:
    turns = case.get("turns") or [case["question"]]
    label = " → ".join(turns)
    if len(turns) > 1 and not SUPPORTS_MEMORY:
        skipped += 1
        test_results.append({"question": label, "skipped": True})
        continue

    # 每个用例一个新的 thread_id，用例之间互不污染
    kwargs = {"thread_id": str(uuid.uuid4()), "my_checkpointer": memory} if SUPPORTS_MEMORY else {}
    start = time.perf_counter()
    tokens = steps = 0
    for q in turns:
        r = run(q, verbose=False, **kwargs)
        tokens += r.tokens
        steps += r.steps
    elapsed = time.perf_counter() - start
    called = set(r.tool_calls)  # 只看最后一轮调用了哪些工具
    expect_tools = case["expect_tools"]
    if expect_tools is None:
        tools_ok = True  # 不检查工具
    elif expect_tools:
        tools_ok = expect_tools <= called  # 期望的工具都被调用了（子集判断）
    else:
        tools_ok = not called  # 期望不调用工具，那就必须一个都没有
    answer = (r.answer or "").lower()  # 关键词不区分大小写，todo / TODO 都算
    missing_kw = [kw for kw in case.get("answer_contains", []) if kw.lower() not in answer]
    any_kw = case.get("answer_any", [])
    if any_kw and not any(kw.lower() in answer for kw in any_kw):
        missing_kw.append("任一: " + "/".join(any_kw))
    missing_kw += ["不该出现: " + kw for kw in case.get("answer_not", []) if kw.lower() in answer]
    answer_ok = not missing_kw
    passed = tools_ok and answer_ok
    if passed:
        passes += 1

    test_results.append({
        "question": label,
        "tool_calls": r.tool_calls,
        "tokens": tokens,
        "steps": steps,
        "elapsed": elapsed,
        "answer": r.answer,
        "tool_not_calls": (expect_tools or set()) - called,
        "unexpected_tools": called if expect_tools == set() else set(),
        "missing_kw": missing_kw,
        "passed": passed,
    })

for test_result in test_results:
    if test_result.get("skipped"):
        print(f"⏭️  {test_result['question']}  （{impl} 不支持多轮，跳过）")
        continue
    print(f"{'✅' if test_result['passed'] else '❌'} {test_result['question']}  "
          f"tools={test_result['tool_calls']}  "
          f"{test_result['tokens']} tok  {test_result['steps']} 步  {test_result['elapsed']:.1f}s"
          f"{'  缺少: ' + ', '.join(test_result['tool_not_calls']) if test_result['tool_not_calls'] else ''}"
          f"{'  不该调用: ' + ', '.join(test_result['unexpected_tools']) if test_result['unexpected_tools'] else ''}"
          f"{'  缺少关键词: ' + ', '.join(test_result['missing_kw']) if test_result['missing_kw'] else ''}"
          )
    if not test_result["passed"]:  # 失败时打印回答，方便判断是模型错了还是关键词定得太死
        print(f"    回答: {(test_result['answer'] or '').strip()[:200]!r}")

ran = [t for t in test_results if not t.get("skipped")]
print(f"通过率: {passes}/{len(ran)}{f'（跳过 {skipped}）' if skipped else ''}  "
      f"总 token: {sum(t['tokens'] for t in ran)}  "
      f"总耗时: {sum(t['elapsed'] for t in ran):.1f}s")
sys.exit(0 if passes == len(ran) else 1)
