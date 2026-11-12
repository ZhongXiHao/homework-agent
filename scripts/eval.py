import subprocess
import sys
from pathlib import Path
import importlib

impl = sys.argv[1] if len(sys.argv) > 1 else "graph"
run = importlib.import_module(f"homework_agent.{impl}").run

print(f"Using implementation: {impl}")

# 每次评测前重新生成模拟数据，保证截止时间相对于“今天”是固定的
GEN_DATA = Path(__file__).resolve().parent / "gen_data.py"
subprocess.run([sys.executable, str(GEN_DATA)], check=True)

CASES = [
    {"question": "这周最紧的是哪门课？",
     "expect_tools": {"list_assignments", "days_until"},  # 至少要调用这些
     "answer_contains": ["操作系统"]},
    {"question": "软件工程的作业要求是什么？",
     "expect_tools": {"list_assignments", "get_assignment"},
     "answer_contains": ["IEEE 830", "todo"]},  # 软件工程有两个作业，两份要求都要答到
    {"question": "你好",
     "expect_tools": set(),  # 空集合：不应该调用任何工具
     "answer_contains": []},
]

test_results = []
passes = 0

print("测试开始")
for case in CASES:
    r = run(case["question"], verbose=False)
    called = set(r.tool_calls)
    expect_tools = case["expect_tools"]
    if expect_tools:
        tools_ok = expect_tools <= called  # 期望的工具都被调用了（子集判断）
    else:
        tools_ok = not called  # 期望不调用工具，那就必须一个都没有
    answer = (r.answer or "").lower()  # 关键词不区分大小写，todo / TODO 都算
    missing_kw = [kw for kw in case["answer_contains"] if kw.lower() not in answer]
    answer_ok = not missing_kw
    passed = tools_ok and answer_ok
    if passed:
        passes += 1

    test_results.append({
        "question": case["question"],
        "tool_calls": r.tool_calls,
        "tokens": r.tokens,
        "steps": r.steps,
        "tool_not_calls": expect_tools - called,
        "missing_kw": missing_kw,
        "passed": passed,
    })

for test_result in test_results:
    print(f"{'✅' if test_result['passed'] else '❌'} {test_result['question']}  "
          f"tools={test_result['tool_calls']}  "
          f"{test_result['tokens']} tok  {test_result['steps']} 步"
          f"{'  缺少: ' + ', '.join(test_result['tool_not_calls']) if test_result['tool_not_calls'] else ''}"
          f"{'  缺少关键词: ' + ', '.join(test_result['missing_kw']) if test_result['missing_kw'] else ''}"
          )
print(f"通过率: {passes}/{len(CASES)}")
sys.exit(0 if passes == len(CASES) else 1)
