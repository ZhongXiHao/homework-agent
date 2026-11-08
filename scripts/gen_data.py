# scripts/gen_data.py
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")


def due(delta):
    t = (datetime.now(TZ) + delta).replace(hour=23, minute=59, second=0, microsecond=0)
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


SEED = [
    # (id, course, name, 截止时间离现在多久, description, attachments)
    ("a1", "软件工程", "需求规格说明书", timedelta(days=4), "按 IEEE 830 模板撰写，不少于 10 页", ["template.docx"]),
    ("a2", "操作系统", "实验三：页面置换", timedelta(days=0), "实现 FIFO 和 LRU 并比较缺页率", []),
    ("a3", "计算机网络", "介绍 TCP 与 UDP 的区别", None,
     "请写一篇不少于 500 字的文章，内容包括 TCP 与 UDP 的区别、各自的优缺点以及应用场景", []),
    ("a4", "软件工程", "lab1", timedelta(days=15), "小组按要求完成对 todo 系统的协作开发", ["lab1-requirements.pdf"]),
]

assignments = [
    {
        "id": aid,
        "course": course,
        "name": name,
        "due_at": due(delta) if delta is not None else None,
        "description": description,
        "attachments": attachments,
    }
    for aid, course, name, delta, description, attachments in SEED
]

ids = [a["id"] for a in assignments]
assert len(ids) == len(set(ids)), f"作业 ID 重复: {ids}"

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "assignments.json"

DATA_FILE.parent.mkdir(parents=True, exist_ok=True)  # data/ 不存在就先创建
DATA_FILE.write_text(
    json.dumps(assignments, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

if __name__ == "__main__":
    print(f"已生成 {DATA_FILE}")