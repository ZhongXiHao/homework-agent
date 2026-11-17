import json
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone

load_dotenv()
# 相对路径按项目根目录解析，而不是按运行时所在的目录，这样从任何目录启动都能找到数据
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_assignment_dir = Path(os.environ.get("ASSIGNMENT_DIR", "data")).expanduser()
ASSIGNMENT_DIR = _assignment_dir if _assignment_dir.is_absolute() else PROJECT_ROOT / _assignment_dir
DEFAULT_TIMEZONE = ZoneInfo("Asia/Shanghai")


def load_json(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


# 未提交作业列表：只返回 ID、课程、标题、截止时间
def list_assignments(course: str | None = None) -> list[dict]:
    assignments = load_json(ASSIGNMENT_DIR / "assignments.json", [])
    if course:
        assignments = [a for a in assignments if a["course"] == course]
    return [
        {
            "id": a["id"],
            "course": a["course"],
            "name": a["name"],
            "due_at": a["due_at"],
        }
        for a in assignments
    ]


# 作业说明、要求、附件名
def get_assignment(assignment_id: str) -> dict | None:
    if assignment_id is None:
        raise ValueError("assignment_id is None")
    assignments = load_json(ASSIGNMENT_DIR / "assignments.json", [])
    return next((a for a in assignments if a["id"] == assignment_id), None)


# 距离截止还有几天几小时
def days_until(deadline: str | None) -> str:
    if deadline is None:
        return "无截止时间"
    try:
        due = datetime.fromisoformat(deadline)
        if due.tzinfo is None:  # 没写时区就当北京时间
            due = due.replace(tzinfo=DEFAULT_TIMEZONE)
    except ValueError as e:
        raise ValueError(f"deadline 不是 ISO 格式: {deadline}") from e

    now = datetime.now(timezone.utc)
    delta = due - now
    if delta.total_seconds() < 0:
        return "已过期"

    days = delta.days
    hours = delta.seconds // 3600
    if days > 0:
        return f"{days}天{hours}小时"

    minutes = delta.seconds // 60
    if hours > 0:
        return f"{hours}小时{minutes % 60}分钟"
    return f"{minutes}分钟"


# 把作业的截止时间加入日历（目前写本地 JSON 模拟，Step 6 换成真实日历 API）
# 出错一律 raise，由 ToolNode / middleware 统一转成 status="error" 的 ToolMessage
def add_to_calendar(assignment_id: str) -> str:
    assignment = get_assignment(assignment_id)
    if assignment is None:
        raise ValueError(f"找不到作业 {assignment_id}，请先用 list_assignments 查询正确的 ID")
    if assignment["due_at"] is None:
        raise ValueError(f"「{assignment['name']}」没有截止时间，无法加入日历")

    title = f"{assignment['course']} - {assignment['name']}"
    path = ASSIGNMENT_DIR / "calendar.json"
    events = load_json(path, [])
    if any(e["assignment_id"] == assignment_id for e in events):  # 按 ID 去重：重复调用结果不变（幂等）
        return f"「{title}」已在日历中，无需重复添加"

    # 日历条目只保留日历需要的字段，结构接近 Google Calendar 的 event
    events.append({"assignment_id": assignment_id, "title": title, "due_at": assignment["due_at"]})
    path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已将「{title}」加入日历，截止时间 {assignment['due_at']}"


if __name__ == "__main__":
    print(list_assignments("软件工程"))
    print(get_assignment("a3"))
    print(days_until("2026-11-09T04:00:00Z"))
    print(days_until("2026-11-12T15:59:00+08:00"))
    print(add_to_calendar("a1"))
