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


# 未提交作业列表：ID、课程、标题、截止时间
def list_assignments(course: str | None = None) -> list[dict]:
    assignments = load_json(ASSIGNMENT_DIR / "assignments.json", [])
    return [a for a in assignments if course is None or a["course"] == course]


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


if __name__ == "__main__":
    print(list_assignments("软件工程"))
    print(get_assignment("a3"))
    print(days_until("2026-11-09T04:00:00Z"))
    print(days_until("2026-11-12T15:59:00+08:00"))
