import json
from datetime import datetime, timezone

import pytest

from homework_agent import tools

SAMPLE = [
    {
        "id": "a1",
        "course": "软件工程",
        "name": "需求规格说明书",
        "due_at": "2026-11-12T15:59:00Z",
        "description": "按 IEEE 830 模板撰写",
        "attachments": ["template.docx"],
    },
    {
        "id": "a2",
        "course": "操作系统",
        "name": "实验三：页面置换",
        "due_at": "2026-11-09T04:00:00Z",
        "description": "实现 FIFO 和 LRU",
        "attachments": [],
    },
    {
        "id": "a3",
        "course": "软件工程",
        "name": "设计文档",
        "due_at": None,
        "description": "",
        "attachments": [],
    },
]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """把 ASSIGNMENT_DIR 指向临时目录，并写入样例数据，不依赖真实的 data/。"""
    (tmp_path / "assignments.json").write_text(
        json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(tools, "ASSIGNMENT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path, monkeypatch):
    """没有 assignments.json 的目录。"""
    monkeypatch.setattr(tools, "ASSIGNMENT_DIR", tmp_path)
    return tmp_path


# ---------- list_assignments ----------

def test_list_all_when_no_course(data_dir):
    assert [a["id"] for a in tools.list_assignments()] == ["a1", "a2", "a3"]


def test_list_filters_by_course(data_dir):
    assert [a["id"] for a in tools.list_assignments("软件工程")] == ["a1", "a3"]


def test_list_unknown_course_returns_empty(data_dir):
    assert tools.list_assignments("不存在的课") == []


def test_list_missing_file_returns_empty(empty_dir):
    assert tools.list_assignments() == []


# ---------- get_assignment ----------

def test_get_existing(data_dir):
    a = tools.get_assignment("a2")
    assert a["name"] == "实验三：页面置换"
    assert a["attachments"] == []


def test_get_unknown_returns_none(data_dir):
    assert tools.get_assignment("a99") is None


def test_get_none_id_raises(data_dir):
    with pytest.raises(ValueError):
        tools.get_assignment(None)


def test_get_missing_file_returns_none(empty_dir):
    assert tools.get_assignment("a1") is None


# ---------- days_until ----------

FIXED_NOW = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def frozen_now(monkeypatch):
    """把 tools 里的 datetime.now() 固定为 FIXED_NOW，让结果不随运行时间变化。"""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_NOW.astimezone(tz) if tz else FIXED_NOW.replace(tzinfo=None)

    monkeypatch.setattr(tools, "datetime", FixedDatetime)


@pytest.mark.parametrize(
    "deadline, expected",
    [
        ("2026-11-12T03:30:00Z", "2天3小时"),           # 超过一天
        ("2026-11-10T05:20:00Z", "5小时20分钟"),        # 不到一天
        ("2026-11-10T00:45:00Z", "45分钟"),             # 不到一小时
        ("2026-11-09T23:00:00Z", "已过期"),             # 已过截止时间
        ("2026-11-12T11:30:00+08:00", "2天3小时"),      # 与第一条是同一时刻
        ("2026-11-10T13:00:00", "5小时0分钟"),          # 不带时区，按北京时间处理
    ],
)
def test_days_until(frozen_now, deadline, expected):
    assert tools.days_until(deadline) == expected


def test_days_until_none(frozen_now):
    assert tools.days_until(None) == "无截止时间"


def test_days_until_invalid_format(frozen_now):
    with pytest.raises(ValueError, match="不是 ISO 格式"):
        tools.days_until("明天")
