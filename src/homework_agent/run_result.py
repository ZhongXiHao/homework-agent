from dataclasses import dataclass, field


@dataclass(slots=True)
class RunResult:
    answer: str | None
    tokens: int = 0
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)  # 按顺序记录调用过的工具名
