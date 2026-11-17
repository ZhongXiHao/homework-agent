from typing import Annotated

from homework_agent import tools
from mcp.server import FastMCP
from pydantic import Field

mcp = FastMCP("homework")


@mcp.tool()
def list_assignments(
        course: Annotated[str | None, Field(description="课程名，可省略")] = None,
) -> list[dict]:
    """列出未提交作业的摘要（id、课程、标题、截止时间），可按课程过滤。要看作业要求和附件，再用 get_assignment 查询。"""
    return tools.list_assignments(course)


@mcp.tool()
def get_assignment(
        assignment_id: Annotated[str, Field(..., description="作业 id")]
) -> dict | None:
    """按作业 ID 获取作业完整信息，包括说明、要求和附件名；找不到时返回 None。"""
    return tools.get_assignment(assignment_id)


@mcp.tool()
def days_until(
        deadline: Annotated[str | None, Field(..., description="截止时间")]
) -> str:
    """按 ISO 格式的截止时间计算当前距离截止的剩余时间。返回"X天Y小时"；不足一天返回"X小时Y分钟"；已过期返回"已过期"；无截止时间返回"无截止时间"。"""
    return tools.days_until(deadline)


if __name__ == "__main__":
    mcp.run()          # 默认走 stdio