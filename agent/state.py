"""State schema for the Document Analyst graph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    """Conversation I/O plus the graph's internal execution scratch space."""

    messages: Annotated[list[Any], add_messages]
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    next_agent: str
    final_answer: str
