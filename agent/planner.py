"""Planner node (Task 1.2)."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _message_text(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _latest_user_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
        if isinstance(message, dict) and message.get("role") in {"human", "user"}:
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _parse_plan(content: str, fallback: str) -> list[str]:
    """Parse a JSON plan, tolerating a fenced response while enforcing the contract."""
    candidate = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("["), candidate.rfind("]")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return [fallback]

    if not isinstance(parsed, list):
        return [fallback]
    plan = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
    return plan[:5] if plan else [fallback]


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        messages = state.get("messages", [])
        question = _latest_user_text(messages)
        if not question.strip():
            question = "Answer the user's question"

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=question),
                ]
            )
            plan = _parse_plan(_message_text(response), question)
        except Exception:
            # Planning should degrade to a useful single-step execution instead of
            # preventing the graph from producing an answer.
            plan = [question]

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        }

    return planner
