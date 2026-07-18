"""Supervisor node and routing edge (Task 1.3)."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

_MATH_TERMS = {
    "calculate",
    "calculation",
    "compute",
    "compound",
    "convert",
    "difference",
    "growth",
    "larger",
    "percent",
    "percentage",
    "project",
    "ratio",
    "smaller",
    "sum",
}


def _fallback_route(step: str) -> str:
    words = {word.strip(".,:;!?()[]").lower() for word in step.split()}
    has_expression = bool(re.search(r"\d\s*(?:[-+*/]|\*\*)\s*\d", step)) or "%" in step
    return MCP if words & _MATH_TERMS or has_expression else RAG


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        if index >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[index]
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SUPERVISOR_PROMPT),
                    HumanMessage(content=f"Current step: {step}"),
                ]
            )
            content = str(getattr(response, "content", response)).lower()
            if MCP in content:
                route = MCP
            elif RAG in content:
                route = RAG
            else:
                route = _fallback_route(step)
        except Exception:
            route = _fallback_route(step)
        return {"next_agent": route}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    route = state.get("next_agent", SYNTH)
    return route if route in {RAG, MCP, SYNTH} else SYNTH
