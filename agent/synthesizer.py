"""Synthesizer node (Task 1.6)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _latest_user_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _content(message)
        if isinstance(message, dict) and message.get("role") in {"human", "user"}:
            return _content(message)
    return _content(messages[-1]) if messages else ""


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        messages = state.get("messages", [])
        question = _latest_user_text(messages)
        plan = state.get("plan", [])
        results = state.get("step_results", [])
        context_lines = [
            f"Step {number} ({plan[number - 1] if number <= len(plan) else 'unknown'}): {result}"
            for number, result in enumerate(results, start=1)
        ]
        context = "\n".join(context_lines) or "No step results were produced."
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYNTHESIZER_PROMPT),
                    HumanMessage(
                        content=f"Original question: {question}\n\nExecution results:\n{context}"
                    ),
                ]
            )
            answer = _content(response).strip()
        except Exception:
            answer = context
        if not answer:
            answer = "I could not produce an answer from the available step results."
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesizer
