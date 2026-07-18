"""Required offline end-to-end smoke test (no Databricks or network calls)."""

from __future__ import annotations

import os
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        self.calls.append(system)
        if "planning component" in system:
            return AIMessage(
                content='["Find FY2023 net revenue in the document", '
                '"Calculate that revenue after 8% annual growth for 3 years"]'
            )
        if "route one plan step" in system:
            route = "mcp_tools" if "Calculate" in human else "rag_agent"
            return AIMessage(content=route)
        if "extract the answer" in system:
            return AIMessage(
                content="Meridian's FY2023 net revenue was ¥16.91 trillion "
                "[source: annual_report.pdf, p.4]"
            )
        if "solve one numerical" in system:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "growth_rate",
                        "args": {"start_value": 16.91, "rate": 0.08, "years": 3},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        if "final Document Analyst" in system:
            return AIMessage(
                content="Revenue was ¥16.91 trillion [source: annual_report.pdf, p.4]; "
                "after three years at 8% CAGR it would be ¥21.30 trillion."
            )
        raise AssertionError(f"Unexpected prompt: {system}")


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def invoke(self, query: str):
        self.queries.append(query)
        return [
            Document(
                page_content="FY2023 net revenue was ¥16.91 trillion.",
                metadata={"source": "/Volumes/main/default/pa4/annual_report.pdf", "page": 4},
            )
        ]


class FakeTool:
    name = "growth_rate"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, arguments: dict) -> str:
        self.calls.append(arguments)
        value = arguments["start_value"] * (1 + arguments["rate"]) ** arguments["years"]
        return f"16.91 at 8% CAGR for 3 years = {value:.2f} trillion"


def test_combined_query_runs_both_specialists_and_surfaces_answer():
    from agent.graph import build_graph

    llm = FakeLLM()
    retriever = FakeRetriever()
    tool = FakeTool()
    graph = build_graph(llm=llm, retriever=retriever, tools=[tool])

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What was Meridian's FY2023 revenue, and what is it after "
                        "three years of 8% growth?"
                    ),
                }
            ]
        }
    )

    assert len(result["plan"]) == 2
    assert len(result["step_results"]) == 2
    assert retriever.queries == ["Find FY2023 net revenue in the document"]
    assert tool.calls == [{"start_value": 16.91, "rate": 0.08, "years": 3}]
    assert result["final_answer"]
    assert result["messages"][-1].content == result["final_answer"]
