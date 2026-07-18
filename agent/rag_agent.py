"""RAG agent node (Task 1.4) backed by Databricks Vector Search."""

from __future__ import annotations

from pathlib import PurePath

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    """Format retrieved documents with stable source/page citations."""
    formatted: list[str] = []
    for document in docs:
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("source_uri") or "unknown"
        source = PurePath(str(source)).name or str(source)
        page = metadata.get("page")
        if page is None:
            page = metadata.get("page_number", "?")
        content = str(getattr(document, "page_content", document)).strip()
        formatted.append(f"{content}\n[source: {source}, p.{page}]")
    return "\n\n".join(formatted)


def make_rag_agent(retriever, llm):
    def _answer_lookup(query: str, step: str) -> str:
        docs = retriever.invoke(query)
        context = format_docs(docs)
        if not context:
            return "not found in documents"
        response = llm.invoke(
            [
                SystemMessage(content=RAG_EXTRACT_PROMPT),
                HumanMessage(content=f"Lookup step: {step}\n\nDocument context:\n{context}"),
            ]
        )
        return str(getattr(response, "content", response)).strip() or "not found in documents"

    def rag_agent(state: AnalystState) -> dict:
        index = state.get("current_step_index", 0)
        step = state.get("plan", [""])[index]
        result = "not found in documents"
        try:
            result = _answer_lookup(step, step)
            if result.strip().lower() == "not found in documents":
                messages = state.get("messages", [])
                original = ""
                if messages:
                    message = messages[-1]
                    original = (
                        str(message.get("content", ""))
                        if isinstance(message, dict)
                        else str(getattr(message, "content", message))
                    )
                if original and original != step:
                    result = _answer_lookup(f"{step}\nOriginal question: {original}", step)
        except Exception as exc:
            result = f"not found in documents (retrieval error: {type(exc).__name__})"

        return {
            "step_results": [*state.get("step_results", []), result],
            "current_step_index": index + 1,
        }

    return rag_agent
