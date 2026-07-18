"""Shared configuration and client factories for the Document Analyst.

Centralising credential loading here keeps every other module free of
`os.environ` calls and makes the LLM / retriever easy to mock in tests.
All clients are created lazily so importing the package never requires
network access or credentials — only *invoking* a node does.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise OSError(
            f"Missing required environment variable: {name}. "
            "Set it in your .env (local) or the endpoint secret scope (deployed)."
        )
    return val


def get_settings() -> dict[str, str]:
    """Return configuration, allowing platform-native auth in Agent Framework."""
    return {
        "host": os.environ.get("DATABRICKS_HOST", "").rstrip("/"),
        "token": os.environ.get("DATABRICKS_TOKEN", ""),
        "model": _require("DATABRICKS_MODEL"),
        "embeddings": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
        "vs_endpoint": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
        "vs_index": os.environ.get("VECTOR_SEARCH_INDEX", ""),
    }


@lru_cache(maxsize=1)
def get_chat_llm(temperature: float = 0.0):
    """Configured ChatOpenAI client pointed at Databricks Model Serving.

    We use the OpenAI-compatible surface (same as PA1–PA3) so the deployed
    endpoint speaks the same protocol the client SDK expects.
    """
    s = get_settings()
    if not s["host"] or not s["token"]:
        from databricks_langchain import ChatDatabricks

        return ChatDatabricks(
            endpoint=s["model"],
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s["model"],
        api_key=s["token"],
        base_url=f"{s['host']}/serving-endpoints",
        temperature=temperature,
    )
