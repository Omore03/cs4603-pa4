"""MLflow models-from-code definition for the production graph."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow

# MLflow imports this source file before it stages ``code_paths`` while logging
# locally. Make sibling packages importable for that initial construction pass.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.graph import build_graph, load_mcp_tools  # noqa: E402
from config import get_chat_llm  # noqa: E402
from rag.store import get_retriever  # noqa: E402

_REQUIRED_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    "EMBEDDINGS_ENDPOINT",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
)
_missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
if _missing:
    raise OSError(
        "Missing required environment variables for the deployed agent: " + ", ".join(_missing)
    )

graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
    tools=load_mcp_tools(),
)
mlflow.models.set_model(graph)
