"""Databricks Vector Search retriever factory shared by local and serving runs."""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()
    missing = [
        name
        for name, value in {
            "VECTOR_SEARCH_ENDPOINT": settings["vs_endpoint"],
            "VECTOR_SEARCH_INDEX": settings["vs_index"],
        }.items()
        if not value
    ]
    if missing:
        raise OSError(f"Missing required environment variables: {', '.join(missing)}")

    kwargs = {
        "endpoint": settings["vs_endpoint"],
        "index_name": settings["vs_index"],
        "columns": [TEXT_COLUMN, *CITATION_COLUMNS],
    }
    if settings["host"] and settings["token"]:
        kwargs["client_args"] = {
            "workspace_url": settings["host"],
            "personal_access_token": settings["token"],
        }

    return DatabricksVectorSearch(
        **kwargs,
    )


def get_retriever(k: int = 4):
    if k < 1:
        raise ValueError("k must be at least 1")
    return get_vector_store().as_retriever(search_kwargs={"k": k})
