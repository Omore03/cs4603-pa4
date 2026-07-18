"""Corpus ingestion into a Databricks Delta table and Vector Search index.

``build_chunks_table`` must run on Databricks Runtime 18.2+ (or a compatible
serverless environment) because it uses ``ai_parse_document`` and
``ai_prep_search``.
"""

from __future__ import annotations

import os
import re

from config import get_settings

_TABLE_NAME = re.compile(r"^[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*$")


def _validated_table_name(name: str) -> str:
    if not _TABLE_NAME.fullmatch(name):
        raise ValueError("chunks_table must be a three-part Unity Catalog name")
    return name


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse one PDF, flatten search-ready chunks, and enable Change Data Feed."""
    table = _validated_table_name(chunks_table)
    escaped_path = volume_path.replace("'", "''")
    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {table}
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS
        WITH parsed_documents AS (
          SELECT path, ai_parse_document(content) AS parsed
          FROM READ_FILES('{escaped_path}', format => 'binaryFile')
        ),
        prepped_documents AS (
          SELECT path, ai_prep_search(parsed) AS result
          FROM parsed_documents
        )
        SELECT
          chunk.value:chunk_id::STRING AS chunk_id,
          chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
          chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
          regexp_extract(path, '[^/]+$', 0) AS source,
          coalesce(
            chunk.value:pages[0]:page_id::INT + 1,
            1
          ) AS page
        FROM prepped_documents,
          LATERAL variant_explode(prepped_documents.result:document.contents) AS chunk
        """
    )


def create_index() -> None:
    """Create (or refresh) the required STANDARD, TRIGGERED Delta Sync index."""
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    source_table = os.environ.get("SOURCE_TABLE", "")
    if not source_table:
        raise OSError("Missing required environment variable: SOURCE_TABLE")
    _validated_table_name(source_table)
    if not settings["host"] or not settings["token"]:
        raise OSError("DATABRICKS_HOST and DATABRICKS_TOKEN are required for ingestion")
    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX are required to create the index"
        )

    client = VectorSearchClient(
        workspace_url=settings["host"],
        personal_access_token=settings["token"],
        disable_notice=True,
    )
    endpoint = settings["vs_endpoint"]
    index_name = settings["vs_index"]

    if not client.endpoint_exists(endpoint):
        client.create_endpoint_and_wait(endpoint, endpoint_type="STANDARD", verbose=True)
    else:
        client.wait_for_endpoint(endpoint, verbose=True)

    if client.index_exists(endpoint_name=endpoint, index_name=index_name):
        index = client.get_index(endpoint_name=endpoint, index_name=index_name)
        index.sync()
        index.wait_until_ready(verbose=True, wait_for_updates=True)
    else:
        client.create_delta_sync_index_and_wait(
            endpoint_name=endpoint,
            index_name=index_name,
            primary_key="chunk_id",
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=settings["embeddings"],
            columns_to_sync=["chunk_to_retrieve", "chunk_to_embed", "source", "page"],
            verbose=True,
        )
