"""Log, register, and serve the Document Analyst (Tasks 2.2 and 2.3).

Run with a populated ``.env`` and an existing secret scope:
``uv run python deployment/deploy.py``.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MODEL_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "langchain-mcp-adapters>=0.0.5",
    "mcp>=1.0.0",
    "openai>=1.40.0",
    "python-dotenv>=1.0.0",
]

load_dotenv(ROOT / ".env")


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value or value.startswith("<"):
        raise OSError(f"Set {name} to a real value before deployment")
    return value


def model_environment(*, include_secrets: bool) -> dict[str, str]:
    """Build endpoint configuration without ever exposing credential values."""
    scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
    if include_secrets:
        environment = {
            name: f"{{{{secrets/{scope}/{name}}}}}"
            for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL")
        }
    else:
        # Agent Framework supplies workspace authentication automatically.
        environment = {"DATABRICKS_MODEL": _required("DATABRICKS_MODEL")}

    environment.update(
        {
            "VECTOR_SEARCH_ENDPOINT": _required("VECTOR_SEARCH_ENDPOINT"),
            "VECTOR_SEARCH_INDEX": _required("VECTOR_SEARCH_INDEX"),
            "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
        }
    )
    if mcp_url := os.environ.get("MCP_SERVER_URL"):
        environment["MCP_SERVER_URL"] = mcp_url
        if include_secrets:
            environment.update(
                {
                    name: f"{{{{secrets/{scope}/{name}}}}}"
                    for name in ("MCP_SERVER_CLIENT_ID", "MCP_SERVER_CLIENT_SECRET")
                }
            )
        else:
            for name in ("MCP_SERVER_CLIENT_ID", "MCP_SERVER_CLIENT_SECRET"):
                if value := os.environ.get(name):
                    environment[name] = value
    return environment


def log_and_register():
    """Log models-from-code and register a new Unity Catalog model version."""
    catalog = _required("UC_CATALOG")
    schema = _required("UC_SCHEMA")
    model_name = os.environ.get("REGISTERED_MODEL_NAME", "document_analyst")
    if model_name.startswith("<"):
        raise OSError("Set REGISTERED_MODEL_NAME to a real value before deployment")
    uc_name = f"{catalog}.{schema}.{model_name}"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "/Shared/cs4603-pa4"))

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=str(ROOT / "deployment" / "agent_model.py"),
            name="agent",
            code_paths=[
                str(ROOT / "agent"),
                str(ROOT / "rag"),
                str(ROOT / "tools"),
                str(ROOT / "config.py"),
            ],
            pip_requirements=MODEL_REQUIREMENTS,
            input_example={
                "messages": [{"role": "user", "content": "What was Meridian's revenue in 2023?"}]
            },
        )
    registered = mlflow.register_model(model_info.model_uri, uc_name)
    version = str(registered.version)
    print(f"Registered model: {uc_name}")
    print(f"Deployed model version: {version}")
    return uc_name, version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or safely update the serving endpoint and wait until it is ready."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    host = _required("DATABRICKS_HOST").rstrip("/")
    endpoint_name = _required("SERVING_ENDPOINT_NAME")
    workspace = WorkspaceClient(host=host, token=_required("DATABRICKS_TOKEN"))
    served_entity = ServedEntityInput(
        name=f"document-analyst-{version}",
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=model_environment(include_secrets=True),
    )

    try:
        workspace.serving_endpoints.get(endpoint_name)
    except NotFound:
        endpoint = workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=[served_entity],
            ),
            timeout=timedelta(minutes=30),
        )
    else:
        endpoint = workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
            timeout=timedelta(minutes=30),
        )

    ready = getattr(getattr(endpoint, "state", None), "ready", None)
    status = getattr(ready, "value", ready) or "UNKNOWN"
    invocation_url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint status: {status}")
    print(f"Endpoint URL: {invocation_url}")
    return invocation_url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
