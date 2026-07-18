"""Full Document Analyst graph (Tasks 1.5 and 1.7)."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer

_MCP_PROCESS_STDERR = None


class _DatabricksM2MAuth(httpx.Auth):
    """Refresh a service-principal OAuth token for Databricks App requests."""

    def __init__(self, host: str, client_id: str, client_secret: str) -> None:
        self._auth_base = httpx.BasicAuth(client_id, client_secret)
        self._token_url = f"{host.rstrip('/')}/oidc/v1/token"
        self._token = ""
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _access_token(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._expires_at - 60:
                return self._token
            response = httpx.post(
                self._token_url,
                auth=self._auth_base,
                data={"grant_type": "client_credentials", "scope": "all-apis"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            self._token = str(payload["access_token"])
            self._expires_at = time.monotonic() + float(payload.get("expires_in", 3600))
            return self._token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._access_token()}"
        yield request


def _run_coroutine(coroutine):
    """Run async MCP operations from sync graph code, including inside notebooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Jupyter already owns an event loop in the main thread. Keep the graph's
    # public API synchronous by running this short MCP operation in a worker.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def load_mcp_tools(server_path: str | None = None):
    """Load MCP tools once, using remote HTTP when configured and stdio otherwise."""
    # MLflow Serving replaces ``sys.stderr`` with a logging proxy that has no
    # ``fileno``. The MCP stdio transport captures stderr as an import-time
    # default and later passes it to subprocess.Popen, which requires fileno.
    # Capture Python's real stderr for that default, then immediately restore
    # MLflow's proxy for the rest of model loading.
    global _MCP_PROCESS_STDERR
    current_stderr = sys.stderr
    process_stderr = sys.__stderr__
    try:
        process_stderr.fileno()
    except (AttributeError, OSError):
        if _MCP_PROCESS_STDERR is None:
            _MCP_PROCESS_STDERR = open(os.devnull, "w")  # noqa: SIM115
        process_stderr = _MCP_PROCESS_STDERR
    try:
        sys.stderr = process_stderr
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from mcp.client import stdio as mcp_stdio
    finally:
        sys.stderr = current_stderr

    # ``mcp.client.stdio.stdio_client`` may have been imported before this
    # function and captured MLflow's logging proxy as its default ``errlog``.
    # asynccontextmanager preserves the original generator on ``__wrapped__``.
    stdio_generator = getattr(mcp_stdio.stdio_client, "__wrapped__", None)
    if stdio_generator is not None and stdio_generator.__defaults__:
        stdio_generator.__defaults__ = (process_stderr,)

    mcp_url = os.environ.get("MCP_SERVER_URL", "").strip().rstrip("/")
    if mcp_url:
        endpoint = mcp_url if mcp_url.endswith("/mcp") else f"{mcp_url}/mcp"
        config: dict[str, object] = {
            "url": endpoint,
            "transport": "streamable_http",
        }
        token = os.environ.get("MCP_SERVER_TOKEN")
        if token:
            config["headers"] = {"Authorization": f"Bearer {token}"}
        elif client_id := os.environ.get("MCP_SERVER_CLIENT_ID"):
            client_secret = os.environ.get("MCP_SERVER_CLIENT_SECRET", "")
            if not client_secret:
                raise OSError("MCP_SERVER_CLIENT_SECRET is required with MCP_SERVER_CLIENT_ID")
            config["auth"] = _DatabricksM2MAuth(
                os.environ["DATABRICKS_HOST"],
                client_id,
                client_secret,
            )
        elif token := os.environ.get("DATABRICKS_TOKEN"):
            config["headers"] = {"Authorization": f"Bearer {token}"}
    else:
        path = (
            Path(server_path) if server_path else Path(__file__).parents[1] / "tools/mcp_server.py"
        )
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP server not found: {path}")
        config = {
            "command": sys.executable,
            "args": [str(path)],
            "transport": "stdio",
        }

    client = MultiServerMCPClient({"analyst": config})
    return _run_coroutine(client.get_tools())


def _tool_name(tool) -> str:
    return str(getattr(tool, "name", ""))


def _invoke_tool(tool, arguments):
    """Invoke local fake tools synchronously and MCP tools through their async API."""
    coroutine = getattr(tool, "coroutine", None)
    if coroutine is not None and hasattr(tool, "ainvoke"):
        return _run_coroutine(tool.ainvoke(arguments))
    result = tool.invoke(arguments)
    if hasattr(result, "__await__"):
        return _run_coroutine(result)
    return result


def _result_text(result) -> str:
    if isinstance(result, list):
        parts = [
            str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in result
        ]
        return "\n".join(parts)
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(getattr(result, "content", result))


def make_mcp_node(tools, llm):
    tools_by_name = {_tool_name(tool): tool for tool in tools}
    tool_llm = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        index = state.get("current_step_index", 0)
        step = state.get("plan", [""])[index]
        previous = "\n".join(
            f"Step {number}: {result}"
            for number, result in enumerate(state.get("step_results", []), start=1)
        )
        prompt = f"Current calculation step: {step}\nPrior step results:\n{previous or '(none)'}"

        try:
            response = tool_llm.invoke(
                [
                    SystemMessage(content=MCP_STEP_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                result = "calculation failed: no tool was called"
            else:
                call = tool_calls[0]
                name = call.get("name", "")
                arguments = call.get("args", {})
                tool = tools_by_name.get(name)
                if tool is None:
                    result = f"calculation failed: unknown tool '{name}'"
                else:
                    result = _result_text(_invoke_tool(tool, arguments))
        except Exception as exc:
            result = f"calculation failed: {type(exc).__name__}: {exc}"

        return {
            "step_results": [*state.get("step_results", []), result],
            "current_step_index": index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Build the graph, lazily constructing production dependencies when omitted."""
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node(RAG, make_rag_agent(retriever, llm))
    builder.add_node(MCP, make_mcp_node(tools, llm))
    builder.add_node(SYNTH, make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: RAG, MCP: MCP, SYNTH: SYNTH},
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)
    return builder.compile()
