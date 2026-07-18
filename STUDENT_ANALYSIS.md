# CS4603 PA4 — Document Analyst (Student Submission)

## Setup

The project uses Python 3.12 and `uv`.

```bash
uv sync --extra dev
cp .env.example .env
```

Populate every placeholder in `.env`. The required values are the Databricks workspace
host and token, the chat and embedding endpoint names, the three-part Unity Catalog
names, the Vector Search endpoint/index, and the serving endpoint name. For Bonus B,
install the optional dependency with `uv sync --extra dev --extra agents`.

### Corpus ingestion

Upload `data/annual_report.pdf` to a Unity Catalog volume. On Databricks Runtime 18.2+
(or a compatible serverless environment), run:

```python
import os

from rag.ingest import build_chunks_table, create_index

build_chunks_table(
    spark,
    volume_path="/Volumes/main/default/pa4/annual_report.pdf",
    chunks_table=os.environ["SOURCE_TABLE"],
)
create_index()
```

`build_chunks_table` uses `ai_parse_document` and `ai_prep_search`, writes the required
chunk and citation columns, and enables Delta Change Data Feed. `create_index` creates
or reuses a STANDARD endpoint, creates a TRIGGERED Delta Sync index with managed
embeddings, triggers a refresh when the index already exists, and waits until it is
ready. Before running the graph, verify a query in the Vector Search UI or SDK.

### Running locally

```python
from agent.graph import build_graph

graph = build_graph()
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What was the net income in 2023?"}]}
)
print(result["messages"][-1].content)
```

The notebook contains retrieval-only, computation-only, and combined queries plus a
streamed execution trace. The credential-free regression test is:

```bash
uv run pytest tests/test_smoke.py -q
```

It injects a fake LLM, retriever, and MCP tool, then verifies that a combined request
passes through both specialists and returns a non-empty final message.

## Deployment

Create the secret scope once; substitute the real values locally and never commit them:

```bash
databricks secrets create-scope cs4603-deploy
databricks secrets put-secret cs4603-deploy DATABRICKS_TOKEN --string-value "$DATABRICKS_TOKEN"
databricks secrets put-secret cs4603-deploy DATABRICKS_HOST --string-value "$DATABRICKS_HOST"
databricks secrets put-secret cs4603-deploy DATABRICKS_MODEL --string-value "$DATABRICKS_MODEL"
```

Then run:

```bash
uv run python deployment/deploy.py
databricks serving-endpoints get "$SERVING_ENDPOINT_NAME"
```

The script logs `deployment/agent_model.py` with all local packages in `code_paths`,
registers a new Unity Catalog model version, creates or updates a Small scale-to-zero
endpoint, waits for the update, and prints the version, status, and invocation URL.
Credentials in endpoint configuration are secret references; Vector Search identifiers
are plain configuration.

The non-bonus manual endpoint was deployed from this checkout. The active Unity Catalog
model is `cs4603.student_27100077.document_analyst` version `8`, served by
`27100077-document-analyst`. The endpoint status is `READY` with `config_update` equal
to `NOT_UPDATING`, workload size `Small`, and scale-to-zero enabled. `DATABRICKS_HOST`,
`DATABRICKS_TOKEN`, and `DATABRICKS_MODEL` are configured as Databricks secret
references; the Vector Search identifiers are plain environment values. The active
non-bonus endpoint does not set `MCP_SERVER_URL`, so MCP tools run through stdio inside
the model container.

Vector Search is ready at `cs4603.student_27100077.analyst_index` on endpoint
`27100077-pa4-vs`; the index reports `ONLINE_NO_PENDING_UPDATE`, `ready=True`, source
table `cs4603.student_27100077.analyst_chunks`, and 7 indexed rows.

The deployed endpoint was invoked successfully with `curl`: return code `0`, HTTP
status `200`, and a non-empty 754-byte response. The returned answer was: "The net
income in 2023 was ¥1,107 billion [source: annual_report.pdf, p.2.0]." The client SDK
health check returned `True`. Live SDK queries returned the retrieval answer above in
3.47s, the calculation answer "15% of 2.4 billion is 3.6e+08, or 360 million" in
8.87s, and the combined answer "The revenue in 2023 was ¥16.91 trillion
[source: annual_report.pdf, p.2.0]. A 10% increase would be ¥1.691 trillion, resulting
in a total of ¥18.601 trillion" in 7.23s. The notebook also shows timeout handling and
404 endpoint-error handling. Bonus Review App evidence is not claimed for the
non-bonus submission. After these outputs were captured, the manual serving endpoint
`27100077-document-analyst` was deleted and verified as not found, matching the
assignment cleanup checklist.

## Design decisions

The graph plans once and executes steps sequentially through a supervisor. Lookup and
calculation are separate nodes so retrieval prompts and deterministic tools can be
tested and tuned independently. Every calculation prompt receives prior step results,
which lets a later calculation consume a value found by RAG. The synthesizer is the
only node that creates the user-facing response and writes it to both `final_answer`
and the reducer-backed `messages` channel.

Production dependencies are lazy in `build_graph`, while tests inject fakes. The same
`rag.store.get_retriever` path is used locally and in serving. MCP tools load once when
the graph is built; stdio is the default, and `MCP_SERVER_URL` switches to remote
streamable HTTP for Bonus C. The client supports both raw LangGraph-state and
chat-native response shapes.

---

## Analysis Questions

### Task 1.2 — Planner

1. **What happens when planner steps depend on one another?**

   Steps run in order, and a node increments `current_step_index` only after recording
   its result. The MCP node receives all earlier `step_results`, so a calculation can
   use a fact retrieved in step 1. A bad plan can still omit or ambiguously reference a
   prerequisite; production recovery would validate required inputs before tool
   execution and ask the planner to repair only the remaining plan.

2. **Would replanning after every execution help?**

   Usually it would hurt this small, bounded workflow: every replan adds model latency,
   cost, and another chance to change a correct plan. For “find revenue, then apply 8%
   CAGR,” the dependency is known up front. Replanning is useful selectively when a
   lookup fails or returns an unexpected unit—for example, when the report gives
   millions but the next step assumes billions. I would trigger it on validation
   failure rather than after every successful step.

### Task 1.3 — Supervisor

1. **What is the failure mode of a misroute, and how would it be recovered?**

   A calculation sent to RAG may return no fact, while a lookup sent to MCP may make no
   valid tool call. Detection signals include empty retrieval, missing citations,
   absent tool calls, tool argument validation errors, and result-type checks. A robust
   recovery policy would mark the attempt, route once to the other specialist, and
   replan or fail explicitly after a bounded number of attempts to avoid loops.

2. **Supervisor versus one ReAct agent**

   ReAct is simpler for short, unpredictable tasks with few tools. The supervisor is
   worthwhile when queries repeatedly mix retrieval and deterministic computation,
   because each node has a narrow prompt, routing and intermediate results are
   observable, and each specialist is independently testable. Its extra model calls
   and state logic are not justified for a single lookup or a tiny tool set.

### Task 1.4 — RAG Agent

1. **How does retrieval on a decomposed step affect quality?**

   An atomic step removes unrelated computation language and usually improves semantic
   precision. It can reduce recall if decomposition strips useful entities, dates, or
   context from the original request. The best retrieval query retains the focused
   intent while carrying forward essential qualifiers from the original question.

2. **How would a vague retrieval step be improved?**

   I would rewrite it using the original question, known entities, period, metric, and
   relevant prior results—for example, replace “find relevant financial data” with
   “Find Meridian Motor Corporation FY2023 net revenue and its reporting unit.” If it
   remains vague, I would generate a small set of specific query variants, retrieve for
   each, deduplicate, and rerank the combined candidates.

### Task 2.1 — Model Definition

1. **Why must models-from-code be self-contained?**

   MLflow rebuilds the model in a fresh serving container. Only the model definition,
   declared `code_paths`, dependencies, and reachable services exist there. A laptop
   process, unshipped module, local path, or local database will be absent or
   unreachable, causing import-time failure or inference-time connection errors.

2. **External index versus baking the corpus into the artifact**

   An external managed index stays fresh without relogging the model and keeps the
   artifact and cold-start footprint small. The costs are a network hop, authentication
   and permission requirements, service latency, and new timeout/availability failure
   modes. A baked corpus is versioned atomically with the model and avoids a remote
   dependency, but increases artifact size and cold start, duplicates storage, and
   becomes stale until the model is rebuilt.

### Task 2.3 — Serving Endpoint

1. **Why does the endpoint still need `DATABRICKS_TOKEN`?**

   Authentication of an inbound caller to Model Serving is separate from credentials
   used by code inside the container for outbound LLM and Vector Search calls. The
   container does not automatically inherit the deployer's personal token on the manual
   path, so those clients need an injected credential. The token is stored in a secret
   scope and referenced, not placed in source or endpoint JSON as plaintext.

2. **What happens during a model-version update?**

   Databricks provisions and health-checks the new served entity while the existing
   configuration remains available, then switches traffic after the update is ready.
   Requests already accepted by the old replica can finish on that version, while later
   requests use the new one. This rolling transition avoids deliberate delete/recreate
   downtime, but clients must tolerate a short period where concurrent responses can
   come from different model versions.

### Task 3.2 — Client

1. **Why exponential backoff?**

   Rate limiting and scale-up are transient load conditions. Exponential backoff rapidly
   reduces retry pressure, gives the endpoint time to recover, and avoids the synchronized
   retry storm caused by a fixed short interval.

2. **What is dangerous about excessive retries?**

   They amplify a partial outage into more traffic, retain connections and worker slots,
   increase user-visible latency and cost, and can repeat non-idempotent work. Production
   systems need bounded retries, deadlines, jitter, circuit breaking, and observability.

3. **When is streaming preferable?**

   Streaming is useful in an interactive analyst chat where a long cited explanation
   should begin rendering immediately. `ask()` is preferable for batch processing or
   code that needs one complete string. Because a generic models-from-code endpoint may
   emit only one completion or reject the streaming flag, this SDK treats a single
   full-answer chunk as valid and falls back to the normal invocation path when
   Databricks reports that streaming is unsupported.

### Bonus A — CI/CD

Pipeline evidence:

- Pull request workflow run: `29657902830` for the Bonus A pipeline branch.
  `lint-and-test` completed successfully and `deploy` was skipped, confirming pull
  requests do not mutate the serving endpoint.
- Main-branch workflow run: `29657943955` after merging PR #1. `lint-and-test`
  completed successfully, then `deploy` completed successfully.
- The deploy log registered
  `cs4603.student_27100077.document_analyst` version `11`, updated
  `27100077-document-analyst`, and printed `Endpoint status: READY`.
- After the Bonus A deploy output was captured, `27100077-document-analyst` was deleted
  again and verified as not found, matching the cost cleanup requirement.

1. **Why deploy only from `main`?**

   `main` is the reviewed source of truth and provides one serialization point for the
   live endpoint. Deploying feature branches would expose unfinished code and let
   concurrent branches overwrite each other's model versions. Pull requests still run
   lint and tests without mutating production.

2. **What performance gate should be added?**

   Add an evaluation job between tests and deployment. Run the candidate on a versioned,
   held-out set containing retrieval, calculation, citation, and failure cases; record
   accuracy, groundedness, citation validity, tool success, and latency in MLflow.
   Compare those metrics with the currently served version and fail the workflow if any
   hard threshold is missed or statistically meaningful regression exceeds the allowed
   margin.

### Bonus B — `databricks-agents`

1. **Manual deployment versus `agents.deploy()`**

   The manual route exposes endpoint naming, served-entity configuration, secret
   references, traffic settings, polling, and update behavior, so it offers maximum
   control and easier inspection of each platform step. `agents.deploy()` removes that
   boilerplate, supplies workspace authentication, and creates the Review App, but it
   hides more lifecycle details and offers fewer low-level customization points.

2. **How would Review App feedback improve the agent?**

   I would join ratings and comments to traces, label the failure stage (planning,
   routing, retrieval, tool arguments, or synthesis), and add representative failures
   to a versioned evaluation set. Repeated retrieval failures would drive query or
   chunking changes; incorrect routes would improve supervisor examples; weak final
   answers would refine synthesis. A candidate change would pass the offline evaluation
   gate, deploy to a small test cohort, and be promoted only if feedback improves.

### Bonus C — Standalone MCP server

1. **Gains and new failure modes**

   Separation allows tools to scale, deploy, patch, log, and set permissions
   independently, and one service can support multiple agents. It introduces DNS/network
   latency, request timeouts, authentication failures, version skew, another quota and
   availability dependency, and the need for retries and circuit breaking.

2. **How should it be secured?**

   Use Databricks identity rather than a public static token where possible: a dedicated
   service principal or workload identity with least privilege, private workspace
   networking or ingress controls, TLS, short-lived credentials, and server-side
   authorization/audit logs. Grant invoke permission only to the serving identity and
   rotate any fallback secret through a Databricks secret scope.

3. **When is bundling better?**

   Bundling is better for a small, stable, agent-specific tool set when minimum latency,
   atomic versioning, and operational simplicity matter more than independent scaling.
   A separate service is worth it when tools are shared, change on a different cadence,
   need distinct security or monitoring, or consume resources that should scale
   independently.
