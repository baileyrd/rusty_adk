# rusty-adk

A Rust implementation of the [Agent Development Kit (ADK) 2.0][adk] architecture.

ADK 2.0 ships SDKs for Python, Go, TypeScript, Java, and Kotlin — but not Rust.
`rusty-adk` is a port of its architecture: the same data model, the same
graph-based execution engine, the same tool and callback contracts, expressed
idiomatically in Rust.

```rust
use rusty_adk::prelude::*;
use std::sync::Arc;

/// Retrieves the current weather for a city.
#[adk_tool(crate = ::rusty_adk::tools)]
async fn get_weather(city: String) -> Result<serde_json::Value> {
    Ok(rusty_adk::tools::success(serde_json::json!({
        "report": format!("It is sunny in {city}."),
    })))
}

# async fn run() -> Result<()> {
let agent = LlmAgent::builder("weather_agent")
    .model(Arc::new(GeminiModel::from_env("gemini-flash-latest")?))
    .instruction("Answer weather questions using the get_weather tool.")
    .tool(get_weather_tool())
    .build()?;

let services = Services::new(Arc::new(InMemorySessionService::new()));
let runner = Runner::new("weather_app", agent.shared(), services);
let session = runner.create_session("user-1", None).await?;

let answer = runner
    .run_to_completion(&session.user_id, &session.id,
                       Content::user_text("Weather in Paris?"), None)
    .await?;
# Ok(()) }
```

## What ADK 2.0 is, and what this implements

Version 2.0 moved ADK from a hierarchical agent executor to a **graph execution
engine**: agents, tools, and plain functions are all evaluated as nodes in a
workflow graph, and data flows between them through `Event.output` rather than
through session state. The `Event` schema gained `node_info` and `output` to
carry that.

This port implements:

| Area | Implemented |
|---|---|
| **Data model** | `Content`/`Part`, `Event` (with `node_info`, `output`, `routes`), `EventActions`, prefix-scoped `State`, `Session`, `InvocationContext`, `RunConfig` |
| **Graph engine** | `Node`, concurrent frontier execution, route matchers (string/int/bool/multi/default), fan-out, `JoinNode` fan-in, per-node retries, step budget |
| **Human-in-the-loop** | `resume_or_request_input` suspends a run; the resume point is persisted and the node re-executes with the answer |
| **Tools** | `Tool` trait, `ToolContext`, `FunctionTool`, `#[adk_tool]`, toolsets, long-running tools, confirmation gating, artifacts, memory search |
| **Agents** | `LlmAgent` with the full tool-calling loop, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `AgentNode` |
| **Callbacks** | before/after agent, model, and tool — returning a value replaces the wrapped step |
| **Runtime** | `Runner`'s yield → commit → resume loop, streaming, cancellation |
| **Services** | `SessionService`, `ArtifactService`, `MemoryService` traits, each with an in-memory and a persistent SQLite implementation |
| **Models** | `Model` trait, `MockModel`, Gemini and Anthropic connectors |
| **Interop** | MCP server (stdio + streamable HTTP) and MCP client toolset for *tools*; an A2A bridge for *agents* |

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together, and for
the places where a Rust idiom differs from the reference SDKs.

## Install

```toml
[dependencies]
rusty-adk = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

Default features bring in the `#[adk_tool]` macro, the MCP transports, and the
live model connectors. Trim what you don't need:

```toml
rusty-adk = { version = "0.1", default-features = false, features = ["macros"] }
```

| Feature | Brings in | Default |
|---|---|---|
| `macros` | the `#[adk_tool]` attribute | yes |
| `mcp` | the MCP server and client transports | yes |
| `models` | the Gemini and Anthropic connectors | yes |
| `sqlite` | `SqliteStore` — durable session, artifact, and memory services | no |
| `a2a` | Serve this agent over Agent2Agent, via `adk-a2a` | no |

## Concepts

### Tools

A tool's doc comment becomes the description the model reads, and its schema is
derived from the signature. An `Option<T>` argument is optional; a
`&ToolContext` argument is injected by the framework and hidden from the model.

```rust
/// Reimburses an amount to the user.
#[adk_tool(crate = ::rusty_adk::tools)]
async fn reimburse(amount: i64, ctx: &ToolContext) -> Result<serde_json::Value> {
    ctx.set_state("user:last_refund", amount);
    Ok(rusty_adk::tools::success(serde_json::json!({"reimbursed": amount})))
}
```

Arguments are validated against the schema before the body runs, and the result
is normalized to ADK's object convention — a scalar return is wrapped under a
`result` key.

### State

State keys are scoped by prefix, and writes are staged until an event carries
them:

| Prefix | Scope | Persisted |
|---|---|---|
| `app:` | every user and session of the app | yes |
| `user:` | one user, across their sessions | yes |
| `temp:` | the current invocation only | no |
| *(none)* | the current session | yes |

A write becomes durable only once the `Runner` has processed the event carrying
its delta. That ordering is the contract: code resuming after a yielded event
can rely on its state having landed.

### Sessions, artifacts, and memory

The in-memory services are the default and keep everything in process memory —
right for tests and one-shot scripts. Enable the `sqlite` feature for storage
that outlives the process:

```rust
let store = SqliteStore::open("agent.db").await?;
let services = store.services();  // all three services, one database
```

Take them individually with `store.sessions()`, `store.artifacts()`, and
`store.memories()`, or open any on its own with the matching `::open`.

Sessions and artifacts reproduce the in-memory semantics exactly — the same
prefix routing, the same hydration of `app:` and `user:` values onto each
thread, the same refusal to record partial events, the same per-filename
versioning — so switching backends changes durability and nothing else. Threads,
history, each state scope, and artifacts get their own table; `temp:` keys get
none, which is what makes them temporary. Events and artifact payloads are
stored as JSON rather than as columns, so schema additions like 2.0's
`node_info` and `output` round-trip without a migration.

That durability is what makes a suspended human-in-the-loop run resumable after
a restart: the interrupt's resume point is ordinary session state, so it lands
in the database with everything else.

`SqliteMemoryService` is the deliberate exception. The in-memory one scores by
counting how many query terms appear in an entry and says of itself that it is a
placeholder to "swap in a real vector store" for — so rather than carry a
stand-in into durable storage, this one indexes with SQLite's FTS5 and ranks by
BM25. Isolation, the ingestion filter, and best-first ordering are identical;
the *scores* are not comparable between the two, which is why the trait calls
that field backend-specific. Re-ingesting a session replaces what it contributed
before, so feeding a growing conversation in repeatedly converges rather than
piling up duplicates.

### Graphs

```rust
let graph = Graph::new(
    vec![triage, billing_agent, technical_agent],
    EdgeBuilder::new()
        .start("triage")
        .add_route("triage", "billing", Route::string("BILLING"))
        .add_route("triage", "technical", Route::string("TECHNICAL"))
        .add_default("triage", "billing")
        .build(),
)?;
```

Nodes in a frontier run concurrently. A node's `NodeOutcome::output` becomes its
successor's input; a `JoinNode` waits for every predecessor and hands its
successor a map keyed by predecessor name.

### Human-in-the-loop

```rust
let approve = FunctionNode::new("approve", NodeConfig::default(), |ctx| {
    let ctx = ctx.clone();
    Box::pin(async move {
        // Suspends on the first pass; returns the answer after resume.
        let answer = ctx.resume_or_request_input("Approve this refund?", None)?;
        Ok(NodeOutcome::output(answer))
    })
})
.shared();
```

The run ends cleanly at the suspension. Resume it with
`runner.resume(user_id, session_id, ResumeRequest::new(interrupt_id, payload))`.

## Interoperating with the other ADK SDKs

ADK defines no language-neutral wire protocol for tools, so a Rust tool reaches
a Python, Go, TypeScript, Java, or Kotlin agent over **MCP**:

```rust
let server = McpServer::new("rust-weather", vec![get_weather_tool()], services);
serve_stdio(&server).await
```

```python
McpToolset(connection_params=StdioConnectionParams(
    server_params=StdioServerParameters(command="./rust-weather-server", args=[]),
))
```

The reverse works too: `McpToolset` in this crate consumes any MCP server's
tools as ADK tools.

### Agents: A2A

ADK's SDKs *do* share a protocol for agents — A2A — and the `a2a` feature
serves a Rust ADK agent over it, so an agent in any language can delegate here:

```rust
let runner = Runner::new("weather_app", agent.shared(), services);
let card = card_for_agent(&agent.shared(), "0.1.0", AgentInterface::json_rpc(&url));
let router = AgentServer::new(card, Arc::new(AdkAgentExecutor::new(Arc::new(runner))))
    .into_router();
```

```python
weather = RemoteA2aAgent(
    name="rust_weather",
    description="A weather agent implemented in Rust.",
    agent_card="http://127.0.0.1:8080/.well-known/agent-card.json",
)
```

One A2A `contextId` is one ADK session, and — the mapping worth having — an ADK
graph suspension becomes an A2A `InputRequired` task that the caller's next
message on that task resumes. See the `a2a-agent-server` example.

## Examples

```bash
cargo run -p weather-agent      # agent + tools, routing, fan-out/join, HITL
cargo run -p mcp-tool-server    # serve Rust tools over MCP (stdio)
cargo run -p mcp-tool-server -- --http
```

`weather-agent` runs offline against `MockModel`, so it needs no API key. Build
it with `--features live` and set `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` to
drive it with a real model.

## Development

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Relationship to Google's ADK

This is an independent implementation of the architecture described at
[adk.dev][adk]. It is not affiliated with or endorsed by Google, and it does not
share code with the official SDKs. Where the published documentation does not
specify an internal layout — `Event.node_info` is the main case — this crate
defines its own and says so in the API docs.

## License

Apache-2.0.

[adk]: https://adk.dev/2.0/
