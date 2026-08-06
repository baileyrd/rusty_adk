# Architecture

How `rusty-adk` is put together, and where it departs from the reference ADK
SDKs.

## Crate layout

Dependencies point downward; nothing below depends on anything above.

```
                        rusty-adk  (facade + prelude)
                             │
   ┌──────────┬──────────────┼──────────────┬──────────┐
   │          │              │              │          │
adk-runner  adk-mcp     adk-agents     adk-macros      │
   │          │         │    │    │                    │
   │          │         │    │    └──── adk-models ────┤
   │          │         │    └───────── adk-graph ─────┤
   │          └─────────┴──────────────  adk-tools ────┤
   │                                                   │
   └────────────────── adk-sessions ─────────────── adk-core
```

| Crate | Responsibility |
|---|---|
| `adk-core` | The data model and the service traits. No runtime. |
| `adk-tools` | The `Tool` trait, `ToolContext`, toolsets, and the framing behaviour around a call. |
| `adk-graph` | The ADK 2.0 workflow graph engine. |
| `adk-models` | The `Model` trait and provider connectors. |
| `adk-sessions` | `SessionService` / `ArtifactService` / `MemoryService` backends: in-memory, plus SQLite session and artifact stores behind the `sqlite` feature. |
| `adk-agents` | `LlmAgent`, the workflow agents, callbacks, and the graph bridge. |
| `adk-runner` | The runtime event loop. |
| `adk-macros` | `#[adk_tool]`. |
| `adk-mcp` | MCP server and client, for cross-SDK interop. |
| `rusty-adk` | Facade and prelude. |

Service **traits** live in `adk-core` rather than `adk-sessions` so that every
crate can depend on the abstraction without pulling in a backend.

## The execution model

### Events are the only currency

Everything an agent, tool, or node wants to report travels as an `Event`. The
`Runner` consumes each one, commits its `EventActions` through the session
service, and forwards it:

```
agent yields Event ──▶ Runner ──▶ SessionService.append_event
                          │            (state_delta, artifact_delta commit)
                          │
                          ├──▶ reflect the commit back into InvocationContext
                          └──▶ forward the Event to the caller
```

State writes are **staged**, not applied. `State` keeps a pending delta
alongside its committed map; a write is readable immediately (ADK calls these
"dirty reads", and they are what lets a tool and a callback coordinate inside
one step) but becomes durable only once an event carrying the delta has been
processed.

Two consequences shaped the implementation:

- The runner reflects a commit by **applying the event's delta**, not by
  replacing the `State` object. Replacing it would discard writes a tool staged
  during the same step but has not yet carried out.
- A graph suspension has to emit an event carrying its resume record. Staging
  the record without an event to carry it would mean it never persists, and the
  run could not be resumed.

### The graph engine

`Graph::run` executes in frontiers:

1. Seed the frontier from the graph's `START` edges — or, when resuming, from
   the single node that suspended.
2. Run every node in the frontier concurrently.
3. For each result, match the node's emitted routes against its outgoing edges.
   Matching edges win; if none match, `Route::Default` edges are used; if there
   are none of those either, that is a routing dead end (`AdkError::NoRoute`)
   rather than a silent stop.
4. A successor that is a join accumulates its predecessor's output and only
   enters the next frontier once every in-edge has delivered.
5. Repeat until the frontier empties or the step budget is exhausted.

A join that can never fill — because a branch failed or was routed around —
would otherwise look like a run that just ended. The engine reports it instead.

Retries consult `AdkError::is_control_flow` first: a suspension or a
confirmation request is control flow, and retrying it would re-ask the user
forever.

### Human-in-the-loop

`NodeContext::resume_or_request_input` is the whole mechanism:

- **First pass** — emits a `RequestInput` event and returns
  `AdkError::NodeInterrupted`. The engine persists a `PendingInterrupt` (which
  node, what input, which step) and ends the run cleanly.
- **After resume** — the same node runs again, and the call returns the payload.

Re-running the node rather than resuming mid-body is what ADK's Go engine does
(`RerunOnResume`), and it means node code does not have to be restructured into
a state machine around each suspension point.

### Persistence

A `SessionService` is where a conversation, its scoped state, and any suspended
resume point actually live; an `ArtifactService` holds the files a run produced.
The choice of backend decides whether either outlives its process. Two of each
ship here, and each pair is behaviourally identical:

| | In-memory | SQLite (feature `sqlite`) |
|---|---|---|
| Storage | `BTreeMap`s behind a mutex | a SQLite file, WAL mode |
| Survives restart | no | yes |
| Scoped state | three maps, keyed by app / (app, user) / thread | three tables, same keys |
| History | a `Vec` on the session | one row per event, ordered by `seq` |
| Artifacts | a `Vec` per key, index = version | one row per key *and* version |

The SQLite schema keeps each state scope in its own table — `app_state`,
`user_state`, `session_state` — with keys stored **including** their prefixes, so
reassembling the flat view an agent sees is a three-query merge in the same order
the in-memory service merges its maps. `temp:` keys have no table at all: they
are dropped on the way in, which is what makes them temporary.

Artifacts are keyed by a *scope* column rather than by session id: a
`user:`-prefixed filename stores under the empty scope so it is reachable from
every one of that user's threads, mirroring the `user:` state prefix. There is
deliberately no foreign key from artifacts to sessions — deleting a thread does
not delete the files it produced, matching the in-memory service.

Events and artifact payloads are stored as serialized JSON in a single `payload`
column rather than as a column per field. ADK 2.0's addition of `node_info` and
`output` is exactly the case where a rigid-column store needs widening and a JSON
store does not, so new fields round-trip with no migration. `PRAGMA user_version`
still records the schema version, and migrations apply forward from whatever
version a file is already at — which is how the artifacts table reached databases
written before it existed.

`SqliteStore` opens the file once and hands out both services sharing one
connection, so a session and the artifacts it produced land in the same database
and the same transaction log. Either service can also be opened alone.

`rusqlite` is blocking, so every query runs on `tokio::task::spawn_blocking` with
the connection lock taken *inside* the blocking closure — consistent with the
"locks are never held across an await" rule below. Busy and locked conditions map
to `AdkError::Storage { retryable: true }`; constraint violations map to the same
variant with `retryable: false`, so callers need not parse error strings.

## Deviations from the reference SDKs

These are places where a faithful transliteration would have been worse Rust.
Each is noted in the API docs at the point it matters.

### `Agent` is not a supertrait of `Node`

ADK 2.0 made `BaseAgent` subclass `BaseNode`. Rust cannot express that here,
because the two have genuinely different execution shapes: an agent streams
`Event`s, a node returns one `NodeOutcome`. A trait with two conflicting `run`
signatures would satisfy the letter and lose the point.

`AgentNode` adapts any agent into a `Node` instead. The composition ADK 2.0 is
after — agents, tools, and functions all appearing as nodes in one graph — works
identically; only the inheritance is expressed as adaptation.

### `Event.node_info` has a defined shape here

ADK 2.0 documents that `node_info` exists and that rigid-column session stores
must be widened for it, but not its internal layout. This crate defines
`NodeInfo { name, node_type, step, predecessor }` — enough for the engine and
for a readable trace. It serializes as a JSON object, so stores that keep events
as serialized JSON (the case ADK says needs no migration) round-trip it.

### Callbacks are async and typed, not keyword-matched

The Python SDK matches callbacks by parameter name. Here each callback is a
distinct boxed async closure. The contract is unchanged: returning `Some(..)`
replaces the wrapped step, returning `None` lets it proceed.

### `#[adk_tool]` derives at compile time

The other SDKs reflect over a function at run time to build its declaration.
The macro does it during compilation, so a malformed tool is a compile error
rather than a startup failure. Because a proc macro cannot know how the caller
reached the ADK, generated paths route through one module and are redirectable
with `#[adk_tool(crate = ::rusty_adk::tools)]`.

### Interop is MCP, because ADK has no tool wire protocol

ADK's SDKs share a protocol for *agents* (A2A) but not for tools; a tool is an
in-process object in each language. The one path a Rust tool has into a Python,
Go, TypeScript, Java, or Kotlin agent is MCP, which every ADK SDK consumes via
`McpToolset`. `adk-mcp` therefore implements both directions.

## Design decisions worth knowing

**Sampling parameters are only sent when set.** Current Claude models reject
`temperature`, `top_p`, and `top_k` with a 400. `GenerateContentConfig` holds
them as `Option` and the Anthropic connector omits an unset field, so the
default configuration is valid on every model.

**A tool failure is a result, not an error.** `invoke_tool` turns a failing tool
into `{"status": "error", "error_message": ...}` so the model can read it and
recover. An `Err` from the framework layer means something the model cannot fix.

**Limits are mandatory where a runaway is possible.** `LoopAgent` requires an
iteration cap, `RunConfig` caps model calls and graph steps, and `LlmAgent` caps
tool round trips. Each failure names the limit it hit.

**Locks are never held across an await.** `InvocationContext` uses a
`std::sync::RwLock` and exposes it only through short closures.

## Testing

Every crate carries unit tests next to the code; behaviour that spans crates is
tested where it is composed (`adk-runner` for commit ordering, `adk-agents` for
the tool loop, `rusty-adk` for the assembled stack). `MockModel` scripts model
turns so the whole suite runs offline and deterministically — no test needs an
API key or a network.

The proc macro is tested from `crates/adk-macros/tests/`, since a proc-macro
crate cannot use its own macros.
