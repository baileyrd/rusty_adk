//! Serves a Rust ADK agent over the Agent2Agent protocol.
//!
//! A2A is how ADK agents talk to each other across languages, so this is the
//! route by which a Python, Go, TypeScript, Java, or Kotlin agent delegates to
//! one written here. It is the agent-level counterpart to `mcp-tool-server`,
//! which does the same job for tools.
//!
//! The root here is an ADK 2.0 **graph**: an approval node that suspends for a
//! human, then an agent that answers. That shape is deliberate — it exercises
//! the correspondence the bridge exists for, between A2A's `InputRequired` and
//! an ADK graph suspension.
//!
//! # Running
//!
//! ```text
//! cargo run -p a2a-agent-server              # listens on 127.0.0.1:8080
//! ```
//!
//! Fetch the card a peer discovers you by:
//!
//! ```text
//! curl -s http://127.0.0.1:8080/.well-known/agent-card.json | jq .
//! ```
//!
//! Every call below carries `A2A-Version: 1.0` — the server rejects a request
//! that does not declare a version it supports (spec Section 3.2.6) — and uses
//! the spec's operation names as JSON-RPC methods.
//!
//! Send a message. The run suspends at the approval node, so the task comes
//! back `TASK_STATE_INPUT_REQUIRED` with the question attached:
//!
//! ```text
//! curl -s http://127.0.0.1:8080/ \
//!   -H 'content-type: application/json' -H 'A2A-Version: 1.0' -d '{
//!   "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
//!   "params": {"message": {
//!     "messageId": "m1", "role": "ROLE_USER",
//!     "parts": [{"text": "What is the weather in Paris?"}]
//!   }}
//! }' | jq '.result.task | {id, contextId, state: .status.state, ask: .status.message.parts}'
//! ```
//!
//! Answer it by sending another message carrying that `taskId` and
//! `contextId`. The graph resumes at the node that asked, and the task
//! completes:
//!
//! ```text
//! curl -s http://127.0.0.1:8080/ \
//!   -H 'content-type: application/json' -H 'A2A-Version: 1.0' -d '{
//!   "jsonrpc": "2.0", "id": 2, "method": "SendMessage",
//!   "params": {"message": {
//!     "messageId": "m2", "role": "ROLE_USER", "taskId": "<TASK_ID>",
//!     "contextId": "<CONTEXT_ID>", "parts": [{"text": "approved"}]
//!   }}
//! }' | jq '.result.task | {state: .status.state, answer: .status.message.parts}'
//! ```
//!
//! # Calling it from a Python ADK agent
//!
//! ```python
//! from google.adk.agents import LlmAgent
//! from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
//!
//! weather = RemoteA2aAgent(
//!     name="rust_weather",
//!     description="A weather agent implemented in Rust.",
//!     agent_card="http://127.0.0.1:8080/.well-known/agent-card.json",
//! )
//!
//! root_agent = LlmAgent(
//!     model="gemini-flash-latest",
//!     name="concierge",
//!     instruction="Delegate weather questions to rust_weather.",
//!     sub_agents=[weather],
//! )
//! ```

use std::sync::Arc;

use adk_a2a::rusty_a2a::server::AgentServer;
use adk_a2a::rusty_a2a::types::{AgentCard, AgentInterface, AgentSkill};
use adk_a2a::AdkAgentExecutor;
use rusty_adk::prelude::*;
use serde_json::json;

/// Retrieves the current weather for a city.
#[adk_tool(crate = ::rusty_adk::tools)]
async fn get_weather(city: String) -> Result<serde_json::Value> {
    Ok(rusty_adk::tools::success(json!({
        "report": format!("It is sunny in {city}, 22°C."),
    })))
}

const HOST: std::net::Ipv4Addr = std::net::Ipv4Addr::new(127, 0, 0, 1);
const PORT: u16 = 8080;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let url = format!("http://{HOST}:{PORT}");

    // An ordinary ADK agent. Nothing about it knows A2A exists.
    let assistant = LlmAgent::builder("assistant")
        .model(Arc::new(
            MockModel::new()
                .push_call_json("get_weather", json!({"city": "Paris"}))
                .push_text("It is sunny in Paris, 22°C."),
        ))
        .description("Answers weather questions.")
        .instruction("Answer weather questions using the get_weather tool.")
        .tool(get_weather_tool())
        .build()
        .expect("agent builds")
        .shared();

    // The approval gate. On the first pass this returns
    // `AdkError::NodeInterrupted`, which the graph turns into a persisted
    // resume point; the bridge reports it to the caller as `InputRequired`.
    let approve = FunctionNode::new("approve", NodeConfig::default(), |ctx| {
        let ctx = ctx.clone();
        Box::pin(async move {
            let answer = ctx.resume_or_request_input("May I answer this request?", None)?;
            Ok(NodeOutcome::output(answer))
        })
    })
    .shared();

    let graph = Arc::new(
        Graph::new(
            vec![approve, AgentNode::new(assistant).shared()],
            chain(["approve", "assistant"]),
        )
        .expect("graph is well formed"),
    );

    let runner = Runner::new(
        "weather_app",
        graph,
        Services::new(Arc::new(InMemorySessionService::new()))
            .with_artifact(Arc::new(InMemoryArtifactService::new())),
    );

    // The card is what a peer reads before talking to us. A graph root has no
    // single agent to describe it, so it is written out here; for an agent
    // root, `adk_a2a::card_for_agent` derives one from the agent itself.
    let card = AgentCard::new(
        "rust-weather-agent",
        "Answers weather questions, after a human approves each request.",
        "0.1.0",
        AgentInterface::json_rpc(&url),
    )
    .with_streaming(true)
    .with_skill(AgentSkill::new(
        "weather",
        "Weather lookup",
        "Reports current conditions for a city.",
    ));

    let executor = Arc::new(AdkAgentExecutor::new(Arc::new(runner)));
    let router = AgentServer::new(card, executor).into_router();

    println!("A2A agent listening on {url}");
    println!("  card: {url}/.well-known/agent-card.json");
    let listener = tokio::net::TcpListener::bind((HOST, PORT)).await?;
    axum::serve(listener, router).await
}
