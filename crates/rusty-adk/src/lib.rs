//! A Rust implementation of the [Agent Development Kit (ADK) 2.0][adk]
//! architecture.
//!
//! ADK 2.0 ships SDKs for Python, Go, TypeScript, Java, and Kotlin, but not
//! Rust. This crate is a faithful port of its architecture: the same data
//! model, the same graph execution engine, the same tool and callback
//! contracts — expressed idiomatically in Rust.
//!
//! This is the facade. It re-exports the workspace crates so an application
//! needs one dependency:
//!
//! | Crate | What it holds |
//! |---|---|
//! | [`core`] | `Event` (with 2.0's `node_info` / `output`), `State`, `Session`, `InvocationContext` |
//! | [`tools`] | The `Tool` trait, `ToolContext`, toolsets |
//! | [`graph`] | The 2.0 workflow graph engine: nodes, routes, joins, interrupts |
//! | [`agents`] | `LlmAgent`, the workflow agents, callbacks |
//! | [`models`] | The `Model` trait, `MockModel`, Gemini and Anthropic connectors |
//! | [`sessions`] | Session, artifact, and memory services — in-memory, or SQLite-backed with the `sqlite` feature |
//! | [`runner`] | The runtime event loop |
//! | [`mcp`] | MCP transports: serve Rust tools to an ADK agent in any language |
//! | `a2a` | The Agent2Agent bridge (feature `a2a`): serve this agent to any A2A caller |
//!
//! # Getting started
//!
//! ```
//! # tokio_test::block_on(async {
//! use rusty_adk::prelude::*;
//! use std::sync::Arc;
//!
//! // A tool: the doc comment is what the model reads to decide when to call it.
//! /// Retrieves the current weather for a city.
//! #[adk_tool(crate = ::rusty_adk::tools)]
//! async fn get_weather(city: String) -> Result<serde_json::Value> {
//!     Ok(rusty_adk::tools::success(serde_json::json!({
//!         "report": format!("It is sunny in {city}."),
//!     })))
//! }
//!
//! // An agent that can call it. MockModel scripts the exchange so this runs
//! // offline; swap in GeminiModel or AnthropicModel for a live one.
//! let model = MockModel::new()
//!     .push_call_json("get_weather", serde_json::json!({"city": "Paris"}))
//!     .push_text("It is sunny in Paris.");
//!
//! let agent = LlmAgent::builder("weather_agent")
//!     .model(Arc::new(model))
//!     .instruction("Answer weather questions using the get_weather tool.")
//!     .tool(get_weather_tool())
//!     .build()?;
//!
//! // The runner owns the session and commits state as events flow.
//! let services = Services::new(Arc::new(InMemorySessionService::new()));
//! let runner = Runner::new("weather_app", agent.shared(), services);
//! let session = runner.create_session("user-1", None).await?;
//!
//! let answer = runner
//!     .run_to_completion(&session.user_id, &session.id, Content::user_text("Weather in Paris?"), None)
//!     .await?;
//!
//! assert_eq!(answer.as_deref(), Some("It is sunny in Paris."));
//! # Ok::<(), AdkError>(())
//! # }).unwrap();
//! ```
//!
//! # Interoperating with other ADK SDKs
//!
//! ADK defines no language-neutral wire protocol for tools, so a Rust tool
//! reaches a Python, Go, TypeScript, Java, or Kotlin agent over **MCP**. Serve
//! tools with [`mcp::McpServer`] and register them from the other side with
//! that SDK's `McpToolset`. See the `mcp-tool-server` example.
//!
//! [adk]: https://adk.dev/2.0/

#![deny(missing_docs)]
#![warn(clippy::all)]

pub use adk_agents as agents;
pub use adk_core as core;
pub use adk_graph as graph;
pub use adk_models as models;
pub use adk_runner as runner;
pub use adk_sessions as sessions;
pub use adk_tools as tools;

#[cfg(feature = "mcp")]
pub use adk_mcp as mcp;

#[cfg(feature = "a2a")]
pub use adk_a2a as a2a;

#[cfg(feature = "macros")]
pub use adk_macros::adk_tool;

/// Everything needed to build an agent, in one import.
///
/// ```
/// use rusty_adk::prelude::*;
/// ```
pub mod prelude {
    pub use crate::agents::{
        Agent, AgentNode, Callbacks, IncludeContents, LlmAgent, LoopAgent, ParallelAgent,
        SequentialAgent, SharedAgent,
    };
    pub use crate::core::{
        AdkError, Args, ArtifactService, Content, Event, EventActions, FunctionCall,
        FunctionDeclaration, FunctionResponse, InvocationContext, NodeInfo, Part, Result, Role,
        RunConfig, Schema, SchemaType, Services, Session, SessionService, State, StreamingMode,
    };
    pub use crate::graph::{
        chain, concat, constant_node, Edge, EdgeBuilder, FunctionNode, Graph, JoinNode, Node,
        NodeConfig, NodeContext, NodeOutcome, ResumeRequest, Route, RouterNode, START,
    };
    pub use crate::models::{
        GenerateContentConfig, LlmRequest, LlmResponse, MockModel, Model, ModelRegistry,
    };
    pub use crate::runner::Runner;
    pub use crate::sessions::{
        InMemoryArtifactService, InMemoryMemoryService, InMemorySessionService,
    };
    #[cfg(feature = "sqlite")]
    pub use crate::sessions::{SqliteArtifactService, SqliteSessionService, SqliteStore};
    pub use crate::tools::{
        invoke_tool, FunctionTool, SharedTool, StaticToolset, Tool, ToolContext, ToolSource,
        Toolset,
    };

    #[cfg(feature = "macros")]
    pub use crate::adk_tool;

    #[cfg(feature = "models")]
    pub use crate::models::{AnthropicModel, GeminiModel};
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::Arc;

    /// Retrieves the current weather for a city.
    // Inside this crate the facade is `crate`, not `::rusty_adk`.
    #[adk_tool(crate = crate::tools)]
    async fn get_weather(city: String) -> Result<serde_json::Value> {
        Ok(crate::tools::success(json!({
            "report": format!("It is sunny in {city}."),
        })))
    }

    #[tokio::test]
    async fn an_agent_a_tool_and_the_runner_work_together() {
        let model = MockModel::new()
            .push_call_json("get_weather", json!({"city": "Paris"}))
            .push_text("It is sunny in Paris.");

        let agent = LlmAgent::builder("weather_agent")
            .model(Arc::new(model))
            .instruction("Answer weather questions using the get_weather tool.")
            .tool(get_weather_tool())
            .output_key("last_answer")
            .build()
            .unwrap();

        let services = Services::new(Arc::new(InMemorySessionService::new()));
        let runner = Runner::new("weather_app", agent.shared(), services);
        let session = runner.create_session("u1", None).await.unwrap();

        let answer = runner
            .run_to_completion(
                "u1",
                &session.id,
                Content::user_text("Weather in Paris?"),
                None,
            )
            .await
            .unwrap();
        assert_eq!(answer.as_deref(), Some("It is sunny in Paris."));

        let saved = runner.session("u1", &session.id).await.unwrap().unwrap();
        assert_eq!(
            saved.state.get("last_answer").unwrap(),
            "It is sunny in Paris."
        );
    }

    /// A conversation, and the files it produced, outlive the process.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn a_sqlite_store_survives_a_restart() {
        let path =
            std::env::temp_dir().join(format!("{}.sqlite3", crate::core::new_id("rusty-adk-test")));

        let agent = |reply: &str| {
            LlmAgent::builder("assistant")
                .model(Arc::new(MockModel::new().push_text(reply)))
                .output_key("last_answer")
                .build()
                .unwrap()
                .shared()
        };

        // First "process": ask one question, save a file, then drop everything.
        let session_id = {
            let store = SqliteStore::open(&path).await.unwrap();
            let runner = Runner::new("support_app", agent("Paris."), store.services());
            let session = runner.create_session("u1", None).await.unwrap();
            runner
                .run_to_completion(
                    "u1",
                    &session.id,
                    Content::user_text("Capital of France?"),
                    None,
                )
                .await
                .unwrap();
            store
                .artifacts()
                .save_artifact(
                    "support_app",
                    "u1",
                    &session.id,
                    "answer.md",
                    Part::text("Paris."),
                )
                .await
                .unwrap();
            session.id
        };

        // Second "process": a fresh store over the same file picks the thread
        // back up with its history, state, and artifacts intact.
        let store = SqliteStore::open(&path).await.unwrap();
        let runner = Runner::new("support_app", agent("Berlin."), store.services());

        let resumed = runner.session("u1", &session_id).await.unwrap().unwrap();
        assert_eq!(resumed.state.get("last_answer").unwrap(), "Paris.");
        assert!(resumed.events.iter().any(|e| e.text() == "Paris."));

        let artifacts = store.artifacts();
        assert_eq!(
            artifacts
                .load_artifact("support_app", "u1", &session_id, "answer.md", None)
                .await
                .unwrap()
                .unwrap(),
            Part::text("Paris.")
        );

        runner
            .run_to_completion(
                "u1",
                &session_id,
                Content::user_text("And of Germany?"),
                None,
            )
            .await
            .unwrap();

        let after = runner.session("u1", &session_id).await.unwrap().unwrap();
        assert_eq!(after.state.get("last_answer").unwrap(), "Berlin.");
        // The second turn appended to the first rather than starting over.
        assert!(after.events.len() > resumed.events.len());
        // And the artifact's version sequence continued rather than restarting.
        assert_eq!(
            artifacts
                .save_artifact(
                    "support_app",
                    "u1",
                    &session_id,
                    "answer.md",
                    Part::text("Berlin.")
                )
                .await
                .unwrap(),
            1
        );

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn a_graph_routes_between_agents() {
        let triage = RouterNode::new("triage", NodeConfig::default(), |_ctx| {
            Box::pin(async { Ok((json!("a refund request"), vec!["BILLING".to_string()])) })
        })
        .shared();

        let billing = AgentNode::new(
            LlmAgent::builder("billing")
                .model(Arc::new(MockModel::new().push_text("Refund issued.")))
                .build()
                .unwrap()
                .shared(),
        )
        .shared();

        let technical = AgentNode::new(
            LlmAgent::builder("technical")
                .model(Arc::new(MockModel::new().push_text("Ticket opened.")))
                .build()
                .unwrap()
                .shared(),
        )
        .shared();

        let graph = Graph::new(
            vec![triage, billing, technical],
            EdgeBuilder::new()
                .start("triage")
                .add_route("triage", "billing", Route::string("BILLING"))
                .add_route("triage", "technical", Route::string("TECHNICAL"))
                .build(),
        )
        .unwrap();

        let services = Services::new(Arc::new(InMemorySessionService::new()));
        let ctx = InvocationContext::new(
            Session::new("s", "app", "u"),
            services,
            RunConfig::default(),
        );

        let events: Vec<Event> = graph
            .run(ctx, None)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|e| e.unwrap())
            .collect();

        assert!(events.iter().any(|e| e.author == "billing"));
        assert!(!events.iter().any(|e| e.author == "technical"));
    }
}
